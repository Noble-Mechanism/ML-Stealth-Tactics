"""Spec 2 tests: datalink timing/loss/range, fusion, IRST triangulation,
launch on remote, support handoff, Red policy switch, dead jets, determinism."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from stealth_tactics.sim.aircraft import Aircraft, AircraftState, bearing_to, _wrap_pi
from stealth_tactics.sim.datalink import Datalink
from stealth_tactics.sim.fusion import triangulate_pair
from stealth_tactics.sim.sensor_config import (
    DEFAULT_SENSOR_CONFIG as CFG, F35_KEY, RED_KEY, NM_M, SensorConfig,
)
from stealth_tactics.sim.sensors import SensorModel
from stealth_tactics.sim.world import World, SimConfig

DEG = math.pi / 180.0


def B(uid, x, y, hdg=0.0, alt=8000.0):
    return Aircraft.make_blue(uid, f"F-35-{uid}", AircraftState(x, y, alt, hdg, 250.0))


def R(uid, x, y, hdg=math.pi, alt=8000.0):
    return Aircraft.make_red(uid, f"Red-{uid}", AircraftState(x, y, alt, hdg, 250.0))


def cfg_with(**dl_kw) -> SensorConfig:
    return dataclasses.replace(CFG, datalink=dataclasses.replace(CFG.datalink, **dl_kw))


def no_loss() -> SensorConfig:
    return cfg_with(loss_prob={"Blue": 0.0, "Red": 0.0})


def static_world(aircraft, secs, cfg=None, seed=1, blue=None, red=None):
    """World where aircraft don't move (speeds kept only for IRST)."""
    noop = lambda w: None  # noqa: E731
    w = World(aircraft, SimConfig(dt=0.5, max_time_s=secs, seed=seed, sensor_config=cfg),
              blue or noop, red or noop, record=False)
    return w


def freeze(w: World):
    """Controller wrapper helper: zero the kinematics by resetting positions."""
    snap = {a.id: (a.state.x, a.state.y, a.state.alt, a.state.heading_rad) for a in w.aircraft}

    def hold(world: World):
        for a in world.aircraft:
            x, y, z, h = snap[a.id]
            a.state.x, a.state.y, a.state.alt = x, y, z
            a.cmd_heading_rad = a.state.heading_rad
    return hold


# ------------------------------------------------------------ timing -------
def _receipt_log(aircraft, cfg, secs, observer, target):
    """Run a static world; record (t, snapshot time of each sender) seen by observer."""
    w = static_world(aircraft, secs, cfg)
    log = []

    def ctl(world: World):
        for a in world.aircraft:  # keep everybody in place
            a.state.speed_mps = 250.0
        tr = world.tracks[observer].get(target) if observer in world.tracks else None
        log.append((world.time_s, dict(tr.remote_msg_t) if tr else {}))
    w.blue_controller = ctl
    w.red_controller = lambda world: None
    # zero movement: set speeds to min and headings pinned by resetting positions
    orig = {a.id: (a.state.x, a.state.y) for a in aircraft}

    def red_ctl(world):
        for a in world.aircraft:
            a.state.x, a.state.y = orig[a.id]
    w.red_controller = red_ctl
    w.run()
    return w, log


def test_blue_link_latency_1s_and_rate_1s() -> None:
    # B1 sees R1 on radar; B2 faces away (R1 outside all its sensors)
    b1, b2 = B("B1", 0, 0, 0.0), B("B2", 2000, 0, math.pi)
    r1 = R("R1", 0, 40_000, math.pi / 2)  # beam to B1 so Red radar can't see B2 stuff
    w, log = _receipt_log([b1, b2, r1], no_loss(), 8.0, "B2", "R1")
    b1_first = next(e["t"] for e in w.sensor_events
                    if e["observer"] == "B1" and e["type"] == "radar_detect")
    first_rx = next(t for t, m in log if "B1" in m)
    assert first_rx == pytest.approx(b1_first + 1.0)          # 1 s latency
    snaps = sorted({m["B1"] for t, m in log if "B1" in m})
    assert np.allclose(np.diff(snaps), 1.0)                    # 1 s updates
    for t, m in log:
        if "B1" in m:
            assert t - m["B1"] >= 1.0 - 1e-9                    # never fresher than latency


def test_red_link_latency_2s_and_rate_2s() -> None:
    r1, r2 = R("R1", 0, 0, 0.0), R("R2", 2000, 0, math.pi)
    b1 = B("B1", 0, 20_000, math.pi / 2)  # beam-on F-35 at 20 km: easy for R1
    w, log = _receipt_log([r1, r2, b1], no_loss(), 16.0, "R2", "B1")
    snaps = sorted({m["R1"] for t, m in log if "R1" in m})
    assert len(snaps) >= 3 and np.allclose(np.diff(snaps), 2.0)
    first_rx = next(t for t, m in log if "R1" in m)
    assert first_rx - snaps[0] == pytest.approx(2.0)
    assert "B1" not in w.tracks["R2"].tracks or all(
        s == "R1" for s in w.tracks["R2"].tracks["B1"].remote)


def test_message_loss_about_5_percent() -> None:
    dl = Datalink(np.random.default_rng(123))
    b1, b2 = B("B1", 0, 0), B("B2", 1000, 0)
    stores = {}
    for k in range(4000):
        dl.send([b1, b2], stores, float(k))
    frac = dl.lost / dl.sent
    assert dl.sent == 8000
    assert frac == pytest.approx(0.05, abs=0.01)


def test_link_range_limit_150nm() -> None:
    dl = Datalink(np.random.default_rng(0), no_loss())
    b1, near, far = B("B1", 0, 0), B("B2", 149 * NM_M, 0), B("B3", 0, -151 * NM_M)
    dl.send([b1, near, far], {}, 0.0)
    rec = {r["sender"]: r["recipients"] for r in dl.log}
    assert rec["B1"] == ["B2"]
    assert "B3" not in rec["B2"]  # B2-B3 ~ 213 NM
    assert rec["B3"] == []


# ------------------------------------------------------------ fusion -------
def test_best_error_fusion_picks_flightmate_track_with_provenance() -> None:
    # B1 far (95 km) from R1, B2 close (25 km): B2's shared radar track is far
    # more accurate even after 1-2 s latency growth
    b1, b2 = B("B1", 0, -70_000, 0.0), B("B2", 0, 0, 0.0)
    r1 = R("R1", 0, 25_000, math.pi / 2)
    w = static_world([b1, b2, r1], 12.0, no_loss())
    w.red_controller = freeze(w)
    w.run()
    f1 = w.tracks["B1"].fused["R1"]
    assert f1.source_jet == "B2" and not f1.own and f1.sensor == "radar"
    assert f1.age_s >= 1.0
    labels = dict(f1.candidates)
    assert labels["B2-radar"] < labels["own-radar"]
    f2 = w.tracks["B2"].fused["R1"]
    assert f2.own and f2.source_jet == "B2"     # own is best for B2
    # RWR contacts are never shared
    for tr in w.tracks["B1"].tracks.values():
        for comps in tr.remote.values():
            assert "rwr" not in comps


# ------------------------------------------------------ triangulation ------
def test_triangulation_geometry_and_min_angle() -> None:
    tgt = np.array([0.0, 50_000.0, 8000.0])
    p1 = np.array([-5_000.0, 0.0, 8000.0])
    p2 = np.array([5_000.0, 0.0, 8000.0])
    az = lambda p: math.atan2(tgt[0] - p[0], tgt[1] - p[1])  # noqa: E731
    fix = triangulate_pair(p1, az(p1), 0.0, p2, az(p2), 0.0, 0.5 * DEG, 10.0)
    assert fix is not None and np.linalg.norm(fix["pos"][:2] - tgt[:2]) < 1e-6
    assert fix["angle_deg"] == pytest.approx(2 * math.degrees(math.atan(5 / 50)))
    # < 10 deg: no fix
    q1, q2 = np.array([-3_000.0, 0, 8000]), np.array([3_000.0, 0, 8000])
    assert triangulate_pair(q1, az(q1), 0, q2, az(q2), 0, 0.5 * DEG, 10.0) is None


@pytest.mark.parametrize("half_base", [4_500.0, 13_000.0, 25_000.0])
def test_triangulation_error_matches_propagation(half_base) -> None:
    rng = np.random.default_rng(5)
    tgt = np.array([0.0, 25_000.0, 8000.0])
    p1, p2 = np.array([-half_base, 0, 8000.0]), np.array([half_base, 0, 8000.0])
    a1 = math.atan2(tgt[0] - p1[0], tgt[1] - p1[1])
    a2 = math.atan2(tgt[0] - p2[0], tgt[1] - p2[1])
    sb = 0.5 * DEG
    errs, pred = [], None
    for _ in range(3000):
        fx = triangulate_pair(p1, a1 + rng.normal(0, sb), 0.0,
                              p2, a2 + rng.normal(0, sb), 0.0, sb, 10.0)
        errs.append(np.linalg.norm(fx["pos"][:2] - tgt[:2]))
        pred = fx
    r = math.hypot(half_base, 25_000.0)
    g = math.radians(pred["angle_deg"])
    sigma_h = sb * math.sqrt(2) * r / math.sin(g)
    rms = math.sqrt(np.mean(np.square(errs)))
    assert rms == pytest.approx(sigma_h, rel=0.1)


def test_triangulation_error_shrinks_with_angle_up_to_90() -> None:
    tgt = np.array([0.0, 30_000.0, 8000.0])
    sig = []
    for hb in (4_000.0, 10_000.0, 20_000.0, 30_000.0):  # angle ~15 -> 90 deg
        p1, p2 = np.array([-hb, 0, 8000.0]), np.array([hb, 0, 8000.0])
        a1 = math.atan2(-p1[0], tgt[1]); a2 = math.atan2(-p2[0], tgt[1])  # noqa: E702
        fx = triangulate_pair(p1, a1, 0, p2, a2, 0, 0.5 * DEG, 10.0)
        sig.append(fx["sigma_geom"] / math.hypot(hb, 30_000.0))  # per unit range
    assert all(a > b for a, b in zip(sig, sig[1:]))


def _radar_off_cfg() -> SensorConfig:
    radar = dataclasses.replace(CFG.radar, ref_range_m={F35_KEY: 1.0, RED_KEY: 1.0})
    return dataclasses.replace(no_loss(), radar=radar)


def test_passive_triangulation_in_fusion_never_fire_control() -> None:
    b1, b2 = B("B1", -12_000, 0, 0.0), B("B2", 12_000, 0, 0.0)
    r1 = R("R1", 0, 20_000, 0.0)   # flying away: hot tail visible to both IRSTs
    w = static_world([b1, b2, r1], 10.0, _radar_off_cfg())
    w.red_controller = freeze(w)
    w.run()
    f = w.tracks["B1"].fused["R1"]
    assert f.sensor == "tri" and set(f.source_jet.split("+")) == {"B1", "B2"}
    own_irst = dict(f.candidates)["own-irst"]
    assert f.sigma_m < own_irst
    assert np.linalg.norm(f.est_pos - r1.state.position()) < 4 * f.sigma_m
    assert not w.tracks["B1"].is_fire_control("R1")
    assert w.fire_control_source(b1, "R1") is None
    # Red has no triangulation
    r2 = R("R2", 5000, 20_000, 0.0)
    assert all(fr.sensor != "tri" for st in w.tracks.values() for fr in st.fused.values()
               if st.owner_id.startswith("R"))


# ------------------------------------------- launch on remote / handoff ----
def _remote_shot_world(side: str, cfg=None, lead_turns_cold=False, seed=3):
    """Lead holds FC on the enemy; wingman faces away, then snaps toward the
    enemy (target 50 deg off its nose) and fires the same step, then turns cold."""
    if side == "blue":
        lead, wing = B("B1", 0, 0, 0.0), B("B2", 8_000, 0, math.pi)
        enemy = R("R1", 0, 40_000, math.pi)
        shot_t = 8.0
    else:
        lead, wing = R("R1", 0, 0, 0.0), R("R2", 8_000, 0, math.pi)
        enemy = B("B1", 0, 18_000, math.pi / 2)  # beam-on F-35: easy lock for Red
        shot_t = 14.0
    state = {"fired": False}

    def wing_ctl(w: World):
        lead.cmd_heading_rad = bearing_to(lead.state, enemy.state)
        if lead_turns_cold and state["fired"]:
            lead.cmd_heading_rad = _wrap_pi(bearing_to(lead.state, enemy.state) + math.pi)
        enemy.cmd_heading_rad = enemy.state.heading_rad
        if not state["fired"] and w.time_s >= shot_t:
            wing.state.heading_rad = _wrap_pi(bearing_to(wing.state, enemy.state) + 50 * DEG)
            wing.cmd_fire, wing.fire_target = True, enemy.id
            state["fired"] = True
        elif state["fired"]:
            wing.cmd_heading_rad = _wrap_pi(bearing_to(wing.state, enemy.state) + math.pi)
        else:
            wing.cmd_heading_rad = wing.state.heading_rad

    noop = lambda w: None  # noqa: E731
    ctl_blue, ctl_red = (wing_ctl, noop) if side == "blue" else (noop, wing_ctl)
    # Spec 3a: Pk forced to 1 via the shared missile config (no per-type Pk)
    cfg = cfg or CFG
    cfg = dataclasses.replace(cfg, missile_kinematics=dataclasses.replace(
        cfg.missile_kinematics, base_pk=1.0, endgame_pk_floor=1.0))
    w = World([lead, wing, enemy], SimConfig(dt=0.5, max_time_s=90.0, seed=seed,
                                             sensor_config=cfg), ctl_blue, ctl_red)
    return w, lead, wing, enemy


def test_blue_launch_on_remote_and_support_handoff() -> None:
    w, lead, wing, enemy = _remote_shot_world("blue", no_loss())
    res = w.run()
    launches = [e for e in res.events if e["type"] == "launch"]
    assert len(launches) == 1 and launches[0]["shooter"] == "B2"
    assert launches[0]["remote_source"] == "B1"
    types = [e["type"] for e in res.events]
    hand = [e for e in res.events if e["type"] == "support_handoff"]
    assert hand and hand[0]["observer"] == "B1"          # support handed to lead
    assert "autonomous" in types and "support_lost" not in types
    assert "hit" in types                                   # Pk forced to 1
    assert not enemy.state.alive


def test_handoff_fails_if_lead_also_turns_cold() -> None:
    w, *_ = _remote_shot_world("blue", no_loss(), lead_turns_cold=True)
    res = w.run()
    types = [e["type"] for e in res.events]
    # Spec 2b: nobody can support -> missile coasts; it is either lost or reaches
    # autonomy with a Pk factor < 1
    assert "support_lost" in types and "support_regained" not in types
    auto = [e for e in res.events if e["type"] == "autonomous"]
    assert (any(t in types for t in ("lost_coast_timeout", "lost_basket"))
            or (auto and auto[0]["pk_factor"] < 1.0))


def test_red_cannot_launch_on_remote_unless_symmetric() -> None:
    w, lead, wing, enemy = _remote_shot_world("red", no_loss())
    res = w.run()
    assert w.tracks["R1"].fused  # lead does see the F-35
    assert any(e["type"] == "fc_track" and e["observer"] == "R1" for e in res.sensor_events)
    assert not [e for e in res.events if e["type"] == "launch"]

    w2, *_ = _remote_shot_world("red", cfg_with(loss_prob={"Blue": 0.0, "Red": 0.0},
                                                symmetric_weapons_policy=True))
    res2 = w2.run()
    launches = [e for e in res2.events if e["type"] == "launch"]
    assert len(launches) == 1 and launches[0]["remote_source"] == "R1"


# --------------------------------------------------------- dead jets -------
def test_dead_jet_tracks_age_out() -> None:
    b1, b2 = B("B1", 0, 0, 0.0), B("B2", 2000, 0, math.pi)
    r1 = R("R1", 0, 40_000, math.pi / 2)
    w = static_world([b1, b2, r1], 30.0, no_loss())
    log = []

    def ctl(world: World):
        if world.time_s >= 8.0:
            b1.state.alive = False
        tr = world.tracks["B2"].get("R1")
        log.append((world.time_s, None if tr is None else
                    max((c.last_t for comps in tr.remote.values() for c in comps.values()),
                        default=None)))
    w.blue_controller = ctl
    w.red_controller = freeze(w)
    w.run()
    last_meas = max(v for t, v in log if v is not None)
    assert last_meas <= 8.0
    present = [t for t, v in log if v is not None]
    assert max(present) <= last_meas + CFG.track.coast_s + 0.5
    assert max(present) >= last_meas + CFG.track.coast_s - 0.5
    assert "R1" not in w.tracks["B2"]


# -------------------------------------------------------- determinism ------
def _flight(seed):
    b1, b2 = B("B1", -5000, -60_000, 0.0), B("B2", 5000, -60_000, 0.0)
    r1, r2 = R("R1", -3000, 60_000), R("R2", 3000, 60_000)
    noop = lambda w: None  # noqa: E731
    w = World([b1, b2, r1, r2], SimConfig(dt=0.5, max_time_s=150.0, seed=seed), noop, noop)
    res = w.run()
    return res.sensor_events, w.datalink.log


def test_datalink_determinism() -> None:
    a, la = _flight(42)
    b, lb = _flight(42)
    c, lc = _flight(43)
    assert a == b and la == lb
    assert [r["lost"] for r in la] != [r["lost"] for r in lc]
    assert any(e["type"] == "link_track" for e in a)


def test_replay_c_lead_trail_handoff_and_no_support_coast():
    """Lead-trail (Spec 2b: 50 NM Blue FC gate + missile coast; Spec 3a missile):
    the trail takes over support before the seeker goes active;
    with nobody supporting, the missile coasts and reaches autonomy with a
    reduced Pk (or is lost); 'delayed' never shoots before the trail has FC."""
    from stealth_tactics.analysis.datalink_replays import run_lead_trail

    # Spec 8: with loft the lead's table Rmax here is ~42-45 NM (was ~31 NM), so
    # a trail 15 NM back is still outside its own 50 NM FC gate for ~40 s after
    # the shot (>= the 40 s coast timeout). The trail flies 4 NM back so the
    # scenario still tests a handoff (not the coast timeout).
    trail_nm = 4.0
    sup = run_lead_trail(seed=1, record=False, variant="support", trail_nm=trail_nm)
    assert sup.fired_t is not None and sup.shot_range_m is not None
    # Spec 3a: the lead now shoots at the table Rmax (~31 NM, was the fixed
    # 45 km = 24.3 NM), usually before the trail holds its own FC, so a short
    # coast gap is allowed as long as the trail takes over before the seeker
    # goes active (Pk factor 1.0).
    assert sup.handoff_to_trail_t is not None
    assert not sup.support_gap or sup.gap_s < CFG.missile.coast_timeout_s
    assert sup.pk_factor == 1.0 and sup.outcome in ("hit", "miss")
    nos = run_lead_trail(seed=1, record=False, variant="no-support", trail_nm=trail_nm)
    assert nos.support_gap and nos.handoff_to_trail_t is None and nos.gap_s > 0
    assert (nos.outcome in ("lost_coast_timeout", "lost_basket")
            or (nos.pk_factor is not None and nos.pk_factor < 1.0))
    dly = run_lead_trail(seed=1, record=False, variant="delayed", trail_nm=trail_nm)
    assert dly.trail_fc_t is not None and dly.fired_t >= dly.trail_fc_t
    assert dly.outcome in ("hit", "miss")
    assert dly.shot_range_m <= sup.shot_range_m + 1e-6
