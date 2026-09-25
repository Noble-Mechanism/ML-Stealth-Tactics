"""Spec 3: RWR modes, maneuver primitives, Red defense state machine, firing
doctrines, a-pole / f-pole, altitude floor / ground, early end, regression."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stealth_tactics.ga.evolution import GAConfig, GeneticAlgorithm
from stealth_tactics.scenarios.loader import build_aircraft, load_scenario
from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, Coalition,
                                          integrate_aircraft, _wrap_pi)
from stealth_tactics.sim.rwr import (LOCK, MISSILE_ACTIVE, SEARCH, SUPPORT, RwrCue,
                                     RwrModel, emitter_mode)
from stealth_tactics.sim.sensor_config import (DEFAULT_SENSOR_CONFIG as CFG, F35_KEY, NM_M,
                                               SignatureConfig)
from stealth_tactics.sim.weapons import WeaponModel
from stealth_tactics.sim.world import (LEGACY, SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS,
                                       SimConfig, World)
from stealth_tactics.tactics import maneuvers as mv
from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.tactics.interpreter import RedCAPController, TacticsController
from stealth_tactics.tactics.red_defense import (AGGRESSIVE, COLD, CONSERVATIVE, DEFENDING,
                                                 DEPARTED, HOT, MIDDLE, PRESSING,
                                                 DefenseConfig, RedDefense)

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180.0


def blue(x=0.0, y=0.0, hdg=0.0, alt=9000.0, uid="B1", spd=260.0):
    return Aircraft.make_blue(uid, uid, AircraftState(x, y, alt, hdg, spd))


def red(x=0.0, y=0.0, hdg=math.pi, alt=9000.0, uid="R1", spd=255.0):
    return Aircraft.make_red(uid, uid, AircraftState(x, y, alt, hdg, spd))


class _FC:
    def __init__(self, ids=()):
        self.ids = set(ids)

    def is_fire_control(self, tid):
        return tid in self.ids


def _msl(target, supporter=None, autonomous=False, alive=True, mid="M1", shooter="B1",
         x=0.0, y=0.0):
    return SimpleNamespace(id=mid, target_id=target, supporter_id=supporter,
                           autonomous=autonomous, alive=alive, shooter_id=shooter,
                           pos=SimpleNamespace(x=x, y=y))


# ------------------------------------------------------------------ RWR ----
def test_rwr_search_gates_range_for_and_emitting():
    rx = red(0.0, 0.0)
    em = blue(0.0, -53_000.0)                     # F-35 LPI gate 0.6 x 90 km = 54 km
    assert emitter_mode(rx, em, {}, [], CFG) == SEARCH
    em.state.y = -55_000.0
    assert emitter_mode(rx, em, {}, [], CFG) is None
    em.state.y = -30_000.0
    em.state.heading_rad = math.radians(70)       # receiver outside the +/-60 deg FOR
    assert emitter_mode(rx, em, {}, [], CFG) is None
    em.state.heading_rad = math.radians(59)
    assert emitter_mode(rx, em, {}, [], CFG) == SEARCH
    em.radar_emitting = False
    assert emitter_mode(rx, em, {}, [], CFG) is None
    # Red radar vs F-35 RWR: 1.5 x 70 km = 105 km (57 NM)
    b = blue(0.0, 0.0)
    r = red(0.0, 104_000.0)
    assert emitter_mode(b, r, {}, [], CFG) == SEARCH
    r.state.y = 106_000.0
    assert emitter_mode(b, r, {}, [], CFG) is None


def test_rwr_lock_support_highest_mode_wins():
    rx = red(0.0, 0.0)
    em = blue(0.0, -40_000.0)
    tracks = {"B1": _FC({"R1"})}
    assert emitter_mode(rx, em, tracks, [], CFG) == LOCK
    ms = [_msl("R1", supporter="B1")]
    assert emitter_mode(rx, em, tracks, ms, CFG) == SUPPORT         # support beats lock
    assert emitter_mode(rx, em, {}, ms, CFG) == SUPPORT
    assert emitter_mode(rx, em, tracks, [_msl("R2", supporter="B1")], CFG) == LOCK
    assert emitter_mode(rx, em, tracks, [_msl("R1", supporter="B1", alive=False)],
                        CFG) == LOCK


def test_rwr_mode_range_factor():
    cfg = dataclasses.replace(CFG, rwr=dataclasses.replace(
        CFG.rwr, mode_range_factor={"search": 1.0, "lock": 0.5, "support": 1.0}))
    rx, em = red(0.0, 0.0), blue(0.0, -40_000.0)       # 40 km > 0.5 x 54 km
    assert emitter_mode(rx, em, {"B1": _FC({"R1"})}, [], cfg) == SEARCH
    em.state.y = -20_000.0
    assert emitter_mode(rx, em, {"B1": _FC({"R1"})}, [], cfg) == LOCK


def test_rwr_missile_active_only_for_target_and_events():
    r1, r2 = red(0.0, 0.0, uid="R1"), red(5_000.0, 0.0, uid="R2")
    b1 = blue(0.0, -200_000.0)                        # far away: no emitter cues
    model = RwrModel(np.random.default_rng(0), CFG)
    ms = [_msl("R1", autonomous=True, x=0.0, y=-20_000.0)]
    ev = model.update([b1, r1, r2], ms, {}, 0.0)
    assert [c.mode for c in model.cues.get("R1", [])] == [MISSILE_ACTIVE]
    assert "R2" not in model.cues
    cue = model.cues["R1"][0]
    assert abs(_wrap_pi(cue.bearing - math.pi)) < 25 * DEG     # sigma 5 deg
    assert cue.source_jet == "B1" and cue.t_first == 0.0
    assert [e["mode"] for e in ev] == [MISSILE_ACTIVE]
    model.update([b1, r1, r2], ms, {}, 0.5)
    assert model.cues["R1"][0].t_first == 0.0
    ms[0].alive = False
    ev = model.update([b1, r1, r2], ms, {}, 1.0)
    assert model.cues.get("R1", []) == [] and ev[0]["old"] == MISSILE_ACTIVE


def test_rwr_uses_own_rng_stream_world_sensor_streams_unchanged():
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ac = build_aircraft(sc)
    w = World(ac, SimConfig(seed=5), lambda w: None, lambda w: None, record=False)
    s0 = np.random.default_rng([5, CFG.rng_salt]).random()
    assert w.sensor_rng.random() == s0
    assert CFG.rwr.mode_rng_salt == 0x3D3F


# ------------------------------------------------------------ maneuvers ----
def test_maneuver_headings():
    ac = red(hdg=10 * DEG)
    threat = 0.0
    c = mv.crank(ac, threat)                       # heading right of threat -> +50
    assert c.heading == pytest.approx(50 * DEG)
    ac.state.heading_rad = -10 * DEG
    assert mv.crank(ac, threat).heading == pytest.approx(-50 * DEG)
    assert mv.crank(ac, threat, side=1).heading == pytest.approx(50 * DEG)
    assert c.speed == pytest.approx(1.1 * ac.params.cruise_speed_mps) and c.alt == 9000.0
    b = mv.beam(ac, threat)
    assert b.heading == pytest.approx(-90 * DEG) and b.speed == ac.params.max_speed_mps
    d = mv.drag(ac, threat, target_alt=4000.0)
    assert abs(_wrap_pi(d.heading - math.pi)) < 1e-9 and d.alt == 4000.0
    assert d.speed == ac.params.max_speed_mps
    h = mv.hot(ac, 1.0)
    assert h.heading == pytest.approx(1.0) and h.speed == pytest.approx(
        1.1 * ac.params.cruise_speed_mps)
    dp = mv.depart(ac, 2.0)
    assert dp.heading == pytest.approx(2.0) and dp.alt == 9000.0


def test_drag_depth_by_aggressiveness_and_floor():
    ac = red(alt=9000.0)
    assert mv.drag_target_alt(ac, 0.0) == pytest.approx(100.0)          # to the floor
    assert mv.drag_target_alt(ac, 1.0) == pytest.approx(8000.0)         # 1,000 m
    assert mv.drag_target_alt(ac, 0.5) == pytest.approx(9000.0 - (0.5 * 8900 + 500))
    ac.state.alt = 700.0
    assert mv.drag_target_alt(ac, 1.0) == pytest.approx(100.0)          # floor clamp
    assert mv.drag_target_alt(red(alt=9000.0), 1.0, depth_a1_m=2000.0) == pytest.approx(7000.0)


def test_altitude_floor_100m_agl():
    ac = red(alt=150.0)
    ac.cmd_alt_m = -500.0
    for _ in range(10):
        integrate_aircraft(ac, 0.5)
    assert ac.state.alt == pytest.approx(100.0)


def test_missile_hitting_ground_is_lost():
    wm = WeaponModel(np.random.default_rng(0))
    sh = blue(0.0, 0.0, alt=300.0)
    tg = red(0.0, 8_000.0, alt=-3000.0, spd=0.0)       # below ground: missile dives in
    m = wm.spawn(sh, tg)
    by_id = {"B1": sh, "R1": tg}
    for i in range(200):
        wm.step([m], by_id, 0.5, tracks={}, t=i * 0.5)
        if not m.alive:
            break
    assert m.outcome == "ground"


# ------------------------------------------------------------- bands -------
@pytest.mark.parametrize("a,band", [(0.0, CONSERVATIVE), (0.33, CONSERVATIVE),
                                    (0.34, MIDDLE), (0.66, MIDDLE), (0.67, AGGRESSIVE),
                                    (1.0, AGGRESSIVE)])
def test_band_edges(a, band):
    assert DefenseConfig().band(a) == band


def test_t_cold_values():
    c = DefenseConfig()
    assert c.t_cold(0.0) == pytest.approx(40.0)
    assert c.t_cold(0.5) == pytest.approx(22.5)
    assert c.t_cold(1.0) == pytest.approx(5.0)


# ------------------------------------------------------- state machine -----
class FakeWorld:
    def __init__(self, aircraft):
        self._by_id = {a.id: a for a in aircraft}
        self.time_s = 0.0
        self.rwr = {}
        self.tracks = {}
        self.events = []
        self.sensor_cfg = CFG

    def get(self, uid):
        return self._by_id.get(uid)

    def log_event(self, ev):
        self.events.append(ev)


def _run(defense, w, ac, cues_at, t_end, dt=0.5):
    """cues_at(t) -> list of RwrCue; returns list of (t, state)."""
    out = []
    while w.time_s <= t_end + 1e-9:
        w.rwr = {ac.id: cues_at(w.time_s)}
        cmd, shoot = defense.step(w, ac)
        if cmd is not None:
            cmd.apply(ac)
        out.append((w.time_s, defense.jet(ac).state, shoot))
        w.time_s += dt
    return out


def _cue(mode, t0=0.0, eid="B1", brg=0.0, src=None):
    return RwrCue(eid, mode, brg, t0, src)


@pytest.mark.parametrize("a,trigger,below", [(0.0, LOCK, SEARCH), (0.5, SUPPORT, LOCK),
                                             (1.0, MISSILE_ACTIVE, SUPPORT)])
def test_triggers_per_band(a, trigger, below):
    ac, b = red(), blue(0.0, -40_000.0, hdg=0.0)
    w = FakeWorld([ac, b])
    d = RedDefense(a)
    eid = "M1" if trigger == MISSILE_ACTIVE else "B1"
    _run(d, w, ac, lambda t: [_cue(below)], 5.0)
    assert d.jet(ac).state == HOT and d.jet(ac).turn_aways == 0
    _run(d, w, ac, lambda t: [_cue(trigger, eid=eid, src="B1")], 6.0)
    assert d.jet(ac).state == DEFENDING and d.jet(ac).turn_aways == 1
    assert d.jet(ac).reaction == {0.0: "drag", 0.5: "beam", 1.0: "crank"}[a]


def test_clear_cold_recommit_and_no_double_count():
    ac, b = red(), blue(0.0, -40_000.0)
    w = FakeWorld([ac, b])
    d = RedDefense(0.0)
    hist = _run(d, w, ac, lambda t: [_cue(LOCK)] if t < 10.0 else [], 60.0)
    states = dict((round(t, 1), s) for t, s, _ in hist)
    assert states[9.5] == DEFENDING
    assert states[11.0] == DEFENDING            # cleared only after 2 s quiet
    assert states[12.0] == COLD                 # last cue 9.5 s + 2 s -> 11.5 s
    assert states[51.0] == COLD and states[52.0] == HOT   # T_cold 40 s at a = 0
    assert [e["type"] for e in w.events] == ["defend", "threat_cleared", "recommit"]
    # a drag at a = 0 targets the 100 m floor
    assert d.jet(ac).drag_alt == pytest.approx(100.0)
    # a trigger while COLD does not count again
    d2 = RedDefense(0.0)
    w2 = FakeWorld([ac, b])
    _run(d2, w2, ac, lambda t: [_cue(LOCK)] if (t < 5.0 or 20.0 <= t < 22.0) else [], 30.0)
    assert d2.jet(ac).turn_aways == 1


def test_clear_waits_for_tof_after_support_cue():
    ac, b = red(), blue(0.0, -40_000.0)
    w = FakeWorld([ac, b])
    d = RedDefense(0.5)
    # support cue 0-5 s; no fused range -> R_est = RWR range vs F-35 54 km -> 77.1 s
    hist = _run(d, w, ac, lambda t: [_cue(SUPPORT)] if t < 5.0 else [], 90.0)
    assert d.jet(ac).t_est_s == pytest.approx(54_000.0 / 700.0)
    states = dict((round(t, 1), s) for t, s, _ in hist)
    assert states[60.0] == DEFENDING
    assert states[77.5] == COLD
    # with a missile-active cue that ends, the TOF wait is cut short
    d = RedDefense(0.5)
    w = FakeWorld([ac, b])

    def cues(t):
        c = [_cue(SUPPORT)] if t < 5.0 else []
        if 3.0 <= t < 15.0:
            c.append(_cue(MISSILE_ACTIVE, 3.0, eid="M1", src="B1"))
        return c
    hist = _run(d, w, ac, cues, 30.0)
    states = dict((round(t, 1), s) for t, s, _ in hist)
    assert states[14.5] == DEFENDING and states[17.0] == COLD


@pytest.mark.parametrize("a,final", [(0.8, PRESSING), (0.2, DEPARTED)])
def test_turn_away_limit_press_or_depart(a, final):
    ac, b = red(), blue(0.0, -40_000.0)
    w = FakeWorld([ac, b])
    d = RedDefense(a)
    mode = MISSILE_ACTIVE if a >= 2 / 3 else LOCK
    period = 100.0     # 5 s of cue every 100 s: defend, clear, cold (<= 40 s), hot

    def cues(t):
        return [_cue(mode, eid="M1" if mode == MISSILE_ACTIVE else "B1", src="B1")] \
            if (t % period) < 5.0 else []
    hist = _run(d, w, ac, cues, 205.0)
    jd = d.jet(ac)
    assert jd.turn_aways == 2 and jd.state == final
    assert ac.departed == (final == DEPARTED)
    shoot_after = [s for t, st, s in hist if t >= 200.0]
    assert all(shoot_after) if final == PRESSING else not any(shoot_after)
    assert [e["type"] for e in w.events].count("defend") == 2


def test_crank_may_shoot_beam_drag_may_not():
    for a, ok in ((1.0, True), (0.5, False), (0.0, False)):
        ac, b = red(), blue(0.0, -40_000.0)
        w = FakeWorld([ac, b])
        d = RedDefense(a)
        hist = _run(d, w, ac, lambda t: [_cue(MISSILE_ACTIVE, eid="M1", src="B1")], 2.0)
        assert hist[-1][1] == DEFENDING and hist[-1][2] is ok


def test_early_end_when_all_red_departed():
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ac = build_aircraft(sc)
    for a in ac:
        if a.coalition == Coalition.RED:
            a.departed = True
    w = World(ac, SimConfig(max_time_s=100.0, seed=1), lambda w: None, lambda w: None,
              record=False)
    r = w.run()
    assert r.ended_early and r.time_s == 0.0
    ac = build_aircraft(sc)
    ac[-1].departed = True                     # one Red departed: fight continues
    r = World(ac, SimConfig(max_time_s=10.0, seed=1), lambda w: None, lambda w: None,
              record=False).run()
    assert not r.ended_early and r.time_s == pytest.approx(10.0)


# ------------------------------------------------------------ doctrine -----
def _doctrine_world(doc, n_red=1, retarget=None):
    b1 = blue(0.0, 0.0, alt=9000.0)
    reds = [red(3000.0 * i, 45_000.0, alt=9000.0, uid=f"R{i + 1}") for i in range(n_red)]
    for r in reds:
        r.ammo = 0

    def blue_ctl(w):
        b1.cmd_heading_rad = math.atan2(reds[0].state.x, reds[0].state.y - b1.state.y)
        live = [r for r in reds if r.state.alive]
        if not live:
            return
        tgt = live[0]
        if retarget is not None and len([m for m in w.missiles]) >= retarget:
            tgt = live[-1]
        b1.cmd_fire, b1.fire_target = True, tgt.id

    def red_ctl(w):
        for r in reds:
            r.cmd_heading_rad = math.pi
    w = World([b1] + reds, SimConfig(max_time_s=25.0, seed=3, blue_doctrine=doc), blue_ctl,
              red_ctl, record=False)
    return w, w.run()


def test_doctrine_shoot_assess_shoot_one_in_flight():
    w, r = _doctrine_world(SHOOT_ASSESS_SHOOT)
    assert r.blue_shots == 1


def test_doctrine_shoot_shoot_assess_pair_then_hold():
    w, r = _doctrine_world(SHOOT_SHOOT_ASSESS, n_red=2, retarget=1)
    ts = [s["launch_t"] for s in r.shots]
    assert len(ts) == 2 and ts[1] - ts[0] == pytest.approx(3.0)
    assert {s["target"] for s in r.shots} == {"R1"}     # second shot redirected to R1
    assert r.shots[0]["outcome"] == "in_flight"          # nothing more while both fly


def test_doctrine_legacy_ripples():
    w, r = _doctrine_world(LEGACY)
    assert r.blue_shots == 4


# ------------------------------------------------------------- poles -------
def test_a_pole_f_pole_static_geometry():
    wm = WeaponModel(np.random.default_rng(1))
    sh = blue(0.0, 0.0, alt=9000.0)
    tg = red(0.0, 50_000.0, alt=9000.0, spd=0.0)
    m = wm.spawn(sh, tg)
    assert m.a_pole_m is None
    by_id = {"B1": sh, "R1": tg}
    tracks = {"B1": {"R1"}}          # legacy set of fire-control ids
    for i in range(400):
        wm.step([m], by_id, 0.5, tracks=tracks, t=i * 0.5)
        if not m.alive:
            break
    assert m.a_pole_m == pytest.approx(50_000.0)
    assert m.f_pole_m == pytest.approx(50_000.0)
    # starts active -> a-pole at launch; shooter dead at the end -> f-pole None
    tg2 = red(0.0, 10_000.0, alt=9000.0, spd=0.0, uid="R2")
    m2 = wm.spawn(sh, tg2)
    assert m2.a_pole_m == pytest.approx(10_000.0)
    sh.state.alive = False
    for i in range(100):
        wm.step([m2], {"B1": sh, "R2": tg2}, 0.5, tracks={}, t=i * 0.5)
        if not m2.alive:
            break
    assert not m2.alive and m2.f_pole_m is None


# ------------------------------------------------------ determinism --------
def test_same_seed_same_result_with_defense():
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ga = GeneticAlgorithm(sc, GAConfig(seed=7, red_aggressiveness=0.5))
    r1 = ga.evaluate(TacticsGenome(), seed_offset=3)
    r2 = ga.evaluate(TacticsGenome(), seed_offset=3)
    assert r1.fitness == r2.fitness and r1.sim.time_s == r2.sim.time_s
    assert r1.sim.shots == r2.sim.shots
    assert [e["type"] for e in r1.sim.events] == [e["type"] for e in r2.sim.events]


# ------------------------------------------------------- regression --------
OLD_F35_TABLE = ((0.0, 0.05), (30.0, 0.10), (60.0, 0.45), (90.0, 0.90), (135.0, 0.55),
                 (180.0, 0.30))
# Pre-spec-3 code (commit 4a5ebf0), default_4v3, default genome, sim seed 1000+i
PRE_SPEC3 = {0: "d0507f1f0ba28783", 1: "788ecbec1bc257f4", 2: "0ee9b85ce5879fc4"}


def _fingerprint(seed):
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ac = build_aircraft(sc)
    b = [a.id for a in ac if a.coalition == Coalition.BLUE]
    r = [a.id for a in ac if a.coalition == Coalition.RED]
    cfg = dataclasses.replace(CFG, signature=SignatureConfig(tables={F35_KEY: OLD_F35_TABLE}))
    w = World(ac, SimConfig(dt=0.5, max_time_s=240.0, seed=1000 + seed, sensor_config=cfg,
                            blue_doctrine=LEGACY, red_doctrine=LEGACY),
              TacticsController(TacticsGenome(), b), RedCAPController(r, mode=sc.red_mode),
              record=False)
    res = w.run()
    evs = [(round(e["t"], 2), e["type"], e.get("missile"), e.get("target"))
           for e in res.events if e["type"] != "rwr_mode"]
    pos = [(round(a.state.x, 3), round(a.state.y, 3), round(a.state.alt, 3)) for a in ac]
    return hashlib.sha1(json.dumps([evs, pos, len(res.sensor_events)]).encode()).hexdigest()[:16]


@pytest.mark.parametrize("seed", [0, 1])
def test_regression_defense_off_legacy_old_rcs_matches_pre_spec3(seed):
    assert _fingerprint(seed) == PRE_SPEC3[seed]
