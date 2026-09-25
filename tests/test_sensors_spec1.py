"""Spec 1 sensor model tests: radar Pd curve, field of regard, signature,
IRST, RWR, tracks (coast/drop, fire-control gating) and determinism."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from stealth_tactics.sim.aircraft import Aircraft, AircraftState, Coalition
from stealth_tactics.sim.sensor_config import (
    DEFAULT_SENSOR_CONFIG as CFG, F35_KEY, RED_KEY, SensorConfig,
)
from stealth_tactics.sim.sensors import (
    SensorModel, pd_curve, radar_pd, radar_r50, effective_rcs, irst_r50,
    irst_pd, rwr_range, rwr_detects,
)
from stealth_tactics.sim.tracks import TrackQuality
from stealth_tactics.sim.weapons import WeaponModel
from stealth_tactics.sim.world import World, SimConfig
from stealth_tactics.acmi.exporter import ACMIExporter

DEG = math.pi / 180.0


def blue(x=0.0, y=0.0, hdg=0.0, alt=8000.0, spd=250.0) -> Aircraft:
    return Aircraft.make_blue("B1", "F-35-1", AircraftState(x=x, y=y, alt=alt,
                                                          heading_rad=hdg, speed_mps=spd))


def red(x=0.0, y=0.0, hdg=math.pi, alt=8000.0, spd=250.0) -> Aircraft:
    return Aircraft.make_red("R1", "Red-1", AircraftState(x=x, y=y, alt=alt,
                                                        heading_rad=hdg, speed_mps=spd))


def _radar_hit_rate(obs, tgt, n=3000) -> float:
    hits = 0
    for seed in range(n):  # "many seeds": one scan per seed
        sm = SensorModel(np.random.default_rng(seed))
        hits += sm.radar_measure(obs, tgt) is not None
    return hits / n


def _run(sm: SensorModel, aircraft, t0: float, secs: float, dt: float = 0.5):
    t = t0
    evts = []
    while t < t0 + secs - 1e-9:
        evts += sm.update(aircraft, t)
        t += dt
    return t, evts


# ------------------------------------------------------------- radar -------
def test_pd_curve_shape() -> None:
    k = CFG.radar.pd_exponent
    assert pd_curve(50_000, 50_000, k) == pytest.approx(0.5)
    assert pd_curve(0.5 * 50_000, 50_000, k) > 0.99
    assert pd_curve(1.5 * 50_000, 50_000, k) < 0.05
    rs = np.linspace(1_000, 150_000, 200)
    pds = [pd_curve(r, 50_000, k) for r in rs]
    assert all(a >= b for a, b in zip(pds, pds[1:]))  # monotone


def test_radar_detection_statistics_over_seeds() -> None:
    b = blue()
    r50 = radar_r50(b, red(0, 10_000))  # Red isotropic -> 90 km
    assert r50 == pytest.approx(90_000.0)
    assert _radar_hit_rate(b, red(0, r50)) == pytest.approx(0.5, abs=0.04)
    assert _radar_hit_rate(b, red(0, 0.5 * r50)) > 0.97
    assert _radar_hit_rate(b, red(0, 1.4 * r50)) < 0.10


def test_field_of_regard_blocks_radar_and_irst() -> None:
    b = blue(hdg=0.0)
    beam = red(x=10_000.0, y=0.0)            # 90 deg off Blue's nose, 10 km
    above = red(x=0.0, y=2_000.0, alt=8000.0 + 6_000.0)  # ~72 deg elevation
    edge_in = red(x=10_000 * math.sin(59 * DEG), y=10_000 * math.cos(59 * DEG))
    assert radar_pd(b, beam) == 0.0 and irst_pd(b, beam) == 0.0
    assert radar_pd(b, above) == 0.0
    assert radar_pd(b, edge_in) > 0.99
    sm = SensorModel(np.random.default_rng(0))
    assert all(sm.radar_measure(b, beam) is None for _ in range(1000))


# --------------------------------------------------------- signature -------
def test_f35_signature_nose_beam_tail() -> None:
    b = blue(hdg=0.0)
    nose = effective_rcs(b, red(0, 30_000))
    beam = effective_rcs(b, red(30_000, 0))
    tail = effective_rcs(b, red(0, -30_000))
    assert nose == pytest.approx(0.05)
    assert 0.85 <= beam <= 1.0
    assert tail == pytest.approx(0.30)
    # smooth + monotone nose -> beam
    vals = []
    for a in np.linspace(0, 90, 91):
        obs = red(30_000 * math.sin(a * DEG), 30_000 * math.cos(a * DEG))
        vals.append(effective_rcs(b, obs))
    assert all(y1 >= y0 - 1e-12 for y0, y1 in zip(vals, vals[1:]))
    assert max(abs(y1 - y0) for y0, y1 in zip(vals, vals[1:])) < 0.03
    # Red isotropic
    r = red(0, 0, hdg=0.0)
    assert effective_rcs(r, blue(0, 30_000)) == effective_rcs(r, blue(30_000, 0)) == 1.0


def test_red_radar_r50_vs_f35_by_aspect() -> None:
    b = blue(hdg=0.0)
    r_nose = radar_r50(red(0, 50_000, hdg=math.pi), b)
    r_beam = radar_r50(red(50_000, 0, hdg=-math.pi / 2), b)
    assert r_nose == pytest.approx(70_000 * 0.05 ** 0.25)
    assert r_beam > 2 * r_nose


# -------------------------------------------------------------- IRST -------
def test_irst_bearing_accurate_range_poor() -> None:
    obs = red(0, 0, hdg=0.0)
    tgt = blue(0, 15_000, hdg=math.pi)  # nose-on, close -> Pd ~ 1
    sm = SensorModel(np.random.default_rng(11))
    brg_err, rng_err = [], []
    for _ in range(2000):
        m = sm.irst_measure(obs, tgt)
        if m is None:
            continue
        brg_err.append(math.degrees(math.atan2(math.sin(m["bearing"]), math.cos(m["bearing"]))))
        rng_err.append(m["range_est"] / m["true_range"] - 1.0)
    assert len(brg_err) > 1500
    assert np.std(brg_err) == pytest.approx(0.5, abs=0.1)   # ~0.5 deg
    assert 0.2 < np.std(rng_err) < 0.4                       # ~30 % range error
    # cross-range error at 15 km (~130 m) << range error (~4.5 km)
    assert 15_000 * math.radians(np.std(brg_err)) < 0.1 * 15_000 * np.std(rng_err)


def test_irst_tail_longer_than_nose_and_speed_effect() -> None:
    obs = red(0, 0, hdg=0.0)
    nose_tgt = blue(0, 40_000, hdg=math.pi)
    tail_tgt = blue(0, 40_000, hdg=0.0)
    assert irst_r50(obs, nose_tgt) == pytest.approx(CFG.irst.nose_r50_m[RED_KEY])
    assert irst_r50(obs, tail_tgt) == pytest.approx(CFG.irst.tail_r50_m[RED_KEY])
    assert irst_pd(obs, tail_tgt) > 0.8 > 0.2 > irst_pd(obs, nose_tgt)
    slow = blue(0, 40_000, hdg=math.pi, spd=150.0)
    fast = blue(0, 40_000, hdg=math.pi, spd=340.0)
    assert irst_r50(obs, fast) > irst_r50(obs, nose_tgt) > irst_r50(obs, slow)


def _no_red_radar_cfg() -> SensorConfig:
    radar = dataclasses.replace(CFG.radar, ref_range_m={F35_KEY: 90_000.0, RED_KEY: 1.0})
    return dataclasses.replace(CFG, radar=radar)


def test_irst_is_passive_and_cannot_support_shot() -> None:
    cfg = _no_red_radar_cfg()   # Red radar effectively off
    b = blue(0, 0, hdg=0.0)     # flying away from Red: Red is behind Blue
    r = red(0, -20_000, hdg=0.0)  # chasing, sees Blue's hot tail
    sm = SensorModel(np.random.default_rng(3), cfg)
    _run(sm, [b, r], 0.0, 10.0)
    red_store, blue_store = sm.stores[r.id], sm.stores[b.id]
    assert red_store.quality(b.id) == TrackQuality.IRST
    assert not red_store.is_fire_control(b.id)
    assert not WeaponModel.has_midcourse_support(r, b, sm.stores)
    assert b.id in red_store and r.id not in blue_store  # target got no warning


# --------------------------------------------------------------- RWR -------
def test_rwr_ranges() -> None:
    b = blue(0, 0, hdg=0.0)
    # Red RWR vs F-35 LPI radar: 0.6 * 90 km = 54 km (Red inside Blue's FOR)
    assert rwr_range(b) == pytest.approx(54_000.0)
    assert rwr_detects(red(0, 53_000), b)
    assert not rwr_detects(red(0, 55_000), b)
    assert not rwr_detects(red(20_000, 0), b)  # outside Blue radar FOR
    # Blue RWR vs Red radar: 1.5 * 70 km = 105 km (Blue inside Red's FOR)
    r = red(0, 0, hdg=math.pi)
    assert rwr_range(r) == pytest.approx(105_000.0)
    assert rwr_detects(blue(0, -104_000), r)
    assert not rwr_detects(blue(0, -106_000), r)
    # Blue RWR sees Red well beyond Red's own 50% range vs even a beam F-35
    assert rwr_range(r) > 1.4 * CFG.radar.ref_range_m[RED_KEY]


def test_rwr_every_step_radar_irst_every_second() -> None:
    b = blue(0, 0, hdg=0.0)
    r = red(0, 40_000, hdg=math.pi)
    sm = SensorModel(np.random.default_rng(5))
    _run(sm, [b, r], 0.0, 10.0)  # 20 steps of 0.5 s
    tr = sm.stores[b.id].get(r.id)
    assert tr.components["rwr"].hits == 20
    assert tr.components["radar"].hits <= 10
    assert tr.components["rwr"].range_est_m is None  # bearing-only


# ------------------------------------------------------------ tracks -------
def test_track_coasts_then_drops_after_10s() -> None:
    b = blue(0, 0, hdg=0.0)
    r = red(0, 30_000, hdg=math.pi)
    sm = SensorModel(np.random.default_rng(7))
    t, _ = _run(sm, [b, r], 0.0, 8.0)
    tr = sm.stores[b.id].get(r.id)
    s_short = tr.position_sigma(tr.last_t)
    t, _ = _run(sm, [b, r], t, 8.0)
    tr = sm.stores[b.id].get(r.id)
    assert tr.position_sigma(tr.last_t) < s_short  # shrinks with hold time
    # Blue turns cold: Red leaves every Blue sensor's FOR (and Blue leaves Red's
    # radar FOR too -> no RWR either once Red also turns away)
    b.state.heading_rad = math.pi
    r.state.heading_rad = 0.0
    last = tr.last_t
    t, evts = _run(sm, [b, r], t, 9.5)
    assert r.id in sm.stores[b.id]
    tr = sm.stores[b.id].get(r.id)
    assert tr.position_sigma(t) > tr.position_sigma(last)  # grows while coasting
    t, evts2 = _run(sm, [b, r], t, 2.0)
    assert r.id not in sm.stores[b.id]
    assert any(e["type"] == "track_lost" and e["observer"] == b.id for e in evts2)
    drop_t = next(e["t"] for e in evts2 if e["type"] == "track_lost")
    assert 10.0 < drop_t - last <= 10.5 + 1e-9


def test_fire_control_requires_hold_time_and_range() -> None:
    b = blue(0, 0, hdg=0.0)
    r = red(0, 40_000, hdg=math.pi)   # 0.44 R50: Pd ~ 1
    sm = SensorModel(np.random.default_rng(9))
    t, evts = _run(sm, [b, r], 0.0, 10.0)
    first = next(e["t"] for e in evts if e["type"] == "radar_detect" and e["observer"] == b.id)
    fc = next(e["t"] for e in evts if e["type"] == "fc_track" and e["observer"] == b.id)
    assert fc - first >= CFG.track.fc_min_hold_s - 1e-9
    assert sm.stores[b.id].is_fire_control(r.id)
    # Beyond 0.7 * R50 a radar track exists but is never fire-control (fraction
    # rule; Spec 2b gives Blue its own 50 NM gate, so disable it here)
    b2 = blue(0, 0, hdg=0.0)
    r2 = red(0, 0.8 * 90_000, hdg=math.pi)
    frac_cfg = dataclasses.replace(CFG, track=dataclasses.replace(CFG.track,
                                                                  fc_range_ref_m={}))
    sm2 = SensorModel(np.random.default_rng(9), frac_cfg)
    for _ in range(40):
        sm2.update([b2, r2], _ * 0.5)
        assert not sm2.stores[b2.id].is_fire_control(r2.id)
    assert sm2.stores[b2.id].quality(r2.id) == TrackQuality.RADAR


def test_world_launch_and_support_gated_by_fire_control() -> None:
    b = blue(0, 0, hdg=0.0)
    r = red(0, 44_000, hdg=math.pi)
    r.params = dataclasses.replace(r.params)  # don't mutate shared params

    def blue_ctl(w: World) -> None:  # try to shoot every step
        ac = w.get("B1")
        ac.cmd_fire, ac.fire_target = True, "R1"

    def red_ctl(w: World) -> None:
        pass

    w = World([b, r], SimConfig(dt=0.5, max_time_s=6.0, seed=1), blue_ctl, red_ctl)
    res = w.run()
    launches = [e for e in res.events if e["type"] == "launch"]
    fc = [e for e in res.sensor_events if e["type"] == "fc_track" and e["observer"] == "B1"]
    assert fc and launches
    assert launches[0]["t"] >= fc[0]["t"] >= CFG.track.fc_min_hold_s
    # Radar-only (immature) or RWR track never supports a missile
    sm = SensorModel(np.random.default_rng(0))
    sm.update([b, r], 0.0)
    assert sm.stores["B1"].quality("R1") == TrackQuality.RADAR
    assert not WeaponModel.has_midcourse_support(b, r, sm.stores)


# ------------------------------------------------------- determinism -------
def _headon_world(seed: int) -> World:
    b = blue(0, -60_000, hdg=0.0)
    r = red(0, 60_000, hdg=math.pi)
    noop = lambda w: None  # noqa: E731
    return World([b, r], SimConfig(dt=0.5, max_time_s=120.0, seed=seed), noop, noop)


def test_sensor_determinism_same_seed() -> None:
    a = _headon_world(42).run().sensor_events
    b = _headon_world(42).run().sensor_events
    c = _headon_world(43).run().sensor_events
    assert a == b and len(a) > 0
    assert [(e["t"], e["type"]) for e in a] != [(e["t"], e["type"]) for e in c]


def test_acmi_contains_locks_and_events(tmp_path: Path) -> None:
    w = _headon_world(1)
    w.record = True
    res = w.run()
    out = ACMIExporter(title="t").export(res.frames, tmp_path / "x.txt.acmi")
    text = out.read_text()
    assert "LockedTarget=" in text and "LockedTargetMode=1" in text
    assert "0,Event=Bookmark|" in text and "0,Event=Message|" in text
    assert "Name=F-35A" in text


def test_replay_beam_then_hot_seed1() -> None:
    """Replay 3: beam at 45 NM, hot after 60 s (no Red lock); Red's radar
    track then drops (nose aspect) and Red only locks much closer."""
    from stealth_tactics.analysis.replays import run_beam_then_hot, BTH_BEAM_RANGE_M
    res = run_beam_then_hot(seed=1, record=False)
    assert res.turn_t is not None and res.hot_t is not None
    assert res.hot_t - res.turn_t == pytest.approx(60.0)
    assert res.hot_reason == "60 s elapsed"
    assert res.hot_range_m < BTH_BEAM_RANGE_M
    red = [e for e in res.world.sensor_events if e["observer"] == "R1"]
    assert any(e["type"] == "radar_detect" and e["t"] < res.hot_t for e in red)
    assert any(e["type"] == "radar_lost" and e["t"] > res.hot_t for e in red)
    fc = [e for e in red if e["type"] == "fc_track"]
    assert fc and fc[0]["t"] > res.hot_t and fc[0]["range_m"] < 0.7 * 70_000 * 0.05 ** 0.25 + 2_000
