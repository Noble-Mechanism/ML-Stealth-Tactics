"""Spec 2b: missile coast after support lapse + Blue 50 NM fire-control gate."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, AircraftState
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG as CFG, NM_M
from stealth_tactics.sim.sensors import SensorModel, radar_r50
from stealth_tactics.sim.weapons import WeaponModel, ACQUISITION_RANGE_M


# Spec 3a: these tests isolate the support / coast rules from the energy model,
# so the Mach-1.2 kinematic defeat is switched off (the missile can coast for
# 30+ s at 8 km without being defeated) and the launch gate is bypassed with
# ``spawn`` (the gate is now the Rmax table; a cold target at 40 km is outside it).
NO_DEFEAT = dataclasses.replace(CFG, missile_kinematics=dataclasses.replace(
    CFG.missile_kinematics, defeat_min_mach=0.0, defeat_on_opening=False))


def _setup(red_y=40_000.0, red_hdg=0.0, red_speed=250.0, pk=1.0):
    blue = Aircraft.make_blue("B1", "Blue1", AircraftState(0.0, 0.0, 8000.0, 0.0, 250.0))
    red = Aircraft.make_red("R1", "Red1", AircraftState(0.0, 40_000.0, 8000.0, red_hdg, red_speed))
    w = WeaponModel(np.random.default_rng(0), NO_DEFEAT)
    m = w.spawn(blue, red)
    assert m is not None and not m.autonomous
    m.pk = pk
    m.remaining_time_s = 200.0
    red.state.y = red_y                     # (then move the target)
    m.last_sup_pos = red.state.position()
    return w, m, blue, red, {blue.id: blue, red.id: red}


SUP = {"B1": {"R1"}, "R1": set()}      # legacy set form: B1 holds FC on R1
NONE = {"B1": set(), "R1": set()}


def _step(w, m, by_id, tracks, n=1, t0=0.0):
    for i in range(n):
        w.step([m], by_id, 0.5, tracks=tracks, t=t0 + i * 0.5)


def test_coast_then_support_regained():
    w, m, blue, red, by_id = _setup(red_y=80_000.0, red_speed=0.0)
    _step(w, m, by_id, NONE, 4)
    assert m.alive and m.coasting and m.t_since_update == 2.0
    _step(w, m, by_id, SUP, 1, t0=2.0)
    assert not m.coasting and m.t_since_update == 0.0 and m.supporter_id == "B1"
    _step(w, m, by_id, NONE, 2, t0=2.5)
    types = [e["type"] for e in w.events]
    assert types.count("support_lost") == 2 and "support_regained" in types
    assert m.t_unsupported == 3.0 and m.t_since_update == 1.0   # cumulative vs since update
    assert any("regained by B1" in e["text"] for e in w.events)


def test_pk_factor_at_autonomy_uses_time_since_update():
    # Red flies straight away at 250 m/s: extrapolation is exact, so no basket loss.
    # Spec 3a: start at 35 km (was 40 km); the physical missile is slower than
    # the old constant 900 m/s (3a fix: coast timeout now 40 s, time scale 40 s).
    w, m, blue, red, by_id = _setup(red_y=35_000.0, red_hdg=0.0, red_speed=250.0, pk=0.8)
    t = 0.0
    while m.alive and not m.autonomous:
        red.state.y += 250.0 * 0.5
        _step(w, m, by_id, NONE, 1, t0=t)
        t += 0.5
    assert m.autonomous and m.outcome == ""
    ts = m.t_since_update
    assert 5.0 < ts < 20.0
    # 3a fix coast factor: exp(-(e/2 km)^2) * (1 - 0.15 min(t_gap/40 s, 1)), e ~ 0 here
    expect = CFG.missile.coast_pk_factor(m.aim_err_at_auto_m, ts)
    assert abs(m.pk_factor - expect) < 1e-12
    assert abs(m.pk_factor - (1 - 0.15 * ts / 40.0)) < 1e-6
    assert abs(m.pk - 0.8 * m.pk_factor) < 1e-12
    auto = [e for e in w.events if e["type"] == "autonomous"][0]
    assert auto["pk_factor"] == m.pk_factor and "Pk factor" in auto["text"]
    assert m.aim_err_at_auto_m < 1.0


def test_supported_missile_has_pk_factor_one():
    w, m, blue, red, by_id = _setup(red_y=40_000.0, red_speed=0.0, pk=0.7)
    while m.alive and not m.autonomous:
        _step(w, m, by_id, SUP, 1)
    assert m.pk_factor == 1.0 and m.pk == 0.7


def test_coast_timeout_after_40s_continuous():
    """3a fix: coast timeout 40 s (was 20 s)."""
    assert CFG.missile.coast_timeout_s == 40.0
    w, m, blue, red, by_id = _setup(red_y=400_000.0, red_speed=0.0)
    _step(w, m, by_id, NONE, 60)          # 30 s
    _step(w, m, by_id, SUP, 1)            # resets continuous timer
    _step(w, m, by_id, NONE, 78)          # 39 s: still alive (cumulative 69 s)
    assert m.alive and m.t_unsupported == 69.0
    _step(w, m, by_id, NONE, 4)
    assert not m.alive and m.outcome == "lost_coast_timeout"
    assert any(e["type"] == "lost_coast_timeout" and "coast timeout" in e["text"]
               for e in w.events)


def test_seeker_basket_loss_and_inside_basket_ok():
    # Aim point extrapolated from a stationary target; target then jinks 8 km
    w, m, blue, red, by_id = _setup(red_y=40_000.0, red_speed=0.0)
    _step(w, m, by_id, NONE, 1)
    red.state.x = 8_000.0
    red.state.y = m.pos.y + 20_000.0      # within 15 NM of the TRUE target
    _step(w, m, by_id, NONE, 1)
    assert not m.alive and m.outcome == "lost_basket"
    assert m.aim_err_at_auto_m > CFG.missile.seeker_basket_m
    # Small jink (2 km) stays inside the basket -> autonomous
    w, m, blue, red, by_id = _setup(red_y=40_000.0, red_speed=0.0)
    _step(w, m, by_id, NONE, 1)
    red.state.x = 2_000.0
    _step(w, m, by_id, NONE, 1)
    while m.alive and not m.autonomous:
        _step(w, m, by_id, NONE, 1)
    assert m.autonomous and m.aim_err_at_auto_m < CFG.missile.seeker_basket_m
    assert math.hypot(red.state.x - m.pos.x, red.state.y - m.pos.y) <= ACQUISITION_RANGE_M + 1


def test_blue_fc_gate_50nm_scaled_red_unchanged():
    sm = SensorModel(np.random.default_rng(0))
    b = Aircraft.make_blue("B1", "B", AircraftState(0.0, 0.0, 9000.0, 0.0, 260.0))
    r = Aircraft.make_red("R1", "R", AircraftState(0.0, 90_000.0, 9000.0, math.pi, 255.0))
    r50_br = radar_r50(b, r)                    # Red is isotropic RCS 1.0
    assert abs(sm.fc_gate_m(b, r50_br) - 50 * NM_M) < 1.0
    # Scaled by rcs^0.25 through R50
    assert abs(sm.fc_gate_m(b, 0.5 * r50_br) - 0.5 * 92_600.0) < 1.0
    r50_rb = radar_r50(r, b)
    assert abs(sm.fc_gate_m(r, r50_rb) - CFG.track.fc_range_frac * r50_rb) < 1e-9


def test_blue_gets_fc_beyond_old_gate():
    """Static Blue vs Red at 80 km (43 NM, > 0.7*R50 = 63 km, < 92.6 km)."""
    b = Aircraft.make_blue("B1", "B", AircraftState(0.0, 0.0, 9000.0, 0.0, 0.0))
    r = Aircraft.make_red("R1", "R", AircraftState(0.0, 80_000.0, 9000.0, math.pi, 0.0))
    sm = SensorModel(np.random.default_rng(3))
    got = False
    for i in range(120):
        sm.update([b, r], i * 0.5)
        got = got or sm.stores["B1"].is_fire_control("R1")
    assert got
    old = dataclasses.replace(CFG, track=dataclasses.replace(CFG.track, fc_range_ref_m={}))
    sm2 = SensorModel(np.random.default_rng(3), old)
    for i in range(120):
        sm2.update([b, r], i * 0.5)
        assert not sm2.stores["B1"].is_fire_control("R1")
