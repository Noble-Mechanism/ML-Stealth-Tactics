"""Spec 8b: jet lift limit and transonic drag rise (jet model only).

Spec 8c retuned thrust and the drag-rise shape; the hump test now checks that
level acceleration at 40 kft is slow rather than blocked (see test_jet_spec8c.py).
"""

from __future__ import annotations

import math

import pytest

from stealth_tactics.analysis.aircraft_sweep import (accel_time_s, bleed_run, bleeds_ok,
                                                     dive_climb_run, level_accel_run,
                                                     lift_limit_g, top_speeds)
from stealth_tactics.sim.aircraft import Aircraft, AircraftState, integrate_aircraft, F35_PARAMS
from stealth_tactics.sim.missile_kinematics import atmosphere
from stealth_tactics.sim.sensor_config import BLUE_ENERGY, RED_ENERGY

FT = 0.3048
TYPES = [("blue", BLUE_ENERGY), ("red", RED_ENERGY)]


def test_clmax_placeholders_per_type():
    # Spec 8c: Red bumped 1.4 -> 1.5 (max g at altitude ~15% above Blue)
    assert BLUE_ENERGY.CLmax == 1.3 and RED_ENERGY.CLmax == 1.5


def test_cd0_transonic_shape():
    for _, e in TYPES:
        mp = e.cd0_peak_mach
        assert e.cd0_factor(0.5) == 1.0 and e.cd0_factor(e.cd0_rise_mach) == 1.0
        assert e.cd0_factor(mp) == pytest.approx(e.cd0_peak_factor)
        assert 1.0 < e.cd0_factor(0.95) < e.cd0_factor(1.0) < e.cd0_peak_factor
        assert e.cd0_factor(mp + 0.1) < e.cd0_factor(mp)      # declines after the peak
        assert e.cd0_factor(1.4) >= e.cd0_supersonic_floor


def test_lift_limit_caps_g_at_40kft_mach09():
    """q S CLmax / W at 40 kft / Mach 0.9 is ~3 g (well under n_max 7)."""
    g = lift_limit_g(BLUE_ENERGY, 40_000.0, 0.9)
    assert 2.8 <= g <= 3.4, g
    # and the integrator really turns no faster than that
    alt = 40_000 * FT
    V0 = 0.9 * atmosphere(alt)[1]
    ac = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt, 0.0, V0))
    ac.params = F35_PARAMS
    ac.cmd_speed_mps, ac.cmd_alt_m, ac.cmd_heading_rad = V0, alt, math.pi / 2
    h0 = ac.state.heading_rad
    integrate_aircraft(ac, 0.05)
    omega = abs(ac.state.heading_rad - h0) / 0.05
    n = math.sqrt(1.0 + (V0 * omega / 9.80665) ** 2)
    assert n == pytest.approx(g, rel=1e-3)
    assert n < 0.5 * BLUE_ENERGY.n_max


def test_slow_jet_up_high_must_sink():
    """Below the 1 g lift speed a jet cannot hold altitude, even commanded to."""
    alt = 40_000 * FT
    ac = Aircraft.make_blue("B1", "F-35-1",
                            AircraftState(0.0, 0.0, alt, 0.0, BLUE_ENERGY.min_speed_mps + 5.0))
    ac.params = F35_PARAMS
    ac.cmd_speed_mps, ac.cmd_alt_m, ac.cmd_heading_rad = 1.0e4, alt, 0.0
    assert lift_limit_g(BLUE_ENERGY, 40_000.0, ac.state.speed_mps / atmosphere(alt)[1]) < 1.0
    for _ in range(100):
        integrate_aircraft(ac, 0.05)
    assert ac.state.alt < alt - 50.0


@pytest.mark.parametrize("name,e", TYPES)
def test_transonic_hump_slows_level_accel_at_40kft(name, e):
    """Spec 8c: the hump no longer blocks level acceleration at 40 kft, but the
    level excess thrust bottoms out past Mach 1.0 at a small fraction of its
    Mach 0.9 value (the slow part of the run)."""
    from stealth_tactics.analysis.aircraft_sweep import level_excess_n
    t = accel_time_s(e, 40_000.0, 0.9, 1.2)
    assert t is not None and t > 60.0, t
    ms = [0.9 + 0.005 * i for i in range(61)]
    ex = [level_excess_n(e, 40_000.0, m) for m in ms]
    m_min = ms[ex.index(min(ex))]
    assert m_min > 1.0 and 0.0 < min(ex) < 0.25 * ex[0]


@pytest.mark.parametrize("name,e", TYPES)
def test_supersonic_at_40kft_sustains_mach_1_2(name, e):
    rep = level_accel_run(e, alt_ft=40_000.0, start_mach=1.2, t_s=300.0)
    assert rep["final_mach"] >= 1.19, rep
    # Spec 8c: no hump gap at 40 kft any more (level accel is slow, not blocked)
    assert top_speeds(e, 40_000.0)["max_sustained"] >= 1.2


@pytest.mark.parametrize("name,e", TYPES)
def test_dive_and_climb_reaches_supersonic_at_40kft(name, e):
    rep = dive_climb_run(e)
    assert rep["t_supersonic"] is not None and rep["alt_supersonic_ft"] < 40_000.0
    assert rep["t_arrive"] is not None and rep["mach_arrive"] > 1.15, rep
    assert rep["final_alt_ft"] == pytest.approx(40_000.0) and rep["final_mach"] >= 1.19


@pytest.mark.parametrize("name,e", TYPES)
def test_bleed_gate_lift_limited(name, e):
    """Spec 8c: with more thrust the old 0.60-0.80 band is not forced; the
    lift-limited turn at 40 kft must still clearly lose speed."""
    rep = bleed_run(energy=e, alt_ft=40_000.0, start_mach=0.9, t_s=20.0)
    assert rep["n_max"] == e.n_max          # no override: lift limit does the capping
    assert bleeds_ok(rep), rep["final_mach"]
