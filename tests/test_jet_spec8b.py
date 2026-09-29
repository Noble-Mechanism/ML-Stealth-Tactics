"""Spec 8b: jet lift limit and transonic drag rise (jet model only)."""

from __future__ import annotations

import math

import pytest

from stealth_tactics.analysis.aircraft_sweep import (bleed_run, dive_climb_run, gate_ok,
                                                     level_accel_run, lift_limit_g,
                                                     top_speeds)
from stealth_tactics.sim.aircraft import Aircraft, AircraftState, integrate_aircraft, F35_PARAMS
from stealth_tactics.sim.missile_kinematics import atmosphere
from stealth_tactics.sim.sensor_config import BLUE_ENERGY, RED_ENERGY

FT = 0.3048
TYPES = [("blue", BLUE_ENERGY), ("red", RED_ENERGY)]


def test_clmax_placeholders_per_type():
    assert BLUE_ENERGY.CLmax == 1.3 and RED_ENERGY.CLmax == 1.4


def test_cd0_transonic_shape():
    for _, e in TYPES:
        assert e.cd0_factor(0.5) == 1.0 and e.cd0_factor(0.85) == 1.0
        assert e.cd0_factor(1.05) == pytest.approx(2.2)
        assert 1.0 < e.cd0_factor(0.95) < 2.2
        assert e.cd0_factor(1.2) < e.cd0_factor(1.05)        # declines after the peak
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
def test_transonic_hump_blocks_level_accel_at_40kft(name, e):
    rep = level_accel_run(e, alt_ft=40_000.0, start_mach=0.9, t_s=300.0)
    assert rep["peak_mach"] < 1.0, rep
    assert rep["final_alt_ft"] == pytest.approx(40_000.0)
    assert top_speeds(e, 40_000.0)["from_subsonic"] < 1.0


@pytest.mark.parametrize("name,e", TYPES)
def test_supersonic_at_40kft_sustains_mach_1_2(name, e):
    rep = level_accel_run(e, alt_ft=40_000.0, start_mach=1.2, t_s=300.0)
    assert rep["final_mach"] >= 1.19, rep
    assert top_speeds(e, 40_000.0)["back_side"] < 1.2


@pytest.mark.parametrize("name,e", TYPES)
def test_dive_and_climb_reaches_supersonic_at_40kft(name, e):
    rep = dive_climb_run(e)
    assert rep["t_supersonic"] is not None and rep["alt_supersonic_ft"] < 40_000.0
    assert rep["t_arrive"] is not None and rep["mach_arrive"] > 1.15, rep
    assert rep["final_alt_ft"] == pytest.approx(40_000.0) and rep["final_mach"] >= 1.19


@pytest.mark.parametrize("name,e", TYPES)
def test_bleed_gate_lift_limited(name, e):
    rep = bleed_run(energy=e, alt_ft=40_000.0, start_mach=0.9, t_s=20.0)
    assert rep["n_max"] == e.n_max          # no override: lift limit does the capping
    assert gate_ok(rep), rep["final_mach"]
