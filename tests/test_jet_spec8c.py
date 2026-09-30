"""Spec 8c: thrust retune (Blue climbs and accelerates, turns still bleed; Red ~25% better).

Spec 8d: the climb schedules fly a direct climb-rate command and gamma is a
load-factor-limited state; the 15 deg check starts established on the climb
angle (entry from level costs ~0.05 Mach at 35 kft, reported in the sweep)."""

from __future__ import annotations

import pytest

from stealth_tactics.analysis.aircraft_sweep import (accel_time_s, bleed_run, bleeds_ok,
                                                     climb_run, gamma_climb_margin,
                                                     gamma_climb_run, lift_limit_g,
                                                     red_blue_ratios, sustained_g)
from stealth_tactics.sim.aircraft import F35_PARAMS, RED_FIGHTER_PARAMS
from stealth_tactics.sim.sensor_config import BLUE_ENERGY, RED_ENERGY


# ---------------------------------------------------------- target 1 --------
def test_blue_sustains_m1_at_15deg_climb_35kft():
    """T - D >= W sin(15 deg) at 35 kft / M1.0 (n = cos 15), and the integrator
    agrees: speed non-decreasing over a 5 s 15 deg climb (~15,000 ft/min)."""
    assert gamma_climb_margin(BLUE_ENERGY, 35_000.0, 1.0, 15.0) >= 1.0
    run = gamma_climb_run(BLUE_ENERGY, 35_000.0, 1.0, 15.0, t_s=5.0)
    assert run["final_mach"] >= 1.0
    assert 14_000.0 <= run["climb_fpm"] <= 16_000.0


# ---------------------------------------------------------- target 2 --------
@pytest.mark.parametrize("schedule", ["max_rate", "gamma", "mach_hold"])
def test_blue_climbs_30_to_40kft_under_60s(schedule):
    r = climb_run(BLUE_ENERGY, schedule, 30_000.0, 40_000.0, start_mach=1.0, gamma_deg=15.0)
    assert r["t_s"] is not None and r["t_s"] < 60.0, r
    assert r["end_mach"] > 0.9


def test_blue_plain_full_power_climb_from_m09():
    r = climb_run(BLUE_ENERGY, "max_rate", 30_000.0, 40_000.0, start_mach=0.9)
    assert r["t_s"] is not None and r["t_s"] < 60.0, r


# ---------------------------------------------------------- target 3 --------
def test_blue_level_accel_40kft_allowed_but_slow():
    t = accel_time_s(BLUE_ENERGY, 40_000.0, 0.9, 1.2)
    assert t is not None, "Blue must be able to reach M1.2 level at 40 kft"
    assert t >= 120.0, t
    assert t <= 300.0, t


def test_red_level_accel_40kft_faster_than_blue_and_reaches_ceiling():
    tb = accel_time_s(BLUE_ENERGY, 40_000.0, 0.9, 1.2)
    tr = accel_time_s(RED_ENERGY, 40_000.0, 0.9, 1.2)
    assert tr is not None and tr < tb
    assert accel_time_s(RED_ENERGY, 40_000.0, 0.9, RED_ENERGY.max_mach) is not None


# ---------------------------------------------------------- target 4 --------
def test_max_g_at_altitude_unchanged_and_turns_bleed():
    # lift limit / n_max unchanged for Blue: ~3.13 g at 40 kft M0.9, n_max 7
    assert BLUE_ENERGY.CLmax == 1.3 and BLUE_ENERGY.n_max == 7.0
    assert lift_limit_g(BLUE_ENERGY, 40_000.0, 0.9) == pytest.approx(3.13, abs=0.01)
    # sustained well below max at altitude -> a max-g turn bleeds
    assert sustained_g(BLUE_ENERGY, 40_000.0, 0.9) < 0.75 * lift_limit_g(BLUE_ENERGY, 40_000.0, 0.9)
    for e in (BLUE_ENERGY, RED_ENERGY):
        assert bleeds_ok(bleed_run(energy=e))


def test_mach_ceilings_kept():
    assert BLUE_ENERGY.max_mach == 1.2 and RED_ENERGY.max_mach == 1.4


# ---------------------------------------------------------- target 5 --------
def test_red_about_25pct_better():
    for q in red_blue_ratios():
        assert 1.15 <= q["ps_ratio"] <= 1.45, q
        assert 1.20 <= q["sust_g_ratio"] <= 1.30, q
        assert 1.10 <= q["max_g_ratio"] <= 1.25, q
    rs = [q["ps_ratio"] for q in red_blue_ratios()]
    assert 1.2 <= sum(rs) / len(rs) <= 1.3
    # Red also sustains the 35 kft 15 deg climb with ~25% more margin
    rb = gamma_climb_margin(BLUE_ENERGY)
    rr = gamma_climb_margin(RED_ENERGY)
    assert 1.15 <= rr / rb <= 1.35


def test_red_climbs_faster():
    tb = climb_run(BLUE_ENERGY, "mach_hold")["t_s"]
    tr = climb_run(RED_ENERGY, "mach_hold")["t_s"]
    assert tr < tb
    assert RED_FIGHTER_PARAMS.max_climb_rate_mps == pytest.approx(1.25 * F35_PARAMS.max_climb_rate_mps)
