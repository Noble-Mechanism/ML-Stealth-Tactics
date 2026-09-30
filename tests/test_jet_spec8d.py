"""Spec 8d: flight-path angle dynamics (no free vertical jinking) and the
look-up launch fix (envelope uses the shooter's altitude when the target is
higher). See docs/specs/08d-flight-path-and-launch.md.

Spec 8e (rolled pull, lift-vector rate limits) changed the default model; the
tests that check 8d-specific allocation details (wings-level 0 g push floor,
vertical-first split, instantaneous load factor) fly the airframe with
spec8d_energy(). The 8e equivalents are in tests/test_jet_spec8e.py."""

from __future__ import annotations

import math

import pytest

from stealth_tactics.analysis.aircraft_sweep import lift_limit_g
from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, F35_PARAMS,
                                          RED_FIGHTER_PARAMS, integrate_aircraft)
from stealth_tactics.sim.missile_envelope import MissileEnvelope
from stealth_tactics.sim.missile_kinematics import atmosphere
import dataclasses

from stealth_tactics.sim.sensor_config import BLUE_ENERGY, RED_ENERGY, spec8d_energy

G0 = 9.80665
FT = 0.3048


def _jet(alt_ft, mach, red=False, gamma_deg=0.0, model8d=False):
    alt = alt_ft * FT
    V = mach * atmosphere(alt)[1]
    mk = Aircraft.make_red if red else Aircraft.make_blue
    ac = mk("J", "J", AircraftState(0.0, 0.0, alt, 0.0, V))
    ac.params = RED_FIGHTER_PARAMS if red else F35_PARAMS
    if model8d:
        ac.params = dataclasses.replace(ac.params, energy=spec8d_energy(ac.params.energy))
    ac.cmd_heading_rad, ac.cmd_speed_mps, ac.cmd_alt_m = 0.0, V, alt
    ac.gamma_rad = math.radians(gamma_deg)
    return ac


def _n_cap(ac):
    e = ac.params.energy
    rho, _ = atmosphere(ac.state.alt)
    qS = 0.5 * rho * ac.state.speed_mps ** 2 * e.S_m2
    return min(e.n_max, qS * e.CLmax / (e.mass_kg * G0))


def _step(ac, dt):
    """One integrator step; returns (n_v, n_h, n_cap) actually used."""
    V0, g0, h0, nc = ac.state.speed_mps, ac.gamma_rad, ac.state.heading_rad, _n_cap(ac)
    integrate_aircraft(ac, dt)
    n_v = math.cos(g0) + V0 * (ac.gamma_rad - g0) / (G0 * dt)
    dpsi = (ac.state.heading_rad - h0 + math.pi) % (2 * math.pi) - math.pi
    n_h = V0 * abs(dpsi) / dt * max(math.cos(g0), 0.05) / G0
    return n_v, n_h, nc


# ------------------------------------------------------------ A: gamma rate --
@pytest.mark.parametrize("alt_ft,mach", [(15_000, 0.9), (25_000, 0.8), (35_000, 1.0), (42_000, 0.6)])
def test_gamma_rate_limited_by_load_factor(alt_ft, mach):
    """Pull-up rate <= g (n_cap - cos gamma) / V; push-over floor 0 g."""
    ac = _jet(alt_ft, mach, model8d=True)
    ac.cmd_alt_m = ac.state.alt + 5000.0
    for _ in range(40):
        n_v, _, nc = _step(ac, 0.05)
        assert n_v <= nc + 1e-6
    # first step from level cannot reach the demanded angle instantly
    ac2 = _jet(alt_ft, mach, model8d=True)
    ac2.cmd_alt_m = ac2.state.alt + 5000.0
    V = ac2.state.speed_mps
    nc = _n_cap(ac2)
    integrate_aircraft(ac2, 0.5)
    assert ac2.gamma_rad <= G0 * (nc - 1.0) / V * 0.5 + 1e-9
    # push-over: gamma decreases no faster than 0 g allows
    ac3 = _jet(alt_ft, mach, gamma_deg=10.0, model8d=True)
    ac3.cmd_alt_m = ac3.state.alt - 5000.0
    for _ in range(40):
        n_v, _, _ = _step(ac3, 0.05)
        assert n_v >= BLUE_ENERGY.n_pushover_min - 1e-6


@pytest.mark.parametrize("alt_ft,mach", [(42_000, 0.53), (42_000, 0.6), (25_000, 0.8), (15_000, 0.9)])
def test_no_free_jinking_alternating_climb_sink(alt_ft, mach):
    """Policy flips cmd_alt between max and min every 1 s (world dt 0.5 s).
    Before 8d the jet reversed +-90 m/s vertical speed every step (~37 g of
    vertical acceleration at ~1 g available). Now the vertical load factor
    stays inside [0, n_cap] and altitude acceleration stays bounded."""
    ac = _jet(alt_ft, mach, model8d=True)
    dt = 0.5
    vz, n_caps, alts = [], [], []
    alt_start = ac.state.alt
    for i in range(120):
        ac.cmd_alt_m = 15_000.0 if int(i * dt) % 2 == 0 else 100.0
        a0 = ac.state.alt
        n_v, n_h, nc = _step(ac, dt)
        assert BLUE_ENERGY.n_pushover_min - 1e-6 <= n_v <= nc + 1e-6
        n_caps.append(nc)
        vz.append((ac.state.alt - a0) / dt)
        alts.append(ac.state.alt)
    az_g = max(abs(vz[i] - vz[i - 1]) / dt / G0 for i in range(1, len(vz)))
    # vertical acceleration <= (n_cap + 1) g plus the along-track term
    assert az_g <= max(n_caps) + 1.0 + 0.5, az_g
    if alt_ft == 42_000 and mach == 0.53:
        # at the 1 g stall line the jet cannot pull: no climb at first (it
        # must dive for speed before it can climb again), and no +90 m/s
        # climb reversals
        assert max(alts[:8]) <= alt_start + 1.0
        assert max(vz) < 60.0
        assert az_g < 2.5


def test_shared_g_budget_with_turn():
    """Total load factor sqrt(n_v^2 + n_h^2) <= n_cap; vertical demand first,
    so a pull-up turns slower than a level lift-limited turn."""
    ac = _jet(25_000, 0.8, model8d=True)
    ac.cmd_alt_m = ac.state.alt + 3000.0
    ac.cmd_heading_rad = math.pi / 2
    first = None
    for _ in range(60):
        n_v, n_h, nc = _step(ac, 0.05)
        assert math.hypot(n_v, n_h) <= nc + 1e-6
        if first is None:
            first = (n_v, n_h, nc)
    lvl = _jet(25_000, 0.8, model8d=True)
    lvl.cmd_heading_rad = math.pi / 2
    _, n_h_lvl, nc_l = _step(lvl, 0.05)
    assert n_h_lvl == pytest.approx(math.sqrt(nc_l ** 2 - 1.0), rel=1e-6)
    assert first[1] < n_h_lvl - 0.5          # pull-up took most of the budget


def test_induced_drag_uses_total_load_factor():
    ac = _jet(25_000, 0.8, model8d=True)
    ac.cmd_speed_mps = 1.0e4
    ac.cmd_alt_m = ac.state.alt + 3000.0
    e = ac.params.energy
    rho, a = atmosphere(ac.state.alt)
    V0 = ac.state.speed_mps
    qS = 0.5 * rho * V0 * V0 * e.S_m2
    W = e.mass_kg * G0
    d0 = qS * e.Cd0 * e.cd0_factor(V0 / a)
    kind = e.k_induced * W * W / qS
    dt = 0.05
    n_v, n_h, _ = _step(ac, dt)
    assert n_v > 2.0 and n_h == 0.0
    expect = V0 + ((e.thrust_n(rho, V0 / a) - d0 - kind * n_v ** 2) / e.mass_kg
                   - G0 * math.sin(ac.gamma_rad)) * dt
    assert ac.state.speed_mps == pytest.approx(expect, rel=1e-9)
    # and more speed lost than the same step at 1 g level
    lvl = _jet(25_000, 0.8, model8d=True)
    lvl.cmd_speed_mps = 1.0e4
    integrate_aircraft(lvl, dt)
    assert ac.state.speed_mps < lvl.state.speed_mps - 0.1


def test_nose_drops_below_1g():
    """Lift below cos(gamma): gamma decreases at g (n_lift - 1) / V from level,
    even when commanded to hold or climb (replaces the 8b ad-hoc sink)."""
    alt = 42_000
    ac = _jet(alt, 0.5)
    nl = lift_limit_g(BLUE_ENERGY, alt, 0.5)
    assert nl < 1.0
    ac.cmd_alt_m = ac.state.alt + 1000.0
    ac.cmd_speed_mps = 1.0e4
    V = ac.state.speed_mps
    integrate_aircraft(ac, 0.05)
    assert ac.gamma_rad == pytest.approx(G0 * (nl - 1.0) / V * 0.05, rel=1e-6)
    alt0 = ac.state.alt
    for _ in range(200):
        integrate_aircraft(ac, 0.05)
        assert ac.state.alt <= alt0 + 1e-9      # never climbs while below 1 g
    assert ac.state.alt < alt0 - 10.0 and ac.gamma_rad < 0.0


def test_alt_hold_levels_off_without_big_overshoot():
    for red in (False, True):
        ac = _jet(25_000, 0.8, red=red)
        target = ac.state.alt + 1500.0
        ac.cmd_alt_m = target
        ac.cmd_speed_mps = 1.0e4
        peak = ac.state.alt
        for _ in range(int(120 / 0.5)):
            integrate_aircraft(ac, 0.5)
            peak = max(peak, ac.state.alt)
        assert abs(ac.state.alt - target) < 5.0
        assert peak - target < 50.0
        assert abs(ac.gamma_rad) < math.radians(0.5)


def test_max_climb_rate_cap_kept():
    for red, cap in ((False, F35_PARAMS.max_climb_rate_mps), (True, RED_FIGHTER_PARAMS.max_climb_rate_mps)):
        ac = _jet(15_000, 0.9, red=red)
        ac.cmd_alt_m = 14_000.0
        ac.cmd_speed_mps = 1.0e4
        vmax = 0.0
        for _ in range(200):
            a0 = ac.state.alt
            integrate_aircraft(ac, 0.05)
            vmax = max(vmax, (ac.state.alt - a0) / 0.05)
        assert vmax <= cap + 1e-6 and vmax > 0.9 * cap


def test_gamma_state_defaults_and_climb_rate_command():
    ac = _jet(30_000, 0.9)
    assert ac.gamma_rad == 0.0 and ac.cmd_climb_rate_mps is None
    ac.cmd_climb_rate_mps = 40.0
    for _ in range(200):
        integrate_aircraft(ac, 0.05)
    vs = ac.state.speed_mps * math.sin(ac.gamma_rad)
    assert vs == pytest.approx(40.0, abs=0.5)


# ------------------------------------------------------------ B: launch fix --
def test_envelope_uses_shooter_alt_when_target_higher():
    lo = AircraftState(0.0, 0.0, 6900.0, math.pi / 2, 0.9 * atmosphere(6900.0)[1])
    hi = AircraftState(30_000.0, 0.0, 12_650.0, -math.pi / 2, 0.53 * atmosphere(12_650.0)[1])
    alt, *_ = MissileEnvelope.geometry(lo, hi)          # look-up shot
    assert alt == pytest.approx(6900.0)
    alt, *_ = MissileEnvelope.geometry(hi, lo)          # look-down shot keeps the mean
    assert alt == pytest.approx(0.5 * (6900.0 + 12_650.0))
    same = AircraftState(30_000.0, 0.0, 6900.0, -math.pi / 2, 250.0)
    alt, *_ = MissileEnvelope.geometry(lo, same)
    assert alt == pytest.approx(6900.0)


def test_lookup_rmax_is_conservative_both_sides():
    from stealth_tactics.sim.weapons import WeaponModel
    import numpy as np
    wm = WeaponModel(np.random.default_rng(0))
    for coal in ("Blue", "Red"):
        env = wm.envelope_for_coalition(coal)
        lo = AircraftState(0.0, 0.0, 6920.0, math.pi / 2, 0.91 * atmosphere(6920.0)[1])
        hi = AircraftState(30_000.0, 0.0, 12_650.0, math.radians(220.0),
                           0.53 * atmosphere(12_650.0)[1])
        r_new = env.rmax_for_states(lo, hi)
        _, ms, asp, mt, off = MissileEnvelope.geometry(lo, hi)
        r_mean = env.rmax_m(0.5 * (6920.0 + 12_650.0), ms, asp, mt, off)
        assert r_new < 0.75 * r_mean
        # the real look-up fly-out reaches ~23.5 NM here; the fix stays below it
        assert r_new < 23.5 * 1852.0
