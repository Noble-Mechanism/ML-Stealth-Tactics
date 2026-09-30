"""Spec 8e: rolled pull (lift vector at any bank, positive g only), max dive
angle, and lift-vector rate limits (roll rate, g onset) against jinking.
See docs/specs/08e-rolled-pull.md."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, F35_PARAMS,
                                          RED_FIGHTER_PARAMS, integrate_aircraft)
from stealth_tactics.sim.missile_kinematics import atmosphere
from stealth_tactics.sim.sensor_config import BLUE_ENERGY, RED_ENERGY, spec8d_energy

G0 = 9.80665
FT = 0.3048


def _jet(alt_ft, mach, red=False, model8d=False, gamma_deg=0.0, **over):
    alt = alt_ft * FT
    V = mach * atmosphere(alt)[1]
    mk = Aircraft.make_red if red else Aircraft.make_blue
    ac = mk("J", "J", AircraftState(0.0, 0.0, alt, 0.0, V))
    p = RED_FIGHTER_PARAMS if red else F35_PARAMS
    e = spec8d_energy(p.energy) if model8d else p.energy
    if over:
        e = dataclasses.replace(e, **over)
    ac.params = dataclasses.replace(p, energy=e)
    ac.cmd_heading_rad, ac.cmd_speed_mps, ac.cmd_alt_m = 0.0, V, alt
    ac.gamma_rad = math.radians(gamma_deg)
    return ac


def _n_cap(ac):
    e = ac.params.energy
    rho, _ = atmosphere(ac.state.alt)
    qS = 0.5 * rho * ac.state.speed_mps ** 2 * e.S_m2
    return min(e.n_max, qS * e.CLmax / (e.mass_kg * G0))


def _step(ac, dt):
    """One step; returns (n_v, n_h, n_cap) inferred from the path change."""
    V0, g0, h0, nc = ac.state.speed_mps, ac.gamma_rad, ac.state.heading_rad, _n_cap(ac)
    integrate_aircraft(ac, dt)
    n_v = math.cos(g0) + V0 * (ac.gamma_rad - g0) / (G0 * dt)
    dpsi = (ac.state.heading_rad - h0 + math.pi) % (2 * math.pi) - math.pi
    n_h = V0 * dpsi / dt * max(math.cos(g0), 0.05) / G0
    return n_v, n_h, nc


def test_defaults():
    for e in (BLUE_ENERGY, RED_ENERGY):
        assert e.rolled_pull and e.max_dive_deg == 60.0
        assert e.roll_rate_deg_s == 120.0 and e.g_onset_g_s == 6.0
        assert e.n_pushover_min == 0.0


# ------------------------------------------------------------ A: lift vector --
@pytest.mark.parametrize("alt_ft,mach", [(40_000, 0.9), (25_000, 0.8), (15_000, 0.9)])
def test_pure_bunt_stays_at_or_above_0g(alt_ft, mach):
    """Wings level the jet can only unload to 0 g: n >= 0 always, and a small
    descent (a 0 g push reaches it within rolled_pull_min_push_s) is flown
    wings level with n_v >= 0. With rolled_pull off every descent is a bunt."""
    small = _jet(alt_ft, mach)
    small.cmd_alt_m = small.state.alt - 30.0
    for _ in range(200):
        n_v, _, _ = _step(small, 0.05)
        assert small.load_factor >= 0.0
        assert abs(small.bank_rad) < 1e-12
        assert n_v >= -1e-6
    for dt in (0.05, 0.5):
        bunt = _jet(alt_ft, mach, rolled_pull=False)
        bunt.cmd_alt_m = bunt.state.alt - 5000.0
        for _ in range(int(20 / dt)):
            n_v, _, _ = _step(bunt, dt)
            assert bunt.load_factor >= 0.0 and abs(bunt.bank_rad) < 1e-12
            assert n_v >= -1e-6


def _nose_down(ac, dt=0.05, target_deg=-15.0):
    ac.cmd_alt_m = ac.state.alt - 8000.0
    t, peak, t_inv = 0.0, 0.0, None
    while math.degrees(ac.gamma_rad) > target_deg and t < 60.0:
        g0 = ac.gamma_rad
        integrate_aircraft(ac, dt)
        t += dt
        peak = max(peak, math.degrees(g0 - ac.gamma_rad) / dt)
        if t_inv is None and abs(ac.bank_rad) >= math.pi - 1e-9:
            t_inv = t
    return t, peak, t_inv


@pytest.mark.parametrize("alt_ft,mach", [(40_000, 0.9), (25_000, 0.8), (15_000, 0.9)])
def test_inverted_pull_noses_down_much_faster_than_8d_pushover(alt_ft, mach):
    t8d, r8d, _ = _nose_down(_jet(alt_ft, mach, model8d=True))
    ac = _jet(alt_ft, mach)
    nc = _n_cap(ac)
    V = ac.state.speed_mps
    t8e, r8e, t_inv = _nose_down(ac)
    # 8d: 0 g push, dgamma/dt <= g cos(gamma) / V
    assert r8d <= math.degrees(G0 / V) * 1.02
    # 8e: rolls inverted (180 deg at 120 deg/s = 1.5 s) and pulls
    assert t_inv == pytest.approx(1.5, abs=0.051)
    assert t8e < 0.45 * t8d, (t8e, t8d)
    assert r8e > 3.5 * r8d, (r8e, r8d)
    assert r8e <= math.degrees((nc + 1.0) * G0 / V) * 1.1


def test_total_g_never_exceeds_n_cap_random_commands():
    rng = np.random.default_rng(8)
    for red in (False, True):
        for alt_ft, mach in [(40_000, 0.9), (30_000, 0.7), (15_000, 1.0), (5_000, 0.5)]:
            ac = _jet(alt_ft, mach, red=red)
            ac.cmd_speed_mps = 1e4
            for i in range(400):
                if i % 4 == 0:
                    ac.cmd_heading_rad = float(rng.uniform(-math.pi, math.pi))
                    ac.cmd_alt_m = float(rng.uniform(0.0, 15_000.0))
                n_v, n_h, nc = _step(ac, 0.5)
                assert 0.0 <= ac.load_factor <= nc + 1e-9
                # the realised path never needs more than the lift vector
                assert math.hypot(n_v, n_h) <= nc + 1e-6
                assert ac.state.alt >= 100.0 - 1e-9
                assert ac.gamma_rad >= -math.radians(60.0) - 1e-9


def test_rolled_descending_pull_allocation():
    """Saturated descent + turn demands -> rolled descending pull: the
    desired vector is scaled proportionally, so the bank ends between 90 and
    180 deg and the total load factor sits at n_cap."""
    ac = _jet(25_000, 0.9, roll_rate_deg_s=None, g_onset_g_s=None)
    ac.cmd_alt_m = 100.0
    ac.cmd_heading_rad = math.pi / 2
    integrate_aircraft(ac, 0.5)
    assert 90.0 < math.degrees(ac.bank_rad) < 180.0
    assert ac.load_factor == pytest.approx(_n_cap(_jet(25_000, 0.9)), rel=0.02)


# ----------------------------------------------------- rate limits (jinking) --
@pytest.mark.parametrize("mode", ["vert", "lat", "both"])
@pytest.mark.parametrize("alt_ft,mach", [(40_000, 0.9), (25_000, 0.8), (15_000, 0.9)])
def test_lift_vector_rate_limited_under_alternating_commands(mode, alt_ft, mach):
    """Alternating climb/dive and/or left/right every 1 s (world dt 0.5 s):
    the bank changes <= 120 deg/s, n changes <= 6 g/s (or drops to a falling
    lift limit), and vertical acceleration stays bounded."""
    ac = _jet(alt_ft, mach)
    e = ac.params.energy
    dt = 0.5
    vz = []
    for i in range(120):
        ph = int(i * dt) % 2 == 0
        if mode in ("vert", "both"):
            ac.cmd_alt_m = 15_000.0 if ph else 100.0
        if mode in ("lat", "both"):
            ac.cmd_heading_rad = 1.5 if ph else -1.5
        b0, n0, a0 = ac.bank_rad, ac.load_factor, ac.state.alt
        _, _, nc = _step(ac, dt)
        dphi = abs((ac.bank_rad - b0 + math.pi) % (2 * math.pi) - math.pi)
        assert dphi <= math.radians(e.roll_rate_deg_s) * dt + 1e-9
        dn = ac.load_factor - n0
        assert dn <= e.g_onset_g_s * dt + 1e-9
        assert dn >= -e.g_onset_g_s * dt - 1e-9 or ac.load_factor == pytest.approx(nc)
        vz.append((ac.state.alt - a0) / dt)
    az_g = max(abs(vz[i] - vz[i - 1]) / dt / G0 for i in range(1, len(vz)))
    assert az_g <= e.n_max + 1.0 + 0.5, az_g


@pytest.mark.parametrize("alt_ft,mach", [(40_000, 0.9), (15_000, 0.9)])
def test_vertical_g_reversal_takes_finite_time(alt_ft, mach):
    """From an established inverted pull (n_v < -1), a climb command cannot
    flip the lift vector at once: it must unload and roll ~180 deg."""
    ac = _jet(alt_ft, mach)
    ac.cmd_alt_m = 100.0
    dt = 0.05
    for _ in range(int(3.0 / dt)):
        integrate_aircraft(ac, dt)
    assert ac.load_factor * math.cos(ac.bank_rad) < -1.0
    ac.cmd_alt_m = 15_000.0
    t = 0.0
    while ac.load_factor * math.cos(ac.bank_rad) < 1.0 and t < 10.0:
        integrate_aircraft(ac, dt)
        t += dt
    assert t >= 0.75, t                      # >= 90 deg of roll at 120 deg/s
    # the 8d model (instantaneous) needs a single step for the same reversal
    b = _jet(alt_ft, mach, model8d=True, gamma_deg=-10.0)
    b.cmd_alt_m = 15_000.0
    n_v, _, _ = _step(b, dt)
    assert n_v > 1.0


def test_time_to_roll_inverted():
    ac = _jet(25_000, 0.8)
    ac.cmd_alt_m = 100.0
    t = 0.0
    while abs(ac.bank_rad) < math.pi - 1e-9:
        integrate_aircraft(ac, 0.05)
        t += 0.05
    assert t == pytest.approx(180.0 / 120.0, abs=0.051)


# ------------------------------------------------------------ out vs level --
def _reversal(alt_ft=40_000, mach=0.9, dalt_ft=0.0, dt=0.5, red=False, model8d=False):
    ac = _jet(alt_ft, mach, red=red, model8d=model8d)
    ac.cmd_speed_mps = 1e4
    ac.cmd_alt_m = ac.state.alt + dalt_ft * FT
    ac.cmd_heading_rad = math.pi - 1e-3
    t, turned, h_prev, alt0, xs = 0.0, 0.0, 0.0, ac.state.alt, []
    while t < 120.0 and turned < math.pi - 2e-3:
        integrate_aircraft(ac, dt)
        t += dt
        turned += (ac.state.heading_rad - h_prev + math.pi) % (2 * math.pi) - math.pi
        h_prev = ac.state.heading_rad
        xs.append(ac.state.x)
    return dict(t=t, alt_lost_ft=(alt0 - ac.state.alt) / FT,
                mach=ac.state.speed_mps / atmosphere(ac.state.alt)[1],
                diameter_m=max(xs) - min(min(xs), 0.0))


@pytest.mark.parametrize("red", [False, True])
def test_out_reverses_faster_than_level_turn_at_40kft(red):
    lvl = _reversal(red=red)
    out = _reversal(dalt_ft=-15_000, red=red)
    assert lvl["alt_lost_ft"] == pytest.approx(0.0, abs=1.0)
    assert out["t"] < 0.6 * lvl["t"], (out, lvl)
    assert out["mach"] > lvl["mach"] + 0.1, (out, lvl)      # less speed loss
    assert out["diameter_m"] < 0.5 * lvl["diameter_m"]
    assert 5_000 < out["alt_lost_ft"] < 15_000


# ------------------------------------------------ level turns / climbs vs 8d --
@pytest.mark.parametrize("alt_ft,mach", [(40_000, 0.9), (15_000, 0.9)])
def test_level_turn_unchanged_vs_8d(alt_ft, mach):
    def turn(model8d):
        ac = _jet(alt_ft, mach, model8d=model8d)
        ac.cmd_speed_mps = 1e4
        total, h = 0.0, 0.0
        for _ in range(60):
            ac.cmd_heading_rad = ac.state.heading_rad + math.pi / 2
            integrate_aircraft(ac, 0.5)
            total += (ac.state.heading_rad - h + math.pi) % (2 * math.pi) - math.pi
            h = ac.state.heading_rad
            assert abs(ac.state.alt - alt_ft * FT) < 1e-6
        return math.degrees(total), ac.state.speed_mps
    d8, v8 = turn(True)
    d, v = turn(False)
    assert d == pytest.approx(d8, rel=0.03)
    assert v == pytest.approx(v8, rel=0.01)


@pytest.mark.parametrize("red", [False, True])
def test_climb_unchanged_vs_8d(red):
    def climb(model8d):
        ac = _jet(30_000, 0.9, red=red, model8d=model8d)
        ac.cmd_alt_m = 40_000 * FT
        t, mx = 0.0, 0.0
        while ac.state.alt < 40_000 * FT - 1.0 and t < 300:
            integrate_aircraft(ac, 0.5)
            t += 0.5
            mx = max(mx, ac.state.alt)
        for _ in range(40):
            integrate_aircraft(ac, 0.5)
            mx = max(mx, ac.state.alt)
        return t, mx
    t8, _ = climb(True)
    t, mx = climb(False)
    assert t == pytest.approx(t8, rel=0.03)
    assert mx <= 40_000 * FT + 1e-6          # no overshoot


# ------------------------------------------------------ dive limit / floor --
@pytest.mark.parametrize("red", [False, True])
@pytest.mark.parametrize("a0_ft,tgt_ft,mach", [(40_000, 20_000, 0.9), (40_000, 0, 0.9),
                                              (10_000, 0, 0.9), (5_000, 0, 0.6)])
def test_dive_limit_and_recovery(red, a0_ft, tgt_ft, mach):
    for dt in (0.05, 0.5):
        ac = _jet(a0_ft, mach, red=red)
        ac.cmd_alt_m = tgt_ft * FT
        tgt = max(tgt_ft * FT, 100.0)
        mn, gmin = 1e9, 0.0
        for _ in range(int(150 / dt)):
            g_before = ac.gamma_rad
            integrate_aircraft(ac, dt)
            mn = min(mn, ac.state.alt)
            gmin = min(gmin, ac.gamma_rad)
            if ac.state.alt <= 100.0 + 1e-6:
                # reached the floor flying level (not the hard clamp)
                assert abs(math.degrees(g_before)) < 1.0
        assert math.degrees(gmin) >= -60.0 - 1e-9
        assert math.degrees(gmin) < -50.0                  # a real dive was flown
        assert mn >= tgt - 10.0 * FT                       # no overshoot below target
        assert ac.state.alt == pytest.approx(tgt, abs=30.0 * FT)
        assert mn >= 100.0 - 1e-9


# ------------------------------------------------------------------- ACMI --
def test_acmi_writes_roll_and_pitch(tmp_path):
    """Jets carry attitude in ACMI: T=lon|lat|alt|roll|pitch|yaw."""
    from pathlib import Path
    from stealth_tactics.acmi.exporter import ACMIExporter
    from stealth_tactics.ga.evolution import GAConfig, GeneticAlgorithm
    from stealth_tactics.scenarios.loader import load_scenario
    from stealth_tactics.tactics.genome import TacticsGenome
    root = Path(__file__).resolve().parents[1]
    ga = GeneticAlgorithm(load_scenario(root / "scenarios" / "default_4v3.yaml"),
                          GAConfig(seed=7, sim_max_time_s=60.0, sim_dt=1.0))
    res = ga.evaluate(TacticsGenome(), record=True)
    fr = res.sim.frames[-1]["aircraft"]
    assert all("roll" in st and "pitch" in st for st in fr.values())
    out = tmp_path / "att.txt.acmi"
    ACMIExporter(title="att").export(res.sim.frames, out)
    t_fields = [l.split("T=")[1].split(",")[0].split("|") for l in out.read_text().splitlines()
                if "T=" in l and "Type=Air+FixedWing" in l]
    assert t_fields and all(len(f) == 6 and f[3] != "" and f[4] != "" for f in t_fields)


# ------------------------------------------------ 8d model reproducibility --
def test_spec8d_energy_reproduces_8d_fingerprint(monkeypatch):
    """spec8d_energy() (no rolled pull, 8d descent cap, instantaneous lift
    vector) gives the Spec 8d world byte for byte."""
    import stealth_tactics.sim.aircraft as A
    import test_missile_defense_spec3 as T3
    monkeypatch.setattr(A, "F35_PARAMS", dataclasses.replace(
        A.F35_PARAMS, energy=spec8d_energy(A.F35_PARAMS.energy)))
    monkeypatch.setattr(A, "RED_FIGHTER_PARAMS", dataclasses.replace(
        A.RED_FIGHTER_PARAMS, energy=spec8d_energy(A.RED_FIGHTER_PARAMS.energy)))
    assert T3._fingerprint(0) == T3.PRE_SPEC3_8D_MODEL[0]
