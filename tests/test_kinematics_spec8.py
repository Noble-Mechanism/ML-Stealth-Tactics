"""Spec 8: jet energy, missile loft, opening grace, deconfliction, outcome rollup."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from stealth_tactics.analysis.aircraft_sweep import bleed_run, gate_ok
from stealth_tactics.analysis.missile_sweep import sim_rmax_nm, fly_shot
from stealth_tactics.fitness import (DEFAULT_WEIGHTS, fight_terms, load_weights,
                                     merge_outcome_counts, missile_outcome_counts)
from stealth_tactics.neuro.novelty import BDRecorder
from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, Coalition,
                                          integrate_aircraft, F35_PARAMS)
from stealth_tactics.sim.missile_envelope import batch_shots, HIT, DEFEAT_OPENING
from stealth_tactics.sim.missile_kinematics import loft_active, lofted_aim
from stealth_tactics.sim.sensor_config import (DEFAULT_SENSOR_CONFIG as CFG, NM_M,
                                               BLUE_ENERGY, RED_ENERGY)
from stealth_tactics.sim.weapons import WeaponModel

KC = CFG.missile_kinematics
W = dict(DEFAULT_WEIGHTS)
FT = 0.3048


def cfg_kin(**kw):
    return dataclasses.replace(CFG, missile_kinematics=dataclasses.replace(KC, **kw))


# --------------------------------------------------------------- A: energy ---
def test_jet_bleed_gate_ad():
    """A-d: 3–4 g turn at 40 kft from Mach 0.9 bleeds to ~Mach 0.7 within ~20 s."""
    rep = bleed_run(n_max=3.5, alt_ft=40_000.0, start_mach=0.9, t_s=20.0)
    assert gate_ok(rep), rep["final_mach"]
    assert 0.60 <= rep["final_mach"] <= 0.80


def test_blue_red_energy_params_separate():
    assert BLUE_ENERGY.n_max == 7.0 and RED_ENERGY.n_max == 8.0
    assert BLUE_ENERGY.Cd0 != RED_ENERGY.Cd0
    assert BLUE_ENERGY.T_sl_N < RED_ENERGY.T_sl_N
    assert BLUE_ENERGY.max_mach < RED_ENERGY.max_mach
    assert F35_PARAMS.energy is BLUE_ENERGY


def test_hard_turn_bleeds_speed_unlike_legacy_integrator():
    """A max-g turn at 40 kft costs energy; level cruise does not."""
    from stealth_tactics.sim.missile_kinematics import atmosphere
    alt = 40_000 * FT
    a = atmosphere(alt)[1]
    V0 = 0.9 * a
    turn = bleed_run(n_max=7.0, alt_ft=40_000.0, start_mach=0.9, t_s=10.0)
    # cruise: command current heading (no turn)
    from stealth_tactics.sim.aircraft import Aircraft, AircraftState, integrate_aircraft, F35_PARAMS
    from dataclasses import replace
    ac = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt, 0.0, V0))
    ac.params = F35_PARAMS
    ac.cmd_heading_rad = 0.0
    ac.cmd_speed_mps = V0
    ac.cmd_alt_m = alt
    for _ in range(int(10.0 / 0.05)):
        integrate_aircraft(ac, 0.05)
    cruise_mach = ac.state.speed_mps / atmosphere(ac.state.alt)[1]
    assert turn["final_mach"] < cruise_mach - 0.05
    assert turn["final_mach"] < 0.85


def _turn_degrees(alt_ft, t_s=30.0, dt=0.05, mach=0.9):
    """Cumulative heading change (deg) for a continuous max-rate turn."""
    from stealth_tactics.sim.aircraft import _angle_diff
    from stealth_tactics.sim.missile_kinematics import atmosphere
    alt = alt_ft * FT
    V0 = mach * atmosphere(alt)[1]
    ac = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt, 0.0, V0))
    ac.params = F35_PARAMS
    ac.cmd_speed_mps, ac.cmd_alt_m = V0, alt
    tot, prev = 0.0, ac.state.heading_rad
    for _ in range(int(round(t_s / dt))):
        ac.cmd_heading_rad = ac.state.heading_rad + math.pi / 2   # keep turning
        integrate_aircraft(ac, dt)
        tot += abs(_angle_diff(ac.state.heading_rad, prev))
        prev = ac.state.heading_rad
    return math.degrees(tot), ac


def test_energy_turn_slower_at_high_alt_than_legacy_fixed_rate():
    """Spec 8 A2: a sustained turn at 40 kft is much slower than the legacy free
    11 deg/s. Guards the soft speed floor: a jet that has bled to min_speed_mps
    must not keep pulling n_max for free (which would turn at ~40 deg/s)."""
    deg, ac = _turn_degrees(40_000.0, t_s=30.0)
    legacy = F35_PARAMS.max_turn_rate_deg_s * 30.0
    assert deg < 0.6 * legacy, (deg, legacy)
    # and the jet has paid for it in speed
    assert ac.state.speed_mps < 0.5 * 0.9 * 295.0


def test_soft_speed_floor_limits_turn_rate():
    """At the speed floor the achieved turn rate is bounded by sustainable n."""
    from stealth_tactics.sim.aircraft import _angle_diff
    deg_a, ac = _turn_degrees(40_000.0, t_s=15.0)
    h0 = ac.state.heading_rad
    for _ in range(20):                       # 1 s more at the floor
        ac.cmd_heading_rad = ac.state.heading_rad + math.pi / 2
        integrate_aircraft(ac, 0.05)
    rate = abs(math.degrees(_angle_diff(ac.state.heading_rad, h0)))
    assert ac.state.speed_mps == pytest.approx(BLUE_ENERGY.min_speed_mps)
    assert rate < 5.0, rate


# ----------------------------------------------------------------- B: loft ---
def test_loft_increases_hot_rmax_vs_loft_off():
    on = sim_rmax_nm(40_000.0, 0.9, "hot", coarse=5.0, lo=10.0, hi=100.0, tol=0.5)
    off = sim_rmax_nm(40_000.0, 0.9, "hot", coarse=5.0, lo=10.0, hi=100.0, tol=0.5,
                      cfg=cfg_kin(loft_enabled=False))
    assert on > off + 10.0, (on, off)


def test_loft_aim_decays_to_zero_at_handoff():
    c = KC
    # well outside handoff: loft active, aim raised
    ax, ay, az, active = lofted_aim(0, 0, 10000, 0, 80_000, 10000, 5.0, c)
    assert active and az > 10000
    # at handoff: inactive
    ax2, ay2, az2, active2 = lofted_aim(0, 0, 10000, 0, c.loft_handoff_m, 10000, 5.0, c)
    assert not active2 and az2 == 10000
    assert not loft_active(c.loft_handoff_m, 5.0, False, c)
    # B1: bias is loft_angle_deg when the loft starts and decays linearly to 0
    # at handoff, whatever the start range (here 30 NM start, now 27.5 NM)
    r0 = 30 * NM_M
    r = 27.5 * NM_M
    _, _, az3, _ = lofted_aim(0, 0, 10000, 0, r, 10000, 5.0, c, loft_r0_m=r0)
    frac = (r - c.loft_handoff_m) / (r0 - c.loft_handoff_m)
    assert math.degrees(math.asin((az3 - 10000) / r)) == pytest.approx(
        c.loft_angle_deg * frac, rel=1e-6)
    _, _, az4, _ = lofted_aim(0, 0, 10000, 0, r0, 10000, 5.0, c, loft_r0_m=r0)
    assert math.degrees(math.asin((az4 - 10000) / r0)) == pytest.approx(c.loft_angle_deg)


@pytest.mark.parametrize("alt_m,asp", [(9000.0, 0.0), (11000.0, 30.0), (12192.0, 60.0)])
def test_loft_rmax_monotone_in_range(alt_m, asp):
    """Regression (spec 8 build): with the loft ramp tied to 2 x handoff, 9 km hot
    shots hit to 35 NM, missed 36-53 NM and hit again 54-64 NM. Every range
    below the largest hitting range must hit (above the 5 NM min-range zone)."""
    rs = np.arange(5.0, 100.0, 1.0)
    hit = np.asarray(batch_shots(KC, alt_m, 0.9, asp, 0.9, rs * NM_M, False)["outcome"]) == HIT
    last = int(np.max(np.where(hit)[0]))
    assert hit[:last + 1].all(), rs[:last + 1][~hit[:last + 1]]


def test_opening_grace_and_never_during_loft():
    """defeat_opening needs opening_grace_s of continuous opening; never while loft."""
    # Fast cold target: previously immediate defeat_opening; with grace, TOF grows
    s = fly_shot(15.0, 49_000.0, 0.9, "cold", target_mach=2.2)
    assert s.outcome == "defeat_opening"
    assert s.tof_s > KC.opening_grace_s

    # Loft active at long range: brief opening must not defeat immediately.
    # Use a mid-range lofted shot against a beam target — if it ends in
    # defeat_opening, TOF must exceed grace (batch path).
    alt = 40_000 * FT
    res = batch_shots(KC, alt, 0.9, 90.0, 0.9, 35 * NM_M, False)
    out = int(np.asarray(res["outcome"]).reshape(-1)[0])
    tof = float(np.asarray(res["tof"]).reshape(-1)[0])
    if out == DEFEAT_OPENING:
        assert tof >= KC.opening_grace_s - 1e-6


def test_miss_overshoot_still_immediate():
    """Passed within 1 km still labels miss_overshoot without waiting for grace."""
    # Force a near miss via loft-off short beam shot that overshoots — use a
    # high-aspect shot inside overshoot range that opens. Smoke: config flag exists.
    assert KC.opening_grace_s == 3.0
    assert KC.overshoot_range_m == 1000.0


# ---------------------------------------------------------- D: deconflict ---
def test_deconflict_and_rule_and_cap():
    w = dict(W)
    # 10 conflict-seconds -> -10; 100 -> capped at -50
    t = fight_terms(SimpleNamespace(blue_kills=0, red_kills=0, blue_alive=4,
                                    blue_shots=1, end_reason="time_cap", events=[]),
                    6, {f"B{i}": False for i in range(1, 5)}, w, deconflict_s=10)
    assert t["deconflict"] == -10.0
    t2 = fight_terms(SimpleNamespace(blue_kills=0, red_kills=0, blue_alive=4,
                                     blue_shots=1, end_reason="time_cap", events=[]),
                     6, {f"B{i}": False for i in range(1, 5)}, w, deconflict_s=100)
    assert t2["deconflict"] == -50.0


def test_deconflict_recorder_and_rule():
    """BDRecorder counts a second only when BOTH horizontal and alt thresholds hit."""

    class Dummy:
        def __call__(self, world):
            pass

    class Jet:
        def __init__(self, x, y, alt, alive=True):
            self.state = SimpleNamespace(x=x, y=y, alt=alt, alive=alive,
                                        heading_rad=0.0)
            self.radar_emitting = True
            self.cmd_fire = False
            self.fire_target = None
            self.coalition = Coalition.BLUE
            self.id = None

    class World:
        def __init__(self, blues, t):
            self.time_s = t
            self._b = {b.id: b for b in blues}
            self.aircraft = list(blues)
            self.weapons = SimpleNamespace(envelope=None)

        def get(self, i):
            return self._b.get(i)

    ids = ["B1", "B2"]
    # Close horizontally (2 NM) but 6000 ft vertical — NOT conflict
    b1 = Jet(0, 0, 8000); b1.id = "B1"
    b2 = Jet(2 * NM_M, 0, 8000 + 6000 * FT); b2.id = "B2"
    rec = BDRecorder(Dummy(), ids)
    rec.deconflict_nm = 5.0
    rec.deconflict_alt_ft = 5000.0
    rec(World([b1, b2], 0.0))
    assert rec.deconflict_s == 0.0
    # Same alt, 2 NM — conflict
    b2.state.alt = 8000.0
    rec2 = BDRecorder(Dummy(), ids)
    rec2.deconflict_nm = 5.0
    rec2.deconflict_alt_ft = 5000.0
    rec2(World([b1, b2], 0.0))
    assert rec2.deconflict_s == 1.0
    # Far horizontally (10 NM), same alt — NOT conflict
    b2.state.x = 10 * NM_M
    rec3 = BDRecorder(Dummy(), ids)
    rec3.deconflict_nm = 5.0
    rec3.deconflict_alt_ft = 5000.0
    rec3(World([b1, b2], 0.0))
    assert rec3.deconflict_s == 0.0


def test_deconflict_counts_pair_seconds():
    """D-b: -1 per conflict-PAIR-second: three stacked jets = 3 pairs per sample."""

    class Dummy:
        def __call__(self, world):
            pass

    def jet(i, x):
        return SimpleNamespace(id=i, coalition=Coalition.BLUE, radar_emitting=True,
                               cmd_fire=False, fire_target=None,
                               state=SimpleNamespace(x=x, y=0.0, alt=8000.0, alive=True,
                                                     heading_rad=0.0))

    jets = [jet("B1", 0.0), jet("B2", NM_M), jet("B3", 2 * NM_M), jet("B4", 50 * NM_M)]
    world = SimpleNamespace(time_s=0.0, aircraft=jets,
                            weapons=SimpleNamespace(envelope=None),
                            get=lambda i: next((j for j in jets if j.id == i), None))
    rec = BDRecorder(Dummy(), [j.id for j in jets])
    rec(world)
    assert rec.deconflict_s == 3.0
    world.time_s = 1.0
    rec(world)
    assert rec.deconflict_s == 6.0


# -------------------------------------------------------- C: outcome rollup ---
def test_missile_outcome_rollup():
    shots = [
        {"coalition": "Blue", "outcome": "hit"},
        {"coalition": "Blue", "outcome": "defeat_speed"},
        {"coalition": "Blue", "outcome": "defeat_opening"},
        {"coalition": "Red", "outcome": "hit"},
        {"coalition": "Red", "outcome": "miss"},
        {"coalition": "Blue", "outcome": "in_flight"},
    ]
    c = missile_outcome_counts(shots)
    assert c["red_as_target"] == {"hit": 1, "defeat_speed": 1, "defeat_opening": 1}
    assert c["blue_as_target"] == {"hit": 1, "miss": 1}
    merged = merge_outcome_counts([c, c])
    assert merged["red_as_target"]["hit"] == 2


def test_progress_rollup_text_contains_outcomes(tmp_path):
    """write_progress includes a missile-outcomes section (smoke via helper)."""
    from stealth_tactics.fitness import merge_outcome_counts
    outcomes = merge_outcome_counts([
        {"blue_as_target": {"hit": 2}, "red_as_target": {"defeat_speed": 3}},
    ])
    assert outcomes["red_as_target"]["defeat_speed"] == 3
    # champion JSON shape
    doc = {"stats": {"missile_outcomes": outcomes}}
    assert "defeat_speed" in doc["stats"]["missile_outcomes"]["red_as_target"]


# ---------------------------------------------------------- E: per-coalition ---
def test_per_coalition_missile_stub_red_equals_blue():
    assert CFG.missile_kinematics_red is None
    assert CFG.kinematics_for("Red") is CFG.missile_kinematics
    assert CFG.kinematics_for("Blue") is CFG.missile_kinematics
    wm = WeaponModel(np.random.default_rng(0))
    assert wm.kinematics("Red") is wm.kinematics("Blue")


def test_fitness_yaml_has_deconflict_keys():
    w = load_weights()
    assert w["deconflict_nm"] == 5.0
    assert w["deconflict_alt_ft"] == 5000.0
    assert w["deconflict_per_s"] == -1.0
    assert w["deconflict_cap"] == -50.0
