"""Spec 3a: missile kinematics, kinematic defeat labels, Pk / support rules,
launch envelope table and the 40,000 ft calibration."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.analysis.missile_sweep import fly_shot, sim_rmax_nm, ENV_ARGS
from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, AircraftTypeParams,
                                          F35_PARAMS, RED_FIGHTER_PARAMS)
from stealth_tactics.sim.missile_envelope import get_envelope, batch_shots, HIT
from stealth_tactics.sim.missile_kinematics import (
    KinState, advance_batch, advance_one, atmosphere, atmosphere_np, derived, G0)
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG as CFG, NM_M
from stealth_tactics.sim.weapons import WeaponModel, OUTCOMES
from stealth_tactics.sim.world import SimConfig, World

KC = CFG.missile_kinematics
FT = 0.3048


def cfg_kin(**kw):
    return dataclasses.replace(CFG, missile_kinematics=dataclasses.replace(KC, **kw))


def jets(range_m=27_000.0, alt=12_000.0, red_hdg=math.pi, red_speed=0.0, blue_speed=265.0):
    b = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt, 0.0, blue_speed))
    r = Aircraft.make_red("R1", "Red-1", AircraftState(0.0, range_m, alt, red_hdg, red_speed))
    return b, r


def fly(w, m, by_id, tracks, n, t0=0.0, move=True):
    """n world steps of 0.5 s; target moves along its heading first (World order)."""
    for i in range(n):
        if not m.alive:
            return
        if move:
            for ac in by_id.values():
                st = ac.state
                st.x += st.speed_mps * math.sin(st.heading_rad) * 0.5
                st.y += st.speed_mps * math.cos(st.heading_rad) * 0.5
        w.step([m], by_id, 0.5, tracks=tracks, t=t0 + i * 0.5)


# ------------------------------------------------------------- physics -------
def test_standard_atmosphere_1976():
    rho0, a0 = atmosphere(0.0)
    assert abs(rho0 - 1.225) < 1e-3 and abs(a0 - 340.29) < 0.05
    rho11, a11 = atmosphere(11_000.0)
    assert abs(rho11 - 0.3639) < 1e-3 and abs(a11 - 295.07) < 0.05
    rho20, _ = atmosphere(20_000.0)
    assert abs(rho20 - 0.08803) < 5e-4
    rho40k, a40k = atmosphere(40_000 * FT)          # 12,192 m
    assert abs(rho40k - 0.302) < 3e-3 and abs(a40k - 295.07) < 0.05
    h = np.array([0.0, 5000.0, 11_000.0, 15_000.0, 25_000.0])
    rv, av = atmosphere_np(h)
    for i, hh in enumerate(h):
        assert abs(rv[i] - atmosphere(hh)[0]) < 1e-12 and abs(av[i] - atmosphere(hh)[1]) < 1e-9


def test_scalar_and_batch_integrators_identical():
    d = derived(KC)
    cases = [(9000.0, 250.0, (20_000.0, 30_000.0, 9500.0), (-200.0, -100.0, 0.0)),
             (12_000.0, 300.0, (-5000.0, 40_000.0, 11_000.0), (250.0, 0.0, 0.0))]
    n = len(cases)
    S = {k: np.zeros(n) for k in ("x", "y", "alt", "vx", "vy", "vz", "mass", "tf")}
    singles = []
    for i, (alt, v, tp, tv) in enumerate(cases):
        S["alt"][i], S["vy"][i], S["mass"][i] = alt, v, KC.launch_mass_kg
        singles.append(KinState(0.0, 0.0, alt, 0.0, v, 0.0, KC.launch_mass_kg, 0.0))
    tps = [np.array(c[2]) for c in cases]
    for step in range(600):                      # 30 s: boost + coast, big turns
        A = np.array([tp + np.array(c[3]) * step * KC.dt_s for tp, c in zip(tps, cases)])
        AV = np.array([c[3] for c in cases], float)
        advance_batch(S, A[:, 0], A[:, 1], A[:, 2], AV[:, 0], AV[:, 1], AV[:, 2], KC.dt_s, d)
        for i, k in enumerate(singles):
            advance_one(k, *A[i], *AV[i], KC.dt_s, d)
    for i, k in enumerate(singles):
        for key, val in (("x", k.x), ("y", k.y), ("alt", k.alt), ("vx", k.vx),
                         ("vy", k.vy), ("vz", k.vz), ("mass", k.mass)):
            assert abs(S[key][i] - val) < 1e-6 * max(1.0, abs(val)), key


def test_boost_adds_about_mach_2_9_and_burns_50_kg():
    d = derived(KC)
    a = atmosphere(40_000 * FT)[1]
    k = KinState(0.0, 0.0, 40_000 * FT, 0.0, 0.9 * a, 0.0, KC.launch_mass_kg, 0.0)
    far = 1e7
    for _ in range(int(round(KC.burn_time_s / KC.dt_s))):
        advance_one(k, 0.0, far, 40_000 * FT, 0.0, 0.0, 0.0, KC.dt_s, d)
    assert abs(k.mass - (KC.launch_mass_kg - KC.propellant_kg)) < 1e-6
    assert 3.7 < k.mach() < 3.9          # spec: peak ~Mach 3.8 from Mach 0.9 at 40k ft
    assert abs(k.alt - 40_000 * FT) < 5.0   # lift carries the weight: holds altitude


def test_turn_limit_from_dynamic_pressure_and_induced_drag():
    d = derived(KC)
    alt = 12_000.0
    rho, a = atmosphere(alt)
    for mach in (1.3, 2.5, 4.0):
        k = KinState(0.0, 0.0, alt, 0.0, mach * a, 0.0, 111.5, 10.0)   # burned out
        # aim point 45 deg off the nose -> saturated PN command
        n = advance_one(k, 3000.0, 3000.0, alt, 0.0, 0.0, 0.0, KC.dt_s, d)
        q_s = 0.5 * rho * (mach * a) ** 2 * d.S
        nmax = min(KC.g_max, q_s * KC.cl_max / (111.5 * G0))
        assert abs(n - nmax) < 1e-6
    # hard turns bleed more speed than straight flight
    k1 = KinState(0.0, 0.0, alt, 0.0, 2.0 * a, 0.0, 111.5, 10.0)
    k2 = k1.copy()
    for _ in range(40):
        advance_one(k1, 0.0, 1e7, alt, 0.0, 0.0, 0.0, KC.dt_s, d)
        advance_one(k2, k2.x + 3000.0, k2.y + 3000.0, alt, 0.0, 0.0, 0.0, KC.dt_s, d)
    assert k2.speed_mps < k1.speed_mps - 20.0


def test_missile_inherits_shooter_velocity():
    wm = WeaponModel(np.random.default_rng(0))
    for v in (200.0, 330.0):
        b, r = jets(blue_speed=v)
        b.state.heading_rad = 0.3
        m = wm.spawn(b, r)
        assert abs(m.pos.speed_mps - v) < 1e-9 and abs(m.pos.heading_rad - 0.3) < 1e-9
        assert (m.pos.x, m.pos.y, m.pos.alt) == (b.state.x, b.state.y, b.state.alt)


# ------------------------------------------------------- defeat labels -------
def test_defeat_speed_on_long_tail_chase():
    s = fly_shot(40.0, 25_000.0, 0.9, "cold")
    assert s.outcome == "defeat_speed" and abs(s.mach_end - KC.defeat_min_mach) < 0.02
    assert s.pk is None and s.f_pole_nm is None


def test_defeat_opening_when_target_outruns_the_missile():
    # Mach 2.2 target running away at 49,000 ft: missile still > Mach 1.2 when it
    # stops closing
    s = fly_shot(15.0, 49_000.0, 0.9, "cold", target_mach=2.2)
    assert s.outcome == "defeat_opening"
    assert s.mach_end > KC.defeat_min_mach


def test_timeout_safety_cap_label():
    cfg = cfg_kin(max_flight_time_s=10.0)
    s = fly_shot(40.0, 40_000.0, 0.9, "hot", cfg=cfg)
    assert s.outcome == "timeout" and abs(s.tof_s - 10.0) < 0.06


def test_head_on_hit_and_outcome_labels_distinct():
    s = fly_shot(40.0, 40_000.0, 0.9, "hot")
    assert s.outcome in ("hit", "miss") and s.fuzed
    assert 70.0 < s.tof_s < 85.0 and 1.3 < s.mach_end < 1.7
    assert s.a_pole_nm is not None and s.f_pole_nm is not None and s.f_pole_nm < s.a_pole_nm
    assert len(set(OUTCOMES)) == len(OUTCOMES)
    from stealth_tactics.acmi.exporter import ALWAYS_BOOKMARK
    for lab in ("defeat_speed", "defeat_opening", "miss_overshoot", "timeout",
                "lost_coast_timeout", "lost_basket"):
        assert lab in ALWAYS_BOOKMARK


# ---------------------------------------------------------- Pk factors -------
def test_endgame_energy_factor():
    f = KC.endgame_factor
    assert f(2.0) == 1.0 and f(3.5) == 1.0
    assert abs(f(1.2) - 0.6) < 1e-12 and abs(f(1.6) - 0.8) < 1e-12
    assert abs(f(1.0) - 0.6) < 1e-12


def test_coast_pk_formula_values():
    c = CFG.missile.coast_pk_factor
    assert abs(c(500.0, 0.0) - 0.94) < 0.005
    assert abs(c(1000.0, 0.0) - 0.78) < 0.005
    assert abs(c(2000.0, 0.0) - 0.37) < 0.005
    assert abs(c(3000.0, 0.0) - 0.105) < 0.005
    # 3a fix: time term uses the 40 s scale, (1 - 0.15 min(t/40, 1))
    assert CFG.missile.coast_time_ref_s == 40.0
    assert abs(c(0.0, 10.0) - (1 - 0.15 * 10 / 40)) < 1e-12
    assert abs(c(0.0, 20.0) - 0.925) < 1e-12
    assert abs(c(0.0, 40.0) - 0.85) < 1e-12 and abs(c(0.0, 60.0) - 0.85) < 1e-12
    # aim-point error sets most of the penalty: 1 km at 20 s -> 0.78 x 0.925
    assert abs(c(1000.0, 20.0) - math.exp(-0.25) * 0.925) < 1e-12


SUP = {"B1": {"R1"}, "B2": set(), "R1": set()}
NONE = {"B1": set(), "B2": set(), "R1": set()}


def _active_at_launch(wing=False, red_shooter=False):
    """Missile launched inside 15 NM (active at launch) at a far-ish stationary
    target at 12 km altitude: ~25 s of flight."""
    b, r = jets(range_m=27_000.0)
    by = {"B1": b, "R1": r}
    if wing:
        by["B2"] = Aircraft.make_blue("B2", "F-35-2", AircraftState(3000.0, 0.0, 12_000.0,
                                                                     0.0, 265.0))
    wm = WeaponModel(np.random.default_rng(0))
    m = wm.spawn(b, r)
    assert m.autonomous and m.sup_at_active
    return wm, m, by


def test_full_support_to_impact_has_no_support_factor():
    wm, m, by = _active_at_launch()
    fly(wm, m, by, SUP, 200, move=False)
    assert m.outcome in ("hit", "miss")
    assert m.f_support == 1.0 and m.pk_factor == 1.0
    assert abs(m.pk_final - KC.base_pk * KC.endgame_factor(m.mach_end)) < 1e-12


def test_support_dropped_after_active_x090_once_and_short_gaps_continuous():
    wm, m, by = _active_at_launch()
    fly(wm, m, by, SUP, 10, move=False)              # tf 0..4.5 supported
    fly(wm, m, by, NONE, 3, t0=5.0, move=False)       # 1.5 s gap
    fly(wm, m, by, SUP, 1, t0=6.5, move=False)
    assert not m.support_dropped and m.pk == KC.base_pk
    fly(wm, m, by, NONE, 4, t0=7.0, move=False)       # 2.0 s gap -> drop
    assert m.support_dropped and abs(m.pk - 0.9 * KC.base_pk) < 1e-12
    fly(wm, m, by, SUP, 2, t0=9.0, move=False)        # regained, then dropped again
    fly(wm, m, by, NONE, 200, t0=10.0, move=False)
    assert m.outcome in ("hit", "miss")
    assert sum(e["type"] == "support_dropped" for e in wm.events) == 1
    assert abs(m.pk_final - KC.base_pk * 0.9 * KC.endgame_factor(m.mach_end)) < 1e-12


def test_gap_under_2s_before_impact_is_not_penalised():
    wm, m, by = _active_at_launch()
    while m.alive:
        fly(wm, m, by, SUP, 1, move=False)
        # look ahead: drop support for the last 1.5 s before the fuze
        probe = m.pos.copy()
        tgt = by["R1"].state
        if math.dist((probe.x, probe.y, probe.alt), (tgt.x, tgt.y, tgt.alt)) \
                < 1.4 * probe.speed_mps:
            break
    fly(wm, m, by, NONE, 10, t0=50.0, move=False)
    assert m.outcome in ("hit", "miss") and not m.support_dropped


def test_any_flightmate_keeps_support_after_active_blue_only():
    wm, m, by = _active_at_launch(wing=True)
    fly(wm, m, by, SUP, 4, move=False)
    wing_only = {"B1": set(), "B2": {"R1"}, "R1": set()}
    fly(wm, m, by, wing_only, 200, t0=2.0, move=False)
    assert m.outcome in ("hit", "miss") and not m.support_dropped and m.f_support == 1.0


def test_coasting_at_handoff_straight_vs_maneuvering_target():
    def run(jink_m):
        b, r = jets(range_m=45_000.0, alt=9000.0, red_speed=250.0)   # head-on
        wm = WeaponModel(np.random.default_rng(0), cfg_kin(defeat_min_mach=0.0))
        m = wm.spawn(b, r)
        by = {"B1": b, "R1": r}
        t = 0.0
        while m.alive and not m.autonomous:
            if jink_m and abs(t - 4.0) < 1e-9:
                r.state.x += jink_m                  # target maneuvered while coasting
            fly(wm, m, by, NONE, 1, t0=t)
            t += 0.5
        assert m.autonomous and m.coasting is False
        return m
    straight, jinked = run(0.0), run(1500.0)
    ts = straight.t_since_update
    assert 5.0 < ts < 40.0
    assert straight.aim_err_at_auto_m < 1.0
    assert abs(straight.pk_factor - (1 - 0.15 * min(ts / 40.0, 1))) < 1e-4
    e = jinked.aim_err_at_auto_m
    assert 1400.0 < e < 1600.0
    assert abs(jinked.pk_factor - CFG.missile.coast_pk_factor(e, jinked.t_since_update)) < 1e-12
    assert jinked.pk_factor < 0.62 * straight.pk_factor
    assert abs(jinked.pk - KC.base_pk * jinked.pk_factor) < 1e-12


def test_loss_rules_kept_coast_timeout_and_basket():
    # coast timeout: > 40 s continuously unsupported before going active (3a fix)
    b, r = jets(range_m=300_000.0, alt=12_000.0)
    wm = WeaponModel(np.random.default_rng(0), cfg_kin(defeat_min_mach=0.0))
    m = wm.spawn(b, r)
    fly(wm, m, {"B1": b, "R1": r}, NONE, 120)
    assert m.outcome == "lost_coast_timeout" and 40.0 < m.t_since_update <= 40.5 + 1e-9
    # basket: aim point > 5 km from the true target at the active point
    b, r = jets(range_m=40_000.0, alt=12_000.0)
    wm = WeaponModel(np.random.default_rng(0))
    m = wm.spawn(b, r)
    by = {"B1": b, "R1": r}
    fly(wm, m, by, NONE, 2)
    r.state.x = 6_000.0
    fly(wm, m, by, NONE, 60, t0=1.0)
    assert m.outcome == "lost_basket" and m.aim_err_at_auto_m > 5000.0
    assert any(e["type"] == "lost_basket" for e in wm.events)


def test_active_range_is_a_config_knob():
    for nm in (15.0, 10.0):
        b, r = jets(range_m=40_000.0, alt=12_000.0)
        wm = WeaponModel(np.random.default_rng(0), cfg_kin(active_range_m=nm * NM_M))
        m = wm.spawn(b, r)
        by = {"B1": b, "R1": r}
        while m.alive and not m.autonomous:
            fly(wm, m, by, SUP, 1, move=False)
        ev = [e for e in wm.events if e["type"] == "autonomous"][0]
        assert abs(ev["range_m"] - nm * NM_M) < 60.0


# ------------------------------------------------ shared missile / gating ----
def test_same_missile_both_sides_no_per_type_range_or_pk():
    names = {f.name for f in dataclasses.fields(AircraftTypeParams)}
    assert "missile_range_m" not in names and "missile_pk" not in names
    wm = WeaponModel(np.random.default_rng(0))
    b, r = jets(range_m=20_000.0)
    mb, mr = wm.spawn(b, r), wm.spawn(r, b)
    assert mb.pk == mr.pk == KC.base_pk == 0.60


def test_launch_gate_uses_rmax_table():
    wm = WeaponModel(np.random.default_rng(0))
    alt = 40_000 * FT
    a = atmosphere(alt)[1]
    b, r = jets(range_m=45 * NM_M, alt=alt, red_speed=0.9 * a, blue_speed=0.9 * a)
    rmax, rne = wm.envelope_for(b, r)
    assert 47 * NM_M < rmax < 52 * NM_M and 22 * NM_M < rne < 26 * NM_M
    assert wm.can_shoot(b, r)                    # 45 NM head-on: inside Rmax
    r.state.heading_rad = 0.0                    # now cold: Rmax ~21 NM
    assert wm.rmax_m(b, r) < 23 * NM_M and not wm.can_shoot(b, r)
    r.state.y = 20 * NM_M
    assert wm.can_shoot(b, r)


def test_envelope_table_monotone_and_rne_below_rmax():
    env = get_envelope(KC)
    t, n = env.rmax_table, env.rne_table
    assert t.shape == (len(KC.env_alt_m), len(KC.env_shooter_mach), len(KC.env_aspect_deg),
                       len(KC.env_target_mach), len(KC.env_off_nose_deg))
    i0 = KC.env_off_nose_deg.index(0.0)
    t0 = t[..., i0]                                                # nose-on slice
    hi_alt = [i for i, v in enumerate(KC.env_alt_m) if v <= 13_000.0]
    assert np.all(np.diff(t0[hi_alt][:, :, 0, :], axis=0) > 0)    # altitude (hot)
    assert np.all(np.diff(t0[:, :, 0, :], axis=1) > 0)            # shooter Mach (hot)
    assert np.all(np.diff(t0[:, :, :, 1], axis=2) < 0)            # aspect hot -> cold
    assert np.all(n <= t + 0.05 * NM_M)
    # hot / cold targets: lead and lag are mirror images (no cross-LOS motion)
    for asp in (0.0, 180.0):
        ia = KC.env_aspect_deg.index(asp)
        for k, off in enumerate(KC.env_off_nose_deg):
            if -off in KC.env_off_nose_deg:
                km = KC.env_off_nose_deg.index(-off)
                assert np.allclose(t[:, :, ia, :, k], t[:, :, ia, :, km], atol=0.06 * NM_M)
    # Rmax falls off the nose (hot target): 0 >= 30 >= 50 deg
    i30, i50 = KC.env_off_nose_deg.index(30.0), KC.env_off_nose_deg.index(50.0)
    assert np.all(t[:, :, 0, :, i0] >= t[:, :, 0, :, i30] - 0.06 * NM_M)
    assert np.all(t[:, :, 0, :, i30] >= t[:, :, 0, :, i50] - 0.06 * NM_M)


# --------------------------------------------------------- calibration -------
@pytest.mark.parametrize("behavior,goal", [("hot", 50.0), ("turncold", 24.0), ("cold", 20.0),
                                           ("beam", 28.8)])
def test_calibration_40k_mach09_sim_path(behavior, goal):
    """Approved calibration: 40,000 ft, Mach 0.9 vs level Mach 0.9 target:
    head-on ~50, turn-cold-at-launch ~24, already-cold ~20 NM (+-2 NM);
    beam checked against the prototype's 28.8 NM."""
    r = sim_rmax_nm(40_000.0, 0.9, behavior, coarse=5.0, lo=5.0, hi=60.0, tol=0.1)
    assert abs(r - goal) <= 2.0, r


def test_table_matches_sim_path_at_calibration_point():
    env = get_envelope(KC)
    alt = 40_000 * FT
    for beh in ("hot", "beam", "turncold", "cold"):
        asp, tc = ENV_ARGS[beh]
        tab = (env.rne_m if tc else env.rmax_m)(alt, 0.9, asp, 0.9) / NM_M
        # a shot 0.3 NM inside the table value fuzes, 0.5 NM outside does not
        assert fly_shot(tab - 0.3, 40_000.0, 0.9, beh).fuzed, (beh, tab)
        assert not fly_shot(tab + 0.5, 40_000.0, 0.9, beh).fuzed, (beh, tab)


def test_batch_engine_agrees_with_sim_path():
    alt = 40_000 * FT
    res = batch_shots(KC, alt, 0.9, [0.0, 0.0, 90.0], 0.9,
                      [40 * NM_M, 55 * NM_M, 25 * NM_M], [False, False, False])
    assert list(res["outcome"] == HIT) == [True, False, True]
    s = fly_shot(40.0, 40_000.0, 0.9, "hot")
    assert abs(res["tof"][0] - s.tof_s) < 0.6 and abs(res["mach_end"][0] - s.mach_end) < 0.02


# ---------------------------------------------------------------- ACMI -------
def test_acmi_missile_mach_and_defeat_bookmark(tmp_path):
    from stealth_tactics.analysis.missile_replays import run_missile_replay
    rep = run_missile_replay("drag", seed=2)
    assert rep.outcome == "defeat_speed"
    out = ACMIExporter().export(rep.world.frames, tmp_path / "d.txt.acmi")
    text = out.read_text()
    assert "Name=Missile,Type=Weapon+Missile" in text and "Mach=" in text and "TAS=" in text
    assert "Name=F-35A" in text
    assert any(l.startswith("0,Event=Bookmark|") and "kinematic defeat" in l
               for l in text.splitlines())
    mid = [l for l in text.splitlines() if "Name=Missile" in l][0].split(",")[0]
    assert f"-{mid}" in text.splitlines()          # object removed at end of flight


def test_climbing_turning_target_uses_true_motion_between_steps():
    """Target motion inside a world step is the chord between its recorded start
    position and its integrated end position (climbs / turns included). Using the
    level heading-only velocity instead makes PN chase a stair-stepping target
    and bleed energy (Mach ~1.76 at the fuze for this case vs ~2.35)."""
    b, r = jets(range_m=30_000.0, alt=9000.0, red_speed=250.0, red_hdg=math.pi)
    wm = WeaponModel(np.random.default_rng(0))
    m = wm.spawn(b, r)
    by = {"B1": b, "R1": r}
    t = 0.0
    while m.alive and t < 120.0:
        r.state.heading_rad += math.radians(2.0) * 0.5
        r.state.alt += 60.0 * 0.5
        fly(wm, m, by, SUP, 1, t0=t)
        t += 0.5
    assert m.outcome in ("hit", "miss"), m.outcome
    assert m.mach_end > 2.2
