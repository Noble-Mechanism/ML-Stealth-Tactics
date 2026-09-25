"""Spec 3a fix (approved): launch off-nose axis in the Rmax table, coast timeout
and coast Pk time scale 40 s, 180 s flight-time cap."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from stealth_tactics.analysis.missile_sweep import FT, fly_shot
from stealth_tactics.sim.aircraft import Aircraft, AircraftState
from stealth_tactics.sim.missile_envelope import (
    ENGINE_VERSION, HIT, _cache_key, batch_shots, get_envelope, off_nose_signed)
from stealth_tactics.sim.missile_kinematics import atmosphere
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG as CFG, NM_M
from stealth_tactics.sim.weapons import WeaponModel

KC = CFG.missile_kinematics


def _st(x, y, hdg_deg, speed=250.0, alt=12_000.0):
    return AircraftState(x, y, alt, math.radians(hdg_deg), speed)


# ------------------------------------------------------------ config ---------
def test_approved_values():
    assert CFG.missile.coast_timeout_s == 40.0
    assert CFG.missile.coast_time_ref_s == 40.0
    assert CFG.missile.seeker_basket_m == 5000.0          # basket rule kept
    assert KC.max_flight_time_s == 180.0
    offs = KC.env_off_nose_deg
    assert 0.0 in offs and min(offs) <= -60.0 and max(offs) >= 60.0
    assert list(offs) == sorted(offs)
    assert KC.max_off_boresight_deg == 60.0               # launch limit unchanged


def test_cache_key_versioned_and_depends_on_off_nose_bins():
    assert ENGINE_VERSION.startswith("3a.2")
    k0 = _cache_key(KC)
    k1 = _cache_key(dataclasses.replace(KC, env_off_nose_deg=(0.0, 30.0)))
    k2 = _cache_key(dataclasses.replace(KC, max_flight_time_s=120.0))
    assert len({k0, k1, k2}) == 3


# --------------------------------------------------------- geometry ----------
def test_off_nose_sign_convention():
    tgt_east = _st(0.0, 30_000.0, 90.0)              # north of the shooter, moving East
    # shooter nose 40 deg right of the LOS (toward the target's motion): lead (+)
    assert abs(off_nose_signed(_st(0.0, 0.0, 40.0), tgt_east) - 40.0) < 1e-9
    # nose 40 deg left of the LOS: lag (-)
    assert abs(off_nose_signed(_st(0.0, 0.0, -40.0), tgt_east) + 40.0) < 1e-9
    # target moving West flips it
    tgt_west = _st(0.0, 30_000.0, -90.0)
    assert abs(off_nose_signed(_st(0.0, 0.0, 40.0), tgt_west) + 40.0) < 1e-9
    # hot target (no cross-LOS motion): magnitude, + by convention
    tgt_hot = _st(0.0, 30_000.0, 180.0)
    assert abs(off_nose_signed(_st(0.0, 0.0, -25.0), tgt_hot) - 25.0) < 1e-9
    assert off_nose_signed(_st(0.0, 0.0, 0.0), tgt_east) == 0.0
    env = get_envelope(KC)
    g = env.geometry(_st(0.0, 0.0, -40.0), tgt_east)
    assert len(g) == 5 and abs(g[2] - 90.0) < 1e-9 and abs(g[4] + 40.0) < 1e-9


def test_batch_off_nose_starts_along_the_shooter_nose():
    """A lag shot must turn through more angle than a lead shot: at the same
    range, lag 50 deg fails where lead 50 deg and nose-on hit (40k ft beam)."""
    alt = 40_000 * FT
    res = batch_shots(KC, alt, 0.9, 90.0, 0.9, 24 * NM_M, False,
                      off_nose_deg=[0.0, 50.0, -50.0])
    assert list(res["outcome"] == HIT) == [True, True, False]


# ------------------------------------------------------------ table ----------
def test_table_off_nose_values_and_lead_lag_asymmetry():
    env = get_envelope(KC)
    alt = 40_000 * FT
    nm = lambda *a: env.rmax_m(alt, 0.9, *a) / NM_M
    assert abs(nm(0.0, 0.9, 0.0) - 49.4) < 0.5
    assert 41.5 < nm(0.0, 0.9, 50.0) < 44.0           # hot, 50 off: ~42.8
    assert abs(nm(0.0, 0.9, 50.0) - nm(0.0, 0.9, -50.0)) < 0.1
    lead, lag = nm(90.0, 0.9, 50.0), nm(90.0, 0.9, -50.0)
    assert lead > lag + 8.0                             # ~29 vs ~17.5 NM


@pytest.mark.parametrize("beh,asp,off", [("hot", 0.0, 50.0), ("beam", 90.0, 50.0),
                                         ("beam", 90.0, -50.0), ("beam", 45.0, -50.0),
                                         ("turncold", 0.0, 50.0), ("cold", 180.0, 30.0)])
def test_table_matches_sim_path_off_nose(beh, asp, off):
    """Table vs full sim (WeaponModel, 0.5 s world step) for off-nose launches:
    a shot 0.5 NM inside the table value fuzes, 0.5 NM outside does not."""
    env = get_envelope(KC)
    alt = 40_000 * FT
    f = env.rne_m if beh == "turncold" else env.rmax_m
    tab = f(alt, 0.9, asp, 0.9, off) / NM_M
    kw = dict(off_nose_deg=off, aspect_deg=asp)
    assert fly_shot(tab - 0.5, 40_000.0, 0.9, beh, **kw).fuzed, (beh, off, tab)
    assert not fly_shot(tab + 0.5, 40_000.0, 0.9, beh, **kw).fuzed, (beh, off, tab)


def test_launch_gate_uses_off_nose():
    """Same range and target geometry: nose-on / lead 50 deg can shoot, lag 50 deg
    cannot (beam target at 25 NM, 40k ft, Mach 0.9 each)."""
    wm = WeaponModel(np.random.default_rng(0))
    alt = 40_000 * FT
    a = atmosphere(alt)[1]

    def pair(hdg_deg):
        b = Aircraft.make_blue("B1", "F-35-1", _st(0.0, 0.0, hdg_deg, 0.9 * a, alt))
        r = Aircraft.make_red("R1", "Red-1", _st(0.0, 25 * NM_M, 90.0, 0.9 * a, alt))
        return b, r
    assert wm.can_shoot(*pair(0.0))
    assert wm.can_shoot(*pair(50.0))                  # lead
    b, r = pair(-50.0)                                # lag
    assert wm.rmax_m(b, r) < 20 * NM_M and not wm.can_shoot(b, r)


def test_rne_capped_at_rmax():
    env = get_envelope(KC)
    assert np.all(env.rne_table <= env.rmax_table)


# ------------------------------------------------------ flight cap -----------
def test_180s_cap_lets_energy_set_range():
    """15 km, shooter Mach 1.3 head-on at 76 NM: fuzes after ~140 s; with the old
    120 s cap the same shot timed out."""
    s = fly_shot(76.0, 15_000 / FT, 1.3, "hot", target_mach=0.9)
    assert s.fuzed and 125.0 < s.tof_s < 170.0, (s.outcome, s.tof_s)
    old = dataclasses.replace(CFG, missile_kinematics=dataclasses.replace(
        KC, max_flight_time_s=120.0))
    assert fly_shot(76.0, 15_000 / FT, 1.3, "hot", target_mach=0.9, cfg=old).outcome == "timeout"


# ---------------------------------------------------------- coast ------------
def test_coast_between_20_and_40s_now_reaches_active():
    """A missile coasting ~25-35 s on a straight target (old rule: lost at 20 s)
    goes active with f_coast = 1 - 0.15 t/40 (aim error ~ 0)."""
    nd = dataclasses.replace(CFG, missile_kinematics=dataclasses.replace(
        KC, defeat_min_mach=0.0, defeat_on_opening=False))
    b = Aircraft.make_blue("B1", "F-35-1", _st(0.0, 0.0, 0.0, 265.0))
    r = Aircraft.make_red("R1", "Red-1", _st(0.0, 47_000.0, 0.0, 250.0))   # flying away
    wm = WeaponModel(np.random.default_rng(0), nd)
    m = wm.spawn(b, r)
    by = {"B1": b, "R1": r}
    none = {"B1": set(), "R1": set()}
    t = 0.0
    while m.alive and not m.autonomous and t < 60.0:
        r.state.y += 250.0 * 0.5
        wm.step([m], by, 0.5, tracks=none, t=t)
        t += 0.5
    assert m.autonomous, m.outcome
    ts = m.t_since_update
    assert 20.0 < ts < 40.0
    assert m.aim_err_at_auto_m < 1.0
    assert abs(m.pk_factor - (1 - 0.15 * ts / 40.0)) < 1e-6
