"""Spec 4: Red presentations (sampling, determinism, formations, maneuvers,
early end, Winchester departure, seeding). docs/specs/04-red-presentations.md"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.ga.evolution import GAConfig, GeneticAlgorithm
from stealth_tactics.presentation_runner import export_presentation_acmi, run_presentation
from stealth_tactics.scenarios.loader import build_aircraft, load_scenario
from stealth_tactics.scenarios.presentation import (
    BAND_RANGES, BANDS, DEFAULT_PRESENTATION_CONFIG, DOCTRINES, NM_M, Presentation,
    PresentationConfig, build_benchmark_set, build_eval_set, derive_sim_seed,
    formations_for, halves_of, load_menus, param_corners, realize_formation,
    sample_presentation, strata)
from stealth_tactics.sim.aircraft import AircraftState, Aircraft, Coalition, distance_3d
from stealth_tactics.sim.world import (END_BLUE_DEAD, END_BOTH_WINCHESTER, END_RED_DEAD,
                                       END_RED_DEPARTED, END_TIME_CAP, SimConfig, World)
from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.tactics.red_defense import COLD

ROOT = Path(__file__).resolve().parents[1]
MENUS = load_menus()
CFG = DEFAULT_PRESENTATION_CONFIG
BOX_NM = 25.0


# ------------------------------------------------------------ sampling -----
def test_sampling_ranges_and_frequencies():
    n = 1200
    ps = [sample_presentation(s) for s in range(n)]
    six = set(formations_for(6, MENUS))
    for p in ps:
        assert p.n_red == 6 and len(p.red_jets) == 6
        assert p.formation in six
        assert 40.0 <= p.range_nm <= 60.0 and -40.0 <= p.azimuth_deg <= 40.0
        assert 6000.0 <= p.base_alt_m <= 12000.0
        lo, hi = BAND_RANGES[p.band]
        assert lo <= p.aggressiveness <= hi
        assert p.aggressiveness < hi or p.band == "aggressive"
        assert p.doctrine in DOCTRINES
        m = p.maneuver
        t_lo, t_hi = MENUS["maneuvers"][m["type"]]["trigger_nm"]
        assert t_lo - 1e-9 <= m["trigger_range_nm"] <= min(t_hi, p.range_nm - 5.0) + 1e-9
        for k, spec in MENUS["maneuvers"][m["type"]]["params"].items():
            v = m["params"][k]
            if isinstance(spec, dict):
                assert v in spec["choice"]
            else:
                assert spec[0] <= v <= spec[1]
        ac = run_free_positions(p)
        blues = [a for a in ac if a.coalition == Coalition.BLUE]
        for r in (a for a in ac if a.coalition == Coalition.RED):
            assert 1000.0 <= r.state.alt <= 13500.0
            assert r.state.speed_mps == 250.0
            assert min(distance_3d(r.state, b.state) for b in blues) >= 35.0 * NM_M
            assert r.firing_doctrine == p.doctrine
        lead = ac[len(blues)]
        assert distance_3d(blues[0].state, lead.state) == pytest.approx(
            math.hypot(p.range_nm * NM_M, lead.state.alt - blues[0].state.alt), rel=1e-6)
    for key, vals, tol in (("maneuver", sorted(MENUS["maneuvers"]), 0.06),
                           ("band", BANDS, 0.06), ("doctrine", DOCTRINES, 0.06),
                           ("formation", sorted(six), 0.05)):
        c = Counter((p.maneuver["type"] if key == "maneuver" else getattr(p, key)) for p in ps)
        for v in vals:
            assert abs(c[v] / n - 1.0 / len(vals)) < tol, (key, v, c)


def run_free_positions(p):
    from stealth_tactics.scenarios.presentation import build_presentation_aircraft
    return build_presentation_aircraft(p)


def test_same_seed_identical_presentation_and_json_round_trip():
    for s in (0, 1, 12345, 2 ** 63 + 7):
        a, b = sample_presentation(s), sample_presentation(s)
        assert a == b and a.to_json() == b.to_json()
        c = Presentation.from_json(a.to_json())
        assert c == a and c.to_json() == a.to_json()
        assert a.sim_seed == derive_sim_seed(s)
    assert sample_presentation(1).to_json() != sample_presentation(2).to_json()


def test_stratum_only_changes_its_fields():
    base = sample_presentation(99)
    for st in strata():
        p = sample_presentation(99, stratum=st)
        assert (p.maneuver["type"], p.band, p.doctrine) == st
        assert p.formation == base.formation and p.formation_params == base.formation_params
        assert p.range_nm == base.range_nm and p.red_jets == base.red_jets
        assert p.sim_seed == base.sim_seed


# ---------------------------------------------------------- formations -----
def _extent(slots):
    r = [s["right_nm"] for s in slots]
    b = [s["back_nm"] for s in slots]
    return max(r) - min(r), max(b) - min(b), min(b)


@pytest.mark.parametrize("name", sorted(MENUS["formations"]))
def test_formation_box_bound_all_corners(name):
    f = MENUS["formations"][name]
    corners = param_corners(f["params"])
    assert corners
    for params in corners:
        for mirror in ((1, -1) if f["mirror"] else (1,)):
            slots = realize_formation(f, params, mirror)
            w, d, bmin = _extent(slots)
            assert w <= BOX_NM + 1e-9 and d <= BOX_NM + 1e-9, (name, params, w, d)
            assert bmin >= -1e-9                          # nobody ahead of the lead
            assert slots[0]["right_nm"] == 0 and slots[0]["back_nm"] == 0
            h = halves_of(f, slots)
            assert sorted(h[0] + h[1]) == list(range(1, len(slots) + 1))
            assert h[0] and h[1]
            # groups within ~3 NM
            for g in {s["group"] for s in slots}:
                gs = [s for s in slots if s["group"] == g]
                for a in gs:
                    for b in gs:
                        assert math.hypot(a["right_nm"] - b["right_nm"],
                                          a["back_nm"] - b["back_nm"]) <= 3.0


def test_formation_box_bound_drawn_spacings():
    for s in range(600):
        p = sample_presentation(s)
        w, d, _ = _extent(p.red_jets)
        assert w <= BOX_NM and d <= BOX_NM


def test_six_ship_menu_and_eight_ship_option():
    assert formations_for(6, MENUS) == sorted(
        ["wall", "box", "ladder", "echelon", "vic", "champagne"])
    cfg8 = PresentationConfig(n_red=8)
    seen = set()
    for s in range(60):
        p = sample_presentation(s, cfg8)
        assert len(p.red_jets) == 8 and p.formation.endswith("_8")
        seen.add(p.formation)
    assert seen == {"wall_8", "box_8", "ladder_8"}
    r = run_presentation(sample_presentation(3, cfg8),
                         cfg=dataclasses.replace(cfg8, max_time_s=30.0))
    assert r.time_s == pytest.approx(30.0) and len(r.presentation["red_jets"]) == 8


# ------------------------------------------------------------ eval sets -----
def test_eval_set_stratified_and_deterministic():
    a = build_eval_set(42, 0, 24)
    assert [p.to_json() for p in a] == [p.to_json() for p in build_eval_set(42, 0, 24)]
    cells = Counter((p.maneuver["type"], p.band, p.doctrine) for p in a)
    assert len(cells) == 24 and set(cells.values()) == {1}
    b = build_eval_set(42, 1, 24)
    assert {p.presentation_seed for p in a}.isdisjoint({p.presentation_seed for p in b})
    bench = build_benchmark_set(42, 64)
    assert len(bench) == 64 and bench == build_benchmark_set(42, 64)
    c = Counter((p.maneuver["type"], p.band, p.doctrine) for p in bench)
    assert min(c.values()) >= 2 and max(c.values()) <= 3


# -------------------------------------------------------- determinism -------
def test_same_presentation_identical_fight_and_byte_identical_acmi(tmp_path):
    p = sample_presentation(4)
    r1 = run_presentation(p, record=True)
    r2 = run_presentation(json.loads(p.to_json()), record=True)   # stored JSON
    assert r1.time_s == r2.time_s and r1.shots == r2.shots
    assert [(e["t"], e["type"]) for e in r1.events] == [(e["t"], e["type"]) for e in r2.events]
    export_presentation_acmi(r1, tmp_path / "a.txt.acmi")
    export_presentation_acmi(r2, tmp_path / "b.txt.acmi")
    a = (tmp_path / "a.txt.acmi").read_bytes()
    assert a == (tmp_path / "b.txt.acmi").read_bytes()
    txt = a.decode()
    assert "Name=F-35A" in txt and "0,Comments=seed 4" in txt
    assert ACMIExporter.validate_header(tmp_path / "a.txt.acmi")


def test_noise_seed_from_presentation_not_genome_index():
    ga = GeneticAlgorithm(load_scenario(ROOT / "scenarios" / "default_4v3.yaml"),
                          GAConfig(seed=5, sim_max_time_s=120.0, presentations_per_gen=2))
    pres = [p.to_dict() for p in build_eval_set(5, 0, 2)]
    g = TacticsGenome()
    other = TacticsGenome.random(np.random.default_rng(3))
    alone = ga.evaluate_presentations([g], pres)[0]
    third = ga.evaluate_presentations([other, other, g], pres)[2]
    assert alone.per_fight == third.per_fight and alone.fitness == third.fitness


def test_ga_presentation_mode_champion_reproduces():
    ga = GeneticAlgorithm(load_scenario(ROOT / "scenarios" / "default_4v3.yaml"),
                          GAConfig(population=3, generations=2, seed=11, elite_count=1,
                                   sim_max_time_s=100.0, presentations_per_gen=2,
                                   benchmark_size=2))
    best = ga.run()                      # raises if the stored-presentation re-run differs
    assert len(best.presentations) == 2 and best.benchmark_fitness is not None
    again = ga.evaluate_presentations([best.genome], best.presentations)[0]
    assert again.fitness == best.fitness
    assert best.sim.frames and best.sim.presentation == best.presentations[best.acmi_index]


# ------------------------------------------------------------ early end -----
def _jet(uid, coal, x, y, heading, alt=9000.0, ammo=4):
    st = AircraftState(x=x, y=y, alt=alt, heading_rad=heading, speed_mps=250.0)
    a = (Aircraft.make_blue if coal == Coalition.BLUE else Aircraft.make_red)(uid, uid, st)
    a.ammo = ammo
    return a


def _straight(w):
    for a in w.aircraft:
        a.cmd_heading_rad, a.cmd_speed_mps, a.cmd_alt_m = (a.state.heading_rad, 250.0,
                                                           a.state.alt)


def _spec4_cfg(**kw):
    base = dict(dt=0.5, max_time_s=120.0, seed=3, early_end_winchester=True,
                finish_missiles_after_wipeout=True)
    base.update(kw)
    return SimConfig(**base)


def test_legacy_defaults_unchanged():
    c = SimConfig()
    assert not c.early_end_winchester and not c.finish_missiles_after_wipeout
    assert c.early_end_on_departure


def test_both_winchester_nothing_in_air_ends():
    ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0, ammo=0),
          _jet("R1", Coalition.RED, 0, 90_000, math.pi, ammo=0)]
    r = World(ac, _spec4_cfg(), _straight, lambda w: None, record=False).run()
    assert r.end_reason == END_BOTH_WINCHESTER and r.ended_early and r.time_s == 0.0
    ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0, ammo=0),
          _jet("R1", Coalition.RED, 0, 90_000, math.pi, ammo=0)]
    r = World(ac, SimConfig(max_time_s=20.0, seed=3), _straight, lambda w: None,
              record=False).run()                           # legacy: runs to the cap
    assert r.end_reason == END_TIME_CAP and not r.ended_early


def _shooter(target="R1", die_after_s=None):
    state = {"launch_t": None}

    def ctrl(w):
        _straight(w)
        b = w.get("B1")
        if b.state.alive and b.ammo > 0:
            b.cmd_fire, b.fire_target = True, target
        if w.missiles and state["launch_t"] is None:
            state["launch_t"] = w.time_s
        if (die_after_s is not None and state["launch_t"] is not None
                and w.time_s >= state["launch_t"] + die_after_s):
            b.state.alive = False                         # Blue shot down
    return ctrl, state


def test_both_winchester_waits_for_missile_in_air():
    ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0, ammo=1),
          _jet("R1", Coalition.RED, 0, 30_000, math.pi, ammo=0),
          _jet("R2", Coalition.RED, 30_000, 60_000, math.pi, ammo=0)]
    ctrl, st = _shooter()
    r = World(ac, _spec4_cfg(), ctrl, _straight, record=False).run()
    assert st["launch_t"] is not None
    assert r.end_reason == END_BOTH_WINCHESTER and r.ended_early
    assert r.time_s > st["launch_t"] + 5.0
    assert r.shots[0]["outcome"] != "in_flight"


def test_blue_alone_winchester_fight_continues():
    ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0, ammo=0),
          _jet("R1", Coalition.RED, 0, 150_000, 0.0, ammo=4)]      # Red flies away
    r = World(ac, _spec4_cfg(max_time_s=30.0), _straight, _straight, record=False).run()
    assert r.end_reason == END_TIME_CAP and not r.ended_early and r.time_s == pytest.approx(30.0)


def test_wipeout_resolves_missiles_in_flight_trade_kill():
    """K2: the last Blue dies with its missile in the air; the fight goes on
    until the missile resolves and a hit counts as a kill."""
    hit_found = False
    for seed in range(12):
        ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0, ammo=1),
              _jet("R1", Coalition.RED, 0, 20_000, math.pi, ammo=0),
              _jet("R2", Coalition.RED, 40_000, 20_000, math.pi, ammo=0)]
        ctrl, st = _shooter(die_after_s=2.0)
        r = World(ac, _spec4_cfg(seed=seed), ctrl, _straight, record=False).run()
        assert r.end_reason == END_BLUE_DEAD and r.ended_early
        assert r.shots[0]["outcome"] != "in_flight"
        assert r.time_s > st["launch_t"] + 2.5
        if r.shots[0]["outcome"] == "hit":
            assert r.blue_kills == 1 and r.red_kills == 1
            hit_found = True
            # legacy: ends at once, the missile is cut off and the kill lost
            ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0, ammo=1),
                  _jet("R1", Coalition.RED, 0, 20_000, math.pi, ammo=0),
                  _jet("R2", Coalition.RED, 40_000, 20_000, math.pi, ammo=0)]
            ctrl, st = _shooter(die_after_s=2.0)
            r0 = World(ac, SimConfig(dt=0.5, max_time_s=120.0, seed=seed), ctrl, _straight,
                       record=False).run()
            assert r0.shots[0]["outcome"] == "in_flight" and r0.blue_kills == 0
            assert r0.end_reason == END_BLUE_DEAD and not r0.ended_early
            break
    assert hit_found


def test_red_wiped_out_nothing_in_air_ends_at_once():
    ac = [_jet("B1", Coalition.BLUE, 0, 0, 0.0), _jet("R1", Coalition.RED, 0, 90_000, math.pi)]
    ac[1].state.alive = False
    r = World(ac, _spec4_cfg(), _straight, _straight, record=False).run()
    assert r.end_reason == END_RED_DEAD and r.ended_early and r.time_s == 0.0


def test_red_departed_end_reason():
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ac = build_aircraft(sc)
    for a in ac:
        if a.coalition == Coalition.RED:
            a.departed = True
    r = World(ac, SimConfig(max_time_s=10.0, seed=1), lambda w: None, lambda w: None,
              record=False).run()
    assert r.end_reason == END_RED_DEPARTED and r.ended_early


# ------------------------------------------------------ Red behaviour ------
def test_red_winchester_departs_not_a_turn_away():
    p = sample_presentation(7)
    seen = {}

    def prep(aircraft, defense):
        for a in aircraft:
            if a.coalition == Coalition.RED:
                a.ammo = 0
        seen["defense"] = defense
    r = run_presentation(p, prepare=prep)
    ev = [e for e in r.events if e["type"] == "winchester"]
    assert len(ev) == 6 and all(e["t"] == 0.0 for e in ev)
    assert r.end_reason == END_RED_DEPARTED and r.time_s == pytest.approx(0.5)
    assert all(j.turn_aways == 0 and j.state == "departed"
               for j in seen["defense"].jets.values())


def test_formation_and_assigned_altitude_held_before_trigger():
    p = next(q for q in (sample_presentation(s) for s in range(200))
             if q.range_nm - q.maneuver["trigger_range_nm"] > 15.0)
    r = run_presentation(p, record=True, cfg=dataclasses.replace(CFG, max_time_s=40.0))
    assert not r.preplanned["fired"]
    fr = r.frames[-1]["aircraft"]
    lead = fr["R1"]
    h = lead["heading"]
    fx, fy, rx, ry = math.sin(h), math.cos(h), math.cos(h), -math.sin(h)
    for j in p.red_jets:
        st = fr[j["id"]]
        dx, dy = st["x"] - lead["x"], st["y"] - lead["y"]
        right, back = (dx * rx + dy * ry) / NM_M, -(dx * fx + dy * fy) / NM_M
        assert abs(right - j["right_nm"]) < 0.5 and abs(back - j["back_nm"]) < 0.5
        assert st["alt"] == pytest.approx(j["hot_alt_m"], abs=1.0)   # not Blue's altitude


def _heading_to_nearest_blue(fr, rid):
    a = fr["aircraft"][rid]
    blues = [b for k, b in fr["aircraft"].items() if k.startswith("B") and b["alive"]]
    b = min(blues, key=lambda b: math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
    brg = math.atan2(b["x"] - a["x"], b["y"] - a["y"])
    return math.degrees((a["heading"] - brg + math.pi) % (2 * math.pi) - math.pi)


@pytest.mark.parametrize("mtype", sorted(MENUS["maneuvers"]))
def test_each_maneuver_fires_on_its_range_trigger(mtype):
    p = sample_presentation(21, stratum=(mtype, "aggressive", "shoot_assess_shoot"))
    r = run_presentation(p, record=True)
    pp = r.preplanned
    trig = p.maneuver["trigger_range_nm"]
    assert pp["fired"] and trig - 0.5 <= pp["fire_range_nm"] <= trig
    starts = [e for e in r.events if e["type"] == "preplanned_start"]
    assert len(starts) == 1 and starts[0]["t"] == pp["fire_t"]
    assert pp["status_counts"].get("done", 0) >= 1
    t_chk = pp["fire_t"] + 15.0
    fr = next(f for f in r.frames if f["t"] >= t_chk)
    halves = p.maneuver["halves"]
    par = p.maneuver["params"]
    if mtype == "split":
        for half, sign in ((halves[0], -1), (halves[1], 1)):
            for slot in half:
                off = _heading_to_nearest_blue(fr, f"R{slot}")
                assert off * sign == pytest.approx(par["angle_deg"], abs=8.0)
    elif mtype == "pump":
        for j in p.red_jets:
            assert abs(_heading_to_nearest_blue(fr, j["id"])) > 150.0
            assert fr["aircraft"][j["id"]]["alt"] == pytest.approx(j["hot_alt_m"], abs=1.0)
    else:
        for j in p.red_jets:
            if mtype == "altitude_change":
                want = par["low_alt_m"] if par["direction"] == "low" else par["high_alt_m"]
            else:
                half = 0 if j["slot"] in halves[0] else 1
                want = (par["low_alt_m"] if half == par["low_half"]
                        else min(13500.0, j["hot_alt_m"] + par["high_delta_m"]))
            assert pp["jets"][j["id"]]["hot_alt_m"] == pytest.approx(want)
        # altitude moving toward the target at the check time
        j = p.red_jets[0]
        a0 = j["alt"]
        want0 = pp["jets"][j["id"]]["hot_alt_m"]
        a1 = fr["aircraft"][j["id"]]["alt"]
        assert abs(a1 - want0) < abs(a0 - want0) or abs(a1 - want0) <= 50.0


def test_defense_overrides_maneuver_abort():
    p = sample_presentation(0, stratum=("split", "conservative", "shoot_assess_shoot"))
    seen = {}
    r = run_presentation(p, prepare=lambda ac, d: seen.setdefault("d", d))
    aborted = [k for k, v in r.preplanned["jets"].items() if v["status"] == "aborted"]
    assert aborted
    for k in aborted:
        t_end = r.preplanned["jets"][k]["t_end"]
        ab = [e for e in r.events if e["type"] == "preplanned_abort" and e["observer"] == k]
        df = [e for e in r.events if e["type"] == "defend" and e["observer"] == k]
        assert ab and ab[0]["t"] == t_end and df and df[0]["t"] == t_end
    # maneuvers never count as turn-aways: counts equal counted defend events
    for k, jd in seen["d"].jets.items():
        n_def = sum(1 for e in r.events if e["type"] == "defend" and e["observer"] == k
                    and e.get("counted"))
        assert jd.turn_aways == n_def


def test_jet_not_hot_at_trigger_skips():
    p = sample_presentation(21, stratum=("pump", "aggressive", "shoot_assess_shoot"))

    def prep(aircraft, defense):
        r2 = next(a for a in aircraft if a.id == "R2")
        jd = defense.jet(r2)
        jd.state, jd.cold_until, jd.reaction = COLD, 1e9, "beam"
    r = run_presentation(p, prepare=prep, cfg=dataclasses.replace(CFG, max_time_s=150.0))
    assert r.preplanned["fired"]
    assert r.preplanned["jets"]["R2"]["status"] == "skipped"
    assert any(e["type"] == "preplanned_skip" and e["observer"] == "R2" for e in r.events)
    assert r.preplanned["status_counts"].get("skipped") == 1
