"""Spec 6b: GA stability (moving-average stagnation trigger, immigration,
fair champion pick on paired benchmark fights, 36 presentations, history
files). docs/specs/06b-ga-stability.md"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from stealth_tactics.neuro.genome import NetGenome
from stealth_tactics.neuro.mlp import Arch
from stealth_tactics.neuro.neuroga import MemberEval, NeuroConfig, NeuroGA
from stealth_tactics.neuro.novelty import hof_cell
from stealth_tactics.neuro.runconfig import add_run_args, config_from_args

SMALL = Arch(hidden=(4,)).to_dict()
QUIET = dict(workers=1, log=lambda *a: None)


class PairedTask:
    """Fake task: eval fitness = mean of weights 0-9 (+ per-generation noise);
    the benchmark has ``n_bench`` fights; fight j scores base + common_j
    (shared by every genome, so the pairing removes it) + own noise."""

    def __init__(self, n_bench=16, flat=False, own=0.05):
        self.n_bench, self.flat, self.own = n_bench, flat, own
        self.bench_log = []
        self.common = np.random.default_rng(7).normal(0, 1.0, n_bench)

    def eval_set(self, gen):
        return [{"presentation_seed": gen}]

    def bench_set(self):
        return [{"presentation_seed": -1 - j} for j in range(self.n_bench)]

    def test_set(self):
        return []

    def evaluate(self, ga, genomes, pres, tag):
        out = []
        for g in genomes:
            w = g.weights
            base = 0.0 if self.flat else float(w[:10].mean())
            raw = {"radar_off_share": float(1 / (1 + np.exp(-10 * w[10]))),
                   "launch_r_rmax": float(0.4 + 0.6 / (1 + np.exp(-10 * w[11])))}
            if tag.startswith("b"):
                rng = np.random.default_rng(abs(hash(w.tobytes())) % 2 ** 32)
                per = [0.0 if self.flat else base + c + self.own * rng.standard_normal()
                       for c in self.common]
                self.bench_log.append(g.lineage["id"])
            else:
                per = [base]
            f = float(np.mean(per))
            out.append(MemberEval(f, per, np.array([raw["radar_off_share"],
                                                    raw["launch_r_rmax"]] + [0.0] * 6),
                                  raw, [{}] * len(per)))
        return out


def _ga(tmp, task=None, **kw):
    base = dict(population=20, elites=2, truncation=5, init="random", arch=SMALL)
    base.update(kw)
    return NeuroGA(NeuroConfig(**base), tmp, task=task or PairedTask(), **QUIET)


# ---------------------------------------------------------------- defaults --
def test_defaults_and_cli_flags():
    c = NeuroConfig()
    assert c.presentations == 36
    assert (c.stagnation_gens, c.stagnation_window, c.stagnation_metric) == (60, 10, "best_ma")
    assert (c.immigrate_every, c.immigrants, c.stagnation_immigrants) == (25, 8, 16)
    assert (c.boost_gens, c.boost_sigma_mult) == (0, 1.0)       # no global boost
    assert (c.top_every, c.top_n, c.champion_margin_k) == (1, 3, 1.0)
    p = argparse.ArgumentParser()
    add_run_args(p, None)
    a = p.parse_args([])
    assert a.presentations == 36
    cfg = config_from_args(a)
    assert cfg.presentations == 36 and cfg.stagnation_gens == 60 and cfg.immigrants == 8
    a = p.parse_args(["--stagnation-gens", "0", "--stagnation-window", "5", "--immigrate-every",
                      "0", "--immigrants", "5", "--immigrant-random-frac", "1",
                      "--stagnation-immigrants", "10", "--champion-k", "2", "--top-n", "4"])
    cfg = config_from_args(a)
    assert (cfg.stagnation_gens, cfg.stagnation_window, cfg.immigrate_every, cfg.immigrants,
            cfg.immigrant_random_frac, cfg.stagnation_immigrants, cfg.champion_margin_k,
            cfg.top_n) == (0, 5, 0, 5, 1.0, 10, 2.0, 4)
    with pytest.raises(ValueError):
        NeuroConfig(stagnation_metric="nope").validate()
    with pytest.raises(ValueError):
        NeuroConfig(immigrant_random_frac=1.5).validate()


# ------------------------------------------------- moving-average trigger --
def test_moving_average_trigger_counts_and_fires(tmp_path):
    ga = _ga(tmp_path, stagnation_gens=5, stagnation_window=3)
    seq = [0, 1, 2, 3, 4] + [4] * 12
    mas, fired = [], []
    for b in seq:
        ma, f = ga._stagnation_update(improved=True, best_now=b)   # champion change ignored
        ga.history.append({"best_fitness": float(b)})
        mas.append(ma)
        fired.append(f)
    assert mas[:7] == pytest.approx([0, 0.5, 1, 2, 3, 11 / 3, 4])
    # last new MA high at index 6; 5 generations without one -> fires at 11, again at 16
    assert [i for i, f in enumerate(fired) if f] == [11, 16]


def test_moving_average_ignores_single_noisy_dip_and_can_be_disabled(tmp_path):
    ga = _ga(tmp_path, stagnation_gens=10, stagnation_window=4)
    for b in [10, 10, 10, 10, -50, 10, 10, 10, 10, 12]:
        ma, f = ga._stagnation_update(False, b)
        ga.history.append({"best_fitness": float(b)})
    # a dip lowers the MA but is not a new high; one +2 spike moves the MA by only 0.5
    assert ga.best_ma_max == pytest.approx(10.5) and ga.stagnation == 0
    ga._stagnation_update(False, 10.0)
    assert ga.stagnation == 1
    ga2 = _ga(tmp_path / "off", stagnation_gens=0)
    assert not any(ga2._stagnation_update(False, 0.0)[1] for _ in range(200))


def test_stagnation_trigger_immigrates_without_global_changes(tmp_path):
    ga = _ga(tmp_path, task=PairedTask(flat=True), stagnation_gens=3, stagnation_window=2,
             stagnation_immigrants=6, immigrate_every=0)
    ga.init_population()
    for _ in range(5):
        sig_before = [p.sigma for p in ga.population]
        row = ga.step()
        if row["stagnation_trigger"]:
            assert row["immigrants"] == 6 and row["immigrant_reason"] == "stagnation"
            assert not row["boosted"]
            # elites keep their sigma: no global sigma x2
            for j, i in enumerate(row["elites"]):
                assert ga.population[j].sigma == sig_before[i]
    h = ga.history
    assert [r["stagnation_trigger"] for r in h] == [False, False, False, True, False]
    assert {r["novelty_w"] for r in h} == {0.5}
    assert [e["gen"] for e in ga.events if e["type"] == "stagnation"] == [3]
    assert [e["gen"] for e in ga.events if e["type"] == "immigration"] == [3]


# -------------------------------------------------------------- immigration --
def _fake_hof_entry(gid, root, bench, w):
    return {"genome": NetGenome([w], 0.01, {"id": gid, "parent": None, "born": 0,
                                            "origin": "random", "root": root}),
            "benchmark_fitness": bench}


def test_immigration_count_sources_and_offspring_unchanged(tmp_path):
    kw = dict(immigrants=6, immigrant_random_frac=0.5, stagnation_gens=0,
              immigrant_hof_lineage="strict", immigrant_parent_slots=1)
    a = _ga(tmp_path / "a", immigrate_every=3, **kw)
    b = _ga(tmp_path / "b", immigrate_every=0, **kw)
    for ga in (a, b):
        ga.init_population()
        for _ in range(2):
            ga.step()
    ch = a.champion["genome"].lineage
    ch_root = ch.get("root", ch["id"])
    w = a.population[0].weights
    # one cell from another lineage, one cell from the champion's own lineage
    extra = {"zz_other": _fake_hof_entry(9001, 9001, 99.0, w + 1.0),
             "zz_champ": _fake_hof_entry(9002, ch_root, 100.0, w - 1.0)}
    a.hof.update(extra)
    b.hof.update(extra)
    sig_a = [p.sigma for p in a.population]
    ra, rb = a.step(), b.step()                          # generation 2: (2 + 1) % 3 == 0
    assert ra["immigrants"] == 6 and ra["immigrant_reason"] == "periodic"
    assert rb["immigrants"] == 0
    imm = [p for p in a.population if p.lineage.get("immig") == 3]
    assert len(imm) == 6 and len(a.population) == 20
    rand = [p for p in imm if p.lineage["origin"] == "immigrant_random"]
    hofd = [p for p in imm if p.lineage["origin"] == "immigrant_hof"]
    assert len(rand) == 3 and len(hofd) == 3
    assert ra["immigrant_sources"]["random"] == 3 and ra["immigrant_sources"]["hof"] == 3
    eligible = {c for c, e in a.hof.items()
                if e["genome"].lineage.get("root", e["genome"].lineage["id"]) != ch_root
                and e["genome"].lineage["id"] != ch["id"]}
    assert "zz_other" in eligible and "zz_champ" not in eligible
    for p in hofd:
        assert p.lineage["hof_cell"] in eligible
        assert p.lineage["root"] != ch_root
        assert p.lineage["parent"] == a.hof[p.lineage["hof_cell"]]["genome"].lineage["id"]
    assert all(p.lineage["root"] == p.lineage["id"] and p.lineage["parent"] is None for p in rand)
    # elites and the other offspring are exactly what a run without immigration
    # breeds (no global sigma or novelty change); immigrants take the last slots
    n = 20 - 6
    for pa, pb in zip(a.population[:n], b.population[:n]):
        assert np.array_equal(pa.weights, pb.weights) and pa.sigma == pb.sigma
        assert pa.lineage == pb.lineage
    for j, i in enumerate(ra["elites"]):
        assert a.population[j].sigma == sig_a[i]
    assert ra["novelty_w"] == rb["novelty_w"] == 0.5
    # the next generation reserves a truncation-parent slot for an immigrant lineage
    before = list(a.population)
    r2 = a.step()
    assert any("immig" in before[i].lineage for i in r2["parents"])
    assert len(r2["parents"]) == 5


def test_immigration_falls_back_to_random_without_eligible_hof(tmp_path):
    ga = _ga(tmp_path, immigrate_every=2, immigrants=4, stagnation_gens=0)
    ga.init_population()
    ga.step()
    ga.immigrant_cells = lambda: []          # no eligible hall-of-fame cell
    row = ga.step()
    assert row["immigrants"] == 4 and row["immigrant_sources"]["random"] == 4
    assert sum(p.lineage.get("origin") == "immigrant_random" for p in ga.population) == 4


def test_immigrants_never_replace_elites(tmp_path):
    ga = _ga(tmp_path, population=6, immigrate_every=1, immigrants=50, stagnation_gens=0)
    ga.init_population()
    row = ga.step()
    assert row["immigrants"] == 6 - 2 - 1
    assert sum("immig" in p.lineage for p in ga.population) == 3


# ---------------------------------------------------------- champion pick --
def _entry(gid, per):
    per = list(map(float, per))
    return {"genome": NetGenome([np.zeros(3)], 0.01, {"id": gid}), "benchmark_per_fight": per,
            "benchmark_fitness": float(np.mean(per))}


def test_champion_margin_paired_difference(tmp_path):
    ga = _ga(tmp_path)
    rng = np.random.default_rng(0)
    common = rng.normal(0, 100, 64)                      # fight difficulty, shared
    ga.champion = _entry(1, common)
    d = rng.normal(0, 8, 64)
    d = d - d.mean()
    se = d.std(ddof=1) / 8
    clear = _entry(2, common + d + 3 * se)               # +3 SE: accepted
    close = _entry(3, common + d + 0.5 * se)             # +0.5 SE: higher but rejected
    dec = ga._champion_decision([ga.champion, close])
    assert dec["challenger_id"] == 3 and not dec["accepted"]
    assert dec["mean_diff"] == pytest.approx(0.5 * se) and dec["se"] == pytest.approx(se)
    assert dec["reason"] == "within k*SE of champion"
    dec = ga._champion_decision([close, clear, ga.champion])
    assert dec["challenger_id"] == 2 and dec["accepted"]
    assert dec["margin"] == pytest.approx(se)
    # pairing matters: the unpaired spread (std 100) would hide a +3 SE gain
    assert np.std(clear["benchmark_per_fight"], ddof=1) / 8 > dec["mean_diff"]
    ga.cfg.champion_margin_k = 4.0
    assert not ga._champion_decision([clear])["accepted"]
    ga.cfg.champion_margin_k = 0.0                       # spec 6 rule: higher wins
    assert ga._champion_decision([close])["accepted"]
    ga.cfg.champion_margin_k = 1.0
    worse = _entry(4, common - 1.0)
    assert ga._champion_decision([worse])["reason"] == "lower benchmark"
    assert ga._champion_decision([ga.champion])["challenger_id"] is None


def test_champion_decisions_logged_and_benchmark_cache(tmp_path):
    task = PairedTask(n_bench=16)
    ga = _ga(tmp_path, task=task, stagnation_gens=0, immigrate_every=0)
    ga.init_population()
    champ = []
    for _ in range(8):
        row = ga.step()
        dec = row["champion_decision"]
        champ.append(row["champion_bench"])
        if dec["accepted"] and dec["champion_id"] is not None:
            assert dec["mean_diff"] > dec["margin"] >= 0
            assert dec["challenger_bench"] > dec["champion_bench"]
        assert row["champion_id"] == ga.champion["genome"].lineage["id"]
    assert all(b2 >= b1 for b1, b2 in zip(champ, champ[1:]))
    # every genome is benchmarked at most once (the incumbent and returning
    # elites reuse their stored scores); top 3 considered every generation
    assert len(task.bench_log) == len(set(task.bench_log))
    assert len(task.bench_log) < 3 * 8
    timing = [json.loads(l) for l in (tmp_path / "timing.jsonl").read_text().splitlines()]
    assert sum(t["bench_runs"] for t in timing) == len(task.bench_log)
    assert all(t["bench_runs"] + t["bench_cached"] == 3 for t in timing)
    # same decisions without the cache (the cache only saves time)
    task2 = PairedTask(n_bench=16)
    ga2 = _ga(tmp_path / "nc", task=task2, stagnation_gens=0, immigrate_every=0,
              bench_cache=False)
    ga2.init_population()
    for _ in range(8):
        ga2.step()
    assert [r["champion_decision"] for r in ga2.history] == [r["champion_decision"]
                                                              for r in ga.history]
    assert len(task2.bench_log) == 3 * 8


# ------------------------------------------------------ history / resume --
def test_history_files_written_every_generation(tmp_path):
    ga = _ga(tmp_path, immigrate_every=2, immigrants=4, stagnation_gens=0)
    ga.init_population()
    for _ in range(3):
        ga.step()
    rows = list(csv.DictReader((tmp_path / "history.csv").open()))
    assert [int(r["generation"]) for r in rows] == [0, 1, 2]
    for col in ("best_fitness", "mean_fitness", "median_fitness", "best_ma", "mean_sigma",
                "immigrants", "immigrant_reason", "stagnation_trigger", "champion_accepted",
                "paired_diff", "paired_se", "champion_bench", "challenger_id"):
        assert col in rows[0]
    assert rows[1]["immigrants"] == "4" and rows[1]["immigrant_reason"] == "periodic"
    js = json.loads((tmp_path / "history.json").read_text())
    assert len(js) == 3 and js[0]["champion_decision"]["reason"] == "first champion"
    assert js[1]["immigrant_sources"]["random"] + js[1]["immigrant_sources"]["hof"] == 4
    from stealth_tactics.neuro.overnight import _chart
    pytest.importorskip("matplotlib")
    assert _chart(ga.history, tmp_path / "fitness.png")
    assert (tmp_path / "fitness.png").stat().st_size > 1000


def test_fixed_seed_deterministic_and_resume_across_immigration(tmp_path):
    kw = dict(immigrate_every=3, immigrants=5, stagnation_gens=2, stagnation_window=2,
              stagnation_immigrants=7)

    def files(d):
        return [(d / "checkpoints" / f"ckpt_g0008{e}").read_bytes() for e in (".npz", ".json")] \
            + [(d / "history.csv").read_bytes()]
    _ga(tmp_path / "a", **kw).run(8, finalize=False)
    _ga(tmp_path / "b", **kw).run(8, finalize=False)
    _ga(tmp_path / "c", **kw).run(4, finalize=False)
    _ga(tmp_path / "c", **kw).run(8, resume=True, finalize=False)
    assert files(tmp_path / "a") == files(tmp_path / "b") == files(tmp_path / "c")
    h = json.loads((tmp_path / "a" / "history.json").read_text())
    assert any(r["immigrants"] for r in h)
    other = _ga(tmp_path / "d", master_seed=7, **kw)
    other.run(8, finalize=False)
    assert files(tmp_path / "d") != files(tmp_path / "a")


def test_sim_immigration_and_top3_identical_across_workers(tmp_path):
    cfg = NeuroConfig(population=4, presentations=2, benchmark=2, test_size=0, truncation=2,
                      init="random", max_time_s=60.0, immigrate_every=2, immigrants=1)
    for name, workers in (("a", 1), ("b", 4)):
        NeuroGA(cfg, tmp_path / name, workers=workers, log=lambda *a: None).run(
            3, finalize=False)
    for e in (".npz", ".json"):
        assert (tmp_path / "a/checkpoints" / f"ckpt_g0003{e}").read_bytes() == \
            (tmp_path / "b/checkpoints" / f"ckpt_g0003{e}").read_bytes()
    h = json.loads((tmp_path / "a" / "history.json").read_text())
    assert h[1]["immigrants"] == 1


# ------------------------------------------------- newcomer fixes (v2) ----
def test_newcomer_defaults():
    c = NeuroConfig()
    assert (c.immigrant_protect_gens, c.immigrant_parent_slots) == (25, 2)
    assert (c.immigrant_max_per_cell, c.immigrant_hof_lineage) == (2, "prefer_other")


def _cells_ga(tmp_path, **kw):
    ga = _ga(tmp_path, immigrate_every=0, stagnation_gens=0, **kw)
    ga.init_population()
    ga.step()
    ch = ga.champion["genome"].lineage
    root = ch.get("root", ch["id"])
    w = ga.population[0].weights
    ch_cell = hof_cell(ga.champion["bench_raw"])
    ga.hof = {"c_champ_genome": {**ga.champion},
              "o1": _fake_hof_entry(9101, 9101, 50.0, w + 0.5),
              "s1": _fake_hof_entry(9201, root, 90.0, w + 1.0),
              "s2": _fake_hof_entry(9202, root, 80.0, w + 1.5),
              "s3": _fake_hof_entry(9203, root, 70.0, w + 2.0)}
    if ch_cell:
        ga.hof[ch_cell] = _fake_hof_entry(9301, root, 99.0, w - 1.0)
    return ga, ch_cell


def test_immigrant_cells_distinct_capped_and_ordered(tmp_path):
    ga, ch_cell = _cells_ga(tmp_path, immigrants=8, immigrant_random_frac=0.0)
    cells = ga.immigrant_cells()
    # other-founder cells first, then same-founder cells by benchmark; never the
    # champion's genome or the champion's own behaviour cell
    assert cells[0] == "o1" and cells[1:] == ["s1", "s2", "s3"]
    assert "c_champ_genome" not in cells and ch_cell not in cells
    imm, src = ga._make_immigrants(8, ga.gen)
    from collections import Counter
    cnt = Counter(src["hof_cells"])
    assert src["hof"] == 8 and len(cnt) == 4 and max(cnt.values()) <= 2
    assert all(p.lineage["origin"] == "immigrant_hof" for p in imm)
    # more slots than cells x 2 -> the rest are random
    imm, src = ga._make_immigrants(12, ga.gen)
    assert src["hof"] == 8 and src["random"] == 4
    # strict = the spec 6b founder rule: only o1, at most 2 newcomers from it
    ga.cfg.immigrant_hof_lineage = "strict"
    assert ga.immigrant_cells() == ["o1"]
    imm, src = ga._make_immigrants(8, ga.gen)
    assert src["hof"] == 2 and src["random"] == 6 and set(src["hof_cells"]) == {"o1"}


def test_immigrant_protection_length_and_parent_slots(tmp_path):
    ga = _ga(tmp_path, immigrate_every=0, stagnation_gens=0)
    g0 = NetGenome([np.zeros(3)], 0.01, {"id": 1, "immig": 10})
    assert ga._protected(g0, 34) and not ga._protected(g0, 35)     # 25 generations
    ga.init_population()
    for i, p in enumerate(ga.population):
        if i in (17, 18, 19):
            p.lineage["immig"] = 0
    comb = np.arange(20, dtype=float)          # member i has combined rank i
    par = ga._select_parents(comb, 5)
    # two reserved slots go to the best protected members (17, 18); 3 by merit
    assert sorted(par) == [0, 1, 2, 17, 18]
    ga.cfg.immigrant_parent_slots = 1
    assert sorted(ga._select_parents(comb, 5)) == [0, 1, 2, 3, 17]
    assert sorted(ga._select_parents(comb, 30)) == [0, 1, 2, 3, 4]  # protection over
