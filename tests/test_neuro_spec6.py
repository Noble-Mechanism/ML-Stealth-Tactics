"""Spec 6: neural policy and neuroevolution (docs/specs/06-neural-policy-neuroevolution.md)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stealth_tactics.neuro.checkpoint import load_npz, save_npz
from stealth_tactics.neuro.evaluate import make_pool, pmap, worker_env
from stealth_tactics.neuro.genome import (NetGenome, NetworkCountError, SIGMA_MAX, SIGMA_MIN,
                                          TAU, check_network_count, load_genome, mutate,
                                          save_genome, unit_swap)
from stealth_tactics.neuro.mlp import (Arch, MLPPolicy, flatten, init_weights,
                                       interface_fingerprint, unflatten)
from stealth_tactics.neuro.neuroga import MemberEval, NeuroConfig, NeuroGA, _rng, P_MUT
from stealth_tactics.neuro.novelty import hof_cell, novelty, ranks
from stealth_tactics.neuro.toy import ToyTask
from stealth_tactics.policy import ACTION_SPEC as A, OBS_SPEC as S

RNG = np.random.default_rng(0)


def _obs(n=16, seed=0):
    """Random but well-formed observation rows (present flags 0/1)."""
    r = np.random.default_rng(seed)
    x = r.uniform(-1, 1, (n, S.size))
    for i in range(S.k):
        x[:, S.contact_slice(i).start] = (r.random(n) < 0.6).astype(float)
    return x


# ------------------------------------------------------------ network (A-E)
def test_param_counts_and_heads():
    assert Arch().n_params == 19853
    assert Arch(hidden=(32, 32)).n_params == 8909
    assert Arch(hidden=(128, 128)).n_params == 47885
    assert Arch(encoder="set").n_params == 13832
    for arch in (Arch(), Arch(encoder="set")):
        w = init_weights(arch, np.random.default_rng(1))
        y = MLPPolicy(w, arch)(_obs())
        assert y.shape == (16, 13) and np.isfinite(y).all()
        assert (np.abs(y[:, :3]) <= 1).all() and (np.abs(y[:, A.fire:]) <= 1).all()
    # target logits linear (flat): not squashed
    w = init_weights(Arch(), np.random.default_rng(2)) * 5
    y = MLPPolicy(w)(_obs())
    assert np.abs(y[:, A.target_slice]).max() > 1.0


def test_flat_layers_round_trip_and_batch_order():
    arch = Arch()
    w = init_weights(arch, np.random.default_rng(3))
    assert np.array_equal(flatten(unflatten(w, arch), arch), w)
    with pytest.raises(ValueError):
        unflatten(w[:-1], arch)
    p = MLPPolicy(w)
    x = _obs(12)
    y = p(x)
    perm = np.random.default_rng(4).permutation(12)
    assert np.array_equal(p(x[perm]), y[perm])
    # one row at a time: BLAS gemv vs gemm may differ in the last bit, so a
    # jet's output is deterministic for a given batch, equal to ~1e-15 across sizes
    assert np.allclose(np.vstack([p(x[i:i + 1]) for i in range(12)]), y, rtol=0, atol=1e-12)


def test_set_encoder_is_slot_equivariant():
    arch = Arch(encoder="set")
    p = MLPPolicy(init_weights(arch, np.random.default_rng(5)), arch)
    x = _obs(6, 7)
    x[:, [S.contact_slice(i).start for i in range(S.k)]] = 1.0
    y = p(x)
    perm = [3, 0, 5, 1, 4, 2]
    xp = x.copy()
    for new, old in enumerate(perm):
        xp[:, S.contact_slice(new)] = x[:, S.contact_slice(old)]
    yp = p(xp)
    assert np.allclose(yp[:, [0, 1, 2, A.fire, A.radar, A.pair]],
                       y[:, [0, 1, 2, A.fire, A.radar, A.pair]])
    assert np.allclose(yp[:, A.target0:A.target0 + 6], y[:, [A.target0 + o for o in perm]])
    # absent slots never win
    x2 = x.copy()
    x2[:, S.contact_slice(2).start] = 0.0
    assert (p(x2)[:, A.target0 + 2] < -1e8).all()


def test_genome_save_load_identical_outputs(tmp_path):
    arch = Arch()
    g = NetGenome([init_weights(arch, np.random.default_rng(8))], 0.03,
                  {"id": 7, "parent": 3, "born": 2, "origin": "clone"})
    save_genome(tmp_path / "g", g, arch)
    g2, arch2, meta = load_genome(tmp_path / "g")
    assert arch2 == arch and meta["n_networks"] == 1 and meta["interface"] == interface_fingerprint()
    assert g2.sigma == 0.03 and g2.lineage == g.lineage
    x = _obs()
    assert np.array_equal(g.policy(arch)(x), g2.policy(arch2)(x))


def test_network_count_field_round_trips_and_other_counts_refused(tmp_path):
    arch = Arch()
    g = NetGenome([init_weights(arch, np.random.default_rng(9))])
    save_genome(tmp_path / "one", g, arch)
    assert json.loads((tmp_path / "one.json").read_text())["n_networks"] == 1
    assert load_genome(tmp_path / "one")[0].n_networks == 1
    for n in (2, 4):
        g2 = NetGenome([g.weights.copy() for _ in range(n)])
        save_genome(tmp_path / f"n{n}", g2, arch)
        assert json.loads((tmp_path / f"n{n}.json").read_text())["n_networks"] == n
        with pytest.raises(NetworkCountError, match="supports only 1 shared network"):
            load_genome(tmp_path / f"n{n}")
    with pytest.raises(NetworkCountError, match="deferred"):
        check_network_count(2)
    with pytest.raises(NetworkCountError):
        NeuroConfig(n_networks=2).validate()
    with pytest.raises(ValueError, match="jet_network"):
        NeuroConfig(jet_network=(0, 0, 1, 1)).validate()
    assert NeuroConfig.from_dict(NeuroConfig().to_dict()).jet_network == (0, 0, 0, 0)


def test_interface_fingerprint_mismatch_refused(tmp_path):
    arch = Arch()
    save_genome(tmp_path / "g", NetGenome([init_weights(arch, np.random.default_rng(1))]), arch)
    meta = json.loads((tmp_path / "g.json").read_text())
    meta["interface"] = "deadbeef"
    (tmp_path / "g.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="interface fingerprint"):
        load_genome(tmp_path / "g")


def test_deterministic_npz_bytes(tmp_path):
    a = {"x": np.arange(5.0), "y": np.eye(3)}
    save_npz(tmp_path / "a.npz", a)
    save_npz(tmp_path / "b.npz", a)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    assert np.array_equal(load_npz(tmp_path / "a.npz")["y"], np.eye(3))


# ------------------------------------------------------------ mutation (I, J)
def test_mutation_statistics():
    arch = Arch()
    w = init_weights(arch, np.random.default_rng(10))
    par = NetGenome([w], 0.02, {"id": 1, "origin": "random"})
    ch = mutate(par, np.random.default_rng(11), 99, 5)
    d = ch.weights - w
    assert abs(d.std() / ch.sigma - 1) < 0.02 and abs(d.mean()) < 0.05 * ch.sigma
    assert ch.lineage == {"id": 99, "parent": 1, "born": 5, "origin": "random", "root": 1}
    logs = []
    for i in range(3000):
        s = mutate(NetGenome([np.zeros(3)], 0.02, {}), np.random.default_rng(i), 0, 0).sigma
        logs.append(math.log(s / 0.02))
    logs = np.array(logs)
    assert abs(logs.std() - TAU) < 0.01 and abs(logs.mean()) < 0.01
    lo = mutate(NetGenome([np.zeros(3)], SIGMA_MIN, {}), np.random.default_rng(1), 0, 0, tau=5)
    hi = mutate(NetGenome([np.zeros(3)], SIGMA_MAX, {}), np.random.default_rng(2), 0, 0, tau=5)
    assert SIGMA_MIN <= lo.sigma <= SIGMA_MAX and SIGMA_MIN <= hi.sigma <= SIGMA_MAX
    # same (generation, child) seed -> identical; different child index -> different
    a = mutate(par, _rng(2026, 3, P_MUT, 4), 0, 0)
    b = mutate(par, _rng(2026, 3, P_MUT, 4), 0, 0)
    c = mutate(par, _rng(2026, 3, P_MUT, 5), 0, 0)
    assert np.array_equal(a.weights, b.weights) and a.sigma == b.sigma
    assert not np.array_equal(a.weights, c.weights)


def test_unit_swap_moves_whole_units():
    arch = Arch()
    a = NetGenome([init_weights(arch, np.random.default_rng(1))])
    b = NetGenome([init_weights(arch, np.random.default_rng(2))])
    c = unit_swap(a, b, np.random.default_rng(3), arch)
    La, Lb, Lc = (unflatten(g.weights, arch) for g in (a, b, c))
    from_b = [u for u in range(64) if np.array_equal(Lc["W0"][:, u], Lb["W0"][:, u])]
    assert 0 < len(from_b) < 64
    for u in range(64):
        src = Lb if u in from_b else La
        assert np.array_equal(Lc["W0"][:, u], src["W0"][:, u])
        assert Lc["b0"][u] == src["b0"][u]
        assert np.array_equal(Lc["W1"][u], src["W1"][u])
    assert np.array_equal(Lc["W2"], La["W2"])


# ------------------------------------------------------------ novelty (L)
def test_novelty_ranks_archive_and_hof_cells():
    bds = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    nov = novelty(bds, np.zeros((0, 2)), k=2)
    assert nov[3] > nov[0] and nov[0] == nov[1] == nov[2] == 0.0   # duplicates lowest
    assert list(ranks(np.array([3.0, 1.0, 3.0, 5.0]))) == [1, 3, 2, 0]
    assert hof_cell({"radar_off_share": 0.0, "launch_r_rmax": 0.5}) == "r0_l0"
    assert hof_cell({"radar_off_share": 0.5, "launch_r_rmax": 0.7}) == "r1_l1"
    assert hof_cell({"radar_off_share": 0.9, "launch_r_rmax": 0.95}) == "r2_l2"
    assert hof_cell({"radar_off_share": 0.9, "launch_r_rmax": None}) is None


class FakeTask:
    """Fitness and behaviour from the weights: fitness = mean of the first
    10 weights; BD / hall-of-fame cell from weights 10-11. Benchmark = same
    formula + 0.01 * weight 12. Optional constant fitness (stagnation)."""

    def __init__(self, flat=False):
        self.flat = flat
        self.bench_log = []

    def eval_set(self, gen):
        return [{"presentation_seed": gen}]

    def bench_set(self):
        return [{"presentation_seed": -1}]

    def test_set(self):
        return []

    def evaluate(self, ga, genomes, pres, tag):
        out = []
        for g in genomes:
            w = g.weights
            f = 0.0 if self.flat else float(w[:10].mean())
            if tag.startswith("b"):
                f = f + (0.0 if self.flat else 0.01 * float(w[12]))
            raw = {"radar_off_share": float(1 / (1 + np.exp(-10 * w[10]))),
                   "launch_r_rmax": float(0.4 + 0.6 / (1 + np.exp(-10 * w[11])))}
            if tag.startswith("b"):
                self.bench_log.append((hof_cell(raw), f, g.lineage["id"]))
            out.append(MemberEval(f, [f], np.array([raw["radar_off_share"],
                                                    raw["launch_r_rmax"]] + [0.0] * 6),
                                  raw, [{}]))
        return out


def test_selection_elites_parents_and_hall_of_fame(tmp_path):
    cfg = NeuroConfig(population=20, elites=2, truncation=5, init="random", top_every=2,
                      bench_cache=False, arch=Arch(hidden=(4,)).to_dict())
    task = FakeTask()
    ga = NeuroGA(cfg, tmp_path, workers=1, task=task, log=lambda *a: None)
    ga.init_population()
    for _ in range(6):
        before = [p.copy() for p in ga.population]
        fits = np.array([p.weights[:10].mean() for p in before])
        row = ga.step()
        top = list(np.argsort(-fits, kind="stable"))
        # elites copied unchanged (weights, sigma, lineage)
        for j, i in enumerate(top[:2]):
            assert np.array_equal(ga.population[j].weights, before[i].weights)
            assert ga.population[j].lineage == before[i].lineage
        assert row["elites"] == [int(i) for i in top[:2]]
        # every child's parent is one of the recorded truncation parents
        pids = {before[i].lineage["id"] for i in row["parents"]}
        assert len(row["parents"]) == 5
        assert all(c.lineage["parent"] in pids for c in ga.population[2:])
        assert all(c.lineage["born"] == row["generation"] + 1 for c in ga.population[2:])
    # hall of fame keeps the best benchmark score per cell
    for cell, entry in ga.hof.items():
        best = max(f for c, f, _ in task.bench_log if c == cell)
        assert entry["benchmark_fitness"] == best
    assert ga.champion["benchmark_fitness"] == max(f for _, f, _ in task.bench_log)
    # top-3 benchmarking on every 2nd generation (top_every=2), else top-1
    assert len(task.bench_log) == 3 * 3 + 1 * 3


def test_stagnation_boost_and_archive_cap(tmp_path):
    # the spec 6 boost, still available through the config (spec 6b defaults differ)
    cfg = NeuroConfig(population=12, elites=2, truncation=4, init="random", stagnation_gens=3,
                      stagnation_metric="champion", boost_gens=2, boost_sigma_mult=2.0,
                      immigrate_every=0, stagnation_immigrants=0, champion_margin_k=0.0,
                      archive_add=2, archive_cap=5, arch=Arch(hidden=(4,)).to_dict())
    ga = NeuroGA(cfg, tmp_path, workers=1, task=FakeTask(flat=True), log=lambda *a: None)
    ga.init_population()
    sig = []
    for _ in range(8):
        sig.append(np.mean([p.sigma for p in ga.population]))
        ga.step()
    h = ga.history
    # champion set at gen 0; no improvement in gens 1-3 -> boost at 3; again at 6
    assert [r["boosted"] for r in h] == [False, False, False, True, False, False, True, False]
    assert [r["novelty_w"] for r in h] == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0]
    assert [e["gen"] for e in ga.events if e["type"] == "stagnation"] == [3, 6]
    assert sig[4] == pytest.approx(2 * sig[3], rel=0.3)          # sigma x2 at the boost
    assert len(ga.archive) == 5


# ------------------------------------------------------------ toy task ---
@pytest.fixture(scope="module")
def toy_data():
    from stealth_tactics.neuro.clone import _collect_job, presentations, CLONE_SALT
    parts = [_collect_job((p, 360.0)) for p in presentations(2026, CLONE_SALT, 4)]
    obs = np.vstack([o for o, _ in parts])
    act = np.vstack([a for _, a in parts])
    return obs[:2000], act[:2000]


def test_toy_task_fitness_rises(tmp_path, toy_data):
    obs, act = toy_data
    assert len(obs) == 2000
    ga = NeuroGA(NeuroConfig(init="random"), tmp_path, workers=1, task=ToyTask(obs, act),
                 log=lambda *a: None)
    ga.run(40, finalize=False)
    best = [r["best_fitness"] for r in ga.history]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))          # elites: never drops
    assert -best[-1] <= 0.5 * -best[0]                               # MSE at least halved
    assert all(SIGMA_MIN <= p.sigma <= SIGMA_MAX for p in ga.population)


def test_toy_checkpoint_resume_byte_identical(tmp_path, toy_data):
    obs, act = toy_data
    cfg = NeuroConfig(population=12, truncation=4, init="random")
    a = NeuroGA(cfg, tmp_path / "a", workers=1, task=ToyTask(obs, act), log=lambda *a: None)
    a.run(6, finalize=False)
    b = NeuroGA(cfg, tmp_path / "b", workers=1, task=ToyTask(obs, act), log=lambda *a: None)
    b.run(3, finalize=False)
    b2 = NeuroGA(cfg, tmp_path / "b", workers=1, task=ToyTask(obs, act), log=lambda *a: None)
    b2.run(6, resume=True, finalize=False)
    for ext in (".npz", ".json"):
        assert (tmp_path / "a/checkpoints" / f"ckpt_g0006{ext}").read_bytes() == \
            (tmp_path / "b/checkpoints" / f"ckpt_g0006{ext}").read_bytes()
    assert len(list((tmp_path / "a/checkpoints").glob("*.json"))) == 3   # last 3 kept
    other = NeuroConfig(population=12, truncation=4, init="random", novelty_w=0.3)
    c = NeuroGA(other, tmp_path / "b", workers=1, task=ToyTask(obs, act), log=lambda *a: None)
    with pytest.raises(ValueError, match="config hash"):
        c.load_checkpoint()


# ------------------------------------------------------------ sim ---------
TINY = dict(population=4, presentations=2, benchmark=2, test_size=0, truncation=2,
            init="random", max_time_s=60.0)


def _ckpt(d, g):
    return [(Path(d) / "checkpoints" / f"ckpt_g{g:04d}{e}").read_bytes() for e in (".npz", ".json")]


def test_sim_same_seed_twice_and_1_vs_8_workers_identical(tmp_path):
    cfg = NeuroConfig(**TINY)
    for name, workers in (("a", 1), ("b", 1), ("c", 8)):
        NeuroGA(cfg, tmp_path / name, workers=workers, log=lambda *a: None).run(2, finalize=False)
    assert _ckpt(tmp_path / "a", 2) == _ckpt(tmp_path / "b", 2) == _ckpt(tmp_path / "c", 2)


def test_sim_resume_byte_identical(tmp_path):
    cfg = NeuroConfig(**TINY)
    NeuroGA(cfg, tmp_path / "a", workers=1, log=lambda *a: None).run(6, finalize=False)
    NeuroGA(cfg, tmp_path / "b", workers=1, log=lambda *a: None).run(3, finalize=False)
    NeuroGA(cfg, tmp_path / "b", workers=1, log=lambda *a: None).run(6, resume=True,
                                                                     finalize=False)
    assert _ckpt(tmp_path / "a", 6) == _ckpt(tmp_path / "b", 6)


def test_workers_pin_blas_threads():
    pool = make_pool(2)
    try:
        envs = pmap(pool, worker_env, [0, 1, 2, 3])
    finally:
        pool.shutdown()
    assert all(v == "1" for e in envs for v in e.values())


def test_sim_smoke_8x4x5_and_champion_replay(tmp_path):
    """Test plan 3: pop 8 x 4 presentations x 5 generations, benchmark 8,
    held-out test 8; writes checkpoint, champion, hall of fame, ACMIs; the
    champion replays byte-exactly from its stored presentations."""
    from stealth_tactics.neuro.neuroga import replay_network_champion
    cfg = NeuroConfig(population=8, presentations=4, benchmark=8, test_size=8, truncation=8,
                      init="random")
    rep = NeuroGA(cfg, tmp_path, workers=8, log=lambda *a: None).run(5)
    for f in ("champion.json", "champion_weights.npz", "champion_weights.json",
              "champion_best.txt.acmi", "champion_worst.txt.acmi", "report.json",
              "checkpoints/ckpt_g0005.npz"):
        assert (tmp_path / f).exists(), f
    assert len(rep["history"]) == 5 and rep["champion"]["test_fitness"] is not None
    ok, text = replay_network_champion(tmp_path, workers=8)
    assert ok, text


def test_clone_small_passes_gate_and_uses_own_seeds():
    from stealth_tactics.neuro.clone import (CLONE_SALT, GATE_SALT, collect, fit, gate,
                                             presentations)
    from stealth_tactics.scenarios.presentation import build_benchmark_set, build_eval_set
    from stealth_tactics.neuro.neuroga import SimTask
    seeds_clone = {p["presentation_seed"] for p in presentations(2026, CLONE_SALT, 200)}
    seeds_gate = {p["presentation_seed"] for p in presentations(2026, GATE_SALT, 100)}
    bench = {p.presentation_seed for p in build_benchmark_set(2026, 64)}
    test = {p["presentation_seed"] for p in SimTask(NeuroConfig()).test_set()}
    evals = {p.presentation_seed for g in range(200) for p in build_eval_set(2026, g, 24)}
    assert not (seeds_clone & (bench | test | seeds_gate | evals))
    assert not (seeds_gate & (bench | test | evals))
    pool = make_pool(8)
    try:
        obs, act = collect(2026, 40, pool)
        w, losses = fit(obs, act, epochs=10)
        g = gate(w, Arch(), 2026, 24, pool)
    finally:
        pool.shutdown()
    assert losses[-1] < losses[0]
    assert g["kills_ratio"] >= 0.8, g
