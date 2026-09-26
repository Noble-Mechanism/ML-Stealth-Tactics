"""Spec 6 F-O: the neuroevolution loop.

Mutation-only GA with elitism (F), population 50 x 24 presentations (G),
2 re-scored elites + truncation parents from the top 10 by fitness rank +
0.5 x novelty rank (H, L), self-adaptive Gaussian mutation (I), no crossover
(J; unit swap behind ``unit_swap``), mixed clone / random start (K),
stagnation boost and 3 x 3 hall of fame (L), champion = best benchmark score
with top-3 benchmarking every 5th generation and a held-out test at the end
(M), seeds per (generation, purpose, index) (N), a byte-deterministic
checkpoint after every generation (O), champion storage (P)."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .checkpoint import load_npz, save_npz, write_json
from .evaluate import make_pool, member_job, pmap
from .genome import (NetGenome, check_network_count, load_genome, mutate, save_genome,
                     unit_swap)
from .mlp import Arch, MLPPolicy, init_weights, interface_fingerprint
from .novelty import BD_NAMES, descriptor, hof_cell, mean_raw, novelty, ranks

# seed purposes (N)
P_INIT, P_CLONE_INIT, P_MUT, P_ARCHIVE = 1, 2, 3, 4
TEST_SALT = 0x7E57


@dataclass
class NeuroConfig:
    population: int = 50
    presentations: int = 24
    benchmark: int = 64
    test_size: int = 256
    elites: int = 2
    truncation: int = 10
    novelty_w: float = 0.5
    novelty_k: int = 10
    archive_add: int = 2
    archive_cap: int = 1000
    stagnation_gens: int = 25
    boost_w: float = 1.0
    boost_gens: int = 10
    boost_sigma_mult: float = 2.0
    sigma_init: float = 0.02
    tau: float = 0.2
    sigma_min: float = 0.002
    sigma_max: float = 0.2
    top_every: int = 5
    top_n: int = 3
    final_top: int = 5
    init: str = "mixed"            # mixed | random | clone
    n_clones: int = 10
    clone_sigma: float = 0.02
    unit_swap: bool = False
    arch: dict = field(default_factory=lambda: Arch().to_dict())
    n_networks: int = 1
    jet_network: Tuple[int, ...] = (0, 0, 0, 0)    # jet slot -> network index
    master_seed: int = 2026
    max_time_s: float = 360.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["jet_network"] = list(self.jet_network)
        return d

    @staticmethod
    def from_dict(d: dict) -> "NeuroConfig":
        d = dict(d)
        d["jet_network"] = tuple(d.get("jet_network", (0, 0, 0, 0)))
        return NeuroConfig(**d)

    def validate(self) -> None:
        check_network_count(self.n_networks)
        if any(j != 0 for j in self.jet_network) or len(self.jet_network) != 4:
            raise ValueError("jet_network must map all 4 jets to network 0 (one shared network)")
        if self.elites >= self.population or self.truncation > self.population:
            raise ValueError("elites / truncation larger than the population")
        if self.init in ("mixed", "clone") and self.init == "mixed" \
                and self.n_clones > self.population:
            raise ValueError("more clones than members")


def config_hash(cfg: NeuroConfig, clone_sha: Optional[str]) -> str:
    blob = {"cfg": cfg.to_dict(), "clone": clone_sha, "interface": interface_fingerprint()}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def _rng(seed: int, *key) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(k) for k in key]]))


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class MemberEval:
    fitness: float
    per_fight: List[float]
    bd: np.ndarray
    raw_mean: dict
    summaries: List[dict]


# ------------------------------------------------------------------ tasks --
class SimTask:
    """Evaluates networks on spec 4 presentations through the spec 5 interface."""

    def __init__(self, cfg: NeuroConfig) -> None:
        self.cfg = cfg

    def eval_set(self, gen: int) -> List[dict]:
        from stealth_tactics.scenarios.presentation import build_eval_set
        return [p.to_dict() for p in build_eval_set(self.cfg.master_seed, gen,
                                                    self.cfg.presentations)]

    def bench_set(self) -> List[dict]:
        from stealth_tactics.scenarios.presentation import build_benchmark_set
        return [p.to_dict() for p in build_benchmark_set(self.cfg.master_seed, self.cfg.benchmark)]

    def test_set(self) -> List[dict]:
        from stealth_tactics.scenarios.presentation import _set, DEFAULT_PRESENTATION_CONFIG
        return [p.to_dict() for p in _set(self.cfg.master_seed, TEST_SALT, 0, self.cfg.test_size,
                                          DEFAULT_PRESENTATION_CONFIG)]

    def evaluate(self, ga: "NeuroGA", genomes: List[NetGenome], pres: List[dict],
                 tag: str) -> List[MemberEval]:
        arch = ga.arch
        path = ga.work / f"pop_{tag}.npz"
        save_npz(path, {"weights": np.stack([g.weights for g in genomes])})
        write_json(path.with_suffix(".json"), {"arch": arch.to_dict()})
        jobs = [(str(path), i, p, self.cfg.max_time_s, False)
                for i in range(len(genomes)) for p in pres]
        out = pmap(ga.pool, member_job, jobs)
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        n = len(pres)
        res = []
        for i in range(len(genomes)):
            ch = out[i * n:(i + 1) * n]
            per = [float(f) for f, _, _, _ in ch]
            raws = [r for _, r, _, _ in ch]
            res.append(MemberEval(float(np.mean(per)), per, descriptor(raws), mean_raw(raws),
                                  [s for _, _, s, _ in ch]))
        return res


# --------------------------------------------------------------------- GA --
class NeuroGA:
    def __init__(self, cfg: NeuroConfig, run_dir, workers: int = 8, task=None,
                 clone_path: Optional[str] = None, log=print) -> None:
        cfg.validate()
        self.cfg = cfg
        self.arch = Arch.from_dict(cfg.arch)
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.work = self.run_dir / "work"
        for d in (self.run_dir, self.ckpt_dir, self.work):
            d.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.task = task or SimTask(cfg)
        self.clone_path = clone_path
        self.log = log
        self.pool = None
        self.clone_sha = None
        if clone_path:
            self.clone_sha = hashlib.sha256(
                Path(clone_path).with_suffix(".npz").read_bytes()).hexdigest()[:16]
        self.hash = config_hash(cfg, self.clone_sha)
        # state
        self.gen = 0
        self.population: List[NetGenome] = []
        self.archive = np.zeros((0, len(BD_NAMES)))
        self.hof: Dict[str, dict] = {}
        self.champion: Optional[dict] = None
        self.bench_top: List[dict] = []
        self.history: List[dict] = []
        self.stagnation = 0
        self.boost_left = 0
        self.next_id = 0
        self.events: List[dict] = []

    # ---------------------------------------------------------- init (K) --
    def init_population(self) -> None:
        c = self.cfg
        pop: List[NetGenome] = []
        n_clone = {"random": 0, "mixed": c.n_clones, "clone": c.population}[c.init]
        clone_w = None
        if n_clone:
            if not self.clone_path:
                raise ValueError(f"init={c.init!r} needs a clone (clone-hand output)")
            g, arch, _ = load_genome(self.clone_path)
            if arch != self.arch:
                raise ValueError(f"clone arch {arch} != run arch {self.arch}")
            clone_w = g.weights
        for i in range(c.population):
            if i < n_clone:
                w = clone_w + c.clone_sigma * _rng(c.master_seed, 0, P_CLONE_INIT, i) \
                    .standard_normal(clone_w.shape)
                origin = "clone"
            else:
                w = init_weights(self.arch, _rng(c.master_seed, 0, P_INIT, i))
                origin = "random"
            pop.append(NetGenome([w], c.sigma_init,
                                 {"id": i, "parent": None, "born": 0, "origin": origin}))
        self.population = pop
        self.next_id = c.population

    # ---------------------------------------------------------- one gen --
    def step(self) -> dict:
        c, g = self.cfg, self.gen
        t0 = time.perf_counter()
        pres = self.task.eval_set(g)
        evals = self.task.evaluate(self, self.population, pres, f"g{g:04d}")
        f = np.array([e.fitness for e in evals])
        bds = np.stack([e.bd for e in evals])
        nov = novelty(bds, self.archive, c.novelty_k)
        w = c.boost_w if self.boost_left > 0 else c.novelty_w
        comb = ranks(f) + w * ranks(nov)
        parent_idx = [int(i) for i in np.argsort(comb, kind="stable")[:c.truncation]]
        by_fit = [int(i) for i in np.argsort(-f, kind="stable")]
        elite_idx = by_fit[:c.elites]
        t_eval = time.perf_counter() - t0

        # benchmark (M)
        cand = by_fit[:c.top_n] if (g + 1) % c.top_every == 0 else by_fit[:1]
        bench = self.task.bench_set()
        bevals = self.task.evaluate(self, [self.population[i] for i in cand], bench, f"b{g:04d}")
        improved = False
        best_bench = None
        for i, be in zip(cand, bevals):
            gnm = self.population[i]
            entry = {"genome": gnm.copy(), "fitness": float(f[i]), "per_fight": evals[i].per_fight,
                     "presentations": pres, "benchmark_fitness": be.fitness,
                     "benchmark_per_fight": be.per_fight, "gen": g,
                     "bench_raw": be.raw_mean, "bench_summaries": be.summaries}
            best_bench = be.fitness if best_bench is None else max(best_bench, be.fitness)
            if self.champion is None or be.fitness > self.champion["benchmark_fitness"]:
                self.champion = entry
                improved = True
            cell = hof_cell(be.raw_mean)
            if cell is not None and (cell not in self.hof
                                     or be.fitness > self.hof[cell]["benchmark_fitness"]):
                self.hof[cell] = entry
            self._bench_top_add(entry)
        t_bench = time.perf_counter() - t0 - t_eval

        # stagnation (L)
        self.stagnation = 0 if improved else self.stagnation + 1
        if self.boost_left > 0:
            self.boost_left -= 1
        boosted = False
        if self.stagnation >= c.stagnation_gens:
            boosted = True
            self.boost_left = c.boost_gens
            self.stagnation = 0
            for gnm in self.population:
                gnm.sigma = float(min(c.sigma_max, gnm.sigma * c.boost_sigma_mult))
            self.events.append({"gen": g, "type": "stagnation",
                                "text": f"no benchmark improvement for {c.stagnation_gens} "
                                        f"generations: novelty w -> {c.boost_w} for "
                                        f"{c.boost_gens} gens, sigma x{c.boost_sigma_mult}"})

        # archive (L)
        ra = _rng(c.master_seed, g, P_ARCHIVE)
        add = ra.choice(len(self.population), size=min(c.archive_add, len(self.population)),
                        replace=False)
        self.archive = np.vstack([self.archive, bds[np.sort(add)]])[-c.archive_cap:]

        best = by_fit[0]
        row = {"generation": g, "best_fitness": float(f[best]), "mean_fitness": float(f.mean()),
               "median_fitness": float(np.median(f)), "best_bench": best_bench,
               "champion_bench": self.champion["benchmark_fitness"],
               "champion_gen": self.champion["gen"],
               "best_origin": self.population[best].lineage.get("origin"),
               "best_id": self.population[best].lineage.get("id"),
               "best_kills": float(np.mean([s.get("kills", 0) for s in evals[best].summaries])),
               "best_losses": float(np.mean([s.get("losses", 0) for s in evals[best].summaries])),
               "mean_sigma": float(np.mean([p.sigma for p in self.population])),
               "n_clone_origin": sum(p.lineage.get("origin") == "clone" for p in self.population),
               "novelty_w": w, "mean_novelty": float(nov.mean()), "stagnation": self.stagnation,
               "boosted": boosted, "hof_cells": sorted(self.hof),
               "parents": parent_idx, "elites": elite_idx,
               "presentation_seeds": [p["presentation_seed"] for p in pres]}
        self.history.append(row)

        # breed (H, I, J)
        parents = [self.population[i] for i in parent_idx]
        nxt = [self.population[i].copy() for i in elite_idx]
        k = 0
        while len(nxt) < c.population:
            r = _rng(c.master_seed, g, P_MUT, k)
            par = parents[int(r.integers(len(parents)))]
            if c.unit_swap:
                other = parents[int(r.integers(len(parents)))]
                par = unit_swap(par, other, r, self.arch)
            nxt.append(mutate(par, r, self.next_id, g + 1, c.tau, c.sigma_min, c.sigma_max))
            self.next_id += 1
            k += 1
        self.population = nxt
        self.gen = g + 1
        self.save_checkpoint()
        timing = {"generation": g, "eval_s": t_eval, "bench_s": t_bench,
                  "total_s": time.perf_counter() - t0,
                  "engagements": len(evals) * len(pres) + len(cand) * len(bench)}
        with open(self.run_dir / "timing.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(timing) + "\n")
        return row

    def _bench_top_add(self, entry: dict) -> None:
        ids = {e["genome"].lineage.get("id") for e in self.bench_top}
        if entry["genome"].lineage.get("id") in ids:
            for e in self.bench_top:
                if e["genome"].lineage.get("id") == entry["genome"].lineage.get("id") and \
                        entry["benchmark_fitness"] > e["benchmark_fitness"]:
                    self.bench_top.remove(e)
                    self.bench_top.append(entry)
                    break
        else:
            self.bench_top.append(entry)
        self.bench_top.sort(key=lambda e: -e["benchmark_fitness"])
        del self.bench_top[self.cfg.final_top:]

    # ------------------------------------------------------ checkpoint (O) --
    @staticmethod
    def _entry_meta(e: Optional[dict]) -> Optional[dict]:
        if e is None:
            return None
        m = {k: v for k, v in e.items() if k != "genome"}
        m["genome_meta"] = e["genome"].meta()
        return m

    def save_checkpoint(self) -> Path:
        g = self.gen
        arrays = {"pop_weights": np.stack([p.weights for p in self.population]),
                  "pop_sigma": np.array([p.sigma for p in self.population]),
                  "archive": self.archive}
        if self.champion is not None:
            arrays["champion"] = self.champion["genome"].weights
        cells = sorted(self.hof)
        for cell in cells:
            arrays[f"hof_{cell}"] = self.hof[cell]["genome"].weights
        for i, e in enumerate(self.bench_top):
            arrays[f"top_{i}"] = e["genome"].weights
        meta = {"format": 1, "gen_next": g, "config": self.cfg.to_dict(), "config_hash": self.hash,
                "interface": interface_fingerprint(), "n_networks": self.cfg.n_networks,
                "jet_network": list(self.cfg.jet_network), "git_commit": _git_commit(),
                "clone_sha": self.clone_sha,
                "lineages": [p.lineage for p in self.population], "next_id": self.next_id,
                "stagnation": self.stagnation, "boost_left": self.boost_left,
                "history": self.history, "events": self.events,
                "champion": self._entry_meta(self.champion),
                "hof": {c: self._entry_meta(self.hof[c]) for c in cells},
                "bench_top": [self._entry_meta(e) for e in self.bench_top]}
        base = self.ckpt_dir / f"ckpt_g{g:04d}"
        save_npz(base.with_suffix(".npz"), arrays)
        write_json(base.with_suffix(".json"), meta)
        olds = sorted(self.ckpt_dir.glob("ckpt_g*.json"))[:-3]
        for o in olds:
            o.unlink(missing_ok=True)
            o.with_suffix(".npz").unlink(missing_ok=True)
        return base

    def load_checkpoint(self, base: Optional[Path] = None) -> None:
        if base is None:
            js = sorted(self.ckpt_dir.glob("ckpt_g*.json"))
            if not js:
                raise FileNotFoundError(f"no checkpoint in {self.ckpt_dir}")
            base = js[-1].with_suffix("")
        meta = json.loads(Path(base).with_suffix(".json").read_text(encoding="utf-8"))
        check_network_count(meta.get("n_networks", 1))
        if meta["interface"] != interface_fingerprint():
            raise ValueError("checkpoint interface fingerprint differs from this build "
                             "(spec 5 interface changed); refusing to resume")
        if meta["config_hash"] != self.hash:
            raise ValueError(f"checkpoint config hash {meta['config_hash']} != this run's "
                             f"{self.hash} (different config or clone); refusing to resume")
        a = load_npz(Path(base).with_suffix(".npz"))

        def ent(m, w):
            if m is None:
                return None
            e = {k: v for k, v in m.items() if k != "genome_meta"}
            gm = m["genome_meta"]
            e["genome"] = NetGenome([w], gm["sigma"], gm["lineage"])
            return e
        self.gen = meta["gen_next"]
        self.population = [NetGenome([a["pop_weights"][i]], float(a["pop_sigma"][i]), lin)
                           for i, lin in enumerate(meta["lineages"])]
        self.archive = a["archive"]
        self.champion = ent(meta["champion"], a.get("champion"))
        self.hof = {c: ent(m, a[f"hof_{c}"]) for c, m in meta["hof"].items()}
        self.bench_top = [ent(m, a[f"top_{i}"]) for i, m in enumerate(meta["bench_top"])]
        self.history, self.events = meta["history"], meta["events"]
        self.stagnation, self.boost_left = meta["stagnation"], meta["boost_left"]
        self.next_id = meta["next_id"]

    # ----------------------------------------------------------------- run --
    def run(self, generations: int, resume: bool = False, finalize: bool = True) -> dict:
        if resume and sorted(self.ckpt_dir.glob("ckpt_g*.json")):
            self.load_checkpoint()
            self.log(f"resumed at generation {self.gen}")
        elif not self.population:
            self.init_population()
        self.pool = make_pool(self.workers)
        try:
            while self.gen < generations:
                row = self.step()
                self.log(f"gen {row['generation']:3d}  best {row['best_fitness']:8.2f}  mean "
                         f"{row['mean_fitness']:8.2f}  bench {row['best_bench']:8.2f}  champion "
                         f"{row['champion_bench']:8.2f} (g{row['champion_gen']})  best origin "
                         f"{row['best_origin']}  sigma {row['mean_sigma']:.4f}  hof "
                         f"{len(row['hof_cells'])}")
            report = self.finalize() if finalize else {}
        finally:
            if self.pool is not None:
                self.pool.shutdown()
                self.pool = None
        return report

    # ------------------------------------------------------ finalize (M, P) --
    def finalize(self, export_acmi: bool = True) -> dict:
        from .evaluate import run_policy_fight
        from stealth_tactics.presentation_runner import export_presentation_acmi
        ch = self.champion
        again = self.task.evaluate(self, [ch["genome"]], ch["presentations"], "check")[0]
        if again.per_fight != ch["per_fight"]:
            raise RuntimeError("champion re-run on its stored presentations differs")
        b2 = self.task.evaluate(self, [ch["genome"]], self.task.bench_set(), "check_b")[0]
        if b2.fitness != ch["benchmark_fitness"]:
            raise RuntimeError("champion benchmark re-run differs")
        # held-out test (M): hall of fame + top by benchmark, unique ids
        cands, seen = [], set()
        for e in list(self.bench_top) + [self.hof[c] for c in sorted(self.hof)]:
            i = e["genome"].lineage.get("id")
            if i not in seen:
                seen.add(i)
                cands.append(e)
        test = self.task.test_set()
        tevals = self.task.evaluate(self, [e["genome"] for e in cands], test, "test") \
            if test else []
        test_rows = []
        for e, te in zip(cands, tevals):
            e["test_fitness"] = te.fitness
            e["test_summary"] = _summ(te.summaries)
            test_rows.append({"id": e["genome"].lineage.get("id"),
                              "origin": e["genome"].lineage.get("origin"), "gen": e["gen"],
                              "benchmark_fitness": e["benchmark_fitness"],
                              "test_fitness": te.fitness, **e["test_summary"],
                              "hof_cells": [c for c in self.hof
                                            if self.hof[c]["genome"].lineage.get("id")
                                            == e["genome"].lineage.get("id")]})
        # store champion + hall of fame (P)
        self._store(ch, self.run_dir, "champion", export_acmi)
        for cell in sorted(self.hof):
            self._store(self.hof[cell], self.run_dir / "hall_of_fame" / cell, "champion", False)
        report = {"champion": {"id": ch["genome"].lineage.get("id"),
                               "origin": ch["genome"].lineage.get("origin"), "gen": ch["gen"],
                               "fitness": ch["fitness"],
                               "benchmark_fitness": ch["benchmark_fitness"],
                               "benchmark_summary": _summ(ch["bench_summaries"]),
                               "test_fitness": ch.get("test_fitness")},
                  "test": test_rows, "history": self.history, "events": self.events,
                  "hof": {c: {"id": self.hof[c]["genome"].lineage.get("id"),
                              "benchmark_fitness": self.hof[c]["benchmark_fitness"],
                              "bench_raw": self.hof[c]["bench_raw"]} for c in sorted(self.hof)}}
        write_json(self.run_dir / "report.json", report)
        return report

    def _store(self, e: dict, out: Path, name: str, export_acmi: bool) -> None:
        out.mkdir(parents=True, exist_ok=True)
        per = e["per_fight"]
        best_k, worst_k = int(np.argmax(per)), int(np.argmin(per))
        save_genome(out / f"{name}_weights", e["genome"], self.arch)
        doc = {"kind": "network", "arch": self.arch.to_dict(),
               "interface": interface_fingerprint(), "n_networks": self.cfg.n_networks,
               "jet_network": list(self.cfg.jet_network), "lineage": e["genome"].lineage,
               "sigma": e["genome"].sigma, "gen": e["gen"], "fitness": e["fitness"],
               "per_fight": per, "presentations": e["presentations"],
               "benchmark_fitness": e["benchmark_fitness"],
               "benchmark_per_fight": e["benchmark_per_fight"],
               "test_fitness": e.get("test_fitness"), "bench_raw": e["bench_raw"],
               "max_time_s": self.cfg.max_time_s, "config_hash": self.hash,
               "best_index": best_k, "worst_index": worst_k,
               "weights": f"{name}_weights.npz"}
        write_json(out / f"{name}.json", doc)
        if export_acmi:
            export_champion_acmis(out, doc, e["genome"], self.arch)


def _summ(summaries: List[dict]) -> dict:
    n = len(summaries)
    shots = sum(s["blue_shots"] for s in summaries)
    return {"n": n, "kills": sum(s["kills"] for s in summaries) / n,
            "losses": sum(s["losses"] for s in summaries) / n, "shots": shots / n,
            "hit_rate": (sum(s["blue_hits"] for s in summaries) / shots) if shots else 0.0,
            "red_shots": sum(s["red_shots"] for s in summaries) / n}


def export_champion_acmis(out: Path, doc: dict, genome: NetGenome, arch: Arch,
                          suffix: str = "") -> List[Tuple[Path, float]]:
    from stealth_tactics.presentation_runner import export_presentation_acmi
    from .evaluate import run_policy_fight
    res = []
    for tag, k in (("best", doc["best_index"]), ("worst", doc["worst_index"])):
        f, _, _, rec = run_policy_fight(MLPPolicy(genome.weights, arch), doc["presentations"][k],
                                        doc["max_time_s"], record=True)
        p = out / f"champion_{tag}{suffix}.txt.acmi"
        export_presentation_acmi(rec, p, title=f"Spec 6 network champion ({tag} fight, "
                                               f"stored presentation {k}, fitness {f:.1f})")
        res.append((p, f))
    return res


def replay_network_champion(run_dir, workers: int = 8) -> Tuple[bool, str]:
    """P: re-run a network champion on its stored presentations (fresh
    process), compare per-fight fitness exactly, re-export the best / worst
    ACMIs and compare bytes with the stored ones."""
    from .evaluate import policy_job
    run_dir = Path(run_dir)
    doc = json.loads((run_dir / "champion.json").read_text(encoding="utf-8"))
    genome, arch, _ = load_genome(run_dir / "champion_weights")
    pol = MLPPolicy(genome.weights, arch)
    jobs = [(pol, p, doc["max_time_s"], False) for p in doc["presentations"]]
    pool = make_pool(workers)
    try:
        out = pmap(pool, policy_job, jobs)
    finally:
        if pool is not None:
            pool.shutdown()
    per = [float(f) for f, _, _, _ in out]
    ok = per == doc["per_fight"]
    lines = [f"Re-run on {len(per)} stored presentations: mean {float(np.mean(per))!r} vs stored "
             f"{doc['fitness']!r} -> {'IDENTICAL' if ok else 'MISMATCH'}"]
    same_all = True
    for (p, f), tag in zip(export_champion_acmis(run_dir, doc, genome, arch, "_replayed"),
                           ("best", "worst")):
        orig = run_dir / f"champion_{tag}.txt.acmi"
        same = orig.exists() and p.read_bytes() == orig.read_bytes()
        same_all &= same
        lines.append(f"{tag} fight: fitness {f!r}; ACMI re-export "
                     f"{'byte-identical' if same else 'DIFFERS'} to {orig.name}")
    return ok and same_all, "\n".join(lines)
