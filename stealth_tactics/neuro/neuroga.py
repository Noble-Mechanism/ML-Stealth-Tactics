"""Spec 6 F-O: the neuroevolution loop.

Mutation-only GA with elitism (F), population 50 x 24 presentations (G),
2 re-scored elites + truncation parents from the top 10 by fitness rank +
0.5 x novelty rank (H, L), self-adaptive Gaussian mutation (I), no crossover
(J; unit swap behind ``unit_swap``), mixed clone / random start (K),
stagnation boost and 3 x 3 hall of fame (L), champion = best benchmark score
with top-3 benchmarking every 5th generation and a held-out test at the end
(M), seeds per (generation, purpose, index) (N), a byte-deterministic
checkpoint after every generation (O), champion storage (P).

Spec 6b (GA stability) changes the defaults: 36 presentations, a stagnation
trigger on the 10-generation moving average of the best eval score (60
generations without a new high) that answers with a large immigration instead
of a global sigma x2 / novelty boost, periodic immigration (8 of 50 every 25
generations: random nets + mutated hall-of-fame cells outside the champion's
lineage), and a fair champion pick (top 3 benchmarked every generation; a
challenger must beat the champion on the same paired benchmark fights by
more than k standard errors of the paired difference). Per-generation history
is written to ``history.csv`` / ``history.json``."""

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

from stealth_tactics.fitness import aggregate, merge_outcome_counts

from .checkpoint import load_npz, save_npz, write_json
from .evaluate import make_pool, member_job, pmap
from .genome import (NetGenome, check_network_count, load_genome, mutate, save_genome,
                     unit_swap)
from .mlp import Arch, MLPPolicy, init_weights, interface_fingerprint
from .novelty import BD_NAMES, descriptor, hof_cell, mean_raw, novelty, ranks

# seed purposes (N)
P_INIT, P_CLONE_INIT, P_MUT, P_ARCHIVE, P_IMMIG = 1, 2, 3, 4, 5
STAGNATION_METRICS = ("best_ma", "champion")
TEST_SALT = 0x7E57


@dataclass
class NeuroConfig:
    population: int = 50
    presentations: int = 36           # spec 6b: 24 -> 36
    benchmark: int = 64
    test_size: int = 256
    elites: int = 2
    truncation: int = 10
    novelty_w: float = 0.5
    novelty_k: int = 10
    archive_add: int = 2
    archive_cap: int = 1000
    # stagnation trigger (spec 6b): no new high of the moving average of the
    # best eval score for ``stagnation_gens`` generations (0 disables);
    # "champion" = the spec 6 rule (no champion improvement)
    stagnation_gens: int = 60
    stagnation_window: int = 10
    stagnation_min_delta: float = 0.0
    stagnation_metric: str = "best_ma"
    stagnation_immigrants: int = 16   # immigrants when the trigger fires
    # legacy global boost on the trigger (spec 6: boost_gens 10, sigma x2);
    # spec 6b default: off (no global sigma / novelty-weight change)
    boost_w: float = 1.0
    boost_gens: int = 0
    boost_sigma_mult: float = 1.0
    # periodic immigration (spec 6b): every N generations (0 disables) the
    # next generation gets ``immigrants`` newcomers in place of offspring
    immigrate_every: int = 25
    immigrants: int = 8
    immigrant_random_frac: float = 0.5   # rest: mutated non-champion-lineage HoF cells
    immigrant_protect_gens: int = 10     # immigrant lineages are protected this long
    immigrant_parent_slots: int = 1      # truncation-parent slots reserved for them
    sigma_init: float = 0.02
    tau: float = 0.2
    sigma_min: float = 0.002
    sigma_max: float = 0.2
    top_every: int = 1             # spec 6b: top-n benchmarked every generation
    top_n: int = 3
    champion_margin_k: float = 1.0 # challenger needs mean paired diff > k * SE; <= 0: spec 6 rule
    bench_cache: bool = True       # reuse stored benchmark scores of unchanged genomes
    final_top: int = 5
    init: str = "random"           # random (default, Rusty 2026-09-26) | mixed | clone
    n_clones: int = 10
    clone_sigma: float = 0.02
    unit_swap: bool = False
    arch: dict = field(default_factory=lambda: Arch().to_dict())
    n_networks: int = 1
    jet_network: Tuple[int, ...] = (0, 0, 0, 0)    # jet slot -> network index
    master_seed: int = 2026
    max_time_s: float = 360.0
    n_red: int = 6                 # Red flight size (spec 4 menus: 6 or 8)
    blue_start: str = "wall"       # wall (30 NM line abreast) | diamond (spec 4)
    # spec 7 fitness weights (scenarios/fitness.yaml unless given); part of the
    # config hash, so resume refuses changed weights
    fitness: dict = field(default_factory=lambda: _default_fitness())

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
        from stealth_tactics.scenarios.presentation import BLUE_STARTS
        if self.blue_start not in BLUE_STARTS:
            raise ValueError(f"blue_start must be one of {sorted(BLUE_STARTS)}")
        if self.init not in ("random", "mixed", "clone"):
            raise ValueError("init must be random, mixed or clone")
        if self.init == "mixed" \
                and self.n_clones > self.population:
            raise ValueError("more clones than members")
        if self.stagnation_metric not in STAGNATION_METRICS:
            raise ValueError(f"stagnation_metric must be one of {STAGNATION_METRICS}")
        if self.stagnation_window < 1 or self.top_every < 1 or self.top_n < 1:
            raise ValueError("stagnation_window, top_every and top_n must be >= 1")
        if min(self.stagnation_gens, self.immigrate_every, self.immigrants,
               self.stagnation_immigrants, self.immigrant_protect_gens,
               self.immigrant_parent_slots) < 0:
            raise ValueError("stagnation / immigration counts must be >= 0")
        if not 0.0 <= self.immigrant_random_frac <= 1.0:
            raise ValueError("immigrant_random_frac must be in [0, 1]")
        if self.immigrant_parent_slots > self.truncation:
            raise ValueError("immigrant_parent_slots larger than truncation")

    def max_immigrants(self) -> int:
        """Immigrants never take elite slots and leave at least one offspring."""
        return max(0, self.population - self.elites - 1)


def _default_fitness() -> dict:
    from stealth_tactics.fitness import load_weights
    return load_weights()


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
        from stealth_tactics.scenarios.presentation import presentation_config
        self.cfg = cfg
        self.pcfg = presentation_config(cfg.blue_start, cfg.n_red, cfg.max_time_s)

    def eval_set(self, gen: int) -> List[dict]:
        from stealth_tactics.scenarios.presentation import build_eval_set
        return [p.to_dict() for p in build_eval_set(self.cfg.master_seed, gen,
                                                    self.cfg.presentations, self.pcfg)]

    def bench_set(self) -> List[dict]:
        from stealth_tactics.scenarios.presentation import build_benchmark_set
        return [p.to_dict() for p in build_benchmark_set(self.cfg.master_seed, self.cfg.benchmark,
                                                         self.pcfg)]

    def test_set(self) -> List[dict]:
        from stealth_tactics.scenarios.presentation import _set
        return [p.to_dict() for p in _set(self.cfg.master_seed, TEST_SALT, 0, self.cfg.test_size,
                                          self.pcfg)]

    def evaluate(self, ga: "NeuroGA", genomes: List[NetGenome], pres: List[dict],
                 tag: str) -> List[MemberEval]:
        arch = ga.arch
        path = ga.work / f"pop_{tag}.npz"
        save_npz(path, {"weights": np.stack([g.weights for g in genomes])})
        write_json(path.with_suffix(".json"), {"arch": arch.to_dict()})
        jobs = [(str(path), i, p, self.cfg.max_time_s, False, self.cfg.fitness)
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
            res.append(MemberEval(aggregate(per, self.cfg.fitness), per, descriptor(raws), mean_raw(raws),
                                  [s for _, _, s, _ in ch]))
        return res


# ---------------------------------------------------------------- history --
HISTORY_COLUMNS = ("generation", "best_fitness", "mean_fitness", "median_fitness", "best_ma",
                   "best_bench", "champion_bench", "champion_gen", "champion_id", "best_id",
                   "best_origin", "best_kills", "best_losses", "mean_sigma", "min_sigma",
                   "max_sigma", "mean_novelty", "novelty_w", "stagnation", "stagnation_trigger",
                   "boosted", "immigrants", "immigrant_reason", "immigrants_random",
                   "immigrants_hof", "n_immigrant_lineage", "n_clone_origin", "hof_cells",
                   "challenger_id", "challenger_bench", "paired_diff", "paired_se",
                   "champion_margin", "champion_accepted", "champion_reason")


def history_csv_row(r: dict) -> dict:
    d = {k: r.get(k) for k in HISTORY_COLUMNS if k in r}
    d["hof_cells"] = len(r.get("hof_cells") or [])
    src = r.get("immigrant_sources") or {}
    d["immigrants_random"] = src.get("random", 0)
    d["immigrants_hof"] = src.get("hof", 0)
    dec = r.get("champion_decision") or {}
    d.update(challenger_id=dec.get("challenger_id"), challenger_bench=dec.get("challenger_bench"),
             paired_diff=dec.get("mean_diff"), paired_se=dec.get("se"),
             champion_margin=dec.get("margin"), champion_accepted=dec.get("accepted"),
             champion_reason=dec.get("reason"))
    return d


def write_history_files(history: List[dict], out_dir: Path) -> None:
    import csv
    import io
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in r.items() if k not in ("parents", "elites", "presentation_seeds")}
            for r in history]
    write_json(out_dir / "history.json", rows)
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(HISTORY_COLUMNS), extrasaction="ignore",
                        lineterminator="\n")
    wr.writeheader()
    for r in history:
        wr.writerow({k: ("" if v is None else v) for k, v in history_csv_row(r).items()})
    tmp = out_dir / "history.csv.tmp"
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(out_dir / "history.csv")


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
        self.best_ma_max: Optional[float] = None
        self._bench_cache: Dict[int, tuple] = {}   # id -> (weights, MemberEval); timing only

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
    def _protected(self, gnm: NetGenome, g: int) -> bool:
        im = gnm.lineage.get("immig")
        return im is not None and g - int(im) < self.cfg.immigrant_protect_gens

    def _select_parents(self, comb: np.ndarray, g: int) -> List[int]:
        """Truncation by combined rank; spec 6b reserves up to
        ``immigrant_parent_slots`` of the slots for the best protected
        (recent-immigrant) members so newcomers get a chance to breed."""
        c = self.cfg
        order = [int(i) for i in np.argsort(comb, kind="stable")]
        n_res = c.immigrant_parent_slots if c.immigrant_protect_gens > 0 else 0
        prot = [i for i in order if self._protected(self.population[i], g)][:n_res]
        rest = [i for i in order if i not in prot][:c.truncation - len(prot)]
        chosen = set(prot) | set(rest)
        return [i for i in order if i in chosen]

    def _bench_eval(self, genomes: List[NetGenome], bench: List[dict], tag: str):
        """Benchmark *genomes*; unchanged genomes (same id, same weights) reuse
        stored scores when ``bench_cache`` is on. The fights are deterministic,
        so the cache changes only the run time. Returns (evals, n_run)."""
        c = self.cfg
        out: List[Optional[MemberEval]] = [None] * len(genomes)
        todo = []
        for j, gnm in enumerate(genomes):
            gid = gnm.lineage.get("id")
            hit = self._bench_cache.get(gid) if c.bench_cache else None
            if hit is not None and np.array_equal(hit[0], gnm.weights):
                out[j] = hit[1]
            else:
                todo.append(j)
        if todo:
            ev = self.task.evaluate(self, [genomes[j] for j in todo], bench, tag)
            for j, e in zip(todo, ev):
                out[j] = e
                if c.bench_cache:
                    self._bench_cache[genomes[j].lineage.get("id")] = (genomes[j].weights.copy(), e)
        return out, len(todo)

    def _champion_decision(self, entries: List[dict]) -> dict:
        """Spec 6b fair champion pick: the best-benchmarked candidate that is
        not the champion challenges it on the same (paired) benchmark fights.
        It wins if mean(d) > k * SE(d), d = challenger - champion per fight,
        SE = std(d, ddof=1) / sqrt(n), and its benchmark score is higher (so
        the champion's score never drops). k <= 0: spec 6 rule (higher
        benchmark score wins)."""
        k = self.cfg.champion_margin_k
        ch = self.champion
        ch_id = ch["genome"].lineage.get("id") if ch is not None else None
        chall = [e for e in entries if e["genome"].lineage.get("id") != ch_id]
        if not chall:
            return {"challenger_id": None, "accepted": False, "reason": "no challenger",
                    "champion_id": ch_id}
        e = max(chall, key=lambda x: x["benchmark_fitness"])   # first on ties (by eval rank)
        dec = {"challenger_id": e["genome"].lineage.get("id"),
               "challenger_bench": float(e["benchmark_fitness"]), "champion_id": ch_id,
               "champion_bench": None if ch is None else float(ch["benchmark_fitness"]),
               "mean_diff": None, "se": None, "margin": None}
        if ch is None:
            dec.update(accepted=True, reason="first champion")
            return dec
        higher = e["benchmark_fitness"] > ch["benchmark_fitness"]
        if k <= 0:
            dec.update(accepted=bool(higher), reason="higher benchmark" if higher else "lower benchmark")
            return dec
        d = np.asarray(e["benchmark_per_fight"], float) - np.asarray(ch["benchmark_per_fight"], float)
        n = len(d)
        se = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        md = float(d.mean())
        ok = bool(md > k * se and higher)
        dec.update(mean_diff=md, se=se, margin=float(k * se), accepted=ok,
                   reason="beats champion by > k*SE" if ok else
                   ("within k*SE of champion" if higher else "lower benchmark"))
        return dec

    def _stagnation_update(self, improved: bool, best_now: float) -> Tuple[float, bool]:
        """Returns (moving average of the best eval score over the last
        ``stagnation_window`` generations including this one, trigger fired)."""
        c = self.cfg
        past = [r["best_fitness"] for r in self.history]
        vals = past[max(0, len(past) - (c.stagnation_window - 1)):] + [float(best_now)]
        ma = float(np.mean(vals))
        if c.stagnation_metric == "champion":
            self.stagnation = 0 if improved else self.stagnation + 1
        else:
            if self.best_ma_max is None or ma > self.best_ma_max + c.stagnation_min_delta:
                self.best_ma_max = ma
                self.stagnation = 0
            else:
                self.stagnation += 1
        fired = c.stagnation_gens > 0 and self.stagnation >= c.stagnation_gens
        if fired:
            self.stagnation = 0           # the all-time MA high stays the reference
        return ma, fired

    def _make_immigrants(self, n: int, g: int) -> Tuple[List[NetGenome], dict]:
        """n newcomers for generation g+1: round(n * immigrant_random_frac)
        random nets (fresh lineage roots), the rest mutated descendants of
        hall-of-fame cells whose lineage root differs from the champion's
        (best benchmark first, cycling); random when no such cell exists."""
        c = self.cfg
        ch = self.champion
        ch_root = ch_id = None
        if ch is not None:
            ch_id = ch["genome"].lineage.get("id")
            ch_root = ch["genome"].lineage.get("root", ch_id)
        cells = [cell for cell in sorted(self.hof, key=lambda x: (-self.hof[x]["benchmark_fitness"], x))
                 if self.hof[cell]["genome"].lineage.get("id") != ch_id
                 and self.hof[cell]["genome"].lineage.get(
                     "root", self.hof[cell]["genome"].lineage.get("id")) != ch_root]
        n_rand = int(round(n * c.immigrant_random_frac))
        if not cells:
            n_rand = n
        out, src = [], {"random": 0, "hof": 0, "hof_cells": []}
        for i in range(n):
            r = _rng(c.master_seed, g, P_IMMIG, i)
            nid = self.next_id
            self.next_id += 1
            if i < n_rand:
                gnm = NetGenome([init_weights(self.arch, r)], c.sigma_init,
                                {"id": nid, "parent": None, "born": g + 1,
                                 "origin": "immigrant_random", "root": nid, "immig": g + 1})
                src["random"] += 1
            else:
                cell = cells[(i - n_rand) % len(cells)]
                gnm = mutate(self.hof[cell]["genome"], r, nid, g + 1, c.tau, c.sigma_min,
                             c.sigma_max)
                gnm.lineage.update(origin="immigrant_hof", immig=g + 1, hof_cell=cell)
                src["hof"] += 1
                src["hof_cells"].append(cell)
            out.append(gnm)
        return out, src

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
        parent_idx = self._select_parents(comb, g)
        by_fit = [int(i) for i in np.argsort(-f, kind="stable")]
        elite_idx = by_fit[:c.elites]
        t_eval = time.perf_counter() - t0

        # benchmark (M, spec 6b)
        cand = by_fit[:c.top_n] if (g + 1) % c.top_every == 0 else by_fit[:1]
        bench = self.task.bench_set()
        if self.champion is not None and c.bench_cache:
            chg = self.champion["genome"]
            self._bench_cache.setdefault(chg.lineage.get("id"), (chg.weights.copy(), MemberEval(
                self.champion["benchmark_fitness"], self.champion["benchmark_per_fight"],
                np.zeros(len(BD_NAMES)), self.champion["bench_raw"],
                self.champion["bench_summaries"])))
        bevals, n_run = self._bench_eval([self.population[i] for i in cand], bench, f"b{g:04d}")
        entries = []
        best_bench = None
        for i, be in zip(cand, bevals):
            gnm = self.population[i]
            entry = {"genome": gnm.copy(), "fitness": float(f[i]), "per_fight": evals[i].per_fight,
                     "presentations": pres, "benchmark_fitness": be.fitness,
                     "benchmark_per_fight": be.per_fight, "gen": g,
                     "bench_raw": be.raw_mean, "bench_summaries": be.summaries}
            entries.append(entry)
            best_bench = be.fitness if best_bench is None else max(best_bench, be.fitness)
            cell = hof_cell(be.raw_mean)
            if cell is not None and (cell not in self.hof
                                     or be.fitness > self.hof[cell]["benchmark_fitness"]):
                self.hof[cell] = entry
            self._bench_top_add(entry)
        dec = self._champion_decision(entries)
        improved = bool(dec["accepted"])
        if improved:
            self.champion = next(e for e in entries
                                 if e["genome"].lineage.get("id") == dec["challenger_id"])
        if dec["challenger_id"] is not None and dec["champion_id"] is not None and \
                (improved or (dec["challenger_bench"] > dec["champion_bench"])):
            self.events.append({"gen": g, "type": "champion",
                                "text": (f"champion {dec['champion_id']} -> {dec['challenger_id']}"
                                         if improved else
                                         f"champion {dec['champion_id']} kept against "
                                         f"{dec['challenger_id']}")
                                + f" (bench {dec['challenger_bench']:.2f} vs "
                                  f"{dec['champion_bench']:.2f}"
                                + (f", paired diff {dec['mean_diff']:.2f}, margin "
                                   f"{dec['margin']:.2f})" if dec["mean_diff"] is not None
                                   else ")")})
        keep = {p.lineage.get("id") for p in self.population}
        keep.add(self.champion["genome"].lineage.get("id"))
        self._bench_cache = {k_: v for k_, v in self._bench_cache.items() if k_ in keep}
        t_bench = time.perf_counter() - t0 - t_eval

        # stagnation (L, spec 6b) and immigration
        if self.boost_left > 0:
            self.boost_left -= 1
        best = by_fit[0]
        best_ma, fired = self._stagnation_update(improved, float(f[best]))
        boosted = False
        n_imm, reason = 0, ""
        if c.immigrate_every > 0 and (g + 1) % c.immigrate_every == 0 and c.immigrants > 0:
            n_imm, reason = c.immigrants, "periodic"
        if fired:
            if c.stagnation_immigrants >= n_imm and c.stagnation_immigrants > 0:
                n_imm, reason = c.stagnation_immigrants, "stagnation"
            acts = []
            if c.boost_gens > 0 or c.boost_sigma_mult != 1.0:
                boosted = True
                self.boost_left = c.boost_gens
                for gnm in self.population:
                    gnm.sigma = float(min(c.sigma_max, gnm.sigma * c.boost_sigma_mult))
                acts.append(f"novelty w -> {c.boost_w} for {c.boost_gens} gens, "
                            f"sigma x{c.boost_sigma_mult}")
            if n_imm:
                acts.append(f"{n_imm} immigrants")
            what = (f"no new high of the {c.stagnation_window}-generation moving average of "
                    f"the best score" if c.stagnation_metric == "best_ma"
                    else "no benchmark improvement")
            self.events.append({"gen": g, "type": "stagnation",
                                "text": f"{what} for {c.stagnation_gens} generations: "
                                        + ("; ".join(acts) or "no action")})
        n_imm = min(n_imm, c.max_immigrants())

        # archive (L)
        ra = _rng(c.master_seed, g, P_ARCHIVE)
        add = ra.choice(len(self.population), size=min(c.archive_add, len(self.population)),
                        replace=False)
        self.archive = np.vstack([self.archive, bds[np.sort(add)]])[-c.archive_cap:]

        sig = np.array([p.sigma for p in self.population])
        row = {"generation": g, "best_fitness": float(f[best]), "mean_fitness": float(f.mean()),
               "median_fitness": float(np.median(f)), "best_ma": best_ma, "best_bench": best_bench,
               "champion_bench": self.champion["benchmark_fitness"],
               "champion_gen": self.champion["gen"],
               "champion_id": self.champion["genome"].lineage.get("id"),
               "best_origin": self.population[best].lineage.get("origin"),
               "best_id": self.population[best].lineage.get("id"),
               "best_kills": float(np.mean([s.get("kills", 0) for s in evals[best].summaries])),
               "best_losses": float(np.mean([s.get("losses", 0) for s in evals[best].summaries])),
               "mean_sigma": float(sig.mean()), "min_sigma": float(sig.min()),
               "max_sigma": float(sig.max()),
               "n_clone_origin": sum(p.lineage.get("origin") == "clone" for p in self.population),
               "n_immigrant_lineage": sum("immig" in p.lineage for p in self.population),
               "novelty_w": w, "mean_novelty": float(nov.mean()), "stagnation": self.stagnation,
               "stagnation_trigger": bool(fired), "boosted": boosted,
               "immigrants": int(n_imm), "immigrant_reason": reason if n_imm else "",
               "champion_decision": dec, "hof_cells": sorted(self.hof),
               "parents": parent_idx, "elites": elite_idx,
               "presentation_seeds": [p["presentation_seed"] for p in pres]}

        # breed (H, I, J); immigrants take the last n_imm offspring slots
        parents = [self.population[i] for i in parent_idx]
        nxt = [self.population[i].copy() for i in elite_idx]
        k = 0
        while len(nxt) < c.population - n_imm:
            r = _rng(c.master_seed, g, P_MUT, k)
            par = parents[int(r.integers(len(parents)))]
            if c.unit_swap:
                other = parents[int(r.integers(len(parents)))]
                par = unit_swap(par, other, r, self.arch)
            nxt.append(mutate(par, r, self.next_id, g + 1, c.tau, c.sigma_min, c.sigma_max))
            self.next_id += 1
            k += 1
        if n_imm:
            imm, src = self._make_immigrants(n_imm, g)
            nxt.extend(imm)
            row["immigrant_sources"] = src
            self.events.append({"gen": g, "type": "immigration",
                                "text": f"{n_imm} immigrants ({reason}): {src['random']} random, "
                                        f"{src['hof']} from hall-of-fame cells "
                                        f"{sorted(set(src['hof_cells']))}"})
        else:
            row["immigrant_sources"] = None
        self.history.append(row)
        self.population = nxt
        self.gen = g + 1
        self.save_checkpoint()
        self.write_history()
        timing = {"generation": g, "eval_s": t_eval, "bench_s": t_bench,
                  "total_s": time.perf_counter() - t0, "bench_runs": n_run,
                  "bench_cached": len(cand) - n_run,
                  "engagements": len(evals) * len(pres) + n_run * len(bench)}
        with open(self.run_dir / "timing.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(timing) + "\n")
        return row

    def write_history(self, out_dir: Optional[Path] = None) -> None:
        """Spec 6b: per-generation history as ``history.json`` (full rows) and
        ``history.csv`` (scalar columns) so later analysis needs no plots."""
        write_history_files(self.history, Path(out_dir or self.run_dir))

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
                "best_ma_max": self.best_ma_max,
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
        self.best_ma_max = meta.get("best_ma_max")
        self._bench_cache = {}
        self.next_id = meta["next_id"]

    # ----------------------------------------------------------------- run --
    def run(self, generations: int, resume: bool = False, finalize: bool = True,
            on_generation=None) -> dict:
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
                if on_generation is not None:
                    on_generation(self, row)
            report = self.finalize() if finalize else {}
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
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
            self._store(self.hof[cell], self.run_dir / "hall_of_fame" / cell, "champion", export_acmi)
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
        outcomes = merge_outcome_counts(
            [s.get("missile_outcomes") or {} for s in e.get("bench_summaries") or []])
        doc = {"kind": "network", "arch": self.arch.to_dict(),
               "interface": interface_fingerprint(), "n_networks": self.cfg.n_networks,
               "jet_network": list(self.cfg.jet_network), "lineage": e["genome"].lineage,
               "sigma": e["genome"].sigma, "gen": e["gen"], "fitness": e["fitness"],
               "per_fight": per, "presentations": e["presentations"],
               "benchmark_fitness": e["benchmark_fitness"],
               "benchmark_per_fight": e["benchmark_per_fight"],
               "test_fitness": e.get("test_fitness"), "bench_raw": e["bench_raw"],
               "max_time_s": self.cfg.max_time_s, "config_hash": self.hash,
               "fitness_weights": self.cfg.fitness, "n_red": self.cfg.n_red,
               "blue_start": self.cfg.blue_start,
               "best_index": best_k, "worst_index": worst_k,
               "weights": f"{name}_weights.npz",
               "stats": {"missile_outcomes": outcomes}}
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
                                        doc["max_time_s"], True, doc.get("fitness_weights"))
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
    jobs = [(pol, p, doc["max_time_s"], False, doc.get("fitness_weights"))
            for p in doc["presentations"]]
    pool = make_pool(workers)
    try:
        out = pmap(pool, policy_job, jobs)
    finally:
        if pool is not None:
            pool.shutdown()
    per = [float(f) for f, _, _, _ in out]
    ok = per == doc["per_fight"]
    lines = [f"Re-run on {len(per)} stored presentations: fitness "
             f"{aggregate(per, doc.get('fitness_weights') or {})!r} vs stored "
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
