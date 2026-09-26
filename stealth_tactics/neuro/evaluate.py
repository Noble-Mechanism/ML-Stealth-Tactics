"""Spec 6 N: evaluation workers.

The generation's population is written once to a deterministic npz file;
each worker loads it once per generation (cached by path) and jobs are only
``(pop_file, member_index, presentation, record)``. Workers run in a
``spawn`` pool whose environment pins BLAS / OpenMP to one thread, so a fight
is bit-identical under any worker count."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import List, Optional, Sequence

import numpy as np

THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

_CACHE = {"path": None, "nets": None, "arch": None}


def pin_threads_env() -> None:
    for k in THREAD_ENV:
        os.environ[k] = "1"


def _pool_init() -> None:
    import signal
    pin_threads_env()
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # Ctrl-C / stop is handled by the parent


def make_pool(workers: int) -> Optional[ProcessPoolExecutor]:
    if workers <= 1:
        return None
    pin_threads_env()                     # inherited by the spawned children
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_pool_init)


def worker_env(_=None) -> dict:
    return {k: os.environ.get(k) for k in THREAD_ENV}


def _population(path: str):
    if _CACHE["path"] != path:
        import json
        from .checkpoint import load_npz
        from .mlp import Arch
        arrays = load_npz(path)
        meta = json.loads(open(path[:-4] + ".json", encoding="utf-8").read())
        _CACHE.update(path=path, nets=arrays["weights"], arch=Arch.from_dict(meta["arch"]))
    return _CACHE["nets"], _CACHE["arch"]


def run_policy_fight(policy, pdict: dict, max_time_s: float = 360.0, record: bool = False,
                     fitness: Optional[dict] = None):
    """One fight of *policy* through the spec 5 controller. Returns
    (fitness, raw behaviour dict, summary, SimResult-or-None). *fitness* =
    spec 7 weights (None -> scenarios/fitness.yaml)."""
    from stealth_tactics.fitness import fight_terms, load_weights
    from stealth_tactics.policy.controller import NetworkBlueController
    from stealth_tactics.presentation_runner import run_presentation
    from stealth_tactics.scenarios.presentation import DEFAULT_PRESENTATION_CONFIG
    from .novelty import BDRecorder
    w = fitness if fitness is not None else load_weights()
    holder = {}

    def fac(ids):
        holder["r"] = BDRecorder(NetworkBlueController(policy, ids), ids)
        return holder["r"]
    pcfg = replace(DEFAULT_PRESENTATION_CONFIG, max_time_s=max_time_s)
    res = run_presentation(pdict, cfg=pcfg, record=record, blue_factory=fac)
    rec = holder["r"]
    terms = fight_terms(res, len(pdict["red_jets"]), rec.egress_by_jet(w["egress_off_deg"]), w)
    shots = res.shots
    summ = {"kills": res.blue_kills, "losses": res.red_kills,
            "egress_losses": terms["n_egress_losses"], "escaped": terms["n_escaped"],
            "red_departs": terms["n_red_departs"],
            "blue_shots": sum(1 for s in shots if s["coalition"] == "Blue"),
            "blue_hits": sum(1 for s in shots if s["coalition"] == "Blue"
                             and s["outcome"] == "hit"),
            "red_shots": sum(1 for s in shots if s["coalition"] == "Red"),
            "end_reason": res.end_reason, "time_s": res.time_s}
    return terms["total"], rec.raw(res), summ, (res if record else None)


def member_job(args):
    pop_path, idx, pdict, max_time_s, record, fitness = args
    from .mlp import MLPPolicy
    nets, arch = _population(pop_path)
    return run_policy_fight(MLPPolicy(nets[idx], arch), pdict, max_time_s, record, fitness)


def policy_job(args):
    """(policy object, presentation, max_time_s, record[, fitness]) - for fixed
    policies (HandBlue, clones)."""
    import copy
    pol, pdict, max_time_s, record = args[:4]
    fitness = args[4] if len(args) > 4 else None
    return run_policy_fight(copy.deepcopy(pol), pdict, max_time_s, record, fitness)   # HandBlue is stateful


def pmap(pool, fn, jobs: Sequence) -> List:
    if pool is None:
        return [fn(j) for j in jobs]
    return list(pool.map(fn, jobs, chunksize=max(1, len(jobs) // (4 * pool._max_workers))))
