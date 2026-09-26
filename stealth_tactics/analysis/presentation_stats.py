"""Spec 4 stats over random presentations (``python -m stealth_tactics
presentation-stats``): default genome (scripted Blue), 360 s cap, spec 4 early
end rules. Also measures runtime per engagement and the per-generation cost."""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from stealth_tactics.presentation_runner import run_presentation
from stealth_tactics.scenarios.presentation import (
    BANDS, DEFAULT_PRESENTATION_CONFIG, sample_presentation)
from stealth_tactics.sim.world import END_REASONS

STATS_SALT = 0x57A7
END_ABBR = {"blue_dead": "Bdead", "red_dead": "Rdead", "both_winchester": "Wch",
            "red_departed": "Rdep", "time_cap": "cap"}


def stats_seeds(n: int, master_seed: int = 2026) -> List[int]:
    ss = np.random.SeedSequence([int(master_seed), STATS_SALT])
    return [int(s) for s in ss.generate_state(n, dtype=np.uint64)]


def run_one(seed: int, blue_test: bool = False) -> dict:
    p = sample_presentation(seed)
    t0 = time.perf_counter()
    r = run_presentation(p, blue_test_defense=blue_test)
    el = time.perf_counter() - t0
    shots = r.shots
    ev = Counter(e["type"] for e in r.events)
    pp = r.preplanned
    return {
        "seed": seed, "maneuver": p.maneuver["type"], "band": p.band, "doctrine": p.doctrine,
        "formation": p.formation, "a": p.aggressiveness, "range_nm": p.range_nm,
        "time_s": r.time_s, "wall_s": el, "end_reason": r.end_reason,
        "blue_kills": r.blue_kills, "red_kills": r.red_kills, "blue_alive": r.blue_alive,
        "blue_shots": sum(1 for s in shots if s["coalition"] == "Blue"),
        "red_shots": sum(1 for s in shots if s["coalition"] == "Red"),
        "blue_hits": sum(1 for s in shots if s["coalition"] == "Blue" and s["outcome"] == "hit"),
        "red_hits": sum(1 for s in shots if s["coalition"] == "Red" and s["outcome"] == "hit"),
        "departs": ev["depart"], "winchester": ev["winchester"], "presses": ev["press"],
        "fired": pp["fired"], "fire_range_nm": pp["fire_range_nm"],
        "status": dict(pp["status_counts"]),
    }


def _job(a):
    return run_one(*a)


def run_stats(n: int = 100, blue_test: bool = False, workers: Optional[int] = None,
              master_seed: int = 2026) -> List[dict]:
    jobs = [(s, blue_test) for s in stats_seeds(n, master_seed)]
    workers = workers or min(8, os.cpu_count() or 1)
    if workers <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs, chunksize=2))


def _row(label: str, runs: List[dict]) -> str:
    n = len(runs)
    bs = sum(r["blue_shots"] for r in runs)
    rs = sum(r["red_shots"] for r in runs)
    bh = sum(r["blue_hits"] for r in runs)
    rh = sum(r["red_hits"] for r in runs)
    endc = Counter(r["end_reason"] for r in runs)
    ends = " ".join(f"{END_ABBR[k]} {endc[k]}" for k in END_REASONS if endc[k])
    dur = [r["time_s"] for r in runs]
    return (f"| {label} | {n} | {bs / n:.1f} | {100 * bh / bs if bs else 0:.0f} % | "
            f"{rs / n:.1f} | {100 * rh / rs if rs else 0:.0f} % | "
            f"{np.mean([r['blue_kills'] for r in runs]):.2f} | "
            f"{np.mean([r['red_kills'] for r in runs]):.2f} | "
            f"{np.median(dur):.0f} | {100 * np.mean([d >= 359.9 for d in dur]):.0f} % | {ends} |")


HEADER = ("| group | n | B shots | B hit | R shots | R hit | B kills | B losses | "
          "dur med s | % cap | end reasons |\n|---|---|---|---|---|---|---|---|---|---|---|")


def format_stats(runs: List[dict], title: str) -> str:
    L = [title, "", HEADER, _row("all", runs)]
    for key, vals in (("maneuver", sorted({r["maneuver"] for r in runs})),
                      ("band", list(BANDS)),
                      ("doctrine", sorted({r["doctrine"] for r in runs})),
                      ("formation", sorted({r["formation"] for r in runs}))):
        for v in vals:
            sub = [r for r in runs if r[key] == v]
            if sub:
                lab = v.replace("shoot_assess_shoot", "SAS").replace("shoot_shoot_assess", "SSA")
                L.append(_row(f"{key}={lab}", sub))
    # maneuver x band where each cell has >= 5
    L += ["", "Maneuver x band (cells with n >= 5):", "", HEADER]
    for m in sorted({r["maneuver"] for r in runs}):
        for b in BANDS:
            sub = [r for r in runs if r["maneuver"] == m and r["band"] == b]
            if len(sub) >= 5:
                L.append(_row(f"{m}/{b}", sub))
    st = Counter()
    for r in runs:
        st.update(r["status"])
    nf = sum(1 for r in runs if not r["fired"])
    L += ["",
          "End reasons: Bdead = blue_dead (every Blue dead), Rdead = red_dead, "
          "Wch = both_winchester, Rdep = red_departed (every live Red departed), "
          "cap = time_cap (360 s). B losses = Red kills.",
          "End reason counts: " + ", ".join(
              f"{k} {sum(1 for r in runs if r['end_reason'] == k)}" for k in END_REASONS),
          f"Pre-planned maneuver: fired in {len(runs) - nf}/{len(runs)} fights; per-jet "
          f"status totals {dict(st)} (done = flown to completion, aborted = defense took "
          f"over, skipped = not HOT at trigger).",
          f"Red departures {sum(r['departs'] for r in runs)} (of which Winchester "
          f"{sum(r['winchester'] for r in runs)}), presses {sum(r['presses'] for r in runs)}.",
          f"Mean wall time per engagement (inside workers): "
          f"{np.mean([r['wall_s'] for r in runs]):.2f} s."]
    return "\n".join(L)


def timing(n_serial: int = 12, n_parallel: int = 96, workers: int = 8,
           master_seed: int = 77) -> Dict[str, float]:
    seeds = stats_seeds(n_serial + n_parallel, master_seed)
    t0 = time.perf_counter()
    sims = [run_one(s)["time_s"] for s in seeds[:n_serial]]
    serial = (time.perf_counter() - t0) / n_serial
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_job, [(s, False) for s in seeds[n_serial:]], chunksize=2))
    par = time.perf_counter() - t0
    return {"serial_s_per_fight": serial, "mean_sim_s": float(np.mean(sims)),
            "parallel_wall_s": par, "parallel_n": n_parallel, "workers": workers,
            "parallel_s_per_fight": par / n_parallel}


def format_timing(t: Dict[str, float], pop: int = 50, n_eval: int = 24,
                  bench: int = 64) -> str:
    fights = pop * n_eval + bench
    per_gen = fights * t["parallel_s_per_fight"]
    return "\n".join([
        f"Serial: {t['serial_s_per_fight']:.2f} s per engagement "
        f"(6 Red, 360 s cap, mean simulated {t['mean_sim_s']:.0f} s).",
        f"Process pool, {t['workers']} workers: {t['parallel_n']} engagements in "
        f"{t['parallel_wall_s']:.1f} s = {t['parallel_s_per_fight']:.3f} s wall per engagement.",
        f"Per generation (population {pop} x {n_eval} presentations + {bench} benchmark "
        f"= {fights} engagements): ~{per_gen:.0f} s wall on {t['workers']} workers "
        f"(~{per_gen / 60:.1f} min) -> ~{36000 / per_gen:.0f} generations per 10 h night "
        f"(scripted Blue; the spec 6 network adds its own cost).",
    ])
