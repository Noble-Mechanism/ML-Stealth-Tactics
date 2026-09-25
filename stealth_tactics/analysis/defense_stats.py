"""Spec 3 100-seed stats (``python -m stealth_tactics defense-stats``).

default_4v3, default genome, Red aggressiveness a in {0, .25, .5, .75, 1},
Blue scripted test reaction off / on, Blue firing doctrine shoot_assess_shoot /
shoot_shoot_assess (Red stays at its default shoot_assess_shoot). Runs in a
process pool (``STEALTH_TACTICS_STATS_WORKERS``, default CPU count).
"""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.sim.world import SimConfig, World, SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS
from stealth_tactics.scenarios.loader import load_scenario, build_aircraft
from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.tactics.interpreter import TacticsController, RedCAPController
from stealth_tactics.tactics.red_defense import RedDefense
from .blue_test_defense import BlueTestDefense

ROOT = Path(__file__).resolve().parents[2]
CAP_S = 240.0


def run_one(scenario_path: str, a: float, blue_test: bool, blue_doctrine: str, seed: int,
            red_defense: bool = True, red_doctrine: str = SHOOT_ASSESS_SHOOT,
            max_time_s: float = CAP_S) -> dict:
    sc = load_scenario(scenario_path)
    aircraft = build_aircraft(sc)
    blue_ids = [x.id for x in aircraft if x.coalition == Coalition.BLUE]
    red_ids = [x.id for x in aircraft if x.coalition == Coalition.RED]
    blue = TacticsController(TacticsGenome(), blue_ids)
    if blue_test:
        blue = BlueTestDefense(blue)
    defense = RedDefense(a) if red_defense else None
    red = RedCAPController(red_ids, mode=sc.red_mode, defense=defense)
    # same seed convention as GAConfig(seed=0).evaluate(seed_offset=seed)
    cfg = SimConfig(dt=0.5, max_time_s=max_time_s, seed=1000 + seed,
                    blue_doctrine=blue_doctrine, red_doctrine=red_doctrine)
    t0 = time.perf_counter()
    w = World(aircraft, cfg, blue, red, record=False)
    r = w.run()
    el = time.perf_counter() - t0
    shots = r.shots
    jets = defense.jets.values() if defense else []
    return {
        "seed": seed, "time_s": r.time_s, "wall_s": el, "ended_early": r.ended_early,
        "blue_kills": r.blue_kills, "red_kills": r.red_kills,
        "blue_shots": sum(1 for s in shots if s["coalition"] == "Blue"),
        "red_shots": sum(1 for s in shots if s["coalition"] == "Red"),
        "blue_hits": sum(1 for s in shots if s["coalition"] == "Blue" and s["outcome"] == "hit"),
        "red_hits": sum(1 for s in shots if s["coalition"] == "Red" and s["outcome"] == "hit"),
        "turn_aways": sum(j.turn_aways for j in jets),
        "departures": sum(1 for j in jets if j.state == "departed"),
        "presses": sum(1 for j in jets if j.state == "pressing"),
        "blue_defends": (sum(j.n_defends for j in blue.jets.values()) if blue_test else 0),
        "shots": [{k: s[k] for k in ("coalition", "outcome", "launch_range_m", "a_pole_m",
                                     "f_pole_m")} for s in shots],
    }


def _job(args):
    return run_one(*args)


CONFIGS = [(a, bt, doc) for doc in (SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS)
           for bt in (False, True) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]


def run_stats(seeds: int = 100, configs=None, scenario: Optional[str] = None,
              workers: Optional[int] = None) -> Dict[tuple, List[dict]]:
    scenario = scenario or str(ROOT / "scenarios" / "default_4v3.yaml")
    configs = configs or CONFIGS
    jobs = [(scenario, a, bt, doc, s) for (a, bt, doc) in configs for s in range(seeds)]
    workers = workers or int(os.environ.get("STEALTH_TACTICS_STATS_WORKERS", os.cpu_count() or 1))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_job, jobs, chunksize=4))
    else:
        res = [_job(j) for j in jobs]
    out: Dict[tuple, List[dict]] = {}
    for j, r in zip(jobs, res):
        out.setdefault(j[1:4], []).append(r)
    return out


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def _dist(vals) -> str:
    v = [x / NM_M for x in vals if x is not None]
    if not v:
        return "-"
    return f"{np.median(v):4.1f} [{_pct(v, 10):4.1f}-{_pct(v, 90):4.1f}] n={len(v)}"


def _res_rate(shots, coal) -> float:
    """Hits / resolved shots (excludes missiles still flying when the fight ended)."""
    res = [s for s in shots if s["coalition"] == coal and s["outcome"] != "in_flight"]
    return sum(1 for s in res if s["outcome"] == "hit") / max(len(res), 1)


def summarize(key, runs: List[dict]) -> dict:
    a, bt, doc = key
    t = np.array([r["time_s"] for r in runs])
    bs = sum(r["blue_shots"] for r in runs)
    rs = sum(r["red_shots"] for r in runs)
    shots = [s for r in runs for s in r["shots"]]
    return {
        "a": a, "blue_test": bt, "doctrine": doc, "n": len(runs),
        "blue_shots": bs / len(runs), "blue_hit": sum(r["blue_hits"] for r in runs) / max(bs, 1),
        "red_shots": rs / len(runs), "red_hit": sum(r["red_hits"] for r in runs) / max(rs, 1),
        "blue_hit_res": _res_rate(shots, "Blue"), "red_hit_res": _res_rate(shots, "Red"),
        "blue_kills": float(np.mean([r["blue_kills"] for r in runs])),
        "red_kills": float(np.mean([r["red_kills"] for r in runs])),
        "dur_med": float(np.median(t)), "dur_p90": _pct(t, 90),
        "cap_pct": 100.0 * float(np.mean(t >= CAP_S - 1e-6)),
        "early_pct": 100.0 * float(np.mean([r["ended_early"] for r in runs])),
        "turn_aways": float(np.mean([r["turn_aways"] for r in runs])),
        "departures": float(np.mean([r["departures"] for r in runs])),
        "presses": float(np.mean([r["presses"] for r in runs])),
        "blue_defends": float(np.mean([r["blue_defends"] for r in runs])),
        "wall_s": float(np.mean([r["wall_s"] for r in runs])),
        "blue_apole": _dist([s["a_pole_m"] for s in shots if s["coalition"] == "Blue"]),
        "blue_fpole": _dist([s["f_pole_m"] for s in shots if s["coalition"] == "Blue"]),
        "red_apole": _dist([s["a_pole_m"] for s in shots if s["coalition"] == "Red"]),
        "red_fpole": _dist([s["f_pole_m"] for s in shots if s["coalition"] == "Red"]),
        "blue_launch": _dist([s["launch_range_m"] for s in shots if s["coalition"] == "Blue"]),
        "red_launch": _dist([s["launch_range_m"] for s in shots if s["coalition"] == "Red"]),
        "blue_outcomes": Counter(s["outcome"] for s in shots if s["coalition"] == "Blue"),
        "red_outcomes": Counter(s["outcome"] for s in shots if s["coalition"] == "Red"),
    }


def format_tables(summaries: List[dict], seeds: int) -> str:
    L = [f"Spec 3 defense stats: default_4v3, default genome, {seeds} seeds per row "
         f"(sim seed 1000+i, cap {CAP_S:.0f} s). Red doctrine shoot_assess_shoot; "
         "'doctrine' = Blue firing doctrine (SAS = shoot-assess-shoot, SSA = "
         "shoot-shoot-assess); 'test' = Blue scripted test reaction.", ""]
    hdr = (f"{'doct':4} {'test':4} {'a':>4} | {'B sh':>5} {'B hit':>5} {'res':>4} {'R sh':>5} "
           f"{'R hit':>5} {'res':>4} | "
           f"{'B kil':>5} {'R kil':>5} | {'dur med':>7} {'p90':>5} {'%cap':>4} {'%early':>6} | "
           f"{'turn':>4} {'dep':>4} {'press':>5} {'Bdef':>4} | flag")
    L += [hdr, "-" * len(hdr)]
    for s in summaries:
        doc = "SAS" if s["doctrine"] == SHOOT_ASSESS_SHOOT else "SSA"
        flag = "MEDIAN > 200 s" if s["dur_med"] > 200 else ""
        L.append(f"{doc:4} {('on' if s['blue_test'] else 'off'):4} {s['a']:4.2f} | "
                 f"{s['blue_shots']:5.2f} {100 * s['blue_hit']:4.0f}% "
                 f"{100 * s['blue_hit_res']:3.0f}% {s['red_shots']:5.2f} "
                 f"{100 * s['red_hit']:4.0f}% {100 * s['red_hit_res']:3.0f}% | "
                 f"{s['blue_kills']:5.2f} {s['red_kills']:5.2f} | "
                 f"{s['dur_med']:7.1f} {s['dur_p90']:5.1f} {s['cap_pct']:4.0f} "
                 f"{s['early_pct']:6.0f} | {s['turn_aways']:4.2f} {s['departures']:4.2f} "
                 f"{s['presses']:5.2f} {s['blue_defends']:4.1f} | {flag}")
    L += ["", "B sh / R sh = shots per engagement; hit = hits / shots; res = hits / resolved "
          "shots (excludes missiles still flying when one side was wiped out); kil = kills per "
          "engagement (B kil = Red jets killed by Blue); dur = engagement duration (s); %cap "
          "= reached the 240 s cap; %early = ended early (all live Red departed, no missile "
          "in flight); turn = Red turn-aways per engagement (sum over 3 jets); dep / press = "
          "Red jets that departed / pressed; Bdef = Blue test-reaction defenses per engagement.",
          "", "Pole distributions (NM, median [p10-p90] n):"]
    for s in summaries:
        doc = "SAS" if s["doctrine"] == SHOOT_ASSESS_SHOOT else "SSA"
        L.append(f"  {doc} test {'on ' if s['blue_test'] else 'off'} a={s['a']:.2f}: "
                 f"Blue launch {s['blue_launch']}; a-pole {s['blue_apole']}; f-pole "
                 f"{s['blue_fpole']}")
        L.append(f"  {'':25} Red  launch {s['red_launch']}; a-pole {s['red_apole']}; f-pole "
                 f"{s['red_fpole']}")
    L += ["", "Outcome labels (all shots):"]
    for s in summaries:
        doc = "SAS" if s["doctrine"] == SHOOT_ASSESS_SHOOT else "SSA"
        fmt = lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1]))
        L.append(f"  {doc} test {'on ' if s['blue_test'] else 'off'} a={s['a']:.2f}: Blue: "
                 f"{fmt(s['blue_outcomes'])} | Red: {fmt(s['red_outcomes']) or '-'}")
    return "\n".join(L) + "\n"
