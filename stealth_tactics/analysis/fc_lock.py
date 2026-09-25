"""Spec 2b: Blue fire-control lock stability head-on vs Red (50 NM FC gate).

Blue F-35 (260 m/s) and Red (255 m/s) fly straight at each other from 70 NM.
Per seed we log Blue's FC status every 0.5 s step and report the first FC range,
fraction of time FC is held inside range bands, and FC drop/reacquire cycles.
The continuity rule is unchanged (>= 3 s held, streak breaks after > 2 s gap).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, AircraftState
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.sim.sensors import distance_3d
from stealth_tactics.sim.world import SimConfig, World

BANDS: List[Tuple[float, float]] = [(50.0, 40.0), (40.0, 30.0)]


def run_one(seed: int, start_nm: float = 70.0, stop_nm: float = 28.0) -> dict:
    b = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, 9000.0, 0.0, 260.0))
    r = Aircraft.make_red("R1", "Red-1",
                          AircraftState(0.0, start_nm * NM_M, 9000.0, math.pi, 255.0))
    samples: List[Tuple[float, float, bool]] = []

    def blue_ctl(w: World) -> None:
        samples.append((w.time_s, distance_3d(b.state, r.state) / NM_M,
                        w.tracks["B1"].is_fire_control("R1")))
        b.cmd_heading_rad, b.cmd_speed_mps = 0.0, 260.0

    def red_ctl(w: World) -> None:
        r.cmd_heading_rad, r.cmd_speed_mps = math.pi, 255.0

    dur = (start_nm - stop_nm) * NM_M / 515.0
    w = World([b, r], SimConfig(dt=0.5, max_time_s=dur, seed=seed), blue_ctl, red_ctl,
              record=False)
    w.run()
    first = next((rng for _, rng, fc in samples if fc), None)
    out = {"seed": seed, "first_fc_nm": first, "bands": {}}
    for hi, lo in BANDS:
        sel = [(rng, fc) for _, rng, fc in samples if lo < rng <= hi]
        drops = reacq = 0
        had = bool(sel) and sel[0][1]
        for (_, f0), (_, f1) in zip(sel, sel[1:]):
            drops += f0 and not f1
            reacq += (not f0) and f1 and had     # re-lock after an earlier lock
            had = had or f1
        out["bands"][(hi, lo)] = {"held_frac": (sum(fc for _, fc in sel) / len(sel))
                                  if sel else 0.0, "drops": drops, "reacq": reacq}
    return out


def lock_stats(seeds: int = 100) -> str:
    runs = [run_one(s) for s in range(seeds)]
    firsts = [r["first_fc_nm"] for r in runs if r["first_fc_nm"] is not None]
    lines = [f"Blue FC lock stability head-on vs Red (F-35 260 m/s, Red 255 m/s, from 70 NM),"
             f" {seeds} seeds",
             "FC gate: Blue 92.6 km (50 NM) x rcs_eff^0.25 (Red isotropic 1.0 -> 50 NM); "
             "continuity rule unchanged (>= 3 s held, break after > 2 s without a hit).",
             f"first FC range: median {np.median(firsts):.1f} NM, p10 "
             f"{np.percentile(firsts, 10):.1f}, p90 {np.percentile(firsts, 90):.1f}, "
             f"min {min(firsts):.1f}, max {max(firsts):.1f} (FC in {len(firsts)}/{seeds} seeds;"
             " gate is applied to the ESTIMATED radar range, so >50 NM true is possible)"]
    for hi, lo in BANDS:
        hf = [r["bands"][(hi, lo)]["held_frac"] for r in runs]
        dr = [r["bands"][(hi, lo)]["drops"] for r in runs]
        rq = [r["bands"][(hi, lo)]["reacq"] for r in runs]
        lines.append(
            f"band {hi:.0f}-{lo:.0f} NM: FC held {np.mean(hf) * 100:.0f}% of time (mean; "
            f"median {np.median(hf) * 100:.0f}%), drops per run: mean {np.mean(dr):.2f}, "
            f"median {np.median(dr):.0f}, max {max(dr)}; reacquisitions mean "
            f"{np.mean(rq):.2f}; runs with >= 1 drop: {sum(d > 0 for d in dr)}/{seeds}")
    return "\n".join(lines)
