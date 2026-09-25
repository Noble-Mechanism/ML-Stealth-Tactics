"""First-detection range table for every Spec 1 sensor pairing.

Analytic column: the 50 %-per-scan range (R50) from the model formulas (RWR:
the deterministic intercept envelope). Empirical column: median (p10-p90) of the
first-detection range over N seeded closing runs that exercise the full
``SensorModel.update`` pipeline (1 Hz radar/IRST scans, 0.5 s RWR updates).

Closing-run geometry: observer at the origin flying north; target on the
observer's nose, heading set to give the requested target aspect (nose / beam /
tail). The target is moved straight down the line of sight at a fixed abstract
closure of 500 m/s (so aspect stays constant and rows are comparable); the
start range is jittered by up to one scan of closure (random scan phase).
Cumulative per-scan Pd means empirical first-detection ranges sit beyond R50.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, AircraftState
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG, SensorConfig, NM_M
from stealth_tactics.sim.sensors import SensorModel, radar_r50, irst_r50, rwr_range

CLOSURE_MPS = 500.0
DT = 0.5
ASPECT_HEADING = {"nose": math.pi, "beam": math.pi / 2, "tail": 0.0}


@dataclass
class Row:
    sensor: str
    observer: str
    target: str
    aspect: str
    analytic_m: float
    fc_m: Optional[float]
    emp_med: float
    emp_p10: float
    emp_p90: float


def _make(side: str, y: float, hdg: float) -> Aircraft:
    if side == "blue":
        ac = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, y, 8000.0, hdg, 0.0))
    else:
        ac = Aircraft.make_red("R1", "Red-1", AircraftState(0.0, y, 8000.0, hdg, 0.0))
    ac.state.speed_mps = ac.params.cruise_speed_mps
    return ac


def _pair(obs_side: str, aspect: str):
    obs = _make(obs_side, 0.0, 0.0)
    tgt = _make("red" if obs_side == "blue" else "blue", 50_000.0, ASPECT_HEADING[aspect])
    return obs, tgt


def empirical_first_detect(obs_side: str, aspect: str, source: str, start_m: float,
                           seeds: int, cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> np.ndarray:
    ranges = []
    for seed in range(seeds):
        rng = np.random.default_rng([seed, 2024])
        obs, tgt = _pair(obs_side, aspect)
        tgt.state.y = start_m + rng.uniform(0.0, CLOSURE_MPS * 1.0)
        sm = SensorModel(np.random.default_rng([seed, 1]), cfg)
        t, found = 0.0, None
        while tgt.state.y > 500.0:
            for e in sm.update([obs, tgt], t):
                if e["observer"] == obs.id and e["type"] == f"{source}_detect":
                    found = e["range_m"]
                    break
            if found is not None:
                break
            tgt.state.y -= CLOSURE_MPS * DT
            t += DT
        ranges.append(found if found is not None else 0.0)
    return np.array(ranges)


def build_table(seeds: int = 300, cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> List[Row]:
    rows: List[Row] = []
    frac = cfg.track.fc_range_frac

    def add(sensor, obs_side, aspect, analytic, start, fc=None):
        emp = empirical_first_detect(obs_side, aspect, sensor, start, seeds, cfg)
        o, t = ("F-35", "Red") if obs_side == "blue" else ("Red", "F-35")
        label = {"radar": "Radar", "irst": "IRST", "rwr": "RWR"}[sensor]
        rows.append(Row(label, o, t, aspect, analytic, fc, float(np.median(emp)),
                        float(np.percentile(emp, 10)), float(np.percentile(emp, 90))))

    # Radar
    for obs_side, aspects in (("blue", ["nose", "beam", "tail"]),
                              ("red", ["nose", "beam", "tail"])):
        for asp in aspects:
            obs, tgt = _pair(obs_side, asp)
            r50 = radar_r50(obs, tgt, cfg)
            cap = cfg.radar.instrumented_range_factor * obs.params.radar_range_m
            add("radar", obs_side, asp, r50, min(2.2 * r50, cap), fc=frac * r50)
    # IRST
    for obs_side in ("blue", "red"):
        for asp in ("nose", "tail"):
            obs, tgt = _pair(obs_side, asp)
            r50 = irst_r50(obs, tgt, cfg)
            add("irst", obs_side, asp, r50, cfg.irst.max_range_factor * r50)
    # RWR (receiver = observer; emitter = the other jet, pointing at receiver)
    for obs_side in ("red", "blue"):
        obs, tgt = _pair(obs_side, "nose")
        rr = rwr_range(tgt, cfg)
        add("rwr", obs_side, "nose", rr, rr + 5_000.0)
    return rows


def format_table(rows: List[Row], seeds: int) -> str:
    def nk(m):
        return f"{m / NM_M:6.1f} NM {m / 1000:6.1f} km"

    hdr = (f"{'Sensor':6} {'Observer->Target':17} {'Tgt aspect':10} "
           f"{'Analytic R50 / envelope':25} {'FC range (0.7 R50)':25} "
           f"{'Empirical median first-detect':31} {'p10 - p90 (NM)':14}")
    lines = [
        "Spec 1 sensor first-detection table (UNCLASSIFIED PLACEHOLDER parameters)",
        f"Empirical: {seeds} seeded closing runs per row, closure {CLOSURE_MPS:.0f} m/s,"
        f" dt {DT} s, radar/IRST 1 Hz, RWR every step.",
        "Targets at cruise speed (F-35 260 m/s, Red 255 m/s). Radar/IRST analytic value"
        " = 50 %-per-scan range; RWR = deterministic intercept envelope.",
        "", hdr, "-" * len(hdr),
    ]
    for r in rows:
        fc = nk(r.fc_m) if r.fc_m else "-"
        lines.append(
            f"{r.sensor:6} {r.observer + '->' + r.target:17} {r.aspect:10} "
            f"{nk(r.analytic_m):25} {fc:25} {nk(r.emp_med):31} "
            f"{r.emp_p10 / NM_M:5.1f} - {r.emp_p90 / NM_M:5.1f}"
        )
    lines += [
        "",
        "Notes: F-35 radar vs Red is aspect-independent (Red isotropic RCS 1.0)."
        " Red radar vs F-35 uses the F-35 aspect table (nose 0.05 / beam 0.90 / tail 0.30).",
        "RWR: Red RWR vs F-35 LPI radar = 0.6 x 90 km; Blue RWR vs Red radar = 1.5 x 70 km;"
        " receiver must be inside the emitter's +/-60 deg field of regard.",
        "IRST tracks are passive and can never support a missile shot.",
    ]
    return "\n".join(lines) + "\n"
