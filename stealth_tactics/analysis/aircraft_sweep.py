"""Spec 8: jet energy bleed calibration sweep (counterpart to missile-sweep)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional

from stealth_tactics.sim.aircraft import Aircraft, AircraftState, integrate_aircraft, F35_PARAMS
from stealth_tactics.sim.missile_kinematics import atmosphere
from stealth_tactics.sim.sensor_config import AircraftEnergyConfig, BLUE_ENERGY

FT = 0.3048


def bleed_run(n_max: float = 3.5, alt_ft: float = 40_000.0, start_mach: float = 0.9,
              t_s: float = 20.0, dt: float = 0.05,
              energy: Optional[AircraftEnergyConfig] = None) -> dict:
    """Command a hard turn (heading 180° off) at constant cmd speed/alt.

    Returns Mach / speed samples and the final Mach. With ``n_max`` around 3–4
    this is the Spec 8 A-d accept gate (Mach 0.9 → ~0.7 in ~20 s at 40 kft).
    """
    e = energy or replace(BLUE_ENERGY, n_max=n_max)
    p = replace(F35_PARAMS, energy=e, min_speed_mps=e.min_speed_mps, max_alt_m=e.max_alt_m)
    alt = alt_ft * FT
    a = atmosphere(alt)[1]
    V0 = start_mach * a
    ac = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt, 0.0, V0))
    ac.params = p
    ac.cmd_heading_rad = math.pi
    ac.cmd_speed_mps = V0
    ac.cmd_alt_m = alt
    samples: List[dict] = [{"t": 0.0, "mach": start_mach, "speed_mps": V0, "alt_m": alt}]
    nsteps = int(round(t_s / dt))
    for i in range(1, nsteps + 1):
        integrate_aircraft(ac, dt)
        if i % int(round(1.0 / dt)) == 0 or i == nsteps:
            _, a_now = atmosphere(ac.state.alt), atmosphere(ac.state.alt)
            a_now = atmosphere(ac.state.alt)[1]
            samples.append({"t": i * dt, "mach": ac.state.speed_mps / a_now,
                            "speed_mps": ac.state.speed_mps, "alt_m": ac.state.alt})
    a_end = atmosphere(ac.state.alt)[1]
    return {"n_max": n_max, "alt_ft": alt_ft, "start_mach": start_mach, "t_s": t_s,
            "final_mach": ac.state.speed_mps / a_end, "samples": samples,
            "energy": e}


def format_bleed(rep: dict) -> str:
    lines = [f"Jet bleed: n_max={rep['n_max']}, alt={rep['alt_ft']:.0f} ft, "
             f"start Mach {rep['start_mach']:.2f}, t={rep['t_s']:.0f} s",
             f"  final Mach {rep['final_mach']:.3f}",
             f"  {'t_s':>6} {'Mach':>7} {'V m/s':>8} {'alt m':>8}"]
    for s in rep["samples"]:
        lines.append(f"  {s['t']:6.1f} {s['mach']:7.3f} {s['speed_mps']:8.1f} {s['alt_m']:8.0f}")
    return "\n".join(lines) + "\n"


def gate_ok(rep: dict, lo: float = 0.60, hi: float = 0.80) -> bool:
    """A-d accept: final Mach in [lo, hi] (~0.7)."""
    return lo <= rep["final_mach"] <= hi
