"""Spec 8 / 8b: jet energy calibration (bleed gate, lift limit, transonic hump)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional

import numpy as np

from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, integrate_aircraft,
                                          F35_PARAMS, RED_FIGHTER_PARAMS)
from stealth_tactics.sim.missile_kinematics import atmosphere
from stealth_tactics.sim.sensor_config import AircraftEnergyConfig, BLUE_ENERGY, RED_ENERGY

G0 = 9.80665

FT = 0.3048


def _params_for(energy: AircraftEnergyConfig):
    base = RED_FIGHTER_PARAMS if energy is RED_ENERGY else F35_PARAMS
    return replace(base, energy=energy, min_speed_mps=energy.min_speed_mps,
                   max_alt_m=energy.max_alt_m)


def _make_jet(energy: AircraftEnergyConfig, alt_m: float, mach: float) -> Aircraft:
    V0 = mach * atmosphere(alt_m)[1]
    ac = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt_m, 0.0, V0))
    ac.params = _params_for(energy)
    ac.cmd_heading_rad, ac.cmd_speed_mps, ac.cmd_alt_m = 0.0, V0, alt_m
    return ac


def _mach(ac: Aircraft) -> float:
    return ac.state.speed_mps / atmosphere(ac.state.alt)[1]


def bleed_run(n_max: Optional[float] = None, alt_ft: float = 40_000.0, start_mach: float = 0.9,
              t_s: float = 20.0, dt: float = 0.05,
              energy: Optional[AircraftEnergyConfig] = None) -> dict:
    """Command a hard turn (heading 180° off) at constant cmd speed/alt.

    Returns Mach / speed samples and the final Mach. Spec 8b: by default the jet
    pulls its available load factor min(n_max, q S CLmax / W) (lift-limited,
    ~3.1 g at 40 kft / Mach 0.9 for Blue); pass ``n_max`` to override the
    structural cap. This is the Spec 8 A-d gate (Mach 0.9 -> ~0.7 in ~20 s).
    """
    e = energy or BLUE_ENERGY
    if n_max is not None:
        e = replace(e, n_max=n_max)
    alt = alt_ft * FT
    a = atmosphere(alt)[1]
    V0 = start_mach * a
    ac = _make_jet(e, alt, start_mach)
    ac.cmd_heading_rad = math.pi
    samples: List[dict] = [{"t": 0.0, "mach": start_mach, "speed_mps": V0, "alt_m": alt}]
    nsteps = int(round(t_s / dt))
    for i in range(1, nsteps + 1):
        integrate_aircraft(ac, dt)
        if i % int(round(1.0 / dt)) == 0 or i == nsteps:
            samples.append({"t": i * dt, "mach": _mach(ac),
                            "speed_mps": ac.state.speed_mps, "alt_m": ac.state.alt})
    return {"n_max": e.n_max, "n_lift0": lift_limit_g(e, alt_ft, start_mach),
            "alt_ft": alt_ft, "start_mach": start_mach, "t_s": t_s,
            "final_mach": _mach(ac), "samples": samples, "energy": e}


# ------------------------------------------------------ Spec 8b analysis ---
def _terms(e: AircraftEnergyConfig, alt_ft: float, mach: float):
    rho, a = atmosphere(alt_ft * FT)
    V = mach * a
    qS = 0.5 * rho * V * V * e.S_m2
    W = e.mass_kg * G0
    return qS, W, e.thrust_n(rho, mach), qS * e.Cd0 * e.cd0_factor(mach), e.k_induced * W * W / qS


def lift_limit_g(e: AircraftEnergyConfig, alt_ft: float, mach: float) -> float:
    """Available load factor min(n_max, q S CLmax / W)."""
    qS, W, *_ = _terms(e, alt_ft, mach)
    return min(e.n_max, qS * e.CLmax / W)


def level_excess_n(e: AircraftEnergyConfig, alt_ft: float, mach: float, n: float = 1.0) -> float:
    """Full-thrust T - D (N) in level flight at load factor n."""
    _, _, T, d0, kind = _terms(e, alt_ft, mach)
    return T - d0 - kind * n * n


def sustained_g(e: AircraftEnergyConfig, alt_ft: float, mach: float) -> float:
    """Max level load factor sustainable (T = D) at this Mach, capped by the lift limit."""
    _, _, T, d0, kind = _terms(e, alt_ft, mach)
    if T <= d0 + kind:
        return 0.0
    return min(lift_limit_g(e, alt_ft, mach), math.sqrt((T - d0) / kind))


def best_sustained_g(e: AircraftEnergyConfig, alt_ft: float) -> tuple:
    """(g, Mach) of the best sustained level turn at this altitude."""
    best = (0.0, 0.0)
    for m in np.arange(0.30, e.max_mach + 1e-9, 0.005):
        g = sustained_g(e, alt_ft, float(m))
        if g > best[0]:
            best = (g, float(m))
    return best


def top_speeds(e: AircraftEnergyConfig, alt_ft: float, dm: float = 0.001) -> dict:
    """Static level top speeds (1 g, full thrust, Mach capped at max_mach).

    ``from_subsonic``: where a jet accelerating level from Mach 0.9 stalls out
    (first Mach with T < D), or max_mach if it punches through the hump.
    ``back_side``: lowest supersonic Mach sustainable when the hump blocks
    level acceleration (None if it does not).
    ``max_sustained``: highest Mach with T >= D (supersonic back side).
    """
    ms = np.arange(0.9, e.max_mach + 1e-9, dm)
    ok = [level_excess_n(e, alt_ft, float(m)) >= 0.0 for m in ms]
    stall = next((float(m) for m, o in zip(ms, ok) if not o), None)
    from_sub = stall if stall is not None else e.max_mach
    top = None
    if ok[-1]:
        top = e.max_mach
    else:
        for m, o in zip(ms[::-1], ok[::-1]):
            if o:
                top = float(m)
                break
    # lowest supersonic Mach from which T >= D all the way up to ``top``
    back = None
    if top is not None and top > 1.0 and stall is not None:
        i = len(ms) - 1 if ok[-1] else int(np.searchsorted(ms, top))
        while i > 0 and ok[i - 1]:
            i -= 1
        back = float(ms[i])
    return {"from_subsonic": from_sub, "max_sustained": top, "back_side": back}


def level_accel_run(e: AircraftEnergyConfig, alt_ft: float = 40_000.0,
                    start_mach: float = 0.9, t_s: float = 300.0, dt: float = 0.05) -> dict:
    """Level, full-thrust acceleration run; returns final / peak Mach."""
    ac = _make_jet(e, alt_ft * FT, start_mach)
    ac.cmd_speed_mps = 1.0e4
    peak = start_mach
    for _ in range(int(round(t_s / dt))):
        integrate_aircraft(ac, dt)
        peak = max(peak, _mach(ac))
    return {"alt_ft": alt_ft, "start_mach": start_mach, "t_s": t_s,
            "final_mach": _mach(ac), "peak_mach": peak, "final_alt_ft": ac.state.alt / FT}


def dive_climb_run(e: AircraftEnergyConfig, top_ft: float = 40_000.0, low_ft: float = 30_000.0,
                   start_mach: float = 0.9, climb_mach: float = 1.18, t_s: float = 900.0,
                   dt: float = 0.05) -> dict:
    """Start level at ``top_ft`` / ``start_mach``; dive to ``low_ft`` at full
    thrust until supersonic (min(climb_mach + 0.03, max_mach - 0.005)), then an
    energy-managed climb back to ``top_ft`` (climb only while Mach > climb_mach),
    then hold level there for the rest of ``t_s``."""
    ac = _make_jet(e, top_ft * FT, start_mach)
    ac.cmd_speed_mps = 1.0e4
    trig = min(climb_mach + 0.03, e.max_mach - 0.005)
    phase, t = "dive", 0.0
    out = {"top_ft": top_ft, "low_ft": low_ft, "climb_mach": climb_mach,
           "t_supersonic": None, "alt_supersonic_ft": None, "mach_at_trigger": None,
           "t_arrive": None, "mach_arrive": None}
    top_m = top_ft * FT
    for _ in range(int(round(t_s / dt))):
        m = _mach(ac)
        if phase == "dive":
            ac.cmd_alt_m = low_ft * FT
            if m >= trig:
                phase = "climb"
                out.update(t_supersonic=t, alt_supersonic_ft=ac.state.alt / FT, mach_at_trigger=m)
        if phase == "climb":
            ac.cmd_alt_m = min(top_m, ac.state.alt + (300.0 if m > climb_mach else 0.0))
            if ac.state.alt >= top_m - 1.0:
                phase = "hold"
                out.update(t_arrive=t, mach_arrive=m)
        if phase == "hold":
            ac.cmd_alt_m = top_m
        integrate_aircraft(ac, dt)
        t += dt
    out.update(final_mach=_mach(ac), final_alt_ft=ac.state.alt / FT, t_s=t)
    return out


def performance_table(alts_ft=(15_000.0, 25_000.0, 40_000.0)) -> List[dict]:
    rows = []
    for name, e in (("Blue", BLUE_ENERGY), ("Red", RED_ENERGY)):
        for h in alts_ft:
            g, gm = best_sustained_g(e, h)
            ts = top_speeds(e, h)
            rows.append({"type": name, "alt_ft": h, "sust_g": g, "sust_g_mach": gm,
                         "sust_g_m09": sustained_g(e, h, 0.9),
                         "lift_g_m09": lift_limit_g(e, h, 0.9), **ts})
    return rows


def format_performance(rows: List[dict]) -> str:
    f = lambda v: "  -  " if v is None else f"{v:5.3f}"
    lines = ["Spec 8b jet performance (1 g level unless noted, full thrust, Mach capped at max_mach)",
             f"{'type':5} {'alt ft':>7} {'best sust g':>11} {'@Mach':>6} {'sust g M0.9':>11} "
             f"{'lift g M0.9':>11} {'top from M0.9':>13} {'max sust M':>10} {'back side':>9}"]
    for r in rows:
        lines.append(f"{r['type']:5} {r['alt_ft']:7.0f} {r['sust_g']:11.2f} {r['sust_g_mach']:6.2f} "
                     f"{r['sust_g_m09']:11.2f} {r['lift_g_m09']:11.2f} {f(r['from_subsonic']):>13} "
                     f"{f(r['max_sustained']):>10} {f(r['back_side']):>9}")
    return "\n".join(lines) + "\n"


def format_bleed(rep: dict) -> str:
    lines = [f"Jet bleed: n_max={rep['n_max']}, lift-limit g at start={rep.get('n_lift0', float('nan')):.2f}, alt={rep['alt_ft']:.0f} ft, "
             f"start Mach {rep['start_mach']:.2f}, t={rep['t_s']:.0f} s",
             f"  final Mach {rep['final_mach']:.3f}",
             f"  {'t_s':>6} {'Mach':>7} {'V m/s':>8} {'alt m':>8}"]
    for s in rep["samples"]:
        lines.append(f"  {s['t']:6.1f} {s['mach']:7.3f} {s['speed_mps']:8.1f} {s['alt_m']:8.0f}")
    return "\n".join(lines) + "\n"


def gate_ok(rep: dict, lo: float = 0.60, hi: float = 0.80) -> bool:
    """A-d accept: final Mach in [lo, hi] (~0.7)."""
    return lo <= rep["final_mach"] <= hi


def calibration_report() -> str:
    """Spec 8b report: performance table plus the 40 kft calibration runs."""
    out = [format_performance(performance_table())]
    for name, e in (("Blue", BLUE_ENERGY), ("Red", RED_ENERGY)):
        b = bleed_run(energy=e)
        la = level_accel_run(e)
        hold = level_accel_run(e, start_mach=1.2)
        dc = dive_climb_run(e)
        out += [f"{name}: lift-limit g at 40 kft / M0.9 = {lift_limit_g(e, 40_000.0, 0.9):.2f}",
                f"  bleed gate (lift-limited turn, 20 s): M0.9 -> {b['final_mach']:.3f}",
                f"  level accel 40 kft from M0.9 (300 s): final {la['final_mach']:.3f}",
                f"  hold from M1.2 at 40 kft (300 s): final {hold['final_mach']:.3f}",
                f"  dive-climb: M{dc['mach_at_trigger']:.3f} at {dc['alt_supersonic_ft']:.0f} ft "
                f"after {dc['t_supersonic']:.0f} s; arrives 40 kft at t={dc['t_arrive']:.0f} s "
                f"M{dc['mach_arrive']:.3f}; final M{dc['final_mach']:.3f} at "
                f"{dc['final_alt_ft']:.0f} ft (t={dc['t_s']:.0f} s)", ""]
    return "\n".join(out) + "\n"
