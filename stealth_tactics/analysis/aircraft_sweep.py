"""Spec 8 / 8b / 8c / 8d: jet energy calibration (bleed, lift limit, transonic hump, climb)."""

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
        back = float(ms[i]) if i > 0 else None   # contiguous from M0.9: no gap
    return {"from_subsonic": from_sub, "max_sustained": top, "back_side": back}


def level_accel_run(e: AircraftEnergyConfig, alt_ft: float = 40_000.0,
                    start_mach: float = 0.9, t_s: float = 300.0, dt: float = 0.05,
                    target_mach: Optional[float] = None) -> dict:
    """Level, full-thrust acceleration run; returns final / peak Mach and the
    time to reach ``target_mach`` (default max_mach - 0.005; None if never)."""
    ac = _make_jet(e, alt_ft * FT, start_mach)
    ac.cmd_speed_mps = 1.0e4
    tgt = (e.max_mach - 0.005) if target_mach is None else target_mach
    peak, t_hit = start_mach, None
    for i in range(int(round(t_s / dt))):
        integrate_aircraft(ac, dt)
        m = _mach(ac)
        peak = max(peak, m)
        if t_hit is None and m >= tgt:
            t_hit = (i + 1) * dt
    return {"alt_ft": alt_ft, "start_mach": start_mach, "t_s": t_s, "target_mach": tgt,
            "t_to_target": t_hit, "final_mach": _mach(ac), "peak_mach": peak,
            "final_alt_ft": ac.state.alt / FT}


def accel_time_s(e: AircraftEnergyConfig, alt_ft: float = 40_000.0, m0: float = 0.9,
                 m1: float = 1.2, t_max: float = 900.0) -> Optional[float]:
    """Simulated level full-thrust time from m0 to m1 (None if not reached in t_max).

    The implicit throttle holds ~0.5 m/s under the Mach ceiling, so a target at
    the ceiling counts as reached at max_mach - 0.005."""
    tgt = min(m1 - 0.001, e.max_mach - 0.005)
    return level_accel_run(e, alt_ft, m0, t_s=t_max, target_mach=tgt)["t_to_target"]


# ------------------------------------------------------ Spec 8c analysis ---
def specific_excess_power(e: AircraftEnergyConfig, alt_ft: float, mach: float,
                          n: float = 1.0) -> float:
    """Ps = (T - D) V / W (m/s) at full thrust and load factor n."""
    a = atmosphere(alt_ft * FT)[1]
    return level_excess_n(e, alt_ft, mach, n) * mach * a / (e.mass_kg * G0)


def gamma_climb_margin(e: AircraftEnergyConfig, alt_ft: float = 35_000.0, mach: float = 1.0,
                       gamma_deg: float = 15.0) -> float:
    """(T - D) / (W sin gamma) in a steady climb at flight-path angle gamma
    (n = cos gamma). >= 1 means the speed is non-decreasing."""
    g = math.radians(gamma_deg)
    return level_excess_n(e, alt_ft, mach, math.cos(g)) / (e.mass_kg * G0 * math.sin(g))


def gamma_climb_run(e: AircraftEnergyConfig, alt_ft: float = 35_000.0, mach: float = 1.0,
                    gamma_deg: float = 15.0, t_s: float = 5.0, dt: float = 0.05,
                    start_gamma_deg: Optional[float] = None) -> dict:
    """Integrator check: full thrust, climb at a fixed flight-path angle."""
    ac = _make_jet(e, alt_ft * FT, mach)
    ac.cmd_speed_mps = 1.0e4
    sg = math.sin(math.radians(gamma_deg))
    # Spec 8d: start already established on the climb angle (steady-climb
    # check); pass start_gamma_deg=0 to include the pull-up from level.
    ac.gamma_rad = math.radians(gamma_deg if start_gamma_deg is None else start_gamma_deg)
    for _ in range(int(round(t_s / dt))):
        # Spec 8d: direct climb-rate command (gamma is a rate-limited state)
        ac.cmd_climb_rate_mps = ac.state.speed_mps * sg
        integrate_aircraft(ac, dt)
    return {"alt_ft": alt_ft, "mach0": mach, "gamma_deg": gamma_deg, "t_s": t_s,
            "final_mach": _mach(ac), "final_alt_ft": ac.state.alt / FT,
            "climb_fpm": ac.state.speed_mps * math.sin(ac.gamma_rad) / FT * 60.0,
            "final_gamma_deg": math.degrees(ac.gamma_rad)}


def climb_run(e: AircraftEnergyConfig, schedule: str = "max_rate", start_ft: float = 30_000.0,
              end_ft: float = 40_000.0, start_mach: float = 1.0, gamma_deg: float = 15.0,
              t_max: float = 600.0, dt: float = 0.05) -> dict:
    """Full-thrust climb from start_ft to end_ft. Schedules:

    - ``max_rate``: climb-rate command = the type's max climb rate (90 m/s
      Blue, 112.5 m/s Red), speed floats; time = first crossing of end_ft.
    - ``alt_hold``: command cmd_alt = end_ft (Spec 8d altitude hold, which
      levels off without overshoot); time = within 30 m of end_ft.
    - ``gamma``: constant flight-path angle ``gamma_deg``.
    - ``mach_hold``: climb at the rate that holds the start Mach (climb rate =
      Ps), capped at the max climb rate.
    """
    ac = _make_jet(e, start_ft * FT, start_mach)
    ac.cmd_speed_mps = 1.0e4
    top = end_ft * FT
    t, t_hit, m_min = 0.0, None, start_mach
    sg = math.sin(math.radians(gamma_deg))
    a0 = atmosphere(start_ft * FT)[1]
    while t < t_max:
        # Spec 8d: schedules fly a direct climb-rate command (gamma follows
        # at the rate the load factor allows); alt_hold uses cmd_alt.
        if schedule == "max_rate":
            ac.cmd_climb_rate_mps = ac.params.max_climb_rate_mps
        elif schedule == "alt_hold":
            ac.cmd_alt_m = top
        elif schedule == "gamma":
            ac.cmd_climb_rate_mps = ac.state.speed_mps * sg
        elif schedule == "mach_hold":
            ps = specific_excess_power(e, ac.state.alt / FT, _mach(ac))
            # climb at Ps, plus/minus a correction toward the start Mach
            err = (_mach(ac) - start_mach) * atmosphere(ac.state.alt)[1]
            ac.cmd_climb_rate_mps = max(0.0, ps + 2.0 * err)
        else:
            raise ValueError(schedule)
        integrate_aircraft(ac, dt)
        t += dt
        m_min = min(m_min, _mach(ac))
        tol = 30.0 if schedule == "alt_hold" else 1.0
        if ac.state.alt >= top - tol:
            t_hit = t
            break
    return {"schedule": schedule, "start_ft": start_ft, "end_ft": end_ft,
            "start_mach": start_mach, "t_s": t_hit, "end_mach": _mach(ac), "min_mach": m_min,
            "gamma_deg": gamma_deg if schedule == "gamma" else None}


RATIO_CONDITIONS = tuple((h, m) for h in (15_000.0, 25_000.0, 35_000.0, 40_000.0)
                         for m in (0.7, 0.8, 0.9))


def red_blue_ratios(conds=RATIO_CONDITIONS) -> List[dict]:
    """Red / Blue ratios of 1 g specific excess power, sustained g and max g."""
    out = []
    for h, m in conds:
        out.append({"alt_ft": h, "mach": m,
                    "ps_blue": specific_excess_power(BLUE_ENERGY, h, m),
                    "ps_red": specific_excess_power(RED_ENERGY, h, m),
                    "ps_ratio": specific_excess_power(RED_ENERGY, h, m)
                    / specific_excess_power(BLUE_ENERGY, h, m),
                    "sust_g_ratio": sustained_g(RED_ENERGY, h, m) / sustained_g(BLUE_ENERGY, h, m),
                    "max_g_ratio": lift_limit_g(RED_ENERGY, h, m) / lift_limit_g(BLUE_ENERGY, h, m)})
    return out


def calibration_8c() -> dict:
    """Spec 8c calibration numbers for Blue and Red."""
    res = {}
    for name, e in (("Blue", BLUE_ENERGY), ("Red", RED_ENERGY)):
        r = {"gamma_margin": gamma_climb_margin(e),
             "gamma_run": gamma_climb_run(e),
             "gamma_run_from_level": gamma_climb_run(e, start_gamma_deg=0.0),
             "climb_max_rate": climb_run(e, "max_rate"),
             "climb_gamma15": climb_run(e, "gamma", gamma_deg=15.0),
             "climb_mach_hold": climb_run(e, "mach_hold"),
             "climb_plain_m09": climb_run(e, "max_rate", start_mach=0.9),
             "climb_alt_hold": climb_run(e, "alt_hold"),
             "accel_40k_12": accel_time_s(e, 40_000.0, 0.9, 1.2),
             "bleed": bleed_run(energy=e)}
        if e.max_mach > 1.2:
            r["accel_40k_max"] = accel_time_s(e, 40_000.0, 0.9, e.max_mach)
        res[name] = r
    return res


def format_calibration_8c(res: dict) -> str:
    f = lambda v, fmt="{:.1f}": "never" if v is None else fmt.format(v)
    lines = ["Spec 8c calibration (full thrust; times in s)"]
    for name, r in res.items():
        b = r["bleed"]
        lines += [
            f"{name}:",
            f"  35 kft M1.0 gamma 15 deg: (T-D)/(W sin g) = {r['gamma_margin']:.3f}; "
            f"5 s run M1.000 -> {r['gamma_run']['final_mach']:.3f} "
            f"({r['gamma_run']['climb_fpm']:.0f} ft/min); from level (incl. pull-up) "
            f"-> M{r['gamma_run_from_level']['final_mach']:.3f}",
            f"  30->40 kft from M1.0, max-rate climb: {f(r['climb_max_rate']['t_s'])} s "
            f"(end M{r['climb_max_rate']['end_mach']:.3f}, min M{r['climb_max_rate']['min_mach']:.3f})",
            f"  30->40 kft from M1.0, gamma 15 deg:  {f(r['climb_gamma15']['t_s'])} s "
            f"(end M{r['climb_gamma15']['end_mach']:.3f})",
            f"  30->40 kft from M1.0, Mach hold:     {f(r['climb_mach_hold']['t_s'])} s "
            f"(end M{r['climb_mach_hold']['end_mach']:.3f})",
            f"  30->40 kft from M1.0, alt hold (cmd_alt 40 kft, within 30 m): "
            f"{f(r['climb_alt_hold']['t_s'])} s (end M{r['climb_alt_hold']['end_mach']:.3f})",
            f"  30->40 kft from M0.9, plain full power (max-rate): "
            f"{f(r['climb_plain_m09']['t_s'])} s (end M{r['climb_plain_m09']['end_mach']:.3f}, "
            f"min M{r['climb_plain_m09']['min_mach']:.3f})",
            f"  40 kft level accel M0.9 -> 1.2: {f(r['accel_40k_12'])} s"]
        if "accel_40k_max" in r:
            lines.append(f"  40 kft level accel M0.9 -> {BLUE_ENERGY.max_mach if name == 'Blue' else RED_ENERGY.max_mach:.1f}: "
                         f"{f(r['accel_40k_max'])} s")
        lines.append(f"  bleed (lift-limited turn, 40 kft, 20 s): M0.9 -> {b['final_mach']:.3f} "
                     f"(start {b['n_lift0']:.2f} g)")
    lines += ["", "Red / Blue ratios (1 g Ps, sustained g, max g)",
              f"  {'alt ft':>7} {'Mach':>5} {'Ps B':>6} {'Ps R':>6} {'Ps R/B':>7} {'sust g R/B':>10} {'max g R/B':>9}"]
    for q in red_blue_ratios():
        lines.append(f"  {q['alt_ft']:7.0f} {q['mach']:5.2f} {q['ps_blue']:6.1f} {q['ps_red']:6.1f} "
                     f"{q['ps_ratio']:7.3f} {q['sust_g_ratio']:10.3f} {q['max_g_ratio']:9.3f}")
    return "\n".join(lines) + "\n"


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
            # Spec 8d: energy-managed Mach-hold climb (climb rate = Ps plus a
            # correction toward climb_mach). The 8c bang-bang schedule (climb
            # only while M > climb_mach) now pays for every pitch change.
            ps = specific_excess_power(e, ac.state.alt / FT, m)
            a_s = atmosphere(ac.state.alt)[1]
            ac.cmd_climb_rate_mps = max(0.0, ps + 2.0 * (m - climb_mach) * a_s)
            if ac.state.alt >= top_m - 30.0:
                ac.cmd_climb_rate_mps = None
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


def bleeds_ok(rep: dict, min_loss_mach: float = 0.08) -> bool:
    """Spec 8c bleed check: the lift-limited turn still clearly loses speed
    (at least ``min_loss_mach`` in the run). Replaces the Spec 8 0.60-0.80 gate
    as the pass criterion; more thrust may leave the end Mach above 0.8."""
    return rep["final_mach"] <= rep["start_mach"] - min_loss_mach


def gate_ok(rep: dict, lo: float = 0.60, hi: float = 0.80) -> bool:
    """A-d accept: final Mach in [lo, hi] (~0.7)."""
    return lo <= rep["final_mach"] <= hi


def calibration_report() -> str:
    """Spec 8b/8c report: performance table plus the 8c calibration runs."""
    return format_performance(performance_table()) + "\n" + format_calibration_8c(calibration_8c())
