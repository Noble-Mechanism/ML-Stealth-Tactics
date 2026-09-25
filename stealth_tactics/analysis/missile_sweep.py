"""Spec 3a missile shot sweep and Rmax / Rne summary (``missile-sweep`` CLI).

Each shot runs through the sim's ``WeaponModel`` (same code path as engagements,
0.5 s world step with 0.05 s internal missile steps). Scripted 1v1, co-altitude:

- the shooter flies straight and level at its launch speed toward the target's
  launch position and keeps full midcourse support (truth fire-control track), so
  there are no support / coast Pk factors: final Pk = 0.60 x f_E;
- the target flies level at ``target_mach`` (0.9): ``hot`` (nose-on), ``beam``
  (90 deg), ``turncold`` (3 g level turn to cold starting at launch, no speed
  change) or ``cold`` (already flying away).

Per shot: outcome (hit / miss from the seeded Pk roll, or the defeat label),
time of flight, missile Mach at the active point (15 NM) and at the end, a-pole
(shooter-target range when the seeker goes active), f-pole (shooter-target range
at fuze), and final Pk. "Rmax" counts any fuzed shot (hit or miss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, AircraftState
from stealth_tactics.sim.missile_kinematics import speed_of_sound
from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG, NM_M, SensorConfig
from stealth_tactics.sim.weapons import WeaponModel

FT = 0.3048
G0 = 9.80665
BEHAVIORS = ("hot", "beam", "turncold", "cold")
BEHAVIOR_LABEL = {"hot": "hot (head-on)", "beam": "beam", "turncold": "turn cold at launch",
                  "cold": "already cold"}
_HDG = {"hot": math.pi, "beam": math.pi / 2, "turncold": math.pi, "cold": 0.0}
# table aspect / turn flag equivalent of each behavior
ENV_ARGS = {"hot": (0.0, False), "beam": (90.0, False), "turncold": (0.0, True),
            "cold": (180.0, False)}


@dataclass
class Shot:
    range_nm: float
    alt_ft: float
    shooter_mach: float
    behavior: str
    target_mach: float
    outcome: str = ""
    tof_s: float = 0.0
    mach_active: Optional[float] = None
    mach_end: Optional[float] = None
    max_mach: float = 0.0
    a_pole_nm: Optional[float] = None
    f_pole_nm: Optional[float] = None
    pk: Optional[float] = None
    frames: List[dict] = field(default_factory=list)

    @property
    def fuzed(self) -> bool:
        return self.outcome in ("hit", "miss")


def fly_shot(range_nm: float, alt_ft: float, shooter_mach: float, behavior: str,
             target_mach: float = 0.9, cfg: Optional[SensorConfig] = None,
             seed: int = 0, dt: float = 0.5, turn_g: float = 3.0,
             off_nose_deg: float = 0.0, aspect_deg: Optional[float] = None) -> Shot:
    """``off_nose_deg``: launch angle off the shooter's nose (signed, + = nose
    offset toward the side the target moves: East for beam, "lead"). With an
    off-nose launch the shooter points at the target after launch (keeps
    support inside 90 deg). ``aspect_deg`` overrides the behavior's target
    heading (0 = hot, 180 = cold; target moves East for 0 < aspect < 180)."""
    cfg = cfg or DEFAULT_SENSOR_CONFIG
    alt = alt_ft * FT
    a = speed_of_sound(alt)
    shooter = Aircraft.make_blue("S", "Shooter", AircraftState(
        0.0, 0.0, alt, math.radians(off_nose_deg), shooter_mach * a))
    pursue = off_nose_deg != 0.0
    hdg0 = _HDG[behavior] if aspect_deg is None else math.pi - math.radians(aspect_deg)
    target = Aircraft.make_red("T", "Target", AircraftState(0.0, range_nm * NM_M, alt,
                                                            hdg0, target_mach * a))
    wm = WeaponModel(np.random.default_rng(seed), cfg)
    m = wm.spawn(shooter, target)
    by_id = {"S": shooter, "T": target}
    tracks = {"S": {"T"}, "T": set()}
    shot = Shot(range_nm, alt_ft, shooter_mach, behavior, target_mach)
    omega = turn_g * G0 / (target_mach * a)
    t = 0.0
    if m.autonomous:
        shot.a_pole_nm = range_nm
    while m.alive:
        for ac in (shooter, target):
            st = ac.state
            if ac is target and behavior == "turncold" and st.heading_rad > 1e-12:
                st.heading_rad = max(0.0, st.heading_rad - omega * dt)
            if ac is shooter and pursue:
                st.heading_rad = math.atan2(target.state.x - st.x, target.state.y - st.y)
            st.x += st.speed_mps * math.sin(st.heading_rad) * dt
            st.y += st.speed_mps * math.cos(st.heading_rad) * dt
        was_active = m.autonomous
        wm.step([m], by_id, dt, tracks=tracks, t=t)
        t += dt
        sep = math.hypot(shooter.state.x - target.state.x, shooter.state.y - target.state.y)
        if not was_active and m.autonomous and shot.a_pole_nm is None:
            shot.a_pole_nm = sep / NM_M
        if m.outcome in ("hit", "miss"):
            shot.f_pole_nm = sep / NM_M
    shot.outcome = m.outcome
    shot.tof_s = m.pos.tf
    shot.mach_active = m.mach_at_active
    shot.mach_end = m.mach_end
    shot.max_mach = m.max_mach
    shot.pk = m.pk_final
    return shot


def sim_rmax_nm(alt_ft: float, shooter_mach: float, behavior: str, target_mach: float = 0.9,
                cfg: Optional[SensorConfig] = None, lo: float = 1.0, hi: float = 120.0,
                coarse: float = 2.5, tol: float = 0.05, off_nose_deg: float = 0.0,
                aspect_deg: Optional[float] = None) -> float:
    """Largest fuzing launch range (NM) through the sim path: coarse scan +
    bisection (same scheme as the envelope table)."""
    grid = list(np.arange(lo, hi + 1e-9, coarse))
    kw = dict(off_nose_deg=off_nose_deg, aspect_deg=aspect_deg)
    ok = [fly_shot(r, alt_ft, shooter_mach, behavior, target_mach, cfg, **kw).fuzed
          for r in grid]
    if not any(ok):
        return 0.0
    i = max(k for k, v in enumerate(ok) if v)
    if i == len(grid) - 1:
        return grid[-1]
    a, b = grid[i], grid[i + 1]
    while b - a > tol:
        mid = 0.5 * (a + b)
        if fly_shot(mid, alt_ft, shooter_mach, behavior, target_mach, cfg, **kw).fuzed:
            a = mid
        else:
            b = mid
    return a


SWEEP_RANGES = tuple(float(r) for r in range(10, 61, 5))
SWEEP_MACHS = (0.7, 0.9, 1.1, 1.3)
SWEEP_ALTS = (15000.0, 25000.0, 35000.0, 45000.0)


def run_sweep(ranges=SWEEP_RANGES, machs=SWEEP_MACHS, alts=SWEEP_ALTS,
              behaviors=BEHAVIORS, target_mach: float = 0.9,
              cfg: Optional[SensorConfig] = None) -> List[Shot]:
    shots = []
    k = 0
    for alt in alts:
        for ms in machs:
            for beh in behaviors:
                for r in ranges:
                    shots.append(fly_shot(r, alt, ms, beh, target_mach, cfg, seed=k))
                    k += 1
    return shots


def _f(v, fmt="{:.2f}", dash="-"):
    return dash if v is None else fmt.format(v)


def format_shots(shots: List[Shot]) -> str:
    hdr = (f"{'alt ft':>6} {'M_s':>4} {'target':<9} {'R NM':>5} {'outcome':<15} "
           f"{'TOF s':>6} {'M@act':>6} {'M@end':>6} {'Mmax':>5} {'a-pole':>6} "
           f"{'f-pole':>6} {'Pk':>5}")
    out = [hdr, "-" * len(hdr)]
    for s in shots:
        out.append(f"{s.alt_ft:6.0f} {s.shooter_mach:4.1f} {s.behavior:<9} {s.range_nm:5.1f} "
                   f"{s.outcome:<15} {s.tof_s:6.1f} {_f(s.mach_active):>6} "
                   f"{_f(s.mach_end):>6} {s.max_mach:5.2f} {_f(s.a_pole_nm, '{:.1f}'):>6} "
                   f"{_f(s.f_pole_nm, '{:.1f}'):>6} {_f(s.pk):>5}")
    return "\n".join(out)


def shots_csv(shots: List[Shot]) -> str:
    rows = ["alt_ft,shooter_mach,target_behavior,target_mach,range_nm,outcome,tof_s,"
            "mach_active,mach_end,max_mach,a_pole_nm,f_pole_nm,pk"]
    for s in shots:
        rows.append(",".join(str(x) for x in (
            s.alt_ft, s.shooter_mach, s.behavior, s.target_mach, s.range_nm, s.outcome,
            round(s.tof_s, 2), _f(s.mach_active, "{:.3f}", ""), _f(s.mach_end, "{:.3f}", ""),
            round(s.max_mach, 3), _f(s.a_pole_nm, "{:.2f}", ""),
            _f(s.f_pole_nm, "{:.2f}", ""), _f(s.pk, "{:.3f}", ""))))
    return "\n".join(rows) + "\n"


def summary_table(alts=SWEEP_ALTS, machs=SWEEP_MACHS, target_mach: float = 0.9,
                  cfg: Optional[SensorConfig] = None, extra_alts: Iterable[float] = (40000.0,)
                  ) -> Tuple[str, Dict]:
    """Rmax (hot / beam / cold) and Rne (turn cold at launch) in NM: sim-path
    bisection and the interpolated startup lookup table."""
    cfg = cfg or DEFAULT_SENSOR_CONFIG
    wm = WeaponModel(np.random.default_rng(0), cfg)
    env = wm.envelope
    all_alts = sorted(set(alts) | set(extra_alts))
    data: Dict = {}
    hdr = (f"{'alt ft':>6} {'M_s':>4} | " + " ".join(f"{b:>9}" for b in BEHAVIORS)
           + " | " + " ".join(f"{'tab ' + b[:5]:>10}" for b in BEHAVIORS))
    lines = [f"Rmax / Rne summary (NM), target Mach {target_mach}, co-altitude. Left: sim-path "
             "bisection (WeaponModel, 0.05 NM); right: startup lookup table (interpolated). "
             "turncold = Rne (no-escape: target turns cold at 3 g at launch).",
             hdr, "-" * len(hdr)]
    for alt in all_alts:
        for ms in machs:
            sim = {b: sim_rmax_nm(alt, ms, b, target_mach, cfg) for b in BEHAVIORS}
            tab = {}
            for b in BEHAVIORS:
                asp, tc = ENV_ARGS[b]
                f = env.rne_m if tc else env.rmax_m
                tab[b] = f(alt * FT, ms, asp, target_mach) / NM_M
            data[(alt, ms)] = {"sim": sim, "table": tab}
            lines.append(f"{alt:6.0f} {ms:4.1f} | " + " ".join(f"{sim[b]:9.1f}" for b in BEHAVIORS)
                         + " | " + " ".join(f"{tab[b]:10.1f}" for b in BEHAVIORS))
    return "\n".join(lines) + "\n", data


def sweep_findings(shots: List[Shot]) -> str:
    """Short text: where kinematic defeats start per (alt, M_s, behavior)."""
    out = ["Shortest non-fuzing range per case (first defeat, NM) and outcome label:"]
    keys = sorted({(s.alt_ft, s.shooter_mach, s.behavior) for s in shots},
                  key=lambda k: (k[0], k[1], BEHAVIORS.index(k[2])))
    for k in keys:
        ss = sorted([s for s in shots if (s.alt_ft, s.shooter_mach, s.behavior) == k],
                    key=lambda s: s.range_nm)
        first = next((s for s in ss if not s.fuzed), None)
        last_ok = max((s.range_nm for s in ss if s.fuzed), default=None)
        out.append(f"  {k[0]:6.0f} ft M{k[1]:.1f} {k[2]:<9} longest fuzed "
                   f"{_f(last_ok, '{:.0f}')} NM; first defeat "
                   + (f"{first.range_nm:.0f} NM ({first.outcome}, M_end {_f(first.mach_end)})"
                      if first else "none <= 60 NM"))
    counts: Dict[str, int] = {}
    for s in shots:
        counts[s.outcome] = counts.get(s.outcome, 0) + 1
    out.append("Outcome counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return "\n".join(out) + "\n"
