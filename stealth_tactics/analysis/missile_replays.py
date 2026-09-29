"""Spec 3a TacView replays of single long shots (full sensors, real World).

Blue F-35 and Red Su-27 co-altitude at 40,000 ft, both Mach 0.9, starting
60 NM apart head-on. Blue fires ONE missile on its own fire-control track at
``shot_nm`` (default 45 NM, inside the head-on Rmax: ~49 NM before Spec 8 loft,
~82 NM with it) and keeps flying at
Red to support. Red (weapons off) reacts at launch:

- ``headon``: keeps coming (long head-on shot);
- ``drag``: 3 g level turn to cold (away from the launch point), commanded speed unchanged
  (Spec 8 energy model: the turn bleeds speed, thrust then recovers it);
- ``beam``: 3 g level turn to put the shooter on its beam (90 deg), holds it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.sim.aircraft import Aircraft, AircraftState, _wrap_pi, bearing_to
from stealth_tactics.sim.missile_kinematics import speed_of_sound
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.sim.sensors import distance_3d
from stealth_tactics.sim.weapons import OUTCOMES
from stealth_tactics.sim.world import SimConfig, World

FT = 0.3048
G0 = 9.80665
KINDS = {"headon": "Red keeps coming (hot)",
         "drag": "Red turns cold at launch (3 g, same commanded speed)",
         "beam": "Red turns to beam at launch (3 g) and holds it"}


@dataclass
class MissileReplay:
    kind: str
    seed: int
    world: World
    shot_nm: float
    fired_t: Optional[float] = None
    fired_range_nm: Optional[float] = None
    outcome: str = ""
    timeline: List[dict] = field(default_factory=list)
    samples: List[dict] = field(default_factory=list)


def run_missile_replay(kind: str = "headon", seed: int = 2, shot_nm: float = 45.0,
                       alt_ft: float = 40000.0, mach: float = 0.9, start_nm: float = 60.0,
                       record: bool = True, turn_g: float = 3.0) -> MissileReplay:
    alt = alt_ft * FT
    v = mach * speed_of_sound(alt)
    b1 = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, alt, 0.0, v))
    r1 = Aircraft.make_red("R1", "Red-1", AircraftState(0.0, start_nm * NM_M, alt, math.pi, v))
    r1.type_name = "Su-27"
    # 3 g level turn at this speed (Red's own 13 deg/s would be ~6 g)
    # Spec 8: the energy integrator ignores max_turn_rate_deg_s and caps the load
    # factor instead; a level turn with turn_g of horizontal acceleration needs
    # n = sqrt(1 + turn_g^2). (The turn now bleeds speed; thrust recovers it.)
    r1.params = dataclasses.replace(r1.params, max_turn_rate_deg_s=math.degrees(turn_g * G0 / v),
                                    max_speed_mps=max(v, r1.params.max_speed_mps),
                                    energy=dataclasses.replace(
                                        r1.params.energy, n_max=math.sqrt(1.0 + turn_g ** 2)))
    b1.params = dataclasses.replace(b1.params, max_speed_mps=max(v, b1.params.max_speed_mps))
    st: Dict = {"cold_hdg": None, "beam_hdg": None}
    rep_box: Dict = {}

    def blue_ctl(w: World) -> None:
        b1.cmd_heading_rad = bearing_to(b1.state, r1.state)
        b1.cmd_speed_mps, b1.cmd_alt_m = v, alt
        rng = distance_3d(b1.state, r1.state)
        if b1.ammo == b1.params.ammo and w.tracks["B1"].is_fire_control("R1") \
                and rng <= shot_nm * NM_M:
            b1.cmd_fire, b1.fire_target = True, "R1"
            st["fire_rng"] = rng
        m = w.missiles[0] if w.missiles else None
        if m is not None and (m.alive or not rep_box["samples"]
                              or rep_box["samples"][-1]["alive"]):
            if abs(w.time_s - round(w.time_s / 5.0) * 5.0) < 1e-6 or not m.alive:
                rep_box["samples"].append({
                    "t": w.time_s, "tof": m.pos.tf, "mach": m.mach, "alive": m.alive,
                    "m_r1_nm": distance_3d(m.pos, r1.state) / NM_M,
                    "b1_r1_nm": rng / NM_M})

    def red_ctl(w: World) -> None:
        r1.cmd_speed_mps, r1.cmd_alt_m = v, alt
        launched = bool(w.missiles)
        if not launched or kind == "headon":
            r1.cmd_heading_rad = bearing_to(r1.state, b1.state)
            return
        if st["cold_hdg"] is None:     # at launch: fix the escape heading
            away = bearing_to(b1.state, r1.state)
            st["cold_hdg"] = away
            st["beam_hdg"] = _wrap_pi(away - math.pi / 2)
        r1.cmd_heading_rad = st["cold_hdg"] if kind == "drag" else st["beam_hdg"]

    w = World([b1, r1], SimConfig(dt=0.5, max_time_s=300.0, seed=seed), blue_ctl, red_ctl,
              record=record)
    rep_box["samples"] = []
    # stop 20 s after the missile ends
    orig_run_end = {"t_end": None}

    def hook(world: World, frame: dict) -> None:
        if world.missiles and not world.missiles[0].alive and orig_run_end["t_end"] is None:
            orig_run_end["t_end"] = world.time_s
        if orig_run_end["t_end"] is not None and world.time_s >= orig_run_end["t_end"] + 20:
            world.config.max_time_s = world.time_s
    w.frame_hook = hook
    res = w.run()
    rep = MissileReplay(kind, seed, w, shot_nm, samples=rep_box["samples"])
    for e in res.events:
        if e["type"] == "launch":
            rep.fired_t = e["t"]
    if rep.fired_t is not None and st.get("fire_rng") is not None:
        rep.fired_range_nm = st["fire_rng"] / NM_M
    outs = [e["type"] for e in res.events if e["type"] in OUTCOMES]
    rep.outcome = outs[0] if outs else ("no_shot" if rep.fired_t is None else "unresolved")
    keep = {("B1", "radar_detect"), ("B1", "fc_track"), ("B1", "fc_lost")}
    seen = set()
    tl = []
    for e in res.sensor_events:
        k = (e["observer"], e["type"])
        if k in keep and (k not in seen or e["type"] == "fc_lost"):
            seen.add(k)
            tl.append(e)
    for e in res.events:
        if e["type"] in ("launch", "autonomous", "support_lost", "support_regained",
                         "support_dropped", "burnout") + OUTCOMES:
            tl.append(e)
    rep.timeline = sorted(tl, key=lambda e: e["t"])
    return rep


def describe(rep: MissileReplay) -> str:
    m = rep.world.missiles[0] if rep.world.missiles else None
    lines = [f"Missile replay '{rep.kind}' (seed {rep.seed}): {KINDS[rep.kind]} -> "
             f"outcome {rep.outcome.upper()}",
             f"  F-35 and Su-27 at 40,000 ft, Mach 0.9, head-on from 60 NM; F-35 fires on "
             f"own FC at <= {rep.shot_nm:.0f} NM and keeps supporting"]
    for e in rep.timeline:
        et = e["type"]
        if et == "launch":
            txt = f"{e['shooter']} LAUNCH {e['missile']} (own FC)"
            if rep.fired_range_nm is not None:
                txt += f"  [{rep.fired_range_nm:5.1f} NM]"
        else:
            txt = e.get("text", et)
        rng = e.get("range_m")
        rs = f"  [{rng / NM_M:5.1f} NM]" if rng is not None and et not in ("hit", "miss") else ""
        lines.append(f"  t={e['t']:6.1f}s  {txt}{rs}")
    if m is not None:
        lines.append(f"  missile: TOF {m.pos.tf:.1f} s, peak Mach {m.max_mach:.2f}, Mach at "
                     f"active {m.mach_at_active if m.mach_at_active is None else round(m.mach_at_active, 2)},"
                     f" Mach at end {m.mach_end if m.mach_end is None else round(m.mach_end, 2)},"
                     f" final Pk {m.pk_final if m.pk_final is None else round(m.pk_final, 2)}")
        lines.append("  samples (every 5 s): t / TOF / missile Mach / missile-Red NM / F-35-Red NM")
        for s in rep.samples:
            lines.append(f"    {s['t']:6.1f}s  {s['tof']:5.1f}s  M{s['mach']:.2f}  "
                         f"{s['m_r1_nm']:5.1f}  {s['b1_r1_nm']:5.1f}"
                         + ("" if s["alive"] else "  (end)"))
    return "\n".join(lines)


def export(rep: MissileReplay, path: Path, title: str) -> Path:
    return ACMIExporter(title=title).export(rep.world.frames, path)
