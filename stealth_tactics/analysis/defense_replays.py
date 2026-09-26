"""Spec 3 TacView replays A-F (``python -m stealth_tactics defense-replays``).

Scripted 1v1 / 1v2 set-ups on the real World (full sensors, datalink, RWR modes,
3a missile) with the Red defense state machine. Blue is a simple scripted
shooter (``BlueShooter``); replay E wraps it in the Blue scripted test reaction.

A  conservative Red (a = 0) drags on the first lock (dive toward 100 m AGL),
   clears, stays cold 40 s, recommits.
B  aggressive Red (a = 1) fires, then cranks 50 deg on missile active while
   still supporting its own shot.
C  middle Red (a = 0.5) beams on the support cue; its own missile coasts.
D  turn-away limit: two cycles, then a = 0.8 presses (D1) and a = 0.2 departs (D2).
   D1/D2 are pinned to DefenseConfig.spec3() (the retired spec 3 limit behaviour);
   since the approved change of 2026-09-26 the default is one reaction, then press.
E  Blue test reaction against a Red shot (crank while supporting, then drag).
F  shoot-shoot-assess: two missiles at one target 3 s apart, then nothing at
   anyone until both are resolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.sim.aircraft import (Aircraft, AircraftState, Coalition, _wrap_pi,
                                          bearing_to)
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.sim.sensors import distance_3d
from stealth_tactics.sim.weapons import OUTCOMES
from stealth_tactics.sim.world import (SimConfig, World, SHOOT_ASSESS_SHOOT,
                                       SHOOT_SHOOT_ASSESS, LEGACY)
from stealth_tactics.tactics.interpreter import RedCAPController
from stealth_tactics.tactics.red_defense import DefenseConfig, RedDefense
from .blue_test_defense import BlueTestDefense  # noqa: F401  (re-export)

KINDS = {
    "A": "conservative Red (a=0) drags on first lock, clears, cold 40 s, recommits",
    "B": "aggressive Red (a=1) fires, then cranks on missile active while supporting",
    "C": "middle Red (a=0.5) beams on the support cue; its own missile coasts",
    "D1": "spec 3 turn-away limit (DefenseConfig.spec3): two cycles, then a=0.8 presses",
    "D2": "spec 3 turn-away limit (DefenseConfig.spec3): two cycles, then a=0.2 departs",
    "E": "Blue scripted test reaction vs a Red shot (crank while supporting, then drag)",
    "F": "Blue shoot-shoot-assess (per contact): pair at R1, then a pair at R2 at once",
}
# Default seeds (chosen so each replay shows its behaviour; the Red / Blue Pk
# rolls decide who survives long enough -- see describe()).
SEEDS = {"A": 1, "B": 4, "C": 4, "D1": 7, "D2": 1, "E": 1, "F": 2}
FILES = {"A": "replayA_conservative_drag.txt.acmi", "B": "replayB_aggressive_crank.txt.acmi",
         "C": "replayC_middle_beam.txt.acmi", "D1": "replayD1_limit_press.txt.acmi",
         "D2": "replayD2_limit_depart.txt.acmi", "E": "replayE_blue_test_reaction.txt.acmi",
         "F": "replayF_shoot_shoot_assess.txt.acmi"}


class BlueShooter:
    """Scripted Blue: fly at the nearest live Red keeping it ``offset_deg`` off
    the nose (0 = pure pursuit), fire per ``fire`` policy:
    'never' | 'range' (FC and range <= fire_nm, or <= shot_frac x Rmax if
    fire_nm is None) | 'reactive' (as 'range', but only after a Red launch)."""

    def __init__(self, blue_ids, fire: str = "range", fire_nm: Optional[float] = None,
                 shot_frac: float = 0.85, offset_deg: float = 0.0, speed: float = 260.0,
                 alt: Optional[float] = None, max_shots: int = 4,
                 refire_delay_s: float = 0.0) -> None:
        self.blue_ids = list(blue_ids)
        self.fire, self.fire_nm, self.shot_frac = fire, fire_nm, shot_frac
        self.offset = math.radians(offset_deg)
        self.speed, self.alt, self.max_shots = speed, alt, max_shots
        self.refire_delay_s = refire_delay_s

    def __call__(self, w: World) -> None:
        reds = w.alive(Coalition.RED)
        for uid in self.blue_ids:
            ac = w.get(uid)
            if ac is None or not ac.state.alive or not reds:
                continue
            tgt = min(reds, key=lambda r: distance_3d(ac.state, r.state))
            brg = bearing_to(ac.state, tgt.state)
            ac.cmd_heading_rad = _wrap_pi(brg + self.offset)
            ac.cmd_speed_mps = self.speed
            ac.cmd_alt_m = self.alt if self.alt is not None else ac.state.alt
            if self.fire == "never" or ac.params.ammo - ac.ammo >= self.max_shots:
                continue
            if self.fire == "reactive" and not any(m.coalition == "Red" for m in w.missiles):
                continue
            own = [m for m in w.missiles if m.shooter_id == uid]
            if self.refire_delay_s > 0 and own:
                if any(m.alive for m in own):
                    continue
                if w.time_s < max(m.end_t or 0.0 for m in own) + self.refire_delay_s:
                    continue
            cands = [tgt]
            if (w.doctrine_of(ac) != LEGACY
                    and not w.may_fire_at(ac, tgt.id)):
                # per-contact doctrine: nearest contact busy -> next nearest
                cands = [r for r in sorted(reds, key=lambda r: distance_3d(ac.state, r.state))
                         if r.id != tgt.id and w.may_fire_at(ac, r.id)]
            for c in cands:
                d = distance_3d(ac.state, c.state)
                lim = (self.fire_nm * NM_M if self.fire_nm is not None
                       else self.shot_frac * w.weapons.rmax_m(ac, c))
                if d <= lim and w.fire_control_source(ac, c.id) is not None:
                    ac.cmd_fire, ac.fire_target = True, c.id
                    break


@dataclass
class DefenseReplay:
    kind: str
    seed: int
    world: World
    result: object
    samples: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _jets(blue_alt: float, red_alt: float, start_nm: float, n_red: int = 1,
          red_ammo: Optional[int] = None, red_spacing_m: float = 6000.0):
    b1 = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, blue_alt, 0.0, 260.0))
    reds = []
    for i in range(n_red):
        x = (i - (n_red - 1) / 2.0) * red_spacing_m
        r = Aircraft.make_red(f"R{i + 1}", f"RedFighter-{i + 1}",
                              AircraftState(x, start_nm * NM_M, red_alt, math.pi, 255.0))
        if red_ammo is not None:
            r.ammo = red_ammo
        reds.append(r)
    return b1, reds


def run_defense_replay(kind: str, seed: int = 1, record: bool = True,
                       max_time_s: Optional[float] = None) -> DefenseReplay:
    kind = kind.upper()
    blue_doc = SHOOT_ASSESS_SHOOT
    n_red, red_ammo, a, defense_on = 1, None, 0.5, True
    blue_alt, red_alt, start_nm, T = 9200.0, 8800.0, 60.0, 240.0
    wrap_test = False
    if kind == "A":
        a, T = 0.0, 260.0
        ctl = dict(fire="range", shot_frac=0.75)
    elif kind == "B":
        a = 1.0
        ctl = dict(fire="reactive", fire_nm=40.0, offset_deg=58.0)
    elif kind == "C":
        a = 0.5
        ctl = dict(fire="reactive", fire_nm=40.0, offset_deg=58.0)
    elif kind == "D1":
        a, T = 0.8, 400.0
        ctl = dict(fire="range", shot_frac=0.85, max_shots=4, refire_delay_s=30.0)
    elif kind == "D2":
        a, T = 0.2, 400.0
        ctl = dict(fire="never")
    elif kind == "E":
        defense_on, T = False, 260.0
        ctl = dict(fire="range", fire_nm=21.5, offset_deg=58.0)
        wrap_test = True
    elif kind == "F":
        defense_on, n_red, red_ammo, T = False, 2, 0, 200.0
        blue_doc = SHOOT_SHOOT_ASSESS
        ctl = dict(fire="range", shot_frac=0.85)
    else:
        raise ValueError(kind)
    dcfg = DefenseConfig.spec3() if kind in ("D1", "D2") else None
    b1, reds = _jets(blue_alt, red_alt, start_nm, n_red, red_ammo)
    blue = BlueShooter(["B1"], **ctl)
    blue_ctl = BlueTestDefense(blue) if wrap_test else blue
    red_ctl = RedCAPController([r.id for r in reds], mode="intercept",
                               defense=RedDefense(a, dcfg) if defense_on else None)
    cfg = SimConfig(dt=0.5, max_time_s=max_time_s or T, seed=seed, blue_doctrine=blue_doc)
    w = World([b1] + reds, cfg, blue_ctl, red_ctl, record=record)
    samples: List[dict] = []

    def hook(world: World, frame: dict) -> None:
        if abs(world.time_s - round(world.time_s / 5.0) * 5.0) < 1e-6:
            row = {"t": world.time_s}
            for ac in world.aircraft:
                row[ac.id] = (ac.state.alt, math.degrees(ac.state.heading_rad) % 360,
                              ac.state.speed_mps, ac.defense_state, ac.state.alive)
            row["range_nm"] = {r.id: distance_3d(b1.state, r.state) / NM_M for r in reds}
            samples.append(row)
    valid_t: Dict[str, List[float]] = {}

    def hook_f(world: World, frame: dict) -> None:
        hook(world, frame)
        # F: times B1 had a valid shot (own FC, <= 0.85 x Rmax, launch gates) on
        # a Red it had no missile at, while other own missiles were in flight
        live = [m for m in world.missiles if m.alive and m.shooter_id == "B1"]
        if not live or not b1.state.alive:
            return
        busy = {m.target_id for m in live}
        for r in reds:
            if r.id in busy or not r.state.alive:
                continue
            if (world.fire_control_source(b1, r.id) is not None
                    and distance_3d(b1.state, r.state) <= 0.85 * world.weapons.rmax_m(b1, r)
                    and world.weapons.can_shoot(b1, r)):
                valid_t.setdefault(r.id, []).append(world.time_s)
    w.frame_hook = hook_f if kind == "F" else hook
    res = w.run()
    rep = DefenseReplay(kind, seed, w, res, samples)
    if kind == "F":
        launches = [e for e in res.events if e["type"] == "launch" and e["shooter"] == "B1"]
        for rid, ts in sorted(valid_t.items()):
            first_launch = min((e["t"] for e in launches if e["target"] == rid), default=None)
            held = [t for t in ts if first_launch is None or t < first_launch]
            if first_launch is None:
                rep.notes.append(f"  {rid}: valid shot from t={ts[0]:.1f} s while other missiles "
                                 f"were in flight, never fired at")
            else:
                rep.notes.append(
                    f"  {rid}: first valid shot while other missiles in flight t={ts[0]:.1f} s, "
                    f"first launch at it t={first_launch:.1f} s (held {len(held) * world_dt(w):.1f} s"
                    f" by doctrine = rest of the open pair at the other contact)")
    return rep


def world_dt(w: World) -> float:
    return w.config.dt


    return rep


# ------------------------------------------------------------------ report --
LOG_TYPES = ("launch", "launch_remote", "autonomous", "support_lost", "support_regained",
             "support_dropped", "defend", "threat_cleared", "recommit", "press", "depart",
             "blue_defend", "blue_recommit") + OUTCOMES


def _nm(v):
    return "-" if v is None else f"{v / NM_M:.1f}"


def describe(rep: DefenseReplay) -> str:
    w, res = rep.world, rep.result
    lines = [f"Replay {rep.kind} (seed {rep.seed}): {KINDS[rep.kind]}",
             f"  end t={res.time_s:.1f}s  blue kills {res.blue_kills}  red kills "
             f"{res.red_kills}  blue shots {res.blue_shots}  red shots {res.red_shots}"
             + ("  (ended early: all Red departed)" if res.ended_early else "")]
    rwr = [e for e in w.rwr_events if e["observer"].startswith("R") or rep.kind == "E"]
    evs = sorted([e for e in res.events if e["type"] in LOG_TYPES]
                 + [e for e in rwr if e.get("mode") in ("lock", "support", "missile_active")
                    or e.get("old") in ("lock", "support", "missile_active")],
                 key=lambda e: e["t"])
    last_rwr = {}
    for e in evs:
        if e["type"] == "rwr_mode":     # de-flicker: skip repeats within 3 s
            k = (e["observer"], e["emitter"], e.get("mode"))
            if k in last_rwr and e["t"] - last_rwr[k] < 3.0:
                continue
            last_rwr[k] = e["t"]
        if e["type"] == "launch":
            sh, tg = w.get(e["shooter"]), w.get(e["target"])
            txt = f"{e['shooter']} LAUNCH {e['missile']} at {e['target']}"
        else:
            txt = e.get("text", e["type"])
        rng = e.get("range_m")
        rs = f"  [{rng / NM_M:5.1f} NM]" if rng is not None and e["type"] not in (
            "hit", "miss") else ""
        lines.append(f"  t={e['t']:6.1f}s  {txt}{rs}")
    lines.append("  shots (range NM: launch / a-pole / f-pole; off-nose deg; outcome; Mach "
                 "end; Pk; target state launch -> end):")
    for s in res.shots:
        lines.append(f"    {s['missile']} {s['shooter']}->{s['target']} t={s['launch_t']:.1f}s "
                     f"{_nm(s['launch_range_m'])} / {_nm(s['a_pole_m'])} / {_nm(s['f_pole_m'])}"
                     f"  off {s['off_nose_deg']:.0f}  {s['outcome']}  M"
                     f"{'-' if s['mach_end'] is None else format(s['mach_end'], '.2f')}  Pk "
                     f"{'-' if s['pk'] is None else format(s['pk'], '.2f')}  "
                     f"{s['target_state_launch']} -> {s['target_state_end']}")
    lines.append("  samples every 10 s: t | per jet alt m / hdg / speed m/s / state | range NM")
    for row in rep.samples[::2]:
        parts = []
        for ac in w.aircraft:
            alt, hdg, spd, stt, alive = row[ac.id]
            parts.append(f"{ac.id} {alt:5.0f} {hdg:3.0f} {spd:3.0f} {stt[:4]}"
                         if alive else f"{ac.id} dead")
        rn = " ".join(f"{k}:{v:4.1f}" for k, v in row["range_nm"].items())
        lines.append(f"    {row['t']:6.1f}s  " + " | ".join(parts) + f" | {rn}")
    lines += rep.notes
    return "\n".join(lines)


def export(rep: DefenseReplay, path: Path, title: str) -> Path:
    return ACMIExporter(title=title).export(rep.world.frames, path)
