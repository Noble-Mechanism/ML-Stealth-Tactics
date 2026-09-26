"""Spec 4 presentation Red controller and pre-planned maneuver runner
(docs/specs/04-red-presentations.md, decisions E, H, I, J, L).

- **E**: wingmen hold their formation slot on the flight leader (lowest live
  slot not broken up) and every jet flies its assigned altitude ``hot_alt_m``
  (start altitude, changed only by an altitude maneuver), not Blue's altitude.
  Break-up per jet at the first of: its pre-planned maneuver starts, it starts
  defending, it gets its own FC track on a Blue, any Blue inside
  ``breakup_range_nm``. After break-up: individual pure pursuit of the nearest
  Blue (true position = perfect GCI, Rusty Q2) at ``hot_alt_m``.
- **H/I**: exactly one pre-planned maneuver per presentation, fired once when
  the smallest true range from any live non-departed Red to any live Blue
  reaches the trigger. Behaviour comes from a registered *kind*
  (``@register_kind``); the menu entries themselves are YAML data.
- **J**: the defense state machine always wins. A jet not HOT at trigger time
  skips the maneuver; a jet that starts defending during it aborts it for good.
  Maneuvers never count as turn-aways and never set ``departed``. The pump
  (kind ``cold``) cannot fire; the other kinds use the normal launch gates.
- **L**: a Red jet with no missiles and none of its own in flight departs
  (D9 departure, not a turn-away).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, Coalition, bearing_to, distance_3d
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.tactics import maneuvers as mv
from stealth_tactics.tactics.interpreter import RedCAPController
from stealth_tactics.tactics.red_defense import HOT, DEPARTED, PRESSING

PENDING, ACTIVE, DONE, SKIPPED, ABORTED = "pending", "active", "done", "skipped", "aborted"
ALT_TOL_M = 50.0
ALT_MANEUVER_MAX_S = 180.0
STATION_LOOKAHEAD_M = 3000.0
STATION_GAIN = 0.03           # m/s of speed per m of along-track slot error


@dataclass
class JetPlan:
    slot: int
    hot_alt_m: float
    right_nm: float
    back_nm: float
    half: int = 0
    broken: bool = False
    status: str = PENDING
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    side: int = -1
    target_alt: Optional[float] = None


@dataclass
class ManeuverKind:
    name: str
    start: Callable        # (ctrl, world, ac, jp, params) -> None
    step: Callable         # (ctrl, world, ac, jp, params, tgt) -> (ManeuverCmd, may_shoot, done)


KINDS: Dict[str, ManeuverKind] = {}


def register_kind(name: str):
    """Register a maneuver behaviour: decorate a class/namespace with static
    ``start`` and ``step``. New menu entries using it are pure YAML."""
    def deco(cls):
        KINDS[name] = ManeuverKind(name, cls.start, cls.step)
        return cls
    return deco


@register_kind("turn_out")
class _TurnOut:
    """Split: half 0 turns left, half 1 right, ``angle_deg`` off the bearing to
    the nearest Blue, 1.1 x cruise at hot_alt, for ``leg_s``."""
    @staticmethod
    def start(ctrl, world, ac, jp, params):
        jp.side = -1 if jp.half == 0 else 1

    @staticmethod
    def step(ctrl, world, ac, jp, params, tgt):
        c = mv.crank(ac, bearing_to(ac.state, tgt.state), params["angle_deg"], side=jp.side)
        cmd = mv.ManeuverCmd(c.heading, c.speed, jp.hot_alt_m)
        return cmd, True, world.time_s - jp.t_start >= params["leg_s"] - 1e-9


@register_kind("cold")
class _Cold:
    """Pump: threat on the tail, max speed, altitude held, for ``leg_s``; no
    shots (support drops like a spec 3 drag)."""
    @staticmethod
    def start(ctrl, world, ac, jp, params):
        pass

    @staticmethod
    def step(ctrl, world, ac, jp, params, tgt):
        cmd = mv.drag(ac, bearing_to(ac.state, tgt.state), target_alt=jp.hot_alt_m)
        return cmd, False, world.time_s - jp.t_start >= params["leg_s"] - 1e-9


@register_kind("altitude")
class _Altitude:
    """Low-high split (divide: halves; half ``low_half`` goes to ``low_alt_m``,
    the other climbs ``high_delta_m``) or flight altitude change (divide: all;
    ``direction`` low -> ``low_alt_m``, high -> ``high_alt_m``). Stays hot; the
    new altitude persists as hot_alt_m. Done when reached (or 180 s)."""
    @staticmethod
    def start(ctrl, world, ac, jp, params):
        lo, hi = ctrl.cfg.alt_clip_m
        if "direction" in params:
            tgt = params["low_alt_m"] if params["direction"] == "low" else params["high_alt_m"]
        elif jp.half == int(params.get("low_half", 0)):
            tgt = params["low_alt_m"]
        else:
            tgt = jp.hot_alt_m + params["high_delta_m"]
        jp.target_alt = float(min(hi, max(lo, tgt)))
        jp.hot_alt_m = jp.target_alt

    @staticmethod
    def step(ctrl, world, ac, jp, params, tgt):
        cmd = mv.hot(ac, bearing_to(ac.state, tgt.state), alt=jp.target_alt)
        done = (abs(ac.state.alt - jp.target_alt) <= ALT_TOL_M
                or world.time_s - jp.t_start >= ALT_MANEUVER_MAX_S - 1e-9)
        return cmd, True, done


class PresentationRedController(RedCAPController):
    """Red controller for a spec 4 presentation (formation, assigned altitude,
    one pre-planned maneuver, Winchester departure) on top of ``RedDefense``."""

    def __init__(self, presentation, defense=None, cfg=None) -> None:
        from stealth_tactics.scenarios.presentation import DEFAULT_PRESENTATION_CONFIG
        p = presentation
        super().__init__([r["id"] for r in p.red_jets], mode="intercept", defense=defense)
        self.p = p
        self.cfg = cfg or DEFAULT_PRESENTATION_CONFIG
        self.m = p.maneuver
        self.kind = KINDS[self.m["kind"]]
        self.trigger_m = float(self.m["trigger_range_nm"]) * NM_M
        halves = self.m.get("halves") or [[], []]
        self.plans: Dict[str, JetPlan] = {}
        for r in p.red_jets:
            half = 0
            if self.m["divide"] == "halves":
                half = 0 if r["slot"] in halves[0] else 1
            self.plans[r["id"]] = JetPlan(slot=r["slot"], hot_alt_m=float(r["hot_alt_m"]),
                                          right_nm=r["right_nm"], back_nm=r["back_nm"], half=half)
        self.fired = False
        self.fire_t: Optional[float] = None
        self.fire_range_nm: Optional[float] = None

    # ------------------------------------------------------------ helpers --
    def _log(self, world, ac, etype, text, target=None, **kw) -> None:
        world.log_event({"t": world.time_s, "type": etype, "observer": ac.id,
                         "target": target or ac.id, "text": f"{ac.name}: {text}", **kw})

    def _state(self, ac) -> str:
        if self.defense is not None:
            return self.defense.jet(ac).state
        return DEPARTED if ac.departed else HOT

    def _breakup(self, world, ac, why: str) -> None:
        jp = self.plans[ac.id]
        if not jp.broken:
            jp.broken = True
            self._log(world, ac, "breakup", f"breaks formation ({why})", reason=why)

    def _leader(self, reds: List[Aircraft]) -> Optional[Aircraft]:
        cands = [r for r in reds if not r.departed and not self.plans[r.id].broken]
        return min(cands, key=lambda r: self.plans[r.id].slot) if cands else None

    def _pursue(self, ac, tgt) -> None:
        mv.hot(ac, bearing_to(ac.state, tgt.state), alt=self.plans[ac.id].hot_alt_m).apply(ac)

    def _station_keep(self, ac, lead) -> None:
        jp, lp = self.plans[ac.id], self.plans[lead.id]
        h = lead.state.heading_rad
        fx, fy = math.sin(h), math.cos(h)
        rx, ry = math.cos(h), -math.sin(h)
        dr = (jp.right_nm - lp.right_nm) * NM_M
        db = (jp.back_nm - lp.back_nm) * NM_M
        sx = lead.state.x + rx * dr - fx * db
        sy = lead.state.y + ry * dr - fy * db
        ax, ay = sx + fx * STATION_LOOKAHEAD_M, sy + fy * STATION_LOOKAHEAD_M
        ac.cmd_heading_rad = float(math.atan2(ax - ac.state.x, ay - ac.state.y))
        along = (sx - ac.state.x) * fx + (sy - ac.state.y) * fy
        base = lead.params.cruise_speed_mps * mv.HOT_SPEED_FACTOR
        ac.cmd_speed_mps = float(np.clip(base + STATION_GAIN * along,
                                         ac.params.min_speed_mps, ac.params.max_speed_mps))
        ac.cmd_alt_m = jp.hot_alt_m

    def _fire_maneuver(self, world, reds, blues, rmin) -> None:
        self.fired, self.fire_t, self.fire_range_nm = True, world.time_s, rmin / NM_M
        lead = min(reds, key=lambda r: self.plans[r.id].slot)
        started, skipped = [], []
        for ac in sorted(reds, key=lambda r: self.plans[r.id].slot):
            jp = self.plans[ac.id]
            st = self._state(ac)
            if ac.departed or st != HOT:
                jp.status = SKIPPED
                skipped.append(ac.id)
                self._log(world, ac, "preplanned_skip",
                          f"skips pre-planned {self.m['type']} (state {st})", maneuver=self.m["type"])
                continue
            jp.status, jp.t_start = ACTIVE, world.time_s
            self.kind.start(self, world, ac, jp, self.m["params"])
            started.append(ac.id)
            self._breakup(world, ac, f"pre-planned {self.m['type']}")
        near = min(blues, key=lambda b: distance_3d(lead.state, b.state))
        self._log(world, lead, "preplanned_start",
                  f"PRE-PLANNED {self.m['type'].upper()} at {rmin / NM_M:.1f} NM "
                  f"(trigger {self.m['trigger_range_nm']:.1f} NM): {len(started)} jets"
                  + (f", skipped {','.join(skipped)}" if skipped else ""),
                  target=near.id, maneuver=self.m["type"], started=started, skipped=skipped)

    def report(self) -> dict:
        counts: Dict[str, int] = {}
        for jp in self.plans.values():
            counts[jp.status] = counts.get(jp.status, 0) + 1
        return {"type": self.m["type"], "fired": self.fired, "fire_t": self.fire_t,
                "fire_range_nm": self.fire_range_nm,
                "trigger_range_nm": self.m["trigger_range_nm"], "status_counts": counts,
                "jets": {k: {"status": v.status, "t_start": v.t_start, "t_end": v.t_end,
                             "hot_alt_m": v.hot_alt_m} for k, v in self.plans.items()}}

    # --------------------------------------------------------------- step --
    def __call__(self, world) -> None:
        self._t = world.time_s
        reds = [world.get(i) for i in self.red_ids]
        reds = [r for r in reds if r and r.state.alive]
        blues = world.alive(Coalition.BLUE)
        if not reds:
            return
        if not blues:
            for ac in reds:
                mv.ManeuverCmd(ac.state.heading_rad, ac.params.cruise_speed_mps,
                               self.plans[ac.id].hot_alt_m).apply(ac)
            return

        # L: Winchester -> leave the fight (not a turn-away)
        for ac in reds:
            if not ac.departed and ac.ammo <= 0 and not world.own_missiles_in_flight(ac):
                if self.defense is not None:
                    self.defense.force_depart(world, ac, "Winchester: no missiles left")
                else:
                    ac.departed = True
                    self._log(world, ac, "depart", "DEPART (Winchester: no missiles left)")
                self._log(world, ac, "winchester", "WINCHESTER, leaving the fight")

        # I: flight-level range trigger (true range)
        if not self.fired:
            live = [r for r in reds if not r.departed]
            if live:
                rmin = min(distance_3d(r.state, b.state) for r in live for b in blues)
                if rmin <= self.trigger_m:
                    self._fire_maneuver(world, reds, blues, rmin)

        leader = self._leader(reds)
        brk_m = self.cfg.breakup_range_nm * NM_M
        for ac in sorted(reds, key=lambda r: self.plans[r.id].slot):
            jp = self.plans[ac.id]
            tgt = min(blues, key=lambda b: distance_3d(ac.state, b.state))
            store = world.tracks.get(ac.id)
            detected = store is not None and tgt.id in store

            if self.defense is not None:
                cmd, may_shoot = self.defense.step(world, ac)
                st = self.defense.jet(ac).state
                if st != HOT:                               # J: defense wins
                    if jp.status == ACTIVE:
                        jp.status, jp.t_end = ABORTED, world.time_s
                        self._log(world, ac, "preplanned_abort",
                                  f"aborts pre-planned {self.m['type']} ({st})",
                                  maneuver=self.m["type"])
                    self._breakup(world, ac, st)
                    if cmd is not None:
                        cmd.apply(ac)
                    else:                                    # PRESSING
                        self._pursue(ac, tgt)
                    if may_shoot and st != DEPARTED:
                        self._maybe_fire(world, ac, tgt, detected)
                    continue
            elif ac.departed:
                away = math.atan2(ac.state.x - tgt.state.x, ac.state.y - tgt.state.y)
                mv.depart(ac, away).apply(ac)
                continue

            if jp.status == ACTIVE:
                cmd, may_shoot, done = self.kind.step(self, world, ac, jp, self.m["params"], tgt)
                cmd.apply(ac)
                if done:
                    jp.status, jp.t_end = DONE, world.time_s
                    self._log(world, ac, "preplanned_end",
                              f"pre-planned {self.m['type']} complete", maneuver=self.m["type"])
                if may_shoot:
                    self._maybe_fire(world, ac, tgt, detected)
                continue

            # HOT: formation until break-up (E)
            if not jp.broken:
                if store is not None and any(store.is_fire_control(b.id) for b in blues):
                    self._breakup(world, ac, "own FC track")
                elif distance_3d(ac.state, tgt.state) <= brk_m:
                    self._breakup(world, ac, f"Blue inside {self.cfg.breakup_range_nm:.0f} NM")
            if jp.broken or leader is None or ac is leader:
                self._pursue(ac, tgt)
            else:
                self._station_keep(ac, leader)
            self._maybe_fire(world, ac, tgt, detected)
