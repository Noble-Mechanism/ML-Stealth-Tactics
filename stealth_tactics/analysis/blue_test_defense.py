"""Spec 3 Blue scripted test reaction (tests / stats / replays only; never used
by the GA). Blue gets its real defensive moves from the network in spec 5.

``BlueTestDefense`` wraps any Blue controller:
- Trigger: a Red ``support`` or ``missile_active`` cue on the jet's RWR (D8).
- Reaction: crank 50 deg while the jet is supporting its own pre-active missile;
  otherwise (and once that missile is active) drag, descending to
  max(100 m AGL, alt - depth(a)) with the jet's ``aggressiveness`` (change B).
- Recommit 20 s after the threat clears (no trigger cue for 2 s). No turn-away
  limit. While dragging the jet does not fire; while cranking the wrapped
  controller's fire requests pass through (world gates decide).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from stealth_tactics.sim.aircraft import Coalition, _wrap_pi
from stealth_tactics.sim.rwr import MODE_RANK, SUPPORT, MISSILE_ACTIVE
from stealth_tactics.tactics import maneuvers as mv


@dataclass
class _BlueJet:
    state: str = "hot"             # hot | defending | cold
    reaction: str = ""
    side: int = 1
    brg: Optional[float] = None
    threat: Optional[str] = None
    last_trig_t: Optional[float] = None
    cold_until: Optional[float] = None
    drag_alt: Optional[float] = None
    n_defends: int = 0


class BlueTestDefense:
    def __init__(self, inner: Callable, recommit_s: float = 20.0, clear_hold_s: float = 2.0,
                 crank_deg: float = 50.0, aggressiveness: float = 0.5,
                 drag_depth_a1_m: float = 1000.0,
                 drag_depth_a0_m: Optional[float] = None) -> None:
        self.inner = inner
        self.recommit_s = recommit_s
        self.clear_hold_s = clear_hold_s
        self.crank_deg = crank_deg
        self.a = aggressiveness
        self.depth_a1 = drag_depth_a1_m
        self.depth_a0 = drag_depth_a0_m
        self.jets: Dict[str, _BlueJet] = {}

    def __getattr__(self, item):          # expose the wrapped controller's fields
        return getattr(self.inner, item)

    def _log(self, world, ac, etype, text, **kw) -> None:
        world.log_event({"t": world.time_s, "type": etype, "observer": ac.id,
                         "target": kw.pop("target", None) or ac.id,
                         "text": f"{ac.name}: {text}", **kw})

    def __call__(self, world) -> None:
        self.inner(world)
        t = world.time_s
        for ac in world.alive(Coalition.BLUE):
            jd = self.jets.setdefault(ac.id, _BlueJet())
            cues = [c for c in world.rwr.get(ac.id, [])
                    if c.rank >= MODE_RANK[SUPPORT]]
            if cues:
                top = max(cues, key=lambda c: (c.rank, -c.t_first))
                if jd.threat != top.emitter_id or jd.brg is None:
                    jd.brg, jd.threat = top.bearing, top.emitter_id
                else:
                    jd.brg = _wrap_pi(jd.brg + 0.5 * _wrap_pi(top.bearing - jd.brg))
                jd.last_trig_t = t
                if jd.state != "defending":
                    jd.state = "defending"
                    jd.n_defends += 1
                    jd.side = mv.short_side(ac, jd.brg)
                    jd.reaction = ""
                    src = top.source_jet or top.emitter_id
                    self._log(world, ac, "blue_defend",
                              f"BLUE TEST DEFENSE on {top.mode} from {top.emitter_id}",
                              target=src if world.get(src) else None, cue=top.mode)
            if jd.state == "defending" and not cues and jd.last_trig_t is not None \
                    and t - jd.last_trig_t >= self.clear_hold_s - 1e-9:
                jd.state = "cold"
                jd.cold_until = t + self.recommit_s
            if jd.state == "cold" and t >= jd.cold_until - 1e-9:
                jd.state = "hot"
                jd.reaction = ""
                self._log(world, ac, "blue_recommit", "BLUE TEST DEFENSE recommit (hot)")
            ac.defense_state = jd.state
            if jd.state == "hot":
                continue
            supporting = any(
                m.alive and m.shooter_id == ac.id and not m.autonomous
                and m.supporter_id == ac.id for m in world.missiles)
            # defending and cold (holding the reaction for 20 s): crank only while
            # supporting its own pre-active missile, else drag
            want = "crank" if supporting else "drag"
            if jd.reaction == "drag" and want == "crank":
                want = "drag"              # never go back from drag to crank
            if want != jd.reaction:
                if want == "drag":
                    jd.drag_alt = mv.drag_target_alt(ac, self.a, self.depth_a1, self.depth_a0)
                self._log(world, ac, "blue_defend",
                          f"BLUE TEST DEFENSE {want}"
                          + (f", descend to {jd.drag_alt:.0f} m" if want == "drag" else ""),
                          reaction=want)
                jd.reaction = want
            if jd.reaction == "crank":
                mv.crank(ac, jd.brg, self.crank_deg, side=jd.side).apply(ac)
            else:
                mv.drag(ac, jd.brg, jd.drag_alt).apply(ac)
                ac.cmd_fire = False
                ac.fire_target = None
