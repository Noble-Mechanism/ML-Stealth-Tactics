"""Spec 3 Red defensive state machine (docs/specs/03-missile-defense.md, section 3).

Per jet (RWR is not shared over the datalink). States:

    HOT -> DEFENDING -> COLD -> HOT,  plus terminal PRESSING and DEPARTED.

- Aggressiveness ``a`` in [0, 1] per Red flight (D1, 3 bands):
  ``a < 1/3`` conservative: trigger lock, reaction drag;
  ``1/3 <= a < 2/3`` middle: trigger support, reaction beam;
  ``a >= 2/3`` aggressive: trigger missile active, reaction crank 50 deg.
- HOT -> DEFENDING on a cue at or above the trigger level; counts one turn-away
  (D10, crank included). A trigger while DEFENDING / COLD does not count again.
- Threat reference: bearing of the highest cue (missile active > support > lock;
  ties: nearest emitter if Red has a fused range on it, else first seen).
- Threat cleared (D4): no trigger-level cue for 2 s AND, if a support cue was
  seen during this defense, the TOF estimate (D5: R_est / 700 m/s from the first
  support cue; R_est = own fused range to that emitter, else the RWR intercept
  range) has run out or a missile-active cue seen during this defense has ended.
- COLD keeps the reaction heading for T_cold = 5 s + 35 s x (1 - a), then HOT.
- Turn-away limit (default 1, approved change 2026-09-26): each jet gets exactly
  ONE defensive reaction (drag / beam / crank per band; cranks count). The next
  trigger after the limit is used -> PRESSING (ignores cues, stays hot and
  shoots) for EVERY jet regardless of ``a``; aggressiveness controls how early
  and how hard Red defends, not whether it stays. The only departure left is
  spec 4 rule L (out of missiles, none of its own in flight; ``force_depart``).
- Spec 3 behaviour (limit 2, then PRESSING if a >= 0.5 else DEPARTED = turn away
  from the Blue centroid at max speed, never fires) is kept behind
  ``DefenseConfig.spec3()`` (``depart_after_limit=True``) for reference only.
- Drag descends to max(100 m AGL, alt - depth(a)); depth(0) = down to the floor,
  depth(1) = 1,000 m (approved change B; endpoints in DefenseConfig).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, Coalition, _wrap_pi
from stealth_tactics.sim.rwr import MODE_RANK, LOCK, SUPPORT, MISSILE_ACTIVE, RwrCue
from stealth_tactics.sim.sensors import rwr_range, distance_3d
from . import maneuvers as mv

HOT, DEFENDING, COLD, PRESSING, DEPARTED = "hot", "defending", "cold", "pressing", "departed"
CONSERVATIVE, MIDDLE, AGGRESSIVE = "conservative", "middle", "aggressive"
TRIGGER = {CONSERVATIVE: LOCK, MIDDLE: SUPPORT, AGGRESSIVE: MISSILE_ACTIVE}
REACTION = {CONSERVATIVE: "drag", MIDDLE: "beam", AGGRESSIVE: "crank"}


@dataclass(frozen=True)
class DefenseConfig:
    band_edges: tuple = (1.0 / 3.0, 2.0 / 3.0)       # D1
    press_threshold: float = 0.5                      # D1 (spec 3 only): presses if a >= 0.5
    crank_deg: float = 50.0                           # D2
    clear_hold_s: float = 2.0                         # D4
    t_cold_base_s: float = 5.0                        # D4: T_cold = base + span (1 - a)
    t_cold_span_s: float = 35.0
    tof_speed_mps: float = 700.0                      # D5
    turn_away_limit: int = 1                          # D10 (2026-09-26: was 2)
    depart_after_limit: bool = False                  # spec 3: a < press_threshold departs
    drag_depth_a0_m: Optional[float] = None           # B: None = down to the floor
    drag_depth_a1_m: float = 1000.0                   # B: 1,000 m descent at a = 1
    bearing_smoothing: float = 0.5                    # EMA weight on new RWR bearings

    @classmethod
    def spec3(cls, **kw) -> "DefenseConfig":
        """Pre-2026-09-26 behaviour: two turn-aways, then press (a >= 0.5) or depart."""
        return cls(turn_away_limit=2, depart_after_limit=True, **kw)

    def band(self, a: float) -> str:
        lo, hi = self.band_edges
        return CONSERVATIVE if a < lo else (MIDDLE if a < hi else AGGRESSIVE)

    def t_cold(self, a: float) -> float:
        return self.t_cold_base_s + self.t_cold_span_s * (1.0 - min(1.0, max(0.0, a)))


@dataclass
class JetDefense:
    jet_id: str
    a: float
    state: str = HOT
    turn_aways: int = 0
    reaction: str = ""
    side: int = 1
    drag_alt: Optional[float] = None
    threat_id: Optional[str] = None
    threat_brg: Optional[float] = None
    last_trigger_t: Optional[float] = None
    support_first_t: Optional[float] = None
    support_emitter: Optional[str] = None
    t_est_s: Optional[float] = None
    active_seen: set = field(default_factory=set)
    active_ended: bool = False
    cold_until: Optional[float] = None
    defend_t: Optional[float] = None
    history: List[dict] = field(default_factory=list)


class RedDefense:
    """Runs the per-jet state machines. ``step(world, ac)`` returns the maneuver
    to fly (or None = fly the normal HOT intercept) and whether the jet may shoot."""

    def __init__(self, aggressiveness: float = 0.5, config: Optional[DefenseConfig] = None,
                 per_jet_a: Optional[Dict[str, float]] = None) -> None:
        self.a = float(aggressiveness)
        self.cfg = config or DefenseConfig()
        self.per_jet_a = dict(per_jet_a or {})
        self.jets: Dict[str, JetDefense] = {}

    def jet(self, ac: Aircraft) -> JetDefense:
        if ac.id not in self.jets:
            self.jets[ac.id] = JetDefense(ac.id, self.per_jet_a.get(ac.id, self.a))
        return self.jets[ac.id]

    # ------------------------------------------------------------ helpers --
    def _threat_cue(self, world, ac: Aircraft, cues: List[RwrCue]) -> Optional[RwrCue]:
        """Highest cue (lock or above); ties -> nearest emitter with a fused
        range, else first seen."""
        cands = [c for c in cues if c.rank >= MODE_RANK[LOCK]]
        if not cands:
            return None
        top = max(c.rank for c in cands)
        cands = [c for c in cands if c.rank == top]
        store = world.tracks.get(ac.id)

        def key(c: RwrCue):
            r = self._fused_range(store, ac, c.emitter_id)
            return (0 if r is not None else 1, r if r is not None else 0.0, c.t_first,
                    c.emitter_id)
        return min(cands, key=key)

    @staticmethod
    def _fused_range(store, ac: Aircraft, eid: str) -> Optional[float]:
        if store is None:
            return None
        ft = store.fused.get(eid)
        if ft is None:
            return None
        p = ft.est_pos
        return math.sqrt((p[0] - ac.state.x) ** 2 + (p[1] - ac.state.y) ** 2
                         + (p[2] - ac.state.alt) ** 2)

    def _update_brg(self, jd: JetDefense, cue: RwrCue) -> None:
        if jd.threat_id != cue.emitter_id or jd.threat_brg is None:
            jd.threat_id, jd.threat_brg = cue.emitter_id, cue.bearing
            return
        w = self.cfg.bearing_smoothing
        d = _wrap_pi(cue.bearing - jd.threat_brg)
        jd.threat_brg = _wrap_pi(jd.threat_brg + w * d)

    def _log(self, world, ac: Aircraft, jd: JetDefense, etype: str, text: str, **kw) -> None:
        ev = {"t": world.time_s, "type": etype, "observer": ac.id,
              "target": kw.pop("target", None) or ac.id, "text": f"{ac.name}: {text}",
              "state": jd.state, "a": jd.a, "turn_aways": jd.turn_aways, **kw}
        jd.history.append(ev)
        world.log_event(ev)

    def _start_defense(self, world, ac: Aircraft, jd: JetDefense, cue: RwrCue,
                       count: bool) -> None:
        band = self.cfg.band(jd.a)
        jd.reaction = REACTION[band]
        jd.state = DEFENDING
        jd.defend_t = world.time_s
        jd.support_first_t = None
        jd.support_emitter = None
        jd.t_est_s = None
        jd.active_seen = set()
        jd.active_ended = False
        jd.cold_until = None
        self._update_brg(jd, cue)
        jd.side = mv.short_side(ac, jd.threat_brg)
        if jd.reaction == "drag":
            jd.drag_alt = mv.drag_target_alt(ac, jd.a, self.cfg.drag_depth_a1_m,
                                             self.cfg.drag_depth_a0_m)
        if count:
            jd.turn_aways += 1
        tgt = cue.source_jet or cue.emitter_id
        extra = (f", descend to {jd.drag_alt:.0f} m" if jd.reaction == "drag" else "")
        self._log(world, ac, jd, "defend",
                  f"DEFEND ({band}, a={jd.a:.2f}) on {cue.mode} from {cue.emitter_id}: "
                  f"{jd.reaction}{extra}; turn-away {jd.turn_aways}/"
                  f"{self.cfg.turn_away_limit}" + ("" if count else " (not counted)"),
                  target=tgt if tgt in world._by_id else None, reaction=jd.reaction,
                  cue=cue.mode, emitter=cue.emitter_id, counted=count)

    def _maneuver(self, ac: Aircraft, jd: JetDefense) -> mv.ManeuverCmd:
        brg = jd.threat_brg if jd.threat_brg is not None else ac.state.heading_rad + math.pi
        if jd.reaction == "crank":
            return mv.crank(ac, brg, self.cfg.crank_deg, side=jd.side)
        if jd.reaction == "beam":
            return mv.beam(ac, brg, side=jd.side)
        return mv.drag(ac, brg, jd.drag_alt)

    def _depart_brg(self, world, ac: Aircraft, jd: JetDefense) -> float:
        store = world.tracks.get(ac.id)
        pts = []
        if store is not None:
            for tid, ft in store.fused.items():
                tgt = world.get(tid)
                if tgt is not None and tgt.coalition == Coalition.BLUE:
                    pts.append(ft.est_pos)
        if pts:
            c = np.mean(np.array(pts), axis=0)
            return math.atan2(ac.state.x - c[0], ac.state.y - c[1])
        if jd.threat_brg is not None:
            return _wrap_pi(jd.threat_brg + math.pi)
        return ac.state.heading_rad

    def force_depart(self, world, ac: Aircraft, reason: str) -> None:
        """Spec 4 L: leave the fight for good (D9 departure) for a non-threat
        reason (e.g. Winchester). Not counted as a turn-away."""
        jd = self.jet(ac)
        if jd.state == DEPARTED:
            return
        jd.state = DEPARTED
        ac.departed = True
        ac.defense_state = DEPARTED
        self._log(world, ac, jd, "depart", f"DEPART ({reason})", reason=reason)

    # --------------------------------------------------------------- step --
    def step(self, world, ac: Aircraft):
        """Returns (ManeuverCmd or None for the normal HOT intercept, may_shoot)."""
        jd = self.jet(ac)
        cfg = self.cfg
        t = world.time_s
        cues = world.rwr.get(ac.id, [])
        band = cfg.band(jd.a)
        trig_rank = MODE_RANK[TRIGGER[band]]
        trig = [c for c in cues if c.rank >= trig_rank]

        if jd.state == DEPARTED:
            ac.defense_state = DEPARTED
            return mv.depart(ac, self._depart_brg(world, ac, jd)), False
        if jd.state == PRESSING:
            ac.defense_state = PRESSING
            return None, True

        if trig:
            cue = self._threat_cue(world, ac, trig)
            jd.last_trigger_t = t
            if jd.state == HOT:
                if jd.turn_aways >= cfg.turn_away_limit:
                    if not cfg.depart_after_limit or jd.a >= cfg.press_threshold:
                        jd.state = PRESSING
                        self._log(world, ac, jd, "press",
                                  f"PRESS (turn-away limit {cfg.turn_away_limit} used, "
                                  f"a={jd.a:.2f}) on {cue.mode} from {cue.emitter_id}")
                        ac.defense_state = PRESSING
                        return None, True
                    jd.state = DEPARTED
                    ac.departed = True
                    self._update_brg(jd, cue)
                    self._log(world, ac, jd, "depart",
                              f"DEPART (turn-away limit {cfg.turn_away_limit} used, "
                              f"a={jd.a:.2f}) on {cue.mode} from {cue.emitter_id}")
                    ac.defense_state = DEPARTED
                    return mv.depart(ac, self._depart_brg(world, ac, jd)), False
                self._start_defense(world, ac, jd, cue, count=True)
            elif jd.state == COLD:
                self._update_brg(jd, cue)
                jd.state = DEFENDING
                jd.cold_until = None
                self._log(world, ac, jd, "defend",
                          f"DEFEND again from COLD on {cue.mode} from {cue.emitter_id} "
                          f"(not counted; {jd.reaction})", reaction=jd.reaction,
                          cue=cue.mode, emitter=cue.emitter_id, counted=False)
            else:
                self._update_brg(jd, cue)

        if jd.state == DEFENDING:
            # D5 bookkeeping: first support cue of this defense starts the TOF clock
            for c in cues:
                if c.mode == SUPPORT and jd.support_first_t is None:
                    jd.support_first_t = t
                    jd.support_emitter = c.emitter_id
                    em = world.get(c.emitter_id)
                    r = self._fused_range(world.tracks.get(ac.id), ac, c.emitter_id)
                    if r is None:
                        r = rwr_range(em, world.sensor_cfg) if em is not None else 54_000.0
                    jd.t_est_s = r / cfg.tof_speed_mps
            now_active = {c.emitter_id for c in cues if c.mode == MISSILE_ACTIVE}
            if jd.active_seen - now_active:
                jd.active_ended = True
            jd.active_seen |= now_active
            quiet = (not trig and jd.last_trigger_t is not None
                     and t - jd.last_trigger_t >= cfg.clear_hold_s - 1e-9)
            if quiet:
                tof_ok = (jd.support_first_t is None or jd.active_ended
                          or t - jd.support_first_t >= jd.t_est_s - 1e-9)
                if tof_ok:
                    jd.state = COLD
                    jd.cold_until = t + cfg.t_cold(jd.a)
                    self._log(world, ac, jd, "threat_cleared",
                              f"threat cleared; cold for {cfg.t_cold(jd.a):.1f} s "
                              f"({jd.reaction})", t_cold=cfg.t_cold(jd.a))
        if jd.state == COLD and t >= jd.cold_until - 1e-9:
            jd.state = HOT
            self._log(world, ac, jd, "recommit",
                      f"RECOMMIT (hot) after {t - (jd.defend_t or t):.1f} s defending/cold; "
                      f"alt {ac.state.alt:.0f} m")
        ac.defense_state = jd.state
        if jd.state in (DEFENDING, COLD):
            return self._maneuver(ac, jd), jd.reaction == "crank"
        return None, True
