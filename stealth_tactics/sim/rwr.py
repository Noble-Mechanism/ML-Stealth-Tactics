"""Spec 3 RWR modes: search / lock / support per emitter, plus missile-active
warnings (docs/specs/03-missile-defense.md, section 1).

Computed every world step AFTER the track stores are maintained, so modes use the
same step's fire-control state. For each receiver and each enemy emitter only the
highest mode is reported (search < lock < support). Gates (D7): the emitter's
radar is emitting, the receiver is inside the emitter's radar field of regard
(+/-60 deg az/el) and within ``emitter_detect_factor x emitter ref range x
mode_range_factor[mode]`` (F-35 LPI 29 NM, Red radar 57 NM). Modes are
identified perfectly. A missile-active warning is reported per live autonomous
missile aimed at the receiver (only the targeted jet gets it; no FOR or range
gate). Bearing noise sigma 5 deg (spec 1), no range. Noise comes from its own RNG
stream (``RWRConfig.mode_rng_salt``) so spec 1-3a draws do not shift.

The spec 1 RWR track component (sim/sensors.py) is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .aircraft import Aircraft
from .sensor_config import SensorConfig
from .sensors import distance_3d, in_field_of_regard, rwr_range

SEARCH, LOCK, SUPPORT, MISSILE_ACTIVE = "search", "lock", "support", "missile_active"
MODE_RANK = {SEARCH: 1, LOCK: 2, SUPPORT: 3, MISSILE_ACTIVE: 4}


@dataclass
class RwrCue:
    emitter_id: str          # enemy jet id, or missile id for missile_active
    mode: str                # search | lock | support | missile_active
    bearing: float           # measured bearing (rad, 0 = N clockwise), sigma 5 deg
    t_first: float           # first time this emitter was seen in this mode (continuous)
    source_jet: Optional[str] = None   # missile_active: the missile's shooter

    @property
    def rank(self) -> int:
        return MODE_RANK[self.mode]


def emitter_mode(receiver: Aircraft, emitter: Aircraft, tracks, missiles: Sequence,
                 cfg: SensorConfig) -> Optional[str]:
    """Highest RWR mode *receiver* reports for *emitter* (None if not seen)."""
    if not emitter.state.alive or not receiver.state.alive:
        return None
    if emitter.coalition == receiver.coalition or not emitter.radar_emitting:
        return None
    rc = cfg.radar
    if not in_field_of_regard(emitter.state, receiver.state, rc.for_az_deg, rc.for_el_deg):
        return None
    d = distance_3d(receiver.state, emitter.state)
    base = rwr_range(emitter, cfg)
    fac = cfg.rwr.mode_range_factor
    for m in missiles:
        if (m.alive and m.target_id == receiver.id and m.supporter_id == emitter.id):
            if d <= base * fac.get(SUPPORT, 1.0):
                return SUPPORT
            break
    store = tracks.get(emitter.id)
    if store is not None and store.is_fire_control(receiver.id):
        if d <= base * fac.get(LOCK, 1.0):
            return LOCK
    if d <= base * fac.get(SEARCH, 1.0):
        return SEARCH
    return None


class RwrModel:
    """Owns ``world.rwr`` (receiver id -> list of RwrCue) and emits ``rwr_mode``
    events on every mode change (including appearing / disappearing)."""

    def __init__(self, rng: np.random.Generator, cfg: SensorConfig) -> None:
        self.rng = rng
        self.cfg = cfg
        self.cues: Dict[str, List[RwrCue]] = {}
        # receiver -> emitter -> (mode, t_first)
        self._state: Dict[str, Dict[str, tuple]] = {}

    def update(self, aircraft: Sequence[Aircraft], missiles: Sequence, tracks,
               t: float) -> List[dict]:
        by_id = {a.id: a for a in aircraft}
        raw = []   # (receiver, emitter_id, mode, true bearing, source_jet)
        for rx in aircraft:
            if not rx.state.alive:
                continue
            for em in aircraft:
                if em.coalition == rx.coalition:
                    continue
                mode = emitter_mode(rx, em, tracks, missiles, self.cfg)
                if mode is not None:
                    raw.append((rx, em.id, mode, _brg(rx.state.x, rx.state.y,
                                                      em.state.x, em.state.y), None))
            for m in missiles:
                if m.alive and m.autonomous and m.target_id == rx.id:
                    raw.append((rx, m.id, MISSILE_ACTIVE,
                                _brg(rx.state.x, rx.state.y, m.pos.x, m.pos.y),
                                m.shooter_id))
        noise = (self.rng.normal(0.0, math.radians(self.cfg.rwr.bearing_sigma_deg), len(raw))
                 if raw else ())
        events: List[dict] = []
        new_cues: Dict[str, List[RwrCue]] = {}
        seen: Dict[str, set] = {}
        for (rx, eid, mode, brg, src), n in zip(raw, noise):
            st = self._state.setdefault(rx.id, {})
            prev = st.get(eid)
            if prev is None or prev[0] != mode:
                t_first = t
                st[eid] = (mode, t)
                events.append(self._event(t, rx, eid, prev[0] if prev else None, mode,
                                          src, by_id))
            else:
                t_first = prev[1]
            b = (brg + float(n) + math.pi) % (2 * math.pi) - math.pi
            new_cues.setdefault(rx.id, []).append(RwrCue(eid, mode, b, t_first, src))
            seen.setdefault(rx.id, set()).add(eid)
        for rid, st in self._state.items():
            for eid in [e for e in st if e not in seen.get(rid, ())]:
                rx = by_id.get(rid)
                old = st.pop(eid)[0]
                if rx is not None and rx.state.alive:
                    events.append(self._event(t, rx, eid, old, None, None, by_id))
        self.cues = new_cues
        return events

    @staticmethod
    def _event(t, rx: Aircraft, eid: str, old: Optional[str], new: Optional[str],
               src: Optional[str], by_id) -> dict:
        is_missile = eid not in by_id
        tgt = (src or eid) if is_missile else eid
        text = f"{rx.name}: RWR {eid} {old or 'none'} -> {new or 'none'}"
        ev = {"t": t, "type": "rwr_mode", "observer": rx.id, "target": tgt,
              "emitter": eid, "old": old, "mode": new, "text": text}
        if is_missile:
            ev["missile"] = eid
        return ev


def _brg(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(x1 - x0, y1 - y0)
