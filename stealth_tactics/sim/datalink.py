"""Spec 2 intra-flight datalink (track sharing).

- Each alive jet broadcasts a snapshot of its OWN radar/IRST track components
  (with error, hold state, fire-control flag and its own position at measurement)
  every ``update_period_s`` of its coalition (Blue 1 s, Red 2 s).
- One message per sender per update; it is lost (for every recipient) with
  probability ``loss_prob`` (5 %), drawn from the datalink's own RNG stream.
- Recipients: alive flightmates within ``max_range_m`` (150 NM) at send time.
- Delivered ``latency_s`` later (Blue 1 s, Red 2 s). The copy keeps the sender's
  measurement time, so the receiver's error grows 50 m/s x (t - meas time).
- Received data are never relayed (only own measurements are sent).
- RWR contacts are not shared. The link is undetectable (no RWR/IRST effect).
- Dead jets stop sending; their shared tracks age out on the 10 s coast rule.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .aircraft import Aircraft
from .fusion import own_sigma_at_meas
from .sensor_config import DEFAULT_SENSOR_CONFIG, SensorConfig
from .tracks import SensorComponent, Track, TrackStore


@dataclass
class LinkMessage:
    sender: str
    coalition: str
    send_t: float
    deliver_t: float
    recipients: List[str]
    # target id -> {"fc": bool, "comps": {source: SensorComponent}}
    tracks: Dict[str, dict] = field(default_factory=dict)


def _dist(a: Aircraft, b: Aircraft) -> float:
    sa, sb = a.state, b.state
    return math.sqrt((sa.x - sb.x) ** 2 + (sa.y - sb.y) ** 2 + (sa.alt - sb.alt) ** 2)


class Datalink:
    def __init__(self, rng: np.random.Generator, config: Optional[SensorConfig] = None) -> None:
        self.rng = rng
        self.cfg = config or DEFAULT_SENSOR_CONFIG
        self._next_send: Dict[str, float] = {}
        self.pending: List[LinkMessage] = []
        self.sent = 0
        self.lost = 0
        self.delivered = 0
        self.log: List[dict] = []   # per-message record (send/lost/deliver)

    # ---------------------------------------------------------------- send --
    def send(self, aircraft: List[Aircraft], stores: Dict[str, TrackStore],
             t: float) -> List[dict]:
        dl = self.cfg.datalink
        events: List[dict] = []
        eps = 1e-9
        for coal in sorted({a.coalition.value for a in aircraft}):
            nxt = self._next_send.get(coal, 0.0)
            if t < nxt - eps:
                continue
            period = dl.update_period_s[coal]
            while nxt <= t + eps:
                nxt += period
            self._next_send[coal] = nxt
            for snd in aircraft:
                if snd.coalition.value != coal or not snd.state.alive:
                    continue
                recips = [a.id for a in aircraft
                          if a is not snd and a.coalition == snd.coalition
                          and a.state.alive and _dist(a, snd) <= dl.max_range_m]
                lost = bool(self.rng.random() < dl.loss_prob[coal])
                self.sent += 1
                rec = {"t": t, "sender": snd.id, "lost": lost, "recipients": recips}
                self.log.append(rec)
                if lost:
                    self.lost += 1
                    continue
                if not recips:
                    continue
                msg = LinkMessage(snd.id, coal, t, t + dl.latency_s[coal], recips)
                store = stores.get(snd.id)
                if store is not None:
                    for tid, tr in store.tracks.items():
                        comps = {}
                        for src in dl.shared_sources:
                            c = tr.components.get(src)
                            if c is None or c.est_pos is None:
                                continue
                            cc = copy.copy(c)
                            cc.est_pos = c.est_pos.copy()
                            cc.est_vel = None if c.est_vel is None else c.est_vel.copy()
                            cc.obs_pos = None if c.obs_pos is None else c.obs_pos.copy()
                            cc.sigma_at_meas = own_sigma_at_meas(c, self.cfg)
                            cc.sender = snd.id
                            cc.origin = "link"
                            comps[src] = cc
                        if comps:
                            msg.tracks[tid] = {"fc": tr.fire_control, "comps": comps}
                self.pending.append(msg)
        return events

    # ------------------------------------------------------------- deliver --
    def deliver(self, aircraft: List[Aircraft], stores: Dict[str, TrackStore],
                t: float) -> List[dict]:
        events: List[dict] = []
        by_id = {a.id: a for a in aircraft}
        due = [m for m in self.pending if m.deliver_t <= t + 1e-9]
        self.pending = [m for m in self.pending if m.deliver_t > t + 1e-9]
        for msg in due:
            for rid in msg.recipients:
                rcv = by_id.get(rid)
                if rcv is None or not rcv.state.alive:
                    continue
                store = stores.setdefault(rid, TrackStore(rid))
                self.delivered += 1
                for tid, info in msg.tracks.items():
                    tgt = by_id.get(tid)
                    if tgt is None or not tgt.state.alive:
                        continue
                    tr = store.tracks.get(tid)
                    if tr is None:
                        tr = Track(target_id=tid, first_t=t)
                        store.tracks[tid] = tr
                    if msg.send_t <= tr.remote_msg_t.get(msg.sender, -1e18):
                        continue
                    if msg.sender not in tr.remote:
                        events.append({
                            "t": t, "type": "link_track", "observer": rid, "target": tid,
                            "range_m": _dist(rcv, tgt), "sender": msg.sender,
                            "text": f"{rcv.name}: link track on {tid} from {msg.sender}"})
                    tr.remote[msg.sender] = dict(info["comps"])
                    tr.remote_msg_t[msg.sender] = msg.send_t
                    tr.remote_fc[msg.sender] = (msg.send_t, bool(info["fc"]))
        return events
