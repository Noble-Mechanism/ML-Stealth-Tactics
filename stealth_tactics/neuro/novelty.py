"""Spec 6 L: behaviour descriptor, novelty, archive, hall of fame.

The behaviour descriptor (BD) describes HOW a network fights, not how well.
It is measured from the World by a recorder wrapped around the Blue controller
(sampled at every 1 s decision); being an analysis quantity, not a policy
input, it may use truth (nearest Red distance and bearing). Spec 7 may change
the axes; everything reads them through ``BD_NAMES``."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from stealth_tactics.sim.aircraft import Coalition

BD_NAMES = ("radar_off_share", "launch_r_rmax", "shots", "closest_approach",
            "altitude", "away_share", "spread", "first_shot_t")
BD_SCALE = {"launch_r_rmax": 1.5, "shots": 8.0, "closest_approach": 150_000.0,
            "altitude": 15_000.0, "spread": 40_000.0, "first_shot_t": 360.0}
AWAY_DEG = 70.0


class BDRecorder:
    """Wraps a Blue controller; samples behaviour at whole seconds."""

    def __init__(self, inner, blue_ids: Sequence[str]) -> None:
        self.inner = inner
        self.blue_ids = list(blue_ids)
        self.n = 0
        self.radar_off = 0
        self.away = 0
        self.alt_sum = 0.0
        self.spread_sum = 0.0
        self.spread_n = 0
        self.closest = math.inf
        self.fire_rr: Dict[tuple, float] = {}
        # spec 7: heading off the bearing to the nearest live Red at each jet's
        # last whole-second sample (egress test for the loss weight)
        self.last_off_deg: Dict[str, float] = {}
        self._next = 0.0

    def __call__(self, world) -> None:
        self.inner(world)
        t = world.time_s
        blues = [world.get(b) for b in self.blue_ids]
        blues = [b for b in blues if b is not None and b.state.alive]
        reds = [a for a in world.aircraft if a.coalition == Coalition.RED and a.state.alive]
        env = world.weapons.envelope
        for b in blues:                       # r / Rmax at every fire request
            if b.cmd_fire and b.fire_target:
                tgt = world.get(b.fire_target)
                if tgt is not None and tgt.state.alive:
                    rmax = env.rmax_for_states(b.state, tgt.state)
                    r = math.dist((b.state.x, b.state.y, b.state.alt),
                                  (tgt.state.x, tgt.state.y, tgt.state.alt))
                    self.fire_rr[(round(t, 3), b.id)] = r / rmax if rmax > 0 else 1.5
        if t < self._next - 1e-9:
            return
        self._next = math.floor(t + 1e-9) + 1.0
        for b in blues:
            self.n += 1
            self.radar_off += not b.radar_emitting
            self.alt_sum += b.state.alt
            if reds:
                d, r = min((math.dist((b.state.x, b.state.y, b.state.alt),
                                      (r.state.x, r.state.y, r.state.alt)), r) for r in reds)
                self.closest = min(self.closest, d)
                brg = math.atan2(r.state.x - b.state.x, r.state.y - b.state.y)
                off = abs((brg - b.state.heading_rad + math.pi) % (2 * math.pi) - math.pi)
                self.away += math.degrees(off) > AWAY_DEG
                self.last_off_deg[b.id] = math.degrees(off)
        if len(blues) >= 2:
            ds = [math.dist((a.state.x, a.state.y), (c.state.x, c.state.y))
                  for i, a in enumerate(blues) for c in blues[i + 1:]]
            self.spread_sum += float(np.mean(ds))
            self.spread_n += 1

    def egress_by_jet(self, off_deg: float) -> Dict[str, bool]:
        """Blue jet id -> heading more than *off_deg* off the nearest live Red
        at its last sample (all Blue ids present)."""
        return {b: self.last_off_deg.get(b, 0.0) > off_deg for b in self.blue_ids}

    def raw(self, res) -> dict:
        blue = set(self.blue_ids)
        launches = [e for e in res.events if e["type"] == "launch" and e["shooter"] in blue]
        rr = [self.fire_rr.get((round(e["t"], 3), e["shooter"])) for e in launches]
        rr = [x for x in rr if x is not None]
        n = max(self.n, 1)
        return {"radar_off_share": self.radar_off / n,
                "launch_r_rmax": float(np.mean(rr)) if rr else None,
                "shots": len(launches),
                "closest_approach": self.closest if math.isfinite(self.closest) else 150_000.0,
                "altitude": self.alt_sum / n,
                "away_share": self.away / n,
                "spread": self.spread_sum / self.spread_n if self.spread_n else 0.0,
                "first_shot_t": min((e["t"] for e in launches), default=360.0)}


def descriptor(raws: List[dict]) -> np.ndarray:
    """Mean over fights, each axis scaled to about [0, 1] (launch r/Rmax is
    averaged over fights with shots; 0 if the network never fired)."""
    out = []
    for k in BD_NAMES:
        vals = [r[k] for r in raws if r[k] is not None]
        v = float(np.mean(vals)) if vals else 0.0
        out.append(min(1.5, v / BD_SCALE.get(k, 1.0)))
    return np.array(out, dtype=np.float64)


def novelty(bds: np.ndarray, archive: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest neighbours among the other members and
    the archive."""
    pool = np.vstack([bds, archive]) if len(archive) else bds
    d = np.sqrt(((bds[:, None, :] - pool[None, :, :]) ** 2).sum(-1))
    n = len(bds)
    d[np.arange(n), np.arange(n)] = np.inf           # exclude self
    kk = min(k, pool.shape[0] - 1)
    if kk <= 0:
        return np.zeros(n)
    return np.sort(d, axis=1)[:, :kk].mean(1)


def ranks(values: np.ndarray, descending: bool = True) -> np.ndarray:
    """0 = best. Ties broken by index (stable)."""
    v = -values if descending else values
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.int64)
    r[order] = np.arange(len(v))
    return r


# ---------------------------------------------------------------- hall of fame
RADAR_BINS = (1 / 3, 2 / 3)          # radar-off share
RR_BINS = (0.6, 0.85)                # mean launch range / Rmax


def hof_cell(raw_mean: dict) -> Optional[str]:
    """3 x 3 cell over radar-off share x launch range / Rmax, e.g. "r0_l2".
    None if the network never fired on the benchmark (no launch range)."""
    rr = raw_mean.get("launch_r_rmax")
    if rr is None:
        return None
    ro = raw_mean["radar_off_share"]
    i = int(ro >= RADAR_BINS[0]) + int(ro >= RADAR_BINS[1])
    j = int(rr >= RR_BINS[0]) + int(rr >= RR_BINS[1])
    return f"r{i}_l{j}"


def mean_raw(raws: List[dict]) -> dict:
    out = {}
    for k in BD_NAMES:
        vals = [r[k] for r in raws if r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out
