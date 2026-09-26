"""Spec 5 C-G: observation vector (231 inputs per Blue jet).

Layout ``[own 21 | contacts 6 x 28 | wingmen 3 x 14]``; ego-centric frame, the
mission (ingress) axis = north is the only absolute direction. Fixed scales
(no running statistics); missing values are 0 with an explicit validity flag.
Built only from a ``BlueView`` (decision A).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from stealth_tactics.sim.aircraft import AircraftState
from stealth_tactics.sim.rwr import MISSILE_ACTIVE

NM_M = 1852.0
MISSION_AXIS_RAD = 0.0      # Blue ingress heading (north); "home" is south

OWN_NAMES = (
    "slot_1", "slot_2", "slot_3", "slot_4", "alt", "speed", "hdg_sin", "hdg_cos",
    "ammo", "time", "radar_on", "msl_in_flight", "msl_need_my_support", "msl_coasting",
    "msl_soonest_active", "ssa_pair_open", "since_last_launch", "maw_count",
    "maw_brg_sin", "maw_brg_cos", "red_kills",
)
CONTACT_NAMES = (
    "present", "range_valid", "range", "brg_sin", "brg_cos", "dalt", "kin_valid",
    "aspect_sin", "aspect_cos", "closure", "tgt_speed", "sigma", "age", "own_sensor",
    "sensor_radar", "sensor_irst", "sensor_tri", "own_fc", "remote_fc", "r_rmax",
    "r_rne", "rmax", "doctrine_ok", "shot_ready", "own_msl_on_it", "mate_msl_on_it",
    "mates_targeting", "rwr_mode",
)
MATE_NAMES = (
    "alive", "range", "brg_sin", "brg_cos", "dalt", "rel_hdg_sin", "rel_hdg_cos", "speed",
    "ammo", "msl_in_flight", "threatened", "radar_on", "has_target", "same_target",
)
# Element-relative wingman order (F): my element mate, other lead, other wing.
MATE_ORDER = {0: (1, 2, 3), 1: (0, 2, 3), 2: (3, 0, 1), 3: (2, 0, 1)}


@dataclass(frozen=True)
class ObsSpec:
    k: int = 6
    n_mates: int = 3
    contact_range_m: float = 80.0 * NM_M
    mate_range_m: float = 40.0 * NM_M
    alt_m: float = 15000.0
    dalt_m: float = 5000.0
    speed_mps: float = 400.0
    closure_mps: float = 1000.0
    time_s: float = 360.0
    coast_s: float = 10.0
    ammo: float = 4.0
    n_red: float = 6.0
    min_launch_m: float = 500.0
    max_off_boresight_deg: float = 60.0
    missile_speed_est_mps: float = 1000.0

    @property
    def own_size(self) -> int:
        return len(OWN_NAMES)

    @property
    def contact_size(self) -> int:
        return len(CONTACT_NAMES)

    @property
    def mate_size(self) -> int:
        return len(MATE_NAMES)

    @property
    def size(self) -> int:
        return self.own_size + self.k * self.contact_size + self.n_mates * self.mate_size

    @property
    def own_slice(self) -> slice:
        return slice(0, self.own_size)

    def contact_slice(self, i: int) -> slice:
        a = self.own_size + i * self.contact_size
        return slice(a, a + self.contact_size)

    @property
    def contacts_slice(self) -> slice:
        return slice(self.own_size, self.own_size + self.k * self.contact_size)

    def mate_slice(self, i: int) -> slice:
        a = self.own_size + self.k * self.contact_size + i * self.mate_size
        return slice(a, a + self.mate_size)

    def names(self) -> List[str]:
        out = [f"own.{n}" for n in OWN_NAMES]
        for i in range(self.k):
            out += [f"c{i}.{n}" for n in CONTACT_NAMES]
        for i in range(self.n_mates):
            out += [f"w{i}.{n}" for n in MATE_NAMES]
        return out

    def index(self, name: str) -> int:
        return self.names().index(name)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Declared (lo, hi) per input (tests check every observation)."""
        lo, hi = [], []
        signed = {"hdg_sin", "hdg_cos", "maw_brg_sin", "maw_brg_cos", "brg_sin", "brg_cos",
                  "dalt", "aspect_sin", "aspect_cos", "closure", "rel_hdg_sin", "rel_hdg_cos"}
        wide = {"range", "rmax", "dalt"}
        for n in self.names():
            base = n.split(".", 1)[1]
            lo.append(-1.5 if base == "dalt" else (-1.0 if base in signed else 0.0))
            hi.append(1.5 if base in wide else 1.0)
        return np.array(lo), np.array(hi)


OBS_SPEC = ObsSpec()


@dataclass
class ObsInfo:
    """Side information for the action decoder (never a network input)."""
    keys: List[Optional[str]]              # track key per contact slot (None = empty)
    bearings: List[Optional[float]]        # absolute perceived bearing per slot
    shot_ready: List[bool] = field(default_factory=list)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def order_contacts(view, current_target: Optional[str]) -> list:
    """D: slot 0 = current target if still in the picture, else the nearest
    ranged contact; the rest: ranged by perceived range (tie: sigma, key), then
    bearing-only by RWR mode (support > lock > search), first seen, key."""
    ox, oy, oa = view.own.x, view.own.y, view.own.alt

    def rng(c):
        return math.sqrt((c.est_pos[0] - ox) ** 2 + (c.est_pos[1] - oy) ** 2
                         + (c.est_pos[2] - oa) ** 2)
    ranged = sorted((c for c in view.contacts if c.ranged),
                    key=lambda c: (rng(c), c.sigma_m, c.key))
    bonly = sorted((c for c in view.contacts if not c.ranged),
                   key=lambda c: (-c.rwr_mode, c.first_t, c.key))
    pool = ranged + bonly
    first = None
    if current_target is not None:
        first = next((c for c in pool if c.key == current_target), None)
    if first is None and pool:
        first = pool[0]
    if first is None:
        return []
    return [first] + [c for c in pool if c is not first]


def build_observation(view, current_target: Optional[str] = None,
                      mate_targets: Optional[Dict[str, Optional[str]]] = None,
                      spec: ObsSpec = OBS_SPEC) -> Tuple[np.ndarray, ObsInfo]:
    mate_targets = mate_targets or {}
    o = view.own
    obs = np.zeros(spec.size, dtype=np.float64)
    own_state = AircraftState(o.x, o.y, o.alt, o.heading, o.speed)
    env = view.envelope
    my_missiles = [m for m in view.missiles if m.shooter == o.id]
    live_blue = [m for m in view.missiles if m.alive]
    contacts = order_contacts(view, current_target)[: spec.k]
    est_by_key = {c.key: c.est_pos for c in view.contacts if c.ranged}

    # ---------------------------------------------------------------- own --
    own = np.zeros(spec.own_size)
    if 0 <= o.slot < 4:
        own[o.slot] = 1.0
    own[4] = _clip(o.alt / spec.alt_m, 0.0, 1.0)
    own[5] = _clip(o.speed / spec.speed_mps, 0.0, 1.0)
    rel = _wrap(o.heading - MISSION_AXIS_RAD)
    own[6], own[7] = math.sin(rel), math.cos(rel)
    own[8] = _clip(o.ammo / spec.ammo, 0.0, 1.0)
    own[9] = _clip(view.t / spec.time_s, 0.0, 1.0)
    own[10] = 1.0 if o.radar_emitting else 0.0
    mine_live = [m for m in my_missiles if m.alive]
    own[11] = _clip(len(mine_live) / spec.ammo, 0.0, 1.0)
    own[12] = _clip(sum(1 for m in live_blue if not m.autonomous and m.supporter == o.id)
                    / spec.ammo, 0.0, 1.0)
    own[13] = _clip(sum(1 for m in mine_live if not m.autonomous
                        and (m.coasting or m.supporter is None)) / spec.ammo, 0.0, 1.0)
    soon = None
    for m in mine_live:
        if m.autonomous or m.target_key not in est_by_key:
            continue
        p = est_by_key[m.target_key]
        d = math.sqrt((p[0] - m.x) ** 2 + (p[1] - m.y) ** 2 + (p[2] - m.alt) ** 2)
        tt = max(0.0, d - view.active_range_m) / max(m.speed, 300.0)
        soon = tt if soon is None else min(soon, tt)
    own[14] = _clip(soon / 60.0, 0.0, 1.0) if soon is not None else 0.0
    own[15] = 1.0 if view.pair_open else 0.0
    last = max((m.launch_t for m in my_missiles), default=None)
    own[16] = 1.0 if last is None else _clip((view.t - last) / 60.0, 0.0, 1.0)
    maw = [c for c in view.cues if c.mode == MISSILE_ACTIVE]
    own[17] = _clip(len(maw) / 2.0, 0.0, 1.0)
    if maw:
        newest = max(maw, key=lambda c: (c.t_first, c.bearing))
        b = _wrap(newest.bearing - o.heading)
        own[18], own[19] = math.sin(b), math.cos(b)
    own[20] = _clip(view.red_kills / spec.n_red, 0.0, 1.0)
    obs[spec.own_slice] = own

    # ----------------------------------------------------------- contacts --
    info = ObsInfo(keys=[None] * spec.k, bearings=[None] * spec.k,
                   shot_ready=[False] * spec.k)
    mates_alive = {m.id for m in view.mates if m.alive}
    for i, c in enumerate(contacts):
        f = np.zeros(spec.contact_size)
        f[0] = 1.0
        if c.ranged:
            dx, dy, dz = c.est_pos[0] - o.x, c.est_pos[1] - o.y, c.est_pos[2] - o.alt
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            brg = math.atan2(dx, dy)
            f[1] = 1.0
            f[2] = _clip(r / spec.contact_range_m, 0.0, 1.5)
            f[5] = _clip(dz / spec.dalt_m, -1.5, 1.5)
        else:
            r, brg = None, c.rwr_bearing
        off = _wrap(brg - o.heading)
        f[3], f[4] = math.sin(off), math.cos(off)
        rmax = rne = None
        if c.ranged and c.vel is not None:
            vx, vy, vz = c.vel
            th = math.atan2(vx, vy)
            ts = math.hypot(vx, vy)
            f[6] = 1.0
            los_t2me = math.atan2(-dx, -dy)
            asp = _wrap(th - los_t2me)
            f[7], f[8] = math.sin(asp), math.cos(asp)
            closure = -(dx * vx + dy * vy + dz * vz) / max(r, 1.0) \
                + (dx * o.speed * math.sin(o.heading) + dy * o.speed * math.cos(o.heading)) \
                / max(r, 1.0)
            f[9] = _clip(closure / spec.closure_mps, -1.0, 1.0)
            f[10] = _clip(ts / spec.speed_mps, 0.0, 1.0)
            tgt_state = AircraftState(c.est_pos[0], c.est_pos[1], c.est_pos[2], th, ts)
            rmax, rne = env.for_states(own_state, tgt_state)
        f[11] = _clip(math.log10(1.0 + c.sigma_m) / 5.0, 0.0, 1.0) if c.ranged else 1.0
        f[12] = _clip(c.age_s / spec.coast_s, 0.0, 1.0)
        f[13] = 1.0 if c.own else 0.0
        if c.sensor in ("radar", "irst", "tri"):
            f[14 + ("radar", "irst", "tri").index(c.sensor)] = 1.0
        f[17] = 1.0 if c.own_fc else 0.0
        f[18] = 1.0 if c.remote_fc else 0.0
        if rmax is not None and rmax > 0.0:
            f[19] = _clip(r / rmax, 0.0, 2.0) / 2.0
            f[20] = _clip(r / rne, 0.0, 2.0) / 2.0 if rne > 0.0 else 1.0
            f[21] = _clip(rmax / spec.contact_range_m, 0.0, 1.5)
        else:
            f[19] = f[20] = 1.0
        f[22] = 1.0 if c.may_fire else 0.0
        ready = bool((c.own_fc or c.remote_fc) and rmax is not None and rmax > 0.0
                     and spec.min_launch_m <= r <= rmax
                     and abs(math.degrees(off)) <= spec.max_off_boresight_deg
                     and c.may_fire and o.ammo > 0)
        f[23] = 1.0 if ready else 0.0
        f[24] = _clip(sum(1 for m in live_blue if m.shooter == o.id and m.target_key == c.key)
                      / 2.0, 0.0, 1.0)
        f[25] = _clip(sum(1 for m in live_blue if m.shooter != o.id and m.target_key == c.key)
                      / 4.0, 0.0, 1.0)
        f[26] = _clip(sum(1 for mid, tk in mate_targets.items()
                          if mid != o.id and mid in mates_alive and tk == c.key) / 3.0,
                      0.0, 1.0)
        f[27] = c.rwr_mode / 3.0
        obs[spec.contact_slice(i)] = f
        info.keys[i] = c.key
        info.bearings[i] = brg
        info.shot_ready[i] = ready

    # ------------------------------------------------------------ wingmen --
    by_slot = {m.slot: m for m in view.mates}
    for j, s in enumerate(MATE_ORDER.get(o.slot, ())[: spec.n_mates]):
        m = by_slot.get(s)
        if m is None or not m.alive:
            continue
        w = np.zeros(spec.mate_size)
        dx, dy, dz = m.x - o.x, m.y - o.y, m.alt - o.alt
        w[0] = 1.0
        w[1] = _clip(math.sqrt(dx * dx + dy * dy + dz * dz) / spec.mate_range_m, 0.0, 1.5)
        b = _wrap(math.atan2(dx, dy) - o.heading) if (dx or dy) else 0.0
        w[2], w[3] = math.sin(b), math.cos(b)
        w[4] = _clip(dz / spec.dalt_m, -1.5, 1.5)
        rh = _wrap(m.heading - o.heading)
        w[5], w[6] = math.sin(rh), math.cos(rh)
        w[7] = _clip(m.speed / spec.speed_mps, 0.0, 1.0)
        w[8] = _clip(m.ammo / spec.ammo, 0.0, 1.0)
        w[9] = _clip(m.missiles_in_flight / spec.ammo, 0.0, 1.0)
        w[10] = 1.0 if m.threatened else 0.0
        w[11] = 1.0 if m.radar_emitting else 0.0
        tk = mate_targets.get(m.id)
        w[12] = 1.0 if tk is not None else 0.0
        w[13] = 1.0 if (tk is not None and tk == current_target) else 0.0
        obs[spec.mate_slice(j)] = w
    return obs, info
