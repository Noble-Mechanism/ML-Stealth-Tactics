"""Spec 5 A: the restricted Blue view (truth firewall).

``build_blue_view(world, ac)`` copies ONLY whitelisted, perception-side data
into frozen plain dataclasses: own and flightmate state (Blue side), the jet's
fused picture (estimate, sigma, sensor, age, radar velocity estimate), own /
remote fire-control availability, doctrine permission, RWR cues, Blue missiles
(own weapon datalink), confirmed kills and time. Red ``Aircraft`` objects, track
components (they carry the diagnostic ``true_range_m``), Pk fields and
``primary_fc_target`` are never copied, so nothing downstream can reach them.
The missile envelope (a pure lookup table) is passed by reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.sim.rwr import MISSILE_ACTIVE, MODE_RANK

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class OwnView:
    id: str
    slot: int                 # 0-based slot (B1 -> 0)
    x: float
    y: float
    alt: float
    heading: float
    speed: float
    ammo: int
    radar_emitting: bool
    doctrine: str


@dataclass(frozen=True)
class MateView:
    id: str
    slot: int
    alive: bool
    x: float
    y: float
    alt: float
    heading: float
    speed: float
    ammo: int
    radar_emitting: bool
    missiles_in_flight: int
    threatened: bool          # a missile-active warning on its RWR


@dataclass(frozen=True)
class ContactView:
    key: str                  # internal track key (never an input)
    ranged: bool              # False = bearing-only (RWR) contact
    est_pos: Optional[Vec3]
    sigma_m: Optional[float]
    sensor: str               # "radar" | "irst" | "tri" | "rwr"
    own: bool
    age_s: float
    vel: Optional[Vec3]       # freshest radar velocity estimate (own or received)
    rwr_bearing: Optional[float]
    own_fc: bool
    remote_fc: bool
    may_fire: bool            # firing doctrine allows a launch at it now
    rwr_mode: int             # 0 none, 1 search, 2 lock, 3 support (from this emitter)
    first_t: float


@dataclass(frozen=True)
class MissileView:
    id: str
    shooter: str
    target_key: str
    alive: bool
    autonomous: bool
    coasting: bool
    supporter: Optional[str]
    launch_t: float
    x: float
    y: float
    alt: float
    speed: float


@dataclass(frozen=True)
class CueView:
    mode: str
    bearing: float
    t_first: float


@dataclass(frozen=True)
class BlueView:
    t: float
    own: OwnView
    mates: Tuple[MateView, ...]          # every other Blue (dead ones alive=False)
    contacts: Tuple[ContactView, ...]
    missiles: Tuple[MissileView, ...]    # all Blue missiles (alive and resolved)
    cues: Tuple[CueView, ...]            # own RWR cues
    red_kills: int
    pair_open: bool
    envelope: object                     # MissileEnvelope (lookup table)
    active_range_m: float


def _freshest_radar_vel(tr) -> Optional[Vec3]:
    best = None
    comps = []
    c = tr.components.get("radar")
    if c is not None:
        comps.append(c)
    for sender in sorted(tr.remote):
        c = tr.remote[sender].get("radar")
        if c is not None:
            comps.append(c)
    for c in comps:
        if c.est_vel is not None and (best is None or c.last_t > best.last_t):
            best = c
    return tuple(float(v) for v in best.est_vel) if best is not None else None


def build_blue_view(world, ac) -> BlueView:
    if ac.coalition != Coalition.BLUE:
        raise ValueError("blue_view is for Blue jets only")
    t = world.time_s
    blues = [a for a in world.aircraft if a.coalition == Coalition.BLUE]
    slot_of = {a.id: i for i, a in enumerate(blues)}
    st = ac.state
    own = OwnView(ac.id, slot_of[ac.id], float(st.x), float(st.y), float(st.alt),
                  float(st.heading_rad), float(st.speed_mps), int(ac.ammo),
                  bool(ac.radar_emitting), world.doctrine_of(ac))
    blue_missiles = [m for m in world.missiles if m.coalition == Coalition.BLUE.value]
    mates = []
    for b in blues:
        if b.id == ac.id:
            continue
        cues = world.rwr.get(b.id, []) if b.state.alive else []
        mates.append(MateView(
            b.id, slot_of[b.id], bool(b.state.alive), float(b.state.x), float(b.state.y),
            float(b.state.alt), float(b.state.heading_rad), float(b.state.speed_mps),
            int(b.ammo), bool(b.radar_emitting),
            sum(1 for m in blue_missiles if m.alive and m.shooter_id == b.id),
            any(c.mode == MISSILE_ACTIVE for c in cues)))
    my_cues = list(world.rwr.get(ac.id, []))
    mode_by_emitter = {}
    for c in my_cues:
        if c.mode != MISSILE_ACTIVE:
            mode_by_emitter[c.emitter_id] = max(mode_by_emitter.get(c.emitter_id, 0),
                                                MODE_RANK[c.mode])
    store = world.tracks.get(ac.id)
    dl = world.sensor_cfg.datalink
    coal = ac.coalition.value
    contacts = []
    if store is not None:
        for tid in sorted(store.tracks):
            tr = store.tracks[tid]
            f = store.fused.get(tid)
            remote = (store.remote_fc_source(tid, t, dl.remote_fc_max_age_s(coal))
                      if dl.weapons_policy_shared(coal) else None)
            common = dict(key=tid, own_fc=bool(tr.fire_control), remote_fc=remote is not None,
                          may_fire=bool(world.may_fire_at(ac, tid)),
                          rwr_mode=int(mode_by_emitter.get(tid, 0)), first_t=float(tr.first_t))
            if f is not None:
                contacts.append(ContactView(
                    ranged=True, est_pos=tuple(float(v) for v in f.est_pos),
                    sigma_m=float(f.sigma_m), sensor=f.sensor, own=bool(f.own),
                    age_s=float(f.age_s), vel=_freshest_radar_vel(tr), rwr_bearing=None,
                    **common))
            elif "rwr" in tr.components:
                c = tr.components["rwr"]
                contacts.append(ContactView(
                    ranged=False, est_pos=None, sigma_m=None, sensor="rwr", own=True,
                    age_s=float(max(0.0, t - c.last_t)), vel=None,
                    rwr_bearing=float(c.bearing_rad), **common))
    missiles = tuple(MissileView(
        m.id, m.shooter_id, m.target_id, bool(m.alive), bool(m.autonomous), bool(m.coasting),
        m.supporter_id, float(m.launch_t), float(m.pos.x), float(m.pos.y), float(m.pos.alt),
        float(m.speed_mps)) for m in blue_missiles)
    blue_ids = set(slot_of)
    red_kills = sum(1 for e in world.events
                    if e.get("type") == "kill" and e.get("target") not in blue_ids)
    cues = tuple(CueView(c.mode, float(c.bearing), float(c.t_first)) for c in my_cues)
    return BlueView(t=float(t), own=own, mates=tuple(mates), contacts=tuple(contacts),
                    missiles=missiles, cues=cues, red_kills=red_kills,
                    pair_open=bool(world.ssa_pair_open(ac)),
                    envelope=world.weapons.envelope,
                    active_range_m=float(world.weapons.kcfg.active_range_m))
