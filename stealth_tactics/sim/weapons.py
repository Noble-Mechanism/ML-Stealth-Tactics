"""Missiles: Spec 3a point-mass fly-out, midcourse datalink support (Spec 2/2b)
and the shared Pk model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import numpy as np

from .aircraft import (Aircraft, AircraftState, distance_3d, bearing_to, _angle_diff,
                       ground_alt_m)
from .tracks import TrackStore
from .sensor_config import DEFAULT_SENSOR_CONFIG, SensorConfig, NM_M
from .missile_kinematics import KinState, advance_one, derived, speed_of_sound

# Per-observer track stores (Spec 1). A plain set is accepted for backward
# compatibility and is interpreted as the set of FIRE-CONTROL target ids.
TracksArg = Mapping[str, Union[TrackStore, Set[str]]]


# Default seeker-active range (15 NM). The live value is
# MissileKinematicsConfig.active_range_m; this constant mirrors the default.
ACQUISITION_RANGE_M = DEFAULT_SENSOR_CONFIG.missile_kinematics.active_range_m
# Shooter must keep target in forward hemisphere to provide midcourse support
SUPPORT_OFF_BORESIGHT_DEG = 90.0

# Terminal outcomes (events + ACMI bookmarks use the same labels)
OUTCOMES = ("hit", "miss", "miss_overshoot", "defeat_speed", "defeat_opening",
            "timeout", "lost_coast_timeout", "lost_basket", "target_dead", "ground")


def _vel(st: AircraftState) -> np.ndarray:
    """Horizontal velocity (ENU) from heading/speed; vertical ignored."""
    return np.array([st.speed_mps * np.sin(st.heading_rad),
                     st.speed_mps * np.cos(st.heading_rad), 0.0])


def has_fire_control(tracks: TracksArg, shooter_id: str, target_id: str) -> bool:
    store = tracks.get(shooter_id)
    if store is None:
        return False
    if isinstance(store, TrackStore):
        return store.is_fire_control(target_id)
    return target_id in store  # legacy set of fire-control ids


@dataclass
class Missile:
    id: str
    shooter_id: str
    target_id: str
    coalition: str
    pos: KinState                          # position + velocity + mass + flight time
    remaining_time_s: float = 180.0        # to the safety cap
    pk: float = 0.60                       # base Pk x support/coast factors so far
    alive: bool = True
    hit: bool = False
    # Midcourse datalink: False until within active range of the TRUE target
    autonomous: bool = False
    support_lost: bool = False
    # Spec 2: launched on a flightmate's fire-control track (id of that jet)
    remote_source: Optional[str] = None
    supporter_id: Optional[str] = None   # jet currently providing midcourse support
    outcome: str = ""                    # one of OUTCOMES
    # Spec 2b coast: aim point = last supported target position + last velocity * dt
    coasting: bool = False
    last_sup_t: Optional[float] = None
    last_sup_pos: Optional[np.ndarray] = None
    last_sup_vel: Optional[np.ndarray] = None
    t_unsupported: float = 0.0           # cumulative unsupported time before autonomy
    t_since_update: float = 0.0          # time since last midcourse update
    pk_factor: float = 1.0               # coast factor applied at the active point
    aim_err_at_auto_m: Optional[float] = None
    # Spec 3a kinematics / Pk bookkeeping
    launch_range_m: float = 0.0
    mach: float = 0.0                    # current Mach
    max_mach: float = 0.0
    burned_out: bool = False
    mach_at_active: Optional[float] = None
    tof_at_active: Optional[float] = None
    mach_end: Optional[float] = None     # at fuze / defeat
    sup_at_active: bool = False          # supported (not coasting) when going active
    post_sup_last_tf: float = 0.0        # flight time of last support after active
    support_dropped: bool = False        # x support_drop_factor applied
    f_support: float = 1.0
    f_endgame: Optional[float] = None
    pk_final: Optional[float] = None
    # Spec 3 shot log
    launch_t: float = 0.0
    off_nose_deg: float = 0.0            # |launch angle off the shooter's nose|
    a_pole_m: Optional[float] = None     # shooter-target range when the seeker goes active
    f_pole_m: Optional[float] = None     # shooter-target range at the end (None: shooter dead)
    end_t: Optional[float] = None

    @property
    def speed_mps(self) -> float:
        return self.pos.speed_mps

    @property
    def tof(self) -> float:
        return self.pos.tf


class WeaponModel:
    """
    Spec 3a missile, same for both sides (docs/specs/03a-missile-kinematics.md).

    - Launch gate: target alive, ammo, min range <= R <= table Rmax for the
      current geometry (``missile_envelope``: altitude, shooter Mach, aspect,
      target Mach and signed launch angle off the nose), within 60 deg of the
      nose, and a
      FIRE-CONTROL track (own, or remote for Blue; gated by World / tactics).
    - Point-mass 3-D fly-out from the shooter's position and velocity, integrated
      at ``MissileKinematicsConfig.dt_s`` inside each world step. PN guidance at
      the supported (or coasting, Spec 2b) aim point until the seeker goes active
      within ``active_range_m`` of the TRUE target, then at the target.
    - Kinematic defeat after burnout: below ``defeat_min_mach`` (defeat_speed) or
      opening (defeat_opening; miss_overshoot if it passed within 1 km outside
      the fuze). ``timeout`` = 180 s safety cap.
    - Pk = base_pk x f_coast (coasting at the active point) x 0.90 (supported to
      active, then support dropped >= 2 s before impact) x f_E(Mach at impact).
    """

    def __init__(self, rng: np.random.Generator,
                 config: Optional[SensorConfig] = None) -> None:
        self.rng = rng
        self.cfg = config or DEFAULT_SENSOR_CONFIG
        self.kcfg = self.cfg.missile_kinematics
        self._kd = derived(self.kcfg)
        self._envelope = None
        self._prev_pos: Dict[str, Tuple[float, float, float, float]] = {}
        self._missile_seq = 0
        # Weapon events since last drain (World moves them into its event log)
        self.events: List[dict] = []
        self._by_id: Dict[str, Aircraft] = {}

    @property
    def envelope(self):
        """Rmax / Rne lookup table (built from the fly-out model, cached)."""
        if self._envelope is None:
            from .missile_envelope import get_envelope
            self._envelope = get_envelope(self.kcfg)
        return self._envelope

    def rmax_m(self, shooter: Aircraft, target: Aircraft) -> float:
        """Table Rmax (m) for the current shooter/target geometry."""
        return self.envelope.rmax_for_states(shooter.state, target.state)

    def rne_m(self, shooter: Aircraft, target: Aircraft) -> float:
        """Table no-escape range (target turns cold at launch), metres."""
        return self.envelope.for_states(shooter.state, target.state)[1]

    def envelope_for(self, shooter: Aircraft, target: Aircraft) -> Tuple[float, float]:
        """(Rmax, Rne) in metres; candidate network inputs (Spec 5)."""
        return self.envelope.for_states(shooter.state, target.state)

    def can_shoot(self, shooter: Aircraft, target: Aircraft) -> bool:
        if not shooter.state.alive or not target.state.alive:
            return False
        if shooter.ammo <= 0:
            return False
        kc = self.kcfg
        d = distance_3d(shooter.state, target.state)
        if d < kc.min_launch_range_m or d > self.rmax_m(shooter, target):
            return False
        brg = bearing_to(shooter.state, target.state)
        off = abs(_angle_diff(brg, shooter.state.heading_rad))
        return off <= np.deg2rad(kc.max_off_boresight_deg)

    def launch(self, shooter: Aircraft, target: Aircraft,
               remote_source: Optional[str] = None) -> Optional[Missile]:
        """remote_source: flightmate whose FC track is used (Spec 2), else None."""
        if not self.can_shoot(shooter, target):
            return None
        return self.spawn(shooter, target, remote_source)

    def spawn(self, shooter: Aircraft, target: Aircraft,
              remote_source: Optional[str] = None) -> Missile:
        """Create a missile without launch gating (analysis scenarios, tests)."""
        kc = self.kcfg
        shooter.ammo -= 1
        self._missile_seq += 1
        mid = f"M{self._missile_seq:04d}"
        st = shooter.state
        v = _vel(st)
        kin = KinState(st.x, st.y, st.alt, v[0], v[1], v[2], kc.launch_mass_kg, 0.0)
        rng = distance_3d(st, target.state)
        # Already inside the active range at launch -> starts autonomous (supported)
        autonomous = rng <= kc.active_range_m
        m = Missile(
            id=mid,
            shooter_id=shooter.id,
            target_id=target.id,
            coalition=shooter.coalition.value,
            pos=kin,
            remaining_time_s=kc.max_flight_time_s,
            pk=kc.base_pk,
            autonomous=autonomous,
            remote_source=remote_source,
            supporter_id=shooter.id,
            last_sup_pos=target.state.position(),
            last_sup_vel=_vel(target.state),
            launch_range_m=rng,
            sup_at_active=autonomous,
        )
        m.mach = m.max_mach = kin.mach()
        m.off_nose_deg = abs(math.degrees(_angle_diff(bearing_to(st, target.state),
                                                      st.heading_rad)))
        if autonomous:
            m.mach_at_active, m.tof_at_active = m.mach, 0.0
            m.a_pole_m = rng          # starts active: a-pole recorded at launch
        return m

    @staticmethod
    def has_midcourse_support(
        shooter: Aircraft,
        target: Aircraft,
        tracks: TracksArg,
    ) -> bool:
        """
        Shooter midcourse support rule:
        - Shooter alive
        - Shooter holds a FIRE-CONTROL quality track on the target (Spec 1;
          IRST / RWR / immature radar tracks do not count)
        - Target within ±SUPPORT_OFF_BORESIGHT_DEG of shooter nose
          (pure egress / turning cold drops support)
        """
        if shooter is None or not shooter.state.alive:
            return False
        if not target.state.alive:
            return False
        if not has_fire_control(tracks, shooter.id, target.id):
            return False
        brg = bearing_to(shooter.state, target.state)
        off = abs(_angle_diff(brg, shooter.state.heading_rad))
        return off <= np.deg2rad(SUPPORT_OFF_BORESIGHT_DEG)

    def _qualifies(self, jet: Optional[Aircraft], target: Aircraft, m: Missile,
                   tracks: TracksArg) -> bool:
        if jet is None or not self.has_midcourse_support(jet, target, tracks):
            return False
        if jet.id == m.shooter_id:
            return True
        # Handoff supporter needs a working weapon link to the missile
        d = float(np.sqrt((jet.state.x - m.pos.x) ** 2 + (jet.state.y - m.pos.y) ** 2
                          + (jet.state.alt - m.pos.alt) ** 2))
        return d <= self.cfg.datalink.weapon_link_range_m

    def pick_supporter(self, m: Missile, target: Aircraft, aircraft_by_id: dict,
                       tracks: TracksArg) -> Optional[str]:
        """
        Red/asymmetric policy: only the shooter can support (Spec 1 rule).
        Blue (or symmetric) policy: any flightmate with its own FC track, target
        within 90 deg of its nose and a working link. Preference: current
        supporter (no flapping) -> shooter -> others by id.
        """
        shooter = aircraft_by_id.get(m.shooter_id)
        if not self.cfg.datalink.weapons_policy_shared(m.coalition):
            return m.shooter_id if self._qualifies(shooter, target, m, tracks) else None
        order = []
        for jid in [m.supporter_id, m.shooter_id] + sorted(aircraft_by_id):
            if jid and jid not in order:
                order.append(jid)
        for jid in order:
            jet = aircraft_by_id.get(jid)
            if jet is None or jet.coalition.value != m.coalition:
                continue
            if self._qualifies(jet, target, m, tracks):
                return jid
        return None

    def _go_autonomous(self, m: Missile, tgt: Aircraft, dist: float, t: float) -> bool:
        """Seeker goes active within active_range_m of the TRUE target. Returns
        False if the missile is lost (aim point outside the seeker basket)."""
        mc = self.cfg.missile
        if m.coasting and m.last_sup_pos is not None:
            aim = m.last_sup_pos + m.last_sup_vel * m.t_since_update
            err = float(np.linalg.norm(aim - tgt.state.position()))
        else:
            err = 0.0
        m.aim_err_at_auto_m = err
        m.mach_at_active, m.tof_at_active = m.mach, m.pos.tf
        if err > mc.seeker_basket_m:
            m.alive = False
            m.outcome = "lost_basket"
            self._event(t, "lost_basket", m,
                        f"{m.id} lost: outside seeker basket (aim error {err / 1000:.1f} km"
                        f" > {mc.seeker_basket_m / 1000:.1f} km)", range_m=dist)
            return False
        if m.coasting:
            m.pk_factor = mc.coast_pk_factor(err, m.t_since_update)
        else:
            m.pk_factor = 1.0
            m.sup_at_active = True
            m.post_sup_last_tf = m.pos.tf
        m.pk *= m.pk_factor
        m.autonomous = True
        m.coasting = False
        shooter = self._by_id.get(m.shooter_id) if self._by_id else None
        if shooter is not None and shooter.state.alive:
            m.a_pole_m = distance_3d(shooter.state, tgt.state)
        rng_nm = self.kcfg.active_range_m / NM_M
        self._event(t, "autonomous", m,
                    f"{m.id} active (within {rng_nm:.0f} NM of {m.target_id}; Mach "
                    f"{m.mach:.2f}; Pk factor {m.pk_factor:.2f}, aim error {err:.0f} m, "
                    f"{m.t_since_update:.1f} s since last update)",
                    range_m=dist, pk_factor=m.pk_factor, mach=m.mach, aim_err_m=err)
        return True

    def _event(self, t, etype, m: Missile, text, jet=None, **kw) -> None:
        self.events.append({"t": t, "type": etype, "missile": m.id,
                            "observer": jet or m.shooter_id, "target": m.target_id,
                            "text": text, **kw})

    def _end(self, m: Missile, t: float, outcome: str, text: str, dist: float) -> None:
        m.alive = False
        m.outcome = outcome
        m.mach_end = m.mach
        self._event(t, outcome, m, text, range_m=dist, mach=m.mach, tof=m.pos.tf)

    def _midcourse_support(self, m: Missile, tgt: Aircraft, aircraft_by_id: dict,
                           tracks: TracksArg, dist: float, t: float, dt: float) -> bool:
        """Spec 2/2b support / coast bookkeeping before the seeker is active.
        Returns False if the missile was lost (coast timeout)."""
        mc = self.cfg.missile
        sup = self.pick_supporter(m, tgt, aircraft_by_id, tracks)
        if sup is None:
            if not m.coasting:
                m.coasting = True
                m.support_lost = True
                self._event(t, "support_lost", m, f"{m.id} support lost (coasting)",
                            jet=m.supporter_id, range_m=dist)
                m.supporter_id = None
            m.t_unsupported += dt
            m.t_since_update += dt
            if m.t_since_update > mc.coast_timeout_s + 1e-9:
                self._end(m, t, "lost_coast_timeout",
                          f"{m.id} lost: coast timeout ({m.t_since_update:.1f} s unsupported)",
                          dist)
                return False
        else:
            if m.coasting:
                m.coasting = False
                self._event(t, "support_regained", m,
                            f"{m.id} support regained by {sup} after "
                            f"{m.t_since_update:.1f} s", jet=sup,
                            range_m=dist, previous=m.supporter_id)
            elif sup != m.supporter_id:
                self._event(t, "support_handoff", m, f"{m.id} support handed to {sup}",
                            jet=sup, range_m=dist, previous=m.supporter_id)
            m.supporter_id = sup
            m.t_since_update = 0.0
            m.last_sup_t = t
            m.last_sup_pos = tgt.state.position()
            m.last_sup_vel = _vel(tgt.state)
        return True

    def _terminal_support(self, m: Missile, tgt: Aircraft, aircraft_by_id: dict,
                          tracks: TracksArg, dist: float, t: float) -> None:
        """Supported to active: if support then drops (gap >= support_gap_tol_s,
        any qualifying flightmate counts), Pk x support_drop_factor once."""
        if not m.sup_at_active or m.support_dropped:
            return
        kc = self.kcfg
        sup = self.pick_supporter(m, tgt, aircraft_by_id, tracks)
        if sup is not None:
            m.post_sup_last_tf = m.pos.tf
            m.supporter_id = sup
            return
        gap = m.pos.tf - m.post_sup_last_tf
        if gap >= kc.support_gap_tol_s - 1e-9:
            m.support_dropped = True
            m.f_support = kc.support_drop_factor
            m.pk *= kc.support_drop_factor
            self._event(t, "support_dropped", m,
                        f"{m.id} support dropped after active ({gap:.1f} s gap): Pk x"
                        f"{kc.support_drop_factor:.2f}", jet=m.supporter_id, range_m=dist)
            m.supporter_id = None

    def step(
        self,
        missiles: List[Missile],
        aircraft_by_id: dict,
        dt: float,
        tracks: Optional[TracksArg] = None,
        t: float = 0.0,
    ) -> List[str]:
        """Advance missiles over [t, t+dt] (aircraft are already at t+dt);
        return list of destroyed aircraft ids."""
        tracks = tracks or {}
        self._by_id = aircraft_by_id
        killed: List[str] = []
        kc = self.kcfg
        kd = self._kd
        nsub = max(1, int(math.ceil(dt / kc.dt_s - 1e-9)))
        h = dt / nsub
        act2 = kc.active_range_m ** 2
        fuze2 = kc.proximity_fuze_m ** 2
        over2 = kc.overshoot_range_m ** 2
        burn = kc.burn_time_s + 1e-9
        for m in missiles:
            if not m.alive:
                continue
            tgt = aircraft_by_id.get(m.target_id)
            if tgt is None or not tgt.state.alive:
                m.alive = False
                m.outcome = m.outcome or "target_dead"
                self._record_end(m, t)
                continue

            k, ts = m.pos, tgt.state
            dist = math.sqrt((ts.x - k.x) ** 2 + (ts.y - k.y) ** 2 + (ts.alt - k.alt) ** 2)
            # Acquire / drop / regain midcourse support before flying
            if not m.autonomous:
                if dist <= kc.active_range_m:
                    if not self._go_autonomous(m, tgt, dist, t):
                        self._record_end(m, t)
                        continue
                elif not self._midcourse_support(m, tgt, aircraft_by_id, tracks,
                                                 dist, t, dt):
                    self._record_end(m, t)
                    continue
            else:
                self._terminal_support(m, tgt, aircraft_by_id, tracks, dist, t)

            # Target motion over the step: straight chord from its position at the
            # start of the step (recorded last step) to its current, already
            # integrated position (captures turns and climbs); first step after
            # launch: current heading / speed, level.
            tx_end, ty_end, tz_end = ts.x, ts.y, ts.alt
            prev = self._prev_pos.get(m.target_id)
            if prev is not None and abs(prev[0] - t) < 1e-6:
                tvx, tvy, tvz = ((tx_end - prev[1]) / dt, (ty_end - prev[2]) / dt,
                                 (tz_end - prev[3]) / dt)
            else:
                tvx = ts.speed_mps * math.sin(ts.heading_rad)
                tvy = ts.speed_mps * math.cos(ts.heading_rad)
                tvz = 0.0
            coast = (not m.autonomous) and m.coasting and m.last_sup_pos is not None
            if coast:
                sp, sv = m.last_sup_pos, m.last_sup_vel
                spx, spy, spz = float(sp[0]), float(sp[1]), float(sp[2])
                svx, svy, svz = float(sv[0]), float(sv[1]), float(sv[2])
                tsu0 = m.t_since_update - dt
            ended = False
            for j in range(nsub):
                back = dt - j * h
                tx0, ty0, tz0 = tx_end - tvx * back, ty_end - tvy * back, tz_end - tvz * back
                r0x, r0y, r0z = tx0 - k.x, ty0 - k.y, tz0 - k.alt
                if coast and not m.autonomous:
                    tau = tsu0 + j * h
                    advance_one(k, spx + svx * tau, spy + svy * tau, spz + svz * tau,
                                svx, svy, svz, h, kd)
                else:
                    advance_one(k, tx0, ty0, tz0, tvx, tvy, tvz, h, kd)
                tx1, ty1, tz1 = tx0 + tvx * h, ty0 + tvy * h, tz0 + tvz * h
                r1x, r1y, r1z = tx1 - k.x, ty1 - k.y, tz1 - k.alt
                V = math.sqrt(k.vx * k.vx + k.vy * k.vy + k.vz * k.vz)
                m.mach = V / speed_of_sound(k.alt)
                if m.mach > m.max_mach:
                    m.max_mach = m.mach
                rng2 = r1x * r1x + r1y * r1y + r1z * r1z
                # proximity fuze: closest approach during the sub-step
                dx, dy, dz = r1x - r0x, r1y - r0y, r1z - r0z
                dd = dx * dx + dy * dy + dz * dz
                s_ = 0.0 if dd < 1e-12 else min(1.0, max(0.0, -(r0x * dx + r0y * dy
                                                                + r0z * dz) / dd))
                cx, cy, cz = r0x + s_ * dx, r0y + s_ * dy, r0z + s_ * dz
                if cx * cx + cy * cy + cz * cz <= fuze2:
                    self._fuze(m, tgt, t, killed)
                    ended = True
                    break
                if k.alt <= ground_alt_m(k.x, k.y):
                    self._end(m, t, "ground", f"{m.id} hit the ground (Mach {m.mach:.2f})",
                              math.sqrt(rng2))
                    ended = True
                    break
                if not m.autonomous and rng2 <= act2:
                    if not self._go_autonomous(m, tgt, math.sqrt(rng2), t):
                        ended = True
                        break
                if k.tf > burn:
                    if not m.burned_out:
                        m.burned_out = True
                        self._event(t, "burnout", m, f"{m.id} motor burnout at Mach "
                                    f"{m.mach:.2f}", range_m=math.sqrt(rng2), mach=m.mach)
                    if m.mach < kc.defeat_min_mach:
                        self._end(m, t, "defeat_speed",
                                  f"{m.id} kinematic defeat: Mach {m.mach:.2f} < "
                                  f"{kc.defeat_min_mach:.1f} after {k.tf:.1f} s",
                                  math.sqrt(rng2))
                        ended = True
                        break
                    if kc.defeat_on_opening and (r1x * (tvx - k.vx) + r1y * (tvy - k.vy)
                                                 + r1z * (tvz - k.vz)) > 0.0:
                        rr = math.sqrt(rng2)
                        if rng2 < over2:
                            self._end(m, t, "miss_overshoot",
                                      f"{m.id} overshoot: passed {rr:.0f} m from "
                                      f"{m.target_id} outside the fuze", rr)
                        else:
                            self._end(m, t, "defeat_opening",
                                      f"{m.id} kinematic defeat: no longer closing "
                                      f"(Mach {m.mach:.2f}, {rr / NM_M:.1f} NM)", rr)
                        ended = True
                        break
                m.remaining_time_s -= h
                if m.remaining_time_s <= 1e-9:
                    self._end(m, t, "timeout",
                              f"{m.id} timed out ({kc.max_flight_time_s:.0f} s safety cap)",
                              math.sqrt(rng2))
                    ended = True
                    break
            if ended:
                self._record_end(m, t)
                continue
        # target positions at the end of this step = start of the next one
        t_end = t + dt
        for uid, ac in aircraft_by_id.items():
            st = ac.state
            self._prev_pos[uid] = (t_end, st.x, st.y, st.alt)
        return killed

    def _record_end(self, m: Missile, t: float) -> None:
        """Spec 3 shot log: f-pole = shooter-target range when the missile ends
        (None if the shooter is dead)."""
        if m.end_t is not None:
            return
        m.end_t = t
        sh = self._by_id.get(m.shooter_id)
        tg = self._by_id.get(m.target_id)
        if sh is not None and sh.state.alive and tg is not None:
            m.f_pole_m = distance_3d(sh.state, tg.state)

    def _fuze(self, m: Missile, tgt: Aircraft, t: float, killed: List[str]) -> None:
        kc = self.kcfg
        m.f_endgame = kc.endgame_factor(m.mach)
        m.pk_final = m.pk * m.f_endgame
        m.mach_end = m.mach
        info = (f"Mach {m.mach:.2f}, TOF {m.pos.tf:.1f} s, Pk {m.pk_final:.2f} = "
                f"{kc.base_pk:.2f} x coast {m.pk_factor:.2f} x support {m.f_support:.2f}"
                f" x energy {m.f_endgame:.2f}")
        m.alive = False
        if self.rng.random() < m.pk_final:
            m.hit = True
            tgt.state.alive = False
            killed.append(tgt.id)
            m.outcome = "hit"
            self._event(t, "hit", m, f"{m.id} HIT {m.target_id} ({info})", range_m=0.0,
                        mach=m.mach, tof=m.pos.tf, pk=m.pk_final)
        else:
            m.outcome = "miss"
            self._event(t, "miss", m, f"{m.id} missed {m.target_id} (Pk roll; {info})",
                        range_m=0.0, mach=m.mach, tof=m.pos.tf, pk=m.pk_final)
