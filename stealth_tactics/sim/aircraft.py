"""Point-mass aircraft state and kinematics (Spec 8 energy model)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .missile_kinematics import G0, atmosphere
from .sensor_config import (
    BLUE_ENERGY, DEFAULT_SENSOR_CONFIG as _SC, F35_KEY, RED_ENERGY, RED_KEY,
    AircraftEnergyConfig,
)

# Spec 3 (approved change B): flat ground at 0 m MSL (no terrain model yet) and
# a hard 100 m AGL floor for every aircraft. Missiles that reach the ground are
# lost (outcome ``ground``, sim/weapons.py).
GROUND_ALT_M = 0.0
ALT_FLOOR_AGL_M = 100.0
RHO_SL = 1.225
TURN_REVERSAL_BAND_RAD = math.radians(10.0)


def ground_alt_m(x: float = 0.0, y: float = 0.0) -> float:
    """Terrain height (m MSL) at (x, y). Flat ground until a terrain model exists."""
    return GROUND_ALT_M


def alt_floor_m(x: float = 0.0, y: float = 0.0) -> float:
    """Lowest altitude any aircraft may fly at (x, y): ground + 100 m AGL."""
    return ground_alt_m(x, y) + ALT_FLOOR_AGL_M


class AircraftType(str, Enum):
    # TacView-friendly display name. Performance/RCS below are generic LO
    # placeholders labeled F-35 for visualization only — NOT real F-35 data.
    F35 = "F-35"
    RED_FIGHTER = "RedFighter"


class Coalition(str, Enum):
    BLUE = "Blue"
    RED = "Red"


@dataclass
class AircraftTypeParams:
    """Generic flight / signature params (unclassified placeholders).

    Spec 8: turn / accel come from ``energy`` (n-limited turn, drag, density
    thrust). ``max_turn_rate_deg_s`` / ``max_speed_mps`` remain as legacy
    command references used by doctrines; the integrator enforces energy limits.
    """

    max_speed_mps: float = 350.0  # ~Mach 1.0 sea level-ish (command reference)
    min_speed_mps: float = 80.0
    cruise_speed_mps: float = 250.0
    max_turn_rate_deg_s: float = 12.0  # legacy; energy model uses n_max
    max_climb_rate_mps: float = 80.0
    max_alt_m: float = 15000.0
    min_alt_m: float = 100.0
    rcs_factor: float = 1.0  # relative RCS (stealth << 1)
    radar_range_m: float = 80000.0
    # Missile range and Pk are no longer per type (Spec 3a): both sides carry the
    # same missile (SensorConfig.missile_kinematics; Rmax from the envelope table).
    ammo: int = 4
    energy: AircraftEnergyConfig = field(default_factory=AircraftEnergyConfig)


# Unclassified generic LO placeholders labeled F-35 for TacView visualization only.
# These are NOT real F-35 performance or RCS numbers — same numeric stealth
# advantage as the former BlueStealth placeholders (rcs_factor=0.05, etc.).
F35_PARAMS = AircraftTypeParams(
    max_speed_mps=340.0,
    min_speed_mps=BLUE_ENERGY.min_speed_mps,
    cruise_speed_mps=260.0,
    max_turn_rate_deg_s=11.0,
    max_climb_rate_mps=90.0,
    max_alt_m=BLUE_ENERGY.max_alt_m,
    # Sensor numbers live in sensor_config.py (single source of truth).
    rcs_factor=_SC.signature.tables[F35_KEY][0][1],  # nose-on value of LO table
    radar_range_m=_SC.radar.ref_range_m[F35_KEY],
    ammo=4,
    energy=BLUE_ENERGY,
)

RED_FIGHTER_PARAMS = AircraftTypeParams(
    max_speed_mps=360.0,
    min_speed_mps=RED_ENERGY.min_speed_mps,
    cruise_speed_mps=255.0,
    max_turn_rate_deg_s=13.0,
    max_climb_rate_mps=112.5,          # Spec 8c: 1.25 x Blue
    max_alt_m=RED_ENERGY.max_alt_m,
    rcs_factor=_SC.signature.isotropic_rcs[RED_KEY],
    radar_range_m=_SC.radar.ref_range_m[RED_KEY],
    ammo=4,
    energy=RED_ENERGY,
)


@dataclass
class AircraftState:
    """ENU position (m), heading (rad from North, clockwise), speed, alt."""

    x: float = 0.0  # East (m)
    y: float = 0.0  # North (m)
    alt: float = 8000.0  # Up (m)
    heading_rad: float = 0.0  # 0 = North, increases clockwise toward East
    speed_mps: float = 250.0
    alive: bool = True

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.alt], dtype=float)

    def copy(self) -> AircraftState:
        return AircraftState(
            x=self.x,
            y=self.y,
            alt=self.alt,
            heading_rad=self.heading_rad,
            speed_mps=self.speed_mps,
            alive=self.alive,
        )


@dataclass
class Aircraft:
    id: str
    name: str  # ship callsign → ACMI Pilot=
    ac_type: AircraftType
    coalition: Coalition
    state: AircraftState
    params: AircraftTypeParams
    # TacView database Name= (F-35A matches Lightning II; callsign stays in Pilot=)
    type_name: str = ""
    ammo: int = 4
    locked_target: Optional[str] = None
    role: str = "fighter"
    # Commanded intents (set by tactics controller each tick)
    cmd_heading_rad: float = 0.0
    cmd_speed_mps: float = 250.0
    cmd_alt_m: float = 8000.0
    cmd_fire: bool = False
    fire_target: Optional[str] = None
    # Spec 3: radar emission (always True here; a network output in spec 5).
    radar_emitting: bool = True
    # Spec 3: firing doctrine override ("shoot_assess_shoot" |
    # "shoot_shoot_assess" | "legacy"); None -> SimConfig per-coalition default.
    firing_doctrine: Optional[str] = None
    # Spec 3: defensive state for logs / shot table ("hot", "defending", "cold",
    # "pressing", "departed"); set by the controllers.
    defense_state: str = "hot"
    # Spec 3 D9: left the fight for good (never fires; early end when every live
    # Red jet is departed and no missile is in flight).
    departed: bool = False
    # Spec 8: sign of the turn in progress (+1 right, -1 left, 0 none). Used for
    # turn-direction hysteresis when the commanded heading is ~180 deg away.
    turn_dir: float = 0.0
    # Spec 8d: flight-path angle (rad, + = climbing). A state: it changes only
    # at the rate the vertical load factor allows (see integrate_aircraft).
    gamma_rad: float = 0.0
    # Spec 8d: optional direct climb-rate command (m/s). None (default) = the
    # altitude hold on cmd_alt_m sets the desired climb rate. Analysis tools
    # use it to fly climb schedules (constant gamma, Mach hold).
    cmd_climb_rate_mps: Optional[float] = None
    # Spec 8e: lift-vector state. bank_rad = bank angle phi of the lift
    # vector around the velocity vector (rad, + = right wing down, |phi| > 90
    # deg = inverted); load_factor = its magnitude n (g, >= 0). Both change at
    # the roll-rate / g-onset limits (AircraftEnergyConfig).
    bank_rad: float = 0.0
    load_factor: float = 1.0

    def __post_init__(self) -> None:
        self.ammo = self.params.ammo
        self.cmd_heading_rad = self.state.heading_rad
        self.cmd_speed_mps = self.state.speed_mps
        self.cmd_alt_m = self.state.alt
        if not self.type_name:
            if self.ac_type == AircraftType.F35:
                self.type_name = "F-35A"
            else:
                # Keep Red ACMI Name= callsign (unchanged behavior)
                self.type_name = self.name

    @staticmethod
    def make_blue(uid: str, name: str, state: AircraftState) -> Aircraft:
        return Aircraft(
            id=uid,
            name=name,
            ac_type=AircraftType.F35,
            coalition=Coalition.BLUE,
            state=state,
            params=F35_PARAMS,
        )

    @staticmethod
    def make_red(uid: str, name: str, state: AircraftState) -> Aircraft:
        return Aircraft(
            id=uid,
            name=name,
            ac_type=AircraftType.RED_FIGHTER,
            coalition=Coalition.RED,
            state=state,
            params=RED_FIGHTER_PARAMS,
        )


def _alt_hold_rate(err: float, e: AircraftEnergyConfig, max_climb: float,
                   cos_g: float, n_cap: float, max_descent: Optional[float] = None,
                   t_delay: float = 0.0) -> float:
    """Spec 8d altitude hold: desired climb rate (m/s) for an altitude error.

    |vs| = min(max rate, |err| / tau, v_stop). a_stop is a fraction of the
    jet's ability to arrest the vertical speed: push-over capability
    (cos gamma - n_pushover_min) when climbing to the target, pull-up
    capability (n_cap - cos gamma) when descending to it. Spec 8e: v_stop
    also covers a reaction delay t_delay (roll back upright / g onset):
    v t_delay + v^2 / (2 a_stop) <= |err| (t_delay = 0 gives the 8d law
    sqrt(2 a_stop |err|)). max_descent (default max_climb) caps descents."""
    if err == 0.0:
        return 0.0
    cap = (cos_g - e.n_pushover_min) if err > 0.0 else (n_cap - cos_g)
    a_stop = e.alt_hold_decel_frac * G0 * max(cap, 0.1)
    ae = abs(err)
    if t_delay > 0.0:
        v_stop = a_stop * (math.sqrt(t_delay * t_delay + 2.0 * ae / a_stop) - t_delay)
    else:
        v_stop = math.sqrt(2.0 * a_stop * ae)
    vmax = max_climb if (err > 0.0 or max_descent is None) else max_descent
    vs = min(vmax, ae / max(e.alt_hold_tau_s, 1e-6), v_stop)
    return math.copysign(vs, err)


def _limit_lift_vector(ac: Aircraft, e: AircraftEnergyConfig, nv: float, nh: float,
                       vertical_first: bool, n_cap: float, dt: float,
                       roll_hint: float) -> Tuple[float, float]:
    """Spec 8e: move the lift vector (n, phi) toward the desired components
    (nv, nh) under the roll-rate and g-onset limits; returns (n, phi).

    1. n_est = n moved toward the desired |(nv, nh)| at <= g_onset_g_s.
    2. Bank target: phi_des = atan2(nh, nv). In vertical-first mode (normal
       turns / climbs) the target is instead the bank at which n_est carries
       nv (cos phi = nv / n_est), so a lagging g onset shallows the bank
       rather than losing altitude.
    3. The bank rolls toward the target (shortest way; an exact 180 deg roll
       goes toward roll_hint's side) at <= roll_rate_deg_s.
    4. Only the projection of the desired lift on the current lift direction
       is pulled, n_goal = max(0, n_des cos(phi_des - phi)) -- a jet rolling
       inverted for a nose-down pull is ~unloaded until past 90 deg -- and n
       moves toward n_goal at <= g_onset_g_s (up and down), 0 <= n <= n_cap
       (the lift limit applies at once). In vertical-first mode n_goal is also
       capped at nv / cos(phi) while the bank is still short of the target,
       so rolling into a level turn does not pop the jet upward."""
    phi = float(ac.bank_rad)
    n0 = float(ac.load_factor)
    n_des = math.hypot(nv, nh)
    dn_max = e.g_onset_g_s * dt if (e.g_onset_g_s is not None and dt > 0) else math.inf
    if n_des > 1e-9:
        phi_des = math.atan2(nh, nv)
        phi_t = phi_des
        if vertical_first and dn_max < math.inf:
            n_est = min(max(n0 + min(max(n_des - n0, -dn_max), dn_max), 0.0), n_cap)
            if n_est > 1e-9:
                c = min(max(nv / n_est, -1.0), 1.0)
                phi_t = math.copysign(math.acos(c), nh) if nh != 0.0 else (0.0 if c >= 0 else math.pi)
        d = _wrap_pi(phi_t - phi)
        if abs(d) > math.pi - 1e-6:
            d = math.pi if roll_hint >= 0.0 else -math.pi
        if e.roll_rate_deg_s is not None and dt > 0:
            p_max = math.radians(e.roll_rate_deg_s) * dt
            d = min(max(d, -p_max), p_max)
        phi_new = _wrap_pi(phi + d)
        # Unload before rolling toward wings level: while g cannot come down
        # fast enough, keep |phi| >= acos(nv / n_min) on the current side so
        # the excess lift stays horizontal instead of popping the jet up.
        n_min = max(0.0, n0 - dn_max)
        if (vertical_first and nv >= 0.0 and n_min > nv + 1e-9
                and abs(phi) > 1e-9 and math.cos(phi_new) * n_min > nv):
            phi_lim = math.acos(nv / n_min)
            if abs(phi) >= phi_lim:
                phi_new = math.copysign(phi_lim, phi)
            else:
                phi_new = phi          # already short of it: do not roll further in
        phi = phi_new
        n_goal = max(0.0, n_des * math.cos(phi_des - phi))
        cphi = math.cos(phi)
        if vertical_first and cphi > 1e-6 and nv >= 0.0:
            n_goal = min(n_goal, nv / cphi)      # still rolling in: hold the vertical
    else:
        n_goal = 0.0
    n = n0 + min(max(n_goal - n0, -dn_max), dn_max)
    n = min(max(n, 0.0), n_cap)
    return n, phi


def integrate_aircraft(ac: Aircraft, dt: float) -> None:
    """Advance point-mass kinematics with the Spec 8 / 8b / 8d / 8e model.

    Commands stay heading / speed / altitude (or an optional direct climb
    rate). Spec 8d: the flight-path angle gamma is a state. Spec 8e: the lift
    vector is a state too, magnitude n (0 <= n <= n_cap = min(n_max,
    q S CLmax / W), positive g only) at bank phi around the velocity vector
    (any angle, inverted included). Its components set the path:
    V dgamma/dt = g (n cos phi - cos gamma), V cos gamma dpsi/dt = g n sin phi.

    Allocation (desired lift vector from the heading / altitude demands):
      n_v_des = cos gamma + V (gamma_des - gamma) / (g dt),
      n_h_des = V cos gamma psidot_cmd / g.
      * n_v_des >= 0 (or a 0 g push is quick enough): Spec 8d vertical first,
        n_v = clip(n_v_des, n_pushover_min, n_cap), n_h <= sqrt(nb^2 - n_v^2).
      * otherwise (rolled pull): the desired vector (n_v_des, n_h_des) is
        scaled proportionally to magnitude <= n_cap, so a big descent + turn
        demand becomes a rolled descending pull (~135 deg bank when both
        saturate) and a big pure descent an inverted pull (phi = 180 deg).
    phi_des = atan2(n_h, n_v); roll-rate and g-onset limits then move the
    actual (n, phi) toward it (_limit_lift_vector). Heading / gamma never
    overshoot their targets (the pilot eases off). Induced drag uses the
    achieved total n. A jet whose lift cannot carry cos gamma drops its nose.
    """
    if not ac.state.alive:
        return

    p = ac.params
    e = p.energy
    st = ac.state
    V = max(float(st.speed_mps), 1.0)
    rho, a_sound = atmosphere(st.alt)
    # Hard altitude / Mach ceilings from energy config
    max_alt = min(p.max_alt_m, e.max_alt_m)
    v_ceil = e.max_mach * a_sound
    floor = max(p.min_alt_m, alt_floor_m(st.x, st.y))
    max_climb = p.max_climb_rate_mps
    s_max = e.max_sin_gamma
    # Spec 8e: descents limited by the dive angle instead of the climb rate
    s_dive = s_max if e.max_dive_deg is None else min(s_max, math.sin(math.radians(e.max_dive_deg)))
    max_descent = None if e.max_dive_deg is None else V * s_dive

    # --- drag / thrust terms (Spec 8b: transonic Cd0 rise, thrust vs Mach) ---
    mach = V / a_sound
    q = 0.5 * rho * V * V
    qS = max(q * e.S_m2, 1e-6)
    W = e.mass_kg * G0
    d0 = qS * e.Cd0 * e.cd0_factor(mach)
    kind = e.k_induced * W * W / qS          # induced drag per n^2
    t_avail = e.thrust_n(rho, mach)
    # Implicit throttle: accelerate when cmd speed > current, else idle
    target_spd = _clip(ac.cmd_speed_mps, e.min_speed_mps, max(v_ceil, e.min_speed_mps))
    thrust = t_avail if target_spd > st.speed_mps + 0.5 else 0.0

    # --- available load factor: min(n_max, lift limit) (Spec 8b) -----------
    n_lift = qS * e.CLmax / W
    n_cap = min(e.n_max, n_lift)             # may be < 1 (Spec 8d: nose drops)
    gam = float(ac.gamma_rad)
    cos_g, sin_g = math.cos(gam), math.sin(gam)

    # Reaction delay for the altitude hold's stopping law (Spec 8e): time to
    # roll back upright and to change g toward the arresting load factor.
    def _t_delay(climbing: bool) -> float:
        t = 0.0
        if e.roll_rate_deg_s is not None:
            t += abs(float(ac.bank_rad)) / math.radians(max(e.roll_rate_deg_s, 1e-6))
        if e.g_onset_g_s is not None:
            n_goal = e.n_pushover_min if climbing else n_cap
            t += 0.5 * abs(n_goal - float(ac.load_factor)) / max(e.g_onset_g_s, 1e-6)
        return t

    def _hold(err: float) -> float:
        return _alt_hold_rate(err, e, max_climb, cos_g, n_cap, max_descent,
                              _t_delay(err > 0.0))

    # --- desired climb rate / flight-path angle -----------------------------
    target_alt = _clip(ac.cmd_alt_m, floor, max_alt)
    if ac.cmd_climb_rate_mps is not None:
        lo = -max_climb if max_descent is None else -max_descent
        vs_des = _clip(ac.cmd_climb_rate_mps, lo, max_climb)
        # never command through the ceiling / floor
        vs_des = min(vs_des, max(0.0, _hold(max_alt - st.alt)))
        vs_des = max(vs_des, min(0.0, _hold(floor - st.alt)))
    else:
        vs_des = _hold(target_alt - st.alt)
    gam_des = math.asin(_clip(vs_des / V, -s_dive, s_max))

    # Soft speed floor (Spec 8 A1.6): at min_speed_mps a jet can only pull the
    # load factor its full thrust sustains; if even 1 g cannot be held there
    # it glides (gamma capped at the glide angle) to hold the floor.
    n_budget = n_cap
    if dt > 0:
        need = (e.min_speed_mps - st.speed_mps) / dt + G0 * sin_g   # required (T-D)/m
        n2_floor = (t_avail - d0 - e.mass_kg * need) / max(kind, 1e-9)
        if n2_floor < n_cap * n_cap:
            if n2_floor >= 1.0:
                n_budget = math.sqrt(n2_floor)
            else:
                n_budget = min(n_cap, 1.0)
                sin_glide = ((t_avail - d0 - kind) / e.mass_kg
                             - (e.min_speed_mps - st.speed_mps) / dt) / G0
                gam_des = min(gam_des, math.asin(_clip(sin_glide, -s_max, s_max)))
                thrust = t_avail
    nb = min(n_budget, n_cap)

    # --- heading demand -------------------------------------------------------
    dh = _angle_diff(ac.cmd_heading_rad, st.heading_rad)
    # Turn-direction hysteresis: with the command within TURN_REVERSAL_BAND of
    # dead astern, keep turning the way the jet already is instead of letting a
    # rounding-level change in the command reverse the turn.
    if (ac.turn_dir != 0.0 and abs(dh) > math.pi - TURN_REVERSAL_BAND_RAD
            and dh * ac.turn_dir < 0.0):
        dh += 2.0 * math.pi * ac.turn_dir
    elif ac.turn_dir == 0.0 and abs(dh) > math.pi - 1e-6:
        dh = abs(dh)          # exact reversal from straight flight: tie-break right
    cos_h = max(cos_g, 0.05)
    omega_cmd = dh / dt if dt > 0 else 0.0
    n_h_des = V * omega_cmd * cos_h / G0     # signed (+ = right)

    # --- desired lift vector (allocation) -------------------------------------
    n_v_des = cos_g + (V * (gam_des - gam) / (G0 * dt) if dt > 0 else 0.0)
    rolled = False
    if e.rolled_pull and n_v_des < e.n_pushover_min:
        if abs(float(ac.bank_rad)) > 0.5 * math.pi:
            rolled = True                    # already inverted-ish: keep pulling
        else:
            push_rate = G0 * max(cos_g - e.n_pushover_min, 0.1) / V
            rolled = (gam - gam_des) / push_rate > e.rolled_pull_min_push_s
    if rolled:
        nv = max(n_v_des, -n_cap)
        nh = min(max(n_h_des, -nb), nb)
        mag = math.hypot(nv, nh)
        if mag > n_cap:
            nv *= n_cap / mag
            nh *= n_cap / mag
            thrust = t_avail                 # saturated pull: full thrust
    else:
        # Spec 8d: vertical first, horizontal gets the remaining budget
        nv = min(max(n_v_des, max(e.n_pushover_min, -n_cap)), n_cap)
        if n_v_des > n_cap + 1e-9:
            thrust = t_avail                 # lift-limited in pitch: full thrust
        n_h_cap = math.sqrt(max(nb * nb - nv * nv, 0.0))
        nh = min(max(n_h_des, -n_h_cap), n_h_cap)
    # --- rate-limited lift vector (roll rate, g onset) ------------------------
    n_act, phi = _limit_lift_vector(ac, e, nv, nh, not rolled, n_cap, dt,
                                    roll_hint=dh if dh != 0.0 else 1.0)
    ac.load_factor = n_act
    ac.bank_rad = phi
    n_v = n_act * math.cos(phi)
    n_hs = n_act * math.sin(phi)

    # --- heading (never past the command) -------------------------------------
    omega = G0 * n_hs / (V * cos_h)
    if dt > 0 and omega * omega_cmd > 0.0 and abs(omega) > abs(omega_cmd):
        omega = omega_cmd
    ac.turn_dir = ((1.0 if omega > 0.0 else (-1.0 if omega < 0.0 else 0.0))
                   if abs(omega_cmd) > abs(omega) + 1e-12 else 0.0)
    st.heading_rad = _wrap_pi(st.heading_rad + omega * dt)
    if abs(omega_cmd) > abs(omega) + 1e-12 and abs(n_hs) > 1e-9 and nb < e.n_max:
        thrust = t_avail                     # limited turn: full thrust

    # --- flight path ------------------------------------------------------------
    # Target capture: if the achieved vertical component has the same sign as,
    # but more magnitude than, the one that lands exactly on gamma_des this
    # step (n_v_des), only n_v_des is flown (the pilot eases the pull; never
    # more lift than the lift vector has). Drag keeps the achieved n.
    if n_v * n_v_des > 0.0 and abs(n_v) > abs(n_v_des):
        n_v = n_v_des
    gam_new = gam + (G0 * (n_v - cos_g) / V * dt if dt > 0 else 0.0)
    gam_new = min(max(gam_new, -math.asin(s_max)), math.asin(s_max))
    drag = d0 + kind * n_act * n_act
    accel = (thrust - drag) / e.mass_kg - G0 * math.sin(gam_new)
    st.speed_mps = float(st.speed_mps + accel * dt)
    # Soft floor (see above), hard Mach ceiling
    st.speed_mps = _clip(st.speed_mps, e.min_speed_mps, v_ceil)
    # Max climb rate cap (Spec 8); descents: max dive angle (8e) or the 8d
    # symmetric rate cap when max_dive_deg is None.
    v_new = max(st.speed_mps, 1e-6)
    s_up = min(s_max, max_climb / v_new)
    s_dn = min(s_max, max_climb / v_new) if e.max_dive_deg is None else s_dive
    if math.sin(gam_new) > s_up:
        gam_new = math.asin(s_up)
    elif math.sin(gam_new) < -s_dn:
        gam_new = -math.asin(s_dn)

    # --- altitude + position ------------------------------------------------
    new_alt = st.alt + st.speed_mps * math.sin(gam_new) * dt
    if new_alt >= max_alt and gam_new > 0.0:
        new_alt, gam_new = max_alt, 0.0      # hard ceiling: level off
    elif new_alt <= floor and gam_new < 0.0:
        new_alt, gam_new = floor, 0.0        # hard floor: level off
    st.alt = _clip(new_alt, floor, max_alt)
    ac.gamma_rad = gam_new
    # Position: heading 0 = North (+Y), clockwise toward East (+X)
    v_h = st.speed_mps * math.cos(gam_new)
    st.x += v_h * math.sin(st.heading_rad) * dt
    st.y += v_h * math.cos(st.heading_rad) * dt


def _clip(x: float, lo: float, hi: float) -> float:
    """Scalar np.clip (same result, incl. hi when lo > hi) without numpy overhead."""
    return float(min(max(x, lo), hi))


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _angle_diff(target: float, current: float) -> float:
    return _wrap_pi(target - current)


def distance_3d(a: AircraftState, b: AircraftState) -> float:
    return float(np.linalg.norm(a.position() - b.position()))


def horizontal_distance(a: AircraftState, b: AircraftState) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def bearing_to(from_st: AircraftState, to_st: AircraftState) -> float:
    """Bearing from A to B (rad, 0=N clockwise)."""
    dx = to_st.x - from_st.x
    dy = to_st.y - from_st.y
    return float(np.arctan2(dx, dy))
