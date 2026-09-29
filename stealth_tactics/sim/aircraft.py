"""Point-mass aircraft state and kinematics (Spec 8 energy model)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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


def integrate_aircraft(ac: Aircraft, dt: float) -> None:
    """Advance point-mass kinematics with the Spec 8 / 8b energy model.

    Commands stay heading / speed / altitude. Turn rate is limited by
    min(n_max, q S CLmax / W), drag (transonic Cd0 rise + induced) and thrust
    (density lapse, ram gain) set along-track accel, and climb costs energy via
    sin(γ). A jet that cannot hold 1 g sinks.
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

    # --- climb rate toward cmd_alt (still rate-limited) ---------------------
    floor = max(p.min_alt_m, alt_floor_m(st.x, st.y))
    target_alt = float(np.clip(ac.cmd_alt_m, floor, max_alt))
    max_climb = p.max_climb_rate_mps
    climb_rate = float(np.clip(target_alt - st.alt, -max_climb * dt, max_climb * dt) / dt) \
        if dt > 0 else 0.0
    # Flight-path angle from the climb/descent; clip so |sin γ| <= 1
    sin_g = float(np.clip(climb_rate / V, -0.95, 0.95))

    # --- drag / thrust terms (Spec 8b: transonic Cd0 rise, thrust vs Mach) ---
    mach = V / a_sound
    q = 0.5 * rho * V * V
    qS = max(q * e.S_m2, 1e-6)
    W = e.mass_kg * G0
    d0 = qS * e.Cd0 * e.cd0_factor(mach)
    kind = e.k_induced * W * W / qS          # induced drag per n^2
    t_avail = e.thrust_n(rho, mach)
    # Implicit throttle: accelerate when cmd speed > current, else idle
    target_spd = float(np.clip(ac.cmd_speed_mps, e.min_speed_mps, max(v_ceil, e.min_speed_mps)))
    thrust = t_avail if target_spd > st.speed_mps + 0.5 else 0.0

    # --- available load factor: min(n_max, lift limit) (Spec 8b) -----------
    n_lift = qS * e.CLmax / W
    n_cap = min(max(e.n_max, 1.0 + 1e-9), n_lift)
    max_sink = -min(0.95, max_climb / V)
    if n_lift < 1.0:
        # Cannot hold 1 g: wings level and sink (lift = W cos(gamma)), full
        # thrust. The jet must descend to regain energy.
        sin_g = min(sin_g, max(-math.sqrt(1.0 - n_lift * n_lift), max_sink))
        climb_rate = sin_g * V
        thrust = t_avail
    elif dt > 0:
        # Soft speed floor (Spec 8 A1.6): at min_speed_mps a jet can only pull
        # the load factor its full thrust sustains. If even 1 g level cannot be
        # held there, it unloads to 1 g and descends (glides) to hold the floor.
        need = (e.min_speed_mps - st.speed_mps) / dt + G0 * sin_g   # required (T-D)/m
        n2_floor = (t_avail - d0 - e.mass_kg * need) / max(kind, 1e-9)
        if n2_floor < n_cap * n_cap:
            if n2_floor >= 1.0:
                n_cap = float(np.sqrt(n2_floor))
            else:
                n_cap = 1.0
                sin_glide = ((t_avail - d0 - kind) / e.mass_kg
                             - (e.min_speed_mps - st.speed_mps) / dt) / G0
                sin_g = min(sin_g, max(sin_glide, max_sink))
                climb_rate = sin_g * V
                thrust = t_avail
    dh = _angle_diff(ac.cmd_heading_rad, st.heading_rad)
    # Turn-direction hysteresis: with the command within TURN_REVERSAL_BAND of
    # dead astern, keep turning the way the jet already is instead of letting a
    # rounding-level change in the command reverse the turn.
    if (ac.turn_dir != 0.0 and abs(dh) > math.pi - TURN_REVERSAL_BAND_RAD
            and dh * ac.turn_dir < 0.0):
        dh += 2.0 * math.pi * ac.turn_dir
    elif ac.turn_dir == 0.0 and abs(dh) > math.pi - 1e-6:
        dh = abs(dh)          # exact reversal from straight flight: tie-break right
    omega_max = G0 * np.sqrt(max(n_cap * n_cap - 1.0, 0.0)) / V
    omega_cmd = dh / dt if dt > 0 else 0.0
    omega = float(np.clip(omega_cmd, -omega_max, omega_max))
    ac.turn_dir = float(np.sign(omega)) if abs(omega_cmd) > omega_max else 0.0
    st.heading_rad = _wrap_pi(st.heading_rad + omega * dt)
    n_horiz = V * abs(omega) / G0
    n = float(np.sqrt(1.0 + n_horiz * n_horiz)) if n_lift >= 1.0 else n_lift
    if n > 1.0 + 1e-9 and n_cap < max(e.n_max, 1.0 + 1e-9):
        thrust = t_avail                     # limited turn: full thrust

    # --- along-track accel ---------------------------------------------------
    drag = d0 + kind * n * n
    accel = (thrust - drag) / e.mass_kg - G0 * sin_g
    st.speed_mps = float(st.speed_mps + accel * dt)
    # Soft floor (see above), hard Mach ceiling
    st.speed_mps = float(np.clip(st.speed_mps, e.min_speed_mps, v_ceil))

    # --- altitude + position ------------------------------------------------
    st.alt = float(np.clip(st.alt + climb_rate * dt, floor, max_alt))
    # Position: heading 0 = North (+Y), clockwise toward East (+X)
    # Horizontal speed = V * cos(γ); use remaining after climb component
    cos_g = float(np.sqrt(max(0.0, 1.0 - sin_g * sin_g)))
    v_h = st.speed_mps * cos_g
    st.x += v_h * np.sin(st.heading_rad) * dt
    st.y += v_h * np.cos(st.heading_rad) * dt


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
