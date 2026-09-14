"""Point-mass aircraft state and kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AircraftType(str, Enum):
    BLUE_STEALTH = "BlueStealth"
    RED_FIGHTER = "RedFighter"


class Coalition(str, Enum):
    BLUE = "Blue"
    RED = "Red"


@dataclass
class AircraftTypeParams:
    """Generic flight / signature params (unclassified placeholders)."""

    max_speed_mps: float = 350.0  # ~Mach 1.0 sea level-ish
    min_speed_mps: float = 80.0
    cruise_speed_mps: float = 250.0
    max_turn_rate_deg_s: float = 12.0  # sustained turn approx
    max_climb_rate_mps: float = 80.0
    max_alt_m: float = 15000.0
    min_alt_m: float = 100.0
    rcs_factor: float = 1.0  # relative RCS (stealth << 1)
    radar_range_m: float = 80000.0
    missile_range_m: float = 40000.0
    missile_pk: float = 0.55
    ammo: int = 4


BLUE_STEALTH_PARAMS = AircraftTypeParams(
    max_speed_mps=340.0,
    min_speed_mps=90.0,
    cruise_speed_mps=260.0,
    max_turn_rate_deg_s=11.0,
    max_climb_rate_mps=90.0,
    rcs_factor=0.05,  # stealth advantage
    radar_range_m=90000.0,
    missile_range_m=45000.0,
    missile_pk=0.60,
    ammo=4,
)

RED_FIGHTER_PARAMS = AircraftTypeParams(
    max_speed_mps=360.0,
    min_speed_mps=85.0,
    cruise_speed_mps=255.0,
    max_turn_rate_deg_s=13.0,
    max_climb_rate_mps=85.0,
    rcs_factor=1.0,
    radar_range_m=70000.0,
    missile_range_m=38000.0,
    missile_pk=0.50,
    ammo=4,
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
    name: str
    ac_type: AircraftType
    coalition: Coalition
    state: AircraftState
    params: AircraftTypeParams
    ammo: int = 4
    locked_target: Optional[str] = None
    role: str = "fighter"
    # Commanded intents (set by tactics controller each tick)
    cmd_heading_rad: float = 0.0
    cmd_speed_mps: float = 250.0
    cmd_alt_m: float = 8000.0
    cmd_fire: bool = False
    fire_target: Optional[str] = None

    def __post_init__(self) -> None:
        self.ammo = self.params.ammo
        self.cmd_heading_rad = self.state.heading_rad
        self.cmd_speed_mps = self.state.speed_mps
        self.cmd_alt_m = self.state.alt

    @staticmethod
    def make_blue(uid: str, name: str, state: AircraftState) -> Aircraft:
        return Aircraft(
            id=uid,
            name=name,
            ac_type=AircraftType.BLUE_STEALTH,
            coalition=Coalition.BLUE,
            state=state,
            params=BLUE_STEALTH_PARAMS,
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
    """Advance point-mass kinematics with turn/climb/speed limits."""
    if not ac.state.alive:
        return

    p = ac.params
    st = ac.state

    # Heading: turn toward commanded heading at max turn rate
    max_turn = np.deg2rad(p.max_turn_rate_deg_s) * dt
    dh = _angle_diff(ac.cmd_heading_rad, st.heading_rad)
    st.heading_rad = _wrap_pi(st.heading_rad + np.clip(dh, -max_turn, max_turn))

    # Speed
    max_ds = 30.0 * dt  # accel/decel m/s^2 approx
    target_spd = float(np.clip(ac.cmd_speed_mps, p.min_speed_mps, p.max_speed_mps))
    st.speed_mps = float(np.clip(st.speed_mps + np.clip(target_spd - st.speed_mps, -max_ds, max_ds),
                                  p.min_speed_mps, p.max_speed_mps))

    # Altitude
    max_climb = p.max_climb_rate_mps * dt
    target_alt = float(np.clip(ac.cmd_alt_m, p.min_alt_m, p.max_alt_m))
    st.alt = float(np.clip(st.alt + np.clip(target_alt - st.alt, -max_climb, max_climb),
                           p.min_alt_m, p.max_alt_m))

    # Position: heading 0 = North (+Y), clockwise toward East (+X)
    st.x += st.speed_mps * np.sin(st.heading_rad) * dt
    st.y += st.speed_mps * np.cos(st.heading_rad) * dt


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
