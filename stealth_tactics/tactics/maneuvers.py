"""Spec 3 maneuver primitives (docs/specs/03-missile-defense.md, section 2).

Pure functions: each takes an aircraft and a bearing (rad, 0 = N clockwise) and
returns a ``ManeuverCmd(heading, speed, alt)``. No decisions, no state. The Red
defense state machine, the Blue scripted test reaction and (spec 4) pre-planned
presentation maneuvers all call these.

``side`` for crank / beam: +1 = threat ends up on the LEFT (turn right of the
threat bearing), -1 = threat on the RIGHT; None = the short way from the
current heading. Callers that must not dither pick the side once and pass it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from stealth_tactics.sim.aircraft import Aircraft, _angle_diff, _wrap_pi, alt_floor_m

HOT_SPEED_FACTOR = 1.1     # x cruise (today's Red intercept speed)
CRANK_SPEED_FACTOR = 1.1   # D2: crank at 1.1 x cruise, altitude held
DEFAULT_CRANK_DEG = 50.0   # D2


@dataclass(frozen=True)
class ManeuverCmd:
    heading: float
    speed: float
    alt: float

    def apply(self, ac: Aircraft) -> None:
        ac.cmd_heading_rad = self.heading
        ac.cmd_speed_mps = self.speed
        ac.cmd_alt_m = self.alt


def short_side(ac: Aircraft, threat_brg: float) -> int:
    """+1 if the current heading is right of (or on) the threat bearing, else -1."""
    return 1 if _angle_diff(ac.state.heading_rad, threat_brg) >= 0.0 else -1


def hot(ac: Aircraft, brg: float, alt: Optional[float] = None) -> ManeuverCmd:
    """Pure pursuit at 1.1 x cruise (alt: commanded altitude, default held)."""
    return ManeuverCmd(_wrap_pi(brg), ac.params.cruise_speed_mps * HOT_SPEED_FACTOR,
                       ac.state.alt if alt is None else alt)


def crank(ac: Aircraft, threat_brg: float, angle_deg: float = DEFAULT_CRANK_DEG,
          side: Optional[int] = None) -> ManeuverCmd:
    """Threat on the nose +/- angle (D2: 50 deg keeps fire control and support),
    1.1 x cruise, altitude held."""
    s = short_side(ac, threat_brg) if side is None else side
    return ManeuverCmd(_wrap_pi(threat_brg + s * math.radians(angle_deg)),
                       ac.params.cruise_speed_mps * CRANK_SPEED_FACTOR, ac.state.alt)


def beam(ac: Aircraft, threat_brg: float, side: Optional[int] = None) -> ManeuverCmd:
    """Threat 90 deg off the nose, max speed, altitude held (D3)."""
    s = short_side(ac, threat_brg) if side is None else side
    return ManeuverCmd(_wrap_pi(threat_brg + s * math.pi / 2), ac.params.max_speed_mps,
                       ac.state.alt)


def drag_depth_m(alt_m: float, aggressiveness: float, depth_a1_m: float = 1000.0,
                 depth_a0_m: Optional[float] = None, floor_m: float = 100.0) -> float:
    """Descent for a drag (approved change B): linear in a from ``depth_a0_m``
    at a = 0 (None = all the way down to the floor) to ``depth_a1_m`` at a = 1."""
    a = min(1.0, max(0.0, aggressiveness))
    d0 = max(0.0, alt_m - floor_m) if depth_a0_m is None else depth_a0_m
    return (1.0 - a) * d0 + a * depth_a1_m


def drag_target_alt(ac: Aircraft, aggressiveness: float, depth_a1_m: float = 1000.0,
                    depth_a0_m: Optional[float] = None) -> float:
    """max(100 m AGL floor, current altitude - depth(a)). Computed once when the
    drag starts and then held."""
    floor = alt_floor_m(ac.state.x, ac.state.y)
    d = drag_depth_m(ac.state.alt, aggressiveness, depth_a1_m, depth_a0_m, floor)
    return max(floor, ac.state.alt - d)


def drag(ac: Aircraft, threat_brg: float, target_alt: Optional[float] = None) -> ManeuverCmd:
    """Threat on the tail, max speed, descend to ``target_alt`` (default: hold)."""
    alt = ac.state.alt if target_alt is None else max(target_alt,
                                                      alt_floor_m(ac.state.x, ac.state.y))
    return ManeuverCmd(_wrap_pi(threat_brg + math.pi), ac.params.max_speed_mps, alt)


def depart(ac: Aircraft, away_brg: float) -> ManeuverCmd:
    """Leave the fight (D9): fly ``away_brg`` at max speed, altitude held."""
    return ManeuverCmd(_wrap_pi(away_brg), ac.params.max_speed_mps, ac.state.alt)
