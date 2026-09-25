"""Aspect-dependent RCS: Blue harder to detect nose-on than beam/aft.

Updated for Spec 1: the old +/-20 deg hard frontal cone (FRONTAL_CONE_DEG /
STEALTH_OFF_ASPECT_RCS) and deterministic ``detect`` were replaced by a smooth
aspect table and a per-scan Pd curve. Same intent, new API.
"""

from __future__ import annotations

import numpy as np

from stealth_tactics.sim.aircraft import Aircraft, AircraftState, F35_PARAMS, RED_FIGHTER_PARAMS
from stealth_tactics.sim.sensors import effective_rcs, aspect_angle_rad, radar_pd


def _blue_at(heading_rad: float = 0.0) -> Aircraft:
    return Aircraft.make_blue(
        "B1", "Blue1",
        AircraftState(x=0.0, y=0.0, alt=8000.0, heading_rad=heading_rad, speed_mps=250.0),
    )


def _red_at(x: float, y: float, heading_rad: float = np.pi) -> Aircraft:
    return Aircraft.make_red(
        "R1", "Red1",
        AircraftState(x=x, y=y, alt=8000.0, heading_rad=heading_rad, speed_mps=250.0),
    )


def test_frontal_rcs_lower_than_beam() -> None:
    blue = _blue_at(heading_rad=0.0)  # nose North
    red_nose = _red_at(0.0, 40_000.0)
    red_beam = _red_at(40_000.0, 0.0)
    red_aft = _red_at(0.0, -40_000.0)

    rcs_nose = effective_rcs(blue, red_nose)
    rcs_beam = effective_rcs(blue, red_beam)
    rcs_aft = effective_rcs(blue, red_aft)

    assert abs(rcs_nose - F35_PARAMS.rcs_factor) < 1e-9
    assert rcs_nose < rcs_aft < rcs_beam
    assert aspect_angle_rad(blue, red_nose) < 1e-6
    assert abs(aspect_angle_rad(blue, red_beam) - np.pi / 2) < 1e-6


def test_red_rcs_isotropic() -> None:
    red = _red_at(0.0, 0.0, heading_rad=0.0)
    blue_nose = _blue_at()
    blue_nose.state.x, blue_nose.state.y = 0.0, 30_000.0
    blue_beam = _blue_at()
    blue_beam.state.x, blue_beam.state.y = 30_000.0, 0.0
    assert effective_rcs(red, blue_nose) == RED_FIGHTER_PARAMS.rcs_factor
    assert effective_rcs(red, blue_beam) == RED_FIGHTER_PARAMS.rcs_factor


def test_blue_harder_to_detect_nose_on_than_beam() -> None:
    """At identical range, Red's per-scan Pd vs Blue is far lower nose-on than beam."""
    blue = _blue_at(heading_rad=0.0)
    range_m = 45_000.0
    red_nose = _red_at(0.0, range_m, heading_rad=np.pi)       # Blue in Red's nose
    red_beam = _red_at(range_m, 0.0, heading_rad=-np.pi / 2)  # facing Blue
    assert radar_pd(red_nose, blue) < 0.1
    assert radar_pd(red_beam, blue) > 0.9
