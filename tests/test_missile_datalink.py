"""Missile midcourse datalink: support until 15 NM acquisition.

Updated for Spec 1: support now requires a FIRE-CONTROL quality track from the
per-jet TrackStore (``SensorModel.update``) instead of the old
``update_tracks`` detection sets; tests build a mature track before launch.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG

from stealth_tactics.sim.aircraft import Aircraft, AircraftState, Coalition
from stealth_tactics.sim.weapons import (
    WeaponModel,
    Missile,
    ACQUISITION_RANGE_M,
)
from stealth_tactics.sim.sensors import SensorModel


def _mature_tracks(sensors: SensorModel, aircraft, t0: float = 0.0, secs: int = 6):
    """Run sensors for a few 1 Hz scans so Blue holds a fire-control track."""
    t = t0
    for _ in range(secs * 2):
        sensors.update(aircraft, t)
        t += 0.5
    return sensors.stores, t


def _pair(range_m: float = 40_000.0):
    """Blue at origin heading North; Red due North at range_m, heading South."""
    blue = Aircraft.make_blue(
        "B1", "Blue1",
        AircraftState(x=0.0, y=0.0, alt=8000.0, heading_rad=0.0, speed_mps=250.0),
    )
    red = Aircraft.make_red(
        "R1", "Red1",
        AircraftState(x=0.0, y=range_m, alt=8000.0, heading_rad=np.pi, speed_mps=250.0),
    )
    return blue, red


def test_acquisition_range_is_15_nm() -> None:
    assert abs(ACQUISITION_RANGE_M - 15.0 * 1852.0) < 1.0


def test_support_lost_when_shooter_dies_before_acquisition() -> None:
    rng = np.random.default_rng(1)
    weapons = WeaponModel(rng)
    sensors = SensorModel(rng)
    blue, red = _pair(40_000.0)
    by_id = {blue.id: blue, red.id: red}
    tracks, t = _mature_tracks(sensors, [blue, red])
    assert tracks[blue.id].is_fire_control(red.id)

    m = weapons.launch(blue, red)
    assert m is not None
    assert not m.autonomous

    # Shooter dies while missile still outside acquisition
    blue.state.alive = False
    sensors.update([blue, red], t)
    # Place missile still far from target
    m.pos.x, m.pos.y, m.pos.alt = 0.0, 5_000.0, 8000.0
    killed = weapons.step([m], by_id, dt=0.5, tracks=tracks)
    assert killed == []
    # Spec 2b: no longer dies on the first unsupported step -> coasts
    assert m.alive and m.coasting and m.support_lost
    assert m.t_unsupported == 0.5 and m.t_since_update == 0.5
    assert not m.hit


def test_support_lost_when_shooter_turns_cold() -> None:
    """Pure egress (180° turn) drops forward-hemisphere support."""
    rng = np.random.default_rng(2)
    weapons = WeaponModel(rng)
    sensors = SensorModel(rng)
    blue, red = _pair(40_000.0)
    by_id = {blue.id: blue, red.id: red}
    tracks, t = _mature_tracks(sensors, [blue, red])
    assert WeaponModel.has_midcourse_support(blue, red, tracks)

    m = weapons.launch(blue, red)
    assert m is not None
    m.pos.x, m.pos.y, m.pos.alt = 0.0, 5_000.0, 8000.0

    # Turn cold — target now aft of shooter
    blue.state.heading_rad = np.pi  # South, away from Red
    sensors.update([blue, red], t)
    # Blue's track is still coasting (and FC within the 2 s gap tolerance),
    # but off-boresight fails support
    assert red.id in tracks[blue.id]
    assert not WeaponModel.has_midcourse_support(blue, red, tracks)

    weapons.step([m], by_id, dt=0.5, tracks=tracks)
    assert m.alive and m.coasting      # Spec 2b: coasting, not dead
    assert m.support_lost
    assert any(e["type"] == "support_lost" and "coasting" in e["text"]
               for e in weapons.events)


def test_supported_until_acquisition_can_hit() -> None:
    """With continuous support, missile acquires then can kill (Pk=1)."""
    rng = np.random.default_rng(3)
    # Spec 3a: Pk forced to 1 via the shared config (base 1.0, no endgame penalty)
    kc = dataclasses.replace(DEFAULT_SENSOR_CONFIG.missile_kinematics, base_pk=1.0,
                             endgame_pk_floor=1.0)
    weapons = WeaponModel(rng, dataclasses.replace(DEFAULT_SENSOR_CONFIG,
                                                   missile_kinematics=kc))
    sensors = SensorModel(rng)
    # Start closer so TOF is short; force Pk = 1
    blue, red = _pair(20_000.0)
    by_id = {blue.id: blue, red.id: red}

    m = weapons.launch(blue, red)
    assert m is not None
    m.pk = 1.0
    # Ensure we exercise midcourse: start missile outside acquisition if needed
    # At 20 km launch, missile starts outside 15 NM? 20km < 27.78km so already
    # autonomous at launch. Use a longer shot.
    blue2, red2 = _pair(40_000.0)
    blue2.ammo = 4
    by_id = {blue2.id: blue2, red2.id: red2}
    tracks, t = _mature_tracks(sensors, [blue2, red2])
    m = weapons.launch(blue2, red2)
    assert m is not None
    assert not m.autonomous
    m.pk = 1.0              # Spec 3a: final Pk = m.pk x f_E(Mach at impact)
    m.remaining_time_s = 120.0

    hit = False
    for _ in range(200):
        sensors.update([blue2, red2], t)
        t += 0.5
        # Spec 3a: fly the target head-on (a static target at 40 km / 8 km alt is
        # at the edge of the new kinematic range)
        red2.state.y -= red2.state.speed_mps * 0.5
        killed = weapons.step([m], by_id, dt=0.5, tracks=tracks)
        if m.autonomous:
            # After acquisition, kill shooter — must still be allowed to guide
            blue2.state.alive = False
        if killed:
            hit = True
            break
        if not m.alive:
            break

    assert m.outcome in ("hit", "miss"), m.outcome     # reached the fuze
    assert hit, "supported missile should acquire and hit with Pk=1"
    assert m.hit
    assert not red2.state.alive


def test_loses_track_before_acquisition_fails() -> None:
    rng = np.random.default_rng(4)
    weapons = WeaponModel(rng)
    blue, red = _pair(40_000.0)
    by_id = {blue.id: blue, red.id: red}
    m = weapons.launch(blue, red)
    assert m is not None
    m.pos.y = 5_000.0
    # Empty tracks → no detection support
    # Spec 2b: coasting with no support at all -> lost after > 40 s (coast timeout;
    # 3a fix, was 20 s)
    empty = {blue.id: set(), red.id: set()}
    m.remaining_time_s = 200.0
    red.state.speed_mps = 0.0             # keep target far away (no autonomy)
    red.state.y = 400_000.0
    steps = 0
    while m.alive and steps < 100:
        weapons.step([m], by_id, dt=0.5, tracks=empty)
        steps += 1
    assert not m.alive and m.outcome == "lost_coast_timeout"
    assert 40.0 < m.t_since_update <= 40.5 + 1e-9
    assert m.support_lost
