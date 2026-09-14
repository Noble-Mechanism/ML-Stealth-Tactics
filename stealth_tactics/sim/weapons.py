"""Missile / shoot envelopes with simple Pk hit model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .aircraft import Aircraft, AircraftState, distance_3d, bearing_to, _angle_diff


@dataclass
class Missile:
    id: str
    shooter_id: str
    target_id: str
    coalition: str
    pos: AircraftState
    speed_mps: float = 900.0
    remaining_time_s: float = 40.0
    pk: float = 0.55
    alive: bool = True
    hit: bool = False


class WeaponModel:
    """
    Fire-and-forget simplified BVR missile.

    Assumptions:
    - Launch if target in range, roughly nose aspect, ammo > 0, lock available
    - Missile flies toward target position (no fancy PN); hits if closes within
      proximity fuse distance before fuel expires
    - On proximity: Bernoulli trial with shooter missile_pk (modified by aspect)
    """

    PROXIMITY_M = 50.0
    MAX_OFF_BORESIGHT_DEG = 60.0

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._missile_seq = 0

    def can_shoot(self, shooter: Aircraft, target: Aircraft) -> bool:
        if not shooter.state.alive or not target.state.alive:
            return False
        if shooter.ammo <= 0:
            return False
        d = distance_3d(shooter.state, target.state)
        if d > shooter.params.missile_range_m or d < 500.0:
            return False
        brg = bearing_to(shooter.state, target.state)
        off = abs(_angle_diff(brg, shooter.state.heading_rad))
        return off <= np.deg2rad(self.MAX_OFF_BORESIGHT_DEG)

    def launch(self, shooter: Aircraft, target: Aircraft) -> Optional[Missile]:
        if not self.can_shoot(shooter, target):
            return None
        shooter.ammo -= 1
        self._missile_seq += 1
        mid = f"M{self._missile_seq:04d}"
        # Time-of-flight approx based on closing
        tof = min(40.0, distance_3d(shooter.state, target.state) / 700.0 + 5.0)
        return Missile(
            id=mid,
            shooter_id=shooter.id,
            target_id=target.id,
            coalition=shooter.coalition.value,
            pos=shooter.state.copy(),
            speed_mps=900.0,
            remaining_time_s=tof,
            pk=shooter.params.missile_pk,
        )

    def step(
        self,
        missiles: List[Missile],
        aircraft_by_id: dict,
        dt: float,
    ) -> List[str]:
        """Advance missiles; return list of destroyed aircraft ids."""
        killed: List[str] = []
        for m in missiles:
            if not m.alive:
                continue
            tgt = aircraft_by_id.get(m.target_id)
            if tgt is None or not tgt.state.alive:
                m.alive = False
                continue

            # Steer toward target
            dx = tgt.state.x - m.pos.x
            dy = tgt.state.y - m.pos.y
            dz = tgt.state.alt - m.pos.alt
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz)) + 1e-6
            step = m.speed_mps * dt
            if step >= dist:
                m.pos.x = tgt.state.x
                m.pos.y = tgt.state.y
                m.pos.alt = tgt.state.alt
                dist = 0.0
            else:
                m.pos.x += dx / dist * step
                m.pos.y += dy / dist * step
                m.pos.alt += dz / dist * step
                dist -= step

            m.remaining_time_s -= dt
            if dist <= self.PROXIMITY_M:
                # Hit roll
                if self.rng.random() < m.pk:
                    m.hit = True
                    tgt.state.alive = False
                    killed.append(tgt.id)
                m.alive = False
            elif m.remaining_time_s <= 0:
                m.alive = False
        return killed
