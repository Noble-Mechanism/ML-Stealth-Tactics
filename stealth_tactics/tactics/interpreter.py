"""Interpret high-level tactics genome into per-tick aircraft commands."""

from __future__ import annotations

from typing import List, Optional, Dict

import numpy as np

from stealth_tactics.sim.aircraft import (
    Aircraft,
    Coalition,
    bearing_to,
    distance_3d,
    horizontal_distance,
    _angle_diff,
    _wrap_pi,
)
from stealth_tactics.sim.world import World
from .genome import (
    TacticsGenome,
    MergeGeometry,
    WeaponDoctrine,
    PostMergeRole,
)


class TacticsController:
    """Blue package controller driven by a TacticsGenome."""

    def __init__(self, genome: TacticsGenome, blue_ids: List[str]) -> None:
        self.genome = genome
        self.blue_ids = list(blue_ids)
        self.committed = False
        self.merge_side: Dict[str, float] = {}  # +1 right, -1 left
        self._assign_sides()

    def _assign_sides(self) -> None:
        n = len(self.blue_ids)
        n_left = max(1, int(round(self.genome.split_left_fraction * n)))
        for i, uid in enumerate(self.blue_ids):
            self.merge_side[uid] = -1.0 if i < n_left else 1.0

    def __call__(self, world: World) -> None:
        blues = [world.get(i) for i in self.blue_ids]
        blues = [b for b in blues if b and b.state.alive]
        reds = world.alive(Coalition.RED)
        if not blues:
            return

        lead = blues[0]
        g = self.genome

        # Nearest threat range
        min_rng = float("inf")
        nearest_red: Optional[Aircraft] = None
        for r in reds:
            d = distance_3d(lead.state, r.state)
            if d < min_rng:
                min_rng = d
                nearest_red = r

        blue_losses = sum(
            1 for uid in self.blue_ids
            if (ac := world.get(uid)) is not None and not ac.state.alive
        )
        abort = blue_losses >= g.abort_loss_threshold

        if nearest_red and min_rng <= g.commit_range_m and not abort:
            self.committed = True

        if abort or not reds:
            self._egress(blues)
            return

        if not self.committed or nearest_red is None:
            self._hold_formation(blues, lead)
            return

        # Committed — apply merge geometry
        self._apply_merge(blues, reds, nearest_red, world)

    def _hold_formation(self, blues: List[Aircraft], lead: Aircraft) -> None:
        g = self.genome
        # Lead flies current heading / cruise
        lead.cmd_heading_rad = lead.state.heading_rad
        lead.cmd_speed_mps = lead.params.cruise_speed_mps
        lead.cmd_alt_m = lead.state.alt

        # Wingmen hold offsets in lead body frame
        for i, ac in enumerate(blues[1:], start=0):
            if i >= len(g.formation):
                break
            off = g.formation[i]
            h = lead.state.heading_rad
            # right = +X in body (east when heading north): rotate
            # body forward = (sin h, cos h), right = (cos h, -sin h)
            fx, fy = np.sin(h), np.cos(h)
            rx, ry = np.cos(h), -np.sin(h)
            tx = lead.state.x + rx * off.right_m + fx * off.forward_m
            ty = lead.state.y + ry * off.right_m + fy * off.forward_m
            brg = float(np.arctan2(tx - ac.state.x, ty - ac.state.y))
            ac.cmd_heading_rad = brg
            dist = float(np.hypot(tx - ac.state.x, ty - ac.state.y))
            ac.cmd_speed_mps = lead.params.cruise_speed_mps * (1.1 if dist > 500 else 1.0)
            ac.cmd_alt_m = lead.state.alt + off.up_m

    def _egress(self, blues: List[Aircraft]) -> None:
        for ac in blues:
            # Turn south / away — reverse heading approx
            ac.cmd_heading_rad = _wrap_pi(ac.state.heading_rad + np.pi)
            ac.cmd_speed_mps = ac.params.max_speed_mps
            ac.cmd_alt_m = max(ac.state.alt, 6000)

    def _apply_merge(
        self,
        blues: List[Aircraft],
        reds: List[Aircraft],
        nearest_red: Aircraft,
        world: World,
    ) -> None:
        g = self.genome
        geom = g.merge_geometry

        for idx, ac in enumerate(blues):
            # Map to original ship index for roles
            ship_i = self.blue_ids.index(ac.id) if ac.id in self.blue_ids else idx
            role = g.post_merge_roles[ship_i] if ship_i < len(g.post_merge_roles) else PostMergeRole.ENGAGE
            side = self.merge_side.get(ac.id, 1.0)

            # Pick target: nearest red that we track, else nearest
            tracks = world.tracks.get(ac.id, set())
            tracked_reds = [r for r in reds if r.id in tracks]
            pool = tracked_reds if tracked_reds else reds
            tgt = min(pool, key=lambda r: distance_3d(ac.state, r.state))

            brg_to_tgt = bearing_to(ac.state, tgt.state)
            dist = distance_3d(ac.state, tgt.state)

            # Geometry-specific aim point offset
            offset_brg = 0.0
            if geom == MergeGeometry.BRACKET:
                offset_brg = side * np.deg2rad(35)
            elif geom == MergeGeometry.HOOK:
                offset_brg = side * np.deg2rad(50)
            elif geom == MergeGeometry.DRAG:
                is_drag = g.drag_ship_mask[ship_i] if ship_i < 4 else False
                if is_drag:
                    # Drag: turn away slightly, suck bandit
                    offset_brg = side * np.deg2rad(-20)
                    role = PostMergeRole.SUPPORT
                else:
                    offset_brg = side * np.deg2rad(45)
            elif geom == MergeGeometry.SANDWICH:
                # High/low: even ships high
                alt_sign = 1.0 if ship_i % 2 == 0 else -1.0
                ac.cmd_alt_m = tgt.state.alt + alt_sign * 1500 + g.commit_alt_bias_m
                offset_brg = side * np.deg2rad(25)
            else:  # HEADON
                offset_brg = 0.0

            if role == PostMergeRole.EGRESS:
                ac.cmd_heading_rad = _wrap_pi(brg_to_tgt + np.pi)
                ac.cmd_speed_mps = ac.params.max_speed_mps
                ac.cmd_alt_m = ac.state.alt + g.commit_alt_bias_m
                continue
            if role == PostMergeRole.CAP:
                # Orbit-ish: circle
                ac.cmd_heading_rad = _wrap_pi(ac.state.heading_rad + np.deg2rad(8))
                ac.cmd_speed_mps = ac.params.cruise_speed_mps
                ac.cmd_alt_m = ac.state.alt
                continue

            aim = _wrap_pi(brg_to_tgt + offset_brg)
            ac.cmd_heading_rad = aim
            ac.cmd_speed_mps = ac.params.cruise_speed_mps * g.commit_speed_factor
            if geom != MergeGeometry.SANDWICH:
                ac.cmd_alt_m = tgt.state.alt + g.commit_alt_bias_m

            # Weapons
            self._maybe_fire(ac, tgt, dist, world)

    def _maybe_fire(
        self,
        ac: Aircraft,
        tgt: Aircraft,
        dist: float,
        world: World,
    ) -> None:
        g = self.genome
        if ac.ammo <= 0:
            return
        if ac.id not in world.tracks or tgt.id not in world.tracks.get(ac.id, set()):
            return

        max_r = ac.params.missile_range_m
        shoot_r = max_r * g.shoot_range_frac

        if g.weapon_doctrine == WeaponDoctrine.CONSERVATIVE:
            shoot_r *= 0.7
            # Prefer nose aspect
            brg = bearing_to(ac.state, tgt.state)
            if abs(_angle_diff(brg, ac.state.heading_rad)) > np.deg2rad(30):
                return
        elif g.weapon_doctrine == WeaponDoctrine.AGGRESSIVE:
            shoot_r = max_r * min(0.98, g.shoot_range_frac + 0.15)

        if dist <= shoot_r:
            ac.cmd_fire = True
            ac.fire_target = tgt.id


class RedCAPController:
    """
    Fixed Red bandit presentation: CAP orbit or intercept toward Blue CAP.
    Not evolved — provides a stable evaluation environment.
    """

    def __init__(self, red_ids: List[str], mode: str = "intercept") -> None:
        self.red_ids = list(red_ids)
        self.mode = mode
        self._t = 0.0

    def __call__(self, world: World) -> None:
        self._t = world.time_s
        reds = [world.get(i) for i in self.red_ids]
        reds = [r for r in reds if r and r.state.alive]
        blues = world.alive(Coalition.BLUE)
        if not reds:
            return

        for ac in reds:
            if blues:
                tgt = min(blues, key=lambda b: distance_3d(ac.state, b.state))
                tracks = world.tracks.get(ac.id, set())
                detected = tgt.id in tracks

                if self.mode == "cap" and not detected:
                    # Lazy CAP: gentle weave
                    ac.cmd_heading_rad = _wrap_pi(ac.state.heading_rad + np.deg2rad(3))
                    ac.cmd_speed_mps = ac.params.cruise_speed_mps
                    ac.cmd_alt_m = ac.state.alt
                else:
                    # Intercept / prosecute
                    brg = bearing_to(ac.state, tgt.state)
                    ac.cmd_heading_rad = brg
                    ac.cmd_speed_mps = ac.params.cruise_speed_mps * 1.1
                    ac.cmd_alt_m = tgt.state.alt
                    dist = distance_3d(ac.state, tgt.state)
                    if detected and dist < ac.params.missile_range_m * 0.8 and ac.ammo > 0:
                        # Red shoots if lock
                        if world.sensors.can_lock(ac, tgt):
                            ac.cmd_fire = True
                            ac.fire_target = tgt.id
            else:
                ac.cmd_heading_rad = ac.state.heading_rad
                ac.cmd_speed_mps = ac.params.cruise_speed_mps
