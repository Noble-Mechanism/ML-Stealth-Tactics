"""High-level tactics genome (not raw continuous control)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any

import numpy as np


class MergeGeometry(str, Enum):
    BRACKET = "bracket"       # split left/right around bandits
    HOOK = "hook"             # all turn same side, flank
    DRAG = "drag"             # lead element drags, trail flanks
    SANDWICH = "sandwich"     # high/low pincer
    HEADON = "headon"         # direct commit


class WeaponDoctrine(str, Enum):
    CONSERVATIVE = "conservative"  # shoot only high-Pk geometry
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"      # shoot at max range early


class PostMergeRole(str, Enum):
    ENGAGE = "engage"
    SUPPORT = "support"
    EGRESS = "egress"
    CAP = "cap"


@dataclass
class FormationOffset:
    """Offset from package lead in meters (right, forward, up)."""

    right_m: float = 0.0
    forward_m: float = 0.0
    up_m: float = 0.0


@dataclass
class TacticsGenome:
    """
    Compact high-level tactics chromosome for a 4-ship Blue package.

    Genes encode formation, commit decision, merge geometry, weapons,
    and post-merge role assignment — not stick-and-throttle traces.
    """

    # Formation: offsets for wingmen 1..3 relative to lead (ship 0)
    formation: List[FormationOffset] = field(default_factory=lambda: [
        FormationOffset(right_m=1500, forward_m=-800, up_m=0),
        FormationOffset(right_m=-1500, forward_m=-800, up_m=0),
        FormationOffset(right_m=0, forward_m=-2000, up_m=500),
    ])

    # Commit: start offensive maneuver when nearest bandit within this range
    commit_range_m: float = 55000.0

    # Abort / disengage if Blue losses exceed this count mid-fight
    abort_loss_threshold: int = 2

    merge_geometry: MergeGeometry = MergeGeometry.BRACKET

    # Bracket/hook split: fraction of package going left (rest right)
    split_left_fraction: float = 0.5

    # Drag: which ship indices (0-3) are drag element
    drag_ship_mask: List[bool] = field(default_factory=lambda: [True, False, False, False])

    weapon_doctrine: WeaponDoctrine = WeaponDoctrine.STANDARD

    # Preferred shoot range as fraction of max missile range
    shoot_range_frac: float = 0.75

    # Post-merge roles per ship
    post_merge_roles: List[PostMergeRole] = field(default_factory=lambda: [
        PostMergeRole.ENGAGE,
        PostMergeRole.ENGAGE,
        PostMergeRole.SUPPORT,
        PostMergeRole.SUPPORT,
    ])

    # Cruise altitude bias (m) applied after commit
    commit_alt_bias_m: float = 0.0

    # Speed factor after commit (1.0 = cruise)
    commit_speed_factor: float = 1.05

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["merge_geometry"] = self.merge_geometry.value
        d["weapon_doctrine"] = self.weapon_doctrine.value
        d["post_merge_roles"] = [r.value for r in self.post_merge_roles]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TacticsGenome:
        form = [FormationOffset(**f) for f in d.get("formation", [])]
        while len(form) < 3:
            form.append(FormationOffset())
        roles = [PostMergeRole(r) for r in d.get("post_merge_roles", ["engage"] * 4)]
        while len(roles) < 4:
            roles.append(PostMergeRole.ENGAGE)
        mask = list(d.get("drag_ship_mask", [True, False, False, False]))
        while len(mask) < 4:
            mask.append(False)
        return cls(
            formation=form[:3],
            commit_range_m=float(d.get("commit_range_m", 55000)),
            abort_loss_threshold=int(d.get("abort_loss_threshold", 2)),
            merge_geometry=MergeGeometry(d.get("merge_geometry", "bracket")),
            split_left_fraction=float(d.get("split_left_fraction", 0.5)),
            drag_ship_mask=mask[:4],
            weapon_doctrine=WeaponDoctrine(d.get("weapon_doctrine", "standard")),
            shoot_range_frac=float(d.get("shoot_range_frac", 0.75)),
            post_merge_roles=roles[:4],
            commit_alt_bias_m=float(d.get("commit_alt_bias_m", 0)),
            commit_speed_factor=float(d.get("commit_speed_factor", 1.05)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> TacticsGenome:
        geometries = list(MergeGeometry)
        doctrines = list(WeaponDoctrine)
        roles = list(PostMergeRole)
        form = [
            FormationOffset(
                right_m=float(rng.uniform(-3000, 3000)),
                forward_m=float(rng.uniform(-3000, 500)),
                up_m=float(rng.uniform(-1000, 1000)),
            )
            for _ in range(3)
        ]
        return cls(
            formation=form,
            commit_range_m=float(rng.uniform(35000, 70000)),
            abort_loss_threshold=int(rng.integers(1, 4)),
            merge_geometry=geometries[int(rng.integers(0, len(geometries)))],
            split_left_fraction=float(rng.uniform(0.25, 0.75)),
            drag_ship_mask=[bool(rng.random() < 0.35) for _ in range(4)],
            weapon_doctrine=doctrines[int(rng.integers(0, len(doctrines)))],
            shoot_range_frac=float(rng.uniform(0.45, 0.95)),
            post_merge_roles=[roles[int(rng.integers(0, len(roles)))] for _ in range(4)],
            commit_alt_bias_m=float(rng.uniform(-2000, 2000)),
            commit_speed_factor=float(rng.uniform(0.9, 1.2)),
        )


def crossover(a: TacticsGenome, b: TacticsGenome, rng: np.random.Generator) -> TacticsGenome:
    """Uniform crossover of high-level genes."""
    child = TacticsGenome.random(rng)  # placeholder; overwrite fields
    child.formation = [
        a.formation[i] if rng.random() < 0.5 else b.formation[i]
        for i in range(3)
    ]
    child.commit_range_m = a.commit_range_m if rng.random() < 0.5 else b.commit_range_m
    child.abort_loss_threshold = (
        a.abort_loss_threshold if rng.random() < 0.5 else b.abort_loss_threshold
    )
    child.merge_geometry = a.merge_geometry if rng.random() < 0.5 else b.merge_geometry
    child.split_left_fraction = (
        a.split_left_fraction if rng.random() < 0.5 else b.split_left_fraction
    )
    child.drag_ship_mask = [
        a.drag_ship_mask[i] if rng.random() < 0.5 else b.drag_ship_mask[i]
        for i in range(4)
    ]
    child.weapon_doctrine = a.weapon_doctrine if rng.random() < 0.5 else b.weapon_doctrine
    child.shoot_range_frac = a.shoot_range_frac if rng.random() < 0.5 else b.shoot_range_frac
    child.post_merge_roles = [
        a.post_merge_roles[i] if rng.random() < 0.5 else b.post_merge_roles[i]
        for i in range(4)
    ]
    child.commit_alt_bias_m = (
        a.commit_alt_bias_m if rng.random() < 0.5 else b.commit_alt_bias_m
    )
    child.commit_speed_factor = (
        a.commit_speed_factor if rng.random() < 0.5 else b.commit_speed_factor
    )
    return child


def mutate(g: TacticsGenome, rng: np.random.Generator, rate: float = 0.15) -> TacticsGenome:
    """In-place-style mutation returning a new genome."""
    d = g.to_dict()
    # Rebuild via random then selectively keep — cleaner to mutate fields
    out = TacticsGenome.from_dict(d)

    if rng.random() < rate:
        i = int(rng.integers(0, 3))
        out.formation[i] = FormationOffset(
            right_m=float(np.clip(out.formation[i].right_m + rng.normal(0, 400), -4000, 4000)),
            forward_m=float(np.clip(out.formation[i].forward_m + rng.normal(0, 400), -4000, 1000)),
            up_m=float(np.clip(out.formation[i].up_m + rng.normal(0, 200), -2000, 2000)),
        )
    if rng.random() < rate:
        out.commit_range_m = float(np.clip(out.commit_range_m + rng.normal(0, 5000), 25000, 80000))
    if rng.random() < rate:
        out.abort_loss_threshold = int(np.clip(out.abort_loss_threshold + rng.integers(-1, 2), 1, 3))
    if rng.random() < rate:
        out.merge_geometry = list(MergeGeometry)[int(rng.integers(0, len(MergeGeometry)))]
    if rng.random() < rate:
        out.split_left_fraction = float(np.clip(out.split_left_fraction + rng.normal(0, 0.1), 0.2, 0.8))
    if rng.random() < rate:
        i = int(rng.integers(0, 4))
        out.drag_ship_mask[i] = not out.drag_ship_mask[i]
    if rng.random() < rate:
        out.weapon_doctrine = list(WeaponDoctrine)[int(rng.integers(0, len(WeaponDoctrine)))]
    if rng.random() < rate:
        out.shoot_range_frac = float(np.clip(out.shoot_range_frac + rng.normal(0, 0.08), 0.4, 0.98))
    if rng.random() < rate:
        i = int(rng.integers(0, 4))
        out.post_merge_roles[i] = list(PostMergeRole)[int(rng.integers(0, len(PostMergeRole)))]
    if rng.random() < rate:
        out.commit_alt_bias_m = float(np.clip(out.commit_alt_bias_m + rng.normal(0, 400), -3000, 3000))
    if rng.random() < rate:
        out.commit_speed_factor = float(np.clip(out.commit_speed_factor + rng.normal(0, 0.05), 0.85, 1.25))
    return out
