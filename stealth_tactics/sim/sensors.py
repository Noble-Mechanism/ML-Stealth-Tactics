"""Simple detection / lock model with stealth RCS advantage."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from .aircraft import Aircraft, Coalition, distance_3d


class SensorModel:
    """
    Detection range scales with target RCS and own radar range.

    Assumptions (see docs/design.md):
    - Detection if range < radar_range * rcs_factor^0.25 * aspect_mod
    - Lock requires detection and within 0.7 * radar_range
    - Blue stealth (rcs_factor=0.05) is much harder for Red to detect
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def detect(
        self,
        observer: Aircraft,
        target: Aircraft,
    ) -> bool:
        if not observer.state.alive or not target.state.alive:
            return False
        if observer.coalition == target.coalition:
            return False

        d = distance_3d(observer.state, target.state)
        # Effective detection range: smaller RCS => shorter detection
        rcs = max(target.params.rcs_factor, 1e-4)
        # Radar equation approximation: range ~ R0 * rcs^0.25
        max_det = observer.params.radar_range_m * (rcs ** 0.25)
        # Mild aspect: head-on slightly better (simplified)
        return d <= max_det

    def can_lock(self, observer: Aircraft, target: Aircraft) -> bool:
        if not self.detect(observer, target):
            return False
        d = distance_3d(observer.state, target.state)
        return d <= observer.params.radar_range_m * 0.7 * (max(target.params.rcs_factor, 1e-4) ** 0.25)

    def update_tracks(
        self,
        aircraft: List[Aircraft],
    ) -> Dict[str, Set[str]]:
        """Return map: observer_id -> set of detected target ids."""
        tracks: Dict[str, Set[str]] = {}
        for obs in aircraft:
            if not obs.state.alive:
                tracks[obs.id] = set()
                continue
            seen: Set[str] = set()
            for tgt in aircraft:
                if self.detect(obs, tgt):
                    seen.add(tgt.id)
            tracks[obs.id] = seen
        return tracks
