"""Per-observer track store (Spec 1).

Each observing aircraft owns one ``TrackStore``: target_id -> ``Track``.
A Track fuses per-sensor *components* (radar / irst / rwr). Each component
coasts for ``TrackConfig.coast_s`` after its last hit and is then dropped; the
track is dropped when it has no components left.

Designed so Spec 2 (datalink) can add ``origin="datalink"`` components or merge
stores without touching sensor code: all state is plain data keyed by target id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


class TrackQuality(IntEnum):
    NONE = 0
    RWR = 1           # bearing-only (emitter)
    IRST = 2          # accurate angles, poor range
    RADAR = 3         # radar position track (not yet fire-control)
    FIRE_CONTROL = 4  # radar, close enough, held long enough -> can shoot / support


SOURCE_QUALITY = {
    "rwr": TrackQuality.RWR,
    "irst": TrackQuality.IRST,
    "radar": TrackQuality.RADAR,
}


@dataclass
class SensorComponent:
    source: str                 # "radar" | "irst" | "rwr"
    first_t: float
    last_t: float
    hits: int = 1
    origin: str = "own"         # reserved for Spec 2 ("datalink")
    bearing_rad: float = 0.0    # measured azimuth (0=N, clockwise)
    elevation_rad: float = 0.0
    range_est_m: Optional[float] = None  # None for RWR
    true_range_m: float = 0.0   # diagnostic only (not for tactics)
    meas_sigma_m: Optional[float] = None
    est_pos: Optional[np.ndarray] = None
    est_vel: Optional[np.ndarray] = None
    # Spec 2: observer position at measurement (needed for IRST triangulation)
    obs_pos: Optional[np.ndarray] = None
    # Spec 2 (remote copies only): position sigma at measurement time as computed
    # by the sender, and the sender jet id. Receiver adds coast growth x age.
    sigma_at_meas: Optional[float] = None
    sender: Optional[str] = None


@dataclass
class FusedTrack:
    """Best available estimate of one enemy in an observer's combined picture."""
    target_id: str
    est_pos: np.ndarray
    sigma_m: float
    sensor: str            # "radar" | "irst" | "tri"
    source_jet: str        # own id, flightmate id, or "B1+B2" for triangulation
    age_s: float           # age of the underlying measurement(s)
    own: bool              # True if from this jet's own sensors
    candidates: List[Tuple[str, float]] = field(default_factory=list)  # (label, sigma)
    tri: Optional[dict] = None  # best triangulation candidate (even if not chosen)


@dataclass
class Track:
    target_id: str
    first_t: float
    components: Dict[str, SensorComponent] = field(default_factory=dict)
    # Fire-control bookkeeping (radar only)
    radar_streak_start: Optional[float] = None
    radar_last_r50_m: float = 0.0
    fire_control: bool = False
    fc_since: Optional[float] = None
    # Spec 2: received copies, sender id -> source -> SensorComponent (origin="link")
    remote: Dict[str, Dict[str, SensorComponent]] = field(default_factory=dict)
    remote_msg_t: Dict[str, float] = field(default_factory=dict)   # snapshot time
    remote_fc: Dict[str, Tuple[float, bool]] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return bool(self.components) or bool(self.remote)

    @property
    def last_t(self) -> float:
        return max(c.last_t for c in self.components.values())

    def remote_fc_source(self, t: float, max_age_s: float) -> Optional[str]:
        """Flightmate whose latest received report says fire-control (fresh enough)."""
        best = None
        for sender, (snap_t, fc) in sorted(self.remote_fc.items()):
            if fc and t - snap_t <= max_age_s + 1e-9:
                if best is None or snap_t > best[1]:
                    best = (sender, snap_t)
        return best[0] if best else None

    @property
    def quality(self) -> TrackQuality:
        if self.fire_control:
            return TrackQuality.FIRE_CONTROL
        if not self.components:
            return TrackQuality.NONE
        return max(SOURCE_QUALITY[s] for s in self.components)

    def best_component(self) -> Optional[SensorComponent]:
        comps = [c for c in self.components.values() if c.est_pos is not None]
        if not comps:
            return None
        return max(comps, key=lambda c: (SOURCE_QUALITY[c.source], c.last_t))

    def position_estimate(self, t: float, cfg_growth_mps: float = 50.0
                          ) -> Optional[np.ndarray]:
        c = self.best_component()
        if c is None:
            return None
        age = max(0.0, t - c.last_t)
        if c.est_vel is not None:
            return c.est_pos + c.est_vel * age
        return c.est_pos.copy()

    def position_sigma(self, t: float, max_integration: float = 10.0,
                       coast_growth_mps: float = 50.0,
                       scan_period_s: float = 1.0) -> Optional[float]:
        """1-sigma 3D position error: shrinks with hold time, grows while coasting."""
        c = self.best_component()
        if c is None or c.meas_sigma_m is None:
            return None
        held = max(0.0, c.last_t - c.first_t)
        n_eff = min(1.0 + held / scan_period_s, max_integration)
        return c.meas_sigma_m / np.sqrt(n_eff) + coast_growth_mps * max(0.0, t - c.last_t)


class TrackStore:
    """All tracks held by one observer. ``tid in store`` = any own or received data.

    ``quality`` / ``is_fire_control`` / ``ids`` describe OWN sensors only;
    ``fused`` (recomputed every sim step) is the combined picture (Spec 2).
    """

    def __init__(self, owner_id: str) -> None:
        self.owner_id = owner_id
        self.tracks: Dict[str, Track] = {}
        self.fused: Dict[str, FusedTrack] = {}

    def __contains__(self, target_id: object) -> bool:
        return target_id in self.tracks

    def __iter__(self) -> Iterator[str]:
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def get(self, target_id: str) -> Optional[Track]:
        return self.tracks.get(target_id)

    def quality(self, target_id: str) -> TrackQuality:
        tr = self.tracks.get(target_id)
        return tr.quality if tr else TrackQuality.NONE

    def is_fire_control(self, target_id: str) -> bool:
        tr = self.tracks.get(target_id)
        return bool(tr and tr.fire_control)

    def ids(self, min_quality: TrackQuality = TrackQuality.RWR) -> List[str]:
        return [tid for tid, tr in self.tracks.items() if tr.quality >= min_quality]

    def fire_control_ids(self) -> List[str]:
        return [tid for tid, tr in self.tracks.items() if tr.fire_control]

    def remote_fc_source(self, target_id: str, t: float, max_age_s: float) -> Optional[str]:
        tr = self.tracks.get(target_id)
        return tr.remote_fc_source(t, max_age_s) if tr else None
