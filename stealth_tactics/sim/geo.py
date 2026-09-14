"""ENU <-> approximate WGS84 helpers for ACMI lon/lat/alt."""

from __future__ import annotations

import math
from typing import Tuple

# Reference origin (somewhere over open ocean / generic range)
REF_LAT_DEG = 35.0
REF_LON_DEG = -115.0
REF_ALT_M = 0.0

_METERS_PER_DEG_LAT = 111_320.0


def enu_to_llh(x_east: float, y_north: float, alt: float,
               ref_lat: float = REF_LAT_DEG,
               ref_lon: float = REF_LON_DEG) -> Tuple[float, float, float]:
    """Convert local ENU meters to lon, lat, alt (degrees, degrees, meters)."""
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    lat = ref_lat + y_north / _METERS_PER_DEG_LAT
    lon = ref_lon + x_east / meters_per_deg_lon
    return lon, lat, alt


def heading_rad_to_yaw_deg(heading_rad: float) -> float:
    """ACMI yaw: 0=North, positive clockwise — same convention as our heading."""
    import math
    return math.degrees(heading_rad) % 360.0
