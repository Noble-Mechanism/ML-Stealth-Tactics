"""Spec 2 fusion: best-error selection over own, received and triangulated tracks.

Candidates per enemy (observer's view at time t):
- own radar / IRST components: sigma = meas_sigma/sqrt(n_eff) + g * age
- received (datalink) radar / IRST components: sigma = sigma_at_meas + g * age,
  where age = t - measurement time (so link latency degrades them), g = 50 m/s
- IRST triangulation (Blue only): two jets' IRST bearings (own or received) whose
  lines of sight differ by >= 10 deg -> passive position fix (never fire-control)
The candidate with the smallest sigma wins.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sensor_config import SensorConfig
from .tracks import FusedTrack, SensorComponent, Track, TrackStore


def own_sigma_at_meas(c: SensorComponent, cfg: SensorConfig) -> Optional[float]:
    """Sigma of an own component at its last hit (hold-time integration only)."""
    if c.meas_sigma_m is None:
        return None
    period = cfg.radar.scan_period_s if c.source == "radar" else cfg.irst.scan_period_s
    held = max(0.0, c.last_t - c.first_t)
    n_eff = min(1.0 + held / period, cfg.track.max_integration)
    return c.meas_sigma_m / math.sqrt(n_eff)


def comp_sigma(c: SensorComponent, t: float, cfg: SensorConfig) -> Optional[float]:
    base = c.sigma_at_meas if c.sigma_at_meas is not None else own_sigma_at_meas(c, cfg)
    if base is None:
        return None
    return base + cfg.track.coast_growth_mps * max(0.0, t - c.last_t)


def comp_pos(c: SensorComponent, t: float) -> Optional[np.ndarray]:
    if c.est_pos is None:
        return None
    if c.est_vel is not None:
        return c.est_pos + c.est_vel * max(0.0, t - c.last_t)
    return c.est_pos.copy()


# ----------------------------------------------------------- triangulation --
def triangulate_pair(p1: np.ndarray, az1: float, el1: float,
                     p2: np.ndarray, az2: float, el2: float,
                     sigma_b_rad: float, min_angle_deg: float,
                     max_angle_deg: float = 170.0) -> Optional[dict]:
    """
    Intersect two horizontal bearing lines (az: 0=N clockwise) from p1, p2.

    Error propagation: a bearing error d1 at sensor 1 slides the fix along line 2
    by r1*d1/sin(g) (g = angle between the lines of sight); likewise for sensor 2.
    With independent errors sigma_b:
        sigma_h = sigma_b * sqrt(r1^2 + r2^2) / sin(g)          (2D RMS)
        sigma_v = sigma_b * r_min                              (elevation)
        sigma   = sqrt(sigma_h^2 + sigma_v^2)
    Returns None if g < min_angle (or > max_angle) or the rays do not meet ahead.
    """
    u1x, u1y = math.sin(az1), math.cos(az1)
    u2x, u2y = math.sin(az2), math.cos(az2)
    cosg = max(-1.0, min(1.0, u1x * u2x + u1y * u2y))
    g = math.acos(cosg)
    if g < math.radians(min_angle_deg) or g > math.radians(max_angle_deg):
        return None
    cross = u1x * u2y - u1y * u2x
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    r1 = (dx * u2y - dy * u2x) / cross
    r2 = (dx * u1y - dy * u1x) / cross
    if r1 <= 0 or r2 <= 0:
        return None
    xy = (p1[0] + r1 * u1x, p1[1] + r1 * u1y)
    z1 = p1[2] + r1 * math.tan(el1)
    z2 = p2[2] + r2 * math.tan(el2)
    w1, w2 = 1.0 / r1 ** 2, 1.0 / r2 ** 2
    z = (w1 * z1 + w2 * z2) / (w1 + w2)
    sin_g = math.sin(g)
    sigma_h = sigma_b_rad * math.sqrt(r1 ** 2 + r2 ** 2) / sin_g
    sigma_v = sigma_b_rad * min(r1, r2)
    return {"pos": np.array([xy[0], xy[1], z]), "sigma_geom": math.hypot(sigma_h, sigma_v),
            "angle_deg": math.degrees(g), "r1": r1, "r2": r2}


def irst_lines(tr: Track, owner_id: str, t: float, max_age: float
               ) -> List[Tuple[str, SensorComponent]]:
    lines = []
    c = tr.components.get("irst")
    if c is not None and c.obs_pos is not None and t - c.last_t <= max_age + 1e-9:
        lines.append((owner_id, c))
    for sender, comps in sorted(tr.remote.items()):
        c = comps.get("irst")
        if c is not None and c.obs_pos is not None and t - c.last_t <= max_age + 1e-9:
            lines.append((sender, c))
    return lines


def best_triangulation(tr: Track, owner_id: str, t: float, cfg: SensorConfig
                       ) -> Optional[dict]:
    dl = cfg.datalink
    lines = irst_lines(tr, owner_id, t, dl.tri_max_age_s)
    sb = math.radians(cfg.irst.bearing_sigma_deg)
    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (j1, c1), (j2, c2) = lines[i], lines[j]
            fix = triangulate_pair(c1.obs_pos, c1.bearing_rad, c1.elevation_rad,
                                   c2.obs_pos, c2.bearing_rad, c2.elevation_rad,
                                   sb, dl.tri_min_angle_deg, dl.tri_max_angle_deg)
            if fix is None:
                continue
            age = t - min(c1.last_t, c2.last_t)
            fix["sigma"] = fix["sigma_geom"] + cfg.track.coast_growth_mps * max(0.0, age)
            fix["age"] = age
            fix["jets"] = (j1, j2)
            if best is None or fix["sigma"] < best["sigma"]:
                best = fix
    return best


# ------------------------------------------------------------------ fusion --
def fuse_track(tr: Track, owner_id: str, t: float, cfg: SensorConfig,
               allow_triangulation: bool) -> Optional[FusedTrack]:
    # (sigma, label, sensor, source jet, own?, component-or-tri, age)
    cands: list = []
    for src, c in tr.components.items():
        if src not in cfg.datalink.shared_sources or c.est_pos is None:
            continue  # RWR: bearing only, no position
        sg = comp_sigma(c, t, cfg)
        if sg is not None:
            cands.append((sg, f"own-{src}", src, owner_id, True, c, t - c.last_t))
    for sender, comps in sorted(tr.remote.items()):
        for src, c in comps.items():
            sg = comp_sigma(c, t, cfg) if c.est_pos is not None else None
            if sg is not None:
                cands.append((sg, f"{sender}-{src}", src, sender, False, c, t - c.last_t))
    tri = None
    if allow_triangulation and ("irst" in tr.components or tr.remote):
        tri = best_triangulation(tr, owner_id, t, cfg)
    if tri is not None:
        cands.append((tri["sigma"], "tri", "tri", "+".join(tri["jets"]), False,
                      None, tri["age"]))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]))
    sg, _label, sensor, src_jet, own, comp, age = cands[0]
    pos = tri["pos"] if sensor == "tri" else comp_pos(comp, t)
    return FusedTrack(target_id=tr.target_id, est_pos=pos, sigma_m=sg, sensor=sensor,
                      source_jet=src_jet, age_s=age, own=own,
                      candidates=[(c[1], c[0]) for c in cands], tri=tri)


def fuse_store(store: TrackStore, coalition: str, t: float, cfg: SensorConfig) -> None:
    allow_tri = bool(cfg.datalink.triangulation.get(coalition, False))
    fused: Dict[str, FusedTrack] = {}
    for tid, tr in store.tracks.items():
        f = fuse_track(tr, store.owner_id, t, cfg, allow_tri)
        if f is not None:
            fused[tid] = f
    store.fused = fused
