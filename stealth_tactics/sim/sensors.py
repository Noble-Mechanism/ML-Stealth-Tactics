"""Spec 1 sensor model: probabilistic radar, IRST, RWR, and per-jet tracks.

All numbers come from ``sensor_config.py`` (unclassified placeholders).

Summary (details + formulas in docs/specs/01-sensors.md):
- Radar (1 Hz): inside a +/-60 deg az/el field of regard, per-scan
  Pd(R) = 1 / (1 + (R/R50)^k), R50 = ref_range * rcs_eff^0.25,
  k = 40 / (snr_slope_db * ln10) (logistic in SNR dB; SNR ~ R^-4).
- Signature: F-35 azimuth-aspect table with piecewise-cosine interpolation;
  Red isotropic.
- IRST (1 Hz, passive): R50 grows nose->tail (hot) and with target speed; accurate
  angles, poor range. IRST tracks never support a missile shot.
- RWR (every step): bearing-only contact when inside an enemy radar's field of
  regard and within emitter_detect_factor * emitter ref range (F-35 LPI 0.6,
  Red 1.5).
- Tracks coast 10 s after the last hit; fire-control quality = radar track held
  >= 3 s continuously and within 0.7 * R50.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aircraft import Aircraft, AircraftState, bearing_to, _angle_diff
from .sensor_config import DEFAULT_SENSOR_CONFIG, SensorConfig
from .tracks import SensorComponent, Track, TrackStore
from .fusion import fuse_store


# ------------------------------------------------------------ geometry -----
def distance_3d(a: AircraftState, b: AircraftState) -> float:
    """Pure-math 3D distance (hot path; avoids numpy overhead)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.alt - b.alt) ** 2)


def velocity_vector(st: AircraftState) -> np.ndarray:
    return np.array([st.speed_mps * math.sin(st.heading_rad),
                     st.speed_mps * math.cos(st.heading_rad), 0.0])


def off_nose_angles(observer: AircraftState, target: AircraftState) -> Tuple[float, float]:
    """(|azimuth off nose|, elevation) of target from observer, radians.

    Point-mass model has no pitch/roll, so the body frame is heading-only.
    """
    brg = bearing_to(observer, target)
    az = abs(_angle_diff(brg, observer.heading_rad))
    horiz = math.hypot(target.x - observer.x, target.y - observer.y)
    el = math.atan2(target.alt - observer.alt, horiz)
    return az, el


def in_field_of_regard(observer: AircraftState, target: AircraftState,
                       az_half_deg: float, el_half_deg: float) -> bool:
    az, el = off_nose_angles(observer, target)
    return az <= math.radians(az_half_deg) + 1e-12 and abs(el) <= math.radians(el_half_deg) + 1e-12


def aspect_angle_rad(target: Aircraft, observer: Aircraft) -> float:
    """
    Aspect of *target* as seen by *observer* (azimuth only): angle between the
    target's heading and the direction target -> observer.
    0 = observer on the target's nose, pi/2 = beam, pi = tail.
    """
    brg = bearing_to(target.state, observer.state)
    return abs(_angle_diff(brg, target.state.heading_rad))


# ----------------------------------------------------------- signature -----
def cosine_interp(table: Sequence[Tuple[float, float]], x: float) -> float:
    """Piecewise-cosine interpolation over sorted (x, y) knots (C1, clamped)."""
    if x <= table[0][0]:
        return float(table[0][1])
    for (x0, y0), (x1, y1) in zip(table[:-1], table[1:]):
        if x <= x1:
            u = (x - x0) / (x1 - x0)
            w = (1.0 - math.cos(math.pi * u)) / 2.0
            return float(y0 + (y1 - y0) * w)
    return float(table[-1][1])


def effective_rcs(target: Aircraft, observer: Aircraft,
                  cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    """Aspect-dependent RCS factor of *target* seen from *observer*."""
    key = target.ac_type.value
    sig = cfg.signature
    table = sig.tables.get(key)
    if table is None:
        return float(sig.isotropic_rcs.get(key, target.params.rcs_factor
                                           if target.params.rcs_factor else sig.default_rcs))
    return cosine_interp(table, math.degrees(aspect_angle_rad(target, observer)))


# --------------------------------------------------------------- radar -----
def radar_ref_range(observer: Aircraft, cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    return float(cfg.radar.ref_range_m.get(observer.ac_type.value,
                                           observer.params.radar_range_m))


def radar_r50(observer: Aircraft, target: Aircraft,
              cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    """50%-per-scan detection range: ref_range * rcs_eff^0.25."""
    return radar_ref_range(observer, cfg) * effective_rcs(target, observer, cfg) ** 0.25


def pd_curve(range_m: float, r50_m: float, k: float) -> float:
    """Pd(R) = 1 / (1 + (R/R50)^k): 1 at R=0, 0.5 at R50, -> 0 far out."""
    if r50_m <= 0:
        return 0.0
    ratio = range_m / r50_m
    if ratio <= 0:
        return 1.0
    lx = k * math.log(ratio)
    if lx > 700:
        return 0.0
    return 1.0 / (1.0 + math.exp(lx))


def radar_pd(observer: Aircraft, target: Aircraft,
             cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    """Per-scan radar Pd including field-of-regard and instrumented range."""
    rc = cfg.radar
    if not in_field_of_regard(observer.state, target.state, rc.for_az_deg, rc.for_el_deg):
        return 0.0
    d = distance_3d(observer.state, target.state)
    if d > rc.instrumented_range_factor * radar_ref_range(observer, cfg):
        return 0.0
    return pd_curve(d, radar_r50(observer, target, cfg), rc.pd_exponent)


# ---------------------------------------------------------------- IRST -----
def irst_speed_factor(speed_mps: float, cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    ic = cfg.irst
    f = (max(speed_mps, 1.0) / ic.ref_speed_mps) ** ic.speed_exponent
    return float(min(max(f, ic.speed_factor_min), ic.speed_factor_max))


def irst_r50(observer: Aircraft, target: Aircraft,
             cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    """
    R50_ir = [nose + (tail - nose) * (1 - cos(aspect))/2] * speed_factor * ir_scale
    """
    ic = cfg.irst
    key = observer.ac_type.value
    nose = ic.nose_r50_m.get(key, 25_000.0)
    tail = ic.tail_r50_m.get(key, 50_000.0)
    asp = aspect_angle_rad(target, observer)
    blend = (1.0 - math.cos(asp)) / 2.0
    scale = ic.target_ir_scale.get(target.ac_type.value, 1.0)
    return (nose + (tail - nose) * blend) * irst_speed_factor(target.state.speed_mps, cfg) * scale


def irst_pd(observer: Aircraft, target: Aircraft,
            cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    ic = cfg.irst
    if not in_field_of_regard(observer.state, target.state, ic.for_az_deg, ic.for_el_deg):
        return 0.0
    d = distance_3d(observer.state, target.state)
    r50 = irst_r50(observer, target, cfg)
    if d > ic.max_range_factor * r50:
        return 0.0
    return pd_curve(d, r50, ic.pd_exponent)


# ----------------------------------------------------------------- RWR -----
def rwr_range(emitter: Aircraft, cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> float:
    """Range at which an enemy RWR intercepts *emitter*'s radar."""
    f = cfg.rwr.emitter_detect_factor.get(emitter.ac_type.value, 1.0)
    return f * radar_ref_range(emitter, cfg)


def rwr_detects(receiver: Aircraft, emitter: Aircraft,
                cfg: SensorConfig = DEFAULT_SENSOR_CONFIG) -> bool:
    """Deterministic: receiver inside emitter's radar FOR and within rwr_range."""
    if not receiver.state.alive or not emitter.state.alive:
        return False
    if receiver.coalition == emitter.coalition:
        return False
    rc = cfg.radar
    if not in_field_of_regard(emitter.state, receiver.state, rc.for_az_deg, rc.for_el_deg):
        return False
    return distance_3d(receiver.state, emitter.state) <= rwr_range(emitter, cfg)


# --------------------------------------------------------- sensor model ----
def _event(t: float, etype: str, obs: Aircraft, tgt_id: str, rng_m: float, text: str) -> dict:
    return {"t": t, "type": etype, "observer": obs.id, "target": tgt_id,
            "range_m": rng_m, "text": text}


class SensorModel:
    """Runs all sensors and owns one ``TrackStore`` per aircraft."""

    def __init__(self, rng: np.random.Generator,
                 config: Optional[SensorConfig] = None) -> None:
        self.rng = rng
        self.cfg = config or DEFAULT_SENSOR_CONFIG
        self.stores: Dict[str, TrackStore] = {}
        self._next_radar_t = 0.0
        self._next_irst_t = 0.0
        self._next_rwr_t = 0.0

    # ---- single-draw helpers (used by update, tests and the sensor table) --
    def radar_measure(self, obs: Aircraft, tgt: Aircraft) -> Optional[dict]:
        pd = radar_pd(obs, tgt, self.cfg)
        if pd <= 0.0 or self.rng.random() >= pd:
            return None
        rc = self.cfg.radar
        d = distance_3d(obs.state, tgt.state)
        sigma = math.hypot(rc.range_sigma_m, d * math.radians(rc.angle_sigma_deg))
        est = tgt.state.position() + self.rng.normal(0.0, sigma / math.sqrt(3.0), 3)
        vel = velocity_vector(tgt.state) + self.rng.normal(0.0, rc.velocity_sigma_mps, 3)
        rel = est - obs.state.position()
        return {
            "range_est": float(np.linalg.norm(rel)),
            "true_range": d,
            "bearing": float(math.atan2(rel[0], rel[1])),
            "elevation": float(math.atan2(rel[2], math.hypot(rel[0], rel[1]))),
            "sigma": sigma, "est_pos": est, "est_vel": vel,
            "r50": radar_r50(obs, tgt, self.cfg),
        }

    def irst_measure(self, obs: Aircraft, tgt: Aircraft) -> Optional[dict]:
        pd = irst_pd(obs, tgt, self.cfg)
        if pd <= 0.0 or self.rng.random() >= pd:
            return None
        ic = self.cfg.irst
        d = distance_3d(obs.state, tgt.state)
        bs = math.radians(ic.bearing_sigma_deg)
        brg = bearing_to(obs.state, tgt.state) + self.rng.normal(0.0, bs)
        el = math.atan2(tgt.state.alt - obs.state.alt,
                        math.hypot(tgt.state.x - obs.state.x, tgt.state.y - obs.state.y))
        el += self.rng.normal(0.0, bs)
        r_est = d * max(0.2, 1.0 + self.rng.normal(0.0, ic.range_error_frac))
        horiz = r_est * math.cos(el)
        est = obs.state.position() + np.array(
            [horiz * math.sin(brg), horiz * math.cos(brg), r_est * math.sin(el)])
        return {
            "range_est": r_est, "true_range": d, "bearing": brg, "elevation": el,
            "sigma": math.hypot(ic.range_error_frac * d, d * bs),
            "est_pos": est, "est_vel": None,
        }

    def rwr_measure(self, receiver: Aircraft, emitter: Aircraft) -> Optional[dict]:
        if not rwr_detects(receiver, emitter, self.cfg):
            return None
        brg = bearing_to(receiver.state, emitter.state) + self.rng.normal(
            0.0, math.radians(self.cfg.rwr.bearing_sigma_deg))
        return {"bearing": brg, "true_range": distance_3d(receiver.state, emitter.state)}

    # ---- main update -----------------------------------------------------
    def store(self, obs_id: str) -> TrackStore:
        if obs_id not in self.stores:
            self.stores[obs_id] = TrackStore(obs_id)
        return self.stores[obs_id]

    def update(self, aircraft: List[Aircraft], t: float) -> List[dict]:
        """Run due sensors at sim time t, maintain tracks, return sensor events."""
        eps = 1e-9
        do_radar = t >= self._next_radar_t - eps
        do_irst = t >= self._next_irst_t - eps
        do_rwr = t >= self._next_rwr_t - eps
        if do_radar:
            self._next_radar_t += self.cfg.radar.scan_period_s
            while self._next_radar_t <= t + eps:
                self._next_radar_t += self.cfg.radar.scan_period_s
        if do_irst:
            self._next_irst_t += self.cfg.irst.scan_period_s
            while self._next_irst_t <= t + eps:
                self._next_irst_t += self.cfg.irst.scan_period_s
        if do_rwr and self.cfg.rwr.update_period_s > 0:
            self._next_rwr_t = t + self.cfg.rwr.update_period_s

        events: List[dict] = []
        by_id = {a.id: a for a in aircraft}
        for obs in aircraft:
            store = self.store(obs.id)
            if not obs.state.alive:
                store.tracks.clear()
                continue
            for tgt in aircraft:
                if tgt.coalition == obs.coalition or not tgt.state.alive:
                    continue
                if do_radar:
                    m = self.radar_measure(obs, tgt)
                    if m:
                        self._apply(store, obs, tgt.id, "radar", m, t, events)
                if do_irst:
                    m = self.irst_measure(obs, tgt)
                    if m:
                        self._apply(store, obs, tgt.id, "irst", m, t, events)
                if do_rwr:
                    m = self.rwr_measure(obs, tgt)
                    if m:
                        self._apply(store, obs, tgt.id, "rwr", m, t, events)
            self._maintain(store, obs, by_id, t, events)
            fuse_store(store, obs.coalition.value, t, self.cfg)
        return events

    def _apply(self, store: TrackStore, obs: Aircraft, tid: str, source: str,
               m: dict, t: float, events: List[dict]) -> None:
        tr = store.tracks.get(tid)
        if tr is None:
            tr = Track(target_id=tid, first_t=t)
            store.tracks[tid] = tr
        comp = tr.components.get(source)
        if comp is None:
            comp = SensorComponent(source=source, first_t=t, last_t=t, hits=0)
            tr.components[source] = comp
            label = {"radar": "radar detect", "irst": "IRST detect",
                     "rwr": "RWR contact"}[source]
            events.append(_event(t, f"{source}_detect", obs, tid, m["true_range"],
                                 f"{obs.name}: {label} on {tid}"))
        # Position estimate: simple 1/n_eff alpha filter so the ACTUAL estimate
        # error matches the reported sigma = meas_sigma / sqrt(n_eff) (Spec 2
        # fusion compares sigmas across jets, so they must be honest).
        meas_pos, meas_vel = m.get("est_pos"), m.get("est_vel")
        if comp.hits > 0 and comp.est_pos is not None and meas_pos is not None:
            period = (self.cfg.radar.scan_period_s if source == "radar"
                      else self.cfg.irst.scan_period_s)
            n_eff = min(1.0 + (t - comp.first_t) / period, self.cfg.track.max_integration)
            dt_c = t - comp.last_t
            pred = comp.est_pos + (comp.est_vel * dt_c if comp.est_vel is not None else 0.0)
            meas_pos = pred + (meas_pos - pred) / n_eff
            if meas_vel is not None and comp.est_vel is not None:
                meas_vel = comp.est_vel + (meas_vel - comp.est_vel) / n_eff
        comp.last_t = t
        comp.hits += 1
        comp.bearing_rad = m["bearing"]
        comp.elevation_rad = m.get("elevation", 0.0)
        comp.range_est_m = m.get("range_est")
        comp.true_range_m = m["true_range"]
        comp.meas_sigma_m = m.get("sigma")
        comp.est_pos = meas_pos
        comp.est_vel = meas_vel
        comp.obs_pos = obs.state.position()
        if source == "radar":
            if tr.radar_streak_start is None:
                tr.radar_streak_start = t
            tr.radar_last_r50_m = m["r50"]

    def fc_gate_m(self, obs: Aircraft, r50_m: float) -> float:
        """FC range gate: per-type reference (x rcs_eff^0.25) or fc_range_frac*R50."""
        tc = self.cfg.track
        ref = tc.fc_range_ref_m.get(obs.ac_type.value)
        if ref is None:
            return tc.fc_range_frac * r50_m
        # r50 = ref_range * rcs_eff^0.25  ->  rcs_eff^0.25 = r50 / ref_range
        return ref * r50_m / radar_ref_range(obs, self.cfg)

    def _maintain(self, store: TrackStore, obs: Aircraft, by_id: Dict[str, Aircraft],
                  t: float, events: List[dict]) -> None:
        tc = self.cfg.track
        for tid in list(store.tracks):
            tr = store.tracks[tid]
            tgt = by_id.get(tid)
            if tgt is None or not tgt.state.alive:
                del store.tracks[tid]  # destroyed target: drop silently
                continue
            for src in list(tr.components):
                c = tr.components[src]
                if t - c.last_t > tc.coast_s + 1e-9:
                    del tr.components[src]
                    events.append(_event(t, f"{src}_lost", obs, tid,
                                         distance_3d(obs.state, tgt.state),
                                         f"{obs.name}: {src.upper()} contact on {tid} dropped"))
            # Spec 2: received copies coast on the same 10 s rule (dead senders
            # stop updating and age out here)
            for sender in list(tr.remote):
                comps = tr.remote[sender]
                for src in list(comps):
                    if t - comps[src].last_t > tc.coast_s + 1e-9:
                        del comps[src]
                if not comps:
                    del tr.remote[sender]
            for sender in list(tr.remote_fc):
                if t - tr.remote_fc[sender][0] > tc.coast_s + 1e-9:
                    del tr.remote_fc[sender]
            radar = tr.components.get("radar")
            if radar is None or t - radar.last_t > tc.fc_max_gap_s + 1e-9:
                tr.radar_streak_start = None
            fc = (
                radar is not None
                and tr.radar_streak_start is not None
                and radar.last_t - tr.radar_streak_start >= tc.fc_min_hold_s - 1e-9
                and radar.range_est_m is not None
                and radar.range_est_m <= self.fc_gate_m(obs, tr.radar_last_r50_m)
            )
            if fc and not tr.fire_control:
                tr.fc_since = t
                events.append(_event(t, "fc_track", obs, tid,
                                     distance_3d(obs.state, tgt.state),
                                     f"{obs.name}: FIRE-CONTROL track on {tid}"))
            elif tr.fire_control and not fc:
                tr.fc_since = None
                events.append(_event(t, "fc_lost", obs, tid,
                                     distance_3d(obs.state, tgt.state),
                                     f"{obs.name}: fire-control track on {tid} lost"))
            tr.fire_control = fc
            if not tr.has_data:
                del store.tracks[tid]
                events.append(_event(t, "track_lost", obs, tid,
                                     distance_3d(obs.state, tgt.state),
                                     f"{obs.name}: track on {tid} dropped"))

    def primary_fc_target(self, obs_id: str) -> Optional[str]:
        """Closest fire-control track (for ACMI LockedTarget)."""
        st = self.stores.get(obs_id)
        if not st:
            return None
        fcs = [st.tracks[t] for t in st.fire_control_ids()]
        if not fcs:
            return None
        return min(fcs, key=lambda tr: tr.components["radar"].true_range_m).target_id
