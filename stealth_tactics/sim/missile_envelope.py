"""Spec 3a launch envelope: Rmax / Rne lookup table built from the fly-out model.

``get_envelope(cfg)`` returns a ``MissileEnvelope`` for a
``MissileKinematicsConfig``. The table is built once (vectorised bisection over
the whole grid with ``advance_batch``), cached in-process and on disk
(``~/.cache/stealth_tactics``, override with ``STEALTH_TACTICS_CACHE_DIR``; set
it to ``off`` to disable the disk cache), keyed by a hash of the config.

Table axes (config ``env_*``): altitude (m), shooter Mach, target aspect (deg,
0 = target nose-on to the shooter / hot, 180 = tail / cold), target Mach, and
launch angle off the shooter's nose (deg, signed: + = nose offset toward the
side the target is moving across the line of sight, "lead"; - = "lag"; the
missile starts along the shooter's velocity, so a lag shot must turn through
more angle). Target co-altitude, straight and level. ``rmax``: largest launch range that
hits. ``rne`` (no-escape): same, but the target turns cold at ``env_turn_g`` at
launch (horizontal turn, constant speed), capped at ``rmax`` (off the nose,
turning cold can help a lag shot, so no-escape = min of the two). Queries
are multilinear and clamped to the grid edges; a query whose interpolation weight is at least half on
no-shot cells (0) returns 0.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .missile_kinematics import (G0, advance_batch, atmosphere, atmosphere_np, derived,
                                  lofted_aim_np)
from .sensor_config import MissileKinematicsConfig, NM_M

ENGINE_VERSION = "8.1"    # Spec 8: loft bias (decay from loft start) + opening_grace_s

# outcome codes for batch shots
RUNNING, HIT, DEFEAT_SPEED, DEFEAT_OPENING, MISS_OVERSHOOT, TIMEOUT = range(6)
OUTCOME_NAMES = {HIT: "hit", DEFEAT_SPEED: "defeat_speed", DEFEAT_OPENING: "defeat_opening",
                 MISS_OVERSHOOT: "miss_overshoot", TIMEOUT: "timeout", RUNNING: "running"}


# ------------------------------------------------------- batched fly-outs ----
def batch_shots(cfg: MissileKinematicsConfig, alt_m, shooter_mach, aspect_deg,
                target_mach, range_m, turn_cold, turn_g: Optional[float] = None,
                off_nose_deg=0.0) -> Dict:
    """Fly many scripted shots in lockstep. Target at (range, 0) co-altitude
    (line of sight +x), heading so that its aspect to the shooter is
    ``aspect_deg`` (it moves toward +y for 0 < aspect < 180); if ``turn_cold``
    it turns (at ``turn_g``) to fly directly away (+x) from launch. Shooter at
    the origin, nose ``off_nose_deg`` off the line of sight (positive = toward
    +y, i.e. toward the side the target is moving: "lead"; negative = "lag");
    the missile starts along the shooter's nose. Missile fully supported
    (guides on the true target). Returns dict of arrays: outcome, tof,
    mach_end, mach_active, t_active."""
    d = derived(cfg)
    turn_g = cfg.env_turn_g if turn_g is None else turn_g
    alt = np.atleast_1d(np.asarray(alt_m, float))
    shape = np.broadcast(alt, np.asarray(shooter_mach), np.asarray(aspect_deg),
                         np.asarray(target_mach), np.asarray(range_m),
                         np.asarray(turn_cold), np.asarray(off_nose_deg)).shape
    alt = np.broadcast_to(alt, shape).ravel().astype(float)
    ms = np.broadcast_to(np.asarray(shooter_mach, float), shape).ravel()
    asp = np.broadcast_to(np.asarray(aspect_deg, float), shape).ravel()
    mt = np.broadcast_to(np.asarray(target_mach, float), shape).ravel()
    R = np.broadcast_to(np.asarray(range_m, float), shape).ravel()
    tc = np.broadcast_to(np.asarray(turn_cold, bool), shape).ravel()
    off = np.deg2rad(np.broadcast_to(np.asarray(off_nose_deg, float), shape).ravel())
    n = alt.size
    _, a = atmosphere_np(alt)
    vt = mt * a
    # struct of arrays for the live shots; compacted whenever shots finish
    S = {"x": np.zeros(n), "y": np.zeros(n), "alt": alt.copy(), "vx": ms * a * np.cos(off),
         "vy": ms * a * np.sin(off), "vz": np.zeros(n), "mass": np.full(n, cfg.launch_mass_kg),
         "tf": np.zeros(n), "tx": R.copy(), "ty": np.zeros(n), "tz": alt.copy(),
         "psi": np.pi - np.deg2rad(asp),
         "omega": np.where(tc, turn_g * G0 / np.maximum(vt, 1.0), 0.0),
         "vt": vt, "idx": np.arange(n), "mact": np.full(n, np.nan),
         "tact": np.full(n, np.nan), "opent": np.zeros(n), "lr0": np.full(n, np.nan)}
    out = np.zeros(n, int)
    tof = np.full(n, np.nan)
    mach_end = np.full(n, np.nan)
    mach_active = np.full(n, np.nan)
    t_active = np.full(n, np.nan)
    h = cfg.dt_s
    fuze2 = cfg.proximity_fuze_m ** 2
    burn = cfg.burn_time_s + 1e-9
    nsteps = int(math.ceil(cfg.max_flight_time_s / h - 1e-9))
    zeros = np.zeros(n)
    for step in range(nsteps):
        m_ = S["idx"].size
        if m_ == 0:
            break
        tvx, tvy = S["vt"] * np.cos(S["psi"]), S["vt"] * np.sin(S["psi"])
        tx0, ty0, tz0 = S["tx"], S["ty"], S["tz"]
        r0x, r0y, r0z = tx0 - S["x"], ty0 - S["y"], tz0 - S["alt"]
        # Spec 8 loft: bias aim upward while outside handoff (batch shots are
        # always fully supported / never autonomous until active_range).
        auto = ~np.isnan(S["mact"])   # already gone active on a prior step
        ax, ay, az, loft_on = lofted_aim_np(
            S["x"], S["y"], S["alt"], tx0, ty0, tz0, S["tf"], cfg, auto, S["lr0"])
        advance_batch(S, ax, ay, az, tvx, tvy, zeros[:m_], h, d)
        tx1, ty1 = tx0 + tvx * h, ty0 + tvy * h
        S["tx"], S["ty"] = tx1, ty1
        S["psi"] = np.maximum(0.0, S["psi"] - S["omega"] * h)
        r1x, r1y, r1z = tx1 - S["x"], ty1 - S["y"], tz0 - S["alt"]
        vx, vy, vz = S["vx"], S["vy"], S["vz"]
        _, a_now = atmosphere_np(S["alt"])
        M = np.sqrt(vx * vx + vy * vy + vz * vz) / a_now
        rng2 = r1x * r1x + r1y * r1y + r1z * r1z
        act = np.isnan(S["mact"]) & (rng2 <= cfg.active_range_m ** 2)
        if act.any():
            S["mact"] = np.where(act, M, S["mact"])
            S["tact"] = np.where(act, S["tf"], S["tact"])
        dx, dy, dz = r1x - r0x, r1y - r0y, r1z - r0z
        dd = np.maximum(dx * dx + dy * dy + dz * dz, 1e-12)
        ts = np.clip(-(r0x * dx + r0y * dy + r0z * dz) / dd, 0.0, 1.0)
        cx, cy, cz = r0x + ts * dx, r0y + ts * dy, r0z + ts * dz
        res = np.where(cx * cx + cy * cy + cz * cz <= fuze2, HIT, RUNNING)
        post = (S["tf"] > burn) & (res == RUNNING)
        slow = post & (M < cfg.defeat_min_mach)
        res = np.where(slow, DEFEAT_SPEED, res)
        if cfg.defeat_on_opening:
            rdot = r1x * (tvx - vx) + r1y * (tvy - vy) - r1z * vz
            opening = post & ~slow & (rdot > 0.0)
            near = rng2 < cfg.overshoot_range_m ** 2
            # miss_overshoot stays immediate; defeat_opening needs grace and
            # never fires while loft bias is still active (Spec 8 B-d).
            res = np.where(opening & near, MISS_OVERSHOOT, res)
            open_count = opening & ~near & ~loft_on & (res == RUNNING)
            S["opent"] = np.where(open_count, S["opent"] + h, 0.0)
            res = np.where((S["opent"] >= cfg.opening_grace_s) & (res == RUNNING),
                           DEFEAT_OPENING, res)
        if step == nsteps - 1:
            res = np.where(res == RUNNING, TIMEOUT, res)
        done = res != RUNNING
        if done.any():
            di = S["idx"][done]
            out[di] = res[done]
            tof[di] = S["tf"][done]
            mach_end[di] = M[done]
            mach_active[di] = S["mact"][done]
            t_active[di] = S["tact"][done]
            keep = ~done
            S = {k: v[keep] for k, v in S.items()}
    return {"outcome": out.reshape(shape), "tof": tof.reshape(shape),
            "mach_end": mach_end.reshape(shape), "mach_active": mach_active.reshape(shape),
            "t_active": t_active.reshape(shape)}


def bisect_rmax(cfg: MissileKinematicsConfig, alt_m, shooter_mach, aspect_deg, target_mach,
                turn_cold, lo_nm: Optional[float] = None, hi_nm: Optional[float] = None,
                tol_nm: Optional[float] = None, turn_g: Optional[float] = None,
                off_nose_deg=0.0) -> np.ndarray:
    """Largest hitting launch range (m), vectorised: coarse scan every
    ``env_coarse_nm`` then bisection above the largest coarse hit. 0 if no
    coarse range hits; hi if the longest still hits."""
    lo_nm = cfg.env_range_lo_nm if lo_nm is None else lo_nm
    hi_nm = cfg.env_range_hi_nm if hi_nm is None else hi_nm
    tol_nm = cfg.env_tol_nm if tol_nm is None else tol_nm
    b = np.broadcast(np.asarray(alt_m), np.asarray(shooter_mach), np.asarray(aspect_deg),
                     np.asarray(target_mach), np.asarray(turn_cold), np.asarray(off_nose_deg))
    args = [np.broadcast_to(np.asarray(x), b.shape).ravel()
            for x in (alt_m, shooter_mach, aspect_deg, target_mach, turn_cold, off_nose_deg)]
    n = args[0].size
    # coarse scan (hit/miss is not monotone near minimum range, e.g. beam
    # targets at very short range), then bisect above the largest coarse hit
    step_nm = cfg.env_coarse_nm
    coarse = np.arange(lo_nm, hi_nm + 1e-9, step_nm)
    if coarse[-1] < hi_nm - 1e-9:
        coarse = np.append(coarse, hi_nm)
    k = coarse.size
    rep = [np.repeat(x, k) for x in args]
    rc = np.tile(coarse * NM_M, n)
    hit_c = (batch_shots(cfg, *rep[:4], rc, rep[4], turn_g, rep[5])["outcome"]
             == HIT).reshape(n, k)
    any_hit = hit_c.any(axis=1)
    last = np.where(any_hit, k - 1 - np.argmax(hit_c[:, ::-1], axis=1), 0)
    lo = coarse[last] * NM_M
    top = last >= k - 1
    hi = np.where(top, lo, coarse[np.minimum(last + 1, k - 1)] * NM_M)
    active = any_hit & ~top
    while active.any() and np.max(hi[active] - lo[active]) > tol_nm * NM_M:
        ia = np.where(active)[0]
        mid = 0.5 * (lo[ia] + hi[ia])
        hit = batch_shots(cfg, *(x[ia] for x in args[:4]), mid, args[4][ia],
                          turn_g, args[5][ia])["outcome"] == HIT
        lo[ia[hit]] = mid[hit]
        hi[ia[~hit]] = mid[~hit]
        active[ia] = (hi[ia] - lo[ia]) > tol_nm * NM_M
    r = np.where(any_hit, lo, 0.0)
    return r.reshape(b.shape)


# ----------------------------------------------------------------- table -----
def _axes(cfg: MissileKinematicsConfig) -> Tuple[Tuple[float, ...], ...]:
    return (tuple(cfg.env_alt_m), tuple(cfg.env_shooter_mach),
            tuple(cfg.env_aspect_deg), tuple(cfg.env_target_mach),
            tuple(cfg.env_off_nose_deg))


def _build_slice(cfg: MissileKinematicsConfig, off_nose: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rmax / Rne over the first four axes for one launch off-nose angle."""
    ax = _axes(cfg)[:4]
    grid = np.meshgrid(*[np.asarray(a, float) for a in ax], indexing="ij")
    shape = grid[0].shape
    flat = [g.ravel() for g in grid]
    both = [np.concatenate([f, f]) for f in flat]
    k = flat[0].size
    tc = np.concatenate([np.zeros(k, bool), np.ones(k, bool)])
    r = bisect_rmax(cfg, *both, tc, off_nose_deg=off_nose)
    return r[:k].reshape(shape), r[k:].reshape(shape)


def _workers(n_jobs: int) -> int:
    env = os.environ.get("STEALTH_TACTICS_ENV_WORKERS")
    if env:
        try:
            return max(1, min(int(env), n_jobs))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, n_jobs))


BUILD_INFO: Dict[str, float] = {}


def build_tables(cfg: MissileKinematicsConfig,
                 workers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Full 5-D tables (alt, shooter Mach, aspect, target Mach, off-nose). One
    job per off-nose bin, run in a process pool (``STEALTH_TACTICS_ENV_WORKERS``,
    default = CPU count; 1 = serial, also the fallback if the pool fails)."""
    import time
    t0 = time.time()
    offs = list(_axes(cfg)[4])
    nw = _workers(len(offs)) if workers is None else max(1, min(workers, len(offs)))
    slices = None
    if nw > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=nw,
                                     mp_context=mp.get_context("spawn")) as ex:
                slices = list(ex.map(_build_slice, [cfg] * len(offs), offs))
        except Exception:        # no process support (sandbox etc.): serial
            slices = None
            nw = 1
    if slices is None:
        slices = [_build_slice(cfg, o) for o in offs]
    rmax = np.stack([s_[0] for s_ in slices], axis=-1)
    # No-escape = the shorter of "target turns cold" and "target holds course":
    # off the nose (lag shots at high aspect) turning cold removes the cross-LOS
    # motion the missile must chase and can make the shot easier, so the raw
    # turn-cold range may exceed Rmax.
    rne = np.minimum(np.stack([s_[1] for s_ in slices], axis=-1), rmax)
    BUILD_INFO.update(seconds=time.time() - t0, workers=nw, cells=float(rmax.size))
    return rmax, rne


# Config fields that do not change the kinematics (Pk model): not part of the key
_NON_KINEMATIC = {"base_pk", "endgame_mach_full", "endgame_mach_floor", "endgame_pk_floor",
                  "support_drop_factor", "support_gap_tol_s", "min_launch_range_m",
                  "max_off_boresight_deg"}


def _cache_key(cfg: MissileKinematicsConfig) -> str:
    items = [(f.name, getattr(cfg, f.name)) for f in dataclasses.fields(cfg)
             if f.name not in _NON_KINEMATIC]
    txt = ENGINE_VERSION + repr(items)
    return hashlib.sha1(txt.encode()).hexdigest()[:16]


def _cache_dir() -> Optional[Path]:
    env = os.environ.get("STEALTH_TACTICS_CACHE_DIR")
    if env and env.lower() == "off":
        return None
    return Path(env) if env else Path.home() / ".cache" / "stealth_tactics"


_MEM: Dict[str, "MissileEnvelope"] = {}


def get_envelope(cfg: MissileKinematicsConfig) -> "MissileEnvelope":
    key = _cache_key(cfg)
    env = _MEM.get(key)
    if env is not None:
        return env
    cdir = _cache_dir()
    path = cdir / f"envelope_{key}.npz" if cdir else None
    rmax = rne = None
    if path is not None and path.is_file():
        try:
            z = np.load(path)
            rmax, rne = z["rmax"], z["rne"]
        except Exception:        # corrupt cache: rebuild
            rmax = rne = None
    if rmax is None:
        rmax, rne = build_tables(cfg)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
                np.savez(tmp, rmax=rmax, rne=rne)
                os.replace(tmp, path)
            except OSError:
                pass
    env = MissileEnvelope(cfg, rmax, rne)
    _MEM[key] = env
    return env


def off_nose_signed(shooter_state, target_state) -> float:
    """Launch angle off the shooter's nose (deg, horizontal): angle between the
    shooter heading and the line of sight to the target. Positive = "lead" (nose
    offset toward the side the target is moving across the line of sight, or
    no cross-LOS target motion), negative = "lag"."""
    dx, dy = target_state.x - shooter_state.x, target_state.y - shooter_state.y
    los = math.atan2(dx, dy)                          # compass bearing to target
    off = (shooter_state.heading_rad - los + math.pi) % (2 * math.pi) - math.pi
    # headings are compass (clockwise), so a positive off = nose clockwise of LOS;
    # target cross-LOS velocity in the same clockwise sense:
    th = target_state.heading_rad
    cross = math.sin(th - los)                        # > 0: target moving clockwise
    mag = abs(math.degrees(off))
    if abs(cross) < 1e-6 or (cross > 0.0) == (off > 0.0) or off == 0.0:
        return mag
    return -mag


class MissileEnvelope:
    """Interpolated Rmax / Rne (no-escape) lookups in metres."""

    def __init__(self, cfg: MissileKinematicsConfig, rmax: np.ndarray, rne: np.ndarray):
        self.cfg = cfg
        self.axes = _axes(cfg)
        self.rmax_table = np.asarray(rmax, float)
        self.rne_table = np.minimum(np.asarray(rne, float), self.rmax_table)
        self._rmax_l = self.rmax_table.tolist()
        self._rne_l = self.rne_table.tolist()

    @staticmethod
    def _locate(axis: Sequence[float], v: float) -> Tuple[int, float]:
        if len(axis) == 1:
            return 0, 0.0
        if v <= axis[0]:
            return 0, 0.0
        if v >= axis[-1]:
            return len(axis) - 2, 1.0
        i = bisect.bisect_right(axis, v) - 1
        return i, (v - axis[i]) / (axis[i + 1] - axis[i])

    def _corners(self, *vals: float):
        """Non-zero-weight corners of the interpolation cell: list of (index
        tuple, weight), clamped at the grid edges."""
        corners = [((), 1.0)]
        for axis, v in zip(self.axes, vals):
            i, f = self._locate(axis, v)
            last = len(axis) - 1
            nxt = []
            for idx, w in corners:
                if f < 1.0:
                    nxt.append((idx + (min(i, last),), w * (1.0 - f)))
                if f > 0.0:
                    nxt.append((idx + (min(i + 1, last),), w * f))
            corners = nxt
        return corners

    @staticmethod
    def _blend(tab, corners) -> float:
        """Multilinear blend. No-shot cells (0: no range hits, e.g. past the lag
        / off-nose cliff): if they carry at least half of the weight the result
        is 0 (the nearest cells say "no shot"); otherwise the plain blend, which
        pulls the value down toward the cliff."""
        tot = zero = 0.0
        for (a, b, c, d, e), w in corners:
            v = tab[a][b][c][d][e]
            if v <= 0.0:
                zero += w
            tot += w * v
        return 0.0 if zero >= 0.5 - 1e-12 else tot

    def _interp(self, tab, *vals: float) -> float:
        return self._blend(tab, self._corners(*vals))

    def rmax_m(self, alt_m: float, shooter_mach: float, aspect_deg: float,
               target_mach: float, off_nose_deg: float = 0.0) -> float:
        return self._interp(self._rmax_l, alt_m, shooter_mach, aspect_deg, target_mach,
                            off_nose_deg)

    def rne_m(self, alt_m: float, shooter_mach: float, aspect_deg: float,
              target_mach: float, off_nose_deg: float = 0.0) -> float:
        return self._interp(self._rne_l, alt_m, shooter_mach, aspect_deg, target_mach,
                            off_nose_deg)

    # ---- geometry helpers (sim aircraft states) ----
    @staticmethod
    def geometry(shooter_state, target_state) -> Tuple[float, float, float, float, float]:
        """(mean altitude m, shooter Mach, target aspect deg, target Mach,
        signed launch off-nose deg)."""
        alt = 0.5 * (shooter_state.alt + target_state.alt)
        ms = shooter_state.speed_mps / atmosphere(shooter_state.alt)[1]
        mt = target_state.speed_mps / atmosphere(target_state.alt)[1]
        # aspect: angle between target velocity and the target->shooter line
        dx, dy = shooter_state.x - target_state.x, shooter_state.y - target_state.y
        los = math.atan2(dx, dy)
        diff = (target_state.heading_rad - los + math.pi) % (2 * math.pi) - math.pi
        return alt, ms, abs(math.degrees(diff)), mt, off_nose_signed(shooter_state,
                                                                      target_state)

    def for_states(self, shooter_state, target_state) -> Tuple[float, float]:
        """(Rmax m, Rne m) for the current shooter/target geometry."""
        c = self._corners(*self.geometry(shooter_state, target_state))
        return self._blend(self._rmax_l, c), self._blend(self._rne_l, c)

    def rmax_for_states(self, shooter_state, target_state) -> float:
        """Rmax (m) only (launch gating)."""
        return self._blend(self._rmax_l,
                           self._corners(*self.geometry(shooter_state, target_state)))
