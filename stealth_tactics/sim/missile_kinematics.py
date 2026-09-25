"""Spec 3a point-mass missile fly-out physics.

One set of equations, two implementations of the same step:

- ``advance_one``: plain-float version used by the sim (few missiles in flight;
  pure Python floats are several times faster than numpy on 3-vectors).
- ``advance_batch``: numpy version used to build the Rmax lookup table (thousands
  of fly-outs in lockstep).

``tests/test_missile_kinematics.py`` checks that both give identical trajectories.

Model (docs/specs/03a-missile-kinematics.md), ENU axes (x East, y North, z Up):

- Boost: thrust = mdot * Isp * g0 along the velocity for ``burn_time_s``.
- Drag: q S Cd0(M) cd_scale + k (n m g)^2 / (q S); 1976 US Standard Atmosphere.
- Lateral acceleration = PN command (N * Vc * Omega x v_hat) plus, with
  ``gravity``, the lift that carries the weight (g perpendicular to v); the total
  is clipped to n_max = min(g_max, q S CLmax / (m g)). Gravity acts on the body.
- Semi-implicit Euler (v first, then x), as in the prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sensor_config import MissileKinematicsConfig

G0 = 9.80665
R_AIR = 287.053
GAMMA = 1.4


# --------------------------------------------------------------- atmosphere --
def atmosphere(h: float):
    """1976 US Standard Atmosphere (0-32 km, geometric ~ geopotential).
    Returns (density kg/m^3, speed of sound m/s)."""
    if h < 0.0:
        h = 0.0
    if h < 11000.0:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** 5.255877
    elif h < 20000.0:
        T = 216.65
        p = 22632.06 * math.exp(-0.000157688 * (h - 11000.0))
    else:
        T = 216.65 + 0.001 * (h - 20000.0)
        p = 5474.889 * (216.65 / T) ** 34.16319
    return p / (R_AIR * T), math.sqrt(GAMMA * R_AIR * T)


def speed_of_sound(h: float) -> float:
    return atmosphere(h)[1]


def atmosphere_np(h: np.ndarray):
    h = np.maximum(h, 0.0)
    T1 = 288.15 - 0.0065 * np.minimum(h, 11000.0)
    T3 = 216.65 + 0.001 * np.maximum(h - 20000.0, 0.0)
    T = np.where(h < 11000.0, T1, np.where(h < 20000.0, 216.65, T3))
    p1 = 101325.0 * (T1 / 288.15) ** 5.255877
    p2 = 22632.06 * np.exp(-0.000157688 * (np.clip(h, 11000.0, 20000.0) - 11000.0))
    p3 = 5474.889 * (216.65 / T3) ** 34.16319
    p = np.where(h < 11000.0, p1, np.where(h < 20000.0, p2, p3))
    return p / (R_AIR * T), np.sqrt(GAMMA * R_AIR * T)


# --------------------------------------------------------------- constants ---
@dataclass(frozen=True)
class Derived:
    """Per-config constants precomputed once."""
    cfg: MissileKinematicsConfig
    S: float
    mdot: float
    thrust: float
    burnout_mass: float

    @staticmethod
    def of(cfg: MissileKinematicsConfig) -> "Derived":
        S = math.pi * (cfg.diameter_m / 2.0) ** 2
        mdot = cfg.propellant_kg / cfg.burn_time_s
        return Derived(cfg, S, mdot, mdot * cfg.isp_s * G0,
                       cfg.launch_mass_kg - cfg.propellant_kg)


_DERIVED: dict = {}


def derived(cfg: MissileKinematicsConfig) -> Derived:
    d = _DERIVED.get(cfg)
    if d is None:
        d = _DERIVED[cfg] = Derived.of(cfg)
    return d


def cd0(M: float, c: MissileKinematicsConfig) -> float:
    if M < c.mach_rise_start:
        return c.cd_subsonic
    if M < c.mach_cd_peak:
        return c.cd_subsonic + (M - c.mach_rise_start) / (c.mach_cd_peak - c.mach_rise_start) \
            * (c.cd_peak - c.cd_subsonic)
    return c.cd_peak * (c.mach_cd_peak / M) ** c.cd_super_exp


def cd0_np(M: np.ndarray, c: MissileKinematicsConfig) -> np.ndarray:
    rise = c.cd_subsonic + (M - c.mach_rise_start) / (c.mach_cd_peak - c.mach_rise_start) \
        * (c.cd_peak - c.cd_subsonic)
    sup = c.cd_peak * (c.mach_cd_peak / np.maximum(M, 1e-6)) ** c.cd_super_exp
    return np.where(M < c.mach_rise_start, c.cd_subsonic,
                    np.where(M < c.mach_cd_peak, rise, sup))


# ------------------------------------------------------------ scalar state ---
class KinState:
    """Missile kinematic state. Quacks like ``AircraftState`` for positions
    (``x``, ``y``, ``alt``, ``position()``, ``heading_rad``, ``speed_mps``)."""

    __slots__ = ("x", "y", "alt", "vx", "vy", "vz", "mass", "tf", "alive")

    def __init__(self, x=0.0, y=0.0, alt=0.0, vx=0.0, vy=0.0, vz=0.0,
                 mass=161.5, tf=0.0):
        self.x, self.y, self.alt = float(x), float(y), float(alt)
        self.vx, self.vy, self.vz = float(vx), float(vy), float(vz)
        self.mass, self.tf = float(mass), float(tf)
        self.alive = True

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.alt], dtype=float)

    @property
    def speed_mps(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    @property
    def heading_rad(self) -> float:
        return math.atan2(self.vx, self.vy)

    @property
    def pitch_rad(self) -> float:
        return math.atan2(self.vz, math.hypot(self.vx, self.vy))

    def mach(self) -> float:
        return self.speed_mps / speed_of_sound(self.alt)

    def copy(self) -> "KinState":
        k = KinState(self.x, self.y, self.alt, self.vx, self.vy, self.vz, self.mass, self.tf)
        k.alive = self.alive
        return k


def advance_one(s: KinState, ax: float, ay: float, az: float,
                avx: float, avy: float, avz: float, h: float, d: Derived) -> float:
    """Advance one step of ``h`` seconds with PN toward aim point (ax, ay, az)
    moving at (avx, avy, avz). Returns the load factor n used (g)."""
    c = d.cfg
    vx, vy, vz = s.vx, s.vy, s.vz
    V = math.sqrt(vx * vx + vy * vy + vz * vz)
    if V < 1e-6:
        V = 1e-6
    ux, uy, uz = vx / V, vy / V, vz / V
    rho, a = atmosphere(s.alt)
    M = V / a
    qS = 0.5 * rho * V * V * d.S
    # PN: acmd = N * Vc * (Omega x u), Omega = (r x vr) / |r|^2
    rx, ry, rz = ax - s.x, ay - s.y, az - s.alt
    wx, wy, wz = avx - vx, avy - vy, avz - vz
    d2 = rx * rx + ry * ry + rz * rz
    if d2 < 1e-6:
        d2 = 1e-6
    dist = math.sqrt(d2)
    vc = -(rx * wx + ry * wy + rz * wz) / dist
    ox, oy, oz = (ry * wz - rz * wy) / d2, (rz * wx - rx * wz) / d2, (rx * wy - ry * wx) / d2
    k = c.pn_gain * vc
    lx, ly, lz = k * (oy * uz - oz * uy), k * (oz * ux - ox * uz), k * (ox * uy - oy * ux)
    if c.gravity:     # lift carrying the weight: g perpendicular to v
        lx -= G0 * uz * ux
        ly -= G0 * uz * uy
        lz += G0 * (1.0 - uz * uz)
    m = s.mass
    nmax = qS * c.cl_max / (m * G0)
    if nmax > c.g_max:
        nmax = c.g_max
    an = math.sqrt(lx * lx + ly * ly + lz * lz)
    if an > nmax * G0:
        f = nmax * G0 / an
        lx, ly, lz = lx * f, ly * f, lz * f
        an = nmax * G0
    n = an / G0
    drag = qS * c.cd_scale * cd0(M, c) + c.k_induced * (n * m * G0) ** 2 / max(qS, 1e-6)
    bf = (c.burn_time_s - s.tf) / h
    bf = 0.0 if bf <= 0.0 else (1.0 if bf >= 1.0 else bf)
    along = (d.thrust * bf - drag) / m
    vx += (along * ux + lx) * h
    vy += (along * uy + ly) * h
    vz += (along * uz + lz - (G0 if c.gravity else 0.0)) * h
    s.vx, s.vy, s.vz = vx, vy, vz
    s.x += vx * h
    s.y += vy * h
    s.alt += vz * h
    s.mass = m - d.mdot * h * bf
    s.tf += h
    return n


def advance_batch(S: dict, ax, ay, az, avx, avy, avz, h: float, d: Derived) -> None:
    """Vectorised ``advance_one`` (in place) over a struct of 1-D arrays
    ``S = {x, y, alt, vx, vy, vz, mass, tf}``."""
    c = d.cfg
    sqrt = np.sqrt
    vx, vy, vz = S["vx"], S["vy"], S["vz"]
    V = np.maximum(sqrt(vx * vx + vy * vy + vz * vz), 1e-6)
    ux, uy, uz = vx / V, vy / V, vz / V
    rho, a = atmosphere_np(S["alt"])
    M = V / a
    qS = 0.5 * d.S * rho * V * V
    rx, ry, rz = ax - S["x"], ay - S["y"], az - S["alt"]
    wx, wy, wz = avx - vx, avy - vy, avz - vz
    d2 = np.maximum(rx * rx + ry * ry + rz * rz, 1e-6)
    vc = -(rx * wx + ry * wy + rz * wz) / sqrt(d2)
    ox, oy, oz = (ry * wz - rz * wy) / d2, (rz * wx - rx * wz) / d2, (rx * wy - ry * wx) / d2
    k = c.pn_gain * vc
    lx, ly, lz = k * (oy * uz - oz * uy), k * (oz * ux - ox * uz), k * (ox * uy - oy * ux)
    if c.gravity:
        lx = lx - G0 * uz * ux
        ly = ly - G0 * uz * uy
        lz = lz + G0 * (1.0 - uz * uz)
    m = S["mass"]
    nmax = np.minimum(qS * c.cl_max / (m * G0), c.g_max)
    an = sqrt(lx * lx + ly * ly + lz * lz)
    f = np.where(an > nmax * G0, nmax * G0 / np.maximum(an, 1e-12), 1.0)
    lx, ly, lz = lx * f, ly * f, lz * f
    n = an * f / G0
    drag = qS * c.cd_scale * cd0_np(M, c) + c.k_induced * (n * m * G0) ** 2 \
        / np.maximum(qS, 1e-6)
    bf = np.clip((c.burn_time_s - S["tf"]) / h, 0.0, 1.0)
    along = (d.thrust * bf - drag) / m
    vx = vx + (along * ux + lx) * h
    vy = vy + (along * uy + ly) * h
    vz = vz + (along * uz + lz - (G0 if c.gravity else 0.0)) * h
    S["vx"], S["vy"], S["vz"] = vx, vy, vz
    S["x"] = S["x"] + vx * h
    S["y"] = S["y"] + vy * h
    S["alt"] = S["alt"] + vz * h
    S["mass"] = m - d.mdot * h * bf
    S["tf"] = S["tf"] + h


def endgame_factor(mach: float, cfg: MissileKinematicsConfig) -> float:
    return cfg.endgame_factor(mach)
