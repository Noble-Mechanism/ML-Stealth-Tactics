"""Spec 5 H-L: action vector (13 outputs per Blue jet) and its mapping.

| idx | output | mapping |
|---|---|---|
| 0 | heading offset h | cmd_heading = ref + h x 180 deg; ref = perceived bearing to the
|   |                  | selected target, else the mission axis (north) |
| 1 | altitude z | cmd_alt = 100 + (z + 1)/2 x 14,900 m |
| 2 | speed s | cmd_speed = 90 + (s + 1)/2 x 250 m/s |
| 3-9 | target logits | slots 0-5 + "none"; argmax over present slots and none |
| 10 | fire | > 0: fire request at the selected target |
| 11 | radar | > 0: emitting (5 s minimum dwell, controller) |
| 12 | pair | > 0: shoot-shoot-assess for the next contact, else shoot-assess-shoot |

Deterministic, no sampling. ``encode_commands`` is the exact inverse used by
the adapter test (layer 1): it nudges each continuous output by a few ulps
until the decoder reproduces the scripted command bit for bit.
"""

from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .observation import MISSION_AXIS_RAD, ObsInfo

ALT_MIN_M, ALT_MAX_M = 100.0, 15000.0
SPD_MIN_MPS, SPD_MAX_MPS = 90.0, 340.0


@dataclass(frozen=True)
class ActionSpec:
    k: int = 6
    heading: int = 0
    alt: int = 1
    speed: int = 2
    target0: int = 3
    fire: int = 10
    radar: int = 11
    pair: int = 12

    @property
    def size(self) -> int:
        return self.target0 + self.k + 1 + 3

    @property
    def target_slice(self) -> slice:
        return slice(self.target0, self.target0 + self.k + 1)


ACTION_SPEC = ActionSpec()


@dataclass(frozen=True)
class Action:
    heading_rad: float
    alt_m: float
    speed_mps: float
    target_slot: Optional[int]
    target_key: Optional[str]
    fire: bool
    radar: bool
    pair: bool


def _c1(v: float) -> float:
    v = float(v)
    if not math.isfinite(v):
        return 0.0
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


_PI = Fraction(math.pi)
_TWO_PI = 2 * _PI


def decode_heading(h: float, ref: float) -> float:
    """ref + h x pi (wrapped once into [-pi, pi]), exact then rounded once."""
    v = Fraction(ref) + Fraction(h) * _PI
    if v > _PI:
        v -= _TWO_PI
    elif v < -_PI:
        v += _TWO_PI
    return float(v)


_ALT_SPAN = Fraction(ALT_MAX_M - ALT_MIN_M) / 2
_SPD_SPAN = Fraction(SPD_MAX_MPS - SPD_MIN_MPS) / 2


def decode_alt(z: float) -> float:
    return float(Fraction(ALT_MIN_M) + (Fraction(z) + 1) * _ALT_SPAN)


def decode_speed(s: float) -> float:
    return float(Fraction(SPD_MIN_MPS) + (Fraction(s) + 1) * _SPD_SPAN)


def select_target(logits, info: ObsInfo) -> Optional[int]:
    """Argmax over present slots and "none" (last logit). None = no target."""
    k = len(info.keys)
    best, best_v = None, None
    for i in range(k + 1):
        if i < k and info.keys[i] is None:
            continue
        v = float(logits[i])
        if not math.isfinite(v):
            continue
        if best_v is None or v > best_v:
            best, best_v = i, v
    if best is None or best == k:
        return None
    return best


def decode_action(out, info: ObsInfo, spec: ActionSpec = ACTION_SPEC) -> Action:
    out = np.asarray(out, dtype=np.float64)
    slot = select_target(out[spec.target_slice], info)
    key = info.keys[slot] if slot is not None else None
    ref = info.bearings[slot] if slot is not None else MISSION_AXIS_RAD
    return Action(
        heading_rad=decode_heading(_c1(out[spec.heading]), ref),
        alt_m=decode_alt(_c1(out[spec.alt])),
        speed_mps=decode_speed(_c1(out[spec.speed])),
        target_slot=slot, target_key=key,
        fire=bool(out[spec.fire] > 0.0) and key is not None,
        radar=bool(out[spec.radar] > 0.0),
        pair=bool(out[spec.pair] > 0.0))


# ------------------------------------------------------------ encoding --
def _nudge(decode, x0: float, target: float, lo: float = -1.0, hi: float = 1.0,
           max_ulps: int = 8) -> Tuple[float, bool]:
    """Find x near x0 (within [lo, hi]) with decode(x) == target exactly."""
    x0 = min(hi, max(lo, x0))
    if decode(x0) == target:
        return x0, True
    up = dn = x0
    for _ in range(max_ulps):
        up = min(hi, float(np.nextafter(up, np.inf)))
        if decode(up) == target:
            return up, True
        dn = max(lo, float(np.nextafter(dn, -np.inf)))
        if decode(dn) == target:
            return dn, True
    return x0, False


def encode_commands(heading_rad: float, alt_m: float, speed_mps: float,
                    target_key: Optional[str], fire: bool, info: ObsInfo,
                    radar: bool = True, pair: bool = False,
                    spec: ActionSpec = ACTION_SPEC) -> Tuple[np.ndarray, bool]:
    """Scripted commands -> 13 outputs. Returns (outputs, exact). ``exact`` is
    False if a continuous command could not be reproduced bit for bit or the
    fire target is not in the contact slots."""
    out = np.full(spec.size, -1.0)
    exact = True
    slot = None
    if target_key is not None:
        if target_key in info.keys:
            slot = info.keys.index(target_key)
        else:
            exact = False
    # heading: the fire target fixes the reference; otherwise any reference
    # (none = mission axis, or any present slot) may be chosen, so try them all
    # for a bit-exact decode (the target choice only matters for firing).
    refs = [slot] if (fire and slot is not None) else \
        [None] + [i for i, k in enumerate(info.keys) if k is not None]
    best = None
    for cand in refs:
        ref = info.bearings[cand] if cand is not None else MISSION_AXIS_RAD
        d = (heading_rad - ref + math.pi) % (2.0 * math.pi) - math.pi
        h, ok = _nudge(lambda x: decode_heading(x, ref), d / math.pi, heading_rad)
        if best is None or ok:
            best = (cand, h, ok)
        if ok:
            break
    slot, h, ok = best
    exact &= ok
    a = min(ALT_MAX_M, max(ALT_MIN_M, alt_m))
    z, ok = _nudge(decode_alt, (a - ALT_MIN_M) / (ALT_MAX_M - ALT_MIN_M) * 2.0 - 1.0, a)
    exact &= ok
    v = min(SPD_MAX_MPS, max(SPD_MIN_MPS, speed_mps))
    s, ok = _nudge(decode_speed, (v - SPD_MIN_MPS) / (SPD_MAX_MPS - SPD_MIN_MPS) * 2.0 - 1.0, v)
    exact &= ok
    out[spec.heading], out[spec.alt], out[spec.speed] = h, z, s
    logits = np.full(spec.k + 1, -1.0)
    logits[slot if slot is not None else spec.k] = 1.0
    out[spec.target_slice] = logits
    out[spec.fire] = 1.0 if (fire and slot is not None) else -1.0
    out[spec.radar] = 1.0 if radar else -1.0
    out[spec.pair] = 1.0 if pair else -1.0
    return out, exact
