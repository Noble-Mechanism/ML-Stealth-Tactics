"""Spec 6 A/B/D: the policy network (numpy, float64, deterministic)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np

from stealth_tactics.policy.action import (ACTION_SPEC, ALT_MAX_M, ALT_MIN_M, SPD_MAX_MPS,
                                           SPD_MIN_MPS)
from stealth_tactics.policy.observation import OBS_SPEC


@dataclass(frozen=True)
class Arch:
    """``encoder="flat"``: n_in -> hidden... -> n_out (tanh hidden).
    ``encoder="set"`` (B, behind a switch, off by default): a shared per-contact
    encoder (contact_size -> set_embed, tanh) applied to each of the K slots,
    mean- and max-pooled over present slots, concatenated with own + wingman
    inputs -> hidden... ; heading / alt / speed / fire / radar / pair from the
    trunk; each slot's target logit from [slot embedding, trunk]; "none" logit
    from the trunk."""
    n_in: int = OBS_SPEC.size
    hidden: Tuple[int, ...] = (64, 64)
    n_out: int = ACTION_SPEC.size
    encoder: str = "flat"
    set_embed: int = 32

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Arch":
        d = dict(d)
        d["hidden"] = tuple(d["hidden"])
        return Arch(**d)

    # ------------------------------------------------------------------
    def shapes(self) -> List[Tuple[str, Tuple[int, ...]]]:
        if self.encoder == "flat":
            sizes = [self.n_in, *self.hidden, self.n_out]
            out = []
            for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
                out += [(f"W{i}", (a, b)), (f"b{i}", (b,))]
            return out
        if self.encoder == "set":
            S = OBS_SPEC
            E = self.set_embed
            trunk_in = S.own_size + S.n_mates * S.mate_size + 2 * E
            sizes = [trunk_in, *self.hidden]
            out = [("We", (S.contact_size, E)), ("be", (E,))]
            for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
                out += [(f"W{i}", (a, b)), (f"b{i}", (b,))]
            h = sizes[-1]
            out += [("Wm", (h, 6)), ("bm", (6,)), ("ws", (E + h,)), ("bs", (1,)),
                    ("wn", (h,)), ("bn", (1,))]
            return out
        raise ValueError(f"unknown encoder {self.encoder!r}")

    @property
    def n_params(self) -> int:
        return int(sum(int(np.prod(s)) for _, s in self.shapes()))


DEFAULT_ARCH = Arch()


def unflatten(w: np.ndarray, arch: Arch) -> dict:
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (arch.n_params,):
        raise ValueError(f"weight vector has {w.size} values, arch needs {arch.n_params}")
    out, i = {}, 0
    for name, shp in arch.shapes():
        n = int(np.prod(shp))
        out[name] = w[i:i + n].reshape(shp)
        i += n
    return out


def flatten(layers: dict, arch: Arch) -> np.ndarray:
    return np.concatenate([np.asarray(layers[n], dtype=np.float64).ravel()
                           for n, _ in arch.shapes()])


def init_weights(arch: Arch, rng: np.random.Generator) -> np.ndarray:
    """D: weights N(0, 1/sqrt(fan_in)), biases N(0, 0.1) (as RandomMLPPolicy)."""
    parts = []
    for name, shp in arch.shapes():
        if len(shp) == 2:
            parts.append(rng.normal(0.0, 1.0 / math.sqrt(shp[0]), shp).ravel())
        elif name in ("ws", "wn"):
            parts.append(rng.normal(0.0, 1.0 / math.sqrt(shp[0]), shp))
        else:
            parts.append(rng.normal(0.0, 0.1, shp))
    return np.concatenate(parts)


def _heads(y: np.ndarray) -> np.ndarray:
    A = ACTION_SPEC
    y = y.copy()
    y[:, :3] = np.tanh(y[:, :3])
    y[:, A.fire:] = np.tanh(y[:, A.fire:])
    return y


class MLPPolicy:
    """Callable (n, 231) -> (n, 13). Stateless between calls (C: no memory)."""

    def __init__(self, weights: np.ndarray, arch: Arch = DEFAULT_ARCH) -> None:
        self.arch = arch
        self.weights = np.asarray(weights, dtype=np.float64)
        self.L = unflatten(self.weights, arch)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        if self.arch.encoder == "flat":
            n = len(self.arch.hidden)
            for i in range(n):
                x = np.tanh(x @ self.L[f"W{i}"] + self.L[f"b{i}"])
            return _heads(x @ self.L[f"W{n}"] + self.L[f"b{n}"])
        return self._set_forward(x)

    def _set_forward(self, x: np.ndarray) -> np.ndarray:
        S, A, L = OBS_SPEC, ACTION_SPEC, self.L
        N = x.shape[0]
        c = x[:, S.contacts_slice].reshape(N, S.k, S.contact_size)
        present = c[:, :, 0] > 0.5                                  # (N, K)
        e = np.tanh(c @ L["We"] + L["be"])                          # (N, K, E)
        m = present[:, :, None]
        cnt = np.maximum(present.sum(1, keepdims=True), 1)
        mean = (e * m).sum(1) / cnt
        mx = np.where(m, e, -1.0).max(1)
        mx = np.where(present.any(1, keepdims=True), mx, 0.0)
        own = x[:, S.own_slice]
        mates = x[:, S.mate_slice(0).start:S.mate_slice(S.n_mates - 1).stop]
        h = np.concatenate([own, mates, mean, mx], axis=1)
        for i in range(len(self.arch.hidden)):
            h = np.tanh(h @ L[f"W{i}"] + L[f"b{i}"])
        main = h @ L["Wm"] + L["bm"]                                 # 6: hdg alt spd fire radar pair
        E = self.arch.set_embed
        slot = e @ L["ws"][:E] + (h @ L["ws"][E:])[:, None] + L["bs"][0]   # (N, K)
        slot = np.where(present, slot, -1e9)
        none = h @ L["wn"] + L["bn"][0]
        y = np.zeros((N, A.size))
        y[:, :3] = main[:, :3]
        y[:, A.target_slice] = np.concatenate([slot, none[:, None]], axis=1)
        y[:, A.fire:] = main[:, 3:]
        return _heads(y)


def interface_fingerprint() -> str:
    """E: hash of the spec 5 interface (observation names and bounds, action
    layout and decode ranges, decision period). An old champion is refused,
    not misread, if any of it changes."""
    lo, hi = OBS_SPEC.bounds()
    blob = {"obs": OBS_SPEC.names(), "lo": lo.tolist(), "hi": hi.tolist(),
            "act": asdict(ACTION_SPEC), "alt": [ALT_MIN_M, ALT_MAX_M],
            "spd": [SPD_MIN_MPS, SPD_MAX_MPS], "decision_period_s": 1.0,
            "radar_dwell_s": 5.0}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]
