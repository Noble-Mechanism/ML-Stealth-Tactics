"""Spec 5 Q: adapters that drive Blue through the interface.

- ``ScriptedViaInterface``: the scripted ``TacticsController`` (which reads
  truth, fact 1) runs unchanged; its commands are encoded into the 13 outputs
  and decoded back through the real decoder (layer 1 at 0.5 s: identical fight;
  layer 2 at 1 s: commands held between decisions).
- ``HandBlue``: a hand policy written ONLY on the observation array (layer 3);
  ``silent_slots`` keeps those slots' radars off (radar-silent variant).
- ``RandomMLPPolicy``: random-weight 231-64-64-13 MLP (end-to-end smoke test).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np

from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.tactics.interpreter import TacticsController

from .action import (ACTION_SPEC, ALT_MAX_M, ALT_MIN_M, SPD_MAX_MPS, SPD_MIN_MPS,
                     decode_action, encode_commands)
from .controller import NetworkBlueController
from .observation import OBS_SPEC, CONTACT_NAMES, MATE_NAMES, OWN_NAMES


class ScriptedViaInterface(NetworkBlueController):
    def __init__(self, blue_ids: List[str], genome: Optional[TacticsGenome] = None,
                 decision_period_s: float = 0.5, **kw) -> None:
        super().__init__(None, blue_ids, decision_period_s=decision_period_s, **kw)
        self.inner = TacticsController(genome if genome is not None else TacticsGenome(),
                                       blue_ids)
        self.n_encoded = 0
        self.n_inexact = 0
        self.n_fire_miss = 0

    def actions(self, world, jets, obs, infos):
        for ac in jets:
            ac.cmd_fire, ac.fire_target = False, None
        self.inner(world)
        acts = []
        for ac, info in zip(jets, infos):
            fire = bool(ac.cmd_fire and ac.fire_target)
            out, exact = encode_commands(ac.cmd_heading_rad, ac.cmd_alt_m, ac.cmd_speed_mps,
                                         ac.fire_target if fire else None, fire, info,
                                         radar=True, pair=False)
            self.n_encoded += 1
            if not exact:
                self.n_inexact += 1
                if fire and ac.fire_target not in info.keys:
                    self.n_fire_miss += 1
            acts.append(decode_action(out, info, self.action_spec))
        return acts


def _idx(prefix: str, names, name: str) -> int:
    return names.index(name)


class HandBlue:
    """Perceived-only hand policy (callable on the (n, 231) observation batch).

    Mirrors the default genome's intent: fly the ingress axis at cruise until a
    ranged contact is within 55 km (commit), then bracket (+/-35 deg by element)
    the nearest fire-control contact (doctrine-permitted first) at its perceived
    altitude, 273 m/s, fire when shot-ready and range <= 0.75 Rmax; egress south
    at max speed after two Blue losses. ``silent_slots`` (1-based) never emit.
    """

    def __init__(self, silent_slots: Iterable[int] = (), commit_range_m: float = 55000.0,
                 shoot_frac: float = 0.75, bracket_deg: float = 35.0,
                 commit_speed_mps: float = 273.0, cruise_mps: float = 260.0) -> None:
        self.silent = {int(s) for s in silent_slots}
        self.commit_range_m = commit_range_m
        self.shoot_frac = shoot_frac
        self.bracket = bracket_deg / 180.0
        self.commit_speed = commit_speed_mps
        self.cruise = cruise_mps
        self.committed = False
        s = OBS_SPEC
        self.o = {n: i for i, n in enumerate(OWN_NAMES)}
        self.c = {n: i for i, n in enumerate(CONTACT_NAMES)}
        self.w = {n: i for i, n in enumerate(MATE_NAMES)}

    @staticmethod
    def _z(alt: float) -> float:
        a = min(ALT_MAX_M, max(ALT_MIN_M, alt))
        return (a - ALT_MIN_M) / (ALT_MAX_M - ALT_MIN_M) * 2.0 - 1.0

    @staticmethod
    def _s(v: float) -> float:
        v = min(SPD_MAX_MPS, max(SPD_MIN_MPS, v))
        return (v - SPD_MIN_MPS) / (SPD_MAX_MPS - SPD_MIN_MPS) * 2.0 - 1.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        S, A = OBS_SPEC, ACTION_SPEC
        obs = np.atleast_2d(obs)
        out = np.full((obs.shape[0], A.size), -1.0)
        o, c, w = self.o, self.c, self.w
        # commit is flight-wide (the scripted lead commits the package)
        for row in obs:
            for i in range(S.k):
                f = row[S.contact_slice(i)]
                if f[c["present"]] > 0 and f[c["range_valid"]] > 0 \
                        and f[c["range"]] * S.contact_range_m <= self.commit_range_m:
                    self.committed = True
        for n, row in enumerate(obs):
            own = row[S.own_slice]
            slot = int(np.argmax(own[:4])) + 1
            alt = own[o["alt"]] * S.alt_m
            mates_alive = sum(row[S.mate_slice(j)][w["alive"]] for j in range(S.n_mates))
            losses = 3 - int(round(mates_alive))
            out[n, A.radar] = -1.0 if slot in self.silent else 1.0
            out[n, A.pair] = -1.0
            logits = np.full(S.k + 1, -1.0)
            if losses >= 2:                                 # abort: egress south
                logits[S.k] = 1.0
                out[n, A.target_slice] = logits
                out[n, A.heading] = 1.0
                out[n, A.speed] = 1.0
                out[n, A.alt] = self._z(max(alt, 6000.0))
                continue
            cands = []
            for i in range(S.k):
                f = row[S.contact_slice(i)]
                if f[c["present"]] <= 0 or f[c["range_valid"]] <= 0:
                    continue
                fc = f[c["own_fc"]] > 0 or f[c["remote_fc"]] > 0
                cands.append((i, f, fc))
            tgt = None
            if self.committed and cands:
                fcs = [x for x in cands if x[2]]
                ok = [x for x in fcs if x[1][c["doctrine_ok"]] > 0]
                pool = ok or fcs or cands
                tgt = min(pool, key=lambda x: (x[1][c["range"]], x[0]))
            if tgt is None:
                logits[S.k] = 1.0
                out[n, A.target_slice] = logits
                out[n, A.heading] = 0.0
                out[n, A.speed] = self._s(self.cruise)
                out[n, A.alt] = self._z(alt)
                continue
            i, f, _ = tgt
            logits[i] = 1.0
            out[n, A.target_slice] = logits
            side = -1.0 if slot <= 2 else 1.0
            out[n, A.heading] = side * self.bracket
            out[n, A.speed] = self._s(self.commit_speed)
            out[n, A.alt] = self._z(alt + f[c["dalt"]] * S.dalt_m)
            fire = f[c["shot_ready"]] > 0 and f[c["r_rmax"]] * 2.0 <= self.shoot_frac
            out[n, A.fire] = 1.0 if fire else -1.0
        return out


class RandomMLPPolicy:
    """Random-weight MLP 231-64-64-13 (tanh hidden; heading/alt/speed and the
    binary heads tanh, target logits linear). Deterministic per seed."""

    def __init__(self, seed: int = 0, hidden=(64, 64), n_in: int = OBS_SPEC.size,
                 n_out: int = ACTION_SPEC.size) -> None:
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0.0, 1.0 / math.sqrt(a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [rng.normal(0.0, 0.1, b) for b in sizes[1:]]
        self.n_params = sum(W.size + b.size for W, b in zip(self.W, self.b))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(obs)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = np.tanh(x @ W + b)
        y = x @ self.W[-1] + self.b[-1]
        A = ACTION_SPEC
        y[:, :3] = np.tanh(y[:, :3])
        y[:, A.fire:] = np.tanh(y[:, A.fire:])
        return y
