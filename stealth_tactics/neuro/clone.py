"""Spec 6 K: behaviour cloning of ``HandBlue`` as a warm start.

1. Run ``HandBlue`` (spec 5 layer 3, observation only) through the spec 5
   controller on ``n_pres`` presentations from their own seed set (never the
   benchmark or held-out test sets) and log every (observation, action) pair
   at the 1 s decisions.
2. Fit the MLP in numpy with Adam: MSE on heading / altitude / speed in the
   tanh domain, cross-entropy on the 7 target logits (absent slots masked),
   MSE on the fire / radar / pair bits against +/-1 (fire positives weighted).
3. Gate: the clone must reach >= 80 % of HandBlue's mean kills on ``n_gate``
   held-out presentations (another seed set)."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from stealth_tactics.policy.action import ACTION_SPEC
from stealth_tactics.policy.adapters import HandBlue
from stealth_tactics.policy.controller import NetworkBlueController
from stealth_tactics.policy.observation import OBS_SPEC

from .evaluate import make_pool, pmap, policy_job
from .genome import NetGenome, save_genome
from .mlp import DEFAULT_ARCH, Arch, MLPPolicy, flatten, init_weights, unflatten

CLONE_SALT, GATE_SALT = 0xC10E, 0x6A7E


class _Logger(NetworkBlueController):
    def __init__(self, policy, ids, sink) -> None:
        super().__init__(policy, ids)
        self.sink = sink

    def actions(self, world, jets, obs, infos):
        out = np.asarray(self.policy(obs), dtype=np.float64)
        self.sink.append((obs.copy(), out.copy()))
        from stealth_tactics.policy.action import decode_action
        return [decode_action(out[i], infos[i], self.action_spec) for i in range(len(jets))]


def _collect_job(args):
    pdict, max_time_s = args
    from dataclasses import replace
    from stealth_tactics.presentation_runner import run_presentation
    from stealth_tactics.scenarios.presentation import DEFAULT_PRESENTATION_CONFIG
    sink: list = []
    run_presentation(pdict, cfg=replace(DEFAULT_PRESENTATION_CONFIG, max_time_s=max_time_s),
                     blue_factory=lambda ids: _Logger(HandBlue(), ids, sink))
    if not sink:
        return np.zeros((0, OBS_SPEC.size)), np.zeros((0, ACTION_SPEC.size))
    return np.vstack([o for o, _ in sink]), np.vstack([a for _, a in sink])


def presentations(master_seed: int, salt: int, n: int, pcfg=None) -> List[dict]:
    from stealth_tactics.scenarios.presentation import _set, DEFAULT_PRESENTATION_CONFIG
    return [p.to_dict() for p in _set(master_seed, salt, 0, n,
                                      pcfg or DEFAULT_PRESENTATION_CONFIG)]


def collect(master_seed: int, n_pres: int, pool, max_time_s: float = 360.0, pcfg=None):
    out = pmap(pool, _collect_job, [(p, max_time_s) for p in
                                    presentations(master_seed, CLONE_SALT, n_pres, pcfg)])
    return np.vstack([o for o, _ in out]), np.vstack([a for _, a in out])


# ------------------------------------------------------------------ fit ----
def _forward(L, x, n_hidden):
    hs = [x]
    for i in range(n_hidden):
        hs.append(np.tanh(hs[-1] @ L[f"W{i}"] + L[f"b{i}"]))
    return hs, hs[-1] @ L[f"W{n_hidden}"] + L[f"b{n_hidden}"]


def fit(obs: np.ndarray, act: np.ndarray, arch: Arch = DEFAULT_ARCH, seed: int = 0,
        epochs: int = 60, batch: int = 256, lr: float = 1e-3, fire_pos_weight: float = 10.0,
        log=None) -> Tuple[np.ndarray, List[float]]:
    """Adam on the flat MLP (encoder must be "flat")."""
    if arch.encoder != "flat":
        raise ValueError("clone fitting implemented for the flat MLP")
    A, S = ACTION_SPEC, OBS_SPEC
    rng = np.random.default_rng(np.random.SeedSequence([seed, CLONE_SALT]))
    w = init_weights(arch, rng)
    L = unflatten(w.copy(), arch)
    names = [n for n, _ in arch.shapes()]
    m = {n: np.zeros_like(L[n]) for n in names}
    v = {n: np.zeros_like(L[n]) for n in names}
    nh = len(arch.hidden)
    tgt = np.argmax(act[:, A.target_slice], axis=1)
    present = np.concatenate([obs[:, [S.contact_slice(i).start for i in range(S.k)]] > 0.5,
                              np.ones((len(obs), 1), bool)], axis=1)
    fire_w = np.where(act[:, A.fire] > 0, fire_pos_weight, 1.0)
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    losses = []
    n = len(obs)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            x, y, tg, pm, fw = obs[idx], act[idx], tgt[idx], present[idx], fire_w[idx]
            B = len(idx)
            hs, z = _forward(L, x, nh)
            dz = np.zeros_like(z)
            # continuous + bits: MSE in tanh domain
            cols = [0, 1, 2, A.fire, A.radar, A.pair]
            th = np.tanh(z[:, cols])
            wcol = np.ones((B, len(cols)))
            wcol[:, 3] = fw
            diff = th - y[:, cols]
            loss = float((wcol * diff ** 2).sum() / B)
            dz[:, cols] = 2 * wcol * diff * (1 - th ** 2) / B
            # target logits: masked softmax cross-entropy
            lg = np.where(pm, z[:, A.target_slice], -1e9)
            lg = lg - lg.max(1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(1, keepdims=True)
            loss += float(-np.log(p[np.arange(B), tg] + 1e-12).sum() / B)
            g = p.copy()
            g[np.arange(B), tg] -= 1.0
            dz[:, A.target_slice] = np.where(pm, g, 0.0) / B
            tot += loss * B
            # backprop
            grads = {}
            d = dz
            for i in range(nh, -1, -1):
                grads[f"W{i}"] = hs[i].T @ d
                grads[f"b{i}"] = d.sum(0)
                if i > 0:
                    d = (d @ L[f"W{i}"].T) * (1 - hs[i] ** 2)
            step += 1
            for nme in names:
                m[nme] = b1 * m[nme] + (1 - b1) * grads[nme]
                v[nme] = b2 * v[nme] + (1 - b2) * grads[nme] ** 2
                mh = m[nme] / (1 - b1 ** step)
                vh = v[nme] / (1 - b2 ** step)
                L[nme] -= lr * mh / (np.sqrt(vh) + eps)
        losses.append(tot / n)
        if log:
            log(f"  clone epoch {ep + 1:2d}/{epochs}: loss {tot / n:.4f}")
    return flatten(L, arch), losses


def gate(weights: np.ndarray, arch: Arch, master_seed: int, n_gate: int, pool,
         max_time_s: float = 360.0, pcfg=None) -> dict:
    """Clone vs HandBlue on held-out presentations: mean kills, losses, shots."""
    pres = presentations(master_seed, GATE_SALT, n_gate, pcfg)
    res = {}
    for name, pol in (("hand", HandBlue()), ("clone", MLPPolicy(weights, arch))):
        out = pmap(pool, policy_job, [(pol, p, max_time_s, False) for p in pres])
        s = [x[2] for x in out]
        res[name] = {"fitness": float(np.mean([x[0] for x in out])),
                     "kills": float(np.mean([x["kills"] for x in s])),
                     "losses": float(np.mean([x["losses"] for x in s])),
                     "shots": float(np.mean([x["blue_shots"] for x in s]))}
    res["kills_ratio"] = res["clone"]["kills"] / res["hand"]["kills"] if res["hand"]["kills"] else 0.0
    res["pass"] = res["kills_ratio"] >= 0.8
    res["n_gate"] = n_gate
    return res


def clone_hand(out_path, master_seed: int = 2026, n_pres: int = 200, n_gate: int = 100,
               epochs: int = 60, workers: int = 8, arch: Arch = DEFAULT_ARCH,
               max_time_s: float = 360.0, log=print, blue_start: str = "wall",
               n_red: int = 6) -> dict:
    from stealth_tactics.scenarios.presentation import presentation_config
    pcfg = presentation_config(blue_start, n_red, max_time_s)
    t0 = time.perf_counter()
    pool = make_pool(workers)
    try:
        obs, act = collect(master_seed, n_pres, pool, max_time_s, pcfg)
        log(f"clone data: {len(obs)} (observation, action) pairs from {n_pres} presentations "
            f"({time.perf_counter() - t0:.0f} s)")
        t1 = time.perf_counter()
        w, losses = fit(obs, act, arch, seed=master_seed, epochs=epochs, log=log)
        t_fit = time.perf_counter() - t1
        g = gate(w, arch, master_seed, n_gate, pool, max_time_s, pcfg) if n_gate else {}
    finally:
        if pool is not None:
            pool.shutdown()
    genome = NetGenome([w], 0.02, {"id": -1, "parent": None, "born": 0, "origin": "clone"})
    info = {"n_samples": int(len(obs)), "n_pres": n_pres, "epochs": epochs,
            "final_loss": losses[-1], "losses": losses, "fit_s": t_fit, "gate": g,
            "master_seed": master_seed, "blue_start": blue_start, "n_red": n_red,
            "fire_share": float((act[:, ACTION_SPEC.fire] > 0).mean()),
            "total_s": time.perf_counter() - t0}
    save_genome(out_path, genome, arch, extra={"clone": info})
    return info
