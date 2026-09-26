"""Spec 6 E/I/J: the network genome and its mutation.

Genome = one flat float64 weight vector per network (``n_networks``; 1 now),
its own mutation size ``sigma`` and a lineage record. The jet -> network map
lives in the config (``NeuroConfig.jet_network``), so per-element (2 nets) or
per-jet (4 nets) genomes later need no format change (Rusty, 2026-09-26: E
addition; multi-network evolution itself is deferred)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .checkpoint import load_npz, save_npz, write_json
from .mlp import DEFAULT_ARCH, Arch, MLPPolicy, interface_fingerprint

SIGMA_INIT = 0.02
SIGMA_MIN, SIGMA_MAX = 0.002, 0.2
TAU = 0.2
SUPPORTED_NETWORK_COUNTS = (1,)


class NetworkCountError(ValueError):
    pass


def check_network_count(n: int) -> None:
    if int(n) not in SUPPORTED_NETWORK_COUNTS:
        raise NetworkCountError(
            f"genome has n_networks={n}; this build supports only 1 shared network "
            f"(spec 6 E). Per-element (2) and per-jet (4) networks are deferred: the "
            f"format carries the count, but multi-network evolution is not implemented.")


@dataclass
class NetGenome:
    nets: List[np.ndarray]                 # one weight vector per network
    sigma: float = SIGMA_INIT
    lineage: dict = field(default_factory=dict)   # id, parent, born, origin

    @property
    def n_networks(self) -> int:
        return len(self.nets)

    @property
    def weights(self) -> np.ndarray:
        return self.nets[0]

    def policy(self, arch: Arch = DEFAULT_ARCH) -> MLPPolicy:
        check_network_count(self.n_networks)
        return MLPPolicy(self.nets[0], arch)

    def copy(self) -> "NetGenome":
        return NetGenome([w.copy() for w in self.nets], float(self.sigma), dict(self.lineage))

    def meta(self) -> dict:
        return {"n_networks": self.n_networks, "sigma": float(self.sigma),
                "lineage": dict(self.lineage)}


def mutate(parent: NetGenome, rng: np.random.Generator, child_id: int, gen: int,
           tau: float = TAU, sigma_min: float = SIGMA_MIN,
           sigma_max: float = SIGMA_MAX) -> NetGenome:
    """I: sigma' = clip(sigma * exp(tau N(0,1))), then every weight + sigma' N(0,1)."""
    s = float(np.clip(parent.sigma * np.exp(tau * rng.standard_normal()), sigma_min, sigma_max))
    nets = [w + s * rng.standard_normal(w.shape) for w in parent.nets]
    lin = {"id": int(child_id), "parent": parent.lineage.get("id"), "born": int(gen),
           "origin": parent.lineage.get("origin", "random")}
    return NetGenome(nets, s, lin)


def unit_swap(a: NetGenome, b: NetGenome, rng: np.random.Generator, arch: Arch,
              rate: float = 0.5) -> NetGenome:
    """J (behind a switch, off by default): swap whole first-hidden-layer units
    (in-weights column, bias, out-weights row) from *b* into a copy of *a*."""
    if arch.encoder != "flat":
        raise ValueError("unit swap is defined for the flat MLP only")
    from .mlp import flatten, unflatten
    child = a.copy()
    La, Lb = unflatten(child.nets[0].copy(), arch), unflatten(b.nets[0], arch)
    units = np.where(rng.random(arch.hidden[0]) < rate)[0]
    La["W0"][:, units] = Lb["W0"][:, units]
    La["b0"][units] = Lb["b0"][units]
    La["W1"][units, :] = Lb["W1"][units, :]
    child.nets[0] = flatten(La, arch)
    return child


def save_genome(path, g: NetGenome, arch: Arch = DEFAULT_ARCH, extra: Optional[dict] = None) -> None:
    """``<path>.npz`` weights + ``<path>.json`` meta (arch, fingerprint, count)."""
    path = Path(path)
    save_npz(path.with_suffix(".npz"), {f"net{i}": w for i, w in enumerate(g.nets)})
    meta = {"kind": "network", "arch": arch.to_dict(), "interface": interface_fingerprint(),
            **g.meta(), **(extra or {})}
    write_json(path.with_suffix(".json"), meta)


def load_genome(path, check_interface: bool = True):
    """Returns (NetGenome, Arch, meta). Refuses network counts other than 1
    and interface-fingerprint mismatches with a clear message."""
    import json
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    check_network_count(meta.get("n_networks", 1))
    if check_interface and meta.get("interface") != interface_fingerprint():
        raise ValueError(f"{path}: interface fingerprint {meta.get('interface')} != current "
                         f"{interface_fingerprint()} (spec 5 interface changed; refusing)")
    arrays = load_npz(path.with_suffix(".npz"))
    nets = [arrays[f"net{i}"] for i in range(meta["n_networks"])]
    arch = Arch.from_dict(meta["arch"])
    return NetGenome(nets, meta["sigma"], meta.get("lineage", {})), arch, meta
