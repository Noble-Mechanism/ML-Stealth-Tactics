"""Shared command-line knobs for ``evolve-net`` and ``overnight``."""

from __future__ import annotations

import argparse
import os

from .neuroga import NeuroConfig


def add_run_args(p: argparse.ArgumentParser, gens_default) -> None:
    p.add_argument("--pop", type=int, default=50, help="population size (default 50)")
    p.add_argument("--gens", type=int, default=gens_default,
                   help="total generations (overnight: default = run until stopped)")
    p.add_argument("--presentations", type=int, default=24,
                   help="presentations per network per generation (default 24)")
    p.add_argument("--benchmark", type=int, default=64)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--n-red", type=int, default=6, help="Red flight size (6 or 8)")
    p.add_argument("--blue-start", choices=["wall", "diamond"], default="wall",
                   help="Blue start: 30 NM line-abreast wall (default) or the spec 4 diamond")
    p.add_argument("--init", choices=["random", "mixed", "clone"], default="random",
                   help="start population: random (default), mixed = 10 clones + rest random")
    p.add_argument("--clone", default=None, help="clone genome path (without suffix)")
    p.add_argument("--fitness", default=None,
                   help="fitness weights YAML (default scenarios/fitness.yaml)")
    p.add_argument("--fw", action="append", default=[], metavar="KEY=VALUE",
                   help="override one fitness weight, e.g. --fw blue_loss=-200 (repeatable)")
    p.add_argument("--encoder", choices=["flat", "set"], default="flat")
    p.add_argument("--unit-swap", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-time", type=float, default=360.0)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                   help="worker processes (default: all cores)")


def config_from_args(args) -> NeuroConfig:
    from stealth_tactics.fitness import load_weights, parse_overrides
    return NeuroConfig(population=args.pop, presentations=args.presentations,
                       benchmark=args.benchmark, test_size=args.test_size, init=args.init,
                       master_seed=args.seed, max_time_s=args.max_time,
                       unit_swap=args.unit_swap, truncation=min(10, args.pop),
                       n_clones=min(10, args.pop), n_red=args.n_red,
                       blue_start=args.blue_start,
                       fitness=load_weights(args.fitness, parse_overrides(args.fw)),
                       arch={**NeuroConfig().arch, "encoder": args.encoder})


def ensure_clone(args, out) -> str | None:
    """Clone path for mixed / clone starts; trains one into <out>/clone if needed."""
    if args.init not in ("mixed", "clone"):
        return args.clone
    if args.clone:
        return args.clone
    from .clone import clone_hand
    cdir = out / "clone"
    cdir.mkdir(parents=True, exist_ok=True)
    if not (cdir / "clone.json").exists():
        info = clone_hand(cdir / "clone", master_seed=args.seed, workers=args.workers,
                          blue_start=args.blue_start, n_red=args.n_red)
        g = info["gate"]
        print(f"clone gate: {100 * g['kills_ratio']:.0f} % of hand kills "
              f"({'PASS' if g['pass'] else 'FAIL'})")
        if not g["pass"]:
            raise SystemExit("clone failed its quality gate; use --init random or retrain")
    return str(cdir / "clone")
