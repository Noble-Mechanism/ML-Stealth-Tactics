"""Command-line interface: python -m stealth_tactics ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stealth_tactics.scenarios.loader import load_scenario
from stealth_tactics.ga.evolution import GeneticAlgorithm, GAConfig
from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.tactics.genome import TacticsGenome


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_evolve(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    if not scenario_path.is_file():
        # try relative to project scenarios/
        alt = _project_root() / "scenarios" / args.scenario
        if alt.is_file():
            scenario_path = alt
        else:
            print(f"Scenario not found: {args.scenario}", file=sys.stderr)
            return 1

    scenario = load_scenario(scenario_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_gen(gen: int, evaluated: list) -> None:
        best = evaluated[0]
        mean = sum(e.fitness for e in evaluated) / len(evaluated)
        print(
            f"Gen {gen:3d}  best={best.fitness:7.2f}  mean={mean:7.2f}  "
            f"kills={best.sim.blue_kills}  losses={best.sim.red_kills}  "
            f"winner={best.sim.winner}"
        )

    ga_cfg = GAConfig(
        population=args.pop,
        generations=args.gens,
        seed=args.seed,
        sim_dt=args.dt,
        sim_max_time_s=args.max_time,
        elite_count=max(1, args.pop // 10),
    )
    ga = GeneticAlgorithm(scenario, ga_cfg, on_generation=on_gen)
    print(f"Evolving tactics: pop={ga_cfg.population} gens={ga_cfg.generations} "
          f"scenario={scenario.name!r} seed={ga_cfg.seed}")
    best = ga.run()

    # Save genome + history
    genome_path = out_dir / "best_genome.json"
    genome_path.write_text(json.dumps(best.genome.to_dict(), indent=2), encoding="utf-8")
    hist_path = out_dir / "history.json"
    hist_path.write_text(json.dumps(ga.history, indent=2), encoding="utf-8")

    # ACMI
    acmi_name = args.acmi or "best_engagement.txt.acmi"
    acmi_path = out_dir / acmi_name
    exporter = ACMIExporter(title=f"{scenario.name} — best genome")
    exporter.export(best.sim.frames, acmi_path)

    print(f"\nBest fitness: {best.fitness:.2f}")
    print(f"  winner={best.sim.winner}  blue_kills={best.sim.blue_kills}  "
          f"blue_losses={best.sim.red_kills}  time={best.sim.time_s:.1f}s")
    print(f"  genome -> {genome_path}")
    print(f"  ACMI   -> {acmi_path}")
    print(f"  Open in TacView: File → Open → {acmi_path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run a single engagement with default or provided genome."""
    scenario_path = Path(args.scenario)
    if not scenario_path.is_file():
        alt = _project_root() / "scenarios" / args.scenario
        scenario_path = alt if alt.is_file() else scenario_path
    scenario = load_scenario(scenario_path)

    if args.genome:
        genome = TacticsGenome.from_dict(json.loads(Path(args.genome).read_text()))
    else:
        genome = TacticsGenome()

    ga = GeneticAlgorithm(scenario, GAConfig(seed=args.seed, sim_max_time_s=args.max_time))
    result = ga.evaluate(genome, record=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    acmi_path = out_dir / (args.acmi or "sim.txt.acmi")
    ACMIExporter(title=scenario.name).export(result.sim.frames, acmi_path)
    print(f"fitness={result.fitness:.2f} winner={result.sim.winner} "
          f"kills={result.sim.blue_kills} losses={result.sim.red_kills}")
    print(f"ACMI -> {acmi_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stealth_tactics",
        description="GA-evolved stealth fighter tactics simulation with TacView ACMI export",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ev = sub.add_parser("evolve", help="Run genetic algorithm evolution")
    p_ev.add_argument("-s", "--scenario", default="default_4v3.yaml")
    p_ev.add_argument("-p", "--pop", type=int, default=16)
    p_ev.add_argument("-g", "--gens", type=int, default=8)
    p_ev.add_argument("--seed", type=int, default=42)
    p_ev.add_argument("--dt", type=float, default=0.5)
    p_ev.add_argument("--max-time", type=float, default=240.0)
    p_ev.add_argument("-o", "--out", default=None,
                      help="Output directory (default: runs/demo)")
    p_ev.add_argument("--acmi", default="best_engagement.txt.acmi")
    p_ev.set_defaults(func=cmd_evolve)

    p_sim = sub.add_parser("simulate", help="Single engagement simulation")
    p_sim.add_argument("-s", "--scenario", default="default_4v3.yaml")
    p_sim.add_argument("--genome", default=None)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--max-time", type=float, default=240.0)
    p_sim.add_argument("-o", "--out", default=None)
    p_sim.add_argument("--acmi", default="sim.txt.acmi")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    if args.out is None:
        root = _project_root()
        args.out = str(root / ("runs/demo" if args.command == "evolve" else "artifacts"))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
