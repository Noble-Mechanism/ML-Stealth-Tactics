"""Simulation and genome smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stealth_tactics.scenarios.loader import load_scenario, build_aircraft
from stealth_tactics.sim.world import World, SimConfig
from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.tactics.genome import TacticsGenome, crossover, mutate
from stealth_tactics.tactics.interpreter import TacticsController, RedCAPController
from stealth_tactics.ga.evolution import GeneticAlgorithm, GAConfig


ROOT = Path(__file__).resolve().parents[1]


def test_scenario_builds_4v3() -> None:
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ac = build_aircraft(sc)
    assert sum(1 for a in ac if a.coalition == Coalition.BLUE) == 4
    assert sum(1 for a in ac if a.coalition == Coalition.RED) == 3


def test_short_sim_runs() -> None:
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    aircraft = build_aircraft(sc)
    blue_ids = [a.id for a in aircraft if a.coalition == Coalition.BLUE]
    red_ids = [a.id for a in aircraft if a.coalition == Coalition.RED]
    world = World(
        aircraft,
        SimConfig(dt=1.0, max_time_s=30.0, seed=1),
        TacticsController(TacticsGenome(), blue_ids),
        RedCAPController(red_ids, mode="intercept"),
        record=True,
    )
    result = world.run()
    assert result.time_s > 0
    assert result.blue_alive + result.red_kills == 4
    assert len(result.frames) > 1


def test_genome_crossover_mutate_seedable() -> None:
    rng = np.random.default_rng(123)
    a = TacticsGenome.random(rng)
    b = TacticsGenome.random(rng)
    child = crossover(a, b, rng)
    mut = mutate(child, rng, rate=1.0)
    assert isinstance(mut.commit_range_m, float)
    d = mut.to_dict()
    restored = TacticsGenome.from_dict(d)
    assert restored.merge_geometry == mut.merge_geometry


def test_ga_one_generation() -> None:
    sc = load_scenario(ROOT / "scenarios" / "default_4v3.yaml")
    ga = GeneticAlgorithm(
        sc,
        GAConfig(population=4, generations=1, seed=99, sim_max_time_s=40.0, sim_dt=1.0),
    )
    best = ga.run()
    assert best.fitness == best.fitness  # not NaN
    assert len(best.sim.frames) > 0
