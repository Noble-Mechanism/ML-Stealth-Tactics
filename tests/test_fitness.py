"""Fitness scoring sanity."""

from __future__ import annotations

from stealth_tactics.ga.evolution import GeneticAlgorithm, GAConfig
from stealth_tactics.sim.world import SimResult
from stealth_tactics.scenarios.loader import load_scenario
from pathlib import Path


def test_fitness_prefers_kills_over_losses() -> None:
    root = Path(__file__).resolve().parents[1]
    sc = load_scenario(root / "scenarios" / "default_4v3.yaml")
    ga = GeneticAlgorithm(sc, GAConfig())

    good = SimResult(
        time_s=100, blue_kills=3, red_kills=0, blue_alive=4, red_alive=0,
        blue_ammo_remaining=8, winner="blue",
    )
    bad = SimResult(
        time_s=100, blue_kills=0, red_kills=3, blue_alive=1, red_alive=3,
        blue_ammo_remaining=2, winner="red",
    )
    assert ga._fitness(good) > ga._fitness(bad)
