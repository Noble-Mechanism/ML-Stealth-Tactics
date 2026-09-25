"""Fitness scoring: kills first, anti-cowardice, faster wins when engaged."""

from __future__ import annotations

from pathlib import Path

from stealth_tactics.ga.evolution import GeneticAlgorithm, GAConfig
from stealth_tactics.sim.world import SimResult
from stealth_tactics.scenarios.loader import load_scenario


def _ga() -> GeneticAlgorithm:
    root = Path(__file__).resolve().parents[1]
    sc = load_scenario(root / "scenarios" / "default_4v3.yaml")
    return GeneticAlgorithm(sc, GAConfig())


def test_fitness_prefers_kills_over_losses() -> None:
    ga = _ga()
    good = SimResult(
        time_s=100, blue_kills=3, red_kills=0, blue_alive=4, red_alive=0,
        blue_ammo_remaining=8, winner="blue", blue_shots=4,
    )
    bad = SimResult(
        time_s=100, blue_kills=0, red_kills=3, blue_alive=1, red_alive=3,
        blue_ammo_remaining=2, winner="red", blue_shots=0,
    )
    assert ga._fitness(good) > ga._fitness(bad)


def test_fleeing_zero_kills_worse_than_fight_with_kills() -> None:
    """Pristine egress (0 kills, 0 losses) must lose to a fight that kills Red."""
    ga = _ga()
    flee = SimResult(
        time_s=30, blue_kills=0, red_kills=0, blue_alive=4, red_alive=3,
        blue_ammo_remaining=16, winner="draw", blue_shots=0,
    )
    # Messy fight: 2 kills, 1 Blue loss
    fight = SimResult(
        time_s=180, blue_kills=2, red_kills=1, blue_alive=3, red_alive=1,
        blue_ammo_remaining=6, winner="draw", blue_shots=4,
    )
    assert ga._fitness(fight) > ga._fitness(flee)


def test_faster_win_same_kills_scores_better() -> None:
    ga = _ga()
    fast = SimResult(
        time_s=60, blue_kills=3, red_kills=0, blue_alive=4, red_alive=0,
        blue_ammo_remaining=8, winner="blue", blue_shots=5,
    )
    slow = SimResult(
        time_s=220, blue_kills=3, red_kills=0, blue_alive=4, red_alive=0,
        blue_ammo_remaining=8, winner="blue", blue_shots=5,
    )
    assert ga._fitness(fast) > ga._fitness(slow)


def test_time_bonus_not_for_zero_kill_flee() -> None:
    """Early exit with 0 kills must not beat late exit with 0 kills via time bonus."""
    ga = _ga()
    early_flee = SimResult(
        time_s=20, blue_kills=0, red_kills=0, blue_alive=4, red_alive=3,
        blue_ammo_remaining=16, winner="draw", blue_shots=0,
    )
    late_flee = SimResult(
        time_s=200, blue_kills=0, red_kills=0, blue_alive=4, red_alive=3,
        blue_ammo_remaining=16, winner="draw", blue_shots=0,
    )
    # Both get cowardice; early flee must NOT score higher via time bonus
    assert ga._fitness(early_flee) <= ga._fitness(late_flee) + 1e-6
