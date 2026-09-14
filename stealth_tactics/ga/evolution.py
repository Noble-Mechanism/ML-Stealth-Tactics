"""Elitist genetic algorithm over TacticsGenome."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import json
from pathlib import Path

import numpy as np

from stealth_tactics.tactics.genome import TacticsGenome, crossover, mutate
from stealth_tactics.sim.world import World, SimConfig, SimResult
from stealth_tactics.sim.aircraft import Aircraft, Coalition
from stealth_tactics.tactics.interpreter import TacticsController, RedCAPController
from stealth_tactics.scenarios.loader import Scenario, build_aircraft


@dataclass
class GAConfig:
    population: int = 20
    generations: int = 10
    elite_count: int = 2
    mutation_rate: float = 0.18
    tournament_k: int = 3
    seed: int = 42
    sim_dt: float = 0.5
    sim_max_time_s: float = 240.0
    # Fitness weights
    w_blue_kills: float = 30.0
    w_blue_losses: float = -40.0
    w_time_win: float = 10.0  # bonus scaled by remaining time fraction if blue wins
    w_ammo: float = 1.0
    w_survival: float = 15.0  # per blue alive


@dataclass
class FitnessResult:
    fitness: float
    sim: SimResult
    genome: TacticsGenome


class GeneticAlgorithm:
    def __init__(
        self,
        scenario: Scenario,
        config: GAConfig,
        on_generation: Optional[Callable[[int, List[FitnessResult]], None]] = None,
    ) -> None:
        self.scenario = scenario
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.on_generation = on_generation
        self.history: List[dict] = []
        self.best: Optional[FitnessResult] = None

    def evaluate(self, genome: TacticsGenome, record: bool = False, seed_offset: int = 0) -> FitnessResult:
        aircraft = build_aircraft(self.scenario)
        blue_ids = [a.id for a in aircraft if a.coalition == Coalition.BLUE]
        red_ids = [a.id for a in aircraft if a.coalition == Coalition.RED]

        blue_ctrl = TacticsController(genome, blue_ids)
        red_ctrl = RedCAPController(red_ids, mode=self.scenario.red_mode)

        sim_cfg = SimConfig(
            dt=self.config.sim_dt,
            max_time_s=self.config.sim_max_time_s,
            seed=self.config.seed + 1000 + seed_offset,
        )
        world = World(aircraft, sim_cfg, blue_ctrl, red_ctrl, record=record)
        result = world.run()
        fitness = self._fitness(result)
        return FitnessResult(fitness=fitness, sim=result, genome=genome)

    def _fitness(self, r: SimResult) -> float:
        c = self.config
        score = 0.0
        score += c.w_blue_kills * r.blue_kills
        score += c.w_blue_losses * r.red_kills  # red_kills == blue losses
        score += c.w_survival * r.blue_alive
        score += c.w_ammo * (r.blue_ammo_remaining * 0.25)
        if r.winner == "blue":
            frac = max(0.0, 1.0 - r.time_s / c.sim_max_time_s)
            score += c.w_time_win * (1.0 + frac)
        elif r.winner == "red":
            score -= 20.0
        return float(score)

    def _tournament(self, pop: List[FitnessResult]) -> TacticsGenome:
        k = min(self.config.tournament_k, len(pop))
        idxs = self.rng.choice(len(pop), size=k, replace=False)
        best = max((pop[i] for i in idxs), key=lambda f: f.fitness)
        return best.genome

    def run(self) -> FitnessResult:
        cfg = self.config
        # Init population
        population: List[TacticsGenome] = [
            TacticsGenome.random(self.rng) for _ in range(cfg.population)
        ]
        # Seed one "sensible default"
        population[0] = TacticsGenome()

        evaluated: List[FitnessResult] = []
        for gen in range(cfg.generations):
            evaluated = [
                self.evaluate(g, record=False, seed_offset=gen * 100 + i)
                for i, g in enumerate(population)
            ]
            evaluated.sort(key=lambda f: f.fitness, reverse=True)

            if self.best is None or evaluated[0].fitness > self.best.fitness:
                self.best = evaluated[0]

            self.history.append({
                "generation": gen,
                "best_fitness": evaluated[0].fitness,
                "mean_fitness": float(np.mean([e.fitness for e in evaluated])),
                "best_kills": evaluated[0].sim.blue_kills,
                "best_losses": evaluated[0].sim.red_kills,
                "best_winner": evaluated[0].sim.winner,
            })

            if self.on_generation:
                self.on_generation(gen, evaluated)

            # Next generation
            elites = [e.genome for e in evaluated[: cfg.elite_count]]
            next_pop: List[TacticsGenome] = list(elites)
            while len(next_pop) < cfg.population:
                p1 = self._tournament(evaluated)
                p2 = self._tournament(evaluated)
                child = crossover(p1, p2, self.rng)
                child = mutate(child, self.rng, rate=cfg.mutation_rate)
                next_pop.append(child)
            population = next_pop

        # Re-evaluate best with recording for ACMI
        assert self.best is not None
        self.best = self.evaluate(self.best.genome, record=True, seed_offset=9999)
        return self.best
