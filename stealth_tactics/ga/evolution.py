"""Elitist genetic algorithm over TacticsGenome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable
import json
from pathlib import Path

import numpy as np

from stealth_tactics.tactics.genome import TacticsGenome, crossover, mutate
from stealth_tactics.sim.world import World, SimConfig, SimResult, SHOOT_ASSESS_SHOOT
from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.tactics.interpreter import TacticsController, RedCAPController
from stealth_tactics.tactics.red_defense import RedDefense, DefenseConfig
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
    # Fitness weights — kills first, anti-cowardice, then time / losses
    w_blue_kills: float = 100.0
    w_blue_losses: float = -75.0
    w_survival: float = 10.0  # per blue alive
    w_shot: float = 3.0  # per Blue launch (capped) — reward engaging
    w_time: float = 40.0  # time bonus *only* when blue_kills > 0
    w_blue_win: float = 25.0
    w_red_win: float = -40.0
    w_coward: float = -50.0  # flat penalty for 0 kills
    w_no_shots: float = -30.0  # extra if also took no shots
    # Spec 3 (None -> scenario value -> default): Red defense on/off, Red
    # aggressiveness, firing doctrines ("shoot_assess_shoot" |
    # "shoot_shoot_assess" | "legacy")
    red_defense: Optional[bool] = None
    red_aggressiveness: Optional[float] = None
    blue_doctrine: Optional[str] = None
    red_doctrine: Optional[str] = None
    defense_config: Optional[DefenseConfig] = None


@dataclass
class FitnessResult:
    fitness: float
    sim: SimResult
    genome: TacticsGenome
    seed_offset: int = 0


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
        # Fitness of the champion as scored during evolution (before ACMI re-run)
        self.evolution_best_fitness: Optional[float] = None

    def evaluate(self, genome: TacticsGenome, record: bool = False, seed_offset: int = 0) -> FitnessResult:
        aircraft = build_aircraft(self.scenario)
        blue_ids = [a.id for a in aircraft if a.coalition == Coalition.BLUE]
        red_ids = [a.id for a in aircraft if a.coalition == Coalition.RED]

        c, sc = self.config, self.scenario
        blue_ctrl = TacticsController(genome, blue_ids)
        defense_on = c.red_defense if c.red_defense is not None else sc.red_defense
        a = c.red_aggressiveness if c.red_aggressiveness is not None else sc.red_aggressiveness
        defense = RedDefense(a, c.defense_config) if defense_on else None
        red_ctrl = RedCAPController(red_ids, mode=sc.red_mode, defense=defense)

        sim_cfg = SimConfig(
            dt=self.config.sim_dt,
            max_time_s=self.config.sim_max_time_s,
            seed=self.config.seed + 1000 + seed_offset,
            blue_doctrine=c.blue_doctrine or sc.blue_doctrine or SHOOT_ASSESS_SHOOT,
            red_doctrine=c.red_doctrine or sc.red_doctrine or SHOOT_ASSESS_SHOOT,
        )
        world = World(aircraft, sim_cfg, blue_ctrl, red_ctrl, record=record)
        result = world.run()
        fitness = self._fitness(result)
        return FitnessResult(
            fitness=fitness, sim=result, genome=genome, seed_offset=seed_offset
        )

    def _fitness(self, r: SimResult) -> float:
        """
        Priorities: (1) kills, (2) finish faster *when engaged*, (3) limit
        Blue losses, (4) crush the flee-with-0-kills local minimum.

        score =
            100×BlueKills − 75×BlueLosses + 10×BlueAlive
          + 3×min(BlueShots, 6)
          + [if BlueKills > 0]:
                40×(1 + time_remaining_frac)
              + (Blue win ? 25 : 0)
            else:
                −50   (cowardice)
              + (BlueShots == 0 ? −30 : 0)
          + (Red win ? −40 : 0)

        Fleeing with 0 kills cannot beat a fight that scores kills even with
        some Blue losses. Time bonus applies only when kills > 0.
        """
        c = self.config
        score = 0.0
        score += c.w_blue_kills * r.blue_kills
        score += c.w_blue_losses * r.red_kills  # red_kills == blue losses
        score += c.w_survival * r.blue_alive
        shots = getattr(r, "blue_shots", 0) or 0
        score += c.w_shot * min(shots, 6)

        if r.blue_kills > 0:
            frac = max(0.0, 1.0 - r.time_s / c.sim_max_time_s)
            score += c.w_time * (1.0 + frac)
            if r.winner == "blue":
                score += c.w_blue_win
        else:
            score += c.w_coward
            if shots == 0:
                score += c.w_no_shots

        if r.winner == "red":
            score += c.w_red_win
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
                champ = evaluated[0]
                # Deep-copy genome so later elite reuse / crossover cannot mutate champion
                self.best = FitnessResult(
                    fitness=champ.fitness,
                    sim=champ.sim,
                    genome=champ.genome.copy(),
                    seed_offset=champ.seed_offset,
                )
                self.evolution_best_fitness = champ.fitness

            self.history.append({
                "generation": gen,
                "best_fitness": evaluated[0].fitness,
                "mean_fitness": float(np.mean([e.fitness for e in evaluated])),
                "best_kills": evaluated[0].sim.blue_kills,
                "best_losses": evaluated[0].sim.red_kills,
                "best_winner": evaluated[0].sim.winner,
                "best_shots": getattr(evaluated[0].sim, "blue_shots", 0),
                "best_seed_offset": evaluated[0].seed_offset,
            })

            if self.on_generation:
                self.on_generation(gen, evaluated)

            # Next generation — copy elites so mutation of shared refs cannot touch them
            elites = [e.genome.copy() for e in evaluated[: cfg.elite_count]]
            next_pop: List[TacticsGenome] = list(elites)
            while len(next_pop) < cfg.population:
                p1 = self._tournament(evaluated)
                p2 = self._tournament(evaluated)
                child = crossover(p1, p2, self.rng)
                child = mutate(child, self.rng, rate=cfg.mutation_rate)
                next_pop.append(child)
            population = next_pop

        # Re-evaluate best with recording for ACMI using the SAME seed_offset
        assert self.best is not None
        recorded = self.evaluate(
            self.best.genome, record=True, seed_offset=self.best.seed_offset
        )
        # Same seed + config ⇒ deterministic; fitness must match the champion
        if abs(recorded.fitness - self.best.fitness) > 1e-6:
            raise RuntimeError(
                f"Recorded ACMI re-run fitness {recorded.fitness:.6f} != "
                f"evolution champion {self.best.fitness:.6f} "
                f"(seed_offset={self.best.seed_offset})"
            )
        self.best = recorded
        return self.best
