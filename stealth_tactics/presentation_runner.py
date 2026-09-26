"""Spec 4: run one engagement from a stored presentation.

``run_presentation(p, genome)`` builds Blue from the presentation's Blue block
and Red from its drawn values, uses ``p.sim_seed`` for every noise stream, the
360 s cap and the spec 4 early-end rules (K, K2) and the presentation Red
controller (E, H-J, L). Same presentation + same genome -> identical result,
whatever process, order or genome index it runs under (decision M).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.scenarios.presentation import (
    DEFAULT_PRESENTATION_CONFIG, Presentation, PresentationConfig,
    build_presentation_aircraft)
from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.sim.world import SHOOT_ASSESS_SHOOT, SimConfig, SimResult, World
from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.tactics.interpreter import TacticsController
from stealth_tactics.tactics.preplanned import PresentationRedController
from stealth_tactics.tactics.red_defense import DefenseConfig, RedDefense


def presentation_sim_config(p: Presentation, cfg: PresentationConfig,
                            blue_doctrine: str = SHOOT_ASSESS_SHOOT) -> SimConfig:
    return SimConfig(dt=cfg.dt, max_time_s=cfg.max_time_s, seed=int(p.sim_seed),
                     blue_doctrine=blue_doctrine, red_doctrine=p.doctrine,
                     early_end_on_departure=True, early_end_winchester=True,
                     finish_missiles_after_wipeout=True)


def run_presentation(p: Union[Presentation, dict], genome: Optional[TacticsGenome] = None,
                     cfg: PresentationConfig = DEFAULT_PRESENTATION_CONFIG,
                     record: bool = False, blue_test_defense: bool = False,
                     blue_doctrine: str = SHOOT_ASSESS_SHOOT,
                     defense_config: Optional[DefenseConfig] = None,
                     blue_factory: Optional[Callable] = None,
                     prepare: Optional[Callable] = None) -> SimResult:
    """``prepare(aircraft, defense)`` (tests only) may adjust the start state."""
    if isinstance(p, dict):
        p = Presentation.from_dict(p)
    aircraft = build_presentation_aircraft(p)
    blue_ids = [a.id for a in aircraft if a.coalition == Coalition.BLUE]
    if blue_factory is not None:
        blue = blue_factory(blue_ids)
    else:
        blue = TacticsController(genome if genome is not None else TacticsGenome(), blue_ids)
    if blue_test_defense:
        from stealth_tactics.analysis.blue_test_defense import BlueTestDefense
        blue = BlueTestDefense(blue)
    defense = RedDefense(p.aggressiveness, defense_config)
    if prepare is not None:
        prepare(aircraft, defense)
    red = PresentationRedController(p, defense=defense, cfg=cfg)
    world = World(aircraft, presentation_sim_config(p, cfg, blue_doctrine), blue, red,
                  record=record)
    res = world.run()
    res.presentation = p.to_dict()
    res.preplanned = red.report()
    return res


def export_presentation_acmi(res: SimResult, path, title: str = "") -> None:
    p = Presentation.from_dict(res.presentation)
    ACMIExporter(title=title or f"Spec 4 presentation {p.presentation_seed}",
                 comments=p.summary()).export(res.frames, path)
