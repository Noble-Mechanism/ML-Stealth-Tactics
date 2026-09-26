"""Spec 5 N: the network Blue controller.

Every ``decision_period_s`` (1 s; aligned to t = 0, 1, 2 ...) it builds the
observations of all alive Blue jets from ``World.blue_view`` into one (n, 231)
batch, calls ``policy(batch) -> (n, 13)``, decodes and holds the commands until
the next decision (the fire request is re-asserted on every sim step). Radar
switches respect a 5 s minimum dwell (L); the pair bit changes the jet's
doctrine only while no shoot-shoot-assess pair is open (K). All launch gates
stay in the World (O).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.sim.world import SHOOT_ASSESS_SHOOT, SHOOT_SHOOT_ASSESS, LEGACY

from .action import ACTION_SPEC, Action, ActionSpec, decode_action
from .observation import OBS_SPEC, ObsSpec, build_observation, ObsInfo


@dataclass
class JetMemory:
    target: Optional[str] = None
    action: Optional[Action] = None
    radar_switch_t: float = -1e9
    radar_denied: int = 0


class NetworkBlueController:
    def __init__(self, policy: Optional[Callable], blue_ids: List[str],
                 decision_period_s: float = 1.0, radar_dwell_s: float = 5.0,
                 obs_spec: ObsSpec = OBS_SPEC, action_spec: ActionSpec = ACTION_SPEC,
                 log_radar: bool = True) -> None:
        self.policy = policy
        self.blue_ids = list(blue_ids)
        self.period = float(decision_period_s)
        self.dwell = float(radar_dwell_s)
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.log_radar = log_radar
        self.mem: Dict[str, JetMemory] = {b: JetMemory() for b in self.blue_ids}
        self._next_t = 0.0
        self.n_decisions = 0
        self.last_obs: Optional[np.ndarray] = None
        self.last_ids: List[str] = []

    # ----------------------------------------------------------------------
    def observe(self, world, jets) -> tuple:
        targets = {b: self.mem[b].target for b in self.blue_ids}
        rows, infos = [], []
        for ac in jets:
            ob, info = build_observation(world.blue_view(ac), self.mem[ac.id].target,
                                         targets, self.obs_spec)
            rows.append(ob)
            infos.append(info)
        return np.array(rows).reshape(len(rows), self.obs_spec.size), infos

    def actions(self, world, jets, obs: np.ndarray, infos: List[ObsInfo]) -> List[Action]:
        out = np.asarray(self.policy(obs), dtype=np.float64).reshape(len(jets), -1)
        return [decode_action(out[i], infos[i], self.action_spec) for i in range(len(jets))]

    def _apply_decision(self, world, ac, a: Action) -> None:
        m = self.mem[ac.id]
        m.target = a.target_key
        m.action = a
        t = world.time_s
        if a.radar != ac.radar_emitting:
            if t - m.radar_switch_t >= self.dwell - 1e-9:
                ac.radar_emitting = a.radar
                m.radar_switch_t = t
                if self.log_radar:
                    world.log_event({"t": t, "type": "radar", "observer": ac.id,
                                     "target": ac.id,
                                     "text": f"{ac.name}: RADAR {'ON' if a.radar else 'OFF (silent)'}"})
            else:
                m.radar_denied += 1
        if world.doctrine_of(ac) != LEGACY and not world.ssa_pair_open(ac):
            ac.firing_doctrine = SHOOT_SHOOT_ASSESS if a.pair else SHOOT_ASSESS_SHOOT

    def __call__(self, world) -> None:
        jets = [world.get(b) for b in self.blue_ids]
        jets = [j for j in jets if j is not None and j.state.alive]
        if not jets:
            return
        t = world.time_s
        if t >= self._next_t - 1e-9:
            while self._next_t <= t + 1e-9:
                self._next_t += self.period
            obs, infos = self.observe(world, jets)
            self.last_obs, self.last_ids = obs, [j.id for j in jets]
            acts = self.actions(world, jets, obs, infos)
            for ac, a in zip(jets, acts):
                self._apply_decision(world, ac, a)
            self.n_decisions += 1
        for ac in jets:
            a = self.mem[ac.id].action
            if a is None:
                continue
            ac.cmd_heading_rad = a.heading_rad
            ac.cmd_alt_m = a.alt_m
            ac.cmd_speed_mps = a.speed_mps
            if a.fire and a.target_key is not None:
                ac.cmd_fire = True
                ac.fire_target = a.target_key
            else:
                ac.cmd_fire = False
                ac.fire_target = None
