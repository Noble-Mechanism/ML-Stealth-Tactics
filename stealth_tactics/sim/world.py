"""Discrete-time air combat world loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import numpy as np

from .aircraft import Aircraft, Coalition, integrate_aircraft, AircraftState
from .sensors import SensorModel
from .weapons import WeaponModel, Missile


@dataclass
class SimConfig:
    dt: float = 0.5  # seconds — coarse for GA speed
    max_time_s: float = 300.0
    seed: int = 42


@dataclass
class SimResult:
    time_s: float
    blue_kills: int
    red_kills: int  # = blue losses
    blue_alive: int
    red_alive: int
    blue_ammo_remaining: int
    winner: str  # "blue", "red", "draw"
    events: List[dict] = field(default_factory=list)
    # Snapshot history for ACMI: list of (t, states dict)
    frames: List[dict] = field(default_factory=list)


class World:
    """Fast point-mass engagement simulation."""

    def __init__(
        self,
        aircraft: List[Aircraft],
        config: SimConfig,
        blue_controller: Callable[[World], None],
        red_controller: Callable[[World], None],
        record: bool = True,
    ) -> None:
        self.aircraft = aircraft
        self.config = config
        self.blue_controller = blue_controller
        self.red_controller = red_controller
        self.record = record
        self.rng = np.random.default_rng(config.seed)
        self.sensors = SensorModel(self.rng)
        self.weapons = WeaponModel(self.rng)
        self.missiles: List[Missile] = []
        self.time_s = 0.0
        self.events: List[dict] = []
        self.frames: List[dict] = []
        self._by_id: Dict[str, Aircraft] = {a.id: a for a in aircraft}
        self.tracks: Dict[str, set] = {}

    def get(self, uid: str) -> Optional[Aircraft]:
        return self._by_id.get(uid)

    def alive(self, coalition: Optional[Coalition] = None) -> List[Aircraft]:
        out = [a for a in self.aircraft if a.state.alive]
        if coalition is not None:
            out = [a for a in out if a.coalition == coalition]
        return out

    def run(self) -> SimResult:
        self._record_frame()
        dt = self.config.dt
        while self.time_s < self.config.max_time_s:
            if not self.alive(Coalition.BLUE) or not self.alive(Coalition.RED):
                break

            self.tracks = self.sensors.update_tracks(self.aircraft)

            # Controllers set cmd_* and cmd_fire
            self.blue_controller(self)
            self.red_controller(self)

            # Process fire commands
            for ac in self.aircraft:
                if ac.cmd_fire and ac.fire_target and ac.state.alive:
                    tgt = self._by_id.get(ac.fire_target)
                    if tgt and self.sensors.can_lock(ac, tgt):
                        m = self.weapons.launch(ac, tgt)
                        if m:
                            self.missiles.append(m)
                            self.events.append({
                                "t": self.time_s,
                                "type": "launch",
                                "shooter": ac.id,
                                "target": tgt.id,
                                "missile": m.id,
                            })
                ac.cmd_fire = False
                ac.fire_target = None

            # Integrate aircraft
            for ac in self.aircraft:
                integrate_aircraft(ac, dt)

            # Missiles
            killed = self.weapons.step(self.missiles, self._by_id, dt)
            for kid in killed:
                self.events.append({"t": self.time_s, "type": "kill", "target": kid})

            self.time_s += dt
            self._record_frame()

        return self._make_result()

    def _record_frame(self) -> None:
        if not self.record:
            return
        states = {}
        for ac in self.aircraft:
            states[ac.id] = {
                "x": ac.state.x,
                "y": ac.state.y,
                "alt": ac.state.alt,
                "heading": ac.state.heading_rad,
                "speed": ac.state.speed_mps,
                "alive": ac.state.alive,
                "name": ac.name,
                "type": ac.ac_type.value,
                "coalition": ac.coalition.value,
                "ammo": ac.ammo,
            }
        mstates = []
        for m in self.missiles:
            if m.alive or (m.hit and abs(m.remaining_time_s) < self.config.dt * 2):
                mstates.append({
                    "id": m.id,
                    "x": m.pos.x,
                    "y": m.pos.y,
                    "alt": m.pos.alt,
                    "alive": m.alive,
                    "coalition": m.coalition,
                    "hit": m.hit,
                })
        self.frames.append({"t": self.time_s, "aircraft": states, "missiles": mstates})

    def _make_result(self) -> SimResult:
        blue_alive = len(self.alive(Coalition.BLUE))
        red_alive = len(self.alive(Coalition.RED))
        n_blue = sum(1 for a in self.aircraft if a.coalition == Coalition.BLUE)
        n_red = sum(1 for a in self.aircraft if a.coalition == Coalition.RED)
        blue_kills = n_red - red_alive
        red_kills = n_blue - blue_alive
        blue_ammo = sum(a.ammo for a in self.aircraft if a.coalition == Coalition.BLUE)

        if red_alive == 0 and blue_alive > 0:
            winner = "blue"
        elif blue_alive == 0 and red_alive > 0:
            winner = "red"
        else:
            winner = "draw"

        return SimResult(
            time_s=self.time_s,
            blue_kills=blue_kills,
            red_kills=red_kills,
            blue_alive=blue_alive,
            red_alive=red_alive,
            blue_ammo_remaining=blue_ammo,
            winner=winner,
            events=self.events,
            frames=self.frames,
        )
