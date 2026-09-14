"""Load engagement scenarios from YAML/JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import yaml

from stealth_tactics.sim.aircraft import Aircraft, AircraftState


@dataclass
class Scenario:
    name: str
    description: str = ""
    red_mode: str = "intercept"  # intercept | cap
    blue: List[Dict[str, Any]] = field(default_factory=list)
    red: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_scenario(path: Path | str) -> Scenario:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return Scenario(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        red_mode=data.get("red_mode", "intercept"),
        blue=list(data.get("blue", [])),
        red=list(data.get("red", [])),
        raw=data,
    )


def build_aircraft(scenario: Scenario) -> List[Aircraft]:
    aircraft: List[Aircraft] = []
    for i, spec in enumerate(scenario.blue):
        uid = spec.get("id", f"B{i+1}")
        name = spec.get("name", f"BlueStealth-{i+1}")
        st = AircraftState(
            x=float(spec.get("x", 0)),
            y=float(spec.get("y", -40000 + i * 500)),
            alt=float(spec.get("alt", 9000)),
            heading_rad=float(spec.get("heading_rad", 0.0)),
            speed_mps=float(spec.get("speed_mps", 260)),
        )
        aircraft.append(Aircraft.make_blue(uid, name, st))

    for i, spec in enumerate(scenario.red):
        uid = spec.get("id", f"R{i+1}")
        name = spec.get("name", f"RedFighter-{i+1}")
        st = AircraftState(
            x=float(spec.get("x", 0)),
            y=float(spec.get("y", 40000)),
            alt=float(spec.get("alt", 8500)),
            heading_rad=float(spec.get("heading_rad", 3.14159)),  # south
            speed_mps=float(spec.get("speed_mps", 250)),
        )
        aircraft.append(Aircraft.make_red(uid, name, st))

    return aircraft
