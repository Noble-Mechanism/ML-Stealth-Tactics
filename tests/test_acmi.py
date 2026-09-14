"""Smoke tests for ACMI 2.2 export."""

from __future__ import annotations

from pathlib import Path

from stealth_tactics.scenarios.loader import load_scenario
from stealth_tactics.ga.evolution import GeneticAlgorithm, GAConfig
from stealth_tactics.tactics.genome import TacticsGenome
from stealth_tactics.acmi.exporter import ACMIExporter


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "default_4v3.yaml"


def test_acmi_header_and_objects(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO)
    ga = GeneticAlgorithm(scenario, GAConfig(seed=7, sim_max_time_s=60.0, sim_dt=1.0))
    result = ga.evaluate(TacticsGenome(), record=True)
    assert len(result.sim.frames) > 0

    out = tmp_path / "smoke.txt.acmi"
    exporter = ACMIExporter(title="smoke")
    exporter.export(result.sim.frames, out)

    assert out.is_file()
    assert ACMIExporter.validate_header(out)

    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "FileType=text/acmi/tacview"
    assert lines[1] == "FileVersion=2.2"
    assert lines[2].startswith("0,ReferenceTime=")

    # Time frames and objects present
    assert any(l.startswith("#") for l in lines)
    assert any("BlueStealth" in l for l in lines)
    assert any("RedFighter" in l for l in lines)
    assert any("Type=Air+FixedWing" in l for l in lines)
    assert any("Coalition=Blue" in l for l in lines)
    assert any("Coalition=Red" in l for l in lines)
    assert any("T=" in l for l in lines)


def test_stable_hex_ids() -> None:
    ex = ACMIExporter()
    a = ex.object_id("B1")
    b = ex.object_id("B1")
    c = ex.object_id("R1")
    assert a == b
    assert a != c
    # Hex-like
    int(a, 16)
