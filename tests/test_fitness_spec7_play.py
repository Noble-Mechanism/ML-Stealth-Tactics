"""Spec 7 fitness v1 terms + aggregation, and the play package (wall start,
random default, fitness weights in the config hash, overnight auto-resume)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from stealth_tactics.fitness import (DEFAULT_WEIGHTS, aggregate, fight_terms, load_weights,
                                     parse_overrides)

W = dict(DEFAULT_WEIGHTS)
BLUE = ["B1", "B2", "B3", "B4"]


def res(kills=0, losses=(), alive=None, shots=0, end="blue_dead", events=()):
    ev = [{"type": "kill", "target": b} for b in losses] + list(events)
    return SimpleNamespace(blue_kills=kills, red_kills=len(losses),
                           blue_alive=4 - len(losses) if alive is None else alive,
                           blue_shots=shots, end_reason=end, events=ev)


def eg(*ids):
    return {b: b in ids for b in BLUE}


def test_kill_term_scales_with_n_red():
    assert fight_terms(res(kills=2, shots=1), 6, eg(), W)["kills"] == 200.0
    assert fight_terms(res(kills=2, shots=1), 8, eg(), W)["kills"] == pytest.approx(150.0)


def test_loss_and_egress_loss_terms():
    t = fight_terms(res(kills=1, losses=["B1", "B2"], shots=2), 6, eg("B2"), W)
    assert t["losses"] == -150.0 and t["egress_losses"] == -250.0
    assert t["n_losses"] == 2 and t["n_egress_losses"] == 1


def test_escape_only_at_time_cap():
    assert fight_terms(res(kills=1, shots=2, end="time_cap"), 6, eg(), W)["escape"] == 40.0
    assert fight_terms(res(kills=1, shots=2, end="red_dead"), 6, eg(), W)["escape"] == 0.0
    w2 = {**W, "escape_only_at_time_cap": False}
    assert fight_terms(res(kills=1, shots=2, end="red_dead"), 6, eg(), w2)["escape"] == 40.0


def test_red_winchester_departure_quarter_kill():
    evs = [{"type": "depart", "observer": "R1", "reason": "Winchester: no missiles left",
            "text": "DEPART"},
           {"type": "depart", "observer": "R2", "text": "R2: DEPART (Winchester: no missiles left)"},
           {"type": "depart", "observer": "R3", "text": "R3: DEPART (turn-away limit 2 used)"},
           {"type": "depart", "observer": "R4", "text": "DEPART (Winchester)"},
           {"type": "kill", "target": "R4"}]                    # killed later: kill only
    t = fight_terms(res(kills=1, shots=3, events=evs), 6, eg(), W)
    assert t["n_red_departs"] == 2 and t["red_winchester_departs"] == 50.0


def test_shot_term_and_total():
    t = fight_terms(res(kills=1, losses=["B3"], shots=5), 6, eg(), W)
    assert t["shots"] == -10.0
    assert t["total"] == 100.0 - 150.0 - 10.0


def test_no_engagement_penalty():
    evs = [{"type": "depart", "observer": "R1", "text": "DEPART (Winchester)"}]
    t = fight_terms(res(losses=["B1"], end="time_cap", events=evs), 6, eg("B1"), W)
    assert not t["engaged"] and t["total"] == -250.0 - 300.0     # losses + penalty only
    t0 = fight_terms(res(end="time_cap"), 6, eg(), W)             # 0 losses
    assert t0["total"] == -300.0 and t0["no_engagement"] == -300.0
    assert fight_terms(res(kills=1, shots=1), 6, eg(), W)["no_engagement"] == 0.0
    assert fight_terms(res(end="time_cap"), 6, eg(), {**W, "no_engagement": 0.0})["total"] == 0.0


def test_fighter_profile_beats_pure_runaway():
    """Last demo's hall-of-fame fighter (~2.32 kills, 1.12 losses, ~14 shots per
    fight) vs a pure runaway (never shoots, never dies), synthetic fights."""
    fighter = []
    for i in range(100):
        k, l = [(2, 1), (3, 1), (2, 1), (3, 2), (2, 1)][i % 5] if i < 64 else [(2, 1), (2, 1), (3, 1), (2, 1)][i % 4]
        fighter.append(fight_terms(res(kills=k, losses=["B1", "B2"][:l], shots=14), 6, eg(), W)["total"])
    kills = np.mean([[(2, 1), (3, 1), (2, 1), (3, 2), (2, 1)][i % 5][0] if i < 64 else
                     [(2, 1), (2, 1), (3, 1), (2, 1)][i % 4][0] for i in range(100)])
    assert 2.2 < kills < 2.5
    runaway = [fight_terms(res(end="time_cap"), 6, eg(), W)["total"]] * 100
    assert aggregate(fighter, W) > aggregate(runaway, W)
    assert aggregate(runaway, W) == -300.0
    assert W["std_coef"] == 0.2


def test_aggregation_mean_minus_half_std():
    per = [100.0, 0.0, 50.0, -50.0]
    assert aggregate(per, W) == pytest.approx(np.mean(per) - 0.2 * np.std(per))
    assert aggregate(per, {**W, "std_coef": 0.0}) == pytest.approx(25.0)
    assert aggregate([7.0] * 5, W) == 7.0


def test_weights_yaml_overrides_and_extensible(tmp_path):
    w = load_weights()
    assert w == {**DEFAULT_WEIGHTS, **w} and w["kill"] == 100.0 and w["std_coef"] == 0.2 and w["no_engagement"] == -300.0
    f = tmp_path / "fw.yaml"
    f.write_text("blue_loss: -200\nmy_new_term: 3.5\n")
    w2 = load_weights(str(f), parse_overrides(["shot=-1", "escape_only_at_time_cap=false"]))
    assert w2["blue_loss"] == -200 and w2["my_new_term"] == 3.5 and w2["shot"] == -1.0
    assert w2["escape_only_at_time_cap"] is False and w2["kill"] == 100.0
    with pytest.raises(FileNotFoundError):
        load_weights(str(tmp_path / "missing.yaml"))


def test_egress_flag_from_recorder():
    from stealth_tactics.neuro.novelty import BDRecorder
    r = BDRecorder(lambda w: None, BLUE)
    r.last_off_deg = {"B1": 170.0, "B2": 100.0, "B3": 121.0}
    assert r.egress_by_jet(120.0) == {"B1": True, "B2": False, "B3": True, "B4": False}


# ------------------------------------------------------------- play package
def test_wall_start_geometry_and_diamond_kept():
    from stealth_tactics.scenarios.presentation import (DEFAULT_PRESENTATION_CONFIG, NM_M,
                                                        presentation_config, sample_presentation)
    pw = sample_presentation(123, presentation_config("wall"))
    xs = sorted(b["x"] for b in pw.blue)
    assert len(pw.blue) == 4 and (xs[-1] - xs[0]) / NM_M == pytest.approx(30.0, abs=0.05)
    assert np.allclose(np.diff(xs), (xs[-1] - xs[0]) / 3, atol=1.0)
    assert len({b["y"] for b in pw.blue}) == 1 and len({b["heading_rad"] for b in pw.blue}) == 1
    # Red placed relative to the wall's centre
    cx, cy = np.mean([b["x"] for b in pw.blue]), pw.blue[0]["y"]
    lead = pw.red_jets[0]
    assert math.hypot(lead["x"] - cx, lead["y"] - cy) / NM_M == pytest.approx(pw.range_nm, rel=0.35)
    # the diamond (spec 4 default) is unchanged
    pd = sample_presentation(123, presentation_config("diamond"))
    assert pd.to_dict() == sample_presentation(123, DEFAULT_PRESENTATION_CONFIG).to_dict()
    with pytest.raises(ValueError):
        presentation_config("vic")


def test_neuro_defaults_random_wall_and_fitness_in_hash():
    from stealth_tactics.neuro.neuroga import NeuroConfig, config_hash
    c = NeuroConfig()
    assert c.init == "random" and c.blue_start == "wall" and c.fitness == load_weights()
    c2 = NeuroConfig(fitness={**c.fitness, "blue_loss": -151.0})
    assert config_hash(c, None) != config_hash(c2, None)
    assert NeuroConfig.from_dict(c2.to_dict()).fitness["blue_loss"] == -151.0


def test_resume_refuses_changed_fitness_weights(tmp_path):
    from stealth_tactics.neuro.mlp import Arch
    from stealth_tactics.neuro.neuroga import NeuroConfig, NeuroGA
    from stealth_tactics.neuro.toy import ToyTask
    rng = np.random.default_rng(0)
    obs, act = rng.uniform(-1, 1, (50, 231)), rng.uniform(-1, 1, (50, 13))
    kw = dict(population=6, truncation=3, arch=Arch(hidden=(4,)).to_dict())
    NeuroGA(NeuroConfig(**kw), tmp_path, workers=1, task=ToyTask(obs, act),
            log=lambda *a: None).run(1, finalize=False)
    w2 = {**load_weights(), "kill": 90.0}
    ga = NeuroGA(NeuroConfig(fitness=w2, **kw), tmp_path, workers=1, task=ToyTask(obs, act),
                 log=lambda *a: None)
    with pytest.raises(ValueError, match="refusing to resume"):
        ga.run(2, resume=True, finalize=False)


def test_overnight_progress_outputs_and_auto_resume(tmp_path):
    from stealth_tactics.neuro.neuroga import NeuroConfig
    from stealth_tactics.neuro.overnight import run_overnight
    cfg = NeuroConfig(population=4, presentations=2, benchmark=2, test_size=0, truncation=2,
                      n_clones=0, max_time_s=60.0)
    ga = run_overnight(cfg, tmp_path, workers=4, every=2, gens=2, log=lambda *a: None)
    prog = tmp_path / "progress"
    for f in ("progress.md", "hall_of_fame.md", "champion.json", "champion_best.txt.acmi",
              "champion_worst.txt.acmi"):
        assert (prog / f).exists(), f
    h2 = [r["best_fitness"] for r in ga.history]
    # a restart with the same config resumes (no re-init) and continues to gen 4
    ga2 = run_overnight(cfg, tmp_path, workers=4, every=2, gens=4, log=lambda *a: None)
    assert ga2.gen == 4 and [r["best_fitness"] for r in ga2.history[:2]] == h2
    assert "generations done: 4" in (prog / "progress.md").read_text()
    from stealth_tactics.neuro.neuroga import replay_network_champion
    ok, msg = replay_network_champion(prog, workers=4)
    assert ok, msg
