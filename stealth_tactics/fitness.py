"""Spec 7 fitness v1 (approved by Rusty 2026-09-26) + Spec 8 deconfliction.

All weights live in ``scenarios/fitness.yaml`` (or a file given per run) and
are overridable key by key. ``fight_fitness`` scores one fight;
``aggregate`` turns a network's per-fight scores into its fitness
(mean - std_coef x std). See ``docs/specs/07-fitness.md`` and
``docs/specs/08-kinematics-realism.md`` (D).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

DEFAULT_FITNESS_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "fitness.yaml"

# Built-in defaults (used if the YAML is missing a key; the YAML ships the same values)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "kill": 100.0, "kill_ref_red": 6, "blue_loss": -150.0, "blue_loss_egress": -250.0,
    "egress_off_deg": 120.0, "escape": 10.0, "escape_only_at_time_cap": True,
    "red_winchester_depart": 25.0, "shot": -2.0, "no_engagement_loss_only": True,
    "no_engagement": -300.0, "std_coef": 0.2,
    # Spec 8 D: Blue–Blue deconfliction (AND rule: < deconflict_nm AND < deconflict_alt_ft)
    "deconflict_nm": 5.0,
    "deconflict_alt_ft": 5000.0,
    "deconflict_per_s": -1.0,
    "deconflict_cap": -50.0,
}


def _coerce(key: str, value):
    ref = DEFAULT_WEIGHTS.get(key)
    if isinstance(ref, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(value, str):
        return float(value)
    return value


def load_weights(path: Optional[str] = None, overrides: Optional[Mapping] = None) -> dict:
    """Defaults <- YAML file (default scenarios/fitness.yaml) <- overrides.
    New keys are allowed (extensible); unknown keys are simply carried along."""
    w = dict(DEFAULT_WEIGHTS)
    p = Path(path) if path else DEFAULT_FITNESS_PATH
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p}: fitness file must be a mapping of weight -> value")
        w.update({k: _coerce(k, v) for k, v in data.items()})
    elif path:
        raise FileNotFoundError(f"fitness weights file not found: {p}")
    for k, v in (overrides or {}).items():
        w[k] = _coerce(k, v)
    return w


def parse_overrides(items: Iterable[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--fw expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = _coerce(k.strip(), v.strip())
    return out


def fight_terms(res, n_red: int, egress_by_jet: Mapping[str, bool], w: Mapping,
                deconflict_s: float = 0.0) -> dict:
    """Per-term breakdown of one fight. ``egress_by_jet``: Blue jet id -> was
    it egressing at its last sample (from the behaviour recorder).
    ``deconflict_s``: Blue–Blue conflict-pair-seconds (Spec 8 D)."""
    kill_ids = [e["target"] for e in res.events if e.get("type") == "kill"]
    blue_ids = set(egress_by_jet)
    blue_lost = [k for k in kill_ids if k in blue_ids] if blue_ids else []
    n_loss = int(res.red_kills)
    n_egress_loss = sum(1 for k in blue_lost if egress_by_jet.get(k))
    killed = set(kill_ids)
    departed = {e.get("observer") for e in res.events
                if e.get("type") == "depart" and "winchester" in
                (str(e.get("reason", "")) + str(e.get("text", ""))).lower()}
    departed = {d for d in departed if d and d not in killed and d not in blue_ids}
    shots = int(getattr(res, "blue_shots", 0) or 0)
    engaged = shots > 0 or res.blue_kills > 0
    escaped = int(res.blue_alive) if (not w["escape_only_at_time_cap"]
                                      or res.end_reason == "time_cap") else 0
    # Spec 8 D: per conflict-pair-second, capped
    per_s = float(w.get("deconflict_per_s", 0.0))
    cap = float(w.get("deconflict_cap", 0.0))
    raw_deconf = per_s * float(deconflict_s)
    if per_s < 0:
        deconf = max(raw_deconf, cap) if cap < 0 else raw_deconf
    else:
        deconf = min(raw_deconf, cap) if cap > 0 else raw_deconf
    t = {"kills": w["kill"] * (w["kill_ref_red"] / max(1, n_red)) * res.blue_kills,
         "losses": w["blue_loss"] * (n_loss - n_egress_loss),
         "egress_losses": w["blue_loss_egress"] * n_egress_loss,
         "escape": w["escape"] * escaped,
         "red_winchester_departs": w["red_winchester_depart"] * len(departed),
         "shots": w["shot"] * shots,
         "no_engagement": 0.0 if engaged else float(w.get("no_engagement", 0.0)),
         "deconflict": float(deconf)}
    if w["no_engagement_loss_only"] and not engaged:
        t.update(kills=0.0, escape=0.0, red_winchester_departs=0.0, shots=0.0)
    t["total"] = float(sum(v for k, v in t.items() if k != "total"))
    t.update(n_losses=n_loss, n_egress_losses=n_egress_loss, n_escaped=escaped,
             n_red_departs=len(departed), engaged=engaged,
             deconflict_s=float(deconflict_s))
    return t


def fight_fitness(res, n_red: int, egress_by_jet: Mapping[str, bool], w: Mapping,
                  deconflict_s: float = 0.0) -> float:
    return fight_terms(res, n_red, egress_by_jet, w, deconflict_s)["total"]


def aggregate(per_fight: Sequence[float], w: Mapping) -> float:
    """Network fitness = mean - std_coef x (population) std over presentations."""
    a = np.asarray(per_fight, dtype=float)
    if a.size == 0:
        return 0.0
    return float(a.mean() - float(w.get("std_coef", 0.2)) * a.std())


def missile_outcome_counts(shots: Sequence[Mapping]) -> dict:
    """Spec 8 C: Blue-as-target and Red-as-target outcome counts by label.

    A Blue shot has Red as target (and vice versa). Returns
    ``{"blue_as_target": {label: n}, "red_as_target": {label: n}}``.
    """
    blue_tgt: Counter = Counter()
    red_tgt: Counter = Counter()
    for s in shots or []:
        out = s.get("outcome") or ""
        if not out or out == "in_flight":
            continue
        coal = s.get("coalition")
        if coal == "Blue":
            red_tgt[out] += 1
        elif coal == "Red":
            blue_tgt[out] += 1
    return {"blue_as_target": dict(blue_tgt), "red_as_target": dict(red_tgt)}


def merge_outcome_counts(parts: Sequence[Mapping]) -> dict:
    """Sum outcome rollups across fights."""
    blue: Counter = Counter()
    red: Counter = Counter()
    for p in parts or []:
        blue.update(p.get("blue_as_target") or {})
        red.update(p.get("red_as_target") or {})
    return {"blue_as_target": dict(blue), "red_as_target": dict(red)}
