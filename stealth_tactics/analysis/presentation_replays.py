"""Spec 4 TacView replays, one per pre-planned maneuver type, each a different
formation (``python -m stealth_tactics presentation-replays``). Default genome
(scripted Blue). Seeds are found by searching the sampler (natural draws, not
stratum overrides) and recorded in the report."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from stealth_tactics.presentation_runner import export_presentation_acmi, run_presentation
from stealth_tactics.scenarios.presentation import Presentation, sample_presentation
from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.sim.sensor_config import NM_M

# key -> (maneuver, formation, band or None, doctrine or None, extra predicate, file)
TARGETS: Dict[str, tuple] = {
    "A": ("split", "wall", "middle", "shoot_assess_shoot", None,
          "replayA_split_wall.txt.acmi"),
    "B": ("pump", "ladder", "aggressive", "shoot_shoot_assess", None,
          "replayB_pump_ladder.txt.acmi"),
    "C": ("low_high_split", "box", "conservative", "shoot_assess_shoot", None,
          "replayC_low_high_split_box.txt.acmi"),
    "D": ("altitude_change", "vic", "aggressive", "shoot_shoot_assess",
          lambda p: (p.maneuver["params"].get("direction") == "high"
                     and p.maneuver["params"]["high_alt_m"] - p.base_alt_m >= 2500.0),
          "replayD_altitude_high_vic.txt.acmi"),
}
TIMELINE_TYPES = {"preplanned_start", "preplanned_skip", "preplanned_abort", "preplanned_end",
                  "defend", "recommit", "press", "depart", "winchester", "hit", "miss",
                  "defeat_speed", "defeat_opening", "miss_overshoot", "ground",
                  "lost_coast_timeout", "lost_basket", "timeout"}


def find_seed(key: str, start: int = 0, limit: int = 20000, min_done: int = 3) -> Tuple[int, object]:
    man, form, band, doc, pred, _ = TARGETS[key]
    for s in range(start, start + limit):
        p = sample_presentation(s)
        if (p.maneuver["type"] != man or p.formation != form
                or (band and p.band != band) or (doc and p.doctrine != doc)
                or (pred and not pred(p))):
            continue
        r = run_presentation(p)
        if r.preplanned["fired"] and r.preplanned["status_counts"].get("done", 0) >= min_done:
            return s, r
    raise RuntimeError(f"no seed found for replay {key}")


def timeline(res) -> str:
    p = Presentation.from_dict(res.presentation)
    names = {r["id"]: r["name"] for r in p.red_jets}
    names.update({b.get("id", f"B{i + 1}"): b.get("name", f"F-35-{i + 1}")
                  for i, b in enumerate(p.blue)})
    L = [p.summary(), ""]
    L.append("Red start (slot: right/back NM, alt m): " + "; ".join(
        f"{r['id']} {r['right_nm']:+.1f}/{r['back_nm']:.1f} {r['alt']:.0f}" for r in p.red_jets))
    L.append("")
    br = [e for e in res.events if e["type"] == "breakup"]
    for e in res.events:
        t = e["t"]
        typ = e["type"]
        if typ == "launch":
            L.append(f"t={t:6.1f}  {names.get(e['shooter'], e['shooter'])} launches "
                     f"{e['missile']} at {names.get(e['target'], e['target'])}")
        elif typ == "kill":
            L.append(f"t={t:6.1f}  KILL: {names.get(e['target'], e['target'])} destroyed")
        elif typ in TIMELINE_TYPES:
            L.append(f"t={t:6.1f}  {e.get('text', typ)}")
    pp = res.preplanned
    L += ["", f"Break-ups: " + ", ".join(f"{e['observer']}@{e['t']:.0f}s ({e['reason']})"
                                         for e in br),
          f"Pre-planned maneuver: {pp['type']} fired={pp['fired']} at t={pp['fire_t']} s, "
          f"{(pp['fire_range_nm'] or 0):.1f} NM; per-jet {pp['status_counts']}",
          f"End: {res.end_reason} at t={res.time_s:.1f} s; Blue kills {res.blue_kills}, "
          f"Blue losses {res.red_kills}; Blue shots {res.blue_shots}, Red shots {res.red_shots}"]
    return "\n".join(L)


def make_replays(out_dir: Path, keys: Optional[List[str]] = None) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ["Spec 4 presentation replays (default genome; Blue Name=F-35A in ACMI)", ""]
    for k in keys or list(TARGETS):
        seed, _ = find_seed(k)
        res = run_presentation(sample_presentation(seed), record=True)
        fname = TARGETS[k][5]
        export_presentation_acmi(res, out_dir / fname, title=f"Spec 4 replay {k}: "
                                 f"{TARGETS[k][0]} / {TARGETS[k][1]}")
        (out_dir / fname.replace(".txt.acmi", ".json")).write_text(
            Presentation.from_dict(res.presentation).to_json(), encoding="utf-8")
        tl = timeline(res)
        (out_dir / fname.replace(".txt.acmi", "_timeline.txt")).write_text(tl + "\n",
                                                                            encoding="utf-8")
        report += [f"=== Replay {k}: {TARGETS[k][0]} / {TARGETS[k][1]} (seed {seed}) -> {fname}",
                   tl, ""]
    text = "\n".join(report)
    (out_dir / "presentation_replays.txt").write_text(text + "\n", encoding="utf-8")
    return text
