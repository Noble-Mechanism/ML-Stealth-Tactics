"""Scripted 1v1 sensor replays (Spec 1) exported to TacView ACMI.

Replay 1 ("headon"): one F-35 flies straight at one Red fighter; Red flies
straight toward the F-35 (pure pursuit, which is straight for a head-on start).
Replay 2 ("beam"): identical, except the F-35 turns to the beam (heading 90 deg
off the line of sight to Red) when range first reaches 30 NM.

Replay 3 ("beam_then_hot"): the F-35 flies hot, turns to the beam at 45 NM,
holds it until Red gains a fire-control lock on it OR 60 s elapse (whichever
first), then turns hot again (nose continuously on Red) and continues.

No weapons are fired: the replays exist to show detections, fire-control
locks (ACMI LockedTarget) and sensor events (ACMI Event=Bookmark/Message).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.sim.aircraft import Aircraft, AircraftState, _wrap_pi, bearing_to
from stealth_tactics.sim.sensor_config import NM_M
from stealth_tactics.sim.sensors import distance_3d
from stealth_tactics.sim.world import SimConfig, World

START_RANGE_M = 70 * NM_M
TURN_RANGE_M = 30 * NM_M
BTH_BEAM_RANGE_M = 45 * NM_M
BTH_MAX_BEAM_S = 60.0
MERGE_EXCLUDE_M = 3 * NM_M
DURATION_S = 280.0
RED_TYPE_NAME = "Su-27"  # generic Red placeholder; TacView DB name for display only


@dataclass
class ReplayResult:
    name: str
    world: World
    turn_t: Optional[float] = None
    hot_t: Optional[float] = None          # beam_then_hot: time turned hot again
    hot_range_m: Optional[float] = None
    hot_reason: str = ""
    first: Dict[str, dict] = field(default_factory=dict)


def _build():
    b = Aircraft.make_blue("B1", "F-35-1", AircraftState(0.0, 0.0, 9000.0, 0.0, 260.0))
    r = Aircraft.make_red("R1", "Red-1",
                          AircraftState(0.0, START_RANGE_M, 8500.0, math.pi, 255.0))
    r.type_name = RED_TYPE_NAME
    return b, r


def run_replay(beam: bool, seed: int = 1, record: bool = True) -> ReplayResult:
    b, r = _build()
    state = {"turn_t": None, "turn_hdg": None}

    def blue_ctl(w: World) -> None:
        if not b.state.alive:
            return
        b.cmd_speed_mps, b.cmd_alt_m = 260.0, 9000.0
        if beam and state["turn_hdg"] is None and distance_3d(b.state, r.state) <= TURN_RANGE_M:
            # beam: 90 deg off the line of sight to Red (turn right)
            state["turn_hdg"] = _wrap_pi(bearing_to(b.state, r.state) + math.pi / 2)
            state["turn_t"] = w.time_s
        b.cmd_heading_rad = state["turn_hdg"] if state["turn_hdg"] is not None else 0.0

    def red_ctl(w: World) -> None:
        r.cmd_heading_rad = bearing_to(r.state, b.state)  # fly toward the F-35
        r.cmd_speed_mps, r.cmd_alt_m = 255.0, 8500.0

    w = World([b, r], SimConfig(dt=0.5, max_time_s=DURATION_S, seed=seed),
              blue_ctl, red_ctl, record=record)
    w.run()
    res = ReplayResult("beam" if beam else "headon", w, state["turn_t"])
    for e in w.sensor_events:
        key = f"{e['observer']}:{e['type']}"
        res.first.setdefault(key, e)
    return res


def run_beam_then_hot(seed: int = 1, record: bool = True,
                      beam_range_m: float = BTH_BEAM_RANGE_M) -> ReplayResult:
    b, r = _build()
    st = {"phase": "hot1", "beam_t": None, "beam_hdg": None,
          "hot_t": None, "hot_r": None, "reason": ""}

    def blue_ctl(w: World) -> None:
        if not b.state.alive:
            return
        b.cmd_speed_mps, b.cmd_alt_m = 260.0, 9000.0
        rng = distance_3d(b.state, r.state)
        if st["phase"] == "hot1" and rng <= beam_range_m:
            st["phase"], st["beam_t"] = "beam", w.time_s
            st["beam_hdg"] = _wrap_pi(bearing_to(b.state, r.state) + math.pi / 2)
        if st["phase"] == "beam":
            red_store = w.tracks.get(r.id)
            locked = red_store is not None and red_store.is_fire_control(b.id)
            if locked or w.time_s - st["beam_t"] >= BTH_MAX_BEAM_S - 1e-9:
                st["phase"], st["hot_t"], st["hot_r"] = "hot2", w.time_s, rng
                st["reason"] = "Red FC lock" if locked else "60 s elapsed"
        if st["phase"] == "beam":
            b.cmd_heading_rad = st["beam_hdg"]
        elif st["phase"] == "hot2":
            b.cmd_heading_rad = bearing_to(b.state, r.state)  # nose on Red
        else:
            b.cmd_heading_rad = 0.0

    def red_ctl(w: World) -> None:
        r.cmd_heading_rad = bearing_to(r.state, b.state)
        r.cmd_speed_mps, r.cmd_alt_m = 255.0, 8500.0

    w = World([b, r], SimConfig(dt=0.5, max_time_s=DURATION_S, seed=seed),
              blue_ctl, red_ctl, record=record)
    w.run()
    res = ReplayResult("beam_then_hot", w, st["beam_t"], st["hot_t"], st["hot_r"],
                       st["reason"])
    for e in w.sensor_events:
        res.first.setdefault(f"{e['observer']}:{e['type']}", e)
    return res


def beam_then_hot_stats(seeds: int = 100,
                        beam_range_m: float = BTH_BEAM_RANGE_M) -> dict:
    """Per-seed outcome of replay 3 (Red's lock/track on the F-35 after going hot)."""
    runs = []
    for s in range(seeds):
        res = run_beam_then_hot(seed=s, record=False, beam_range_m=beam_range_m)
        ht = res.hot_t
        red = [e for e in res.world.sensor_events
               if e["observer"] == "R1" and e["target"] == "B1"]
        fc_before = any(e["type"] == "fc_track" and e["t"] <= ht for e in red)
        # FC state at the moment of turning hot (last fc_track/fc_lost before hot_t)
        fc_state = False
        for e in red:
            if e["t"] <= ht and e["type"] in ("fc_track", "fc_lost"):
                fc_state = e["type"] == "fc_track"
        # Post-hot events, excluding the merge/pass (< 3 NM) where every track
        # drops because the jets pass out of each other's field of regard.
        after = [e for e in red if e["t"] > ht and e["range_m"] > MERGE_EXCLUDE_M]
        fc_lost = next((e for e in after if e["type"] == "fc_lost"), None)
        radar_lost = next((e for e in after if e["type"] == "radar_lost"), None)
        radar_state_at_hot = any(e["type"] == "radar_detect" and e["t"] <= ht for e in red)
        t_ref = fc_lost["t"] if fc_lost else ht
        relock = next((e for e in after if e["type"] == "fc_track" and e["t"] > t_ref), None)
        reacq = next((e for e in after if e["type"] == "radar_detect"
                      and radar_lost and e["t"] > radar_lost["t"]), None)
        irst_after = [e for e in after if e["type"] in ("irst_detect", "irst_lost")]
        runs.append(dict(seed=s, hot_t=ht, hot_r=res.hot_range_m, reason=res.hot_reason,
                         beam_t=res.turn_t, fc_during_beam=fc_before,
                         fc_at_hot=fc_state, radar_at_hot=radar_state_at_hot,
                         fc_lost=fc_lost, radar_lost=radar_lost, relock=relock,
                         reacq=reacq, irst_after=irst_after))
    return {"runs": runs}


def format_bth_stats(stats: dict) -> str:
    runs = stats["runs"]
    n = len(runs)
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    fc = [r for r in runs if r["fc_at_hot"]]
    broke = [r for r in fc if r["fc_lost"]]
    lines = [f"Replay 3 statistics over {n} seeds (0..{n - 1}); post-hot events "
             "inside 3 NM (the merge/pass) excluded:"]
    lines.append(f"  F-35 turned hot: median range {med([r['hot_r'] for r in runs]) / NM_M:.1f} NM, "
                 f"median t {med([r['hot_t'] for r in runs]):.1f} s; reason: "
                 f"{sum(r['reason'] == 'Red FC lock' for r in runs)} Red FC lock, "
                 f"{sum(r['reason'] == '60 s elapsed' for r in runs)} 60 s timeout")
    lines.append(f"  Red radar track on F-35 at turn-hot: {sum(r['radar_at_hot'] for r in runs)}/{n}")
    lines.append(f"  Red FC lock achieved during beam: {len(fc)}/{n} "
                 f"({100.0 * len(fc) / n:.0f} %)")
    if fc:
        lines.append(f"    lock broke after going hot: {len(broke)}/{len(fc)}")
        if broke:
            lines.append(
                "    turn-hot -> lock loss: median "
                f"{med([r['fc_lost']['t'] - r['hot_t'] for r in broke]):.1f} s, at "
                f"{med([r['fc_lost']['range_m'] for r in broke]) / NM_M:.1f} NM")
    rl = [r for r in runs if r["radar_lost"]]
    lines.append(f"  Red lost radar track entirely after going hot: {len(rl)}/{n}")
    if rl:
        lines.append(
            f"    turn-hot -> radar track drop: median "
            f"{med([r['radar_lost']['t'] - r['hot_t'] for r in rl]):.1f} s, at "
            f"{med([r['radar_lost']['range_m'] for r in rl]) / NM_M:.1f} NM")
    ra = [r for r in runs if r["reacq"]]
    lines.append(f"  Red radar re-detection after drop: {len(ra)}/{len(rl) or 1}"
                 + (f", median {med([r['reacq']['range_m'] for r in ra]) / NM_M:.1f} NM"
                    if ra else ""))
    relock = [r for r in runs if r["relock"]]
    lines.append(f"  Red (re)gained FC lock after going hot: {len(relock)}/{n}"
                 + (f", median {med([r['relock']['range_m'] for r in relock]) / NM_M:.1f} NM"
                    if relock else ""))
    first_fc_all = [r for r in runs if r["relock"] and not r["fc_at_hot"]]
    if first_fc_all:
        lines.append(f"    (of which first-ever lock, no beam lock: {len(first_fc_all)})")
    return "\n".join(lines)


def describe(res: ReplayResult) -> str:
    title = {
        "beam": "Replay 2: F-35 turns to the beam at 30 NM",
        "headon": "Replay 1: head-on, both straight",
        "beam_then_hot": "Replay 3: F-35 beams at 45 NM, then turns hot "
                         "(on Red FC lock or after 60 s)",
    }[res.name]
    lines = [title, f"  start range {START_RANGE_M / NM_M:.0f} NM, F-35 260 m/s, "
             f"Red 255 m/s pursuing, dt 0.5 s, seed {res.world.config.seed}"]
    if res.turn_t is not None:
        lines.append(f"  F-35 beam turn commanded at t={res.turn_t:.1f} s")
    if res.hot_t is not None:
        lines.append(f"  F-35 turn HOT commanded at t={res.hot_t:.1f} s, range "
                     f"{res.hot_range_m / NM_M:.1f} NM ({res.hot_range_m / 1000:.1f} km), "
                     f"reason: {res.hot_reason}")
    who = {"B1": "F-35", "R1": "Red"}
    for e in res.world.sensor_events:
        lines.append(f"  t={e['t']:6.1f}s  {who[e['observer']]:5} {e['type']:13} "
                     f"range {e['range_m'] / NM_M:5.1f} NM ({e['range_m'] / 1000:5.1f} km)")
    return "\n".join(lines)


def export(res: ReplayResult, path: Path) -> Path:
    title = {"headon": "Spec1 replay 1 - head-on",
             "beam": "Spec1 replay 2 - F-35 beams at 30 NM",
             "beam_then_hot": "Spec1 replay 3 - F-35 beams at 45 NM then turns hot",
             }[res.name]
    return ACMIExporter(title=title).export(res.world.frames, path)


def first_detect_stats(beam: bool, seeds: int = 100) -> Dict[str, List[float]]:
    """First-detection ranges (m) per observer/sensor across seeds."""
    out: Dict[str, List[float]] = {}
    for s in range(seeds):
        res = run_replay(beam, seed=s, record=False)
        for key, e in res.first.items():
            if key.endswith("_detect") or key.endswith(":fc_track"):
                out.setdefault(key, []).append(e["range_m"])
    return out
