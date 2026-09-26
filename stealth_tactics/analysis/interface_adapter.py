"""Spec 5 Q: the three-layer adapter test, the radar-silent replay, the
random-weights smoke run and the timing (``python -m stealth_tactics
interface-adapter-test``). Writes to ``/workspace/spec5_outputs/``.

Variants (all on the 100 spec 4 stats seeds, 360 s cap, spec 4 end rules):
- ``scripted``: default genome, direct ``TacticsController`` (reads truth);
- ``layer2``: the same script through the 13-output encoder/decoder at 1 s hold;
- ``hand``: ``HandBlue`` on the 231-input observation only, 1 s;
- ``silent``: ``HandBlue`` with slots 2 and 3 radar-silent for the whole fight.
Layer 1 (0.5 s, byte-identical ACMI) runs on the first 20 seeds.
"""

from __future__ import annotations

import filecmp
import json
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stealth_tactics.presentation_runner import export_presentation_acmi, run_presentation
from stealth_tactics.scenarios.presentation import sample_presentation
from stealth_tactics.sim.aircraft import Coalition
from stealth_tactics.sim.rwr import MISSILE_ACTIVE
from stealth_tactics.sim.world import END_REASONS

from stealth_tactics.policy.adapters import HandBlue, RandomMLPPolicy, ScriptedViaInterface
from stealth_tactics.policy.controller import NetworkBlueController
from stealth_tactics.policy.observation import OBS_SPEC
from .presentation_stats import END_ABBR, stats_seeds

OUT = Path("/workspace/spec5_outputs")
SILENT_SLOTS = (2, 3)
VARIANTS = ("scripted", "layer2", "hand", "silent")
LABELS = {"scripted": "Scripted Blue (truth, direct)",
          "layer2": "Layer 2: script via interface, 1 s hold",
          "hand": "Layer 3: HandBlue on obs only, 1 s",
          "silent": "Layer 3 variant: HandBlue, B2+B3 radar-silent",
          "random": "Random-weights MLP 231-64-64-13, 1 s"}


_LO, _HI = OBS_SPEC.bounds()


class CheckedNet(NetworkBlueController):
    """Counts observation vectors and any non-finite / out-of-declared-range input."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.obs_checked = 0
        self.obs_violations = 0

    def actions(self, world, jets, obs, infos):
        self.obs_checked += obs.shape[0]
        bad = ~np.isfinite(obs) | (obs < _LO - 1e-12) | (obs > _HI + 1e-12)
        self.obs_violations += int(bad.any(axis=1).sum())
        return super().actions(world, jets, obs, infos)


class SilentAudit:
    """Wraps a Blue controller; every step (after sensors + RWR update) counts
    Red RWR cues and Red spec 1 RWR track components whose emitter is a silent
    Blue jet, and the silent jets' own radar tracks / own FC."""

    def __init__(self, inner, silent_ids) -> None:
        self.inner = inner
        self.silent = set(silent_ids)
        self.red_cues = 0            # search / lock / support cues from silent jets
        self.red_rwr_components = 0  # spec 1 RWR track components on silent jets
        self.missile_cues = 0        # missile_active cues from missiles a silent jet fired
        self.own_radar_components = 0
        self.own_fc = 0
        self.emitting_steps = 0

    def __call__(self, world) -> None:
        for a in world.aircraft:
            if a.coalition == Coalition.RED:
                for c in world.rwr.get(a.id, []):
                    if c.mode == MISSILE_ACTIVE:
                        self.missile_cues += c.source_jet in self.silent
                    elif c.emitter_id in self.silent:
                        self.red_cues += 1
                st = world.tracks.get(a.id)
                if st is not None:
                    for b in self.silent:
                        tr = st.tracks.get(b)
                        if tr is not None and "rwr" in tr.components:
                            self.red_rwr_components += 1
            elif a.id in self.silent and a.state.alive:
                self.emitting_steps += bool(a.radar_emitting)
                st = world.tracks.get(a.id)
                if st is not None:
                    for tr in st.tracks.values():
                        self.own_radar_components += "radar" in tr.components
                        self.own_fc += bool(tr.fire_control)
        self.inner(world)

    def report(self) -> dict:
        return {k: getattr(self, k) for k in ("red_cues", "red_rwr_components", "missile_cues",
                                              "own_radar_components", "own_fc",
                                              "emitting_steps")}


class PhaseShiftedScripted(ScriptedViaInterface):
    """Layer 2 noise baseline: the same 1 s hold, decisions at t = 0, 1.5, 2.5 ...
    (half a period later). Statistically the same treatment as layer 2, so the
    difference between the two measures pure fight-to-fight chaos."""

    def __call__(self, world) -> None:
        super().__call__(world)
        if self.n_decisions == 1 and abs(self._next_t - self.period) < 1e-9:
            self._next_t = 1.5 * self.period


def _silent_prepare(ids):
    def prep(aircraft, defense):
        for a in aircraft:
            if a.id in ids:
                a.radar_emitting = False
    return prep


def blue_ids_of(p) -> List[str]:
    from stealth_tactics.scenarios.presentation import build_presentation_aircraft
    return [a.id for a in build_presentation_aircraft(p) if a.coalition == Coalition.BLUE]


def run_variant(seed: int, variant: str, record: bool = False, policy_seed: int = 0,
                period: float = 1.0):
    """Returns (SimResult, extra dict)."""
    p = sample_presentation(seed)
    extra: Dict = {}
    holder: Dict = {}
    prepare = None
    if variant == "scripted":
        factory = None
    elif variant == "layer2":
        def factory(ids):
            holder["c"] = ScriptedViaInterface(ids, decision_period_s=period)
            return holder["c"]
    elif variant == "layer1":
        def factory(ids):
            holder["c"] = ScriptedViaInterface(ids, decision_period_s=0.5)
            return holder["c"]
    elif variant == "layer2_shift":
        def factory(ids):
            holder["c"] = PhaseShiftedScripted(ids, decision_period_s=period)
            return holder["c"]
    elif variant == "hand":
        def factory(ids):
            holder["c"] = CheckedNet(HandBlue(), ids, decision_period_s=period)
            return holder["c"]
    elif variant == "silent":
        ids = blue_ids_of(p)
        silent_ids = [ids[s - 1] for s in SILENT_SLOTS]
        prepare = _silent_prepare(silent_ids)

        def factory(ids_):
            net = CheckedNet(HandBlue(silent_slots=SILENT_SLOTS), ids_,
                                        decision_period_s=period)
            holder["c"] = net
            holder["a"] = SilentAudit(net, silent_ids)
            return holder["a"]
        extra["silent_ids"] = silent_ids
    elif variant == "random":
        def factory(ids):
            holder["c"] = CheckedNet(RandomMLPPolicy(policy_seed), ids,
                                                decision_period_s=period)
            return holder["c"]
    else:
        raise ValueError(variant)
    t0 = time.perf_counter()
    r = run_presentation(p, record=record, blue_factory=factory, prepare=prepare)
    extra["wall_s"] = time.perf_counter() - t0
    c = holder.get("c")
    if isinstance(c, ScriptedViaInterface):
        extra.update(n_encoded=c.n_encoded, n_inexact=c.n_inexact, n_fire_miss=c.n_fire_miss)
    if isinstance(c, NetworkBlueController):
        extra["n_decisions"] = c.n_decisions
        extra["radar_denied"] = sum(m.radar_denied for m in c.mem.values())
    if isinstance(c, CheckedNet):
        extra["obs_checked"], extra["obs_violations"] = c.obs_checked, c.obs_violations
    if "a" in holder:
        extra["audit"] = holder["a"].report()
    return r, extra


def summarize(seed: int, variant: str, r, extra: dict) -> dict:
    shots = r.shots
    launches = [e for e in r.events if e["type"] == "launch"]
    blue = set(blue_ids_of(sample_presentation(seed)))
    sil = set(extra.get("silent_ids", []))
    d = {"seed": seed, "variant": variant, "time_s": r.time_s, "end_reason": r.end_reason,
         "blue_kills": r.blue_kills, "red_kills": r.red_kills,
         "blue_shots": sum(1 for s in shots if s["coalition"] == "Blue"),
         "blue_hits": sum(1 for s in shots if s["coalition"] == "Blue" and s["outcome"] == "hit"),
         "red_shots": sum(1 for s in shots if s["coalition"] == "Red"),
         "blue_remote_launches": sum(1 for e in launches
                                     if e["shooter"] in blue and e["remote_source"]),
         "silent_remote_launches": sum(1 for e in launches
                                       if e["shooter"] in sil and e["remote_source"]),
         "silent_own_launches": sum(1 for e in launches
                                    if e["shooter"] in sil and not e["remote_source"]),
         "support_handoffs": sum(1 for e in r.events if e.get("type") == "support_handoff"),
         "radar_events": sum(1 for e in r.events if e.get("type") == "radar")}
    for k in ("wall_s", "n_encoded", "n_inexact", "n_fire_miss", "n_decisions",
              "radar_denied", "audit", "obs_checked", "obs_violations"):
        if k in extra:
            d[k] = extra[k]
    return d


def _job(a):
    seed, variant = a
    r, extra = run_variant(seed, variant)
    return summarize(seed, variant, r, extra)


def run_all(n: int = 100, workers: int = 8, master_seed: int = 2026,
            variants=VARIANTS) -> Dict[str, List[dict]]:
    seeds = stats_seeds(n, master_seed)
    jobs = [(s, v) for v in variants for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=2))
    out: Dict[str, List[dict]] = {v: [] for v in variants}
    for r in res:
        out[r["variant"]].append(r)
    return out


# ---------------------------------------------------------------------------
def layer1_job(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        r0, _ = run_variant(seed, "scripted", record=True)
        r1, ex = run_variant(seed, "layer1", record=True)
        a, b = Path(td) / "a.acmi", Path(td) / "b.acmi"
        export_presentation_acmi(r0, a)
        export_presentation_acmi(r1, b)
        same = filecmp.cmp(a, b, shallow=False)
    s0 = [(s["shooter"], s["target"], s["outcome"]) for s in r0.shots]
    s1 = [(s["shooter"], s["target"], s["outcome"]) for s in r1.shots]
    return {"seed": seed, "acmi_identical": same,
            "result_identical": (r0.time_s, r0.blue_kills, r0.red_kills, r0.end_reason, s0)
            == (r1.time_s, r1.blue_kills, r1.red_kills, r1.end_reason, s1),
            "events_identical": r0.events == r1.events,
            **{k: ex[k] for k in ("n_encoded", "n_inexact", "n_fire_miss")}}


def run_layer1(n: int = 20, workers: int = 8, master_seed: int = 2026) -> List[dict]:
    seeds = stats_seeds(n, master_seed)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(layer1_job, seeds))


# ---------------------------------------------------------------------------
def _mean(runs, k):
    return float(np.mean([r[k] for r in runs]))


def row(label: str, runs: List[dict]) -> str:
    n = len(runs)
    bs = sum(r["blue_shots"] for r in runs)
    bh = sum(r["blue_hits"] for r in runs)
    endc = Counter(r["end_reason"] for r in runs)
    ends = " ".join(f"{END_ABBR[k]} {endc[k]}" for k in END_REASONS if endc[k])
    return (f"| {label} | {n} | {_mean(runs, 'blue_kills'):.2f} | {_mean(runs, 'red_kills'):.2f} | "
            f"{bs / n:.2f} | {100 * bh / bs if bs else 0:.0f} % | {_mean(runs, 'red_shots'):.2f} | "
            f"{np.median([r['time_s'] for r in runs]):.0f} | {ends} |")


HEADER = ("| variant | n | Blue kills | Blue losses | Blue shots | Blue hit rate | Red shots | "
          "median dur s | end reasons |\n|---|---|---|---|---|---|---|---|---|")


def tol_check(ref: List[dict], got: List[dict], rel: float, abs_: Optional[float]) -> List[str]:
    L = []
    for k, lab in (("blue_kills", "Blue kills"), ("red_kills", "Blue losses"),
                   ("blue_shots", "Blue shots")):
        a, b = _mean(ref, k), _mean(got, k)
        d = b - a
        rd = d / a if a else float("inf")
        ok = abs(rd) <= rel + 1e-12 or (abs_ is not None and abs(d) <= abs_ + 1e-12)
        L.append(f"- {lab}: {a:.2f} -> {b:.2f} ({100 * rd:+.1f} %, {d:+.2f} abs) "
                 f"{'PASS' if ok else 'FAIL'}")
    return L


def end_check(ref, got, tol=5) -> str:
    ca = Counter(r["end_reason"] for r in ref)
    cb = Counter(r["end_reason"] for r in got)
    diffs = {k: cb[k] - ca[k] for k in END_REASONS}
    ok = all(abs(v) <= tol for v in diffs.values())
    return (f"- End-reason counts (within ±{tol} per class): "
            + ", ".join(f"{END_ABBR[k]} {ca[k]}->{cb[k]}" for k in END_REASONS)
            + f" {'PASS' if ok else 'FAIL'}")


def format_report(runs: Dict[str, List[dict]], l1: List[dict]) -> str:
    L = ["# Spec 5 Q adapter test (100 spec 4 stats seeds, master 2026)", "",
         HEADER]
    for v in runs:
        L.append(row(LABELS[v], runs[v]))
    L += ["", "Per-engagement means; hit rate = Blue hits / Blue shots over all fights; "
          "Blue losses = Red kills. End reasons: Bdead = every Blue dead, Rdead = every Red "
          "dead, Wch = both Winchester, Rdep = every live Red departed, cap = 360 s.", ""]
    sc = runs["scripted"]
    L += ["## Layer 1 (0.5 s decisions, encode -> decode of the script, 20 seeds)", ""]
    n_id = sum(x["acmi_identical"] for x in l1)
    enc = sum(x["n_encoded"] for x in l1)
    inx = sum(x["n_inexact"] for x in l1)
    miss = sum(x["n_fire_miss"] for x in l1)
    L += [f"- Byte-identical ACMI: {n_id}/{len(l1)} "
          f"{'PASS' if n_id == len(l1) else 'FAIL'}",
          f"- Identical SimResult (time, kills, losses, end reason, every shot): "
          f"{sum(x['result_identical'] for x in l1)}/{len(l1)}; identical event log: "
          f"{sum(x['events_identical'] for x in l1)}/{len(l1)}",
          f"- Fire targets missing from the jet's contact slots: {miss} (must be 0)",
          f"- Commands encoded: {enc}; not reproducible bit-exactly by the decoder: {inx} "
          f"({100 * inx / max(enc, 1):.1f} %; all within 8 ulp, see spec 5 Implementation)",
          "", "## Layer 2 (1 s hold) vs scripted: ±10 % or ±0.2 absolute", ""]
    L += tol_check(sc, runs["layer2"], 0.10, 0.2)
    L.append(end_check(sc, runs["layer2"]))
    L += ["", "## Layer 3 (HandBlue, obs only) vs scripted: ±15 %", ""]
    L += tol_check(sc, runs["hand"], 0.15, None)
    ob = [r for v in ("hand", "silent") for r in runs[v]]
    L += ["", f"Observation range check (test plan 1, 100 presentations x every decision x "
          f"every live jet, hand + silent variants): {sum(r['obs_checked'] for r in ob)} "
          f"vectors, {sum(r['obs_violations'] for r in ob)} with a non-finite or "
          f"out-of-range input (must be 0)."]
    L += ["", "## Radar-silent variant (B2 + B3 radar off all fight)", ""]
    s = runs["silent"]
    aud = Counter()
    for r in s:
        aud.update(r["audit"])
    L += [f"- Remote launches by silent jets: {sum(r['silent_remote_launches'] for r in s)} "
          f"in {sum(1 for r in s if r['silent_remote_launches'])}/{len(s)} fights; own-FC "
          f"launches by silent jets: {sum(r['silent_own_launches'] for r in s)} (must be 0)",
          f"- Red RWR cues (search/lock/support) from silent jets, summed over every step: "
          f"{aud['red_cues']} (must be 0); Red spec 1 RWR track components on silent jets: "
          f"{aud['red_rwr_components']} (must be 0)",
          f"- Silent jets' own radar track components: {aud['own_radar_components']}, own FC "
          f"tracks: {aud['own_fc']}, steps emitting: {aud['emitting_steps']} (all must be 0)",
          f"- Red missile-active cues from missiles fired by silent jets (the missile's own "
          f"seeker, expected): {aud['missile_cues']}",
          f"- Support handoffs (all fights): {sum(r['support_handoffs'] for r in s)}",
          f"- Remote launches by all Blue jets: silent variant "
          f"{sum(r['blue_remote_launches'] for r in s)}, hand "
          f"{sum(r['blue_remote_launches'] for r in runs['hand'])}, scripted "
          f"{sum(r['blue_remote_launches'] for r in sc)}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def silent_replay(seeds: List[int], out_dir: Path = OUT) -> dict:
    """First seed whose silent variant has a silent jet launching on a wingman's
    FC track that later hits or is handed over; exports the ACMI + timeline."""
    best = None
    for seed in seeds:
        r, ex = run_variant(seed, "silent", record=True)
        sil = set(ex["silent_ids"])
        rem = [e for e in r.events if e["type"] == "launch" and e["shooter"] in sil
               and e["remote_source"]]
        if not rem:
            continue
        hit = any(s["shooter"] in sil and s["outcome"] == "hit" for s in r.shots)
        if best is None or (hit and not best[3]):
            best = (seed, r, ex, hit)
        if hit:
            break
    if best is None:
        return {}
    seed, r, ex, _ = best
    sil = set(ex["silent_ids"])
    path = out_dir / f"spec5_radar_silent_seed{seed}.acmi"
    export_presentation_acmi(r, path, title=f"Spec 5 radar-silent HandBlue, seed {seed} "
                                            f"({', '.join(sorted(sil))} silent)")
    p = sample_presentation(seed)
    L = [f"# Radar-silent replay: presentation seed {seed}", "",
         f"Presentation: {p.summary()}", f"Silent jets (radar off all fight): "
         f"{', '.join(sorted(sil))}. Blue = HandBlue on the observation only, 1 s decisions.",
         f"Result: {r.end_reason} at {r.time_s:.1f} s; Blue kills {r.blue_kills}, Blue "
         f"losses {r.red_kills}; Blue shots {sum(1 for s in r.shots if s['coalition'] == 'Blue')}, "
         f"Red shots {sum(1 for s in r.shots if s['coalition'] == 'Red')}. Audit: {ex['audit']}",
         f"ACMI: {path.name}", "", "| t (s) | event |", "|---|---|"]
    shot_by_mid = {s.get("missile"): s for s in r.shots}
    for e in r.events:
        ty = e.get("type")
        if ty == "launch":
            s = shot_by_mid.get(e["missile"], {})
            src = (f"on {e['remote_source']}'s FC track (remote, shooter radar "
                   f"{'SILENT' if e['shooter'] in sil else 'on'})" if e["remote_source"]
                   else "own FC")
            rng = s.get("launch_range_m")
            L.append(f"| {e['t']:.1f} | {e['shooter']} launches {e['missile']} at {e['target']} "
                     f"{src}{f', range {rng / 1852:.1f} nm' if rng else ''}"
                     + (" -> " + str(s.get("outcome")) if s else "") + " |")
        elif ty in ("support_handoff", "support_lost", "support_dropped", "autonomous",
                    "hit", "miss", "kill", "radar", "defend", "depart", "winchester"):
            L.append(f"| {e['t']:.1f} | {ty}: {e.get('text') or e.get('target', '')} |")
    (out_dir / "radar_silent_timeline.md").write_text("\n".join(L) + "\n")
    return {"seed": seed, "acmi": str(path), "timeline": "\n".join(L)}


def random_smoke(n: int = 4, policy_seed: int = 7) -> str:
    seeds = stats_seeds(n, 99)
    L = ["# Random-weights network smoke run (231-64-64-13 MLP, tanh, seed "
         f"{policy_seed}, {RandomMLPPolicy(policy_seed).n_params} parameters, 1 s decisions)", "",
         "| seed | end | dur s | decisions | Blue shots | Blue kills | Blue losses | radar events | "
         "radar flips refused (dwell) | wall s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        r, ex = run_variant(s, "random", policy_seed=policy_seed)
        d = summarize(s, "random", r, ex)
        L.append(f"| {s} | {r.end_reason} | {r.time_s:.0f} | {ex['n_decisions']} | "
                 f"{d['blue_shots']} | {r.blue_kills} | {r.red_kills} | {d['radar_events']} | "
                 f"{ex['radar_denied']} | {ex['wall_s']:.2f} |")
    L += ["", "Proves the full path runs end to end: World.blue_view -> build_observation "
          "(231) -> policy -> decode (13) -> commands / radar / pair -> World gates."]
    return "\n".join(L)


def _tjob(a):
    seed, variant, period = a
    r, ex = run_variant(seed, variant, period=period)
    return ex["wall_s"], r.time_s


def timing(n_serial: int = 12, n_parallel: int = 96, workers: int = 8,
           master_seed: int = 77) -> Dict[str, dict]:
    seeds = stats_seeds(n_serial + n_parallel, master_seed)
    out = {}
    for label, variant, period in (("scripted", "scripted", 1.0),
                                   ("random MLP 1 s", "random", 1.0),
                                   ("random MLP 0.5 s", "random", 0.5)):
        t0 = time.perf_counter()
        sims = [_tjob((s, variant, period))[1] for s in seeds[:n_serial]]
        serial = (time.perf_counter() - t0) / n_serial
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_tjob, [(s, variant, period) for s in seeds[n_serial:]], chunksize=2))
        par = time.perf_counter() - t0
        out[label] = {"serial": serial, "mean_sim": float(np.mean(sims)), "par_wall": par,
                      "par_per": par / n_parallel}
    return out


def format_timing(t: Dict[str, dict], pop: int = 50, n_eval: int = 24, bench: int = 64,
                  workers: int = 8, n_parallel: int = 96) -> str:
    fights = pop * n_eval + bench
    L = ["# Spec 5 timing (this box, 8 workers)", "",
         "| Blue | serial s / engagement | mean simulated s | pool wall s / engagement | "
         f"per generation ({pop} x {n_eval} + {bench} = {fights}) | generations / 10 h |",
         "|---|---|---|---|---|---|"]
    for k, v in t.items():
        g = fights * v["par_per"]
        L.append(f"| {k} | {v['serial']:.2f} | {v['mean_sim']:.0f} | {v['par_per']:.3f} | "
                 f"{g:.0f} s ({g / 60:.1f} min) | {36000 / g:.0f} |")
    L += ["", f"Serial: 12 engagements in one process; pool: {n_parallel} engagements on "
          f"{workers} workers (seeds from master 77, as spec 4). Random MLP cost includes "
          "blue_view + observation + forward + decode for 4 jets per decision. Note that a "
          "random policy's fights differ in length from the scripted ones, so the per-"
          "engagement cost also reflects different simulated durations (column 3)."]
    return "\n".join(L)


def layer2_noise_check(runs: Dict[str, List[dict]], workers: int = 8,
                       n_extra: int = 300, master_seed: int = 2026) -> str:
    """Is a layer 2 end-reason difference a decision-rate effect or noise?
    (a) the phase-shifted 1 s replicate on the same 100 seeds; (b) scripted,
    layer 2 and the replicate on the next ``n_extra`` stats seeds."""
    n = len(runs["scripted"])
    seeds = stats_seeds(n + n_extra, master_seed)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        sh = list(ex.map(_job, [(s, "layer2_shift") for s in seeds[:n]], chunksize=2))
        big = list(ex.map(_job, [(s, v) for v in ("scripted", "layer2", "layer2_shift")
                                 for s in seeds[n:]], chunksize=2))
    B = {v: [r for r in big if r["variant"] == v] for v in ("scripted", "layer2", "layer2_shift")}
    sc = {r["seed"]: r["end_reason"] for r in runs["scripted"]}
    l2 = {r["seed"]: r["end_reason"] for r in runs["layer2"]}
    flips = sum(sc[s] != l2[s] for s in sc)
    flips_sh = sum(sc[r["seed"]] != r["end_reason"] for r in sh)
    L = ["# Layer 2 end-reason check: decision-rate effect or noise?", "",
         f"## Same {n} seeds", "", HEADER,
         row(LABELS["scripted"], runs["scripted"]), row(LABELS["layer2"], runs["layer2"]),
         row("Layer 2 replicate: 1 s hold, decisions half a period later", sh), "",
         f"Fights whose end reason differs from scripted: layer 2 {flips}/{n}, replicate "
         f"{flips_sh}/{n}.", "", "Replicate vs scripted:"]
    L += tol_check(runs["scripted"], sh, 0.10, 0.2) + [end_check(runs["scripted"], sh)]
    L += ["", "Layer 2 vs its own replicate (same treatment):",
          end_check(runs["layer2"], sh), "",
          f"## Next {n_extra} stats seeds (fresh sample)", "", HEADER]
    for v, lab in (("scripted", LABELS["scripted"]), ("layer2", LABELS["layer2"]),
                   ("layer2_shift", "Layer 2 replicate (half-period phase)")):
        L.append(row(lab, B[v]))
    tol = round(5 * n_extra / 100)
    L += ["", f"Layer 2 vs scripted (end-reason tolerance scaled to ±{tol} for n = {n_extra}):"]
    L += tol_check(B["scripted"], B["layer2"], 0.10, 0.2) + [end_check(B["scripted"], B["layer2"], tol)]
    return "\n".join(L)


def main(n: int = 100, workers: int = 8, out_dir: Path = OUT, do_timing: bool = True,
         noise_check: bool = True) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    runs = run_all(n, workers)
    l1 = run_layer1(20, workers)
    rep = format_report(runs, l1)
    (out_dir / "adapter_results.md").write_text(rep + "\n")
    (out_dir / "adapter_runs.json").write_text(json.dumps({"runs": runs, "layer1": l1},
                                                          indent=1, default=str))
    # replay: prefer seeds where the silent variant made remote launches with a hit
    cand = [r["seed"] for r in runs["silent"] if r["silent_remote_launches"]]
    rp = silent_replay(cand or stats_seeds(n, 2026))
    nz = layer2_noise_check(runs, workers) if noise_check else ""
    if nz:
        (out_dir / "layer2_noise_check.md").write_text(nz + "\n")
    sm = random_smoke()
    (out_dir / "random_network_smoke.md").write_text(sm + "\n")
    parts = [rep, "", nz, "", rp.get("timeline", "(no silent remote launch found)"), "", sm]
    if do_timing:
        tm = format_timing(timing(workers=workers))
        (out_dir / "timing.md").write_text(tm + "\n")
        parts += ["", tm]
    parts.append(f"\nTotal wall {time.perf_counter() - t0:.0f} s")
    return "\n".join(parts)
