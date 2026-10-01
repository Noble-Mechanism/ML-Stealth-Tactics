"""Play package: run evolve-net until stopped, auto-resuming.

``python -m stealth_tactics overnight -o runs/overnight`` (or
``scripts/overnight.sh``). Every generation writes a checkpoint; a restart
with the same command resumes from the latest one (byte-identical to an
uninterrupted run). Every ``--every`` generations (default 10) it refreshes
``<out>/progress/``: fitness chart PNG, champion best / worst ACMIs (+ weights,
replayable with ``replay-champion <out>/progress``), the hall of fame (each
cell replayable the same way), ``hall_of_fame.md``, ``progress.md`` and the
per-generation history (``history.csv`` / ``history.json``, spec 6b; the run
directory has the same files updated every generation).
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import numpy as np

from .checkpoint import write_json
from stealth_tactics.fitness import merge_outcome_counts
from .neuroga import NeuroGA, _summ, write_history_files


def _chart(history, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    g = [r["generation"] for r in history]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=110, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(g, [r["best_fitness"] for r in history], label="best (eval set)")
    ax.plot(g, [r["mean_fitness"] for r in history], label="mean (eval set)")
    if any(r.get("median_fitness") is not None for r in history):
        ax.plot(g, [r.get("median_fitness") for r in history], alpha=0.6, label="median (eval set)")
    if any(r.get("best_ma") is not None for r in history):
        ax.plot(g, [r.get("best_ma") for r in history], "--", label="best, moving average")
    ax.plot(g, [r["champion_bench"] for r in history], "k:", label="champion (benchmark)")
    first = {"immig": True, "stag": True, "champ": True}
    for r in history:
        if r.get("immigrants"):
            ax.axvline(r["generation"], color="g", alpha=0.3,
                       label="immigration" if first["immig"] else None)
            first["immig"] = False
        if r.get("stagnation_trigger") or r.get("boosted"):
            ax.axvline(r["generation"], color="r", ls="--", alpha=0.5,
                       label="stagnation trigger" if first["stag"] else None)
            first["stag"] = False
        dec = r.get("champion_decision") or {}
        if dec.get("accepted") and dec.get("champion_id") is not None:
            ax.plot(r["generation"], r["champion_bench"], "k^", ms=5,
                    label="new champion" if first["champ"] else None)
            first["champ"] = False
    ax.set_ylabel("fitness (mean - std_coef x std)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2.plot(g, [r["mean_sigma"] for r in history], label="mean sigma")
    if any("min_sigma" in r for r in history):
        ax2.fill_between(g, [r.get("min_sigma", r["mean_sigma"]) for r in history],
                         [r.get("max_sigma", r["mean_sigma"]) for r in history], alpha=0.2,
                         label="min-max sigma")
    ax2.set_yscale("log")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("mutation sigma")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_progress(ga: NeuroGA, out: Path, started: float, gens_this_session: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h = ga.history
    has_png = _chart(h, out / "fitness.png")
    write_history_files(h, out)
    ga._store(ga.champion, out, "champion", True)
    hof_lines = ["# Hall of fame", "",
                 "Cells: radar-off share (r0 < 1/3, r1 < 2/3, r2) x launch range / Rmax "
                 "(l0 < 0.6, l1 < 0.85, l2). Replay: `python -m stealth_tactics "
                 "replay-champion <cell dir>`.", "",
                 "| cell | id | origin | gen | benchmark fitness | kills | losses | shots | "
                 "radar-off share | launch r/Rmax |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cell in sorted(ga.hof):
        e = ga.hof[cell]
        ga._store(e, out / "hall_of_fame" / cell, "champion", True)
        s = _summ(e["bench_summaries"])
        raw = e["bench_raw"]
        hof_lines.append(f"| {cell} | {e['genome'].lineage.get('id')} | "
                         f"{e['genome'].lineage.get('origin')} | {e['gen']} | "
                         f"{e['benchmark_fitness']:.1f} | {s['kills']:.2f} | {s['losses']:.2f} | "
                         f"{s['shots']:.1f} | {raw['radar_off_share']:.2f} | "
                         f"{raw['launch_r_rmax'] if raw['launch_r_rmax'] is None else round(raw['launch_r_rmax'], 2)} |")
    (out / "hall_of_fame.md").write_text("\n".join(hof_lines) + "\n", encoding="utf-8")
    ch = ga.champion
    cs = _summ(ch["bench_summaries"])
    timing = [json.loads(l) for l in (ga.run_dir / "timing.jsonl").read_text().splitlines()[-10:]]
    spg = float(np.mean([t["total_s"] for t in timing])) if timing else float("nan")
    last = h[-1]
    # Spec 8 C: champion-fight missile outcome rollup
    outcomes = merge_outcome_counts([s.get("missile_outcomes") or {}
                                     for s in ch.get("bench_summaries") or []])

    def _out_table(title, d):
        if not d:
            return [f"### {title}", "", "_none_", ""]
        labels = sorted(d)
        rows = ["| label | count |", "|---|---|"] + [f"| {k} | {d[k]} |" for k in labels]
        return [f"### {title}", ""] + rows + [""]

    lines = ["# Progress", "",
             f"- generations done: {ga.gen} (this session {gens_this_session}, "
             f"{(time.time() - started) / 60:.1f} min)",
             f"- seconds per generation (last {len(timing)}): {spg:.0f}",
             f"- last generation: best {last['best_fitness']:.1f}, mean {last['mean_fitness']:.1f}, "
             f"best origin {last['best_origin']}",
             f"- champion: id {ch['genome'].lineage.get('id')} ({ch['genome'].lineage.get('origin')}), "
             f"generation {ch['gen']}, benchmark fitness {ch['benchmark_fitness']:.1f}; per "
             f"benchmark fight: kills {cs['kills']:.2f}, losses {cs['losses']:.2f}, "
             f"shots {cs['shots']:.1f}, hit rate {cs['hit_rate']:.2f}",
             f"- hall of fame cells: {', '.join(sorted(ga.hof)) or 'none'}",
             f"- chart: {'fitness.png' if has_png else 'not written (pip install .[plot])'}",
             "- full per-generation history: `history.csv` / `history.json` (best, mean, median, "
             "moving average, sigma, immigration / stagnation events, champion decisions)",
             "- replay the champion: `python -m stealth_tactics replay-champion "
             f"{out}`", "",
             "## Champion missile outcomes (benchmark fights)", ""]
    lines += _out_table("Red as target (Blue shots)", outcomes.get("red_as_target") or {})
    lines += _out_table("Blue as target (Red shots)", outcomes.get("blue_as_target") or {})
    ev = [e for e in ga.events if e.get("type") in ("stagnation", "immigration", "champion")]
    lines += ["## Recent events (stagnation, immigration, champion decisions)", ""]
    lines += [f"- gen {e['gen']}: {e['text']}" for e in ev[-15:]] or ["_none_"]
    lines += [""]

    def _f(v, fmt="{:.1f}"):
        return "" if v is None else fmt.format(v)

    def _dec(r):
        d = r.get("champion_decision") or {}
        if d.get("challenger_id") is None:
            return ""
        s_ = f"{d['challenger_id']} {_f(d.get('challenger_bench'))}"
        if d.get("mean_diff") is not None:
            s_ += f" (d {d['mean_diff']:.1f}, k*SE {d['margin']:.1f})"
        return s_ + (" accepted" if d.get("accepted") else " kept champion")
    lines += ["| gen | best | mean | median | best MA | mean sigma | champion benchmark | "
              "challenger | immigrants |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['generation']} | {r['best_fitness']:.1f} | {r['mean_fitness']:.1f} | "
              f"{_f(r.get('median_fitness'))} | {_f(r.get('best_ma'))} | "
              f"{r['mean_sigma']:.4f} | {r['champion_bench']:.1f} | {_dec(r)} | "
              f"{r.get('immigrants') or ''} {r.get('immigrant_reason') or ''} |" for r in h[-30:]]
    (out / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_overnight(cfg, out, workers: int, clone_path=None, every: int = 10,
                  gens=None, log=print) -> NeuroGA:
    out = Path(out)
    ga = NeuroGA(cfg, out, workers=workers, clone_path=clone_path, log=log)
    write_json(out / "run_config.json", {"config": cfg.to_dict(), "config_hash": ga.hash,
                                         "clone": clone_path})
    started = time.time()
    session = {"n": 0}

    def _term(signum, frame):              # systemd stop / kill: leave cleanly
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    def hook(g, row):
        session["n"] += 1
        if g.gen % every == 0:
            write_progress(g, out / "progress", started, session["n"])
            log(f"progress written at generation {g.gen} -> {out / 'progress'}")
    has_ckpt = bool(sorted((out / "checkpoints").glob("ckpt_g*.json")))
    try:
        ga.run(gens if gens else 10 ** 9, resume=has_ckpt, finalize=False, on_generation=hook)
    except KeyboardInterrupt:
        log(f"stopped at generation {ga.gen}; the same command resumes from the last checkpoint")
    return ga
