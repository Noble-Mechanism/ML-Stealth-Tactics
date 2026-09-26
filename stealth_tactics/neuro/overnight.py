"""Play package: run evolve-net until stopped, auto-resuming.

``python -m stealth_tactics overnight -o runs/overnight`` (or
``scripts/overnight.sh``). Every generation writes a checkpoint; a restart
with the same command resumes from the latest one (byte-identical to an
uninterrupted run). Every ``--every`` generations (default 10) it refreshes
``<out>/progress/``: fitness chart PNG, champion best / worst ACMIs (+ weights,
replayable with ``replay-champion <out>/progress``), the hall of fame (each
cell replayable the same way), ``hall_of_fame.md`` and ``progress.md``.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import numpy as np

from .checkpoint import write_json
from .neuroga import NeuroGA, _summ


def _chart(history, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    g = [r["generation"] for r in history]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    ax.plot(g, [r["best_fitness"] for r in history], label="best (eval set)")
    ax.plot(g, [r["mean_fitness"] for r in history], label="mean (eval set)")
    ax.plot(g, [r["champion_bench"] for r in history], "k:", label="champion (benchmark)")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness v1 (mean - 0.5 std)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_progress(ga: NeuroGA, out: Path, started: float, gens_this_session: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h = ga.history
    has_png = _chart(h, out / "fitness.png")
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
             "- replay the champion: `python -m stealth_tactics replay-champion "
             f"{out}`", "",
             "| gen | best | mean | champion benchmark |", "|---|---|---|---|"]
    lines += [f"| {r['generation']} | {r['best_fitness']:.1f} | {r['mean_fitness']:.1f} | "
              f"{r['champion_bench']:.1f} |" for r in h[-30:]]
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
