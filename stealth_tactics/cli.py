"""Command-line interface: python -m stealth_tactics ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stealth_tactics.scenarios.loader import load_scenario
from stealth_tactics.ga.evolution import GeneticAlgorithm, GAConfig
from stealth_tactics.acmi.exporter import ACMIExporter
from stealth_tactics.tactics.genome import TacticsGenome


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_evolve(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    if not scenario_path.is_file():
        # try relative to project scenarios/
        alt = _project_root() / "scenarios" / args.scenario
        if alt.is_file():
            scenario_path = alt
        else:
            print(f"Scenario not found: {args.scenario}", file=sys.stderr)
            return 1

    scenario = load_scenario(scenario_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_gen(gen: int, evaluated: list) -> None:
        best = evaluated[0]
        mean = sum(e.fitness for e in evaluated) / len(evaluated)
        print(
            f"Gen {gen:3d}  best={best.fitness:7.2f}  mean={mean:7.2f}  "
            f"kills={best.sim.blue_kills}  losses={best.sim.red_kills}  "
            f"winner={best.sim.winner}"
        )

    ga_cfg = GAConfig(
        population=args.pop,
        generations=args.gens,
        seed=args.seed,
        sim_dt=args.dt,
        sim_max_time_s=args.max_time,
        elite_count=max(1, args.pop // 10),
    )
    ga = GeneticAlgorithm(scenario, ga_cfg, on_generation=on_gen)
    print(f"Evolving tactics: pop={ga_cfg.population} gens={ga_cfg.generations} "
          f"scenario={scenario.name!r} seed={ga_cfg.seed}")
    best = ga.run()

    # Save genome + history
    genome_path = out_dir / "best_genome.json"
    genome_path.write_text(json.dumps(best.genome.to_dict(), indent=2), encoding="utf-8")
    hist_path = out_dir / "history.json"
    hist_path.write_text(json.dumps(ga.history, indent=2), encoding="utf-8")

    # ACMI
    acmi_name = args.acmi or "best_engagement.txt.acmi"
    acmi_path = out_dir / acmi_name
    exporter = ACMIExporter(title=f"{scenario.name} — best genome")
    exporter.export(best.sim.frames, acmi_path)

    evo_fit = ga.evolution_best_fitness
    if evo_fit is None:
        evo_fit = best.fitness
    match = abs(best.fitness - evo_fit) <= 1e-6
    print(f"\nEvolution best fitness: {evo_fit:.2f}")
    print(
        f"Recorded engagement: fitness={best.fitness:.2f}  "
        f"kills={best.sim.blue_kills}  losses={best.sim.red_kills}  "
        f"winner={best.sim.winner}  time={best.sim.time_s:.1f}s  "
        f"seed_offset={best.seed_offset}  "
        f"{'MATCHES evolution best' if match else 'MISMATCH'}"
    )
    print(f"  genome -> {genome_path}")
    print(f"  ACMI   -> {acmi_path}")
    print(f"  Open in TacView: File → Open → {acmi_path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run a single engagement with default or provided genome."""
    scenario_path = Path(args.scenario)
    if not scenario_path.is_file():
        alt = _project_root() / "scenarios" / args.scenario
        scenario_path = alt if alt.is_file() else scenario_path
    scenario = load_scenario(scenario_path)

    if args.genome:
        genome = TacticsGenome.from_dict(json.loads(Path(args.genome).read_text()))
    else:
        genome = TacticsGenome()

    ga = GeneticAlgorithm(scenario, GAConfig(seed=args.seed, sim_max_time_s=args.max_time))
    result = ga.evaluate(genome, record=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    acmi_path = out_dir / (args.acmi or "sim.txt.acmi")
    ACMIExporter(title=scenario.name).export(result.sim.frames, acmi_path)
    print(f"fitness={result.fitness:.2f} winner={result.sim.winner} "
          f"kills={result.sim.blue_kills} losses={result.sim.red_kills}")
    print(f"ACMI -> {acmi_path}")
    return 0


def cmd_sensors_table(args: argparse.Namespace) -> int:
    """Print (and save) the Spec 1 first-detection range table."""
    from stealth_tactics.analysis.sensor_table import build_table, format_table

    rows = build_table(seeds=args.seeds)
    text = format_table(rows, args.seeds)
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nSaved -> {out}")
    return 0


def cmd_sensor_replays(args: argparse.Namespace) -> int:
    """Scripted 1v1 head-on vs beam-at-30-NM replays with sensor events."""
    import numpy as np
    from stealth_tactics.analysis.replays import (
        run_replay, describe, export, first_detect_stats,
        run_beam_then_hot, beam_then_hot_stats, format_bth_stats,
    )
    from stealth_tactics.sim.sensor_config import NM_M

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    which = args.scenario
    for beam, fname, name in ((False, "replay1_headon.txt.acmi", "headon"),
                              (True, "replay2_beam_at_30nm.txt.acmi", "beam")):
        if which not in ("all", name):
            continue
        res = run_replay(beam, seed=args.seed)
        path = export(res, out_dir / fname)
        report.append(describe(res))
        report.append(f"  ACMI -> {path}")
        if args.stats_seeds > 0:
            st = first_detect_stats(beam, seeds=args.stats_seeds)
            report.append(f"  First-event range over {args.stats_seeds} seeds "
                          "(median [p10-p90] NM):")
            for key in sorted(st):
                v = np.array(st[key]) / NM_M
                report.append(f"    {key:18} {np.median(v):5.1f} "
                              f"[{np.percentile(v, 10):5.1f}-{np.percentile(v, 90):5.1f}]"
                              f"  (n={len(v)})")
        report.append("")
    if which in ("all", "beam-then-hot"):
        bth_m = args.bth_beam_nm * NM_M
        res = run_beam_then_hot(seed=args.seed, beam_range_m=bth_m)
        path = export(res, out_dir / "replay3_beam45_then_hot.txt.acmi")
        report.append(describe(res))
        report.append(f"  ACMI -> {path}")
        if args.stats_seeds > 0:
            report.append(format_bth_stats(beam_then_hot_stats(args.stats_seeds, bth_m)))
        report.append("")
    text = "\n".join(report)
    print(text)
    log = out_dir / "replay_events.txt"
    if which == "all" or not log.exists():
        log.write_text(text + "\n", encoding="utf-8")
    else:  # single scenario: append its log to the existing file
        with log.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


def cmd_datalink_replays(args: argparse.Namespace) -> int:
    """Spec 2 replays: A launch-on-remote, B passive IRST triangulation."""
    from stealth_tactics.analysis.datalink_replays import (
        run_launch_on_remote, describe_a, hit_rate_a, run_triangulation,
        triangulation_table, export, run_lead_trail, describe_c, stats_lead_trail,
        comparison_table, LT_VARIANTS, shot_frac_table,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    if args.scenario in ("all", "launch-on-remote"):
        rep = run_launch_on_remote(seed=args.seed_a, shot_frac=args.shot_frac)
        p = export(rep.world, out_dir / "replayA_launch_on_remote.txt.acmi",
                   "Spec2 replay A - launch on remote + support handoff")
        report += [describe_a(rep), f"  ACMI -> {p}"]
        if args.stats_seeds > 0:
            hr = hit_rate_a(args.stats_seeds, shot_frac=args.shot_frac)
            n = hr["seeds"]
            f2 = (lambda v, fmt: fmt.format(v) if v is not None else "-")
            report.append(
                f"  {n}-seed outcomes: " + ", ".join(
                    f"{k}: {v}" for k, v in sorted(hr["outcomes"].items()))
                + f"  -> hit rate {hr['outcomes'].get('hit', 0) / n * 100:.0f}% "
                f"(shot at {args.shot_frac:.2f} x table Rmax for the 50 deg off-nose shot "
                f"geometry) (remote launch + handoff to B1 in "
                f"{hr['remote_launch_with_handoff']}/{n})")
            report.append(
                f"  medians: shot {f2(hr['median_shot_nm'], '{:.1f}')} NM, "
                f"{f2(hr['median_off_nose_deg'], '{:.0f}')} deg off the nose at launch "
                f"(table Rmax there {f2(hr['median_rmax_nm'], '{:.1f}')} NM); "
                f"missile Mach at end {f2(hr['median_mach_end'], '{:.2f}')}, "
                f"at fuze {f2(hr['median_mach_fuze'], '{:.2f}')}, "
                f"Pk at fuze {f2(hr['median_pk_final'], '{:.2f}')}")
        if args.shot_frac_sweep:
            fracs = [float(x) for x in args.shot_frac_sweep.split(",") if x.strip()]
            txt = shot_frac_table(fracs, args.stats_seeds if args.stats_seeds > 0 else 100)
            (out_dir / "launch_on_remote_shot_frac.txt").write_text(txt, encoding="utf-8")
            report += [txt, f"  -> {out_dir / 'launch_on_remote_shot_frac.txt'}"]
        report.append("")
    if args.scenario in ("all", "triangulation"):
        rep = run_triangulation(seed=args.seed_b)
        p = export(rep.world, out_dir / "replayB_passive_triangulation.txt.acmi",
                   "Spec2 replay B - passive IRST triangulation")
        table = triangulation_table(rep, every_s=5.0)
        (out_dir / "triangulation_error.txt").write_text(table, encoding="utf-8")
        tri_evs = [e for e in rep.world.sensor_events
                   if e["observer"] == "B1" and e["type"] in
                   ("radar_detect", "irst_detect", "irst_lost", "link_track", "fc_track")]
        tri_evs += [e for fr in rep.world.frames for e in fr.get("events", [])
                    if e.get("type") == "tri_fix"]
        tri_evs.sort(key=lambda e: e["t"])
        report.append(f"Replay B: passive triangulation (seed {args.seed_b})")
        for e in tri_evs:
            report.append(f"  t={e['t']:6.1f}s  {e['text']}  [{e['range_m'] / 1852:5.1f} NM]")
        report += [f"  ACMI -> {p}", f"  table -> {out_dir / 'triangulation_error.txt'}",
                   "", table]
    if args.scenario in ("all", "fc-lock"):
        from stealth_tactics.analysis.fc_lock import lock_stats
        txt = lock_stats(args.stats_seeds if args.stats_seeds > 0 else 100)
        (out_dir / "fc_lock_stability.txt").write_text(txt + "\n", encoding="utf-8")
        report += ["Blue 50 NM FC gate - lock stability", txt,
                   f"  -> {out_dir / 'fc_lock_stability.txt'}", ""]
    lt_marker = "##### Replay C: lead-trail #####"
    lt_report: list[str] = []
    if args.scenario in ("all", "lead-trail"):
        lt_report.append(lt_marker)
        rep = run_lead_trail(seed=args.seed_c, variant="support")
        p = export(rep.world, out_dir / "replayC_lead_trail.txt.acmi",
                   "Spec2 replay C - lead-trail: lead shoots + turns out, trail supports")
        lt_report += [describe_c(rep), f"  ACMI -> {p}", ""]
        rep_d = run_lead_trail(seed=args.seed_c, variant="delayed")
        p = export(rep_d.world, out_dir / "replayC_lead_trail_delayed.txt.acmi",
                   "Spec2 replay C (variant delayed) - lead waits for trail FC")
        lt_report += [describe_c(rep_d), f"  ACMI -> {p}", ""]
        rep_n = run_lead_trail(seed=args.seed_c, variant="no-support")
        p = export(rep_n.world, out_dir / "replayC_lead_trail_nosupport.txt.acmi",
                   "Spec2 replay C (variant no-support) - missile coasts after lead turns out")
        lt_report += [describe_c(rep_n), f"  ACMI -> {p}", ""]
        if args.stats_seeds > 0:
            stats = [stats_lead_trail(args.stats_seeds, v) for v in LT_VARIANTS]
            lt_report.append(f"Replay C comparison over seeds 0..{args.stats_seeds - 1}:")
            lt_report += [f"  {k:<11} = {d}" for k, d in LT_VARIANTS.items()]
            lt_report.append(comparison_table(stats))
            lt_report.append(
                "  gap = support lapsed before autonomy (missile coasts on the extrapolated "
                "aim point; Spec 2b); med gap = cumulative unsupported time; Pk factor = "
                "coast factor at the active point, exp(-(e/2 km)^2) x (1 - 0.15 min(t_gap/40 s,"
                " 1)) (Spec 3a fix); lost:timeout = > 40 s continuously unsupported; lost:basket "
                "= aim point > 5 km from the true target at 15 NM; med M@fuze / med Pk = "
                "missile Mach and final Pk (0.60 x factors x endgame energy) over fuzed "
                "shots; defeat_* = kinematic defeat (Spec 3a).")
        lt_report.append("")
    text = "\n".join(report)
    print("\n".join([text] + lt_report))
    ev_path = out_dir / "datalink_events.txt"
    if args.scenario == "all" or not ev_path.exists():
        ev_path.write_text("\n".join([text] + lt_report).strip() + "\n", encoding="utf-8")
    else:  # single scenario: replace a previous Replay C section / append
        old = ev_path.read_text(encoding="utf-8")
        if lt_report and lt_marker in old:
            old = old[:old.index(lt_marker)].rstrip() + "\n"
        body = "\n".join(x for x in [text] + lt_report if x)
        ev_path.write_text(old.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")
    return 0


def cmd_missile_sweep(args: argparse.Namespace) -> int:
    """Spec 3a: shot sweep, Rmax / Rne summary, calibration check, ACMI replays."""
    import time
    from stealth_tactics.analysis import missile_sweep as ms
    from stealth_tactics.analysis.missile_replays import (
        KINDS, describe, export, run_missile_replay)
    from stealth_tactics.sim.sensor_config import DEFAULT_SENSOR_CONFIG, NM_M

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kc = DEFAULT_SENSOR_CONFIG.missile_kinematics
    # Calibration check: 40,000 ft, shooter and target Mach 0.9
    targets = {"hot": 50.0, "beam": None, "turncold": 24.0, "cold": 20.0}
    cal = ["Calibration (40,000 ft, shooter M0.9, target M0.9 level; sim path, NM):",
           f"  cd_scale = {kc.cd_scale}, gravity/lift = {kc.gravity}, dt = {kc.dt_s} s",
           f"  {'target':<22} {'Rmax':>6} {'goal':>6} {'proto':>6}"]
    proto = {"hot": 49.0, "beam": 28.8, "turncold": 23.7, "cold": 20.4}
    for b in ms.BEHAVIORS:
        r = ms.sim_rmax_nm(40000.0, 0.9, b)
        goal = f"{targets[b]:.0f}" if targets[b] else "-"
        cal.append(f"  {ms.BEHAVIOR_LABEL[b]:<22} {r:6.1f} {goal:>6} {proto[b]:6.1f}")
    cal_txt = "\n".join(cal) + "\n"
    print(cal_txt)
    report = [cal_txt]
    if not args.skip_sweep:
        shots = ms.run_sweep()
        (out_dir / "missile_sweep.csv").write_text(ms.shots_csv(shots), encoding="utf-8")
        table = ms.format_shots(shots)
        hdr = ("Spec 3a missile sweep: WeaponModel fly-outs, co-altitude, target Mach 0.9; "
               "shooter flies straight at launch speed and supports to impact.\n"
               "outcome = hit / miss (seeded Pk roll) or defeat label; M@act = missile Mach "
               "when the seeker goes active (15 NM from target; launch Mach if launched inside 15 NM); a-pole / f-pole = "
               "shooter-target range (NM) at active / at fuze; Pk = final Pk (0.60 x f_E).\n\n")
        (out_dir / "missile_sweep.txt").write_text(hdr + table + "\n", encoding="utf-8")
        findings = ms.sweep_findings(shots)
        report.append(findings)
        print(findings)
    if not args.skip_summary:
        summ, _ = ms.summary_table()
        report.append(summ)
        print(summ)
    (out_dir / "rmax_summary.txt").write_text("\n".join(report), encoding="utf-8")
    if not args.skip_replays:
        names = {"headon": "missile_headon_hit.txt.acmi",
                 "drag": "missile_drag_defeat.txt.acmi",
                 "beam": "missile_beam_defeat.txt.acmi"}
        log = []
        for kind in KINDS:
            rep = run_missile_replay(kind, seed=args.seed)
            p = export(rep, out_dir / names[kind], f"Spec3a missile replay - {KINDS[kind]}")
            log += [describe(rep), f"  ACMI -> {p}", ""]
        (out_dir / "missile_replays.txt").write_text("\n".join(log), encoding="utf-8")
        print("\n".join(log))
    print(f"missile-sweep done in {time.time() - t0:.0f} s -> {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stealth_tactics",
        description="GA-evolved stealth fighter tactics simulation with TacView ACMI export",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ev = sub.add_parser("evolve", help="Run genetic algorithm evolution")
    p_ev.add_argument("-s", "--scenario", default="default_4v3.yaml")
    p_ev.add_argument("-p", "--pop", type=int, default=16)
    p_ev.add_argument("-g", "--gens", type=int, default=8)
    p_ev.add_argument("--seed", type=int, default=42)
    p_ev.add_argument("--dt", type=float, default=0.5)
    p_ev.add_argument("--max-time", type=float, default=240.0)
    p_ev.add_argument("-o", "--out", default=None,
                      help="Output directory (default: runs/demo)")
    p_ev.add_argument("--acmi", default="best_engagement.txt.acmi")
    p_ev.set_defaults(func=cmd_evolve)

    p_sim = sub.add_parser("simulate", help="Single engagement simulation")
    p_sim.add_argument("-s", "--scenario", default="default_4v3.yaml")
    p_sim.add_argument("--genome", default=None)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--max-time", type=float, default=240.0)
    p_sim.add_argument("-o", "--out", default=None)
    p_sim.add_argument("--acmi", default="sim.txt.acmi")
    p_sim.set_defaults(func=cmd_simulate)

    p_tab = sub.add_parser("sensors-table",
                           help="Spec 1 first-detection range table (NM / km)")
    p_tab.add_argument("--seeds", type=int, default=300)
    p_tab.add_argument("-o", "--out", default="",
                       help="Also save the table to this file")
    p_tab.set_defaults(func=cmd_sensors_table)

    p_rep = sub.add_parser("sensor-replays",
                           help="Scripted head-on / beam-at-30-NM ACMI replays")
    p_rep.add_argument("--seed", type=int, default=1)
    p_rep.add_argument("--stats-seeds", type=int, default=100)
    p_rep.add_argument("-o", "--out", default=None)
    p_rep.add_argument("--scenario", default="all",
                       choices=["all", "headon", "beam", "beam-then-hot"],
                       help="Which replay(s); a single one appends to replay_events.txt")
    p_rep.add_argument("--bth-beam-nm", type=float, default=45.0,
                       help="beam-then-hot: range (NM) at which the F-35 turns to the beam")
    p_rep.set_defaults(func=cmd_sensor_replays)

    p_dl = sub.add_parser("datalink-replays",
                          help="Spec 2 replays: launch-on-remote and IRST triangulation")
    p_dl.add_argument("--scenario", default="all",
                      choices=["all", "launch-on-remote", "triangulation", "lead-trail", "fc-lock"])
    p_dl.add_argument("--seed-c", type=int, default=1)
    p_dl.add_argument("--seed-a", type=int, default=1)
    p_dl.add_argument("--seed-b", type=int, default=1)
    p_dl.add_argument("--stats-seeds", type=int, default=100)
    p_dl.add_argument("--shot-frac", type=float, default=0.95,
                      help="launch-on-remote: wingman shoots at this fraction of table Rmax")
    p_dl.add_argument("--shot-frac-sweep", default="",
                      help="launch-on-remote: comma list of shot fractions for the "
                           "hit-rate vs shot-range table (e.g. 0.95,0.85,0.75,0.65)")
    p_dl.add_argument("-o", "--out", default=None)
    p_dl.set_defaults(func=cmd_datalink_replays)

    p_ms = sub.add_parser("missile-sweep",
                          help="Spec 3a missile shot sweep, Rmax/Rne summary + ACMI replays")
    p_ms.add_argument("--seed", type=int, default=2, help="seed for the ACMI replays")
    p_ms.add_argument("--skip-sweep", action="store_true")
    p_ms.add_argument("--skip-summary", action="store_true")
    p_ms.add_argument("--skip-replays", action="store_true")
    p_ms.add_argument("-o", "--out", default=None)
    p_ms.set_defaults(func=cmd_missile_sweep)

    args = parser.parse_args(argv)
    if args.command == "sensors-table":
        return args.func(args)
    if args.out is None:
        root = _project_root()
        args.out = str(root / {"evolve": "runs/demo",
                               "sensor-replays": "artifacts/spec1",
                               "datalink-replays": "artifacts/spec2",
                               "missile-sweep": "artifacts/spec3a"}.get(args.command,
                                                                       "artifacts"))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
