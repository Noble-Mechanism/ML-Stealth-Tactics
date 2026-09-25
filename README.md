# ML Stealth Tactics

Simulation testbed for **novel fighter tactics** evolved with a **genetic algorithm**,
with **TacView ACMI 2.2** playback.

MVP: GA evolves high-level tactics for a **4-ship of generic stealth fighters (Blue)**
versus a **fixed bandit presentation (Red)**. The best engagement exports as
`.txt.acmi` openable in TacView.

> Blue ACMI `Name=F-35A` (TacView Lightning II DB match); ship callsigns (`F-35-1`…) go in `Pilot=`. Performance/RCS remain unclassified generic LO placeholders — not real F-35 data. Red remains `RedFighter`.

## Install

Requires **Python 3.11+**.

```bash
cd /workspace/ML-Stealth-Tactics
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

## Run evolution (demo)

```bash
python -m stealth_tactics evolve \
  --scenario default_4v3.yaml \
  --pop 12 --gens 5 --seed 42 \
  --out runs/demo
```

Outputs:

- `runs/demo/best_genome.json` — best tactics chromosome
- `runs/demo/history.json` — per-generation fitness
- `runs/demo/best_engagement.txt.acmi` — TacView recording

Single engagement with default (or saved) genome:

```bash
python -m stealth_tactics simulate -s default_4v3.yaml -o artifacts
```

Spec 3 options for `evolve` and `simulate`: `--red-aggressiveness A` (0–1, default
0.5 or the scenario's `red_aggressiveness`), `--no-red-defense` (old pure-pursuit
Red), `--blue-doctrine` / `--red-doctrine` (`shoot_assess_shoot` default,
`shoot_shoot_assess`, or `legacy` = ripple all missiles, kept for regression).

## Sensor model tools (Spec 1)

```bash
# First-detection range table (analytic R50 + empirical closing runs)
python -m stealth_tactics sensors-table --seeds 300 -o spec1_outputs/sensor_table.txt
# Scripted 1v1 replays: head-on vs F-35 beaming at 30 NM (ACMI + event log)
python -m stealth_tactics sensor-replays --seed 1 -o spec1_outputs
```

See `docs/specs/01-sensors.md` (all parameters in `stealth_tactics/sim/sensor_config.py`).

## Track sharing tools (Spec 2)

```bash
# Replay A (launch on remote + support handoff, with 100-seed hit rate) and
# Replay B (passive IRST triangulation: ACMI with 'TRI est' marker + error table)
python -m stealth_tactics datalink-replays --seed-a 1 --seed-b 1 --stats-seeds 100 -o spec2_outputs
# Replay C (lead-trail: lead shoots at max range + turns out, trail supports) + 3-way comparison
python -m stealth_tactics datalink-replays --scenario lead-trail --seed-c 1 -o spec2_outputs
# Spec 2b: Blue 50 NM FC gate lock stability (head-on, 100 seeds)
python -m stealth_tactics datalink-replays --scenario fc-lock -o spec2_outputs
```

See `docs/specs/02-track-sharing.md` (parameters in `SensorConfig.datalink`) and
`docs/specs/02b-coast-and-lock.md` (missile coast, Blue 50 NM FC gate).

## Missile kinematics tools (Spec 3a)

```bash
# Calibration check, 704-shot sweep (10-60 NM x shooter Mach x altitude x target
# behavior), Rmax / Rne summary (sim path + lookup table) and 3 TacView replays
python -m stealth_tactics missile-sweep -o spec3a_outputs
# Off-nose axis of the Rmax table vs the full sim (+ random-grid check)
python -m stealth_tactics.analysis.off_nose_check
# Launch on remote: hit rate vs shot range (fraction of table Rmax)
python -m stealth_tactics datalink-replays --scenario launch-on-remote \
    --shot-frac-sweep 0.95,0.85,0.75,0.65 -o spec3a_outputs
```

Both sides carry the same missile: point-mass fly-out (boost, Mach-dependent drag,
1976 standard atmosphere, induced drag, g limit), PN guidance, kinematic defeat
below Mach 1.2 / when no longer closing, Pk 0.60 × endgame-energy × support /
coast factors. Launches are gated by an Rmax lookup table (altitude, shooter
Mach, target aspect, target Mach, launch angle off the shooter's nose) built
from the model at first use (~1.5 min on 8 cores in a process pool;
`STEALTH_TACTICS_ENV_WORKERS=1` for serial, ~9 min) and cached in
`~/.cache/stealth_tactics` (`STEALTH_TACTICS_CACHE_DIR` to move it, `off` to
disable). Coast timeout 40 s, flight-time cap 180 s. See
`docs/specs/03a-missile-kinematics.md` (parameters in
`SensorConfig.missile_kinematics`).

## Missile defense tools (Spec 3)

```bash
# TacView replays A-F (drag/recommit, crank, beam, turn-away limit press/depart,
# Blue test reaction, shoot-shoot-assess) + event logs with a-pole / f-pole
python -m stealth_tactics defense-replays -o spec3_outputs
# 100-seed stats: a in {0,.25,.5,.75,1} x Blue test reaction off/on x Blue doctrine
python -m stealth_tactics defense-stats --seeds 100 -o spec3_outputs
```

Red jets hear Blue radar modes on an RWR (search / lock / support / missile
active) and defend by aggressiveness band: conservative drags on a lock (dives
toward the 100 m AGL floor), middle beams on support, aggressive cranks 50° on
missile active and keeps shooting. After two turn-aways a jet presses (a ≥ 0.5)
or leaves. Every jet fires shoot-assess-shoot (one missile in flight) or
shoot-shoot-assess (two at one target, 3 s apart). See
`docs/specs/03-missile-defense.md`.

## Tests

```bash
pytest -q
```

## Open ACMI in TacView

1. Install [TacView](https://www.tacview.net/) (not required for tests).
2. **File → Open** (or drag-and-drop) the `.txt.acmi` file.
3. Play the engagement; Blue = blue coalition, Red = red.

ACMI files are UTF-8 text (`FileType=text/acmi/tacview`, `FileVersion=2.2`).

## Genome & fitness (short)

The genome encodes **formation offsets**, **commit range**, **merge geometry**
(bracket / hook / drag / sandwich / head-on), **weapon doctrine**, and
**post-merge roles** — not raw stick inputs. Fitness prioritizes **kills**,
penalizes Blue losses, rewards faster wins **only when kills > 0**, and
explicitly punishes flee-with-0-kills. Sensors (Spec 1): probabilistic radar
with ±60° field of regard, smooth aspect-dependent F-35 RCS, passive IRST, RWR,
and per-jet tracks; launches and midcourse support (until ~15 NM) need a
fire-control quality track. Missiles (Spec 3a) fly a point-mass kinematic model
and launch inside the table Rmax. Red defends against missiles (Spec 3).
See `docs/design.md` for models and assumptions.

## Layout

```
stealth_tactics/
  sim/        # point-mass world, sensors (+ sensor_config, tracks), datalink, fusion,
              # weapons, missile_kinematics (fly-out), missile_envelope (Rmax table), rwr
  analysis/   # sensor table, scripted sensor / datalink / missile / defense replays,
              # missile sweep, defense stats, Blue test reaction
  tactics/    # genome + interpreter, maneuvers (primitives), red_defense (state machine)
  ga/         # elitist GA
  acmi/       # ACMI 2.2 exporter
  scenarios/  # YAML/JSON loader
scenarios/    # default_4v3.yaml, cap_4v2.yaml
docs/design.md, docs/specs/01-sensors.md, docs/specs/02-track-sharing.md,
docs/specs/02b-coast-and-lock.md, docs/specs/03a-missile-kinematics.md,
docs/specs/03-missile-defense.md
tests/
```

## License

MIT
