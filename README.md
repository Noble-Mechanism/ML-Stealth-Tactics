# ML Stealth Tactics

Simulation testbed for **novel fighter tactics** evolved with a **genetic algorithm**,
with **TacView ACMI 2.2** playback.

MVP: GA evolves high-level tactics for a **4-ship of generic stealth fighters (Blue)**
versus **randomized Red presentations** (spec 4: a 6-ship Red flight in a random
formation, geometry, aggressiveness and doctrine, with one pre-planned maneuver)
or a fixed scenario YAML. The best engagement exports as `.txt.acmi` openable in
TacView.

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

Spec 4: evolve against randomized presentations (24 per generation, one per
maneuver x aggressiveness band x doctrine cell, resampled each generation; a fixed
64-presentation benchmark picks the champion):

```bash
python -m stealth_tactics evolve --pop 50 --gens 100 --seed 42 \
  --presentations 24 --benchmark 64 --workers 8 --out runs/spec4
# reproduce the champion from its stored presentations (fresh process) and
# check the ACMI re-export is byte-identical
python -m stealth_tactics replay-champion runs/spec4
```

The run directory also gets `champion.json` (genome, fitness, per-fight scores
and every stored presentation). The hard sim cap is now **360 s** (`--max-time`).

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
# TacView replays A-F (drag/recommit, crank, beam, spec 3 turn-away limit press/depart,
# Blue test reaction, shoot-shoot-assess) + event logs with a-pole / f-pole
python -m stealth_tactics defense-replays -o spec3_outputs
# 100-seed stats: a in {0,.25,.5,.75,1} x Blue test reaction off/on x Blue doctrine
python -m stealth_tactics defense-stats --seeds 100 -o spec3_outputs
```

Red jets hear Blue radar modes on an RWR (search / lock / support / missile
active) and defend by aggressiveness band: conservative drags on a lock (dives
toward the 100 m AGL floor), middle beams on support, aggressive cranks 50° on
missile active and keeps shooting. Each jet gets exactly one defensive
reaction; after it every jet presses back in, whatever its aggressiveness
(approved change 2026-09-26). A Red jet only leaves when it is out of missiles
with none of its own in flight. Every jet fires shoot-assess-shoot (one missile in flight per contact) or
shoot-shoot-assess (a pair per contact, 3 s apart), and can engage several
contacts at once. See
`docs/specs/03-missile-defense.md`.

## Red presentations (Spec 4)

```bash
python -m stealth_tactics sample-presentation --seed 7 --json p7.json
# 100 random presentations with the scripted (default-genome) Blue, with and
# without the Blue test reaction, + runtime and per-generation cost estimate
python -m stealth_tactics presentation-stats -n 100 --blue-test --timing -o spec4_outputs
# One TacView replay + text timeline per pre-planned maneuver (split / pump /
# low-high split / altitude change), each a different formation
python -m stealth_tactics presentation-replays -o spec4_outputs
```

A presentation is one 6-ship Red flight (`PresentationConfig.n_red`; 8-ship
formations are already in the menu) at 40-60 NM, up to 40 deg off Blue's nose,
6-12 km base altitude, in a formation from the data file
`stealth_tactics/scenarios/presentation_menus.yaml` (wall, box, ladder,
echelon, vic, champagne; every formation fits a 25 x 25 NM box). It draws the
flight's aggressiveness (band, then uniform inside it), SAS/SSA 50/50 and exactly
one pre-planned maneuver on a range trigger (30-45 NM). Red holds formation and
its assigned altitude until break-up, the defense state machine always wins over
the maneuver, and a Red jet out of missiles leaves the fight. Fights end early when
one side is dead (missiles in the air still resolve), both sides are out of
missiles with nothing in the air, or every live Red has left. The presentation
stores every drawn value, and its seed also sets the sim's noise seed, so the
same genome against the same presentation replays byte-for-byte. See
`docs/specs/04-red-presentations.md`.

## Network interface (Spec 5)

```bash
# Adapter test (layers 1-3 + radar-silent variant), layer 2 noise check,
# radar-silent TacView replay + timeline, random-weights smoke run, timing
python -m stealth_tactics interface-adapter-test -o /workspace/spec5_outputs
# One jet's named 231-input observation at a decision time
python -m stealth_tactics obs-dump --presentation 0 --stats-index --t 120 --jet B2
# Random-weights 231-64-64-13 MLP flying Blue through the full interface
python -m stealth_tactics network-smoke -n 4
```

Each Blue jet sees only its own perceived picture: its sensors, the datalink,
its RWR, plus exact own-side information. The picture is built by
`World.blue_view` and flattened into 231 inputs (`OBS_SPEC`). The jet
outputs 13 numbers (`ACTION_SPEC`):
- heading relative to the chosen target, or to the ingress axis if there is
  no target;
- altitude and speed;
- a target slot;
- fire, radar on/off and pair (SSA) bits.

`NetworkBlueController(policy, blue_ids)` decides once per second. All launch
gates stay in the sim. A radar-off jet makes no radar tracks and is not heard
by Red's RWR, but it can still fire on a wingman's fire-control track. See
`docs/specs/05-network-interface.md`.

## Neural policy and neuroevolution (Spec 6)

```bash
# Behaviour-clone HandBlue into the 231-64-64-13 network (gate: >= 80 % of its kills)
python -m stealth_tactics clone-hand -o runs/clone
# Evolve the network: 50 x 24 presentations, 10 clones + 40 random, novelty on
python -m stealth_tactics evolve-net --pop 50 --presentations 24 --gens 10 \
    --init mixed --clone runs/clone/clone.npz -o runs/net
# Continue from the last checkpoint up to generation 20 in total (byte-identical)
python -m stealth_tactics evolve-net --gens 20 --resume -o runs/net
# Random-only control night (no human prior)
python -m stealth_tactics evolve-net --init random -o runs/net_random
# Re-run the stored champion fights and compare byte-for-byte
python -m stealth_tactics replay-champion runs/net
```

A run directory holds `checkpoints/` (the last 3), `champion.json`,
`champion_weights.npz/.json`, `champion_best.txt.acmi`,
`champion_worst.txt.acmi`, `hall_of_fame/<cell>/`, `report.json` (history,
benchmark, held-out test, champion lineage) and `timing.jsonl`. Genomes carry
`n_networks` (1 today; per-element / per-jet networks are deferred) and an
interface fingerprint, so an incompatible file is refused, not misread.
Fitness is still the placeholder; spec 7 owns it. See
`docs/specs/06-neural-policy-neuroevolution.md`.

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
              # missile sweep, defense stats, Blue test reaction, spec 5 interface adapter test
  tactics/    # genome + interpreter, maneuvers (primitives), red_defense (state machine),
              # preplanned (spec 4 presentation Red controller + maneuver kinds)
  ga/         # elitist GA (scenario YAML or spec 4 presentations)
  acmi/       # ACMI 2.2 exporter
  scenarios/  # YAML/JSON loader, presentation sampler + presentation_menus.yaml
  policy/     # spec 5 network interface: view (truth firewall), observation, action,
              # controller, adapters (scripted-via-interface, HandBlue, random MLP)
  neuro/      # spec 6: MLP policy, network genome, novelty, mutation GA (evolve-net),
              # behaviour cloning (clone-hand), checkpoints, toy task
  presentation_runner.py  # run / export one presentation
scenarios/    # default_4v3.yaml, cap_4v2.yaml
docs/design.md, docs/specs/01-sensors.md, docs/specs/02-track-sharing.md,
docs/specs/02b-coast-and-lock.md, docs/specs/03a-missile-kinematics.md,
docs/specs/03-missile-defense.md, docs/specs/04-red-presentations.md,
docs/specs/05-network-interface.md,
docs/specs/06-neural-policy-neuroevolution.md
tests/
```

## License

MIT
