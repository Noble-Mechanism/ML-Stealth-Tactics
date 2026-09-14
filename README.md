# ML Stealth Tactics

Simulation testbed for **novel fighter tactics** evolved with a **genetic algorithm**,
with **TacView ACMI 2.2** playback.

MVP: GA evolves high-level tactics for a **4-ship of generic stealth fighters (Blue)**
versus a **fixed bandit presentation (Red)**. The best engagement exports as
`.txt.acmi` openable in TacView.

> Generic labels only (`BlueStealth` / `RedFighter`) — no classified aircraft data.

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
**post-merge roles** — not raw stick inputs. Fitness rewards Blue kills and
package survival, penalizes Blue losses, and adds a bonus for faster Blue wins
plus a small ammo term. See `docs/design.md` for models and assumptions.

## Layout

```
stealth_tactics/
  sim/        # point-mass world, sensors, weapons
  tactics/    # genome + interpreter
  ga/         # elitist GA
  acmi/       # ACMI 2.2 exporter
  scenarios/  # YAML/JSON loader
scenarios/    # default_4v3.yaml, cap_4v2.yaml
docs/design.md
tests/
```

## License

MIT
