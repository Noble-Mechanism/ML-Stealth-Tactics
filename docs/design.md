# Design Notes — ML Stealth Tactics MVP

## Purpose

A **fast**, seedable discrete-time air-combat testbed so a genetic algorithm can
evolve **high-level fighter tactics** for a 4-ship of generic stealth aircraft
(Blue) against a fixed Red presentation. Best engagements export to TacView
ACMI 2.2 for visual playback.

## Simulation model

- **Point-mass** kinematics in local ENU (East, North, Up) meters.
- State per aircraft: `x, y, alt, heading, speed, alive`.
- Heading convention: **0 = North**, increasing **clockwise** (aviation/TacView yaw).
- Limits: max turn rate (°/s), climb rate (m/s), speed band, altitude band.
- Time step default **0.5 s** — coarse enough for hundreds of evals/generation on a laptop; no game engine.

## Sensors (assumptions)

- Detection if `range ≤ radar_range × rcs_factor^0.25`.
- Blue stealth uses `rcs_factor ≈ 0.05` → Red detects Blue much later than reverse.
- Lock requires detection and roughly `0.7 ×` detection range.
- No multipath, clutter, or EW — intentional MVP simplification.
- Friendly aircraft are never “detected” as threats.

## Weapons (assumptions)

- Simplified BVR fire-and-forget missile.
- Launch constraints: ammo > 0, range ≤ missile range, within ~60° of nose, lock available.
- Missile steers toward target position (not true PN); proximity fuse ~50 m.
- Hit resolved by Bernoulli trial with shooter `missile_pk` (Blue ~0.60, Red ~0.50).
- No loft, no notch, no supporting radar after launch.

## Tactics genome (high-level)

Not stick-and-throttle traces. Genes encode:

| Gene | Meaning |
|------|---------|
| Formation offsets | Wingman right/forward/up relative to lead |
| Commit range | Start merge when nearest bandit within range |
| Abort loss threshold | Disengage after N Blue losses |
| Merge geometry | `bracket` / `hook` / `drag` / `sandwich` / `headon` |
| Split / drag mask | Who goes left vs right; who drags |
| Weapon doctrine | Conservative / standard / aggressive + shoot-range fraction |
| Post-merge roles | Engage / support / egress / CAP per ship |
| Alt/speed biases | Commit altitude offset and speed factor |

Crossover = uniform gene mix; mutation = per-gene Gaussian / categorical flips;
selection = tournament + elitism.

## Fitness

```
score = 30×BlueKills − 40×BlueLosses + 15×BlueAlive
      + 0.25×BlueAmmoLeft
      + (Blue win ? 10×(1 + time_remaining_frac) : 0)
      − (Red win ? 20 : 0)
```

Encourages killing bandits, preserving the package, winning quickly, and not
wasting missiles.

## ACMI export

- UTF-8 text, `FileType=text/acmi/tacview`, `FileVersion=2.2`.
- `0,ReferenceTime=...` then `#<seconds>` frames.
- Stable hex object IDs; `T=lon|lat|alt|||yaw`; Name/Type/Coalition/Color.
- Destroyed objects emitted as `-id`.
- ENU → lon/lat via small-angle offset from a fixed reference (35°N, 115°W).

## Deliberate non-goals (MVP)

- No classified aircraft performance or RCS tables (labels: BlueStealth / RedFighter only).
- No datalink fusion, AWACS, or SAMs.
- No continuous RL control or neural policies.
- TacView is optional for playback; tests never require it installed.
