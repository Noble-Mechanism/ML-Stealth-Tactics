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

## Sensors (Spec 1 — see `docs/specs/01-sensors.md`)

All sensor numbers (unclassified placeholders) live in
`stealth_tactics/sim/sensor_config.py`.

- **Radar** (1 Hz): per-scan `Pd = 1/(1+(R/R50)^8.69)`,
  `R50 = ref_range × rcs_eff^0.25` (F-35 90 km, Red 70 km), ±60° az/el field of
  regard, seeded RNG.
- **Signature:** F-35 smooth azimuth-aspect table (nose 0.05, 30° 0.10, 60° 0.45,
  beam 0.90, 135° 0.55, tail 0.30; piecewise-cosine); Red isotropic 1.0.
- **IRST** (1 Hz, passive): R50 nose→tail (Red 30→60 km, F-35 25→50 km) × speed
  factor; bearing σ 0.5°, range error σ 30 %; never fire-control.
- **RWR** (every step): bearing-only when inside an enemy radar's FOR and within
  0.6 × 90 km (vs F-35 LPI) / 1.5 × 70 km (vs Red radar).
- **Tracks:** per-jet `TrackStore`; components coast 10 s then drop; position
  σ grows with range / coasting and shrinks with hold time. **Fire-control** =
  radar track held ≥ 3 s continuously (one missed scan tolerated) and within
  0.7 × R50 (Spec 2b: Blue uses a 50 NM × rcs_eff^0.25 gate) — required to launch and to keep supporting a missile.
- Friendly aircraft are never tracked. No clutter or EW.

## Track sharing (Spec 2 — see `docs/specs/02-track-sharing.md`)

- **Datalink** (`sim/datalink.py`, own RNG stream): Blue 1 s update / 1 s latency,
  Red 2 s / 2 s. 5 % loss per sender per update. Flightmates only, within 150 NM.
  Radar and IRST are shared; RWR is not. The link is undetectable.
- **Degradation:** received σ = sender σ at measurement + 50 m/s × (t − t_meas).
- **Fusion** (`sim/fusion.py`): per enemy, the smallest-σ candidate among own,
  received and (Blue) triangulated wins → `TrackStore.fused[tid]` (`FusedTrack`
  with source jet, age, own/shared, candidate list).
- **IRST triangulation** (Blue only): LOS angle γ ≥ 10°,
  σ = 0.5°·√(r1²+r2²)/sin γ (+ vertical and age terms). Never fire-control.
- **Weapons policy:** Blue may launch on a flightmate's remote FC track (≤ 3 s old).
  Support hands off to any flightmate holding its own FC track with the target
  within 90° of its nose. Red launches on its own FC only and only the shooter
  supports. `DatalinkConfig.symmetric_weapons_policy` makes Red match Blue.
- Dead jets stop sending; their data ages out on the 10 s coast.

## Weapons (Spec 3a — see `docs/specs/03a-missile-kinematics.md`)

Blue and Red carry the **same missile**. All numbers are in
`SensorConfig.missile_kinematics` (`MissileKinematicsConfig`) and
`SensorConfig.missile` (`MissileCoastConfig`, coast rules) in
`stealth_tactics/sim/sensor_config.py`. Prototype calibration values, not
sourced missile data.

- **Fly-out** (`sim/missile_kinematics.py`): 3-D point mass integrated at
  0.05 s inside each 0.5 s world step (pure-Python floats per missile; a numpy
  twin, checked identical by a test, builds the envelope table). Starts at the
  shooter's position and velocity. 6 s boost (Isp 250 s, 50 of 161.5 kg);
  drag `q S Cd0(M) cd_scale` with 1976 US Standard Atmosphere, **cd_scale 1.10**
  (retuned from the prototype's 1.25 because the sim adds lift = weight);
  induced drag `0.08 (n m g)^2/(q S)`; `n_max = min(40, q S 12/(m g))`.
- **Guidance:** PN, N = 4, no loft. Midcourse on the supported / coasting aim
  point; after going active (15 NM from the true target) on the target. Target
  motion inside a world step is the chord between its start and end positions
  (captures climbs and turns).
- **Kinematic defeat** after burnout: `defeat_speed` (< Mach 1.2) or
  `defeat_opening` (no longer closing; `miss_overshoot` if it passed within
  1 km outside the 50 m fuze). `timeout` = 180 s safety cap (3a fix, was 120 s,
  so energy rather than the cap sets Rmax).
- **Launch gate:** ammo, 500 m minimum, ≤ 60° off the nose, fire-control track
  (own, or remote for Blue) and range ≤ **table Rmax** (`sim/missile_envelope.py`).
  The table (altitude × shooter Mach × target aspect × target Mach × signed
  launch angle off the shooter's nose, plus the turn-cold no-escape Rne,
  capped at Rmax) is built from the model at first use (~83 s on 8 cores in a
  process pool, `STEALTH_TACTICS_ENV_WORKERS` to limit; serial ~9 min),
  cached in memory and in `~/.cache/stealth_tactics` (override with
  `STEALTH_TACTICS_CACHE_DIR`, `off` disables; the key includes an engine
  version and the kinematic config), and interpolated. Off-nose is + for
  "lead" (nose toward the side the target moves across the line of sight) and
  − for "lag"; lag shots past ~55° (vs Mach 0.9 targets) cannot turn without
  bleeding below Mach 1.2. Lookups with at least half their interpolation
  weight on no-shot cells return 0.
  `WeaponModel.envelope_for(shooter, target)` returns (Rmax, Rne) for the
  tactics (and Spec 5 network inputs). The genome's shoot-range fraction and
  Red's 0.8 factor now scale the table Rmax.
- **Pk** = 0.60 × f_coast × f_support × f_E:
  f_E = 1.0 at ≥ Mach 2.0, linear to 0.6 at Mach 1.2 (Mach at the fuze);
  f_support = 0.90 once if support drops for ≥ 2 s after going active (any
  qualifying flightmate under the coalition's weapons policy keeps it);
  f_coast = exp(-(e/2 km)^2) × (1 − 0.15 min(t_gap/40 s, 1)) when coasting at
  the active point (3a fix: 40 s time scale, so the aim-point error sets most
  of the penalty).

### Midcourse datalink / acquisition

- **`MissileKinematicsConfig.active_range_m` = 15 NM** (`ACQUISITION_RANGE_M`
  mirrors the default) — once the missile is this close to the TRUE target its
  seeker goes active (`autonomous`) and it guides on the target.
- **Until active**, the missile needs midcourse support:
  1. Supporting jet **alive**
  2. It holds a **fire-control quality** track on the target
  3. Target within **±90°** of its nose (`SUPPORT_OFF_BORESIGHT_DEG`) —
     pure reverse egress / turning cold drops support immediately
- Spec 2b: if support is lost before the active point the missile **coasts** on
  the extrapolated last supported position; support can resume. Lost after more
  than 40 s of continuous coast (`lost_coast_timeout`; 3a fix, was 20 s), or if the aim point is
  more than 5 km from the target at the active point (`lost_basket`).
- Spec 2: under the shared policy (Blue by default), **buddy support** is allowed
  (`support_handoff`). Red stays shooter-only unless `symmetric_weapons_policy`.
- ACMI: missiles carry orientation, `Mach=` and `TAS=`; a dead missile is
  removed (`-id`) at the end of its flight; defeats and `support_dropped` are
  bookmarks, `burnout` a message.

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

Priorities: **(1) kills**, **(2) finish sooner when engaged**, **(3) penalize Blue
losses**, **(4) kill the flee-with-0-kills local minimum**.

```
score =
    100 × BlueKills
  −  75 × BlueLosses
  +  10 × BlueAlive
  +   3 × min(BlueShots, 6)

  + if BlueKills > 0:
        40 × (1 + time_remaining_frac)
      + (Blue win ? 25 : 0)
    else:
      − 50                          # cowardice
      + (BlueShots == 0 ? −30 : 0)  # never even shot

  + (Red win ? −40 : 0)
```

Rationale:

- Strong per-kill reward so a fight that kills Red (even with some Blue losses)
  beats pristine egress with 0 kills.
- Time bonus applies **only when `BlueKills > 0`** — fleeing early does not earn
  a “fast win” bonus.
- Explicit cowardice / no-shots penalties stop the GA from parking on
  “egress immediately, take no risk.”
- Small per-shot term rewards committing weapons.

Weights live on `GAConfig` (`stealth_tactics/ga/evolution.py`).

## ACMI export

- UTF-8 text, `FileType=text/acmi/tacview`, `FileVersion=2.2`.
- `0,ReferenceTime=...` then `#<seconds>` frames.
- Stable hex object IDs; `T=lon|lat|alt|||yaw`; Name/Type/Coalition/Color/Pilot.
- Spec 1: `LockedTarget=` / `LockedTargetMode` for fire-control tracks; `0,Event=Bookmark|…` / `0,Event=Message|…` for detections, RWR contacts, FC gained/lost, tracks dropped.
- Blue: `Name=F-35A` (TacView DB), `Pilot=` = scenario callsign (`F-35-1`…), optional `ShortName=F-35`.
- Red: `Name=` / `Pilot=` keep scenario callsigns (`RedFighter-1`…).
- Destroyed objects emitted as `-id`.
- Spec 2: `frame["markers"]` → marker objects (e.g. `TRI est` Navaid/Waypoint). Link, launch-on-remote, support-handoff, autonomous and hit/miss events are exported as Bookmarks/Messages.
- Spec 3a: missile objects `T=lon|lat|alt|0|pitch|yaw,Mach=…,TAS=…`; kinematic defeat labels as Bookmarks.
- ENU → lon/lat via small-angle offset from a fixed reference (35°N, 115°W).

## GA champion capture

Each evaluation uses `sim_seed = GAConfig.seed + 1000 + seed_offset`. Generation
trials use `seed_offset = gen * 100 + i`. The champion stores that
`seed_offset` (and a deep-copied genome). The final ACMI recording re-runs with
the **same** `seed_offset` so Bernoulli Pk / stochastic outcomes match the
logged generation best — not a fresh `seed_offset=9999` re-roll.

## Deliberate non-goals (MVP)

- No classified aircraft performance or RCS tables. Blue TacView `Name=F-35A` for DB matching only; params are generic LO placeholders (NOT real F-35 performance/RCS). Red remains `RedFighter`.
- No AWACS/GCI, SAMs, link jamming, or network inputs. Buddy support exists only within the flight (Spec 2).
- No continuous RL control or neural policies.
- TacView is optional for playback; tests never require it installed.
