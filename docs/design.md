# Design Notes — ML Stealth Tactics MVP

## Purpose

A **fast**, seedable discrete-time air-combat testbed so a genetic algorithm can
evolve **high-level fighter tactics** for a 4-ship of generic stealth aircraft
(Blue) against randomized Red presentations (Spec 4) or a fixed scenario YAML. Best engagements export to TacView
ACMI 2.2 for visual playback.

## Simulation model

- **Point-mass** kinematics in local ENU (East, North, Up) meters.
- State per aircraft: `x, y, alt, heading, speed, alive`.
- Heading convention: **0 = North**, increasing **clockwise** (aviation/TacView yaw).
- Limits: max turn rate (°/s), climb rate (m/s), speed band, altitude band.
- **Ground (Spec 3):** flat ground at 0 m (`aircraft.GROUND_ALT_M`); hard
  **100 m AGL floor** for every aircraft (`ALT_FLOOR_AGL_M`, enforced in
  `integrate_aircraft`). A missile that reaches the ground ends with outcome `ground`.
- Time step default **0.5 s** — coarse enough for hundreds of evals/generation on a laptop; no game engine.

## Sensors (Spec 1 — see `docs/specs/01-sensors.md`)

All sensor numbers (unclassified placeholders) live in
`stealth_tactics/sim/sensor_config.py`.

- **Radar** (1 Hz): per-scan `Pd = 1/(1+(R/R50)^8.69)`,
  `R50 = ref_range × rcs_eff^0.25` (F-35 90 km, Red 70 km), ±60° az/el field of
  regard, seeded RNG.
- **Signature:** F-35 smooth azimuth-aspect table (Spec 3 change A: nose 0.05,
  20° 0.05, 45° 0.12, 70° 0.50, beam 0.90, 135° 0.55, tail 0.30; piecewise-cosine);
  Red isotropic 1.0. Red detection range vs nose-on: ×1.18 at 35°, ×1.24 at 45°.
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

## Missile defense and firing doctrine (Spec 3 — see `docs/specs/03-missile-defense.md`)

- **RWR modes** (`sim/rwr.py`, own RNG stream salt 0x3D3F, after sensors each
  step): per receiver and enemy emitter the highest of search < lock < support,
  plus a per-missile `missile_active` cue for the targeted jet only. Deterministic
  spec 1 range gates (F-35 LPI 29 NM, Red radar 57 NM), ±60° emitter FOR, perfect
  mode ID, 5° bearing noise, per-mode range factors. `world.rwr[id]` → `RwrCue`s.
  Both sides get them. Only `radar_emitting` jets are seen (always on until spec 5).
- **Maneuver primitives** (`tactics/maneuvers.py`, spec 4 hook): `hot`, `crank`
  (50°), `beam` (90°, max speed, hold altitude), `drag` (threat on tail, max speed,
  descend `depth(a) = (1−a)(alt − floor) + a × 1000 m`, never below 100 m AGL),
  `depart`.
- **Red defense** (`tactics/red_defense.py`, `DefenseConfig`), per jet, one
  aggressiveness `a` per flight (scenario `red_aggressiveness`, default 0.5):
  a < 1/3 lock → drag; 1/3–2/3 support → beam; ≥ 2/3 missile active → crank
  (keeps firing and supporting). Threat cleared after 2 s without a qualifying cue
  and the TOF estimate (range / 700 m/s) run out; cold `5 + 35(1−a)` s; turn-away
  limit 1 (cranks count), then every jet presses regardless of `a` (approved
  change 2026-09-26; the spec 3 rule "limit 2, then press if a ≥ 0.5 else depart"
  is kept only as `DefenseConfig.spec3()`). The only Red departure is Winchester
  (spec 4 L). The fight ends early when all live Red have departed and nothing
  is in flight.
- **Firing doctrine** per jet (`SimConfig.blue_doctrine` / `red_doctrine`,
  `Aircraft.firing_doctrine` override), both **per contact**:
  `shoot_assess_shoot` (default; max 1 own missile in flight per target, several
  contacts at once) or `shoot_shoot_assess` (2 at one target 3 s apart, then
  nothing more at that target until both resolve; other targets at once). The
  network decides when and how many shots; doctrine only limits per contact.
  Controllers fall back to the next-nearest enemy when their target is blocked.
  `legacy` (ripple everything) is kept for regression.
- **Shot log:** `SimResult.shots` / `red_shots` with launch range, off-nose,
  a-pole, f-pole, outcome, Mach, Pk, target defense state at launch and end.
- Blue scripted test reaction (`analysis/blue_test_defense.py`) is for tests and
  replays only; the GA does not use it.

## Red presentations (Spec 4 — see `docs/specs/04-red-presentations.md`)

- **Presentation** (`scenarios/presentation.py`): one Red flight of `n_red` = 6
  jets (Rusty; configurable, 8-ship formations already in the menu) relative to
  the fixed Blue start of `default_4v3.yaml`: formation + spacings, range 40-60 NM,
  azimuth ±40° off Blue's nose, Red pointed at Blue lead, base altitude
  6-12 km (+ formation stack, ±150 m jitter, clipped 1-13.5 km), speed fixed
  250 m/s; aggressiveness band (1/3 each) then uniform inside it; SAS/SSA 50/50;
  exactly one pre-planned maneuver with its range trigger. Stores every drawn
  value, the Blue block and the absolute Red start states (JSON round trip).
- **Menus are data** (`scenarios/presentation_menus.yaml`): formations = ordered
  groups (≤ ~3 NM each) with slot offsets as expressions over drawn params, plus
  `halves` for maneuvers that divide the flight; 6-ship wall, box, ladder,
  echelon, vic, champagne; 8-ship wall_8, box_8, ladder_8. Every formation fits
  a 25 x 25 NM box at every parameter corner (tested). Maneuvers = YAML entries
  over registered kinds (`tactics/preplanned.py`, `@register_kind`): split
  (turn_out, halves), pump (cold, all), low_high_split (altitude, halves),
  altitude_change (altitude, all).
- **Red in presentation mode** (`PresentationRedController`): wingmen station-keep
  on the flight leader and every jet flies its assigned altitude (not Blue's)
  until break-up (maneuver start, defending, own FC track, Blue inside 20 NM),
  then individual pure pursuit on true positions (perfect GCI; GCI quality is a
  future knob). The maneuver fires once when the true range from any live Red to
  any live Blue reaches the trigger. Defense always wins (skip if not HOT at the
  trigger, abort for good if defending). Not a turn-away. A Red jet with no
  missiles and none in flight departs (not a turn-away).
- **Early end** (`SimConfig`, both flags off by default so YAML / regression
  runs are unchanged; the presentation runner turns them on):
  `early_end_winchester` (no jet on either side can shoot and nothing in the
  air) and `finish_missiles_after_wipeout` (K2: after one side is wiped out,
  missiles in the air still resolve, so trade kills count). The D9 all-Red-departed
  end is unchanged. `SimResult.end_reason` ∈ blue_dead, red_dead,
  both_winchester, red_departed, time_cap. A Blue-only Winchester fight goes on
  (Red can chase; surviving to the cap = got away).
- **Seeding:** named sub-streams per field group from the presentation seed;
  `sim_seed` derived from the presentation seed (never from a genome's index).
  `build_eval_set(master, gen, 24)` = one maneuver x band x doctrine crossing per
  generation (resampled each generation); `build_benchmark_set(master, 64)`.
- **Runner** (`presentation_runner.run_presentation`): 360 s cap, SAS Blue by
  default; `export_presentation_acmi` writes the presentation summary into
  `0,Comments=`.

## Network interface (Spec 5 — see `docs/specs/05-network-interface.md`)

The seam for the spec 6+ neural network. Spec 6 needs only `OBS_SPEC.size`
(231), `ACTION_SPEC.size` (13) and a `policy(obs_batch) -> out_batch` callable.

- **Truth firewall** (`policy/view.py`): `World.blue_view(ac)` builds a frozen
  `BlueView` from whitelisted perception only. It holds own and flightmate
  state (exact; own side), the jet's fused picture (estimate, sigma, sensor,
  age, radar velocity estimate), own and remote FC, doctrine permission, RWR
  cues, Blue missiles (weapon datalink) and confirmed kills. It never holds a
  Red `Aircraft`, a track component (`true_range_m`), Pk or
  `primary_fc_target`.
- **Observation** (`policy/observation.py`), per jet, 231 inputs:
  - own 21: slot one-hot, altitude, speed, heading vs the ingress axis (north),
    ammo, time, radar on, own-missile states, pair open, missile warnings,
    kills;
  - 6 contact slots × 28: slot 0 holds the current target, the rest are
    ordered by perceived range, then bearing-only RWR contacts. Each has
    perceived range, bearing, altitude delta, aspect, closure, speed, sigma,
    age, sensor, own/remote FC, r/Rmax and r/Rne from the envelope on
    perceived states, doctrine OK, shot ready, missiles on it, mates
    targeting it, and RWR mode;
  - 3 element-relative wingmen × 14.

  Fixed scales; declared ranges [0, 1], [−1, 1] or [0, 1.5] (altitude delta
  ±1.5).
- **Action** (`policy/action.py`), 13 outputs:
  - heading offset ±180° about the chosen target's bearing (the ingress axis
    if there is no target);
  - altitude 100–15,000 m and speed 90–340 m/s (linear);
  - 6 + 1 target logits (argmax over present slots and "none");
  - fire, radar on/off, pair (SSA) bits.
- **Controller** (`policy/controller.py`): `NetworkBlueController` decides
  every 1 s (t = 0, 1, 2 …), batches all live jets, holds commands between
  decisions, re-asserts fire each step, enforces a 5 s radar dwell, and sets
  the per-jet doctrine from the pair bit only while no pair is open. All
  launch gates (FC, range, off-boresight, ammo, doctrine) stay in the World,
  so the network cannot cheat.
- **Radar off** (`Aircraft.radar_emitting`, default on): no own radar tracks
  or FC; not heard by Red RWR (modes or spec 1 component); IRST, datalink and
  RWR still work. The jet can launch on a flightmate's FC, and support is
  handed to the FC holder. ACMI `RadarMode` is written only on a change.
- **Adapters** (`policy/adapters.py`): `ScriptedViaInterface` runs the script
  through encode → decode, `HandBlue` is an obs-only hand policy, and
  `RandomMLPPolicy` is the smoke-test net. `interface-adapter-test` runs the
  three-layer sufficiency test (results in the spec's Implementation section).
- **Cost:** +11 % per generation at 1 s (187 s vs 169 s on 8 workers).

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
- Spec 3: `rwr_mode` messages; `defend`, `recommit`, `press`, `depart`, `blue_defend`, `blue_recommit` and `ground` bookmarks.
- ENU → lon/lat via small-angle offset from a fixed reference (35°N, 115°W).

## GA champion capture

**Spec 4 presentation mode** (`GAConfig.presentations_per_gen > 0`): every genome
in a generation plays the same N presentations (fitness = mean per-fight fitness,
a placeholder until spec 7), `workers` > 1 uses a process pool. The generation's
best is scored on the fixed benchmark set; the champion is the best benchmark
score. The champion keeps its stored presentation dicts; at the end the GA
re-runs it on those stored presentations (and the benchmark) and raises if
anything differs, then records its best fight for the ACMI. `champion.json` +
`replay-champion` reproduce it from a fresh process. `GAConfig.sim_max_time_s`
default is now 360 s (Rusty), also in YAML mode.

**Scenario YAML mode:**

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
