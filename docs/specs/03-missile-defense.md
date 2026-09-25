# Spec 3 — Missile defense (RWR modes, Red reactions, maneuver primitives)

Status: **implemented (2026-09-25).** Rusty approved D1–D10 exactly as recommended,
plus changes **A–C** (F-35 RCS table, 100 m AGL floor with aggressiveness-based drag
depth, per-jet firing doctrines). See "Implementation" at the end for deviations and
results. All numbers are prototype placeholders. Spec 3a (missile
kinematics) is unchanged. Whether a defense works is decided only by the fly-out
model (energy bleed below Mach 1.2, or the missile no longer closing).

## What it does

Right now Red flies pure pursuit and fires every step it is allowed to, so it never
defends. This spec adds four things:
1. A warning receiver (RWR) that reports what kind of threat it sees.
2. A Red defensive state machine driven by one aggressiveness value per flight.
3. A reusable module of maneuver primitives.
4. A-pole and f-pole in the shot log.

Goal: skate vs banzai becomes a real trade-off for both sides. A Red jet that
defends early survives but rarely gets its own shot off. A Red jet that presses
threatens Blue but flies into no-escape range.

## Decided with Rusty

- **No Pk modifiers** for crank, beam or drag. They work only when the 3a
  kinematics say they do.
- **RWR modes.** The RWR reports Blue radar **search**, **fire-control lock** and
  **missile support** (datalink). It also gives a **missile-active** warning when a
  missile's seeker goes active within seeker range. Only an emitting radar is
  detected. New field `Aircraft.radar_emitting` is always `True` here and becomes a
  network output in spec 5.
- **Aggressiveness** `a ∈ [0, 1]` per Red flight. Spec 4 will draw it; for now it is
  set in the scenario YAML as `red_aggressiveness`, default 0.5. It sets:
  - the trigger: conservative reacts to the first lock, middle to missile support,
    aggressive only to missile active;
  - the reaction: aggressive cranks (keeping radar and support on its own shot),
    middle beams, conservative drags;
  - how long the jet stays cold before recommitting after the threat drops.
- **Turn-away limit** per Red jet (default 2, see D10). Once the limit is used, the
  jet either presses (aggressive) or leaves the fight for good (conservative). A
  departed jet is "out of the fight"; spec 7 decides how fitness scores that. The
  hard sim time limit stays.
- **A-pole and f-pole** are recorded for every shot.
- **Maneuver primitives** (crank, beam, drag, hot) go in their own module. Spec 4
  will reuse them for one pre-planned maneuver per Red presentation (split, pump,
  altitude change on a time or range trigger). Only the hook is added here.
- **Blue** gets its defensive moves from the network later. In this spec Blue only
  has a scripted test reaction (below).
- **Deferred:** notching, chaff/countermeasures and lookdown/lookup clutter. The
  sensor model has no Doppler or clutter. Chaff must never be very effective.

## Model

### 1. RWR modes (`sim/rwr.py`, new)

Modes are computed every step (0.5 s) **after** all track stores are maintained,
so they use the same step's fire-control state. For each receiver and each enemy
emitter, only the highest mode is reported, in this order: search < lock < support.

| Mode | Condition (emitter radar emitting, receiver inside its ±60° FOR and inside RWR range) |
|---|---|
| search | none of the below |
| lock | the emitter holds a fire-control track on the receiver (`is_fire_control`) |
| support | the emitter is `supporter_id` of a live missile aimed at the receiver |
| missile active | the missile aimed at the receiver is `autonomous`; reported per missile with the missile's bearing (no FOR or emitter gate) |

- The bearing σ is 5° (spec 1). There is no range.
- Noise comes from a **new RNG stream** (`rng_salt 0x3D3F`), so spec 1–3a results
  do not shift.
- Output: `world.rwr[receiver_id]` → list of `RwrCue(emitter_id, mode, bearing, t_first)`.
- Events `rwr_mode` (on change) go to the log and to ACMI.
- The existing spec 1 RWR track component is unchanged.

**Consequence worth noting:** Blue can launch on a remote FC track. The shooter then
never locks the Red jet, so Red sees a lock only from the jet that actually holds
the FC track, and only if that jet is within 29 NM and has Red inside its field of
regard.

### 2. Maneuver primitives (`tactics/maneuvers.py`, new)

These are pure functions. Each takes an aircraft and a threat bearing and returns a
`ManeuverCmd(heading, speed, alt)`. They make no decisions and hold no state.

- `hot(ac, brg)`: pure pursuit at 1.1 × cruise (today's Red intercept).
- `crank(ac, threat_brg, angle=D2)`: threat on the nose ± angle, turning the short way.
- `beam(ac, threat_brg)` and `drag(ac, threat_brg)`: see D3.
- `depart(ac, away_brg)`: see D9.

Spec 4 hook: a presentation can later call the same functions on a time or range
trigger. The defense state machine takes priority (spec 4 confirms).

### 3. Red defense state machine (`tactics/red_defense.py`, new)

It runs inside `RedCAPController` and is per jet, because RWR is not shared over the
datalink (spec 2). States:

`HOT → DEFENDING → COLD → HOT`, plus the terminal `PRESSING` and `DEPARTED`.

- **HOT → DEFENDING:** a cue at or above the jet's trigger level (D1). If the jet
  was HOT when triggered, this counts as one turn-away (D10). A new cue while
  already DEFENDING or COLD does not count again.
- **Threat reference:** the bearing of the highest cue. Missile active beats support,
  which beats lock. Ties go to the nearest emitter if Red has a range on it, else the
  first one seen.
- **DEFENDING → COLD:** when the threat clears (D4). COLD keeps the reaction heading
  for `T_cold` (D4), then the jet goes HOT.
- **Limit reached:** on the next trigger after the limit is used, the jet goes to
  PRESSING (ignores cues, stays hot and shoots) or DEPARTED, depending on D1.
- **Shooting** follows D6.

### 4. Shot log

`Missile` gains:
- `a_pole_m`: shooter–target range when the seeker goes active, or `None`. A missile
  that starts active records it at launch.
- `f_pole_m`: shooter–target range when the missile ends, or `None` if the shooter
  is dead.
- `launch_range_m`, which already exists.

`SimResult.shots` is a list with one entry per missile: shooter, target, launch t
and range, off-nose, a-pole, f-pole, outcome, Mach at fuze, Pk, and the target's
defense state at launch and at the end. These fields feed the ACMI bookmarks and the
test tables.

### 5. Blue scripted test reaction (`analysis/blue_test_defense.py`, tests only)

`BlueTestDefense` wraps any Blue controller.
- **Trigger:** on a Red support or missile-active cue (Blue RWR, D8).
- **Reaction:** crank 50° while the jet is supporting its own pre-active missile;
  otherwise, and once that missile is active, drag.
- **Recommit:** 20 s after the threat clears.
- No turn-away limit.
- It is not used by the GA.

## Decisions (all approved by Rusty)

**D1 — Aggressiveness mapping (approved).** Use **3 bands** for trigger and reaction:
- `a < 1/3`: conservative (lock → drag)
- `1/3 ≤ a < 2/3`: middle (support → beam)
- `a ≥ 2/3`: aggressive (missile active → crank)

Recommit time is **continuous** (D4). Once the limit is used, the jet **presses if
`a ≥ 0.5`**, otherwise it leaves.

*Why:* discrete doctrines are easy for pilots to read in replays and to test. The
continuous parts keep the spec 4 random draw meaningful inside a band.

**D2 — Crank angle 50° (approved).**
- Radar field of regard ±60°, minus 10° of margin. The margin covers the 5° RWR
  bearing noise and the 0.5 s step at 13°/s.
- It keeps fire control (±60°) and support (±90°).
- It keeps the jet's own new shots off the 55–60° lag cliff (3a fix).
- Speed 1.1 × cruise, altitude held.

**D3 — Beam and drag (approved; drag depth replaced by change B).**
- **Beam:** heading 90° off the threat bearing, turning the short way, **max speed**,
  altitude held.
- **Drag:** threat on the tail, **max speed**, descend to
  `max(100 m AGL, alt − depth(a))` (change B). The draft's 3,000 m / 4,500 m floor
  is gone.

*Why:* both work only through the 3a energy bleed. Beam RCS does not matter because
Red is isotropic and there is no Doppler notch. Descending thickens the air the
missile must fly through, but a low jet recommits with a much shorter shot (hot
Rmax 28.8 NM at 25 kft vs 20.6 NM at 15 kft). That is a real cost of dragging low.

**D4 — Threat cleared and recommit timing (approved).** The threat is cleared when no cue at
or above the trigger level has been seen for **2 s** (this rides out FC flicker,
spec 2b) **and**, if a support cue was seen during this defense, the estimated
time of flight (D5) has run out or the missile-active cue has ended (missile gone).
Then the jet stays cold for **`T_cold = 5 s + 35 s × (1 − a)`**: 40 s at a = 0,
22.5 s at 0.5, 5 s at 1.

*Why:* it uses only what an RWR can know. It is short enough that a conservative
jet can make two full cycles inside the 240 s GA limit.

**D5 — Red's time-of-flight estimate (approved).** Red has no missile launch warning and the
RWR gives no range. So:
- `t_est = R_est / 700 m/s`, counted from the first support cue.
- `R_est` is Red's own fused range to that emitter (radar, IRST or datalink) if it
  has one. Otherwise it is the RWR intercept range for that emitter type (54 km vs
  F-35 → 77 s).

*Why:* 700 m/s is a deliberately low average closing speed from the 3a fly-outs
(hot 40 NM: about 1,000 m/s; cold 20 NM: about 620 m/s), so the estimate errs long.

**D6 — Shooting while defending (approved); shot doctrine replaced by change C.**
- No special rule: the existing gates decide (own FC, ≤ 60° off the nose, 0.8 ×
  table Rmax).
- A cranking jet can still fire and support. A beaming or dragging jet cannot fire,
  and it drops support, so its missile coasts (spec 2b). PRESSING jets shoot
  normally. DEPARTED jets never fire.
- ~~At most one live missile per shooter per target~~ — superseded by the firing
  doctrines of change C. (Before spec 3 every jet rippled all 4 missiles in 2 s:
  in `default_4v3` B1 and B2 put 8 missiles on R1 between 65 and 69 s, which made
  any defense meaningless.) This **changes Blue baseline results.**

**D7 — RWR detection model (approved).**
- Deterministic, using the spec 1 range gates for **all** modes: F-35 LPI 29 NM,
  Red radar 57 NM.
- Modes are identified perfectly, with no search/lock ambiguity for now.
- Missile active goes only to the targeted jet.
- Config gets per-mode range factors (default 1.0), so an LPI lock can be made
  harder to see later.

*Why:* simple, testable, and it fits the rest of spec 1. Adding ambiguity later is a
config change.

**D8 — Blue gets the same RWR modes: yes (approved).** Red radar always emits. Blue uses them
only for the test reaction here, and they are ready as spec 5 network inputs.

**D9 — "Leaves the fight" (approved).**
- The jet turns directly away from the Blue centroid in its own fused picture
  (else the last threat bearing), at max speed, holding altitude.
- It is marked `departed` right away and stops shooting.
- It can still be shot. A tail chase is a hard shot (cold Rmax 11.6 NM at 25 kft),
  so this matters little.
- **The engagement ends early** once every live Red jet is departed and no missile
  is in flight. The hard time limit is unchanged.

*Why:* this stops Blue from chasing a runner until the time limit and keeps fights
short.

**D10 — Turn-away limit: default 2 (approved).**
- Each HOT → DEFENDING transition counts, crank included.
- At a = 0 one cycle is about 60–90 s (drag, clear, 40 s cold). Two cycles plus a
  press or departure fits inside 240 s.

Also approved with the draft: the Blue scripted test reaction (tests only),
`tactics/maneuvers.py` as the spec 4 hook, a-pole/f-pole logged per shot, a new RNG
stream for RWR noise, notching/chaff deferred.

## Additional approved changes (A–C)

**A — F-35 RCS curve (approved).** `SignatureConfig` F-35 aspect table becomes
(0°, 0.05) (20°, 0.05) (45°, 0.12) (70°, 0.50) (90°, 0.90) (135°, 0.55) (180°, 0.30),
same piecewise-cosine interpolation. Flat best RCS within 20° of the nose; a
35–45° crank costs little (Red detection range ×1.18 at 35°, ×1.24 at 45°); steep
rise to the beam. Spec 1 doc and tests updated (slope bound 0.03 → 0.035/deg).

**B — Altitude floor and drag depth (approved).**
- Hard **100 m AGL floor** for all aircraft; flat ground at 0 m (no terrain yet).
  Enforced in aircraft integration, so every controller respects it.
- Drag target altitude = `max(100 m, alt − depth(a))`, with
  `depth(a) = (1 − a) × depth0 + a × 1,000 m`; `depth0` defaults to "all the way to
  the floor" (`alt − 100 m`). Endpoints are `DefenseConfig.drag_depth_a0_m` (None =
  to the floor) and `drag_depth_a1_m` (1,000 m). Computed once when the drag starts.
  Used by every drag (spec 4 pre-planned maneuvers reuse `maneuvers.drag`).
- A missile that reaches the ground is defeated with outcome **`ground`**
  (ACMI bookmark).
- Lookdown/lookup clutter is deferred (below).

**C — Firing doctrine per jet, both sides (approved).**
- `shoot_assess_shoot` (SAS): at most one own missile in flight.
- `shoot_shoot_assess` (SSA): two missiles at the same target, the second 3 s
  after the first (`SimConfig.ssa_interval_s`), then no shots at anyone until
  both are resolved.
- Defaults: Red SAS (spec 4 will randomize), Blue configurable
  (`SimConfig.blue_doctrine`, default SAS; spec 5 may make it a network output).
  Per-jet override `Aircraft.firing_doctrine`.
- Replaces the old "ripple all 4 in 2 s" behaviour, which remains as `legacy`
  for regression tests only.

## Deferred

- Notching and chaff/countermeasures (sensor model has no Doppler or clutter;
  chaff must never be very effective).
- Lookdown/lookup clutter (a low dragging jet is as visible as a high one).
- Radar on/off control (spec 5).
- Red missile launch warning.
- Flight-level reactions (a wingman defending on a flightmate's cue).
- The pre-planned maneuver itself (spec 4).

## What changes

- **Code:**
  - `sim/rwr.py` (new), `tactics/maneuvers.py` (new), `tactics/red_defense.py`
    (new, `DefenseConfig`)
  - `sim/world.py`: RWR step, early end, `SimResult.shots`
  - `sim/weapons.py`: a-pole and f-pole, `ground` outcome
  - `sim/world.py`: firing doctrines
  - `sim/aircraft.py`: `radar_emitting`, 100 m AGL floor
  - `tactics/interpreter.py`: Red controller uses the state machine; Blue
    doctrine switch
  - `sim/sensor_config.py`: RWR mode factors, new F-35 RCS table
  - scenario loader: `red_aggressiveness`
  - `acmi/exporter.py`: `rwr_mode`, `defend`, `recommit`, `press`, `depart`
    bookmarks
  - CLI: `defense-replays`, `defense-stats`
- **Tests:** existing tests pass unchanged with `red_defense=False` and the old
  shot doctrine. Any test that changes gets a stated reason.

## Test plan

1. **Unit tests.**
   - RWR: each mode, the 29 NM and ±60° FOR gates, `radar_emitting=False` gives no
     cue, highest mode wins, missile active only for the target.
   - Maneuver headings: crank ±50° the short way, beam 90°, drag 180° with the
     altitude target.
   - Band edges at a = 0, 0.33, 0.34, 0.66, 0.67, 1, and the `T_cold` values.
   - State machine: triggers per band, threat-clear and TOF logic, turn-away
     counting, press vs depart, early end.
   - Firing doctrines (SAS one in flight; SSA pair 3 s apart then hold).
   - 100 m floor, drag depth vs a, missile `ground` outcome.
   - A-pole and f-pole on a static geometry.
   - Same seed gives the same result.
2. **TacView replays** (`defense-replays -o spec3_outputs`), each with an event log
   including a-pole and f-pole:
   - **A.** Conservative Red (a = 0) drags on the first lock, clears, stays cold
     40 s, recommits.
   - **B.** Aggressive Red (a = 1) fires, then cranks on missile active while still
     supporting its own shot.
   - **C.** Middle Red (a = 0.5) beams on the support cue; its own missile coasts.
   - **D.** Turn-away limit: two cycles, then a = 0.8 presses and a = 0.2 departs.
   - **E.** Blue test reaction against a Red shot (crank, then drag).
   - **F.** Blue shoot-shoot-assess.
3. **100-seed stats** (`defense-stats`): `default_4v3`, default genome,
   a ∈ {0, 0.25, 0.5, 0.75, 1}, Blue test reaction off and on. Report:
   - Blue shots and hit rate, Red shots and hit rate, kills on each side
   - engagement duration (median, p90, % reaching the 240 s cap)
   - turn-aways, departures, defeat labels, a-pole and f-pole distributions

   Expected: as a falls, Blue hit rate and Red kills fall and fights get longer.
   Flag it if the median duration exceeds 200 s.
4. **Regression.** Re-run the spec 2 and 3a replays with defense off; results must be
   identical.

## Runtime estimate

- **Today:** 0.47 s per engagement (`default_4v3`, 240 s, measured today). Missile
  integration takes about 25 % of that and sensors about 50 %.
- **Added cost:** RWR modes are O(Blue × Red) lookups per step, and the state
  machine is O(n). Together about **+3–5 %**.
- **Saved cost:** one live missile per shooter per target cuts missile-seconds.
- **Duration:** fights cannot grow past the cap, and today's runs already reach
  240 s.
- **Net expectation: 0.42–0.52 s per engagement**, measured and reported.
- The envelope table is not rebuilt (no kinematic change, so the cache key is the
  same).
- The stats run (10 configurations × 100 seeds) takes about 8 min serial.

## Implementation (2026-09-25)

**Code.** `sim/rwr.py`, `tactics/maneuvers.py`, `tactics/red_defense.py`
(`DefenseConfig`, `RedDefense`), `analysis/blue_test_defense.py`,
`analysis/defense_replays.py`, `analysis/defense_stats.py`; changes in
`sim/world.py` (RWR step after sensors, doctrines, early end, `SimResult.shots` /
`red_shots` / `ended_early`), `sim/weapons.py` (poles, `ground`), `sim/aircraft.py`
(floor, `radar_emitting`, `firing_doctrine`, `defense_state`, `departed`),
`tactics/interpreter.py` (`RedCAPController(defense=...)`), scenario loader
(`red_aggressiveness`, `red_defense`, `blue_doctrine`, `red_doctrine`), GA config
(same four fields; default defense on, a = 0.5, SAS both sides), ACMI exporter,
CLI (`defense-replays`, `defense-stats`; `evolve`/`simulate` get
`--red-aggressiveness`, `--no-red-defense`, `--blue-doctrine`, `--red-doctrine`).
**Tests:** 138 pass (106 before; +32 in `tests/test_missile_defense_spec3.py`;
`tests/test_sensors_spec1.py` updated for change A).

**Deviations and judgment calls.**
- Blue test reaction lives in `analysis/blue_test_defense.py` (re-exported from
  `defense_replays`), not inside `defense_replays.py`. It drags with a = 0.5 depth.
- RWR bearings (5° noise) are smoothed with an EMA (weight 0.5) inside the defense
  so the crank/beam heading does not jitter; the crank/beam side (short way) is
  picked once when the defense starts and then held.
- SSA: if the jet's normal target choice changes between the two shots, the second
  shot is redirected to the salvo target (the pair always goes at one target).
- Shot table records the target's defense state at launch and at missile end.
- `a_pole_m` is recorded only if the shooter is alive when the seeker goes active.
- Regression: with `red_defense=False` and `legacy` firing the code is
  bit-for-bit identical to pre-spec-3 on 100/100 seeds of `default_4v3` **with the
  old RCS table**. With the new table (change A), default_4v3 outcomes are still
  identical; spec 3a replays identical; spec 2 hit rates identical (FC gap % and
  event timings move by ~1 point because the sensor RNG consumes different draws);
  spec 1 beam-then-hot shifts slightly (Red detection 28.9 → 28.7 NM).
- Replay B: Red's FC range vs an offset F-35 is only ~20 NM, so Red's shot goes
  active (186 s) before Blue's missile goes active and triggers the crank (188 s).
  The replay shows the crank keeping radar and FC and a **new shot fired while
  cranking** (M0003 at 216 s, supported), rather than supporting a pre-active
  missile through the crank.
- Within the aggressive band a = 0.75 and a = 1 give almost identical stats: only
  T_cold differs and a cranking jet in COLD still fires and supports.
- a ≤ 0.25: Red hears Blue's lock at 29 NM, before Blue's shot range
  (~0.75 × Rmax ≈ 24 NM), and drags; nobody shoots. Median duration > 200 s is
  flagged for a ≤ 0.5 (see stats). Tuning (Blue shooting on a remote/IRST track,
  shorter Blue lock range, fitness for chasing a runner) is left to specs 4–7.
- Stats vary Blue's doctrine (SAS/SSA); Red is SAS in all rows.

**Replays** (`spec3_outputs/defense_replays.txt` + ACMI; 1v1 unless noted).
- **A** (a = 0, seed 1): lock cue at 106 s (27.9 NM) → drag toward 100 m;
  threat cleared 154.5 s, cold 40 s, recommit 194.5 s at ~1,680 m; second lock →
  drag at 210 s (2/2), reaches the 100 m floor by 250 s. Blue never gets a shot.
- **B** (a = 1, seed 4): R1 fires at 177 s (19.8 NM), B1 fires back 177.5 s;
  R1 cranks on missile active at 188 s (1/2), both missiles miss on the Pk roll
  (Pk 0.59 / 0.40); R1 fires again while cranking (216 s); recommits after 5 s
  cold and cranks again at 231.5 s (2/2).
- **C** (a = 0.5, seed 4): R1 beams on the support cue at 178 s; its own missile
  loses support at 184.5 s and coasts to active (Pk factor 0.99); Blue's missile
  is defeated by speed at 228.5 s (Mach 1.20, 4.4 NM); cold 22.5 s.
- **D1** (a = 0.8, seed 7): cranks at 129 s and 197 s, recommits after 12 s cold;
  presses at 251.5 s on the third missile-active cue (M0005 defeated opening).
  No kills.
- **D2** (a = 0.2, seed 1): drags at 106 s (to 1,720 m) and 201.5 s (to 572 m),
  departs on the third lock at 293 s; engagement ends early at 293.5 s.
- **E** (Red defense off, seed 1): B1 fires at 170 s; R1 fires at 177 s; B1
  cranks 50° while supporting (177.5 s) – the lower aspect costs Red its FC, so
  Red's missile coasts – then drags from 183.5 s when its own missile goes active;
  Red's missile is defeated by speed at 237 s (6.1 NM). Blue's missile misses
  (Pk 0.43).
- **F** (Blue SSA vs two unarmed Reds, seed 2): M1/M2 at R1 at 105.5/108.5 s
  (29 NM); B1 holds fire for 54.5 s with a valid shot on R2; M1 hits R1 at 160 s;
  new pair at R2 at 160.5/163.5 s; M3 hits at 181.5 s. 2 kills, 4 shots.

**100-seed stats** (`spec3_outputs/defense_stats.txt`, `default_4v3`, default
genome, cap 240 s, Red SAS). Hit = hits/shots.

| Blue doct | test | a | B shots | B hit | R shots | R hit | B kills | R kills | dur med | p90 | % cap | turn-aways | departs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAS | off | 0 | 0 | – | 0 | – | 0 | 0 | 240 | 240 | 100 | 6.0 | 1.32 |
| SAS | off | 0.25 | 0 | – | 0 | – | 0 | 0 | 222.8 | 224.5 | 0 (100 % early end) | 6.0 | 3.0 |
| SAS | off | 0.5 | 12.05 | 23 % | 0.90 | 40 % | 2.80 | 0.36 | 204.2 | 240 | 18 | 4.44 | 0 |
| SAS | off | 0.75 | 10.28 | 29 % | 4.80 | 27 % | 2.93 | 1.29 | 143.2 | 155.6 | 5 | 3.04 | 0 |
| SAS | off | 1 | 10.27 | 29 % | 4.81 | 27 % | 2.93 | 1.29 | 143.2 | 155.6 | 5 | 3.04 | 0 |
| SAS | on | 0.5 | 11.44 | 23 % | 0.92 | 14 % | 2.59 | 0.13 | 205.5 | 240 | 33 | 4.45 | 0 |
| SAS | on | 1 | 8.89 | 33 % | 3.13 | 1 % | 2.97 | 0.04 | 146.2 | 165.3 | 2 | 3.11 | 0 |
| SSA | off | 0.5 | 13.45 | 16 % | 1.59 | 46 % | 2.20 | 0.73 | 240 | 240 | 63 | 4.10 | 0 |
| SSA | off | 1 | 15.85 | 19 % | 4.41 | 32 % | 2.94 | 1.39 | 145.0 | 150.6 | 3 | 2.97 | 0 |
| SSA | on | 0.5 | 13.91 | 16 % | 1.26 | 22 % | 2.23 | 0.28 | 240 | 240 | 63 | 4.14 | 0 |
| SSA | on | 1 | 13.56 | 21 % | 3.32 | 4 % | 2.89 | 0.13 | 145.5 | 240 | 11 | 2.98 | 0 |

a = 0 and 0.25 rows are identical for all four Blue settings. Median duration
> 200 s is flagged for every a ≤ 0.5 row. Pre-spec-3 (legacy firing, no defense):
Blue 16 shots at 12.5 %, Red 12 at 8.3 %, 2 kills / 1 loss every seed. Pole
medians (SAS, test off, a = 1): Blue launch/a-pole/f-pole 12.2/12.2/11.0 NM, Red
14.5/14.5/8.1 NM. Full distributions and defeat labels in the stats file.

**Runtime.** 0.44 s per engagement at the default (a = 0.5, SAS; pre-spec-3 0.47 s;
legacy + defense off 0.50 s, +5 % for RWR). Stats run: 2,000 engagements in 112 s
on a process pool. Smoke evolve (pop 12, gens 3, seed 42): 15.7 s wall, best
fitness 439.83 (3 kills, 0 losses, 139 s), reproducible.

**Outputs** (`/workspace/spec3_outputs/`): `defense_replays.txt`, `replayA`–`F`
ACMI files, `defense_stats.txt`, `regression.txt`, `smoke_evolve.txt`,
`rcs_change/` (aspect table, spec 2 re-run), `regression/`, `baseline_pre/`.
