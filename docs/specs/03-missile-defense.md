# Spec 3 — Missile defense (RWR modes, Red reactions, maneuver primitives)

Status: **implemented (2026-09-25).** Rusty approved D1–D10 exactly as recommended,
plus changes **A–C** (F-35 RCS table, 100 m AGL floor with aggressiveness-based drag
depth, per-jet firing doctrines). **Changed 2026-09-26 (approved change D):**
one defensive reaction per Red jet, then every jet presses; see "Approved change D".
See "Implementation" at the end for deviations and results. All numbers are prototype placeholders. Spec 3a (missile
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
  hard sim time limit stays. *Superseded 2026-09-26 by approved change D: limit 1,
  then every jet presses; aggressiveness sets how early and how hard Red defends,
  not whether it stays.*
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
  PRESSING (ignores cues, stays hot and shoots). Since change D (2026-09-26) this
  holds for every `a`; before it, `a < 0.5` went to DEPARTED (D1/D10). The only
  way into DEPARTED now is spec 4 rule L (out of missiles, none of its own in
  flight).
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
- At most one live missile per shooter per target — now the per-contact
  `shoot_assess_shoot` doctrine of change C, with `shoot_shoot_assess` as the
  pair alternative. (Before spec 3 every jet rippled all 4 missiles in 2 s:
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

**D10 — Turn-away limit: default 2 (approved; superseded by change D, now 1).**
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
Both doctrines are **per contact** (follow-up approved by Rusty 2026-09-25; the
first implementation was global, i.e. SAS = one missile in flight overall and SSA =
nothing at anyone until the pair resolved).
- `shoot_assess_shoot` (SAS): at most **one own missile in flight per target**. A
  jet can engage several contacts at once, one missile each, and cannot re-fire
  at a target until its missile at that target resolves.
- `shoot_shoot_assess` (SSA): a pair at one target, the second 3 s after the
  first (`SimConfig.ssa_interval_s`); then no more shots at **that** target until
  both resolve. The jet may immediately engage a different target with its own
  pair (inventory and the normal launch gates still apply).
- *Why per contact:* the network (specs 5/6) should decide when and how many
  shots come off; the doctrine only limits missiles per contact.
- Defaults: Red SAS (spec 4 will randomize), Blue configurable
  (`SimConfig.blue_doctrine`, default SAS; spec 5 may make it a network output).
  Per-jet override `Aircraft.firing_doctrine`.
- Replaces the old "ripple all 4 in 2 s" behaviour, which remains as `legacy`
  for regression tests only.

## Approved change D — one reaction, then press (Rusty, 2026-09-26)

**Change.**
- Turn-away limit goes from 2 to **1**: each Red jet gets exactly **one**
  defensive reaction (drag, beam or crank as its band dictates; cranks count).
- After that one reaction, **every** Red jet presses on the next trigger,
  regardless of aggressiveness (the `a < 0.5` depart branch is removed).
  Aggressiveness now controls **how early and how hard** Red defends (trigger
  level, reaction, drag depth, `T_cold`), not whether it stays.
- The only reason a Red jet departs is being out of missiles with none of its
  own in flight (spec 4 rule L). The D9 "every live Red departed" early end
  stays in the code but can now only fire through those Winchester departures;
  `end_reason = red_departed` keeps meaning "every live Red departed".
- The 360 s cap (spec 4) is unchanged.

**Rusty's reasoning.** Conservative Red was dragging twice and going home with
no shots in about a third of presentations, producing no-kill fights. One
reaction then press keeps fights meaningful, and teaches the network that the
first shot makes a conservative flight turn and the follow-up kills.

**As built.**
- `DefenseConfig.turn_away_limit = 1` and a new `depart_after_limit = False`.
  The old behaviour is kept, for reference only, as `DefenseConfig.spec3()`
  (limit 2, `depart_after_limit=True`, press if `a >= press_threshold`). Spec 3
  replays D1/D2 (`defense-replays`) are pinned to `DefenseConfig.spec3()` so they
  still show the retired press/depart split; everything else (YAML scenarios
  with defense on, GA, presentations) uses the new default.
- **Legacy/regression mode is untouched.** The exact-match regression
  (`test_regression_defense_off_legacy_old_rcs_matches_pre_spec3`) runs with
  Red defense **off** and `legacy` doctrine, so the state machine never runs and
  the fingerprint is unchanged; no extra gate was needed.
- Tests: every band presses after one reaction; a conservative jet drags once,
  then presses and never departs over 360 s while it has missiles; a second
  threat (new emitter, missile active) after the one reaction does not start
  another turn-away; a pressing jet out of missiles still departs
  (`force_depart`); in presentations only Winchester departures occur. The
  spec 3 press/depart test runs on `DefenseConfig.spec3()`.
- Effect on the spec 4 stats (100 presentations, default Blue): see
  `docs/specs/04-red-presentations.md`, "Approved change 2026-09-26".

## Deferred

- Notching and chaff/countermeasures (sensor model has no Doppler or clutter;
  chaff must never be very effective).
- Lookdown/lookup clutter (a low dragging jet is as visible as a high one).
- **Follow-up (intentional):** conservative Red (a < 1/3) drags on Blue's lock
  before Blue reaches shot range, so the default genome gets no shots and fights
  run to the cap. This is left for the network (specs 5/6) to solve (e.g. shoot
  on a remote/IRST track, delay the lock, pincer). Fallback if it cannot:
  increase the sim time cap.
  *(2026-09-26: change D means conservative Red now drags once and then presses
  back in, so these fights produce shots and kills; the network still gains by
  making the first shot count.)*
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
   - Firing doctrines per contact (SAS one per target, several contacts at once;
     SSA pair 3 s apart, R1 blocked while its pair flies, R2 engaged at once).
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
3. **100-seed stats** (`spec3_outputs/defense_stats.txt`, `default_4v3`, default
genome, cap 240 s, Red SAS). Per-contact doctrines (current). The previous global
doctrine run is kept in `defense_stats_global_doctrine_old.txt`; old values in
brackets. Hit = hits/shots.

| Blue doct | test | a | B shots | B hit | R shots | R hit | B kills | R kills | dur med | p90 | % cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| any | any | 0 | 0 | – | 0 | – | 0 | 0 | 240 | 240 | 100 |
| any | any | 0.25 | 0 | – | 0 | – | 0 | 0 | 222.8 | 224.5 | 0 (100 % early end) |
| SAS | off | 0.5 | 11.07 [12.05] | 18 % [23] | 1.63 [0.90] | 39 % [40] | 2.04 [2.80] | 0.64 [0.36] | 240 [204.2] | 240 | 74 [18] |
| SAS | off | 0.75/1 | 14.25 [10.3] | 20 % [29] | 9.80 [4.8] | 10 % [27] | 2.88 [2.93] | 1.02 [1.29] | 122.8 [143.2] | 147.1 [155.6] | 7 [5] |
| SAS | on | 0.5 | 11.30 [11.44] | 18 % [23] | 1.12 [0.92] | 10 % [14] | 2.02 [2.59] | 0.11 [0.13] | 240 [205.5] | 240 | 74 [33] |
| SAS | on | 0.75/1 | 12.69 [8.9] | 22 % [33] | 9.28 [3.15] | 0 % [1–2] | 2.77 [2.98] | 0.00 [0.05] | 123.5 [146.2] | 240 [165.3] | 24 [1–2] |
| SSA | off | 0.5 | 13.92 [13.45] | 17 % [16] | 1.17 [1.59] | 44 % [46] | 2.30 [2.20] | 0.52 [0.73] | 240 [240] | 240 | 55 [63] |
| SSA | off | 0.75/1 | 16.00 [15.85] | 12 % [19] | 10.72 [4.42] | 26 % [32] | 1.99 [2.94] | 2.78 [1.40] | 240 [145.0] | 240 [150.6] | 84 [2–3] |
| SSA | on | 0.5 | 14.13 [13.91] | 16 % [16] | 1.05 [1.26] | 8 % [22] | 2.31 [2.23] | 0.08 [0.28] | 240 [240] | 240 | 55 [63] |
| SSA | on | 0.75/1 | 16.00 [13.57] | 12 % [21] | 10.69 [3.30] | 0 % [3–4] | 1.96 [2.89] | 0.00 [0.11] | 240 [145.5] | 240 | 100 [11] |

- Median duration > 200 s is flagged for every a ≤ 0.5 row and now also SSA at
  a ≥ 0.75.
- Per contact, every jet spreads missiles over all contacts in range: Red at
  a ≥ 0.75 fires 9–11 of its 12 missiles (was 3–5) at lower Pk.
- With SSA, Blue empties all 16 missiles in pairs early (12 % hits), kills ~2 and
  then can't finish: 84–100 % of fights hit the cap. In 20-seed spot checks at
  a = 1, SSA ends 16 Blue shots, 2 kills, at the cap in most seeds; SAS ends with
  3 kills before the cap.
- a = 0 and 0.25 are unchanged (no shots). a = 0.75 and 1 are identical to the
  printed precision.
- Pre-spec-3 (legacy firing, no defense): Blue 16 shots at 12.5 %, Red 12 at
  8.3 %, 2 kills / 1 loss every seed.
- Pole medians (SAS, test off, a = 1): Blue launch/a-pole/f-pole 23.6/19.8/12.5 NM
  (was 12.2/12.2/11.0: first shots now come off at long range on several
  contacts), Red 14.6/14.6/8.6 NM. Full distributions and defeat labels are in the
  stats file.

**Runtime.** 0.49 s per engagement at the default (a = 0.5, per-contact SAS; mean
fight 229 s; SSA 0.46 s; pre-spec-3 0.47 s; legacy + defense off 0.50 s, +5 % for
RWR). Stats run: 2,000 engagements in 122 s on a process pool. Smoke evolve
(pop 12, gens 3, seed 42): 15.4 s wall, best fitness 442.42 (3 kills, 0 losses,
123.5 s), reproducible (global-doctrine run: 439.83).

**Outputs** (`/workspace/spec3_outputs/`): `defense_replays.txt`, `replayA`–`F`
ACMI files, `defense_stats.txt`, `defense_stats_global_doctrine_old.txt`, `regression.txt`, `smoke_evolve.txt`,
`rcs_change/` (aspect table, spec 2 re-run), `regression/`, `baseline_pre/`.
