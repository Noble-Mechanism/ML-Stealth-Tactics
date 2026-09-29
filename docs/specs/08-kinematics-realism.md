# Spec 8 — Kinematics realism (implemented)

Status: **implemented** (approved 2026-09-28; built 2026-09-28, finished and pushed 2026-09-29). Intent (Rusty 2026-09-26 / 2026-09-28): make
kinematic missile defeats more realistic so Blue has to rely more on stealth.
Overnight run observation: a bait F-35 held high/fast flank aspect, defeated
many close-in shots, and the rest of the flight held flank until squaring up to
shoot. Root cause in code: jets turn at a fixed rate with **no energy cost**,
while the missile model already has burn, drag, g-limits and Mach/opening
defeats (spec 3a) but **no loft**.

All of A–E landed: jet energy model (placeholders hit the A-d bleed gate with
no number changes), loft-bias midcourse + opening grace, defeat-reason rollup
in progress/champion stats, Blue deconfliction fitness term, and a
per-coalition missile config stub (Red = Blue).

## Scope

| # | Change | In this build? |
|---|---|---|
| A | Jet energy model (separate Blue / Red params) | Yes |
| B | Missile loft | Yes |
| C | Defeat-reason rollup in progress / replays | Yes |
| D | Blue deconfliction fitness penalty (5000 ft **and** 5 NM) | Yes |
| E | Superior Red missile (config only) | Setting stub; leave off until A–D prove insufficient |

Out of scope: real classified F-35 / AIM / adversary numbers (placeholders
only, same disclaimer as today), continuous RL, per-jet networks, chosen lock,
memory, handoff limits.

## Why the bait works today

`integrate_aircraft` (`sim/aircraft.py`):
- turn rate is a fixed `max_turn_rate_deg_s` (11°/s Blue, 13°/s Red) at any
  speed or altitude;
- turning does not change speed;
- speed slews at a flat ±30 m/s² toward the commanded speed, clipped to a
  fixed min/max, with no thrust, drag or altitude dependence.

So a jet can hold a hard flank turn indefinitely at Mach ~1 and 40,000 ft.
Close shots that would kill a real jet that bled energy are defeated by
`defeat_speed` / `defeat_opening` instead.

Missile side (already real-ish, spec 3a): motor burn, Mach/altitude drag,
induced drag, q-limited g, PN with **no loft**, outcomes `hit` / `miss` /
`miss_overshoot` / `defeat_speed` / `defeat_opening` / `timeout` /
`lost_basket` / `ground`. Outcomes are already on each shot event and in ACMI
bookmarks; overnight `progress.md` does not summarize them yet.

---

## A — Jet energy model

Replace the fixed turn-rate / flat-accel integrator with a simple energy model.
Still a point mass (no attitude dynamics). Commands stay heading / speed /
altitude; the physics decides what is achievable.

### A1. Equations (recommendation)

1. **Atmosphere.** Reuse the 1976 US Standard Atmosphere already in
   `missile_kinematics.atmosphere` (density ρ, speed of sound a).
2. **Load factor and turn.** Commanded heading change implies a horizontal
   turn rate ω. Convert to demanded load factor
   `n_horiz = V · |ω| / g`, then
   `n = sqrt(1 + n_horiz²)` (level turn). Cap by `n_max` from config
   (sustained-ish; not instantaneous). Achieved turn rate is
   `ω_max = g · sqrt(n_max² − 1) / V` (and 0 if V is tiny).
3. **Drag.** `D = q · S · Cd0 + k · (n · W)² / (q · S)` with
   `q = ½ ρ V²`, `W = m · g`. Cd0 is a single placeholder (no full Mach
   table for the jet in this spec).
4. **Thrust.** `T = T_sl · (ρ / ρ_sl)^α`, clipped so the jet cannot exceed
   `max_mach` at the current altitude (and a hard `max_alt_m`). Throttle is
   implicit: accelerate when commanded speed is above current, idle/drag when
   below. No afterburner schedule in this spec (one thrust number).
5. **Along-track acceleration.**
   `a = (T − D) / m − g · sin(γ)` where γ is the flight-path angle from the
   climb/descent toward `cmd_alt`. Climb rate is still limited, but climbing
   now costs energy through the `sin(γ)` term and through induced drag if
   turning.
6. **Speed limits.** Soft floor near corner speed / stall placeholder
   (`min_speed_mps`); hard ceiling from `max_mach · a(h)`.

All constants live in a new `AircraftEnergyConfig` (or fields on
`AircraftTypeParams`) keyed per type (`F35`, `RedFighter`).

### A2. Placeholder numbers (recommendation — **not** real F-35 data)

Tune so that at 25–40 kft the jet can still cruise near Mach 0.9, a hard turn
bleeds speed visibly within a few seconds, and a sustained turn at high
altitude is much slower than today's free 11°/s. Starting targets for Blue
(calibrate in a small sweep before merging):

| Param | Blue (F-35 label) | Red | Notes |
|---|---|---|---|
| mass_kg | 18000 | 18000 | same mass; performance differs via T/Cd |
| S_m2 | 40 | 40 | wing ref area placeholder |
| Cd0 | 0.025 | 0.028 | Red slightly dirtier |
| k_induced | 0.08 | 0.08 | same form as the missile |
| n_max | 7 | 8 | Red turns harder in the clear |
| T_sl_N | 110000 | 130000 | sea-level thrust placeholder |
| thrust_density_exp α | 0.7 | 0.7 | thrust falls with density |
| max_mach | 1.2 | 1.4 | Blue slower top end than Red |
| max_alt_m | 15000 | 15000 | keep today's ceiling |
| min_speed_mps | 90 | 85 | keep today's floors |

Exact numbers are calibration knobs. A short `aircraft-sweep` (or extend
`missile-sweep` with a turning target that now bleeds) should show: free
flank-turn bait at 40 kft no longer defeats a 10 NM shot the way it does
today.

### A3. Policy / network impact

Observables and outputs stay the same (spec 5). The network will feel slower
turns and speed bleed through state. No input-schema change. HandBlue and Red
doctrines keep commanding the same primitives; the integrator enforces the
new limits. Expect HandBlue scores and overnight champions to drop until the
population re-learns.

### A4. Open choices for Rusty

- **A-a.** Approve the energy model shape above (n-limited turn, drag, density
  thrust, climb costs energy).
- **A-b.** Blue and Red get **separate** param sets (recommended), vs one
  shared set.
- **A-c.** Keep a single `n_max` (sustained) for now, vs separate instantaneous
  / sustained. Recommendation: single `n_max`.
- **A-d.** Calibration target: "a 3–4 g turn at 40 kft from Mach 0.9 bleeds to
  ~Mach 0.7 within ~20 s" as the accept gate, with numbers adjusted to hit it.

---

## B — Missile loft

Spec 3a explicitly deferred loft. Without it, high-altitude shots stay at
launch altitude and low-altitude Rmax is short (~19 NM head-on at 15 kft in
the prototype).

### B1. Model (recommendation)

Simple loft bias on the midcourse aim point, no full optimal-control loft:

1. While range-to-aim > `loft_handoff_m` (default ~25 NM) and `tf` is past a
   short boost settle, bias the aim point upward by
   `loft_angle_deg` (default ~20°) along the line of sight, decaying linearly
   to 0 as range reaches the handoff.
2. After handoff (or once active), pure PN on the real aim / target as today.
3. All loft knobs on `MissileKinematicsConfig` (`loft_enabled`,
   `loft_angle_deg`, `loft_handoff_m`). Default **on** for both sides (same
   missile).
4. Rebuild the Rmax / Rne table with loft on; shoot rules and network Rmax
   inputs pick it up automatically.

### B2. Open choices

- **B-a.** Approve the simple loft-bias model (vs a more complex loft-to-
  altitude then dive).
- **B-b.** Default loft on for both sides (recommended), vs off until
  calibrated.
- **B-d. Opening defeat gets a grace window** (Rusty 2026-09-28). Today a
  missile past burnout is lost the first 0.05 s step its closure goes
  negative. With loft, closure can briefly open near the loft peak against a
  dragging target, and the missile could still reach if the target turns
  back hot. Recommendation: `defeat_opening` fires only after closure has
  been negative **continuously for 3 s** (`opening_grace_s`), and never
  while the loft bias is still active. `defeat_speed` (below Mach 1.2) stays
  immediate, since a slow missile cannot regain energy. `miss_overshoot`
  (passed within 1 km) stays immediate. The Rmax table uses the same rule.
- **B-c.** Accept that loft will raise high-alt Rmax and help low-alt shots;
  re-run `missile-sweep` and update the calibration tables in 03a notes.

---

## C — Defeat-reason rollup

Outcomes already exist on every shot. Add visibility.

### C1. Recommendation

1. Per fight (and therefore per presentation in neuro eval): count Blue-as-
   target and Red-as-target outcomes by label (`defeat_speed`,
   `defeat_opening`, `miss_overshoot`, `hit`, `miss`, `timeout`, …).
2. Overnight `progress.md`: a short table of champion-fight missile outcomes
   (and optionally hall-of-fame cells).
3. `replay-champion` / ACMI: keep existing bookmarks; no change required
   beyond making sure every terminal outcome is still bookmarked (already
   true).
4. Optional: one line in the champion JSON under `stats.missile_outcomes`.

No new defeat physics. This answers "what defeated the bait's close-in
shots?" from the overnight data.

### C2. Open choice

- **C-a.** Approve the progress rollup as above.

---

## D — Blue deconfliction penalty

Rusty 2026-09-28: if Blue jets go inside **5000 ft and 5 NM** of one another
there should be a fitness penalty. Emphasis on **and** (both must be true).
Rationale: at those ranges human pilots worry more about deconfliction than
tactics.

### D1. Definition (recommendation)

A Blue–Blue pair is **in conflict** at a 1 s sample when:
- horizontal range < **5 NM**, **and**
- altitude difference < **5000 ft**.

Both thresholds are config keys (`deconflict_nm`, `deconflict_alt_ft`).

### D2. Fitness term (recommendation)

- Each second any Blue pair is in conflict: add
  `deconflict_per_s` (default **−1**) to that fight's score.
- Cap per fight at `deconflict_cap` (default **−50**) so one tangled fight
  cannot dominate, but sustained mushing is still expensive.
- Keys in `scenarios/fitness.yaml`; part of the config fingerprint (resume
  refuses a weight change, same as today).
- Uses truth geometry from the behaviour recorder (scorer only — never a
  policy input), same as the egress term.
- Only Blue–Blue. Red packing is free (their problem / doctrine).

### D3. Open choices

- **D-a.** Approve the AND rule (5 NM horizontal **and** 5000 ft vertical).
- **D-b.** Scoring: −1 per conflict-pair-second with a −50 cap
  (recommended), vs a flat per-fight penalty on "any conflict", vs continuous
  with no cap.
- **D-c.** Default weights: `deconflict_per_s: -1`, `deconflict_cap: -50`.

---

## E — Superior Red missile (stub only)

If A–D are not enough to stop free kinematic baiting, a better Red missile
should be a **config choice**, not a rewrite: e.g. lower `cd_scale`, higher
`propellant_kg` / `isp_s`, or a second `MissileKinematicsConfig` selected per
coalition. Recommendation for this build: add the per-coalition config hook
and leave Red = Blue. Do not tune a "superior" Red missile until we see post-
A/B overnight behaviour.

### E1. Open choice

- **E-a.** Stub the per-coalition missile config, keep Red = Blue for now
  (recommended), vs skip the stub entirely until needed.

---

## What changes (when built)

- **Code:** `sim/aircraft.py` (energy integrator), `sim/sensor_config.py`
  (energy + loft knobs; optional per-coalition missile),
  `sim/missile_kinematics.py` / `weapons.py` / `missile_envelope.py` (loft),
  `fitness.py` + `scenarios/fitness.yaml` (deconfliction),
  `neuro/overnight.py` + eval stats (outcome rollup), small CLI sweep for
  jet bleed calibration.
- **Tests:** energy turn-bleed gate; loft increases Rmax vs loft-off;
  deconfliction AND rule and cap; outcome rollup; regression of existing
  suite (HandBlue / Red doctrines still run; expect score shifts).
- **Docs:** this file; short notes in `design.md` and 03a calibration.
- **Runs:** new overnight `-o` folder after fitness keys change; old
  checkpoints will not resume under the new fingerprint.

## Test plan (build time)

1. Jet bleed: from Mach 0.9 / 40 kft, command a max-rate turn, report speed
   vs time; gate per A-d.
2. Missile loft: `missile-sweep` loft on vs off; Rmax tables refreshed.
3. Bait sanity: scripted high-fast flank bait vs a 10–15 NM shot; with A+B,
   close-in defeat rate should fall vs today's build (report both).
4. Deconfliction: two Blue jets inside / outside each threshold alone, and
   both; only the AND case scores the penalty; cap binds.
5. Progress rollup: synthetic fight with known outcomes appears in
   `progress.md`.
6. Full pytest regression.

## Decision checklist (for widgets)

| ID | Recommendation |
|---|---|
| A-a | Energy model shape as written |
| A-b | Separate Blue / Red param sets |
| A-c | Single `n_max` (no inst/sustained split yet) |
| A-d | Bleed accept gate ~Mach 0.9→0.7 in ~20 s at 40 kft, 3–4 g |
| B-a | Simple loft-bias midcourse model |
| B-b | Loft on by default, both sides |
| B-c | Re-calibrate Rmax tables after loft |
| B-d | Opening defeat only after 3 s continuous opening, not during loft |
| C-a | Outcome rollup in progress / champion stats |
| D-a | AND rule: <5 NM and <5000 ft |
| D-b | −1 / conflict-pair-second, cap −50 |
| D-c | Those defaults in `fitness.yaml` |
| E-a | Per-coalition missile config stub; Red = Blue for now |

## References

- Existing missile model: `docs/specs/03a-missile-kinematics.md`.
- Fitness extensibility: `docs/specs/07-fitness.md`, `scenarios/fitness.yaml`.
- U.S. Standard Atmosphere 1976 (already used by missiles).
- Numbers above are unclassified calibration placeholders, not sourced
  aircraft or missile performance data.


## Implementation notes (2026-09-28, finished 2026-09-29)

### Numbers (verified 2026-09-29)

- **A-d bleed gate** (`aircraft-sweep`, Blue placeholders unchanged, 40 kft,
  Mach 0.9, 20 s turn): n_max 3.0 -> Mach 0.795; **3.5 -> 0.675 (gate)**;
  4.0 -> 0.539. At the default n_max 7 the jet is at the speed floor
  (90 m/s, ~Mach 0.31) within ~10 s.
- **Sustained turn at 40 kft** (default n_max 7, continuous turn, 30 s): 117°
  total (legacy integrator: 330° at 11°/s). At 15 kft: 532° in 30 s.
- **Rmax** (sim path, shooter M0.9 vs M0.9 level, co-altitude, NM), loft off -> on:
  40 kft hot 49.4 -> **82.5**, beam 29.6 -> 34.8, turn-cold 24.4 -> 24.4,
  cold 21.1 -> 21.1; 25 kft hot 28.8 -> 33.0; 15 kft hot 20.5 -> 20.5 (loft
  never engages under the 25 NM handoff). Table vs sim path agrees within the
  original 0.3 / 0.5 NM gates at every tested point.

### Changes made while finishing the build (2026-09-29)

- **Loft decay (bug fix).** The first build ramped the loft angle from full at
  2 x handoff (50 NM) to 0 at handoff, keyed on the current range. Shots
  launched between ~1 and 2 x handoff got a shallow, energy-wasting loft, so
  Rmax was not monotone in range (9 km hot: hits to 35 NM, misses 36-53 NM,
  hits again 54-64 NM; 42 of 378 sampled geometries had such holes). The
  table's "largest hit" then advertised shots that fail. Now, as B1 reads, the
  bias is `loft_angle_deg` at loft start and decays linearly to 0 at handoff
  (start range stored per missile / per batch shot). Mid-range holes are gone
  (only the known sub-5 NM minimum-range holes remain). This moved the 40 kft
  hot Rmax from ~78.3 (first build) to 82.5 NM. `ENGINE_VERSION` "8.1".
- **Aspect bins 10° to 120°** (were 20°). They were needed with the first-build
  loft (Rmax cliffs vs aspect, up to ~11 NM interpolation error). They are kept
  for table fidelity. Table build: ~6 min on 8 cores (was ~3.4 min).
- **Soft speed floor (A1.6).** The first build clipped speed at
  `min_speed_mps` but still let the jet pull n_max there. At 40 kft that gave
  ~43°/s turns at the floor (1163° in 30 s vs 330° legacy), the opposite of A2.
  Now, at the floor, n is capped at what full thrust sustains (1 g = wings
  level if even that cannot be held).
- **Turn-direction hysteresis.** Within 10° of a reversal command, keep turning
  the current way, and break an exact-reversal tie to the right when flying
  straight. Rounding-level command differences (spec 5 layer-1 encode/decode)
  no longer flip the turn direction. Layer 1 is back to ACMI and result
  identical on all 20 seeds.
- **Deconfliction counts pair-seconds** (D-b: -1 per conflict-*pair*-second).
  The first build counted seconds with any pair in conflict.
- **Missile replay (3a)** `turn_g`: it now sets energy `n_max = sqrt(1 + turn_g²)`,
  because the energy integrator ignores `max_turn_rate_deg_s`.

### Test expectations changed by the physics (not weakened)

- 40 kft calibration / envelope tests: new loft-on values (hot 82.5, beam 34.8,
  turn-cold 24.4, cold 21.1; hot 50° off ~71.5). Table-vs-sim gates are back to
  the original 0.3 / 0.5 NM.
- 40 NM head-on at 40 kft: TOF ~69.6 s, end Mach ~2.1 (lofted, was ~M1.5 flat).
  76 NM / 15 km / M1.3: fuzes at ~122 s (was ~140 s; still times out with a
  120 s cap).
- Spec 2 lead-trail replay test: the lead's table Rmax is now ~42-45 NM (was
  ~31), so a trail 15 NM back is outside its own 50 NM FC gate for about the
  whole 40 s coast timeout. The test flies the trail 4 NM back so it still
  checks a handoff. The CLI replay keeps 15 NM, and there the "support"
  variant now shows a coast-timeout loss.
- Regression fingerprint (defense off, old RCS): re-baselined (trajectories
  change).
- `test_energy_turn_slower_at_high_alt_than_legacy_fixed_rate` (A2) is
  restored, plus soft-floor, loft-decay, Rmax monotonicity and pair-second
  tests.

### Other

- **Opening grace:** 3 s continuous opening, never while loft bias active;
  `defeat_speed` / `miss_overshoot` immediate (sim and table).
- **Fitness:** `deconflict_nm/alt_ft/per_s/cap` in `scenarios/fitness.yaml`
  (in the config fingerprint; old checkpoints will not resume).
- **CLI:** `aircraft-sweep` for the bleed gate; `missile-sweep` cal targets updated.
- **Bait sanity** (test plan 3; scripted, not a policy): a Red shooter at M0.9
  fires at a 40 kft M0.95 F-35 bait from 10-15 NM. At launch the bait turns at
  max rate to flank (beam the missile) or drag, at max commanded speed.
  With the shooter at 40 kft, every shot fuzes, old and new, from both hot and
  beam starts. With the shooter at 25 kft, the old build lost 24 of 96 shots
  to `defeat_speed` (all drag cases: 8/24 from hot, 16/24 from beam). The new
  build lost 0 of 96. The drag turn now costs the bait its energy (minimum
  Mach 0.31-0.62 vs >= 0.95 before).
- Per fight wall time (24 spec 4 seeds, box, single process): HandBlue 1.44 ->
  1.59 s (+11%), random MLP 1.25 -> 1.30 s (+4%). Per 100 sim-s: +14% / +4%.
