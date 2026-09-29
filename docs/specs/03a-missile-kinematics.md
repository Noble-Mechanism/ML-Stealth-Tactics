# Spec 3a — Missile kinematics

Status: **implemented** (approved by Rusty as written, with the prototype
calibration 50 / 24 / 20 NM), plus the approved **3a fix** (launch off-nose
axis in the Rmax table, 40 s coast timeout and coast-Pk time scale, 180 s
flight cap; see "3a fix" in the implementation notes). Parameters: `SensorConfig.missile_kinematics`
(`MissileKinematicsConfig`) and `SensorConfig.missile` (`MissileCoastConfig`) in
`stealth_tactics/sim/sensor_config.py`. See **Implementation notes** at the end
for deviations and results.

## What it does

The current missile flies at a constant 900 m/s along a fixed flight time, with a fixed max range and a flat Pk. It cannot run out of energy, so beaming or dragging never defeats a long shot. This spec replaces it with a rough point-mass fly-out model. Max range stops being a setting and falls out of the physics, which creates the gap between Rmax and no-escape range that skate vs banzai depends on.

Blue and Red carry the **same missile** (same kinematics, same base Pk), so results measure sensors and stealth rather than weapons.

## Model (decided with Rusty)

The missile is a point mass flown in 3D at a 0.05 s internal step.

1. **Launch.** Starts at the shooter's position and velocity, so a high and fast shooter hands over more energy.
2. **Boost.** Solid motor, 6 s burn, specific impulse 250 s, 50 kg propellant out of a 161.5 kg launch mass. That adds roughly Mach 2.9 on top of launch speed. Peak is about Mach 3.8 from a Mach 0.9 launch at 40,000 ft.
3. **Drag.** `D = q · S · Cd0(M) · cd_scale + induced`. `S` is the frontal area of a 178 mm body. `Cd0` is 0.35 subsonic, rising to 0.70 at Mach 1.1, then falling as `(1.1/M)^0.6`. `cd_scale` is the calibration knob: 1.25 in the prototype (no gravity), **1.10 in the sim** (see Implementation notes). Air density and speed of sound come from the 1976 US Standard Atmosphere, so thin air at altitude means much less drag.
4. **Maneuver energy bleed.** Induced drag is `k · (n · m · g)^2 / (q · S)` with `k` = 0.08. Hard turns cost speed, and they cost more when the missile is slow or in thin air.
5. **Turn limit.** `n_max = min(40 g, q · S · CLmax / (m · g))` with `CLmax` = 12, so the missile can pull less as it slows or climbs.
6. **Guidance.** Proportional navigation with N = 4. The commanded acceleration is proportional to the line-of-sight rotation rate times the closing speed, clipped to `n_max`. Midcourse it steers at the supported (or coasting) aim point. After going active it steers at the true target. There is no lofting; that stays a future setting.
7. **Kinematic defeat.** After burnout the missile is lost if it drops below **Mach 1.2**, or if it is opening from the target (it can no longer close). This replaces the old fixed flight-time timeout. A 180 s hard cap remains as a safety net (3a fix: was 120 s, which bound some high-altitude shots before energy did).
8. **Gravity.** The missile carries lift equal to its weight, which adds a small induced-drag cost. It holds the launch altitude unless guidance commands a climb or dive.

Every constant above lives in config (`MissileKinematicsConfig`).

## Calibration (prototype results)

The target flies level at Mach 0.9. Numbers come from a throwaway prototype (`/workspace/scratch/proto.py`) with the constants above, so they are **prototype values**, not sourced figures.

Rmax in NM at 40,000 ft, shooter at Mach 0.9:

| Target behavior | Prototype | Your starting target |
|---|---|---|
| Hot, head-on | 49.0 | ~50 |
| Beam | 28.8 | not set |
| Turns cold at launch (3 g turn, no acceleration) | 23.7 | ~20 (no-escape range) |
| Already cold (tail chase) | 20.4 | ~15 |

**My earlier placeholders contradict each other.** A target that turns cold at launch has to be harder to kill than one that's already cold at a shorter range. So a no-escape range of 20 NM can't coexist with a 15 NM tail chase. The prototype gives 50 / 24 / 20. Pushing the tail chase down to 15 NM with more drag also drops head-on Rmax to about 38 NM. See open question 1.

Hot and cold Rmax in NM, by altitude and launch Mach:

| Altitude | Mach 0.7 | Mach 0.9 | Mach 1.3 |
|---|---|---|---|
| 15,000 ft | 17.6 / 6.6 | 18.6 / 7.2 | 20.6 / 8.5 |
| 25,000 ft | 25.0 / 9.7 | 26.4 / 10.6 | 29.1 / 12.4 |
| 40,000 ft | 46.5 / 18.8 | 49.0 / 20.4 | 53.7 / 23.5 |
| 45,000 ft | 58.2 / 23.7 | 60.0 / 25.5 | 63.5 / 29.1 |

Altitude has a strong effect, because the missile stays at launch altitude with no loft. Launch speed is worth about 10% going from Mach 0.9 to Mach 1.3.

Example fly-outs at 40,000 ft and Mach 0.9:
- A hot shot at 40 NM hits after 75 s at Mach 1.55.
- A 20 NM shot on a target that turns cold at launch hits after 60 s at Mach 1.8.

## Kill probability

Base Pk is **0.60 for both sides** (Red was 0.50). Final Pk is the base value times the factors below.

1. **Endgame energy.** `f_E` is 1.0 at Mach 2.0 or faster at intercept, falling linearly to 0.6 at Mach 1.2. A slow missile at the end of a long shot is less lethal.
2. **Supported all the way to impact:** no further factor.
3. **Supported to active (15 NM), then support drops:** `× 0.90`, applied once if support ends at any time between going active and impact. Recommendation: any flightmate can keep support, and it counts as continuous if gaps are shorter than 2 s.
4. **Not supported to active (coasting at handoff):**
   `f_coast = exp(-(e / 2 km)^2) × (1 - 0.15 · min(t_gap / 40 s, 1))` (3a fix: time scale 40 s, was 20 s)
   Here `e` is the aim-point error from the true target at the 15 NM handoff, and `t_gap` is the time since the last update. Example values:
   - 0.5 km: 0.94. A target flying straight barely matters.
   - 1 km: 0.78.
   - 2 km: 0.37. A target that maneuvered is punished hard.
   - 3 km: 0.11.
   The time term alone costs at most 15% at the 40 s timeout (7.5% at 20 s), so the aim-point error sets most of the penalty. This replaces `exp(-t/12 s)`.
5. **Kept rules:** the missile is lost after more than 40 s unsupported (3a fix: was 20 s; long shots need more than 20 s to reach the 15 NM active point), or if the aim point is more than 5 km from the target at handoff. The formula already makes a 5 km error nearly zero Pk. I recommend keeping the explicit loss anyway, because it gets its own label in logs and replays.

## Launch decision

The scripted shoot rule needs to know whether a shot can reach. Recommendation: precompute an **Rmax lookup table** from this model when the sim starts, indexed by altitude, shooter Mach, target aspect, target Mach and (3a fix) launch angle off the shooter's nose, then interpolate it. Shots are allowed inside the table's Rmax, and the old fixed `missile_range_m` goes away. The table's Rmax and its turn-cold value (no-escape range) become candidate network inputs in spec 5, so the network can learn when to shoot.

## Test plan

1. **Shot sweep.** Launch range 10–60 NM in 5 NM steps. Shooter Mach 0.7, 0.9, 1.1, 1.3. Altitude 15,000, 25,000, 35,000, 45,000 ft. Target hot, beam, turning cold at launch, and already cold. Report per shot: outcome (hit, miss, or kinematic defeat), time of flight, missile Mach at the active point and at intercept, a-pole, f-pole, and final Pk. Output tables plus an Rmax and no-escape summary.
2. **TacView replays.** A long head-on shot that hits, the same shot defeated by a drag, a beam defeat, and a lead-trail re-run with the new missile.
3. **Support rules.** Unit tests for full support, supported to active then dropped (0.9), coasting against a straight target versus a maneuvering one, and both loss rules.
4. **Regression.** All existing tests either pass or are updated with a stated reason. Re-run the Spec 2 replays (launch on remote, lead-trail). Report runtime per engagement against today's 0.31 s.

## What changes

- **Code:** `sim/weapons.py` (new fly-out and Pk model), `sim/aircraft.py` (drop `missile_range_m` and per-side Pk in favor of the shared config), `sim/sensor_config.py` (new `MissileKinematicsConfig`), a new `sim/missile_envelope.py` for the Rmax table, `acmi/exporter.py` (missile speed in replays), and the CLI (`missile-sweep` scenario).
- **Tests:** the missile datalink and coast tests (the Pk formula changes) and any test that assumes 900 m/s or a fixed range.
- **Expected side effect:** shot ranges and hit rates in the Spec 2 replays will change.

## Open questions for Rusty

1. **Calibration conflict.** Keep the prototype's 50 / 24 / 20 NM (head-on, turn cold at launch, already cold), or add drag to get the tail chase closer to 15 NM at the cost of head-on Rmax?
2. **Altitude effect.** At 15,000 ft, head-on Rmax is about 19 NM. Is that steep a drop acceptable for now, with lofting later?
3. **Endgame energy factor.** Is a 0.6 floor at Mach 1.2 right, or should a slow missile be punished more or less?

## References

- AIM-120 AMRAAM, Wikipedia: https://en.wikipedia.org/wiki/AIM-120_AMRAAM. Used for launch mass (161.5 kg) and diameter (178 mm) only. Motor, drag, and turn constants are calibration choices.
- U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF). Used for the troposphere and lower-stratosphere density and speed-of-sound formulas.
- propNav, a proportional navigation reference implementation: https://github.com/gedeschaines/propNav. Used for the structure of the guidance law.
- BVRGym (JSBSim-based BVR environment with an AIM model): https://github.com/xcwoid/BVRGym. Reviewed for how an open project structures missile fly-out; no numbers taken.
- bvr-marl: https://github.com/simldl/bvr-marl. Reviewed as a comparable RL air-combat baseline; no numbers taken.

## Implementation notes (as built)

Code: `sim/missile_kinematics.py` (atmosphere, drag, PN step; scalar + numpy
twins), `sim/missile_envelope.py` (Rmax / Rne table), `sim/weapons.py`
(fly-out, support, Pk), `sim/aircraft.py` (per-type `missile_range_m` /
`missile_pk` removed), `tactics/interpreter.py` (shoot ranges scale the table
Rmax), `acmi/exporter.py`, `analysis/missile_sweep.py`,
`analysis/missile_replays.py`, `analysis/off_nose_check.py` (3a fix), CLI
`missile-sweep`.

### Calibration

The sim carries lift equal to weight (item 8), which the prototype did not.
With the prototype's `cd_scale` 1.25 that costs about 10 % of range (head-on
44.1 NM), so **`cd_scale` was retuned to 1.10** (the only constant changed).
Without gravity the sim reproduces the prototype within 0.2 NM
(48.8 / 28.6 / 23.6 / 20.2).

Rmax at 40,000 ft, shooter Mach 0.9, target level Mach 0.9 (sim path, NM),
unchanged by the 3a fix:

| Target behavior | Sim | Goal | Prototype |
|---|---|---|---|
| Hot, head-on | 49.4 | ~50 | 49.0 |
| Beam | 29.6 | – | 28.8 |
| Turns cold at launch (Rne) | 24.4 | ~24 | 23.7 |
| Already cold | 21.1 | ~20 | 20.4 |

Hot / cold Rmax by altitude and launch Mach (NM, sim path, 180 s cap):

| Altitude | Mach 0.7 | Mach 0.9 | Mach 1.1 | Mach 1.3 |
|---|---|---|---|---|
| 15,000 ft | 19.4 / 7.2 | 20.6 / 8.0 | 21.7 / 8.8 | 22.8 / 9.5 |
| 25,000 ft | 27.2 / 10.6 | 28.8 / 11.6 | 30.3 / 12.6 | 31.9 / 13.7 |
| 35,000 ft | 38.5 / 15.7 | 40.8 / 17.2 | 43.0 / 18.6 | 45.2 / 20.1 |
| 40,000 ft | 46.5 / 19.2 | 49.4 / 21.1 | 52.2 / 23.0 | 54.9 / 24.8 |
| 45,000 ft | 55.5 / 23.4 | 59.3 / 25.8 | 62.9 / 28.1 | 66.4 / 30.4 |

(45,000 ft Mach 1.1 / 1.3 were 62.4 / 28.0 and 64.5 / 30.1 under the old
120 s cap.) Full tables (beam and Rne, lookup-table values next to the
sim-path values; table vs sim within 0.3 NM everywhere in the grid):
`spec3a_fix_outputs/rmax_summary.txt`.

### Deviations and judgment calls

- **`cd_scale` 1.10** instead of 1.25 (gravity / lift, see above).
- **Fuze:** 50 m proximity fuze checked as the closest approach inside every
  0.05 s step (prototype: 20 m at sample points).
- **Target motion inside a world step** is the straight chord between the
  target's position at the start and end of the step (includes climbs and
  turns). Using the level heading velocity made PN chase a stair-stepping
  target and bleed energy in climbing engagements.
- **Defeat labels:** `defeat_speed`, `defeat_opening`, plus `miss_overshoot`
  (opening after passing within 1 km, i.e. a guidance miss outside the fuze)
  and `timeout` (180 s cap). Existing `lost_coast_timeout`, `lost_basket`,
  `hit`, `miss` (Pk roll), `target_dead` unchanged.
- **Support after active** follows the coalition's weapons policy: any Blue
  flightmate with its own FC track, target within 90° and a working link keeps
  support; Red stays shooter-only unless `symmetric_weapons_policy` is set.
  A gap still open at impact counts only if it is already ≥ 2 s. A missile
  launched inside 15 NM counts as "supported to active" at launch, so the
  0.90 rule applies to it too.
- **Rmax table** (details under "3a fix"): axes altitude, shooter Mach,
  aspect, target Mach, signed launch off-nose angle; co-altitude, straight
  target. Lookup uses the mean of shooter and target altitude. Rne = target
  turns cold at 3 g at launch from its current aspect, capped at Rmax.
- **Not in the table:** target maneuvers after launch other than the Rne
  turn-cold (a target that turns toward the beam after launch shortens the
  shot), and altitude differences between shooter and target.
- `burnout` and `support_dropped` events added (ACMI message / bookmark).
  ACMI missiles export orientation, `Mach=` and `TAS=`, and are removed at
  the end of their flight.

### 3a fix (approved by Rusty)

1. **Launch off-nose axis.** The table gains a fifth axis: the angle between
   the shooter's heading and the line of sight to the target (horizontal).
   The missile starts along the shooter's velocity, so an off-nose shot must
   turn and bleeds energy. The axis is **signed**, because lead and lag differ
   a lot: + = nose offset toward the side the target is moving across the
   line of sight ("lead"), − = away from it ("lag"). A hot or cold target (no
   cross-LOS motion) counts as +; lead and lag are mirror images there. At
   40,000 ft, Mach 0.9 vs Mach 0.9: hot 49.4 / 47.1 / 42.8 NM at 0 / 30 / 50°
   off; beam 29.6 nose-on, 28.9 at 50° lead, **17.3 at 50° lag**.
   - **Bins:** −75, −60, −55, −50, −45, −40, −35, −30, −15, 0, 15, 30, 40,
     50, 55, 60, 75° (17). Rmax is flat near the nose (15° steps are enough)
     and bends toward the 60° launch limit, most on the lag side. Lag shots
     then **fall off a cliff**: the turn bleeds the missile below Mach 1.2 at
     any range. That happens at about 55–60° lag against a Mach 0.9 target,
     40–45° against Mach 1.3, and 55–60° either side against a cold target.
     ±75 only bounds interpolation past the 60° launch limit.
   - **Aspect bins refined** to 0, 20, 40, 60, 80, 100, 120, 150, 180° (was
     0/45/90/135/180). Rmax is curved in aspect, and 45° steps interpolated
     up to ~1.9 NM wrong between bins. That was already true before the fix,
     but the old summary only checked on-bin aspects.
   - **Interpolation at no-shot cells:** a cell with no hitting range stores
     0. If no-shot corners carry at least half of the interpolation weight,
     the lookup returns 0. Otherwise it returns the plain multilinear blend,
     which pulls the value down toward the cliff. On 1000 random grid points
     this was never optimistic by more than 1 NM. Two alternatives were
     rejected: a plain blend allowed short shots in 59 no-hit cells, and
     "any zero corner → 0" refused 13 % of feasible shots.
   - **Rne capped at Rmax.** On lag shots at high aspect, turning cold
     removes the cross-LOS motion the missile has to chase and can make the
     shot easier. The raw turn-cold range can then exceed Rmax, so no-escape =
     min(turn-cold, straight).
   - **Build:** 11 × 5 × 9 × 3 × 17 = 25,245 cells × (Rmax, Rne). One job per
     off-nose bin in a process pool (`STEALTH_TACTICS_ENV_WORKERS`, default
     = CPU count; falls back to serial). **~83 s on 8 cores** (about 10 CPU
     minutes; serial 557 s). The cache key includes `ENGINE_VERSION` "3a.2"
     and every kinematic config field, so old caches are not reused.
   - **Accuracy vs the full sim** (`spec3a_fix_outputs/off_nose_table_vs_sim.txt`):
     18 off-nose / off-bin cases within **0.35 NM** (8 at ±50° off: max
     0.23 NM). The 1000-point random check against the batch engine had a
     median error of 0.09 NM, p90 0.62 NM, and 87 % of points within 0.5 NM.
     The larger misses are pessimistic cells next to the cliff. The table is
     only approximate between the 55 and 60° bins. Example: cold target at 56°
     off, table 10.6 NM, sim no hit (the cliff sits between 55 and 56° there).
   - Shoot gating uses the new axis for both sides (`WeaponModel.rmax_m`,
     genome shoot fraction, Red's 0.8 factor). `envelope_for` returns (Rmax,
     Rne) for the current off-nose.
   - Launch-on-remote replay: the wingman now decides to turn in when Red is
     inside 0.95 × table Rmax **for the shot geometry** (Red 50° off the nose
     after the turn-in), not for its current wide 65° heading.
2. **Coast timeout 40 s** (was 20 s) and coast-Pk time term
   `(1 − 0.15 · min(t / 40 s, 1))`. The 5 km basket rule is kept.
3. **Flight cap 180 s** (was 120 s). The cap no longer binds anywhere in the
   table, checked against a 400 s cap on the high-altitude corners. The
   missile sweep has 0 `timeout` outcomes (was 29, all at 45,000 ft).

### Results (re-runs after the 3a fix, 100 seeds)

| Replay | Before fix | After fix | Shot range | Mach at fuze / Pk |
|---|---|---|---|---|
| Lead-trail, trail supports | 29 % | 29 % | 31.6 NM | 1.23 / 0.37 |
| Lead-trail, delayed | 29 % | 29 % | 31.6 NM | 1.23 / 0.37 |
| Lead-trail, no support | 0 % (all coast timeouts) | **23 %** (28 s coast, f_coast 0.87) | 31.6 NM | 1.23 / 0.32 |
| Launch on remote (0.95 × Rmax) | 0 % (29.0 NM, all defeat_speed) | **32 %** (0 defeats) | 26.3 NM, 54° off | 1.29 / 0.39 |

The lead-trail shot is nose-on at 9 km and is not cap-limited, so its range
is unchanged. The no-support gain comes only from the 40 s coast timeout.
Launch on remote at 0.85 / 0.75 / 0.65 × Rmax (23.5 / 20.6 / 17.3 NM):
34 / 39 / 44 % (`spec3a_fix_outputs/launch_on_remote_shot_frac.txt`).

- Runtime: 0.45–0.47 s per engagement (default_4v3, default genome), the
  same as the pre-fix code timed on the same machine at the same time. The
  earlier 0.41–0.45 s figure was measured under a different machine load.
- Smoke evolve (pop 12, gens 3, seed 42): best fitness 213 (unchanged), 15.7 s.


## Spec 8 loft update (2026-09-28, numbers re-verified 2026-09-29)

Loft-bias midcourse is now **on by default** (`loft_angle_deg=20`, handoff 25 NM,
settle 1 s): the aim point is raised 20° when the loft starts, decaying linearly
to 0 at the handoff. Opening defeat waits `opening_grace_s=3` and never fires
while loft bias is active. Aspect bins are 10° to 120° (were 20°). Rebuilt
Rmax / Rne table (`ENGINE_VERSION` `"8.1"`, ~6 min on 8 cores).

Rmax, shooter Mach 0.9, target Mach 0.9 level, co-altitude (sim path, NM):

| altitude | target | loft off (3a) | loft on (Spec 8) |
|---|---|---|---|
| 40,000 ft | head-on | 49.4 | **82.5** |
| 40,000 ft | beam | 29.6 | 34.8 |
| 40,000 ft | turn cold at launch | 24.4 | 24.4 |
| 40,000 ft | already cold | 21.1 | 21.1 |
| 25,000 ft | head-on | 28.8 | 33.0 |
| 15,000 ft | head-on | 20.5 | 20.5 |

Loft only acts while range-to-aim is beyond the 25 NM handoff, so shots whose
Rmax is under ~25 NM (low altitude, cold / turn-cold) are unchanged. The gain is
at high altitude. See `docs/specs/08-kinematics-realism.md`.
