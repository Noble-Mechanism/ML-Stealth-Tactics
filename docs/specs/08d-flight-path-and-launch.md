# Spec 8d: Flight-path angle dynamics and the look-up launch fix (implemented)

Status: **implemented** (approved by Rusty 2026-09-29; built and pushed
2026-09-29). Follows Spec 8c. The missile fly-out, loft, defeat rules and the
Rmax table (and its cache key, `ENGINE_VERSION` 8.1) are unchanged.

## Why

Rusty's first overnight champion on 5dea297 flew its F-35s at about M0.53 at
41-43 kft (the model's 1 g stall line) and defeated all 21 Red shots
(19 `defeat_speed`, 2 `miss_overshoot`; 0 hits). The analysis
(`/workspace/acmi_analysis`) found:

1. **Free vertical jinking (jet-model bug).** `integrate_aircraft` set the climb
   rate straight from `cmd_alt` (clipped only by the 90 m/s climb cap), and the
   load factor only counted the horizontal turn. The F-35s alternated +90 m/s
   climb and -15 to -37 m/s sink every ~1-1.5 s (61-79% of steps at >= 85 m/s,
   65-99 reversals in 120 s): up to **36.7 g** of vertical acceleration with
   about **1.0 g** available. PN missiles at 41 kft (lift-limited to 5-13 g)
   saturated in the last seconds and bled below Mach 1.2. Re-flying the same
   shots with the F-35 altitude smoothed over 10 s: 17 / 21 fuze.
2. **Launch gate over-read look-up Rmax.** The co-altitude table was read at the
   mean of shooter and target altitude. For Red at 22.7 kft / M0.9 shooting at
   41.5 kft: table 25-35 NM, real fly-out 21-23.5 NM.

## A. Flight-path angle (sim/aircraft.py)

- New state `Aircraft.gamma_rad` (flight-path angle, + climbing; starts 0).
  Optional `Aircraft.cmd_climb_rate_mps` (None = use the altitude hold); used by
  analysis schedules (constant gamma, Mach hold).
- `n_cap = min(n_max, q S CLmax / W)` (may now be < 1).
- Desired climb rate from `cmd_alt` (**altitude hold**):
  `|vs| = min(max_climb, |err| / alt_hold_tau_s, sqrt(2 a_stop |err|))`, with
  `a_stop = alt_hold_decel_frac x g x capability`: push-over capability
  `cos(gamma) - n_pushover_min` when climbing to the target, pull-up
  capability `n_cap - cos(gamma)` when descending to it (floor 0.1 g). Levels
  off without overshoot (test: +1500 m at 25 kft, overshoot < 50 m, settles
  within 5 m). `gamma_des = asin(clip(vs / V, +-max_sin_gamma))`.
- **Vertical load factor first.** `n_v_des = cos(gamma) + V (gamma_des - gamma) / (g dt)`,
  clipped to `[max(n_pushover_min, -n_cap), n_cap]`. So
  `dgamma/dt = g (n_v - cos gamma) / V`: pull-up rate <= `g (n_cap - cos gamma) / V`,
  push-over floor **0 g** (`n_pushover_min = 0.0`: conservative, unloaded
  bunt; -1 g is a one-line config change).
- **Shared g budget.** The horizontal turn gets what is left:
  `n_h <= sqrt(n_cap^2 - n_v^2)`, turn rate `omega = g n_h / (V cos gamma)`.
  Priority rule: **vertical demand first** (in level flight n_v = 1, so level
  turns are exactly as in 8c: `sqrt(n_cap^2 - 1)`). During a hard pull-up the
  jet turns little until the pitch change is done.
- **Induced drag uses the total load factor** `n_v^2 + n_h^2`.
- **Nose drop replaces the 8b ad-hoc sink.** When lift cannot carry
  `cos(gamma)` (`n_cap < cos gamma`), `n_v = n_cap` and gamma decreases at
  `g (n_cap - cos gamma) / V`; full thrust.
- Kept: max climb / descent rate cap (`|V sin gamma| <= max_climb_rate_mps`,
  applied as a gamma clamp), `|sin gamma| <= 0.95`, soft speed floor (turn
  budget limited to what thrust sustains; glide = gamma capped at the glide
  angle), turn-direction hysteresis, hard ceiling / 100 m AGL floor (gamma set
  to 0 when the jet is clamped there), Mach ceiling.
- New `AircraftEnergyConfig` fields (both types): `n_pushover_min = 0.0`,
  `max_sin_gamma = 0.95`, `alt_hold_tau_s = 2.0`, `alt_hold_decel_frac = 0.5`.
- Per-call cost: 22.5 us vs 25.2 us before (micro-benchmark, 1e5 calls).

### No free jinking (policy flips cmd_alt 15,000 m / 100 m every 1 s, dt 0.5 s)

| Start | n_cap | 8c: max vertical accel | 8d: max vertical accel | 8d n_v range |
|---|---|---|---|---|
| 42 kft M0.53 | 0.99 | 36.7 g (ACMI) | 1.9 g | 0 .. n_cap |
| 42 kft M0.60 | 1.26 | 36.7 g | 1.6 g | 0 .. n_cap |
| 25 kft M0.80 | 4.96 | 36.7 g | 3.7 g | 0 .. n_cap |
| 15 kft M0.90 | 7.00 | 36.7 g | 5.8 g | 0 .. n_cap |

(Vertical accel = change of dh/dt per step; it also contains the along-track
term, so it can exceed n_cap - 1 slightly. The load factor itself never
exceeds n_cap.) At the stall line the jet cannot climb at all at first; each
"sink" command is a 0 g push-over, so the policy trades altitude for speed.

## B. Launch fix (sim/missile_envelope.py)

`MissileEnvelope.geometry()` looks the table up at the **shooter's altitude when
the target is above the shooter**, else at the mean (look-down and co-altitude
unchanged). Both coalitions, launch gate, Rne and the Spec 5 network inputs all
go through `geometry()`. Table and cache key unchanged (no rebuild).

| Look-up geometry (shooter M0.9) | Mean-alt table (8c) | Fly-out Rmax (straight target) | 8d table (shooter alt) |
|---|---|---|---|
| 22.7 -> 41.5 kft, target M0.53, aspect 50 | 35.1 NM | 23.5 NM | 21.0 NM |
| 23.0 -> 41.4 kft, M0.52, aspect 58 | 32.6 | 23.0 | 20.6 |
| 22.4 -> 41.6 kft, M0.54, aspect 65 | 29.9 | 21.5 | 19.5 |
| 22.6 -> 38.4 kft, M0.68, aspect 72 | 25.0 | 21.0 | 18.5 |

For the 21 Red shots in Rusty's replay the gate Rmax falls from 25.2-35.4 NM
(mean 31.9) to 18.7-21.1 NM (mean 20.1), i.e. x0.60-0.74; 8 of the 21 shots
(19.7-21.7 NM) would now be outside the gate. The fix is conservative by
~2-3 NM against the fly-out. Blue look-down shots are unaffected.

## C. Calibration re-check (`python -m stealth_tactics aircraft-sweep`)

Analysis schedules now fly `cmd_climb_rate_mps` (gamma follows at the allowed
rate); the steady 15 deg climb check starts established on 15 deg.

| Target | Blue 8d | Blue 8c | Red 8d | Red 8c |
|---|---|---|---|---|
| 1. 35 kft M1.0, 15 deg: (T-D)/(W sin g) | 1.033 | 1.033 | 1.271 | 1.271 |
| 1. 5 s steady 15 deg run | M1.000 -> 1.005 (15,100 ft/min) | -> 1.004 | -> 1.014 | -> 1.014 |
| 1. same, entered from level (pull-up ~2 s at 4.9 g) | -> 0.954 | n/a (instant) | -> 0.973 | n/a |
| 2. 30 -> 40 kft from M1.0, max rate | 34.8 s (M0.937) | 33.9 s (M0.975) | 28.1 s (M0.931) | 27.1 s |
| 2. constant 15 deg | 40.7 s (M0.990) | 38.9 s | 39.5 s (M1.028) | 38.2 s |
| 2. Mach hold | 42.3 s (M1.005) | 38.7 s | 34.5 s (M1.004) | 31.4 s |
| 2. alt hold, cmd_alt 40 kft (within 30 m) | 40.6 s (M1.003) | n/a | 36.4 s (M1.027) | n/a |
| 2. plain from M0.9, max rate | 35.1 s (M0.814) | 33.9 s | 28.4 s (M0.809) | 27.1 s |
| 3. 40 kft level M0.9 -> 1.2 | 131.5 s | 131.5 s | 88.7 s | 88.7 s |
| 3. 40 kft level M0.9 -> 1.4 | - | - | 202 s | 202 s |
| 6. Bleed, 20 s lift-limited turn at 40 kft | M0.749 (3.13 g) | 0.749 | M0.772 (3.61 g) | 0.772 |
| Dive to 30 kft + energy climb back to 40 kft | arrive 167 s at M1.18, hold M1.198 | 138 s | 120 s, then M1.398 | 98 s |

All climb targets stay under 60 s (the pull-up / level-off transients cost
1-4 s). Level-flight numbers (targets 3 and 6, the sustained / max g table
and the Red / Blue ratios) are unchanged, because level flight has n_v = 1
exactly as before:

| Type | Alt | Best sustained g (Mach) | Sustained g M0.9 | Max g M0.9 |
|---|---|---|---|---|
| Blue | 15 / 25 / 40 kft | 5.31 (0.95) / 3.86 (0.96) / 2.17 (0.97) | 5.23 / 3.78 / 2.11 | 7.00 / 6.28 / 3.13 |
| Red | 15 / 25 / 40 kft | 6.64 (0.95) / 4.83 (0.96) / 2.71 (0.97) | 6.54 / 4.72 / 2.63 | 8.00 / 7.25 / 3.61 |

Red / Blue: 1 g Ps 1.19-1.38 (mean ~1.25), sustained g 1.25, max g 1.14-1.15.
Sustained max-rate turn (Blue from M0.9, 30 s): 157 deg at 40 kft (M0.71),
392 deg at 15 kft, as in 8c. No thrust retune.

The dive-and-climb check now uses an energy-managed (Mach-hold) climb; the 8c
bang-bang schedule (climb only while M > 1.18) paid for every pitch reversal
and stalled at M1.18 at 34 kft.

## D. Re-fly of Rusty's 21 Red shots (`acmi_analysis/spec8d_refly.py`, `repro_variants.py`)

- Expected behaviour (ACMI track, F-35 altitude smoothed over 10 s; missile
  code unchanged): **17 / 21 fuze** (M1.25-1.97 at the fuze), 4 speed-defeated
  (the 19.7-21.1 NM shots). With the ACMI jinking: 0 / 21.
- Jinking policy test (F-35 flown live from its launch state, ACMI heading and
  speed, cmd_alt flipped 15,000 m / 100 m every 1 s; shooter on its ACMI track):
  - 8c code: F-35 vertical accel 36.7 g, **0 / 21 fuze** (all `defeat_speed`).
  - 8d code: F-35 vertical accel <= 1.84 g, n_v in [0, 2.84] (<= n_cap), the
    jet sinks to ~35 kft; **19 / 21 fuze** (end Mach 1.20-1.97, summed Pk
    8.6 expected kills); the 2 speed defeats are the 21.1 and 21.7 NM shots,
    both beyond the new 19.7 NM gate. All 13 shots inside the new gate fuze.

## E. Timing (evolve-net --pop 16 --gens 3 --presentations 8 --seed 123 --workers 4)

| | 5dea297 | 8d | Change |
|---|---|---|---|
| Seconds / generation (eval + benchmark) | 76.0 | 62.4 | -18% |
| Eval only | 51.0 | 42.9 | -16% |
| Benchmark | 24.9 | 19.5 | -22% |

The integrator itself is ~10% cheaper per call; most of the drop is behaviour
(different fights and end times with the same seed), so expect run-to-run
variation of that order.

## F. Tests

- New `tests/test_jet_spec8d.py` (16 tests): gamma rate limit (4 conditions),
  no free jinking (4 conditions), shared g budget, induced drag with total n,
  nose drop below 1 g, altitude hold without overshoot, climb-rate cap kept,
  climb-rate command, envelope altitude rule, look-up Rmax conservative for
  both coalitions.
- `test_kinematics_spec8.py::test_soft_speed_floor_limits_turn_rate`: the sink is
  now a nose drop, so the 10 m altitude check is after 3 s (was 1 s).
- `test_missile_defense_spec3.py::test_altitude_floor_100m_agl`: altitude hold
  eases onto the floor; 10 s, within 1 m, never below (was 5 s, exact).
- Regression fingerprint `PRE_SPEC3` re-baselined (jet paths and look-up gates
  changed; old values kept in the comment).

## Caveats

- The evolved population from 5dea297 exploited the jinking; re-evolve.
- Vertical-first priority means a policy that keeps commanding big altitude
  changes turns less; proportional sharing would be the alternative.
- The 0 g push-over floor makes dives slow to start (about 3.6 deg/s at
  157 m/s, 2.2 deg/s at 260 m/s); -1 g would double that.
- The look-up fix is conservative by 2-3 NM; a proper fix adds an altitude
  difference axis to the table (about 5x build time and cache size).
- The Mach 1.2 speed-defeat threshold was not changed; with jinking gone it
  only decides marginal shots.
