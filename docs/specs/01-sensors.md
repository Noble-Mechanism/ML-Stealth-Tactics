# Spec 1 — Sensor model (radar, signature, IRST, RWR, tracks)

**Status:** implemented. **All numbers are unclassified placeholders** (not real
F-35 / threat data). Every tunable lives in one file:
`stealth_tactics/sim/sensor_config.py` (`SensorConfig` dataclass tree,
`DEFAULT_SENSOR_CONFIG`). `aircraft.py` pulls `radar_range_m` and the base
`rcs_factor` from it. Pass a modified config with
`SimConfig(sensor_config=dataclasses.replace(DEFAULT_SENSOR_CONFIG, ...))`.

Code: `sim/sensors.py` (physics + `SensorModel`), `sim/tracks.py` (per-jet
`TrackStore`), `sim/weapons.py` (fire-control gating of support),
`sim/world.py` (update cadence, launch gating), `acmi/exporter.py` (TacView
markup), `analysis/` (table + replays).

Geometry conventions: ENU metres, heading 0 = North clockwise, point-mass
(no pitch/roll), so the sensor body frame is heading-only. *Aspect* = angle
between the target's heading and the direction target→observer (0 nose, 90
beam, 180 tail), azimuth only.

## 1. Probabilistic radar detection (per scan)

```
R50 = ref_range(observer) * rcs_eff(target, aspect) ** 0.25
Pd(R) = 1 / (1 + (R / R50) ** k)              # per scan
k = 40 / (snr_slope_db * ln 10)                # = 8.69 for 2.0 dB
```

Derivation: radar SNR ∝ R⁻⁴ ⇒ SNR_dB − SNR50_dB = 40·log10(R50/R). A logistic
Pd in SNR-dB with scale `snr_slope_db` gives exactly the log-logistic curve
above (Pd(0)=1, Pd(R50)=0.5, Pd(0.78·R50)=0.9, Pd(1.29·R50)=0.1).
2 dB ≈ Swerling-like steepness. No detections beyond the instrumented range
`1.5 × ref_range`. Draws come from a seeded RNG stream
(`default_rng([sim_seed, rng_salt])`, independent from the weapon-Pk stream) —
fully deterministic per seed.

| Param | Value |
|---|---|
| `ref_range_m` F-35 / Red | 90 km / 70 km |
| `snr_slope_db` (→ k) | 2.0 dB (k ≈ 8.69) |
| `instrumented_range_factor` | 1.5 × ref range |
| measurement σ (track estimate) | range 50 m, angle 0.5°, velocity 10 m/s |
| `scan_period_s` | 1.0 s |

**Per-scan vs cumulative:** because a closing target gets many scans, the
*first* detection typically happens beyond R50 (≈1.35–1.4 × R50 at 500 m/s
closure; see `sensors-table`). R50 is the per-scan 50 % point as the spec
defines; to pull first-detection ranges in, lower `ref_range_m` or reduce
`snr_slope_db` (steeper curve).

## 2. Field of regard

Radar (and IRST) only look within **±60° azimuth AND ±60° elevation** of the
nose (`for_az_deg`, `for_el_deg`). Elevation = atan2(Δalt, horizontal range).
Outside ⇒ Pd = 0.

## 3. Signature (RCS vs aspect)

F-35: azimuth-aspect table, **piecewise-cosine interpolation**
(`y = y0 + (y1−y0)·(1−cos(πu))/2`, u = fractional position in the segment ⇒
C¹-smooth, monotone per segment, zero slope at each knot):

| aspect | 0° | 20° | 45° | 70° | 90° | 135° | 180° |
|---|---|---|---|---|---|---|---|
| rcs_factor | 0.05 | 0.05 | 0.12 | 0.50 | 0.90 | 0.55 | 0.30 |

**Changed in spec 3 (approved change A).** The original table was 0/30/60/90/135/180°
→ 0.05/0.10/0.45/0.90/0.55/0.30. The new knots give a flat best RCS within 20° of the
nose, a modest penalty for a 35–45° crank (Red detection range ×1.18 at 35°, ×1.24 at
45° vs nose-on) and a steep rise to the beam; 90° and aft are unchanged. Red-radar
median first detection of a closing F-35: 20° 28.9 → 24.6 NM, 35° 31.7 → 29.4 NM,
45° 38.8 → 32.1 NM, 50° 42.2 → 33.3 NM, 60° 45.0 → 42.3 NM (nose 25.0 NM unchanged).
The spec 1 slope test bound moved from 0.03 to 0.035 per degree (the 70–90° segment
peaks at 0.031/deg). Full table: `spec3_outputs/rcs_change/rcs_aspect_table.txt`.

Red: isotropic 1.0. Resulting Red-radar R50 vs F-35: nose 33.1 km, beam
68.2 km, tail 51.8 km. F-35 radar vs Red: 90 km all aspects.

## 4. IRST (both sides, passive)

```
R50_ir = [nose + (tail − nose) * (1 − cos(aspect)) / 2]
         * clip((v_target / 250 m/s) ** 1.0, 0.5, 1.5)     # speed ≈ engine power
         * target_ir_scale (1.0 both types)
Pd(R) = 1 / (1 + (R / R50_ir) ** k_ir),  k_ir = 20 / (1.0 dB * ln 10) ≈ 8.69
```

IR point-source signal ∝ R⁻² (hence 20 not 40); atmospheric extinction makes
the real fall-off steeper, folded into a 1.0 dB slope (same steepness as radar).

| Observer | nose-on R50 | tail-on R50 (ref speed 250 m/s) |
|---|---|---|
| Red IRST (Red's counter to stealth) | 30 km | 60 km |
| F-35 IRST | 25 km | 50 km |

* Field of regard ±60° az/el (same as radar — simple, keeps both sensors
  forward-looking; no DAS-style spherical coverage in this spec).
* Measurement: bearing & elevation σ = 0.5°; range multiplicative error
  σ = 30 % (clamped ≥ 0.2 × true). Position σ reported ≈ 0.3·R.
* **Passive:** the target gets no RWR or any other cue.
* **IRST-only tracks can never be fire-control** ⇒ no launch, no midcourse support.
* No draw beyond 2 × R50_ir (Pd < 0.3 %).

## 5. RWR

Receiver gets a **bearing-only** contact (σ 5°, no range) on an enemy emitter
when (a) the receiver is inside the emitter's radar field of regard (±60°) and
(b) range ≤ `emitter_detect_factor × emitter ref_range`. Deterministic; radars
are always on.

| Emitter | factor | Intercept range |
|---|---|---|
| F-35 (LPI) → Red RWR | 0.60 | 54 km (29.2 NM) |
| Red radar → F-35 RWR | 1.50 | 105 km (56.7 NM) |

Updated **every sim step (0.5 s)** (`update_period_s = 0` ⇒ every step).

## 6. Tracks, quality, fire control

Each jet owns a `TrackStore` (target id → `Track`). A `Track` holds per-sensor
*components* (`radar`, `irst`, `rwr`), each with first/last hit time, hit count,
bearing, range estimate (None for RWR), measurement σ, position/velocity
estimate. Reserved `origin` field (`"own"`) for Spec 2 datalink.

* **Coasting:** a component is dropped when `t − last_hit > 10 s`; the track is
  dropped when no component remains (`radar_lost` / `irst_lost` / `rwr_lost` /
  `track_lost` events). Destroyed targets are dropped immediately.
* **Position error:** `σ = meas_σ / sqrt(n_eff) + 50 m/s × time_since_last_hit`,
  `n_eff = min(1 + held_s / 1 s, 10)`; radar `meas_σ = hypot(50 m, R·0.5°)`,
  IRST `meas_σ = hypot(0.3·R, R·0.5°)` ⇒ grows with range, shrinks with hold
  time, grows while coasting.
* **Quality order:** RWR (1) < IRST (2) < RADAR (3) < FIRE_CONTROL (4).
* **Fire-control quality** (all required):
  1. radar-derived component present;
  2. held **continuously ≥ 3 s** — the streak tolerates one missed 1 Hz scan
     and breaks if no radar hit for > 2 s (`fc_max_gap_s`);
  3. last measured range ≤ **0.7 × R50** (kept the old 0.7 idea; at 0.7 R50 the
     per-scan Pd is ≈ 0.96, so the track is solid enough to guide a missile).
* **Weapons:** a launch requires a fire-control track on that target (World
  gate + tactics). Midcourse support requires shooter alive, **fire-control
  track still held**, target within **90°** of the nose, until the missile is
  within **15 NM (27 780 m)** of the target, after which it is autonomous
  (unchanged rules, new track source).

FC ranges: F-35 vs Red 63 km; Red vs F-35 nose 23.2 km / beam 47.7 km / tail 36.3 km.

## 7. Update cadence

Radar and IRST: every 1.0 s (every 2nd step at dt = 0.5 s, scheduled by sim
time so other dt values work). RWR and track maintenance (coast/drop,
fire-control state): every step.

## Tactics integration

* Blue target choice: nearest fire-control track, else nearest radar/IRST
  track, else nearest (truth) — steering still uses truth positions.
* Blue fires only with a fire-control track. Red CAP wakes up on any-quality
  track (incl. RWR); Red fires only with a fire-control track.
* `world.tracks` is `{observer_id: TrackStore}`; `id in store` = any track.

## TacView markup

* `LockedTargetMode=1,LockedTarget=<hex id>` on the observer while it holds a
  fire-control track (closest one); cleared with `LockedTargetMode=0,LockedTarget=`.
* `0,Event=Bookmark|<obs>|<tgt>|…` for the first radar / IRST / RWR / FC event
  per pair; later detections and all losses as `0,Event=Message|<obs>|<tgt>|…`
  (text includes range in NM and km).

## Tools

```bash
python -m stealth_tactics sensors-table --seeds 300 -o spec1_outputs/sensor_table.txt
python -m stealth_tactics sensor-replays --seed 1 -o spec1_outputs
# Replay 3: beam at 45 NM, hot again on Red FC lock or after 60 s (appends to log)
python -m stealth_tactics sensor-replays --scenario beam-then-hot --seed 1 -o spec1_outputs
```

## Out of scope (later specs)

Datalink / track sharing (Spec 2 — `TrackStore`/`origin` ready), chaff /
flares / notching, jamming, radar emission control, terrain, weather.
