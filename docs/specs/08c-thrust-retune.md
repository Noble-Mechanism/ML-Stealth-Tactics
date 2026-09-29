# Spec 8c: Jet thrust retune (implemented)

Status: **implemented** (approved by Rusty 2026-09-29; built and pushed 2026-09-29).
It follows up Spec 8b and changes the jet model only. The missile, loft and Rmax
table are unchanged (same envelope cache key, `ENGINE_VERSION` 8.1). All numbers
are unclassified placeholders in `AircraftEnergyConfig` (`sim/sensor_config.py`).

Rusty's feedback on 8b: Blue was underpowered, and being high and fast (to
exploit missile kinematics) must stay viable. The F-35 has lots of thrust and
stubby wings: it accelerates and climbs well, but turning bleeds energy fast.

## Model changes

- **Thrust lapse, two segments.**
  - Below the 11 km tropopause, thrust = T_sl × σ^0.8.
  - Above it, thrust = T_sl × σ_tropo^0.8 × (ρ/ρ_tropo)^1.0.
  - New config fields: `thrust_density_exp_strat` and `thrust_tropo_rho`.
    `thrust_density_exp_strat=None` gives the single 8b-style exponent.
- **No thrust gain with Mach.** `thrust_ram_gain` is 0 (it was 1.5 in 8b). The
  parameter is kept for later use.
- **Cd0 rise shape** is now `1 + (peak - 1) × x^cd0_rise_power`, where x is the
  fraction of the way from `cd0_rise_mach` to `cd0_peak_mach`. With power 3 the
  rise starts slowly and is steep near the peak. After the peak, Cd0 falls by
  `cd0_decline_per_mach` down to `cd0_supersonic_floor`. `cd0_rise_exp` (8b) was
  renamed `cd0_rise_power` because the shape formula changed.
- **Induced drag.** `k_induced` is 0.16 for Blue (the stubby wing: turns bleed)
  and 0.12 for Red. Both were 0.08.
- **Red climb-rate cap** is 112.5 m/s (1.25 × Blue's 90 m/s; it was 85).

| Param | Blue | Red | 8b (Blue / Red) |
|---|---|---|---|
| T_sl (kN) | 200 | 232 | 146 / 163 |
| Lapse exp. below / above 11 km | 0.8 / 1.0 | 0.8 / 1.0 | 1.2 (single) |
| Thrust gain per Mach above M0.9 | 0 | 0 | 1.5 |
| Cd0 | 0.025 | 0.028 | same |
| k_induced | 0.16 | 0.12 | 0.08 / 0.08 |
| Cd0 rise: start / peak Mach | 0.85 / 1.10 | 0.85 / 1.10 | 0.85 / 1.05 |
| Peak factor | 3.08 | 3.20 | 2.2 |
| Rise power | 3 | 3 | (1-(1-x)^2) |
| Decline per Mach / floor | 4.0 / 1.5 | 4.0 / 1.5 | 2.5 / 1.5 |
| CLmax / n_max | 1.3 / 7 | **1.5** / 8 | 1.3 / 7, 1.4 / 8 |
| Mach ceiling | 1.2 | 1.4 | same |
| Max climb rate (m/s) | 90 | **112.5** | 90 / 85 |

Resulting Cd0 factor (Blue): M0.90 1.02, M0.95 1.13, M1.00 1.45, M1.05 2.07,
M1.10 3.08, M1.20 2.68, M1.30 2.28.

Thrust at M0.9 (Blue / Red, kN): 15 kft 138 / 160, 25 kft 105 / 122, 35 kft
78 / 91, 40 kft 63 / 73.

## Calibration results (verified 2026-09-29, `python -m stealth_tactics aircraft-sweep`)

| Target | Blue | Red | Red/Blue |
|---|---|---|---|
| 1. 35 kft, M1.0, 15° climb: (T-D)/(W sin γ) | **1.033** (speed M1.000 → 1.004 in 5 s, 15,100 ft/min) | 1.271 (→ 1.014) | 1.23 |
| 2. 30 → 40 kft from M1.0, max-rate climb (rate cap) | **33.9 s** (ends M0.975) | 27.1 s (M0.974) | 0.80× time |
| 2. 30 → 40 kft from M1.0, constant 15° | 38.9 s (ends M1.001) | 38.2 s (M1.031) | |
| 2. 30 → 40 kft from M1.0, Mach-hold (climb rate = Ps) | 38.7 s (ends M1.005) | 31.4 s (M1.007) | 0.81× time |
| 2. Plain full-power climb from 30 kft M0.9 (max rate) | 33.9 s (ends M0.898) | 27.1 s (M0.894) | |
| 3. 40 kft level acceleration M0.9 → 1.2 | **131.5 s** | 88.7 s | 0.67× time |
| 3. 40 kft level acceleration M0.9 → 1.4 | (ceiling 1.2) | 202 s | |
| 6. Bleed: 20 s lift-limited turn, 40 kft, from M0.9 | M0.749 (starts at 3.13 g) | M0.772 (starts at 3.61 g) | |

The 8b dive-and-climb still works. Blue reaches 40 kft at M1.18 at 138 s, then
holds M1.198 (Red: 98 s, then accelerates to M1.398).

Sustained max-rate turn (Blue, from M0.9, 30 s): 157° at 40 kft, settling near
M0.71 (8b: 164°), and 392° at 15 kft.

### Sustained vs max g (1 g level, full thrust)

| Type | Alt | Best sustained g (Mach) | Sustained g at M0.9 | Max g at M0.9 (lift limit / n_max) | Level top speed from M0.9 |
|---|---|---|---|---|---|
| Blue | 15 kft | 5.31 (0.95) | 5.23 | 7.00 | 1.09 |
| Blue | 25 kft | 3.86 (0.96) | 3.78 | 6.28 | 1.20 |
| Blue | 40 kft | 2.17 (0.97) | 2.11 | 3.13 | 1.20 |
| Red | 15 kft | 6.64 (0.95) | 6.54 | 8.00 | 1.09 |
| Red | 25 kft | 4.83 (0.96) | 4.72 | 7.25 | 1.40 |
| Red | 40 kft | 2.71 (0.97) | 2.63 | 3.61 | 1.40 |

(8b sustained g at M0.9 for Blue: 4.22, 2.82 and 1.37.)

### Red / Blue ratios (15, 25, 35 and 40 kft × M0.7, 0.8, 0.9)

- 1 g specific excess power: 1.19 to 1.38 (mean about 1.25). The ratio is
  lowest low down and highest at 40 kft.
- Sustained g: 1.246 to 1.250.
- Max g: 1.14 to 1.15 at altitude (CLmax 1.5 / 1.3; n_max 8 / 7 at 15 kft).

## Tests

- `tests/test_jet_spec8c.py` covers:
  - target 1 (static margin and a 5 s integrator run);
  - target 2 (three schedules under 60 s, plus the plain M0.9 climb);
  - target 3 (Blue between 120 and 300 s; Red faster and reaches M1.4);
  - max g unchanged, and turns bleed;
  - the Mach ceilings;
  - the Red ratios (Ps 1.15 to 1.45 with a mean of 1.2 to 1.3, sustained g
    1.20 to 1.30, max g 1.10 to 1.25, 15° climb margin 1.15 to 1.35);
  - Red climbs faster.
- `tests/test_jet_spec8b.py` updates:
  - Red CLmax is 1.5;
  - the Cd0 shape test is written against the config values;
  - "hump blocks level acceleration at 40 kft" is now "hump slows it": the
    run is finite and over 60 s, and the minimum excess comes after M1.0 at
    under 25% of the M0.9 excess;
  - the 40 kft M1.2 hold now checks the max sustained Mach instead of the
    back side, because at 40 kft there is no hump gap any more;
  - the bleed test is now "still loses ≥ 0.08 Mach" (`bleeds_ok`).
- `tests/test_kinematics_spec8.py`: the A-d bleed test uses `bleeds_ok` plus
  end Mach < 0.82. The 0.60-0.80 band is reported as info only.
- The regression fingerprint (`PRE_SPEC3`) was re-baselined because jet paths
  changed. The missile is untouched.

## Conflicts and caveats

- **Target 1 vs target 3.** Sustaining a 15° climb at 35 kft / M1.0 needs
  about 0.26 W of excess thrust. At 40 kft that same thrust (about 20% lower)
  would accelerate a jet through Mach quickly. The only way to make 40 kft
  level acceleration slow is a narrow, tall drag peak just past M1.0: ×3.08 at
  M1.10, compared with ×1.45 at M1.0. That is taller than typical real
  transonic rises (about ×2-2.5). It also means Blue can't go supersonic in
  level flight at 15 kft (it stalls at M1.09).
- The 40 kft time is sensitive: the minimum excess near M1.15-1.2 is only
  about 5.5 kN. A few percent more thrust or less drag moves the 131 s a lot.
- **Red "25% better" in the transonic region.** Excess-power ratios blow up
  where Blue's excess is near zero. Red's peak factor (3.20) is a compromise:
  its 40 kft M0.9 → 1.2 time is 89 s (0.67× Blue's, not 0.8×). A taller Red
  peak (3.31 gives 108 s) would stop Red going supersonic at 25 kft (it would
  stall at M1.12).
- Max-rate climbs are capped by the climb-rate limit (90 / 112.5 m/s), not by
  thrust. Red's cap was raised to keep Red better.
- The bleed is still strong at altitude (the doubled k is the "stubby wing").
  The 20 s turn ends at M0.75 (Blue) or M0.77 (Red).
