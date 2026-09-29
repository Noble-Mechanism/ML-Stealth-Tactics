# Spec 8b: Jet lift limit and transonic drag rise (implemented)

Status: **implemented** (approved by Rusty 2026-09-29; built and pushed 2026-09-29).

> **Superseded numbers:** Spec 8c (thrust retune, 2026-09-29) replaced the thrust,
> induced-drag and drag-rise placeholders below. The lift limit is kept (Red CLmax
> is now 1.5). See `08c-thrust-retune.md` for the current numbers.
This changes the jet model only. The missile, loft and Rmax table are unchanged
(same envelope cache key, `ENGINE_VERSION` 8.1, so no rebuild). All numbers are
unclassified placeholders in `AircraftEnergyConfig` (`sim/sensor_config.py`),
not real F-35 or adversary data.

## Model

1. **Lift limit.** The available load factor is `min(n_max, q S CLmax / W)`, with
   `CLmax` 1.3 for Blue and 1.4 for Red. It applies everywhere, including at the
   speed floor. When `q S CLmax / W < 1`, the jet cannot hold 1 g: it rolls wings
   level at full thrust and sinks (`sin γ = -sqrt(1 - n_lift²)`, limited by the
   climb-rate cap). So a slow jet up high has to descend or unload. The soft
   floor from Spec 8 is still there. If even 1 g can't be held at the floor, the
   jet unloads to 1 g and glides.
2. **Transonic Cd0 rise** (`cd0_factor(M)`, per type):
   - flat (×1) up to Mach 0.85;
   - rises as `1 - (1 - x)²` to ×2.2 at Mach 1.05;
   - then falls by 2.5 per Mach to a floor of ×1.5 (reached at about Mach 1.33).

   Resulting factor: M0.90 1.53, M0.95 1.90, M1.00 2.13, M1.05 2.20, M1.10 2.08, M1.20 1.83.
3. **Thrust** = `T_sl (ρ/ρ_sl)^1.2 × (1 + 1.5 × max(0, M - 0.9))`. The lapse
   exponent was 0.7. The ram gain is new. `T_sl` is now 146 kN for Blue (was
   110) and 163 kN for Red (was 130). This keeps 40 kft thrust at M0.9 about the
   same for Blue and about 5% lower for Red, and gives more margin at 25-35 kft.
   So a jet can punch through the transonic hump lower down but not at 40 kft.

## Calibration (verified 2026-09-29, `python -m stealth_tactics aircraft-sweep`)

Sustained level g is the best over Mach (the Mach is in brackets). Top speed
"from M0.9" is where a level, full-thrust acceleration from M0.9 stalls (max
Mach = punches through). "Max sust." is the highest sustainable Mach (the
max_mach ceiling is 1.2 for Blue and 1.4 for Red). "Back side" is the lowest
supersonic Mach that can be held once the jet is past the hump.

| Type | Alt | Best sustained g | Sust. g @M0.9 | Lift-limit g @M0.9 | Top from M0.9 | Max sust. | Back side |
|---|---|---|---|---|---|---|---|
| Blue | 15 kft | 5.04 (M0.85) | 4.22 | 7.00 (n_max) | 1.20 | 1.20 | - |
| Blue | 25 kft | 3.35 (M0.85) | 2.82 | 6.28 | 1.20 | 1.20 | - |
| Blue | 40 kft | 1.64 (M0.85) | 1.37 | **3.13** | **0.973** | 1.20 | 1.154 |
| Red | 15 kft | 6.33 (M1.39) | 4.45 | 8.00 (n_max) | 1.40 | 1.40 | - |
| Red | 25 kft | 4.29 (M1.40) | 2.97 | 6.76 | 1.40 | 1.40 | - |
| Red | 40 kft | 2.02 (M1.37) | 1.44 | **3.37** | **0.981** | 1.40 | 1.145 |

Runs at 40 kft (integrator, dt 0.05 s):

- **Bleed gate** (lift-limited turn, heading 180° off, 20 s, from M0.9): Blue
  M0.779, Red M0.763 (gate 0.60-0.80). Blue starts at 3.13 g, and the g falls
  as the jet slows.
- **Level acceleration from M0.9** (300 s, full thrust): Blue stalls at
  M0.970, Red at M0.979. The hump blocks both.
- **Hold from M1.2** (300 s): Blue holds 1.198 (its ceiling). Red accelerates
  to 1.398 (its ceiling).
- **Dive and climb** (`dive_climb_run`):
  - The jet starts at 40 kft, M0.9, and dives at full thrust toward 30 kft.
  - It trips supersonic at M1.195 at 30.4 kft after 33 s (Red: M1.21 at 30.0
    kft, 36 s).
  - It then does an energy-managed climb, only climbing while above M1.18.
  - It arrives at 40 kft at t = 451 s at M1.18 (Red: 372 s) and holds M1.198
    (Red accelerates to 1.398).
- Lower down, a level acceleration from M0.9 over 300 s reaches M1.079 at
  30 kft and M1.033 at 35 kft for Blue (Red: 1.106 / 1.041).
- **Sustained max-rate turn** (Blue, 30 s, from M0.9): 164° at 40 kft,
  settling near M0.74 where lift-limited g equals sustainable g (Spec 8: 117°,
  bled to the floor). 428° at 15 kft.

## Tests

`tests/test_jet_spec8b.py`:
- CLmax placeholders and Cd0 shape;
- the lift limit caps g to about 3 at 40 kft / M0.9, and the integrator
  obeys it;
- a slow jet at 40 kft sinks;
- the hump blocks level acceleration at 40 kft (both types);
- supersonic at 40 kft sustains M1.2;
- dive and climb reaches M1.19 or more at 40 kft;
- the lift-limited bleed gate (both types).

Spec 8 test changes:
- The bleed gate now uses the lift-limited g instead of a fixed 3.5 g.
- The A2 turn test now expects the jet to settle near M0.74, because the lift
  limit stops it bleeding to the floor.
- The soft-floor test now checks that at the floor at 40 kft (lift < weight)
  the jet cannot turn and sinks.
- The regression fingerprint (`PRE_SPEC3`) was re-baselined because jet
  trajectories change. The missile is unchanged; the Rmax table and its tests
  are untouched.

## Caveats

- Meeting "no level acceleration through the hump at 40 kft, but M1.2
  sustainable once past it" needs thrust that rises with Mach (ram gain 1.5
  per Mach above M0.9) and a fairly steep Cd0 decline after the peak (2.5 per
  Mach). A gentler decline with flat thrust cannot do both, because q grows
  with M².
- The dive-and-climb only works as an energy-managed climb. Excess power near
  40 kft supersonic is small, so the 30 → 40 kft climb takes about 6-7 min. A
  straight 90 m/s climb from M1.15 at ~32 kft arrives at 40 kft at M0.92, below
  the back side, and settles at M0.97 (Blue) or M0.98 (Red).
- The Blue bleed gate is near the top of its band (0.779 of 0.80), because
  the lift-limited g falls as the jet slows.
