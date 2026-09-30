# Spec 8e: Rolled pull (lift vector at any bank) and lift-vector rate limits (implemented)

Status: **implemented** (approved by Rusty 2026-09-29, including his addition of
roll-rate and g-onset limits; built and pushed 2026-09-29). Follows Spec 8d.
The model is still point-mass (no 6DOF, no angle of attack). The missile, the
Rmax table and the network interface (observation/action layout and
fingerprint) are unchanged.

## Why

In 8d the jet could only unload to 0 g to get its nose down (a wings-level
bunt: at most g cos(gamma) / V, ~2 deg/s at 40 kft / M0.9). Real pilots roll
inverted and pull. Blue must be able to use the vertical (an "out" / split-S
style descending reversal) to get lower and turn around quicker. A pure bunt
(wings-level push, negative g) stays capped at 0 g. Rusty's addition: the lift
vector must not flip instantly (anti-jinking), so bank and load factor are
rate limited.

## A. Lift vector (sim/aircraft.py `integrate_aircraft`)

State (new on `Aircraft`): `bank_rad` (phi, bank of the lift vector around
the velocity vector, + = right wing down, |phi| > 90 deg = inverted) and
`load_factor` (n, >= 0). With `gamma_rad` (8d) the path follows

    V dgamma/dt        = g (n cos phi - cos gamma)
    V cos gamma dpsi/dt = g n sin phi          (cos gamma floored at 0.05)
    0 <= n <= n_cap = min(n_max, q S CLmax / W)   (positive g only)

so the nose can come down at up to (n_cap + cos gamma) g / V (inverted pull).
Induced drag uses the achieved total n (k W^2 / qS * n^2).

**Allocation** (from the existing commands: heading and altitude / climb rate;
no new network outputs):

    n_v_des = cos gamma + V (gamma_des - gamma) / (g dt)
    n_h_des = V cos gamma psidot_cmd / g         (psidot_cmd = heading error / dt)

- *Vertical first (8d rule)* when n_v_des >= n_pushover_min (0 g), or when a
  0 g push would reach gamma_des within `rolled_pull_min_push_s` (3 s): n_v =
  clip(n_v_des, 0, n_cap), n_h <= sqrt(nb^2 - n_v^2) (nb = speed-floor budget).
  Small descents are therefore flown wings level (the pure bunt, >= 0 g).
- *Rolled pull* otherwise (a big nose-down demand; kept while |phi| > 90 deg):
  the desired vector (max(n_v_des, -n_cap), clip(n_h_des, +-nb)) is scaled
  proportionally to magnitude <= n_cap. A big pure descent becomes an inverted
  pull (phi = 180 deg); big descent + big turn demands become a rolled
  descending pull (~135 deg bank when both saturate), i.e. the "out".
- phi_des = atan2(n_h, n_v).

**Rate limits** (`_limit_lift_vector`, Rusty's addition):

- `roll_rate_deg_s` = 120 deg/s (F-35-ish): the bank rolls toward the target
  the shortest way (an exact 180 deg roll goes toward the commanded turn side),
  <= 60 deg per 0.5 s world step. Time to roll inverted: **1.5 s**.
- `g_onset_g_s` = 6 g/s, up and down: <= 3 g per world step. The lift limit
  applies at once (n is clipped to n_cap when it falls).
- While the bank is still off, only the projection of the desired lift on the
  current lift direction is pulled: n_goal = max(0, n_des cos(phi_des - phi)).
  A jet rolling inverted for a nose-down pull is ~unloaded until it passes
  90 deg, so a vertical g reversal takes real time (>= 0.75 s, ~1.5 s from an
  established inverted pull back to a >1 g pull-up).
- Vertical-first mode keeps the vertical while rolling: the bank target is the
  bank at which the reachable n carries n_v (a lagging g onset shallows the
  bank instead of losing altitude), n is capped at n_v / cos(phi) while rolling
  in, and when g cannot come off fast enough the jet does not roll toward wings
  level past acos(n_v / n_min) (unload before rolling out/reversing; the excess
  lift stays horizontal instead of popping the jet up).
- Target capture: when the achieved vertical component has the same sign as,
  but more than, the one that lands on gamma_des this step, only the needed
  part is flown (the pilot eases the pull; never more lift than the lift
  vector has). Heading likewise never passes the command. Drag keeps the
  achieved n (conservative).
- Both limits are config (`None` = instantaneous). Why not instantaneous (the
  original draft): at a 0.5 s world step an instantaneous lift vector lets a
  policy flip between +n_cap and an inverted pull every step, i.e. a new form
  of free jinking.

## B. Kept from 8d, plus dive limit and recovery

Altitude hold without overshoot, max climb-rate cap, nose drop below 1 g,
max_sin_gamma 0.95, soft speed floor, hard 100 m floor: all kept.

- `max_dive_deg` = 60: descents are limited by the dive angle (V sin 60 deg)
  instead of the 8d symmetric +-max_climb rate cap (the climb cap is kept).
- Recovery: the altitude hold's stopping law now includes a reaction delay
  t_d (time to roll back upright at the roll rate + half the g-onset time to
  the arresting load factor): v t_d + v^2 / (2 a_stop) <= |err|. Probes (Blue
  and Red, dt 0.05 and 0.5): 40->20 kft, 20 kft/10 kft/5 kft/40 kft -> floor
  all bottom out at the target with **no undershoot**, reach the 100 m floor
  flying level (|gamma| < 1 deg, never the hard clamp), and dive at the full
  60 deg on the way.

`spec8d_energy(e)` (sensor_config) returns the same airframe with the 8d model
(`rolled_pull=False, max_dive_deg=None, roll_rate_deg_s=None,
g_onset_g_s=None`); it reproduces the 8d regression fingerprints byte for byte
(`PRE_SPEC3_8D_MODEL`, tested).

New `AircraftEnergyConfig` fields (Blue and Red): `rolled_pull=True`,
`rolled_pull_min_push_s=3.0`, `max_dive_deg=60.0`, `roll_rate_deg_s=120.0`,
`g_onset_g_s=6.0`.

ACMI: jets now carry attitude, `T=lon|lat|alt|roll|pitch|yaw` (roll = bank of
the lift vector, pitch = flight-path angle); frames record `roll` / `pitch`.

## C. Results (tests/test_jet_spec8e.py, /workspace/acmi_analysis)

### Out vs level max-rate turn, 40 kft M0.9, full power, time to 180 deg

| Jet  | Maneuver                                  | t 180 deg | alt lost | end Mach | turn "diameter" | max n |
|------|-------------------------------------------|----------:|---------:|---------:|----------------:|------:|
| Blue | level max-rate turn, 8d                   | 35.5 s | 0 ft      | 0.70 | 5.10 km | 3.13 |
| Blue | level max-rate turn, 8e                   | 35.5 s | 0 ft      | 0.70 | 5.09 km | 3.14 |
| Blue | descending turn, 8d (cmd -15 kft)         | 25.0 s | 6,047 ft  | 0.86 | 3.76 km | 3.70 |
| Blue | **out, 8e (cmd -15 kft)**                 | **16.0 s** | 9,317 ft | **0.97** | **1.40 km** | 5.38 |
| Blue | out, 8e, cmd only -10 kft (hold stops it) | 25.5 s | 9,989 ft  | 0.84 | 3.50 km | 5.64 |
| Blue | out, instantaneous lift vector            | 15.5 s | 9,290 ft  | 0.96 | 1.36 km | 5.36 |
| Red  | level max-rate turn, 8d / 8e              | 28.0 / 28.5 s | 0 ft | 0.75 | 4.24 km | 3.6 |
| Red  | descending turn, 8d (cmd -15 kft)         | 21.0 s | 5,639 ft  | 0.91 | 3.17 km | 4.60 |
| Red  | **out, 8e (cmd -15 kft)**                 | **14.5 s** | 8,516 ft | **0.99** | **1.27 km** | 6.17 |

("diameter" = lateral extent of the path during the reversal.) The out
reverses in ~45% of the level-turn time, ends ~M0.27 faster (it gains speed
instead of bleeding it) and in about a quarter of the lateral space, for ~9 kft.
The roll-rate / g-onset limits cost it only ~0.5 s. The altitude block matters:
with only 10 kft commanded the altitude hold pulls out before the turn is done.

### Nose-down rate (level, commanded 8 km lower, dt 0.05, time to gamma -15 deg)

| Condition     | 8d (0 g push)          | 8e (roll inverted + pull)          | 8e, instantaneous |
|---------------|------------------------|------------------------------------|-------------------|
| 40 kft M0.9   | 7.20 s, peak 2.1 deg/s | 2.80 s, peak 8.8 deg/s (inverted at 1.5 s)  | 1.75 s |
| 25 kft M0.8   | 6.75 s, peak 2.3 deg/s | 2.35 s, peak 13.7 deg/s             | 1.15 s |
| 15 kft M0.9   | 7.85 s, peak 1.9 deg/s | 2.35 s, peak 15.0 deg/s             | 1.00 s |

(8d could not reach -30 deg at all at these speeds: the descent was capped at
the climb rate.)

### Anti-jinking re-fly of Rusty's 21 Red shots (spec8e_refly.py)

The target F-35 is flown live from its ACMI launch state with commands flipped
every 1 s (world dt 0.5 s). Pre-8d: 0/21 fuzed (36.7 g jinks).

| Target policy                                 | 8d model | 8e |
|-----------------------------------------------|---------:|---:|
| alternate climb / dive (cmd_alt 15 km / 100 m) | 19/21 | **14/21** (exp. kills 6.3) |
| alternate left / right (+-86 deg heading)      | 19/21 | 18/21 |
| both                                           | 21/21 | 15/21 |
| steady max descent (cmd_alt 100 m, no jinking) | 21/21 | **10/21** |

In 8e every step's bank change is <= 60 deg and |dn| <= 3 g (tested);
vertical acceleration up to ~5 g at the lower, faster end of the dive
(n up to 6.3 once the jet is at ~28 kft / M1). The 8e losses are long shots
(R0 ~18-21.7 NM vs a 19-21 NM gate) where the target ends 13 kft lower at
~M1.0: a steady dive with no jinking defeats more of them (10/21 fuze) than
the alternating commands do (14/21), so the alternation itself buys nothing -
it is the new, legitimate diving capability that costs these long shots.

### Level turns and climbs vs 8d

Level turns: 30 s max-rate turn 157.1 -> 156.7 deg at 40 kft M0.9 and
391.9 -> 388.3 deg at 15 kft M0.9 (the 6 g/s onset needs ~1 s to reach 7 g);
180 deg at 40 kft 35.5 s both; altitude held exactly. Climbs 30->40 kft +0.4-0.5 s (g onset in the
pull-up). See D.

## D. Calibration deltas (8d -> 8e)

| Item (full thrust)                        | Blue 8d | Blue 8e | Red 8d | Red 8e |
|-------------------------------------------|--------:|--------:|-------:|-------:|
| 30->40 kft M1.0 max-rate climb            | 34.8 s | 35.2 s | 28.1 s | 28.6 s |
| 30->40 kft M1.0 gamma 15 deg              | 40.7 s | 41.2 s | 39.5 s | 39.9 s |
| 30->40 kft M1.0 Mach hold                 | 42.3 s | 42.8 s | 34.5 s | 34.9 s |
| 30->40 kft alt hold                       | 40.6 s | 41.1 s | 36.4 s | 36.9 s |
| 30->40 kft M0.9 plain                     | 35.1 s | 35.4 s | 28.4 s | 28.8 s |
| 35 kft M1.0 15 deg margin / 5 s run       | 1.033 / M1.005 | same | 1.271 / M1.014 | same |
| 40 kft level accel M0.9->1.2 (->1.4 Red)  | 131.5 s | same | 88.7 s (202 s) | same |
| bleed (lift-limited turn, 40 kft, 20 s)   | M0.749 | M0.753 | M0.772 | M0.776 |
| sustained / max g tables, 1 g Ps          | -      | unchanged | - | unchanged |
| dive-and-climb, default (alt-hold dive)   | 166.9 s | 292.1 s | 120.0 s | 220.2 s |
| dive-and-climb, 30 m/s constant-rate dive | -       | **146.6 s** | - | **127.1 s** |

The dive-and-climb change is procedural: the default dive is now the altitude
hold's 60 deg inverted-pull dive, which reaches 30 kft quickly and then has to
accelerate level through the drag hump there; a shallow constant-rate dive
(`dive_climb_run(..., dive_rate_mps=30)`) goes supersonic while still
descending and is faster than 8d's for Blue.

## E. Timing

Same command as 8d (`evolve-net --pop 16 --gens 3 --presentations 8 --seed 123
--workers 4`), timing.jsonl, same box, run back to back:

| code    | s/gen (mean of 3) | eval | bench | gen 0 (same random pop) | wall |
|---------|------------------:|-----:|------:|------------------------:|-----:|
| 64cb49f (8d) | 61.3 | 41.9 | 19.4 | 59.5 | 7 m 23 s |
| 8e           | 63.0 | 42.2 | 20.8 | 56.4 | 7 m 51 s |

Gen 0 (identical random population) is ~5% faster; gens 1-2 evolve different
policies (different fights), so their cost is not directly comparable.
Integrator microbenchmark (100k calls,
alternating heading/altitude commands, dt 0.5): 64cb49f 22.3 us/call; 8e 29.2
us/call before and **10.2 us/call** after replacing scalar `np.clip` /
`np.sign` / `np.sqrt` with plain Python in `integrate_aircraft` (results
bit-identical, fingerprints unchanged).

## F. Tests and re-baselines

- `tests/test_jet_spec8e.py`: pure bunt >= 0 g (wings level); inverted pull
  noses down > 3.5x faster than the 0 g push, rolled inverted at 1.5 s; total g
  <= n_cap under random commands (Blue/Red, 4 conditions, path-inferred and
  state); rolled descending pull allocation; bank/n rate limits under
  alternating vertical / lateral / both commands; finite vertical g reversal;
  time to roll inverted; out vs level turn (Blue/Red); level turns and climbs
  vs 8d; dive limit, recovery and floor; ACMI roll/pitch; `spec8d_energy`
  reproduces the 8d fingerprint.
- `tests/test_jet_spec8d.py`: the tests of 8d-specific allocation details
  (0 g push floor, vertical-first split, instantaneous load factor, induced
  drag in one step) now fly `spec8d_energy()`.
- `tests/test_jet_spec8b.py::test_lift_limit_caps_g_at_40kft_mach09`: the
  one-step check of the lift-limited turn uses an instantaneous lift vector.
- `PRE_SPEC3` re-baselined (jet paths change): 8d values
  {0: 4dcf69e1363d65db, 1: 24b05e6373e53513, 2: 785ed021d347d0d8} ->
  {0: bf7aca62278997e9, 1: c3326114a8f940f7, 2: 9432844bb8d191de}.

## Concerns

- Diving is now a strong missile defense for long shots (10/21 fuze with a
  steady dive), for both sides. Expect evolved Blue policies to dive more and
  Red launches from above to lose Pk; revisit if the fights turn into dives to
  the floor.
- Point-mass: no angle of attack or pitch-rate limit beyond the g-onset rate;
  the altitude hold, not a pilot model, flies the recovery.
- Target capture (easing the pull to land on gamma_des / the commanded heading)
  is instantaneous; it only ever reduces lift and drag keeps the achieved n.
