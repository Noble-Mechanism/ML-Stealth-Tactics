# Spec 2b — Missile coast after support lapse, and Blue 50 NM fire-control gate

Status: **implemented**. Parameters live in `stealth_tactics/sim/sensor_config.py`:
`SensorConfig.missile` (`MissileCoastConfig`) and `TrackConfig.fc_range_ref_m`.
Applies to both sides unless noted.

## 1. Missile coast (replaces "dies on the first unsupported step")

Before autonomy (15 NM from the **true** target), each step:

- **Supported** (existing rules; Blue: any flightmate with its own FC track, a
  working link and the target within 90° of its nose; Red: shooter only):
  - the missile guides on the target;
  - `last_sup_pos` and `last_sup_vel` are updated;
  - `t_since_update` resets to 0.
- **Unsupported:**
  - the missile **coasts** toward `last_sup_pos + last_sup_vel × t_since_update`
    (last supported position extrapolated with the last known horizontal velocity);
  - `t_unsupported` (cumulative) and `t_since_update` both grow by dt.
  - Support may resume at any time (`support regained by X`).
- **Coast timeout:** if `t_since_update > coast_timeout_s` (**20 s**, continuous),
  the missile is lost (`lost: coast timeout`, outcome `lost_coast_timeout`).
- **Autonomous point:** the seeker goes active when the missile is within 15 NM of
  the true target.
  - **Seeker basket:** if the extrapolated aim point is more than
    `seeker_basket_m` (**5000 m**) from the true target, the missile is lost
    (`lost: outside seeker basket`, outcome `lost_basket`).
  - Otherwise `Pk *= exp(-t_since_update / pk_decay_tau_s)`, with τ = **12 s**
    (0.66 at 5 s, 0.43 at 10 s). A supported missile gets a factor of 1.0.
    **Spec 3a replaced this factor** with
    `exp(-(e / 2 km)^2) × (1 - 0.15 · min(t_gap / 20 s, 1))` (`pk_decay_tau_s` removed).
  - Event: `autonomous (… Pk factor f …)`.
- At launch, the last supported position and velocity are the target's.

Events and ACMI bookmarks:
- `support_lost` — "support lost (coasting)"
- `support_regained`
- `support_handoff`
- `autonomous` (carries `pk_factor`)
- `lost_coast_timeout`
- `lost_basket`

Missile attributes: `coasting`, `t_unsupported`, `t_since_update`, `pk_factor`,
`aim_err_at_auto_m`.

## 2. Blue fire-control range gate = 50 NM

- `TrackConfig.fc_range_ref_m = {"F-35": 92600}`. For a listed observer type, the
  FC gate is `92.6 km × rcs_eff^0.25`, computed as `ref × R50 / radar_ref_range`
  since R50 = ref_range × rcs_eff^0.25.
- Unlisted types (Red) keep `fc_range_frac × R50` (0.7 × R50).
- Detection ranges are unchanged. The continuity rule is unchanged: at least 3 s
  held, and the streak breaks after more than 2 s without a hit.
- The gate is applied to the **estimated** radar range, as before.

### Lock stability (head-on vs Red, 100 seeds; `datalink-replays --scenario fc-lock`)

- First FC: median **48.3 NM** (p10 45.8, p90 50.0).
- In the 50–40 NM band, FC is held **61 %** of the time. There are 1.48 drops per
  run on average (median 2, max 3), and 88 of 100 runs have at least one drop.
- In the 40–30 NM band, FC is held 97 % of the time, with 20 of 100 runs having a
  drop.
- Output: `spec2_outputs/fc_lock_stability.txt`.

## 3. Effects on the replays

The lead-trail replay (C) was re-run with the coast rule and the 50 NM gate; see
`datalink_events.txt`. The trail now holds its own FC before the lead's
max-range shot, so support is handed over with no or a very short gap.
