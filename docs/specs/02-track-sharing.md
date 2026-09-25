# Spec 2 — Track sharing within the flight (datalink, fusion, triangulation, remote launch)

Status: **implemented**. All numbers are unclassified placeholders and live in
`stealth_tactics/sim/sensor_config.py` (`SensorConfig.datalink`, class `DatalinkConfig`).
Everything is deterministic per seed; the datalink has its own RNG stream
(`rng_salt = 0xD17A`), separate from sensors (`0x5E45`) and weapons.

Out of scope: AWACS/GCI, link jamming, network inputs to the genome.

## 1. Datalink (`stealth_tactics/sim/datalink.py`)

| Parameter | Blue | Red |
|---|---|---|
| update period | 1 s | 2 s |
| latency | 1 s | 2 s |
| message loss | 5 % | 5 % |
| max range between flightmates | 150 NM | 150 NM |

- Each alive jet broadcasts a snapshot of its **own** radar and IRST track
  components every update period: estimated position, `sigma_at_meas` (1-σ when
  measured), measurement time, observer position (for IRST lines of sight), and a
  fire-control flag. **RWR contacts are not shared.** Nothing is relayed; only the
  sender's own sensor data is sent.
- **Loss:** one Bernoulli draw per sender per update (p = 0.05). A lost message is
  lost for every recipient (judgment call: "each message (per sender per update)").
- **Range:** a recipient must be an alive flightmate within 150 NM of the sender
  **at send time**.
- **Delivery** happens at `t_send + latency`. The recipient keeps the newest
  snapshot per sender and ignores older ones.
- Per-step order in `World._update_sensors`: `datalink.deliver` → `sensors.update`
  (own sensors + fusion) → `datalink.send`.
- **Dead jets** stop sending. Their last received data ages out on the normal 10 s
  coast (`TrackConfig.coast_s`), and so do remote FC reports.
- Datalinks are **undetectable**: no RWR or IRST interaction.

## 2. Shared-track degradation

Received components keep the sender's error and grow with age, like coasting:

    sigma_rx(t) = sigma_at_meas + 50 m/s × (t − t_meas)

`t − t_meas` includes the link latency, so a Blue report is always at least 1 s
(+50 m) stale when it arrives and a Red report at least 2 s (+100 m).
The received position is dead-reckoned with the sender's velocity estimate.

Own components: `sigma_own = meas_sigma / sqrt(n_eff) + 50 m/s × age`.

## 3. Fusion (`stealth_tactics/sim/fusion.py`)

For every enemy in a jet's `TrackStore`, the candidates are its own radar, its own
IRST, each flightmate's received radar and IRST, and (Blue only) the best
triangulation fix. The **smallest-σ candidate wins**. The result is stored in
`TrackStore.fused[tid]` as a `FusedTrack`:
`target_id, est_pos, sigma_m, sensor (radar|irst|tri), source_jet, age_s, own, candidates [(label, σ)], tri`.
Provenance (source jet id, age, own vs shared) is always visible.

`TrackStore.quality()` and `is_fire_control()` still describe **own** sensors
only (spec 1 semantics). Remote FC is queried separately via
`TrackStore.remote_fc_source(tid, t, max_age)`.

Supporting filter change (judgment call): own radar/IRST position updates now use
`est = pred + (meas − pred)/n_eff` (velocity smoothed the same way) instead of
jumping to each raw measurement. With this change the actual estimate error
matches the σ that fusion ranks candidates by. Spec 1 detection logic,
probabilities and FC rules are unchanged.

## 4. IRST triangulation (Blue only)

Two F-35 IRST lines of sight to the same target (own and/or received, each ≤ 5 s
old) are intersected in the horizontal plane. Altitude comes from the
elevation of the nearer line.

    gamma   = angle between the two lines of sight (horizontal)
    no fix  if gamma < 10°  (or > 170°, or the rays don't meet in front of both jets)
    sigma_h = sigma_b × sqrt(r1² + r2²) / sin(gamma)      sigma_b = 0.5° (IRST bearing noise)
    sigma_v = sigma_b × min(r1, r2)
    sigma   = hypot(sigma_h, sigma_v) + 50 m/s × age_of_older_bearing

This is standard first-order error propagation for two bearing lines crossing at
angle γ. Each range rᵢ contributes rᵢ·σ_b of cross-bearing error, magnified by
1/sin γ. The fix enters fusion as a candidate (`sensor="tri"`) and wins only if
its σ is smallest. It is **never fire-control quality**. Red has no
triangulation (`triangulation = {Blue: True, Red: False}`).

## 5. Launch on remote and midcourse support handoff

`World.fire_control_source(shooter, target)` returns:
- the shooter's own id if it holds its own FC track (always allowed), otherwise
- under the **shared policy** only: the id of a flightmate whose FC report was
  received over the link with snapshot age ≤ `latency + 2 × update period`
  (Blue 3 s, tolerating one lost message; Red 6 s if symmetric), otherwise `None`
  (no launch).

The usual shooter launch checks still apply: ammo, missile range, and target
within 60° off-boresight. The shooter does **not** need its own track.

Midcourse support (`WeaponModel.pick_supporter`), checked every step until the
missile is within 15 NM of the target (then autonomous, as in spec 1):
- **Shared policy:** any flightmate (shooter included) that is alive, holds its
  **own** FC track on the target, has the target within 90° of its nose, and is
  within the weapon link range (150 NM) of the missile. Preference order: current
  supporter, then shooter, then others by id. A change emits `support handed to X`.
  The weapon link is modelled as lossless (judgment call: the 5 % loss applies to
  track messages).
- **Red policy (default):** only the shooter can support, and the shooter can
  launch only on its own FC track (spec 1 behaviour).
- If nobody qualifies before acquisition, the missile **coasts** (Spec 2b, see
  `02b-coast-and-lock.md`). It is no longer lost on the first unsupported step.

Policy switches:
- `remote_launch_and_handoff = {Blue: True, Red: False}`.
- **Single switch:** `symmetric_weapons_policy = False`. Set it to True to give
  Red the Blue policy.

Weapon events (ACMI bookmarks and `world.weapons.events`): `launch`,
`launch_remote`, `support_handoff`, `autonomous`, `support_lost`, `hit`, `miss`,
`timeout`.

## 6. Tactics interpreter (minimal adaptation)

Blue chooses targets first from reds it can fire on (own or remote FC), then from
its fused picture, then from all reds (old behaviour). It fires via
`fire_control_source`. Red fires via `fire_control_source` too, which means own
FC only unless the symmetric switch is on. The genome is unchanged.

## 7. ACMI

- `frame["markers"]` are exported as extra objects (default
  `Type=Navaid+Static+Waypoint`) and removed with `-id` when they disappear.
  Replay B uses one yellow marker named **`TRI est`**, with σ and LOS angle in
  its label.
- New event types are exported: `link_track` (Message), `launch`,
  `launch_remote`, `support_handoff`, `autonomous`, `support_lost`, `hit`,
  `miss`, `timeout` (Bookmark), `tri_fix`, `note`.

## 8. Replays (`python -m stealth_tactics datalink-replays -o spec2_outputs`)

Code: `stealth_tactics/analysis/datalink_replays.py`. Options: `--scenario
{all,launch-on-remote,triangulation,lead-trail}`, `--seed-a`, `--seed-b`, `--seed-c`,
`--stats-seeds` (default 100).

**A — launch on remote.**
- Setup: B1 (lead) flies pure pursuit on the Red (Su-27, weapons off). Red starts
  60 NM north and flies at B1. B2 starts 20 NM east.
- Deviation: B2 also starts 15 NM forward. With a pure 20 NM abeam start it only
  reached missile range at the merge, after the lead's FC geometry was gone.
- B2 holds Red 65° off its nose (outside its ±60° radar field of regard). When it
  has B1's remote FC track and range ≤ 0.95 × missile range, it turns in to put
  Red 50° off its nose. This is needed because the existing 60° launch
  off-boresight limit still applies. It fires one missile, then turns cold.
- B1 keeps supporting until 15 NM.
- Missiles are enabled only for this replay.

**B — passive triangulation.**
- Setup: two F-35s start 4 NM apart, 36 NM behind a Red flying away (hot tail).
  They chase at 300 m/s, splay ±10° to 24 NM lateral separation, then fly
  parallel. The run lasts 600 s.
- Deviation: tail chase instead of head-on. Head-on, the IRST only detects at
  about 14–18 NM, where the LOS angle is already over 55°. The tail chase keeps
  range between 25 and 36 NM while the LOS angle sweeps from 6° to 56°, so the
  1/sin γ tightening is visible.
- Radars stay on (spec 1), so fusion normally picks radar. The replay logs the
  fused source every step.
- Outputs: `triangulation_error.txt` (time, true range, triangulated error,
  LOS angle, fused source) and a `TRI est` ACMI marker.

**C — lead-trail (launch then turn out, trail supports).**
- Setup: two F-35s in a 15 NM lead-trail on the same track, both hot on a Red
  (Su-27, weapons off) that flies at the lead. Speeds are the same as replay A.
- The lead fires one missile on its own FC track at the earliest valid shot
  (existing rules; range-limited at 24.3 NM), then turns 180° to cold at max speed.
- The trail stays hot and may take over support via handoff.
- **Result:** the lead loses FC about 8 s into its turn, when Red leaves its ±60°
  radar field of regard. The trail only gets its own FC at about 34 NM (t = 148 s,
  every seed), after the missile would already have gone autonomous (about 143 s).
  So support always lapses. Under the existing rule the missile dies on the first
  unsupported step (`support_lost`); nothing was changed in the sim.
- **Variants** (same 100 seeds):
  - `no-support`: shooter-only support, via a replay-level override of
    `pick_supporter`.
  - `delayed`: the lead waits until the trail holds FC, then shoots (about 18.8 NM)
    and turns out.
- Outputs: `replayC_lead_trail.txt.acmi`, `replayC_lead_trail_delayed.txt.acmi`,
  and the comparison table in `datalink_events.txt`.

> **Spec 2b update:** the Replay C results above (every missile lost) were
> produced under the old "dies on first unsupported step" rule and the 0.7 × R50
> Blue FC gate. See `docs/specs/02b-coast-and-lock.md` and `datalink_events.txt`
> for the current numbers.
