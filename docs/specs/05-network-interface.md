# Spec 5 — Network interface (what the Blue network sees, what it outputs, how outputs become sim commands)

Status: **approved and implemented (2026-09-26).** Rusty approved every
decision A–R exactly as recommended, one at a time, on 2026-09-26, and answered
all ten questions (see "Questions for Rusty"). The build is described in
"Implementation" at the end. Every key choice is a lettered decision (A–R)
with options, a recommendation and a short why. The numbers are prototype
placeholders, as in specs 1–4.

## What it does

Specs 6–9 evolve **one small neural network shared by all four Blue F-35s**.
Each jet runs the same weights and is told its slot number (1–4). Spec 6 picks
the architecture and the neuroevolution method, spec 7 the fitness, spec 8 the
overnight runner (Ubuntu, Quadro P1000, about 8 cores), and spec 9 the demo
outputs.

Spec 5 defines only the **interface**:
1. the **observation**: a fixed-length vector per jet built only from that jet's
   own sensor, datalink and warning-receiver picture, plus own-side information;
2. the **action**: a fixed-length output vector per jet;
3. the **mapping** from the action to sim commands (`cmd_heading_rad`,
   `cmd_alt_m`, `cmd_speed_mps`, `cmd_fire` / `fire_target`,
   `radar_emitting`, per-jet firing doctrine), plus the decision rate;
4. the **adapter test**, which proves the interface is enough: the current
   scripted Blue is expressed through the same action mapping, and a hand
   policy written only on the observation vector reproduces its results.

It does **not** choose the network (spec 6), the fitness (spec 7) or the runner
(spec 8). The only sim change it needs is the radar on/off switch (**L**).

## Already decided (carried in, not re-opened)

- One network for all four Blue jets, told its slot (1–4). GA / neuroevolution.
- Blue has no truth. It fights on its own and shared picture (spec 1
  probabilistic radar and IRST, spec 2 datalink, fusion and triangulation, spec
  2b coast and lock, spec 3 RWR modes).
- Missile: spec 3a fly-out model, with an Rmax / Rne (no-escape) table indexed by
  altitude, shooter Mach, target aspect, target Mach and the **signed** launch
  angle off the nose. Launch is allowed inside table Rmax.
- Firing doctrine is per contact. SAS allows one own missile in flight per
  contact; SSA fires a pair 3 s apart, then holds that contact until both
  resolve. Rusty: the **network decides when and how many shots**; doctrine
  only limits missiles per contact.
- Spec 4 presentations: 6 Red, 360 s cap, one Red defensive reaction and then
  press (approved change 2026-09-26). Blue starts fixed: a 4-ship heading north
  at about 9,000 m.
- Radar on/off was floated in spec 3 (`Aircraft.radar_emitting`, "a network
  output in spec 5").

## Facts found in the code (they shaped the decisions)

1. **The scripted Blue steers and shoots on truth.** `TacticsController`
   (`tactics/interpreter.py`) uses Red truth in six places:
   - it commits on the **true** range from Blue lead to the nearest Red
     (`world.alive(RED)` + `distance_3d`);
   - it picks each jet's target as the nearest Red by **true** distance from
     the pool "Red with FC available → Red in the fused picture → **any live
     Red**" (the last fallback is pure truth);
   - it aims at the target's **true** bearing, plus a geometry offset
     (bracket ±35°, hook ±50°, and so on);
   - it commands the target's **true** altitude;
   - it fires when the **true** range ≤ `shoot_range_frac × Rmax(true geometry)`;
   - it aborts on Blue losses.

   The FC requirement is its only perception-based gate. So **the adapter test
   cannot demand bit-identity from observations alone** (see **Q**).
2. **Radar on/off does nothing yet.** `radar_emitting` is read in one place only:
   `rwr.emitter_mode` (spec 3 RWR modes). Three gaps remain:
   - `SensorModel.radar_measure` (own radar detections, hence FC) ignores it;
   - the spec 1 RWR track component (`sensors.rwr_measure` → `rwr_detects`)
     ignores it;
   - midcourse support (`has_midcourse_support`) needs own FC, so it would drop
     only as a side effect once radar measurements stop.

   A radar output therefore needs a small sim change (**L**).
3. **Fire control is automatic and undirected.** Every Red that a Blue radar
   tracks inside the FC gate (0.7 × R50, held 3 s, gaps ≤ 2 s) becomes an FC
   track. Red's RWR shows **lock** for every such track (within the F-35 LPI
   intercept range of 29 NM: 0.6 × 90 km). Blue cannot "lock only its target".
   That is why conservative Red (trigger: lock) drags before Blue's shot range.
   With radar on/off the network gets some control; a directed lock is
   deferred.
4. **Tracks are keyed by the true aircraft id.** Correlation is perfect across
   sensors and across the flight: a Red track seen by B1 and B3 is the same key.
   This makes shared-target features ("a wingman is targeting / shooting this
   contact") well defined without an association model. The network never sees
   the id, only slot positions.
5. **The fused track** (`store.fused[tid]`: `est_pos`, `sigma_m`, `sensor` ∈
   {radar, irst, tri}, `source_jet`, `age_s`, `own`) has **no velocity**.
   Velocity exists only on radar components (`est_vel`, own or received). IRST
   and triangulated tracks have none. Aspect, closure and Rmax / Rne therefore
   exist only when a radar velocity estimate exists. FC (own or remote) implies
   radar, so every shootable track has them.
6. **RWR-only contacts** (Red radar always emits, and Blue intercepts it at
   57 NM: 1.5 × 70 km) live in `store.tracks` with quality `RWR` but are **not**
   in `store.fused` (no position). The bearing has 5° noise and there is no
   range. RWR mode cues (`world.rwr[id]`) are per emitter, keyed by the Red id,
   plus one `missile_active` cue per live autonomous missile aimed at this jet
   (bearing, no range).
7. **Diagnostic truth fields sit next to perceived ones.**
   - `SensorComponent.true_range_m` and `sensors.primary_fc_target` (which uses
     it; ACMI only) are truth;
   - `Missile.autonomous` flips on the **true** range to the target, but the
     shooter knowing "my missile went active" is legitimate (weapon datalink);
   - Pk values and `pk_*` fields are truth-side.

   The observation builder must read only whitelisted fields (**A**).
8. **Blue can fire on a flightmate's FC** (remote launch, fresh within 3 s), and
   any flightmate with own FC and the target within 90° of its nose can take over
   midcourse support (Blue policy). Red cannot. A radar-silent Blue shooter is
   therefore already possible in the weapons code once radar-off exists.
9. **Command plumbing.** Controllers are called every 0.5 s sim step and set:
   - `cmd_heading_rad` (absolute; the aircraft turns at up to 11°/s);
   - `cmd_alt_m` (clipped to the 100 m AGL floor and 15,000 m; climb ≤ 90 m/s);
   - `cmd_speed_mps` (clipped to 90–340 m/s; accel 30 m/s²);
   - `cmd_fire` + `fire_target` (reset after each step).

   `World` checks every launch: doctrine (`_doctrine_target`), FC source,
   `WeaponModel.can_shoot` (ammo, 500 m minimum range, **true** range ≤ table
   Rmax for the **true** geometry, off-boresight ≤ 60°), not departed. The
   per-jet doctrine override `Aircraft.firing_doctrine` already exists.
   `run_presentation(..., blue_factory=...)` already accepts any Blue
   controller.
10. **Envelope lookups are cheap.** `MissileEnvelope.for_states` returns (Rmax,
    Rne) in about 7 µs, and a 230-64-64-13 numpy MLP forward is about 5 µs for a
    batch of 4. A crude observation builder prototype (6 fused contacts with
    Rmax / Rne, own and wingman blocks, 1 Hz, all four jets) added **0.17 s per
    engagement** (6 presentations: 1.11 → 1.29 s). No GPU is needed for the
    interface.
11. There is **no fuel model** and **no Red missile track** (Blue sees a Red
    missile only through its own `missile_active` warning).

## Model

### 1. Observation (per alive Blue jet, one flat float32 vector, fixed layout)

`obs = [ own (21) | contacts K=6 × 28 (168) | wingmen 3 × 14 (42) ]` = **231
inputs**. The layout is described by an `ObsSpec` with named slices, so spec 6
can also reshape the contact block to (6, 28) for a set encoder.

Frame: **ego-centric**. Bearings are relative to the jet's own nose, and
altitude deltas are target minus own. The one absolute direction is the
**mission axis** (the Blue ingress heading, north), used for own heading and
as the heading reference when there is no target (**B**, **H**).

Own block (21):

| # | Input | Encoding / normalization |
|---|---|---|
| 1–4 | slot one-hot (1–4) | 0/1 |
| 5 | altitude | alt / 15,000 m |
| 6 | speed | v / 400 m/s |
| 7–8 | heading vs mission axis | sin, cos |
| 9 | missiles left | ammo / 4 |
| 10 | time elapsed | t / 360 s |
| 11 | radar emitting | 0/1 (always 1 if **L** is out) |
| 12 | own missiles in flight | n / 4 |
| 13 | own missiles in flight that need my support (pre-active, I am the supporter) | n / 4 |
| 14 | own missiles coasting (pre-active, no supporter) | n / 4 |
| 15 | soonest own missile to seeker-active (estimate from perceived range, 1,000 m/s) | t / 60 s, clip [0, 1]; 0 if none |
| 16 | SSA pair open (second shot pending) | 0/1 |
| 17 | time since my last launch | min(t, 60) / 60; 1 if never |
| 18 | missile-active warnings on me | min(n, 2) / 2 |
| 19–20 | bearing of the newest missile-active warning | sin, cos (0, 0 if none) |
| 21 | Red kills confirmed so far (kill events) | n / n_red |

Contact block: K = 6 slots × 28 (see **D**, **E**). Wingman block: 3 × 14 (see
**F**).

### 2. Action (per jet, 13 raw outputs, all squashed by the policy's output layer)

| # | Output | Mapping |
|---|---|---|
| 1 | heading offset `h ∈ [−1, 1]` | `cmd_heading = ref + h × 180°`; `ref` = perceived bearing to the selected target, else the mission axis (**H**) |
| 2 | altitude `z ∈ [−1, 1]` | `cmd_alt = 100 + (z + 1)/2 × (15,000 − 100)` m (sim clips to the 100 m AGL floor) |
| 3 | speed `s ∈ [−1, 1]` | `cmd_speed = 90 + (s + 1)/2 × (340 − 90)` m/s |
| 4–10 | target logits: slots 0–5 + "none" | argmax over present slots and "none" (**I**) |
| 11 | fire `f` | `f > 0` → `cmd_fire` at the selected target (**J**) |
| 12 | radar `r` | `r > 0` → emitting; minimum dwell 5 s per state (**L**) |
| 13 | pair `p` | `p > 0` → SSA for the next new contact engaged, else SAS (**K**) |

The mapping is deterministic, with no sampling (reproducibility, spec 4 M).

### 3. Controller loop

`NetworkBlueController(policy, cfg)` implements the existing Blue controller
call. Every `decision_period_s` (1 s, **N**), aligned to sensor scans at t = 0,
1, 2 …, it builds the observations of all alive Blue jets into one (n, 231)
batch and calls `policy(batch) → (n, 13)`. It then decodes the actions and holds
the commands until the next decision; the fire request is re-asserted on both
0.5 s steps. Dead jets are skipped. `policy` is any callable, so the network
comes from spec 6 and the scripted adapters from **Q**.

## Decisions

**A — Observation source and the truth firewall.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **Blue's own picture only**: own state, own track store (`fused` +
     `tracks` quality / FC / remote FC), own RWR cues, own-side jets and
     missiles, sim time;
  2. same plus a "GCI" truth summary (e.g. true Red count);
  3. truth (not acceptable).
- **Recommendation: (1).** The builder reads a `BlueView` object, not the
  `World`. The view exposes only whitelisted fields:
  - fused `est_pos`, `sigma_m`, `sensor`, `own`, `age_s`;
  - radar component `est_vel` (own or received);
  - FC flag and remote FC availability (`world.fire_control_source` is
    perception-based);
  - RWR cues;
  - own and flightmate `Aircraft` state (Blue only);
  - Blue missiles (`target_id`, `autonomous`, `supporter_id`, `coasting`,
    launch time);
  - kill events;
  - `time_s`.

  Red `Aircraft` objects, `true_range_m`, Pk fields and `primary_fc_target` are
  not reachable from the view.
- **Uncertain and coasting tracks** are represented, not hidden:
  - position is the fused estimate extrapolated to now (as fusion already does);
  - `sigma_m` and `age_s` are inputs;
  - a track disappears when fusion drops it (10 s coast);
  - bearing-only RWR contacts appear with `range_valid = 0` (**D**).

  No memory of dropped tracks in spec 5 (a recurrent policy is spec 6's
  choice).
- *Why:* the whole point is tactics that work on what the jet knows. A
  view object makes leakage a test failure instead of a code-review question.

**B — Frame.** *(approved by Rusty 2026-09-26)*
- Options:
  1. absolute world frame (x, y, heading);
  2. **ego-centric, with a single mission-axis reference**;
  3. fully ego-centric (no absolute direction at all).
- **Recommendation: (2).** All contact and wingman geometry is relative to own
  position and nose. Own heading is given relative to the mission axis
  (north = the direction Blue ingresses, so "home" is south). Absolute x / y
  are never input.
- *Why:* there is no terrain or wind, so absolute position carries no
  information (spec 4 A). Keeping the mission axis gives the network a notion
  of "press / egress" direction that spec 7 may reward (escape rule, Rusty Q4
  in spec 4). Option (3) would make egress unlearnable.

**C — Own-ship inputs** *(approved by Rusty 2026-09-26)* (table above, 21).
- In: slot one-hot, altitude, speed, heading vs mission axis, missiles left,
  time elapsed (the 360 s cap is part of the problem), radar state, own
  missiles in flight / needing my support / coasting / soonest active, SSA pair
  open, time since last launch, missile-active warnings (count + newest
  bearing), confirmed kills.
- Out:
  - **fuel**: no fuel model; time elapsed stands in;
  - absolute position;
  - vertical speed (the sim has no climb dynamics beyond a rate limit);
  - Pk or any missile truth (hit or miss is known only from kill events).
- *Why:* each input answers a question the network must ask. "Can I still
  shoot?" is ammo; "do I have to keep my nose on it?" is needs-support; "am I
  being shot?" is missile active; "is time running out?" is time elapsed.
  Support status is what makes crank vs drag a real trade-off.

**D — Contact set: K, selection, order, masking.** *(approved by Rusty 2026-09-26)*
- Options for K: 4, **6**, 8. Options for order:
  1. by range;
  2. by threat (e.g. RWR mode, then range);
  3. by range/Rmax;
  4. sticky slots (a contact keeps its slot while tracked);
  5. **current target pinned to slot 0, the rest by range**.
- **Recommendation: K = 6, order (5).**
  - The contact pool is every entry in the jet's store: ranged contacts from
    `fused`, plus bearing-only RWR contacts from `tracks` that are not in
    `fused`.
  - **Slot 0** is the jet's current target (from its previous decision) if that
    target is still in the picture; else the nearest ranged contact.
  - **Slots 1–5** hold the other ranged contacts by perceived range ascending,
    then bearing-only contacts by RWR mode (support > lock > search), then
    oldest first seen.
  - Ties break on the lower `sigma_m`, then on the stable internal key (never
    exposed).
  - Empty slots are **all zeros with present = 0**.
  - Contacts beyond K are dropped. With 6 Red and one pinned target, every Red
    fits.
- *Why:*
  - Range order is the most learnable fixed order for a small MLP: slot 1 is
    always "the nearest other contact".
  - Pinning the current target gives the target output a natural "keep" answer
    (slot 0). Without pinning, the target slot jumps whenever ranges reorder,
    and targets flap.
  - K = 6 covers the full 6-ship. At `n_red = 8` the nearest 6 still hold
    what matters, and K is a config value.
  - Permutation: an MLP over ordered slots is not permutation-invariant, so
    the fixed ordering rule stands in for invariance. Spec 6 may instead use a
    shared per-slot encoder (deep sets / attention) on the (6, 28) block; the
    layout supports both.

**E — Per-contact features (28 per slot).** *(approved by Rusty 2026-09-26)*

| # | Feature | Normalization | Notes |
|---|---|---|---|
| 1 | present | 0/1 | mask |
| 2 | range valid | 0/1 | 0 = bearing-only (RWR) |
| 3 | range | r / 80 NM, clip [0, 1.5] | perceived (fused est) |
| 4–5 | bearing off own nose | sin, cos | RWR bearing for bearing-only |
| 6 | altitude delta | Δalt / 5,000 m, clip ±1.5 | 0 if bearing-only |
| 7 | kinematics valid | 0/1 | radar velocity estimate available |
| 8–9 | target aspect (target heading vs LOS to me) | sin, cos | 0 if no kinematics |
| 10 | closure rate | Vc / 1,000 m/s, clip ±1 | 0 if no kinematics |
| 11 | target speed | v / 400 m/s | 0 if no kinematics |
| 12 | position sigma | log10(1 + σ m) / 5 | 1 if bearing-only |
| 13 | age of measurement | age / 10 s, clip [0, 1] | coasting shows here |
| 14 | own-sensor track | 0/1 | vs received / triangulated |
| 15–17 | sensor one-hot: radar, IRST, triangulation | 0/1 | all 0 = RWR only |
| 18 | own FC | 0/1 | |
| 19 | remote FC available (flightmate, fresh) | 0/1 | Blue remote launch |
| 20 | range / Rmax (perceived geometry) | clip [0, 2] / 2 | 1 if no kinematics or Rmax = 0 |
| 21 | range / Rne (perceived geometry) | clip [0, 2] / 2 | same |
| 22 | Rmax | Rmax / 80 NM, clip [0, 1.5] | 0 if no kinematics |
| 23 | doctrine allows a launch at it now | 0/1 | `world.may_fire_at` (Blue-side state) |
| 24 | shot ready (perceived) | 0/1 | FC (own or remote) ∧ r ≤ Rmax ∧ \|off-nose\| ≤ 60° ∧ r ≥ 500 m ∧ doctrine ∧ ammo, all on perceived values |
| 25 | own missiles in flight at it | n / 2 | |
| 26 | flightmates' missiles in flight at it | n / 4 | |
| 27 | flightmates currently targeting it | n / 3 | from their last decision |
| 28 | RWR mode from this emitter on me | 0, 1/3 search, 2/3 lock, 1 support | per-emitter cue |

- Rmax / Rne use the **table** on the **perceived** geometry:
  - own state;
  - target = fused position with heading and speed from the radar `est_vel`;
  - mean altitude, own Mach, perceived aspect, perceived target Mach and the
    **signed** off-nose angle (`off_nose_signed` on perceived states).

  The ratios r/Rmax and r/Rne are the shoot-timing inputs. Raw Rmax is kept so
  the network can see "the envelope is growing" when it climbs or speeds up.
- Out:
  - Red track id;
  - Red type (all Red are the same type);
  - Red defense state (Blue cannot see it; it must infer it from aspect and
    closure);
  - Red ammo;
  - the true Pk.
- *Why:* everything the scripted Blue uses (range, bearing, altitude, Rmax, FC,
  doctrine) plus what a network needs for the trade-offs this project is
  about: aspect and closure show a drag or beam; sigma and age show a coasting
  track; RWR mode shows who is locking or supporting against me;
  flightmate-missile and targeting counts allow sorting and target
  deconfliction. `shot_ready` is a perceived hint; the sim still checks the
  truth (**O**).

**F — Wingman inputs (3 × 14).** *(approved by Rusty 2026-09-26)*
- Order options:
  1. by slot ascending (self removed);
  2. cyclic (slot + 1, + 2, + 3);
  3. **element-relative: my element mate, the other element's lead, the other
     element's wing** (elements 1–2 and 3–4). Slot 1 sees [2, 3, 4], slot 2
     sees [1, 3, 4], slot 3 sees [4, 1, 2], slot 4 sees [3, 1, 2].
- **Recommendation: (3).**
- Per wingman:

  | # | Feature | Normalization |
  |---|---|---|
  | 1 | alive | 0/1 (all zeros if dead) |
  | 2 | range | r / 40 NM, clip [0, 1.5] |
  | 3–4 | bearing off my nose | sin, cos |
  | 5 | altitude delta | Δalt / 5,000 m, clip ±1.5 |
  | 6–7 | heading relative to mine | sin, cos |
  | 8 | speed | v / 400 m/s |
  | 9 | missiles left | ammo / 4 |
  | 10 | missiles in flight | n / 4 |
  | 11 | threatened (missile-active warning on it) | 0/1 |
  | 12 | radar emitting | 0/1 |
  | 13 | has a target | 0/1 |
  | 14 | its target is my current target | 0/1 |

- The **wingman's target** also enters per contact (**E** #27), so "who is
  targeting what" is visible from both sides.
- Friendly state is taken **exact and current** (own-side info over the link;
  the link is never out of range at these distances). A 1 s delayed copy is the
  alternative (question 4).
- *Why:* element-relative order gives each slot the same meaning ("my wingman")
  whatever the slot number, so one shared network can learn element tactics
  once. The slot one-hot still lets it specialise per slot.

**G — Normalization (all inputs).** *(approved by Rusty 2026-09-26)*
- Rules:
  - every input is in [−1, 1] or [0, 1.5];
  - angles as sin / cos (no wrap discontinuity);
  - distances scaled by a fixed engagement scale (contacts 80 NM, wingmen
    40 NM) and clipped;
  - velocities / 400 m/s (closure / 1,000 m/s);
  - altitude / 15,000 m, deltas / 5,000 m;
  - counts by their maximum;
  - sigma on a log scale;
  - ratios to Rmax / Rne clipped at 2;
  - missing values = 0, with an explicit validity flag (present, range valid,
    kinematics valid).
- Scales are **fixed constants** in `ObsSpec` (not running statistics), so
  a champion replays identically forever and runs are comparable.
- *Why:* neuroevolution is sensitive to input scale (a single large input
  saturates tanh units). Fixed scales keep reproducibility (spec 4 M).
  Validity flags stop "0 NM" from meaning both "unknown" and "on top of me".

**H — Heading, altitude and speed outputs.** *(approved by Rusty 2026-09-26)*
- Heading options:
  1. absolute heading;
  2. offset from own nose (±180°);
  3. turn rate;
  4. **offset from the perceived bearing to the selected target; mission axis
     when there is no target**.
- **Recommendation: (4)** for heading, **absolute** altitude
  (100–15,000 m) and **absolute** speed (90–340 m/s).
- *Why:*
  - In the target frame every BVR maneuver is a constant: 0° pursuit, ±50°
    crank, ±90° beam, 180° drag. So is the scripted Blue's bracket / hook
    offset (fact 1). A GA finds a constant far more easily than a function of
    bearing.
  - With no target, output 0 means "fly the ingress axis", not "circle".
  - Turn rate (3) and nose-relative (2) make holding a course depend on
    feeding back the current heading.
  - Absolute altitude and speed matter physically (missile energy, the
    envelope), and a constant output holds them.
- Risk: when a threat comes from another contact than the target, the target
  frame is not the threat frame. The network sees the missile-active bearing
  and can switch target to the threat or change the offset. A separate
  "threat frame" flag is deferred.

**I — Target selection.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **K + 1 logits (slots + none), argmax over present slots**;
  2. a per-slot score from a shared head (needs a set architecture, spec 6);
  3. a fixed rule (nearest FC track).
- **Recommendation: (1).** Empty slots are masked. "None" is allowed (the jet
  flies the mission axis and does not fire). Slot 0 = keep the current target
  (**D**). No extra hysteresis; the pinning is the hysteresis.
- *Why:* target choice (sorting, deconfliction) is a tactic the GA should
  learn. Option (1) works with any architecture, and (2) remains possible
  inside spec 6 without changing the interface.

**J — Shoot decision.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **binary "fire now at the selected target" (output > 0)**;
  2. a shoot-range fraction `f` with an automatic launch when perceived
     r ≤ f × Rmax (the scripted rule);
  3. a probability with sampling.
- **Recommendation: (1).**
  - It is deterministic.
  - It is re-asserted on both sim steps of the decision.
  - It is only a request: the world gates decide (**O**).
  - The number of shots = how often the network asks. Per contact, doctrine
    caps it.
- *Why:* Rusty wants the network to decide when and how many. Option (2)
  hard-codes "shoot at a range fraction" and cannot express "hold fire until
  the conservative Red turns, then shoot". Option (3) breaks exact
  reproducibility unless seeded per decision, and adds noise to fitness.

**K — Doctrine: fixed setting or output.** *(approved by Rusty 2026-09-26)*
- Options:
  1. fixed SAS for all Blue (setting; today's default);
  2. **a per-jet "pair" output**: at the next launch at a contact with nothing
     open, SSA (pair) if `p > 0`, else SAS. Applied by setting
     `Aircraft.firing_doctrine` at the decision, and only when no SSA pair is
     open (so a pair is never cut in half);
  3. a doctrine per contact chosen by the network (a K-wide output).
- **Recommendation: (2).** Doctrine stays a *limit* enforced by the sim. The
  network chooses shots-per-contact 1 or 2 for the next engagement, which is
  "how many shots".
- *Why:* SAS alone cannot express a two-missile salvo (it blocks the second
  shot until the first resolves), yet salvo vs single is a real missile-economy
  trade-off with 16 missiles against 6 Red. One output bit covers it cheaply.
  Option (3) costs 6 outputs for little gain.

**L — Radar on/off output (and the sim change it needs).** *(approved by Rusty 2026-09-26)*
- Options:
  1. out (radar always on);
  2. **in, with a 5 s minimum dwell per state**;
  3. in, per contact (directed lock / STT).
- **Recommendation: (2)**, with this sim change (default `True`, so every
  existing result and the legacy regression stay unchanged):
  - `radar_measure` returns nothing when the observer is not emitting. Own
    radar tracks coast, and own FC drops after the 2 s gap rule.
  - The spec 1 RWR track component (`rwr_detects`) requires the emitter to be
    emitting (RWR modes already do).
  - Midcourse support drops naturally (it needs own FC). A flightmate with FC
    can take over (Blue policy).
  - Switching back on re-acquires through the normal Pd and the 3 s FC hold.
  - Red stays always-on.
- *Why:*
  - Emission control is the core stealth tactic: a silent shooter launches
    on a wingman's FC while Red's RWR shows nothing from it.
  - It is also the only way Blue can avoid giving conservative Red a lock
    warning (fact 3).
  - The dwell stops "blinking" radar from being a free exploit.
  - Option (3) changes spec 1 / 3 behaviour (who gets a lock warning) and is
    deferred.

**M — Defend output.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **none**: defense is expressed through heading / altitude / speed (crank
     = target-frame ±50°, drag = 180° plus descent);
  2. a discrete "defend" macro that hands the jet to a scripted reaction (like
     `BlueTestDefense`).
- **Recommendation: (1).** `BlueTestDefense` stays tests-only.
- *Why:* a macro hard-codes when and how Blue defends, which is exactly what
  the GA should find (crank while supporting vs drag). The target-relative
  heading frame already makes crank and drag single constants. Inputs 13
  (needs my support), 18–20 (missile active) and per-contact RWR mode give the
  trigger information. If spec 6 shows the GA cannot find defense, a macro is
  an easy add (question 9).

**N — Decision rate and smoothing.** *(approved by Rusty 2026-09-26)*
- Options:
  1. every 0.5 s sim step;
  2. **every 1 s (2 steps), aligned to the 1 s sensor scan and datalink
     period**;
  3. every 2 s.
- **Recommendation: (2).**
  - Commands are held between decisions.
  - There is no extra smoothing: the aircraft model already limits turn
    (11°/s), climb (90 m/s) and acceleration (30 m/s²).
  - Radar has the 5 s dwell (**L**). Target has the slot 0 pinning (**I**).
- *Why:*
  - Sensors and the link update once per second, so a 0.5 s decision mostly
    sees the same picture twice.
  - 1 s is fast relative to threats: a missile-active warning comes about
    15 NM out, roughly 25–30 s before impact.
  - Measured cost at 1 Hz is +0.17 s per engagement (fact 10). 0.5 s would
    double that, and 2 s halves it but lags the SSA 3 s pair and crank timing.
  - Rate is a config value (the adapter parity test runs at 0.5 s, **Q**).

**O — Legal-action enforcement (the network cannot cheat).** *(approved by Rusty 2026-09-26)*
- **Recommendation: all launch legality stays in the sim, unchanged:**
  - doctrine (`_doctrine_target`), FC source (own or remote fresh FC), ammo,
    500 m minimum range;
  - **true** range ≤ table Rmax for the **true** geometry, off-boresight ≤ 60°;
  - not departed.

  The network only requests: `fire_target` is the internal key of the
  selected track, which exists only if the jet has that contact in its
  picture. It cannot name a Red it does not track, so a request on a
  bearing-only contact fails the FC gate. Commands are clipped by the
  aircraft model. A denied request costs nothing in the sim; spec 7 may count
  denied requests (deferred).
- *Why:* keeping the gates in one place (the world) means the network, the
  scripted Blue and Red all obey identical rules, and `shot_ready` being a
  *perceived* hint cannot turn into a truth leak.

**P — Sizes and runtime handed to spec 6.** *(approved by Rusty 2026-09-26)*
- **231 inputs** (own 21 + contacts 6 × 28 + wingmen 3 × 14). **13 outputs**:
  3 continuous (heading, altitude, speed), 7 target logits, 3 binary (fire,
  radar, pair).
- Reference size: an MLP 231-64-64-13 has **19,853 parameters** (tanh hidden;
  output squash per head). That is small for neuroevolution (ES / GA with
  population 50 is fine at this size), but spec 6 decides.
- Runtime:
  - Prototype: +0.17 s per engagement at 1 Hz (serial 0.99 → about 1.16–1.25 s
    with the full 28-feature builder; estimate +0.2–0.25 s).
  - Per generation (50 × 24 + 64 = 1,264 engagements on 8 workers):
    **about 166 → 195–205 s**, about 175–185 generations per 10 h night.
  - At 0.5 s decisions: about 230–250 s per generation.
- The GPU (Quadro P1000) does not help at a batch of 4 per call; CPU numpy is
  faster. GPU batching across engagements would need a lock-step runner
  (spec 8, not needed).

**Q — Adapter test (sufficiency proof).** *(approved by Rusty 2026-09-26)* Three layers, because the scripted
Blue uses truth (fact 1):
1. **Action-mapping parity (exact).** `ScriptedViaInterface` runs
   `TacticsController` unchanged. After each call it:
   - **encodes** the jet's commands into the 13-output action (heading →
     offset from the reference, altitude, speed; `fire_target` → its slot in
     the jet's observation, which must exist because firing needs FC; fire and
     pair bits);
   - **decodes** that action through the real decoder, and applies it.

   At decision period 0.5 s the resulting fight must be **identical** to the
   direct controller (same `SimResult`, shots and events, and a byte-identical
   ACMI) on 20 presentations. It proves the action space can express the
   scripted Blue exactly and the decoder introduces no drift. It also checks
   that every requested fire target appears in the jet's contact slots (0
   misses allowed).
2. **Decision-rate effect (near-identical).** The same wrapped policy at
   1 s hold on the 100 spec 4 stats seeds. Tolerance:
   - mean Blue kills and losses within ±10 % (or ±0.2 absolute);
   - end-reason counts within ±5 per class.
3. **Perceived-only hand policy (sufficiency).** `HandBlue` is written **only**
   on the observation vector:
   - target = nearest shot-ready slot, else slot 0;
   - heading offset = bracket ±35° by element;
   - altitude = target altitude from the altitude delta;
   - speed = 273 m/s;
   - fire when `shot_ready` ∧ r/Rmax ≤ 0.75.

   It is run on the same 100 seeds. It must land within ±15 % of the default
   scripted Blue on Blue kills, losses and shots, and it must use no truth (it
   receives only the obs array). A second variant with two silent shooters
   (slots 2 and 3 radar off, firing on remote FC) must produce remote launches
   and zero Red RWR cues from the silent jets. That proves **L** end to end.

   Results are reported as a before/after table like spec 4.
- *Why:* (1) proves the action side exactly; (2) isolates the decision-rate
  choice; (3) proves the observation side carries enough to fight as well as
  the truth-fed script, which is the real question.

**R — Scope and where the code lives.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - `stealth_tactics/policy/` (new):
    - `observation.py`: `BlueView`, `ObsSpec` (named slices and scales),
      `build_observations`;
    - `action.py`: `ActionSpec`, `decode_actions`, `encode_commands` for the
      adapter;
    - `controller.py`: `NetworkBlueController(policy, decision_period_s=1.0)`;
    - `adapters.py`: `ScriptedViaInterface`, `HandBlue`.
  - Sim: the radar-emitting gates (**L**) in `sim/sensors.py`, plus a
    `World.blue_view(ac)` helper.
  - `run_presentation(blue_factory=...)` is used as is.
  - The GA, genome and fitness are untouched: spec 6 swaps the
    `TacticsGenome` for network weights.
- *Why:* a clean seam. Spec 6 needs only `ObsSpec.size`, `ActionSpec.size`
  and a `policy` callable.

## Deferred / out of scope

- Network architecture, recurrence / memory, set encoders, weight encoding and
  the GA operators (spec 6).
- Fitness, penalties for denied fire requests or radar use, and the escape rule
  (spec 7).
- Fuel model; weapon bay / launch delay; directed lock (STT vs TWS) and lock
  warnings only on the targeted Red (**L** option 3).
- Red missile tracks (Blue sees only its own missile-active warning),
  missile-launch warning, IFF / ID ambiguity and track correlation errors
  (fact 4).
- Memory of dropped tracks; time since a contact was lost.
- A threat-frame heading flag; a scripted defend macro (**M** option 2).
- Formation-keeping outputs (station relative to lead). The opening formation
  emerges from heading, altitude and speed.
- Stochastic actions; running-statistics normalization.
- Notching, chaff, lookdown clutter (spec 3 deferred).

## What changes (implemented 2026-09-26; see Implementation)

- New `stealth_tactics/policy/` package (**R**).
- `sim/sensors.py`: radar measurements and the spec 1 RWR component honour
  `radar_emitting` (default `True`: fingerprints and the legacy regression are
  unchanged).
- `sim/world.py`: `blue_view(ac)`; optional per-jet doctrine set by the
  controller (`Aircraft.firing_doctrine` already exists).
- CLI: `interface-adapter-test` (runs **Q** layers 1–3 and writes the tables
  to `/workspace/spec5_outputs/`), `obs-dump --presentation SEED --t T --jet B2`
  (prints the named observation for one jet, for debugging and for Rusty).
- Docs: this spec (with an Implementation section after the build), README and
  `design.md` interface summary.

## Test plan

1. **Observation unit tests:**
   - Shape and layout: `ObsSpec.size == 231` and the named slices cover the
     vector exactly once.
   - Masking: empty contact slots and dead wingmen are all zeros with the flag
     at 0; bearing-only contacts have range fields at 0 and `range_valid = 0`;
     tracks without a radar velocity have kinematics fields at 0.
   - Ordering: the current target is pinned to slot 0; the rest are by
     perceived range; bearing-only contacts come after ranged ones; tie-breaks
     are deterministic; a contact dropping out shifts the others in order.
   - Normalization ranges: over 100 presentations × all decision steps × all
     jets, every input is finite and within its declared range ([0, 1],
     [−1, 1] or [0, 1.5]); sin² + cos² = 1 where present.
   - **No truth leakage:**
     - (a) build obs, move every Red's true position / heading without
       re-running sensors → obs unchanged;
     - (b) set every `true_range_m` to NaN → obs unchanged and finite;
     - (c) `BlueView` raises on any access to Red `Aircraft` objects, Pk fields
       or `primary_fc_target` (test with a strict proxy);
     - (d) two worlds with identical Blue pictures but different Red truth give
       identical obs.
   - Rmax / Rne inputs equal `envelope.for_states` on the perceived states
     (built by hand in the test), with the signed off-nose convention.
   - `shot_ready` agrees with the world's gates when perception is exact (a test
     with sensor noise forced to 0).
   - Element-relative wingman order for each slot; the slot one-hot is correct.
2. **Action unit tests:**
   - Decode ranges: heading offset ±180° about the target bearing / mission
     axis; altitude 100–15,000 m; speed 90–340 m/s.
   - Argmax ignores masked slots; "none" means no fire and the mission-axis
     reference.
   - Fire is re-asserted on both steps and passes through the world gates
     unchanged (a denied request → no launch).
   - Pair bit: the doctrine switches only when no SSA pair is open.
   - Radar dwell: a flip is refused within 5 s.
   - `encode → decode` round trip is exact for scripted commands.
3. **Radar-off sim tests:**
   - An emitting-off Blue gets no own radar hits, drops own FC within the gap
     rule, produces no Red RWR mode or spec 1 RWR component, and still launches
     on a flightmate's remote FC.
   - A flightmate takes over midcourse support.
   - With everything emitting (the default), all spec 1–4 tests and the legacy
     exact-match regression are unchanged.
4. **Controller tests:**
   - Decisions happen at t = 0, 1, 2 … only.
   - Same presentation + same policy gives an identical fight and ACMI;
     independent of batch order / worker.
   - Dead jets are skipped.
5. **Adapter test (Q):** layers 1–3 with the stated tolerances; outputs to
   `/workspace/spec5_outputs/`.
6. **Timing:** per-engagement and per-generation cost with a random-weight MLP
   policy at 1 s and 0.5 s decisions, reported like spec 4 `timing.txt`.

## Questions for Rusty

All answered by Rusty on 2026-09-26 (answers in bold after each question).

1. **Radar on/off (L):** include it now as an output (with the small sim change
   and a 5 s minimum dwell)? Recommended: yes. It is the stealth core, and the
   only way to avoid giving conservative Red a lock warning.
   **Answer (Rusty, 2026-09-26): yes: radar on/off included now, with the sim change and the 5 s minimum dwell.**

2. **Heading frame (H):** target-relative offset (crank / beam / drag are
   constants; mission axis when there is no target) vs nose-relative?
   Recommended: target-relative.
   **Answer (Rusty, 2026-09-26): target-relative heading (relative to the target bearing; the ingress direction when there is no target).**

3. **Salvo size (K):** a per-jet "pair" output (SAS vs SSA for the next
   contact), or keep doctrine a fixed SAS setting? Recommended: the pair
   output.
   **Answer (Rusty, 2026-09-26): the pair/single output per jet (the doctrine switches only when no pair is open).**

4. **Friendly positions (F):** exact and current (own-side info), or a 1 s
   delayed datalink copy? Recommended: exact.
   **Answer (Rusty, 2026-09-26): exact friendly positions.**

5. **Mission axis (B):** OK to give the network its heading relative to the
   ingress axis (so "home is south")? It implies spec 7 can reward escaping in
   that direction.
   **Answer (Rusty, 2026-09-26): the ingress-relative heading reference is OK.**

6. **Decision rate (N):** 1 s (about +15–20 % runtime, ~200 s per
   generation)? Recommended: yes.
   **Answer (Rusty, 2026-09-26): 1 s decisions.**

7. **Directed lock (fact 3):** Blue radar puts an FC "lock" on every tracked
   Red automatically, which triggers conservative Red. Keep that (radar on/off
   is the control) and defer a directed lock / STT? Recommended: defer.
   **Answer (Rusty, 2026-09-26): keep the automatic lock for now; a directed lock is deferred to a later small spec.**

8. **K = 6 contacts**, current target pinned to slot 0: OK?
   **Answer (Rusty, 2026-09-26): K = 6 with the current target pinned to slot 0.**

9. **No defend macro (M):** defense emerges from heading / altitude / speed
   only? Recommended: yes; add a macro only if spec 6 shows the GA cannot
   find it.
   **Answer (Rusty, 2026-09-26): no defend output.**

10. **Adapter tolerances (Q):** ±10 % for the decision-rate check and ±15 % for
    the perceived-only hand policy vs the scripted Blue. OK?
   **Answer (Rusty, 2026-09-26): ±10 % for the 1 s layer and ±15 % for the hand policy.**

## Implementation (2026-09-26)

Built as decided in **R**; the GA, genome and fitness are untouched.

**Code**
- `stealth_tactics/policy/` (new):
  - `view.py`: `build_blue_view(world, ac)` copies only whitelisted perception
    data into frozen dataclasses (`BlueView`, `OwnView`, `MateView`,
    `ContactView`, `MissileView`, `CueView`). No Red `Aircraft`, no track
    component (so no `true_range_m`), no Pk, no `primary_fc_target`. The missile
    envelope lookup table is passed by reference. `World.blue_view(ac)` wraps it.
  - `observation.py`: `ObsSpec` (231 = 21 + 6 × 28 + 3 × 14, named slices,
    `names()`, `index()`, `bounds()`), `order_contacts` (**D**),
    `build_observation(view, current_target, mate_targets)` → `(obs, ObsInfo)`.
    Rmax / Rne come from `envelope.for_states` on the perceived states (heading
    and speed from the freshest radar velocity estimate).
  - `action.py`: `ActionSpec` (13: heading, altitude, speed, 7 target logits,
    fire, radar, pair), `decode_action`, `select_target`, `encode_commands` (for
    the adapter). The decode maps are the approved linear maps, evaluated with
    exact rational arithmetic and a single rounding.
  - `controller.py`: `NetworkBlueController(policy, blue_ids,
    decision_period_s=1.0, radar_dwell_s=5.0)`. Decisions at t = 0, 1, 2 …; one
    (n, 231) batch per decision; commands held between decisions; the fire
    request is re-asserted every step; 5 s radar dwell (a refused flip is
    counted); the pair bit sets the jet's doctrine only when no SSA pair is open.
    Radar switches are logged as `radar` events (ACMI bookmark + `RadarMode`).
  - `adapters.py`: `ScriptedViaInterface`, `HandBlue`, `RandomMLPPolicy`.
  - `debug.py`: `obs-dump`.
- Sim:
  - `sim/sensors.py`: a jet with `radar_emitting = False` gets no own radar
    measurements (so its own FC drops by the gap rule) and is not heard by the
    spec 1 RWR track path. `sim/rwr.py` already honoured the flag for RWR modes.
    It can still use IRST, datalink and its RWR, and can launch on a
    flightmate's remote FC, with support handed to the FC holder.
  - `sim/world.py`: `blue_view`, `ssa_pair_open` (read-only), and `radar` in the
    frame state.
  - `acmi/exporter.py`: `RadarMode=0/1` is written only when a radar state
    changes. The default is on, so every existing ACMI and the legacy
    exact-match regression are byte-identical.
- CLI: `interface-adapter-test`, `obs-dump --presentation N [--stats-index] --t T --jet B2`,
  `network-smoke`.
- Tests: `tests/test_network_interface_spec5.py` (23 tests: layout, masking,
  ordering, bounds over fights, Rmax/Rne, shot_ready vs the world gates, truth
  leakage (move Red, NaN `true_range_m`, strict-proxy world, view walk, Red-truth
  swap), decode ranges and masking, round trip, fire gates, decision times,
  dwell, pair bit, determinism and ACMI, dead jets, radar-off gates, FC drop,
  default unchanged, layer 1 on 3 seeds, HandBlue obs-only). 208 tests pass.

**Adapter test results** (`/workspace/spec5_outputs/adapter_results.md`, 100 spec 4 stats seeds)

| variant | Blue kills | Blue losses | Blue shots | hit rate | Red shots | median dur s | end reasons |
|---|---|---|---|---|---|---|---|
| Scripted Blue (truth, direct) | 2.76 | 3.37 | 13.68 | 20 % | 15.18 | 282 | Bdead 58 Rdead 4 Wch 7 Rdep 6 cap 25 |
| Layer 2: script via interface, 1 s | 2.73 | 3.31 | 13.81 | 20 % | 15.41 | 294 | Bdead 52 Rdead 1 Wch 6 Rdep 5 cap 36 |
| Layer 3: HandBlue, obs only, 1 s | 2.95 | 3.22 | 14.04 | 21 % | 14.78 | 298 | Bdead 46 Rdead 3 Wch 5 Rdep 10 cap 36 |
| HandBlue, B2 + B3 radar-silent | 2.83 | 3.05 | 14.13 | 20 % | 14.25 | 312 | Bdead 39 Rdead 4 Wch 6 Rdep 9 cap 42 |

- **Layer 1 (0.5 s, 20 seeds): 19/20 byte-identical ACMIs, not 20/20.** All 20
  have an identical `SimResult` (time, kills, losses, end reason, every shot and
  outcome), and there are 0 fire-target misses. The cause is mathematical.
  With the approved linear maps (heading = ref + h·π, altitude = 100 + (z+1)·7450)
  the decoder cannot produce every float64 command. Near 0.4 rad, consecutive
  h values give headings about 1.7e-16 apart, but the float spacing there is
  5.5e-17, so about 2 in 3 headings cannot be reached. Altitude is similar.
  The encoder searches ±8 ulp and, when not firing, every heading reference;
  7.5 % of commands still differ by 1–2 ulp. In 19 fights that never shows at
  ACMI precision. In the 20th (seed 830255951060554192) one Red missile's pitch
  is about −1e-17 instead of +1e-17, so 117 lines print `-0.0` instead of `0.0`.
  Nothing else in the file differs. Three event logs differ only in the 16th
  digit of a float field. This was **not** loosened. Two options for Rusty:
  normalise `-0.0` to `0.0` in the exporter (which makes 20/20), or accept
  "outcome-identical, ulp-level" as the layer 1 criterion.
- **Layer 2 (±10 % or ±0.2): kills, losses and shots PASS.** The end-reason
  criterion (±5 per class) **FAILS** on the 100 seeds: cap 25 → 36, Bdead 58 → 52.
  Investigation (`layer2_noise_check.md`) points to noise, not a decision-rate
  effect:
  - The same 1 s hold with decisions half a period later gives cap 26 and
    Bdead 57 on the same seeds.
  - The two statistically identical 1 s runs differ from each other by 10 in cap.
  - 35/100 fights change end reason under either variant (chaotic divergence).
  - On the next 300 seeds, layer 2 goes the other way (cap 81 → 75).

  At n = 100 the ±5-per-class band is inside the fight-to-fight noise. Not loosened.
- **Layer 3 (±15 %): PASS** on kills (+6.9 %), losses (−4.5 %) and shots
  (+2.6 %). `HandBlue` sees only the (n, 231) array. It differs from the sketch
  in **Q** in three ways, which it needed to fight like the script:
  - it holds on the ingress axis until a ranged contact is within 55 km,
    shared across the flight, matching the genome's commit;
  - it prefers FC and doctrine-permitted contacts;
  - it egresses south after two losses.
- **Radar-silent variant:** 708 remote launches by the silent jets, in 100/100
  fights, and 0 own-FC launches by them. Summed over every step of 100 fights,
  the silent jets produced 0 Red RWR search/lock/support cues, 0 Red spec 1 RWR
  components, 0 own radar components and 0 own FC tracks. Support was handed
  over 612 times.
- **Observation ranges:** 200,519 vectors (100 presentations × every decision ×
  every live jet, hand and silent variants), 0 non-finite or out of range.
- **Timing** (8 workers, pop 50 × 24 + 64 = 1,264 engagements):

  | Blue | per generation | generations / 10 h |
  |---|---|---|
  | scripted | 169 s | 214 |
  | random MLP at 1 s | 187 s (+11 %) | 192 |
  | random MLP at 0.5 s | 218 s | 165 |

  Serial cost is 1.00 / 1.08 / 1.30 s per engagement.
