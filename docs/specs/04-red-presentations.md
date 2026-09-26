# Spec 4 — Red presentations (randomized Red flights, one pre-planned maneuver, early end, seeding)

Status: **implemented (2026-09-25).** Rusty approved the draft with changes:
**B** (every presentation is 6 Red, configurable), **C** (6-ship, data-driven
formations extensible to 8, bounded by a 25 x 25 NM box), **H** (data-driven
maneuver registry; how split and low-high divide 6 ships), the 360 s cap (Q1),
perfect Red GCI kept (Q2), Winchester Red leaves (Q3), surviving to the cap = got
away (Q4), and N re-costed by measurement. Everything else approved as
recommended. Each change is recorded under its decision as **Rusty's decision**
with his reasoning. See "Implementation" at the end for results, judgment calls
and outputs. All numbers are prototype placeholders, like specs 1–3a.

## What it does

Today Blue fights one fixed Red presentation from a YAML file (`default_4v3`:
3 Red, 43 NM, co-altitude, hot; or `cap_4v2`). A network trained on one
presentation learns that one fight. The goal for specs 5–9 is one shared Blue
network, evolved by a GA against **randomized** Red presentations, that finds
tactics that hold up across them. Blue's score across presentations is average
minus an inconsistency penalty (spec 7; only referenced here).

This spec adds five things and nothing else:
1. A **presentation**: a small, fully serializable description of one Red
   flight (6 jets, formation, start geometry, aggressiveness `a`, doctrine, one
   pre-planned maneuver) placed relative to a fixed Blue start.
2. A **sampler** that draws a presentation from a seed, and an **evaluation-set
   builder** that draws N stratified presentations per generation.
3. **Red HOT behaviour in presentation mode** so the drawn formation and
   altitude actually survive past the first few seconds (see facts below).
4. **Exactly one pre-planned maneuver per presentation** on a range trigger,
   built from `tactics/maneuvers.py` (the spec 3 hook).
5. The **early-end rule** Rusty approved in principle, plus an `end_reason` on
   every result.

It does **not** touch the Blue controller or the fitness function (specs 5–7).
The GA got an opt-in presentation mode (N presentations per generation,
benchmark set, process pool, champion reproduced from stored presentations) so
the smoke evolve can prove reproducibility; its fitness is the mean of today's
per-fight fitness until spec 7 replaces it.

## Already decided (carried in, not re-opened)

- Aggressiveness `a ∈ [0, 1]` **per Red flight**; 3 bands (spec 3 D1); drag
  depth `depth(a) = (1 − a)(alt − 100 m) + a × 1,000 m` (spec 3 change B);
  `T_cold = 5 + 35(1 − a)` s (D4); turn-away limit 2, then press if `a ≥ 0.5`
  else depart (D1/D10). Spec 4 only **draws** `a`.
- Doctrines are per contact, SAS or SSA (spec 3 change C). Red currently
  defaults to SAS; spec 4 draws it per flight.
- The defense state machine has priority over a pre-planned maneuver (spec 3,
  section 2: "spec 4 confirms"). Confirmed and detailed in **J**.
- Fight ends once every live Red has departed and nothing is in the air
  (spec 3 D9). Kept.
- GA cap: **raised from 240 s to 360 s** (Rusty, Q1).
- **Early-end rule (approved by Rusty in principle):** recorded as **K**.

## Facts found in the code before spec 4 (they shaped the decisions)

1. **One side dead already ends the fight.** `World.run` checks
   `not alive(BLUE) or not alive(RED)` at the top of every step and breaks.
   But: (a) `ended_early` is **not** set in that case (only the D9 departure
   end sets it), and (b) it ends **immediately**, even with missiles in flight.
   A Blue missile already fired at a live Red is cut off when the last Blue
   dies, so a trade kill is never counted (missiles aimed at dead jets already
   die with outcome `target_dead`). See **K**.
2. **Both-sides-Winchester does not end the fight.** Nothing checks ammo; such
   fights run to the cap. **Blue-only Winchester** also runs on, and Red keeps
   pursuing (what Rusty wants). **Red-only Winchester**: Red keeps flying pure
   pursuit with no missiles, which is meaningless (see **L**).
3. **Red HOT steers on truth.** `RedCAPController` (intercept mode) points each
   Red jet at the true position of its **nearest** Blue from t = 0, i.e. perfect
   GCI. Consequences:
   - The initial Red **heading/aspect is irrelevant**: every jet turns to pure
     pursuit within seconds (13°/s).
   - **Formation is not held**: each jet pursues its own nearest Blue, so a wall
     collapses as it closes.
   - HOT sets `cmd_alt = target altitude`, so any **starting altitude offset is
     erased** in ~30–40 s (85 m/s climb) and after every drag Red climbs or
     descends back to Blue's altitude.
   So formation and altitude draws are cosmetic unless Red HOT changes (**E**).
4. **Speed draws are cosmetic.** Aircraft accelerate at 30 m/s², so Red reaches
   its commanded 1.1 × cruise (280 m/s) within ~2 s of any start speed.
5. **Scenario support today.** `build_aircraft` takes explicit per-jet x, y,
   alt, heading, speed from YAML; any number of Red jets works (`R1..Rn`);
   `red_aggressiveness` is one value per scenario; `RedDefense` already accepts
   `per_jet_a`; `Aircraft.firing_doctrine` overrides doctrine per jet. There is
   no formation, no randomization and no sampler. Start ranges: `default_4v3`
   80 km (43 NM) lead to lead, `cap_4v2` 70 km (38 NM). Blue starts heading north
   at ~9,000–9,500 m, 255–260 m/s. Only one Red type exists (`RedFighter`,
   4 missiles, isotropic RCS 1.0).
6. **Relevant ranges (specs 1–3a).** Blue radar R50 vs Red 90 km (49 NM); Blue
   FC gate 50 NM. Red radar R50 vs F-35 nose 33 km (18 NM), FC ≈ 12.5 NM nose-on
   (≈ 16 NM at 45° aspect). Red RWR sees an F-35 radar inside 29 NM (so Red
   cannot see Blue's lock or support beyond 29 NM). Blue RWR sees Red radar at
   57 NM. Hot Rmax ≈ 28.8 NM at 25 kft, 20.6 NM at 15 kft. Default genome
   commit range 55 km (30 NM), shoot fraction 0.75.
7. **Seeding today.** `sim_seed = GAConfig.seed + 1000 + seed_offset` with
   `seed_offset = gen × 100 + i`, so **every genome in a generation fights with
   a different noise seed** (sensor, datalink, RWR and Pk streams). Fitness
   differences inside a generation therefore mix genome quality and luck. The
   earlier ACMI bug (re-run with `seed_offset = 9999`) is fixed by storing the
   champion's `seed_offset` and asserting the re-run fitness matches; spec 4
   must keep that guarantee when the presentation is also random (**M**).
8. **Runtime.** 0.49 s per engagement (4v3, spec 3). 4v4 will be somewhat
   slower (sensors scale with jet pairs); estimate ~0.6 s.

## Model

### 1. The presentation (`scenarios/presentation.py`, new)

A frozen dataclass, JSON round-trippable, holding **every drawn value** (not
just the seed), so a replay never depends on re-running the sampler:

```
Presentation(
  sampler_version, presentation_seed, sim_seed,
  n_red,                      # B (6)
  formation, spacing params,  # C
  range_nm, azimuth_deg, base_alt_m,   # D (relative to Blue lead)
  aggressiveness,             # F
  doctrine,                   # G
  maneuver: PreplannedManeuver(type, trigger_range_nm, params...),  # H, I
  red_jets: [ {id, slot, x, y, alt, heading, speed, hot_alt_m} ]   # derived
)
```

`build_presentation_aircraft(blue_scenario, presentation)` takes the Blue block
from a scenario YAML (default `default_4v3` Blue) and the Red block from the
presentation. The YAML path (`load_scenario` + `build_aircraft`) is unchanged,
so existing tests and scenario runs are unaffected.

### 2. Sampler and evaluation set

`sample_presentation(seed, cfg: PresentationConfig, stratum=None)`. All ranges
in `PresentationConfig`. Draws use independent named sub-streams (M), so adding
a field later does not shift the others. `build_eval_set(master_seed, gen, N)`
returns N presentations (N).

### 3. Red HOT in presentation mode (E)

A `PresentationRedController` (subclass or mode of `RedCAPController`) that
flies the formation and assigned altitudes until break-up, runs the pre-planned
maneuver, and otherwise defers to the existing `RedDefense`.

### 4. Pre-planned maneuver runner (H, I, J)

Per flight: fires once when the trigger range is reached. Per jet: flies the
maneuver's `ManeuverCmd` (from `maneuvers.py`) for its duration unless the
defense takes over. Events `preplanned_start`, `preplanned_skip`,
`preplanned_end` go to the log and ACMI bookmarks.

### 5. Early end and `end_reason` (K, L)

`SimResult.end_reason ∈ {blue_dead, red_dead, both_winchester, red_departed,
time_cap}`; `ended_early` is true for all but `time_cap` (presentation mode, K2 switch on).

## Decisions (Rusty approved 2026-09-25; his changes marked "Rusty's decision")

**A (approved) — What a presentation is, and the frame.**
- Options: (1) randomize Red only, relative to a fixed Blue start; (2) randomize
  both sides; (3) rotate/translate the whole picture too.
- **Recommendation: (1).** Blue starts as in the scenario YAML (4-ship heading
  north at ~9,000 m). A presentation is **one Red flight** placed relative to
  Blue lead. One Red flight per presentation; multiple Red flights are deferred.
- *Why:* there is no terrain or wind, so absolute position and heading carry no
  information; only relative geometry matters. A fixed Blue start keeps the
  network's job about reading Red, and Blue's opening formation belongs to the
  network (spec 5).

**B — Red flight size and composition. Rusty's decision: every presentation is
6 Red jets** (`PresentationConfig.n_red = 6`, all `RedFighter`; 8 is a config
change, and 8-ship formations are already in the menu). The draft's 2-ship/4-ship
mix is removed.
- *Rusty's reasoning:* Blue has a big advantage; with fewer Red the GA wins too
  easily and settles into a local minimum. Six pairs well with the Pk and
  missile counts (16 Blue missiles at Pk ≤ 0.6); eight would require
  near-perfect shooting.
- **Flags spec 7:** kills and "Red dead" should still be scaled by `n_red` when
  averaging (fixed at 6 today, but configurable).

**C — Formations and spacing. Rusty's decision: keep the menu idea, redesign
it for 6 ships, keep it extensible to 8, data-driven, and bound every formation
inside a 25 x 25 NM box.** More than one contact in a group (within ~3 NM) is
fine.
- **Data format** (`stealth_tactics/scenarios/presentation_menus.yaml`): each
  formation is an ordered list of **groups**. A group has an anchor
  `[right NM, back NM(, up m)]` and **slot offsets** inside it (or the shape
  `pair` = two abreast `elem` apart, or `single`). Values are numbers or
  expressions over the formation's drawn `params` (`+ - * /`, `sind`, `cosd`);
  `params` are `[lo, hi]` (uniform) or `{choice: [...]}`; `mirror: true` flips
  left/right with p = 1/2. Slot numbers follow group order; slot 1 (the lead)
  is at the origin and nobody is ahead of the lead. `halves` names the two
  halves (groups or `Group.k` slots) for maneuvers that divide the flight. A new
  formation is a YAML entry, no code. A formation is eligible when its slot
  count equals `n_red`.
- **6-ship menu (uniform draw):**

| Formation | Groups (2 jets each unless noted; pair spacing `elem` 1–2.5 NM) | Spacings | Halves (3/3) |
|---|---|---|---|
| wall | A centre, B left, C right, all abreast | `lat` 4–8 NM | B + A.1 / A.2 + C |
| box | FL, FR abreast in front; TL, TR singles in trail (mirrorable) | `lat` 5–10, `trail` 5–10 NM | FL + TL / FR + TR |
| ladder | A, B, C in trail, stepped up or down | `trail` 4–8 NM, `step` 500–1,000 m, sign 50/50 | A + B.1 / B.2 + C |
| echelon | A, B, C in echelon (mirrorable) | `gap` 3–6 NM at `ang` 30–60° aft | A + B.1 / B.2 + C |
| vic | A forward, B back-left, C back-right | `lat` 4–8, `back` 3–8 NM | B + A.1 / A.2 + C |
| champagne | A, B abreast in front, C trailing in the middle (mirrorable) | `lat` 5–10, `back` 4–9 NM | A + C.1 / C.2 + B |

- **8-ship entries** (not drawn at `n_red = 6`): `wall_8` (4 groups abreast,
  `lat` 3–6), `box_8` (4 pairs), `ladder_8` (4 pairs in trail, `trail` 3–6,
  `step` 400–800 m).
- **Box bound:** the largest extents are wall/vic 18.5 NM wide, wall_8 20.5 NM,
  ladder 16 NM and ladder_8 18 NM deep, all inside 25 x 25 NM. Tests check every
  formation at every parameter corner (both mirror signs) and 600 drawn
  presentations.

**D (approved) — Start geometry, altitude and speed.**
- **Range** (Blue lead to Red lead): uniform **40–60 NM**. Options: 30–50,
  40–60, 40–80.
- **Azimuth** of Red lead off Blue's nose: uniform **±40°**. Options: 0 (always
  on the nose), ±40°, ±90° (flank).
- **Red heading:** pointed at Blue lead (hot). No aspect draw (fact 3: Red turns
  to pursuit within seconds, so an aspect draw would be cosmetic).
- **Red base altitude:** uniform **6,000–12,000 m** (Blue ~9,000 m, i.e. Red
  3 km below to 3 km above), plus the formation stack; clipped to 1,000–13,500 m.
- **Speed:** **fixed 250 m/s** (as today). Not randomized.
- **Approved as recommended.**
- *Why:* 40 NM is just outside the hot Rmax (28.8 NM) and 60 NM is outside Blue's
  radar R50 vs Red (49 NM), so Blue sometimes starts with a picture and sometimes
  must find Red first. At ~540 m/s closure, 60 NM reaches Blue's shot range in
  ~110 s, well inside the 360 s cap (Rusty Q1). ±40° keeps Red inside Blue's
  ±60° radar field of regard with margin; flank/beam starts are a different
  problem (deferred). Speed is cosmetic (fact 4); a Red commit-speed factor
  (which would change Red missile range) is deferred to keep stats readable.

**E — Red HOT behaviour in presentation mode (approved).** Needed, or C and D
are cosmetic (fact 3). Red GCI quality (today perfect: true-position steering) is
a **future knob**.
- Options: (1) keep today's HOT (each jet pure-pursues its nearest Blue at
  Blue's altitude); (2) **hold formation until break-up, then individual
  pursuit; always fly the assigned altitude**; (3) full Red flight tactics.
- **Recommendation: (2).**
  - Lead pursues the nearest live Blue (truth, as today) at 1.1 × cruise.
    Wingmen station-keep on their slot offset in the lead's frame (same method
    as Blue `_hold_formation`, up to max speed to catch up). If the lead dies,
    the lowest live slot leads.
  - **Break-up per jet** at the first of: its pre-planned maneuver starts; it
    starts defending; it gets its own FC track on a Blue; any Blue within
    20 NM. After break-up: today's individual pursuit.
  - **Altitude:** every HOT/recommit command uses the jet's `hot_alt_m` (start
    altitude, changed only by an altitude maneuver), not the target's altitude.
    After a drag, a recommitting jet climbs back to `hot_alt_m`.
  - YAML scenario mode keeps today's behaviour exactly.
- *Why:* smallest change that makes formation, altitude and the altitude
  maneuvers mean something. The climb-back after a drag keeps the real cost of
  dragging low (a slow re-climb at 85 m/s). Truth steering (perfect GCI) is kept
  (Rusty, Q2).

**F (approved) — Aggressiveness draw (per flight).**
- Options: (1) uniform [0, 1]; (2) **banded: pick a band, then uniform inside
  it**; (3) Beta / clustered near the middle.
- **Recommendation: (2) with equal band weights (1/3 each), which is the same
  distribution as uniform, but the band is an explicit stratum** so the
  evaluation set gets every band equally (N). Drag depth, `T_cold` and
  press/depart follow `a` exactly as in spec 3 (unchanged).
- *Why:* band decides trigger and reaction, the biggest behavioural jump, so
  stratifying by band removes most of the set-to-set luck. Uniform inside the
  band keeps the continuous parts (drag depth, `T_cold`, press at 0.5)
  exercised. Weights stay configurable (e.g. if conservative Red, which drags
  before Blue's shot range (spec 3 follow-up), dominates the scores).

**G (approved) — Doctrine draw (per flight).**
- Options: (1) keep SAS always; (2) **SAS/SSA 50/50, independent of `a`**;
  (3) tie to `a` (aggressive → SSA).
- **Recommendation: (2)**, set on each Red jet via `Aircraft.firing_doctrine`,
  stratified in the evaluation set.
- *Why:* spec 3 stats show SSA Red is much more lethal at `a ≥ 0.75` (2.78 vs
  1.02 Blue losses), so Blue must learn to face both. Independence gives all
  six band × doctrine combinations; tying them would hide half the space.

**H — Pre-planned maneuver menu (exactly one per presentation). Rusty's
decision: menu approved, but data-driven so new maneuvers are easy to add, and
define how split and low-high divide 6 ships.**
- **Registry:** each menu entry is YAML (`maneuvers:` in
  `presentation_menus.yaml`): `kind`, `divide` (`all` | `halves`),
  `trigger_nm` window, `weight`, `params`. The behaviour of a kind is one
  decorated class in `tactics/preplanned.py` (`@register_kind("name")` with
  `start` and `step`), built on `maneuvers.py`. A new maneuver that reuses a
  kind is YAML only; a new kind is one small class.
- **Dividing 6 ships:** by the formation's `halves` (3 + 3 in every 6-ship
  formation, table in C). After mirroring, the half with the smaller mean
  `right` offset is the **left** half.
- **Menu** (equal weight, stratified in the evaluation set):

| Type | Kind / divide | What each jet flies | Params (uniform) |
|---|---|---|---|
| **split** | `turn_out` / halves | left half turns left, right half right (outward), `θ` off the bearing to its nearest Blue, 1.1 × cruise at its assigned altitude, for the leg | `θ` 30–60°, leg 30–60 s |
| **pump** | `cold` / all | threat on the tail, max speed, altitude held, for the leg; no shots | leg 20–40 s |
| **low_high_split** | `altitude` / halves | stays hot; half `low_half` descends to the low block, the other climbs; new altitudes persist | low 1,000–3,000 m; high `+1,500–3,000 m` (≤ 13,500 m); `low_half` 0/1 |
| **altitude_change** | `altitude` / all | whole flight stays hot and moves to a block; persists | low 1,000–3,000 m or high 11,500–13,500 m, 50/50 |

- An altitude maneuver is "done" when the jet is within 50 m of the target (or
  after 180 s); its target stays the jet's assigned altitude either way.
- *Why:* unchanged from the draft (sorting, shot timing, altitude sort and
  envelope). Without lookdown clutter (deferred), low options give Red no sensor
  benefit.

**I (approved) — Trigger.**
- Options: (1) time trigger; (2) **range trigger**; (3) draw one of the two per
  presentation.
- **Recommendation: (2).** Trigger range = the smallest **true** range from any
  live, non-departed Red in the flight to any live Blue (same truth picture Red
  already steers on). Windows (uniform):
  - split: 35–45 NM (needs room to develop a pincer)
  - pump: 30–40 NM (just outside Blue's hot Rmax, denies the first long shot)
  - low-high split: 32–45 NM
  - altitude change: 30–40 NM
  - The draw is capped at `start range − 5 NM` so a maneuver never fires at
    t = 0. If the trigger is never reached (e.g. Blue runs), it never fires
    (logged at the end as `preplanned_not_triggered`).
- *Why:* a time trigger fires at a random place because start range and Blue's
  behaviour vary; a range trigger fires at a meaningful point in the geometry.
  Every window is ≥ 30 NM, outside Red's 29 NM RWR detection of an F-35, so Red
  is always still HOT when it fires (see J). This is the classic way pilots
  brief a pre-planned move ("split at 40").

**J (approved) — Interaction with defense reactions and the turn-away limit.**
- **Recommendation:**
  1. The defense state machine always wins. A jet already DEFENDING, COLD,
     PRESSING or DEPARTED at trigger time **skips** the maneuver
     (`preplanned_skip`). A jet that starts defending during the maneuver
     **aborts** it for good (not resumed). Altitude set by an altitude maneuver
     stays as `hot_alt_m`.
  2. Pre-planned maneuvers **never count** as a turn-away and never set
     `departed` (the pump included). The limit stays about threat reactions only.
  3. RWR cues keep triggering the defense normally during a maneuver.
  4. **Shooting during the maneuver:** split, low-high and altitude change use
     the normal gates (≤ 60° off the nose etc.); **pump cannot fire** and drops
     support, like a spec 3 drag.
  5. When the maneuver ends the jet is broken up (E) and flies individual
     pursuit at `hot_alt_m`.
- Options for 1: skip / defer until HOT again / override the defense. Skip is
  the simplest and deterministic; deferring makes a "pre-planned" move happen
  at an unplanned point; overriding breaks spec 3's priority rule.
- *Why:* with ≥ 30 NM triggers, skips should be rare; the stats run measures it.

**K — Early-end rule (Rusty's decision; K2, `end_reason` and the switches approved).**
The fight ends early **only** when:
- **(a)** both sides are out of missiles and nothing is in the air. "Out" = no
  jet on that side can still shoot (every live jet has 0 missiles, or is a
  departed Red);
- **(b)** every jet on one side is dead;
- plus the **existing** D9 rule: every live Red has departed and nothing is in
  the air.
- If **only Blue** is out of missiles, the fight **continues** so Red can chase.
  *Rusty's reasoning:* killing as many as possible and then running away bravely
  may be a fine strategy, but only if Blue actually gets away, and jets lost on
  the way out should be a big penalty (spec 7 prices that penalty).
- Code today (fact 1): (b) is **already implemented**, but it ends at once and
  does not set `ended_early`. (a) is **not** implemented.
- Sub-decision **K2 — missiles in flight when a side is wiped out.** Options:
  (1) end at once (today); (2) **keep flying until every missile in flight has
  resolved, then end**. **Recommendation: (2).** *Why:* a Blue jet that dies
  after firing a good shot should still get the kill (a real trade). It adds
  at most tens of seconds and only matters in trades.
- Also add `SimResult.end_reason` and set `ended_early` for every early end.
- **Switches** on `SimConfig`: `early_end_winchester`,
  `finish_missiles_after_wipeout`. As built they default **off**, so scenario YAML
  and the exact-match legacy regression are unchanged; the presentation runner
  turns both on (judgment call, see Implementation). `end_reason` is always set;
  with K2 on, every early end also sets `ended_early`. If both sides die in the
  same step the label is `blue_dead`.
- **Q4 (Rusty):** surviving to the cap counts as getting away for now; an escape
  rule is deferred to spec 7.

**L (approved) — Red out of missiles (Winchester).**
- Options: (1) keep pursuing (today; fact 2); (2) **a Red jet with 0 missiles
  and none of its own in flight departs** (spec 3 D9 departure: `departed`,
  max speed away, never fires); (3) leave it to Rusty's early-end rule only.
- **Recommendation: (2).** It does not count as a turn-away.
- *Why:* a Winchester jet pressing a Blue flight with no weapons is not
  realistic and just burns sim time. With (2), the D9 rule then ends the fight
  once every live Red has left and nothing is in the air. Blue can still shoot
  a leaving Winchester Red while its missiles are in flight (a hard tail shot).
  **Rusty (Q3):** approved: a Red jet with no missiles and none of its own in
  flight tries to leave (existing departure behaviour, not a turn-away). As
  built it applies in presentation mode; YAML Red is unchanged.

**M (approved) — Seeding and reproducibility.**
- **Recommendation:**
  1. A presentation is a pure function of an integer `presentation_seed`:
     `rng = default_rng([presentation_seed, PRESENTATION_SALT])`, with named
     sub-streams per field group (size, formation, geometry, a, doctrine,
     maneuver) via `SeedSequence.spawn`, so adding a field later does not shift
     the others. No other RNG (not `world.rng`) is ever touched by the sampler.
  2. `sim_seed` (sensors, datalink, RWR, Pk streams) is **derived from the
     presentation seed**, not from the genome's index. Every genome facing
     presentation k sees the same starting noise seed (common random numbers),
     so within a generation differences come from the genome, not the dice.
  3. The full `Presentation` (every drawn value, `sampler_version`,
     `presentation_seed`, `sim_seed`) is stored with each champion and each
     history entry. The ACMI export and any replay **load the stored
     presentation and sim seed**; they never re-sample or pick a new seed. The
     existing check (re-run fitness must equal the logged fitness) stays and is
     extended to each presentation.
  4. `sampler_version` bumps whenever sampling changes; old runs replay from the
     stored JSON regardless.
- Options considered: seed only (re-sample on replay: breaks when the sampler
  changes); keep `gen × 100 + i` sim seeds (keeps genome-dependent luck).
- *Why:* this is exactly the failure mode of the earlier ACMI bug (a different
  seed on replay). Storing values, not just seeds, closes it.

**N (approved; cost re-measured) — Evaluation set per generation.** *(Touches spec 7 and 8.)*
- **Recommendation:**
  - **N = 24**, one full crossing of maneuver type (4) × `a` band (3) ×
    doctrine (2). Formation, geometry, `a` inside its band and
    maneuver params drawn freely from each cell's own seed.
  - **Same 24 for every genome in a generation; resampled each generation**
    from `SeedSequence([master_seed, gen])`. Elites are re-scored on the new set
    (no carried-over fitness).
  - Plus a fixed **benchmark set of 64** (seeded once from `master_seed`) used
    only to score each generation's champion for charts and champion selection
    (spec 9), so curves are comparable across generations.
- Options: N = 8 / 24 / 48; fixed set forever; resample each generation; a
  rolling mix.
- *Why:* a fixed set invites overfitting to 24 pictures; resampling with
  stratification keeps the set fair and the noise low. Because fitness comes
  from a different set each generation, "best ever" is judged on the benchmark
  set (spec 7 may refine this).
- **Measured cost (6 Red, 360 s cap, scripted Blue):** 0.97 s per engagement
  serial; 0.13 s wall per engagement on an 8-worker pool. Population 50 × 24 +
  64 benchmark = 1,264 engagements ≈ **165 s (2.7 min) per generation on 8
  cores, ≈ 218 generations per 10 h night** before the spec 6 network's own cost
  (the draft guessed ~90 s with 4 Red and 240 s).

**O (approved) — Scope and where the code lives.** (As built: see "What changed".)
- **Recommendation:** new `scenarios/presentation.py` (`PresentationConfig`,
  `Presentation`, `sample_presentation`, `build_eval_set`,
  `build_presentation_aircraft`); new `tactics/preplanned.py` (maneuver runner);
  presentation mode in the Red controller (E); `sim/world.py` early end (K, L)
  and `end_reason`; a `run_presentation(presentation, blue_controller)` helper;
  ACMI bookmarks and an ACMI header comment summarizing the presentation; CLI
  `sample-presentation`, `presentation-stats`, `presentation-replays`.
  **The GA loop is not changed** (multi-presentation evaluation and aggregation
  are specs 6/7); YAML scenario mode stays byte-for-byte the same behaviour.
- *Why:* keeps spec 4 small and testable with the scripted Blue, and gives
  specs 5–8 a stable interface.

## Deferred / out of scope

- Multiple Red flights per presentation, mixed Red types, flank/beam/rear starts
  (> ±40°), offset or flanking initial legs.
- Red GCI quality (today perfect true-position steering; a future knob).
- Winchester departure and the new early-end switches for YAML scenarios.
- A Red commit-speed factor (randomized Red speed / Mach).
- Randomizing Blue's start (Blue formation is the network's job, spec 5).
- CAP presentations (`red_mode: cap`) in the sampler.
- More than one pre-planned maneuver, or chained maneuvers.
- Flight-level Red reactions (wingman reacting to a flightmate's cue; spec 3
  deferred).
- A Blue "escaped" early end (Rusty Q4: spec 7).
- Lookdown/lookup clutter, notching, chaff (spec 3 deferred).
- Fitness, aggregation and the inconsistency penalty (spec 7), network inputs
  (spec 5), GA wiring (spec 6), overnight runner (spec 8), demo outputs (spec 9).

## What changed

- **Code:**
  - `stealth_tactics/scenarios/presentation.py` (new): `PresentationConfig`,
    `Presentation` (JSON round trip), menu loader + expression evaluator,
    `sample_presentation`, `build_presentation_aircraft`, `build_eval_set`,
    `build_benchmark_set`, `derive_sim_seed`.
  - `stealth_tactics/scenarios/presentation_menus.yaml` (new): formation and
    maneuver menus (data).
  - `stealth_tactics/tactics/preplanned.py` (new): `PresentationRedController`
    (E, H, I, J, L) and the kind registry (`turn_out`, `cold`, `altitude`).
  - `stealth_tactics/presentation_runner.py` (new): `run_presentation`,
    `export_presentation_acmi`.
  - `sim/world.py`: `early_end_winchester`, `finish_missiles_after_wipeout`,
    `END_*` reasons, `SimResult.end_reason` / `presentation` / `preplanned`,
    `World.can_still_shoot`.
  - `tactics/red_defense.py`: `RedDefense.force_depart` (Winchester).
  - `ga/evolution.py`: presentation mode (`presentations_per_gen`,
    `benchmark_size`, `presentation_config`, `workers`), `fitness_of`,
    `evaluate_presentations`, champion reproduction check; `sim_max_time_s`
    default 360 s.
  - `acmi/exporter.py`: optional `0,Comments=`; spec 4 events as bookmarks /
    messages.
  - CLI: `evolve --presentations N --benchmark M --workers W` (writes
    `champion.json`), `replay-champion`, `sample-presentation`,
    `presentation-stats`, `presentation-replays`; `--max-time` default 360.
  - `analysis/presentation_stats.py`, `analysis/presentation_replays.py` (new).
- **Tests:** 140 existing tests pass unchanged; 33 new in
  `tests/test_presentations_spec4.py` (173 total).

## Test plan (as built)

1. **Unit tests** (`tests/test_presentations_spec4.py`, 33 tests):
   - Sampling ranges over 1,200 presentations: 6 jets, 6-ship formations only,
     range / azimuth / altitude / speed / `a`-in-band / doctrine / trigger window
     (≤ start − 5 NM) / maneuver params; every Red ≥ 35 NM from every Blue at
     t = 0; maneuver, band, doctrine and formation frequencies within tolerance.
   - Determinism: same seed gives an identical presentation and JSON; the JSON
     round trip is identical; a stratum override changes only its three fields.
   - Same presentation (object or stored JSON) gives an identical fight, and the
     ACMI export is **byte-identical**; Blue is `Name=F-35A`; the summary is in
     `0,Comments=`.
   - 25 x 25 NM box: every formation at every parameter corner and both mirror
     signs (also: lead at the origin, nobody ahead, halves partition the slots,
     groups within 3 NM), plus 600 drawn presentations; 8-ship option samples and
     runs.
   - Evaluation set: 24 cells exactly once, deterministic, different per
     generation; benchmark 64 deterministic and near-balanced.
   - Noise seed from the presentation, not the genome's index (same per-fight
     fitness at index 0 and index 2); GA presentation mode reproduces its
     champion from the stored presentations.
   - Early end: both Winchester with nothing in the air ends (legacy flags off:
     runs to the cap); both Winchester waits for the missile in the air; Blue
     alone Winchester continues to the cap; one side wiped out with a missile in
     flight resolves it and counts the trade kill (legacy: cut off, no kill); Red
     wiped out with nothing in the air ends at once; `red_departed` label;
     legacy defaults unchanged.
   - Red: Winchester jets depart (not turn-aways); formation and assigned
     altitude held before the trigger; each maneuver fires once at its range
     trigger (within 0.5 NM) with the right headings / altitudes; defense aborts
     a maneuver (abort time = defend time, turn-aways = counted defends only); a
     jet not HOT at the trigger skips.
2. **Stats run:** `presentation-stats -n 100 --blue-test --timing` (see
   Implementation).
3. **TacView replays:** `presentation-replays`, one per maneuver type, each a
   different formation (see Implementation).

## Questions for Rusty (answered)

1. **Cap:** raised from 240 s to 360 s.
2. **Red GCI:** keep perfect true-position steering before contact; GCI quality
   is a future knob.
3. **Winchester Red:** tries to leave (existing departure, not a turn-away).
4. **Blue escape:** surviving to the cap counts as getting away for now; escape
   rule deferred to spec 7.
5. **Flight size:** always 6 Red (configurable; 8 later).

## Implementation (2026-09-25)

### Judgment calls

- **Legacy paths unchanged.** The new early-end switches default **off** in
  `SimConfig` and only the presentation runner turns them on; the Winchester
  departure lives in the presentation Red controller. So scenario YAML runs and
  the pre-spec-3 exact-match regression fingerprint are untouched. The one global
  change is Rusty's cap: `GAConfig.sim_max_time_s` and the CLI `--max-time`
  default are 360 s in YAML mode too, which changes YAML-mode fitness values
  (the time bonus is a fraction of the cap). Tests that pin a cap set it
  explicitly, so none changed.
- **GA presentation mode was added** (the draft deferred GA wiring to 6/7) so the
  smoke evolve can prove reproducibility. Its fitness is the mean of the existing
  per-fight fitness, a placeholder for spec 7 (no inconsistency penalty yet).
- **Champion selection** uses the benchmark score of each generation's best
  (decision N), so the champion is not always the highest eval-set fitness.
- **Halves:** 3 + 3 in every 6-ship formation, even when that splits a pair
  (wall, ladder, echelon, vic, champagne). A split turns the left half left.
- **Pump** holds the jet's assigned altitude (`drag` with the assigned
  altitude as target). **Altitude maneuvers** count as "done" within 50 m of the
  target (180 s cap); an aborted jet keeps the new assigned altitude.
- **Break-up** also happens if a wingman's leader dies: the next lowest live slot
  that has not broken up leads.
- **Replays** are natural sampler draws found by seed search (not stratum
  overrides). Replay D requires a real climb (high block ≥ 2,500 m above the base).

### Stats (`/workspace/spec4_outputs/presentation_stats.txt`)

100 random presentations (6 Red), default genome, 360 s cap, spec 4 early end.
B losses = Red kills. End reasons: Bdead = every Blue dead, Rdead = every Red
dead, Wch = both Winchester, Rdep = every live Red departed, cap = 360 s.

**Scripted Blue (default genome):**

| group | n | B shots | B hit | R shots | R hit | B kills | B losses | dur med s | % cap | end reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 100 | 10.5 | 16 % | 9.3 | 23 % | 1.66 | 2.11 | 295 | 29 % | Bdead 31 Rdead 1 Wch 6 Rdep 33 cap 29 |
| split | 32 | 9.5 | 20 % | 8.3 | 24 % | 1.94 | 2.03 | 300 | 22 % | Bdead 11 Rdead 1 Wch 2 Rdep 11 cap 7 |
| pump | 20 | 13.2 | 16 % | 12.0 | 20 % | 2.10 | 2.40 | 309 | 45 % | Bdead 5 Wch 2 Rdep 4 cap 9 |
| low_high_split | 28 | 10.0 | 12 % | 10.3 | 22 % | 1.21 | 2.21 | 296 | 32 % | Bdead 9 Wch 1 Rdep 9 cap 9 |
| altitude_change | 20 | 10.1 | 14 % | 6.8 | 26 % | 1.40 | 1.80 | 284 | 20 % | Bdead 6 Wch 1 Rdep 9 cap 4 |
| conservative | 33 | 2.8 | 0 % | 0.0 | – | 0.00 | 0.00 | 296 | 6 % | Rdep 31 cap 2 |
| middle | 36 | 14.1 | 10 % | 12.1 | 26 % | 1.44 | 3.17 | 321 | 44 % | Bdead 17 Wch 2 Rdep 1 cap 16 |
| aggressive | 31 | 14.5 | 25 % | 16.0 | 20 % | 3.68 | 3.13 | 258 | 35 % | Bdead 14 Rdead 1 Wch 4 Rdep 1 cap 11 |
| SAS | 46 | 10.6 | 16 % | 7.9 | 28 % | 1.65 | 2.24 | 294 | 22 % | Bdead 19 Wch 2 Rdep 15 cap 10 |
| SSA | 54 | 10.4 | 16 % | 10.5 | 19 % | 1.67 | 2.00 | 302 | 35 % | Bdead 12 Rdead 1 Wch 4 Rdep 18 cap 19 |

**Same, with the Blue scripted test reaction:** all: B shots 9.9, hit 16 %,
R shots 9.1, hit 3 %, B kills 1.61, B losses 0.30, median 360 s, 59 % at the cap
(Wch 4, Rdep 37, cap 59). Middle band: 100 % at the cap.

- The pre-planned maneuver fired in 100/100 fights; per jet 458 done, 142
  aborted by the defense, 0 skipped (every trigger is outside Red's 29 NM RWR
  range, as designed). Red departures 278 (84 Winchester), presses 0.
- Per-formation rows and a maneuver x band table are in the stats file.

### Surprises

- **The scripted Blue is too weak against 6.** Without the test reaction it loses
  2.11 jets per fight against 1.66 kills and is wiped out in 31 % of fights.
  The default genome ripples all 16 missiles at long range (mostly beamed or
  cranked out of energy), then has nothing left and no egress logic, and the
  recommitting Red finish it. With the test reaction Blue losses fall to 0.30
  but most fights run to the cap. This is the gap the GA should fill:
  missile economy and egress.
- **Conservative Red gives no fight.** In 33/33 conservative presentations Red
  drags on the first lock, uses both turn-aways and leaves with zero shots
  either way (spec 3 follow-up, now visible as a third of the evaluation set).
  Blue scores 0 kills there. Spec 7 must decide how to score "Red ran away"
  (the network may learn to shoot before locking, e.g. on remote or IRST tracks).
- Red SSA does not look deadlier than SAS against the scripted Blue (Red hit
  rate 19 % vs 28 %). Spec 3 saw the opposite with 3 Red.
- The draft's cost guess (~90 s per generation) was low: it measured 165 s
  (6 Red, 360 s cap).

### Replays (`/workspace/spec4_outputs/`)

Each has `.txt.acmi` (Blue `Name=F-35A`, summary in `Comments`), the stored
presentation `.json` and a `_timeline.txt`; `presentation_replays.txt` has all four.

- **A. split / wall** (seed 30; 57.5 NM, a = 0.41 middle, SAS): splits at 39.9 NM
  (31°, 45 s legs, 3 left / 3 right); Blue ripples 13 missiles within 8 s, all six
  Red beam; 1 hit; Red recommit, kill all 4 Blue by 333 s (2 Red lost); two Red go
  Winchester and leave. `blue_dead`.
- **B. pump / ladder** (seed 4; 48.5 NM, a = 0.91 aggressive, SSA): pumps at
  35.5 NM for 31 s, recommits; four Red crank on missile active and still die
  (4 kills); Red SSA pairs kill all 4 Blue by 249 s. `blue_dead`.
- **C. low-high split / box** (seed 51; 40.3 NM, a = 0.01 conservative, SAS):
  splits low (1,620 m) and high (+2,900 m) at 34.2 NM; the low half drags on
  Blue's lock at 38 s (aborting the maneuver); two drag cycles each, then all
  six depart; no shots. `red_departed` at 264 s.
- **D. altitude change (high) / vic** (seed 359; 46.4 NM, a = 0.90 aggressive,
  SSA): climbs from ~9,900 m to 12,750 m at 30.1 NM (two jets abort on
  missile-active cranks); 3 Red killed; Red kill all 4 Blue by 191 s.
  `blue_dead`.

Byte-identical check: re-running A and D from their stored JSON reproduces the
ACMI files byte for byte (also a unit test).

### Timing and cost (`timing.txt`)

- 0.97 s per engagement serial (mean fight 293 s simulated); 0.13 s wall per
  engagement on 8 workers.
- Per generation, population 50 × 24 + 64 benchmark = 1,264 engagements:
  ~165 s wall on 8 cores, ~218 generations per 10 h night (scripted Blue).

### Smoke evolve (`smoke_evolve.txt`, `smoke_evolve/`)

`evolve -p 12 -g 3 --seed 42 --presentations 24 --benchmark 64 --workers 8`:
134 s wall (1,056 engagements). Gen best (eval-set mean / benchmark): 157.30 /
151.72, 187.67 / 136.68, 190.16 / 156.87. The champion (gen 2) was re-run from its
24 stored presentations and the benchmark inside the GA, and again from
`champion.json` in a fresh process (`replay-champion`): mean fitness
190.15972222222226 identical, recorded fight (presentation 23, 441.67, 5 kills,
2 losses, `red_departed` at 237 s) identical, ACMI re-export byte-identical.
