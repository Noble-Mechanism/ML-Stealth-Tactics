# Spec 7 — Fitness v2 (compact)

Status: **v1 approved by Rusty (2026-09-26) and implemented; v2 (Red-alive
penalty, participation-gated survival bonus) approved 2026-10-02 and
implemented** (section "Fitness v2" at the end). It replaces the
placeholder fitness for the neural pipeline (`evolve-net`, `overnight`,
`clone-hand`'s scores). The genome GA (`evolve`) keeps its own `GAConfig`
fitness. Code: `stealth_tactics/fitness.py`. Weights:
`scenarios/fitness.yaml`.

## Weights live in a config file

- Every weight is in `scenarios/fitness.yaml`. A run can use another file
  (`--fitness my.yaml`) and override single keys (`--fw blue_loss=-200`,
  repeatable).
- New keys are allowed, so adding a term later means adding one key and one
  line in `fight_terms`.
- The weights a run used are stored in its config (`run_config.json`, every
  checkpoint and every `champion.json`). They are part of the config
  fingerprint, so `--resume` (and the overnight auto-resume) **refuses** a
  run whose weights changed. Start a new output directory instead.

## Terms (per fight), approved by Rusty 2026-09-26

| # | Term | Default | Key |
|---|---|---|---|
| 1 | each Red kill | 100 × (6 / n_red) | `kill`, `kill_ref_red` |
| 2 | each Blue loss | −150 | `blue_loss` |
| 3 | each Blue loss while egressing (instead of 2) | −250 | `blue_loss_egress`, `egress_off_deg` |
| 4 | each Blue jet alive at the 360 s cap that fired ≥ 1 missile (v2 gate) | +10 | `escape`, `escape_only_at_time_cap`, `escape_min_shots` (v2, 1) |
| 5 | each Red jet that left the fight out of missiles and was not killed (a quarter kill) | +25 | `red_winchester_depart` |
| 6 | each Blue missile fired | −2 | `shot` |
| 6b | a fight with no Blue shots and no kills (v1.1 fix) | −300 | `no_engagement` |
| 6c | each Red jet still alive and not departed at the end of the fight (v2) | −40 | `red_alive`, `red_alive_counts_departed` (false) |
| 7 | network fitness = mean over its presentations − 0.2 × std (v1.1; was 0.5) | 0.2 | `std_coef` |

Definitions:
- **Egress:** the dying Blue jet's heading was more than **120°** off the
  bearing to the nearest live Red jet at its last whole-second sample
  (running away or dragging cold). A beaming jet (90°) is not egressing. This
  uses truth geometry from the behaviour recorder, as a scorer, never as a
  policy input.
- **Escape:** Blue jets alive when the fight ends at the time cap
  (`end_reason == time_cap`). Fights that end early (all Red dead or
  departed, everyone Winchester) give no escape credit. Set
  `escape_only_at_time_cap: false` to count survivors at any end.
- **Red out-of-missiles departure:** a Red `depart` event whose reason is
  Winchester (since spec 4 L the only Red departure that ends a fight). A Red
  that departs and is then killed scores the kill only.
- **No engagement:** a fight with no Blue shots and no kills scores the loss
  terms plus the **−300 no-engagement penalty** (v1.1). It gets no escape and
  no Red-departure credit (`no_engagement_loss_only`). Since v2 it also gets
  the Red-alive penalty (a penalty, not a credit).
- **Aggregation:** population std (ddof 0) over the network's presentations
  in that generation. The benchmark and held-out test use the same formula.
- Novelty behaviour measures are unchanged (spec 6 L).

## Tests

`tests/test_fitness_spec7_play.py`:
- one test per term: kill scaling with n_red, loss vs egress loss, escape
  only at the cap, Winchester departure (not for turn-away departures or
  jets killed later), shots and total, and the no-engagement rule;
- aggregation (mean − 0.2 std, configurable coefficient);
- v1.1: a no-engage fight with 0 losses scores −300, and the demo's
  hall-of-fame fighter profile (~2.3 kills, ~1.1 losses, 14 shots,
  synthetic fights) aggregates above a pure runaway;
- YAML load, overrides and extra keys;
- egress flag from the recorder;
- changed weights change the config hash, and resume refuses.

## First demo (2026-09-26): 50 × 24, 5 generations, random start, wall

- The champion (benchmark −1.7, held-out −1.0) is a **non-engager**: it
  turns away, almost never shoots, and scores about 0.
- HandBlue scores −194 on the same benchmark (3.02 kills, 2.58 losses).
- The best engaging network (hall of fame r0_l2: 2.3 kills vs 1.1 losses on
  the held-out test) scores −23, because of the −0.5 × std term and −2 per
  shot.
- So v1 as approved makes "don't fight" a strong early attractor. This is
  the flee-with-0-kills local minimum the old placeholder punished.
- Options for Rusty, not applied (weights are in `scenarios/fitness.yaml`):
  - a no-engagement penalty;
  - a smaller `std_coef`;
  - or counting escape credit only for fights with kills.

## Fix approved by Rusty 2026-09-26 (fitness v1.1)

**The runaway hole.** Under v1 a fight with no shots and no kills scored only
its losses, which is 0 for a jet that simply turns away. Real fights have a
large spread, and the −0.5 × std term punished that spread. So in the first
demo a runaway (about 0 kills, about 0 losses) was champion at −1.7, above
the best fighter (−22) and above the hand policy (−194).

**The change** (`scenarios/fitness.yaml`, keys are configurable):
- `no_engagement: -300`: a fight with no Blue shots and no kills scores the
  loss terms plus −300 (a new term in `fight_terms`).
- `std_coef: 0.2` (was 0.5).

**Recomputed on the demo's saved per-fight results** (checkpoint g5, the
64-presentation benchmark; no re-simulation; v1 values reproduce exactly):

| network | v1 | v1.1 | no-engage fights | kills / losses per fight |
|---|---|---|---|---|
| champion runaway (id 108) | −1.7 | **−302.6** | 63 / 64 | 0.02 / 0.02 |
| hall-of-fame fighter (id 158, r0_l2) | −22.0 | **+36.7** | 0 / 64 | 2.41 / 1.22 |
| hall-of-fame r0_l1 (id 7) | −45.2 | −314.8 | 60 / 64 | 0.00 / 0.11 |

The hand policy's per-fight results were not saved (only its averages), so
its v1.1 score is not recomputed here.

## Fitness v2 (approved by Rusty 2026-10-02)

**Why: the sacrificial lamb.** Rusty's 6b overnight run (190 generations on
2f00bda) improved through kills (1.3 → 2.35 per fight) while Blue losses
stayed flat at about 1.05-1.1 per fight. A re-fly of the earlier run's
champion on the 64-fight benchmark (current physics) shows the tactic:
- In 48 of 64 fights exactly one jet dies. In all 48 it is the jet that
  advanced furthest (21 NM toward Red). It takes 90 % of Red's missiles and
  fires most of Blue's shots (4.0).
- The other three end 17 NM behind their start, point away from Red 72 % of
  the time, never come closer than 34 NM, fire 0.56 shots each, and are alive
  at the 360 s cap.

Under v1 this pays:
- A loss (−150) costs more than a kill earns (+100), so a second jet only
  joins in if it adds more than 1.5 kills per extra loss it risks.
- One shooter avoids the no-engagement penalty.
- The runners earn +10 each at the cap without any risk: +28 per fight, more
  than the +25 Red-departure term earns (+4 per fight; Red fires only 6.5 of
  its 24 missiles).
- The −250 egress loss never touches them, because they do not die.

**Changes** (keys in `scenarios/fitness.yaml`, all configurable):
1. **Red alive: `red_alive: -40`** for each Red jet still alive and still in
   the fight at the end.
   - A Red that left out of missiles (Winchester) is out of the fight, so it
     does **not** count as alive (`red_alive_counts_departed: false`). It
     keeps the +25 departure term.
   - The penalty also applies in no-engagement fights. Otherwise not fighting
     (−300) would beat a fight with one loss, one shot and no kill
     (−150 − 2 − 240 = −392). With it, not fighting scores −540.
   - Red jets surviving is mission failure, so the runners now cost the team
     about −40 for every Red jet they leave alive, and each kill is worth about
     140.
2. **Participation-gated survival bonus: `escape_min_shots: 1`.**
   - The +10 per Blue jet alive at the time cap goes only to jets that fired
     at least one missile during the fight. Participation means firing a
     missile, with no range condition, so this does not force banzai
     attacks.
   - `escape_min_shots: 0` restores v1.

**Re-scored on the 64-fight benchmark** (re-fly on this build, exact
`fight_terms`; v1 = these two keys switched off):

| network | v1 | v2 | kills | losses | escape v1 → v2 | Red alive | departs |
|---|---|---|---|---|---|---|---|
| earlier run's champion (lamb) | −6.4 (mean 23.0) | **−199.7** (mean −158.9) | +179.7 | −175.0 | +27.7 → +7.0 | −161.2 (4.03 Red/fight) | +4.3 |
| HandBlue | −273.8 (mean −230.5) | **−407.5** (mean −357.5) | +218.8 | −449.2 | +11.2 → +4.8 | −120.6 (3.02) | +19.9 |

Synthetic check (test): a lamb profile (one jet fires 4 and dies, three
runners, 1.5 kills) beats a shared fight (all four fire, 2 kills, 1.5
losses) under v1 (22 vs −16) and loses under v2 (−188 vs −176).

The weights are part of the config hash, so a v1 run directory does not
resume under v2. Start a new `-o`.
