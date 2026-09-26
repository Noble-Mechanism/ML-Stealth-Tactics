# Spec 7 — Fitness v1 (compact)

Status: **approved by Rusty (2026-09-26) and implemented.** It replaces the
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
| 4 | each Blue jet alive at the 360 s cap ("escaped", placeholder rule) | +10 | `escape`, `escape_only_at_time_cap` |
| 5 | each Red jet that left the fight out of missiles and was not killed (a quarter kill) | +25 | `red_winchester_depart` |
| 6 | each Blue missile fired | −2 | `shot` |
| 7 | network fitness = mean over its presentations − 0.5 × std | 0.5 | `std_coef` |

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
- **No engagement:** a fight with no Blue shots and no kills scores only the
  loss terms (no escape and no Red-departure credit). No extra penalty term
  is added (`no_engagement_loss_only`).
- **Aggregation:** population std (ddof 0) over the network's presentations
  in that generation. The benchmark and held-out test use the same formula.
- Novelty behaviour measures are unchanged (spec 6 L).

## Tests

`tests/test_fitness_spec7_play.py`:
- one test per term: kill scaling with n_red, loss vs egress loss, escape
  only at the cap, Winchester departure (not for turn-away departures or
  jets killed later), shots and total, and the no-engagement rule;
- aggregation (mean − 0.5 std, configurable coefficient);
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
