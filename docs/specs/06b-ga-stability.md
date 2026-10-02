# Spec 6b: GA stability (immigration, moving-average stagnation trigger, fair champion pick)

Status: **implemented** (approved by Rusty 2026-09-30 as options 1, 2, 3 and 5
of the progress analysis of his first overnight run; the long-shot option was
not taken). Builds on Spec 6 (`06-neural-policy-neuroevolution.md`) and Spec 7.
The network, the interface, fitness and the simulation are unchanged.
Changed defaults change the config hash, so an old run directory is refused
on resume: start a new `-o` folder.

## Why (findings from the overnight run)

- The best eval score plateaued from about generation 30-43 on.
- The spec 6 stagnation boost (25 generations without a champion improvement
  → novelty weight 1.0 for 10 generations and σ × 2 for **every** genome)
  fired at generations 68 and 93 and crashed the population mean from about
  −46 to −130 each time, without finding anything better.
- The champion's benchmark (6.7, from generation 43) was likely inflated by
  the winner's curse: the best of many noisy candidates replaced the champion
  on any higher score, even one inside the noise. The per-fight standard
  deviation is about 147, so 24 fights give a standard error of about 30.
- The population had collapsed to one lineage (weight distances 2.5-11 inside
  it vs 18-22 between unrelated nets), with σ near the floor.

## A. Presentations 24 → 36

`NeuroConfig.presentations` and `--presentations` now default to 36: the eval
standard error falls by √(36/24) ≈ 1.22 (≈ 30 → ≈ 25 at std 147). Cost: 50 %
more eval fights per generation (see Timing).

## B. Stagnation trigger on the moving average of the best score

- Each generation computes `best_ma` = mean of the best eval score over the
  last `stagnation_window` (default 10) generations, including this one.
- The counter resets when `best_ma` sets a new all-time high (by more than
  `stagnation_min_delta`, default 0). Otherwise it goes up by 1.
- When the counter reaches `stagnation_gens` (default **60**), the trigger
  fires and the counter resets. The all-time high stays the reference, so on
  a lasting plateau it fires again every 60 generations.
- `--stagnation-gens 0` disables the trigger.
- Why a moving average: a champion change is a noisy event, and a single
  lucky or unlucky generation moves a 10-generation average by only a tenth.
- `stagnation_metric="champion"` restores the spec 6 counter (generations
  without a champion improvement).
- What happens when it fires (default): a stagnation immigration of
  `stagnation_immigrants` = **16** newcomers (C). Nothing global changes.
- The spec 6 global boost is still in the config (off by default):
  `boost_gens` (novelty weight `boost_w` for N generations) and
  `boost_sigma_mult` (σ multiplier for every genome). Spec 6 behaviour is
  `stagnation_metric="champion", stagnation_gens=25, boost_gens=10,
  boost_sigma_mult=2.0, immigrate_every=0, stagnation_immigrants=0,
  champion_margin_k=0, top_every=5, presentations=24`.

## C. Immigration ("fresh blood") instead of the global boost

- Periodic immigration: every `immigrate_every` generations (default **25**;
  0 = off), on generations g with (g + 1) % 25 == 0, the next generation gets
  `immigrants` newcomers (default **8** of 50).
- The newcomers take the last offspring slots. The next generation is the
  2 elites (unchanged), then 50 − 2 − 8 = 40 offspring of the truncation
  parents, then the 8 immigrants. This is equivalent to replacing the 8 worst
  non-elite members: under truncation selection the bottom of the population
  never breeds anyway.
- Elites are never replaced: at most `population − elites − 1` immigrants,
  so at least one offspring remains.
- The elites and the remaining offspring are exactly what a run without
  immigration breeds (same parents, same random draws). No σ, novelty weight
  or selection rule changes for the rest of the population.
- Sources (documented ratio, `immigrant_random_frac`, default **0.5**):
  - round(8 × 0.5) = **4 random nets**: fresh `init_weights`, σ = `sigma_init`
    (0.02), origin `immigrant_random`, a new lineage root.
  - **4 mutated descendants of hall-of-fame cells outside the champion's
    lineage**. These are cells whose genome's lineage root (founder id,
    inherited by every descendant) differs from the champion's root, and
    whose id is not the champion's. The best benchmark cell comes first, and
    the eligible cells are cycled through. Each is one `mutate` step (its own
    self-adaptive σ) from the cell's genome, with origin `immigrant_hof` and
    `hof_cell` in its lineage.
  - If no cell is eligible (common late in a run that collapsed to one
    lineage), all immigrants are random.
  - **Since 2026-10-02 (newcomer fixes): see the update at the end.** Cells
    are now drawn round-robin over distinct cells, at most 2 per cell, with
    other-founder cells first and then the other filled cells.
- Protection: random immigrants score far below a tuned population (around
  −300 vs −50), so plain truncation would discard them at once.
  - Members whose lineage has an `immig` mark (inherited by descendants) from
    the last `immigrant_protect_gens` (default 10; **25 since 2026-10-02**)
    generations are protected.
  - `immigrant_parent_slots` (default 1; **2 since 2026-10-02**) of the 10
    truncation-parent slots go to the best protected members by combined
    rank, so newcomer lineages produce about 10 % (now 20 %) of the offspring.
  - A protected member that ranks in the top 10 on merit takes that slot; no
    extra slot is added.
- Interaction with the stagnation trigger:
  - The trigger fires: the next generation gets a *larger immigration*
    (16 = 2 × 8, same random/hall-of-fame mix) instead of a global σ × 2 /
    novelty boost.
  - Both fall on the same generation: only the larger one is done (16), not
    24.
  - The periodic schedule is not shifted by a stagnation immigration.
- Logging: `history` rows carry `immigrants`, `immigrant_reason` (`periodic`
  / `stagnation`), `immigrant_sources` (random / hof counts and cells),
  `n_immigrant_lineage` and `stagnation_trigger`. Events of type
  `immigration` and `stagnation` are written too.

## D. Fair champion pick (paired benchmark, k·SE margin)

- Every generation (`top_every` = 1, was 5), the top `top_n` = **3** members
  by eval fitness are benchmarked on the fixed 64-fight benchmark.
- The challenger is the benchmarked member (not the champion itself) with the
  highest benchmark score.
- With d_j = challenger − champion per benchmark fight j (the same 64 fights
  and seeds, so the comparison is paired), SE = std(d, ddof=1) / √64.
- The challenger replaces the champion only if **mean(d) > k · SE** (k =
  `champion_margin_k`, `--champion-k`, default **1**) **and** its benchmark
  score (mean − 0.2 × std) is higher, so the reported champion score never
  drops.
- The first benchmarked generation sets the first champion. With k ≤ 0 the
  spec 6 rule applies (a higher benchmark score wins).
- Pairing removes the fight-difficulty part of the noise: the unpaired SE of
  a 64-fight mean is about 147 / 8 ≈ 18. The paired SE is much smaller when
  two networks are related (the usual case: a child of the champion).
- The hall of fame and the top-5 list (`final_top`) still take every
  benchmarked member, as before.
- Logging: every history row has `champion_decision` (challenger id and
  benchmark, champion id and benchmark, `mean_diff`, `se`, `margin` = k·SE,
  `accepted`, `reason`) and `champion_id`. An event is written when the
  champion changes, or when a challenger had a higher score but failed the
  margin ("kept").
- Benchmark cache (`bench_cache`, default on): the incumbent champion's stored
  64 per-fight scores, and those of any unchanged genome benchmarked before
  (returning elites; matched by id and identical weights), are reused instead
  of re-running the same fights and seeds. Benchmarks are deterministic, so
  the cache changes only the run time. Decisions are identical with the
  cache off (tested), and resume does not need it (it is not checkpointed).
  `timing.jsonl` records `bench_runs` and `bench_cached` per generation.
  Typically 1-3 new networks are benchmarked per generation (64-192 fights)
  vs 64 before (and 192 every 5th generation).

## E. History file

Every generation the run directory gets `history.json` (full rows, without
the long parent / elite / seed lists) and `history.csv` (one row per
generation). The overnight `progress/` folder gets copies with each progress
refresh.

CSV columns:

- Scores: generation, best_fitness, mean_fitness, median_fitness, best_ma,
  best_bench.
- Champion: champion_bench, champion_gen, champion_id.
- Best member: best_id, best_origin, best_kills, best_losses.
- Mutation σ: mean_sigma, min_sigma, max_sigma.
- Novelty: mean_novelty, novelty_w.
- Stagnation and immigration: stagnation, stagnation_trigger, boosted,
  immigrants, immigrant_reason, immigrants_random, immigrants_hof,
  n_immigrant_lineage, n_clone_origin.
- Hall of fame: hof_cells (count).
- Champion decision: challenger_id, challenger_bench, paired_diff, paired_se,
  champion_margin, champion_accepted, champion_reason.

`fitness.png` now shows best / mean / median / moving average, the champion
benchmark, markers for immigrations, stagnation triggers and champion
changes, and a σ panel (mean, min-max band, log scale). `progress.md` shows
the recent events and a 30-generation table with median, moving average,
σ, the challenger decision and immigrants.

## Knobs

| config field | CLI flag | default | meaning |
|---|---|---|---|
| presentations | `--presentations` | 36 | eval fights per network per generation |
| stagnation_gens | `--stagnation-gens` | 60 | generations without a new MA high before the trigger (0 = off) |
| stagnation_window | `--stagnation-window` | 10 | MA window |
| stagnation_min_delta | | 0.0 | a new high must exceed the old by more than this |
| stagnation_metric | | best_ma | `champion` = spec 6 counter |
| stagnation_immigrants | `--stagnation-immigrants` | 16 | immigrants when the trigger fires |
| boost_gens / boost_w / boost_sigma_mult | | 0 / 1.0 / 1.0 | spec 6 global boost (off) |
| immigrate_every | `--immigrate-every` | 25 | periodic immigration (0 = off) |
| immigrants | `--immigrants` | 8 | immigrants per periodic immigration |
| immigrant_random_frac | `--immigrant-random-frac` | 0.5 | random share; rest from hall-of-fame cells outside the champion's lineage |
| immigrant_max_per_cell | | 2 | at most this many newcomers per hall-of-fame cell per immigration |
| immigrant_hof_lineage | | prefer_other | `strict` = only cells with another founder (6b rule) |
| immigrant_protect_gens | `--immigrant-protect-gens` | 25 (was 10) | protection window for immigrant lineages |
| immigrant_parent_slots | `--immigrant-parent-slots` | 2 (was 1) | parent slots reserved for protected members |
| top_every / top_n | `--top-n` | 1 / 3 | benchmark the top n every `top_every` generations |
| champion_margin_k | `--champion-k` | 1.0 | k in mean(d) > k·SE (≤ 0: higher score wins) |
| bench_cache | | true | reuse stored benchmark scores of unchanged genomes |

## Timing

These were measured on the box (8 cores) with `overnight --workers 4` from a
fresh all-random population (generations 0 and 1), with nothing else running.

| build | gen 0 total (eval + bench) | gen 1 total (eval + bench) | fights per generation |
|---|---|---|---|
| 8dcf6ea (24 presentations, top-1 benchmark) | 398 s (378 + 19) | 445 s (421 + 23) | 1,200 + 64 |
| spec 6b (36 presentations, top-3 benchmark) | 632 s (566 + 66) | 703 s (630 + 72) | 1,800 + 3 × 64 |

- Ratio: about **1.6×** per generation. Of that, 1.5× comes from the eval fights
  (36 vs 24). The benchmark adds about +45-50 s per generation for two extra
  benchmarks at 4 workers (about 22 s per 64-fight benchmark).
- The cache does not help in the first generations: elites did not stay in
  the top 3 of a fresh eval set. In a tuned population, the incumbent and
  returning elites are often in the top 3 and are not re-run, so expect 1-2
  benchmark runs per generation (+0-25 s on the box at 4 workers).
- Before 6b, every 5th generation already ran three benchmarks.
- Scaling: time per generation is roughly proportional to fights / workers
  (per-fight cost depends on how long the fights last, which changes as the
  policy evolves). On an 8-core server at `--workers 8` it is about half the
  box's 4-worker numbers if the cores are comparable. Expect about 1.5-1.6×
  whatever that server measured per generation with the old defaults: 200 s
  → about 310-320 s; 604 s → about 950 s.

## Tests (`tests/test_neuro_spec6b.py`)

- Defaults (36 presentations; trigger and immigration defaults) and the CLI
  flags → config.
- Moving-average trigger: MA values, counting, firing every N generations on
  a plateau; a single dip or spike does not reset it; 0 disables it.
- The stagnation trigger immigrates 16 without changing σ (elites keep σ) or
  the novelty weight.
- Immigration:
  - Count and sources: half random, half from eligible hall-of-fame cells
    only, never a cell of the champion's lineage.
  - Elites and the other offspring are byte-identical to a run without
    immigration.
  - A reserved parent slot goes to an immigrant lineage.
  - Random fallback when no cell is eligible; elites are never replaced.
- Champion margin on paired fights:
  - +3 SE is accepted, +0.5 SE is rejected.
  - k = 4 rejects; k = 0 gives the spec 6 rule; a lower score is rejected.
  - Pairing beats the unpaired spread.
- Champion decisions logged, the champion score never drops, each genome is
  benchmarked at most once with the cache, and decisions are identical
  without the cache.
- `history.csv` / `history.json` written every generation with the new
  columns; chart rendering (skipped without matplotlib).
- Fixed seed: two runs and a run resumed across an immigration and a
  stagnation trigger are byte-identical (checkpoint and history.csv); another
  seed differs.
- Sim: immigration with top-3 benchmarking gives byte-identical checkpoints
  with 1 and 4 workers.
- Spec 6 tests: the old boost is tested through the legacy config; the
  selection test runs with the cache off (it counts benchmark calls).

## Update 2026-10-02: newcomer fixes (approved by Rusty with fitness v2)

Rusty's 6b overnight run (190 generations) showed two problems:
- All 4 hall-of-fame newcomers were copies of the same cell (r0_l2) at every
  immigration. It was the only cell whose founder differed from the
  champion's; the population had collapsed to one founder.
- Newcomer lineages fell from 8-9 members to 2-7 within 10 generations, and
  none ever produced the best network.

Changes (all in `NeuroConfig`):
- **Distinct cells** (`immigrant_cells`):
  - Cells holding the champion's genome, and the champion's own behaviour
    cell, are excluded.
  - Draw order: cells with another founder first (best benchmark first),
    then (`immigrant_hof_lineage="prefer_other"`, the default) the other
    filled cells.
  - Round-robin, with at most `immigrant_max_per_cell` = 2 newcomers per cell
    per immigration. Slots left over are random nets.
  - `"strict"` keeps the 6b founder rule, with the per-cell cap.
- **Longer protection:** `immigrant_protect_gens` 10 → **25**, which equals
  the immigration period. `immigrant_parent_slots` 1 → **2**, so newcomer
  lineages can breed for a whole period and produce about 20 % of the
  offspring meanwhile.
- New CLI flags: `--immigrant-protect-gens`, `--immigrant-parent-slots`.

Tests (`tests/test_neuro_spec6b.py`):
- New defaults.
- Cell order, distinctness, the cap of 2 per cell and the random remainder.
- The strict rule.
- 25-generation protection and 2 reserved parent slots.
