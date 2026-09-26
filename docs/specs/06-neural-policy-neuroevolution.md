# Spec 6 — Neural policy and neuroevolution (the network, how it evolves, checkpoints, reproducibility)

Status: **approved by Rusty (2026-09-26) and implemented.** Rusty went
decision by decision and approved all 18 decisions (A–R) and the test plan
exactly as recommended, with one addition on **E**: per-element and per-jet
networks are added to the deferred list, and the genome/checkpoint format
carries a network count (1 for now) with the jet-to-network map in the config.
The questions at the end are now recorded answers, and the Implementation
section at the end gives the results. Numbers are still prototype
placeholders, as in specs 1–5; fitness is today's placeholder (spec 7 owns it).

**Update (Rusty, 2026-09-26, play package):** fitness is now spec 7 v1
(`docs/specs/07-fitness.md`; network fitness = mean − 0.5 × std). The default
start population is **all random** (`--init mixed` keeps 10 clones + 40
random), and the default Blue start is the **30 NM line-abreast wall**
(`--blue-start diamond` keeps the old geometry). The numbers below were
measured before this change.

## What it does

Spec 5 fixed the seam: each Blue jet gets a 231-input observation and returns
13 outputs, and `NetworkBlueController(policy, blue_ids)` turns them into sim
commands once per second. Spec 6 fills in the `policy`:
1. the **network**: architecture, activations, output squashing and parameter
   count (one network shared by all four jets);
2. the **evolution algorithm**: population, selection, mutation, crossover,
   elitism, initialization and diversity preservation;
3. the **bookkeeping**: re-evaluation, champion selection, checkpoint and
   resume, determinism, and how champion replays are stored.

It does **not** choose the fitness (spec 7; spec 6 uses today's placeholder
`fitness_of`) or the overnight runner, scheduling, monitoring and GPU question
(spec 8). The demo outputs are spec 9.

## Already decided (carried in, not re-opened)

- **One network for all four Blue jets**, told its slot (one-hot in the
  observation). GA / neuroevolution, no gradient RL.
- **Interface (spec 5, approved):** 231 inputs (own 21 + 6 contacts × 28 +
  3 wingmen × 14), 13 outputs (heading, altitude and speed in [−1, 1]; 7 target
  logits; fire, radar and pair bits), 1 s decisions, all launch gates in the
  World, and no truth anywhere in the observation.
- **Evaluation set (spec 4 N):** 24 stratified presentations per generation,
  the same for every member, resampled each generation. Elites are re-scored on
  the new set. A fixed 64-presentation benchmark scores each generation's best,
  and the champion is the best benchmark score. Same presentation + same policy
  → byte-identical fight (spec 4 M).
- **Measured cost (spec 5):** random-weight 231-64-64-13 MLP at 1 s decisions:
  **187 s per generation** at population 50 × 24 + 64 benchmark (1,264
  engagements) on 8 workers, i.e. **0.148 s wall per engagement**, **≈ 192
  generations per 10 h night**.

## Facts found in the code (they shaped the decisions)

1. **The current GA** (`ga/evolution.py`, `GeneticAlgorithm._run_presentations`)
   is a genome GA:
   - tournament selection (k = 3) with 2 elites;
   - uniform crossover plus per-gene mutation on `TacticsGenome`;
   - a process pool per call;
   - the champion is reproduced from its stored presentations, then the
     benchmark is re-run, with a byte-exact check.

   It has no checkpoint or resume and no diversity mechanism. Its per-job
   payload is a small genome dict.
2. **A network genome is big to ship.** 19,853 float64 weights = 159 KB. Sending
   weights with every one of the 1,264 jobs per generation would pickle about
   200 MB per generation. Sending each member once per worker per generation
   (about 8 MB) is negligible.
3. **The forward pass is not the cost.** One forward for a 4-jet batch takes
   about 5 µs. The spec 5 overhead (+11 %) is `blue_view` plus observation
   building. Architecture size up to about 50k weights does not change the
   per-generation cost noticeably. Population size and presentations do.
4. **Random networks fight badly but not zero.** In the spec 5 smoke run (4
   presentations), a random MLP made 1–13 shots and 0–3 kills, and flipped
   radar constantly (up to 103 flips refused by the dwell). Early generations
   will be mostly noise with occasional lucky kills, which makes the early
   fitness signal very noisy.
5. **A good obs-only teacher exists.** `HandBlue` (spec 5 layer 3) matches the
   scripted Blue within 7 % on kills, losses and shots using only the
   observation. It can label observation → action pairs for behaviour cloning.
   The scripted Blue itself uses truth, so cloning it maps "truth decisions" to
   "perceived inputs", which is an imperfect teacher.
6. **Evaluation noise is large.** Spec 5's layer 2 check showed that 35 of 100
   fights change end reason from a half-second shift in decision phase.
   Per-fight outcomes are chaotic, so 24-fight means have a real standard
   error. The scripted Blue's per-fight fitness spread (spec 4 stats) is of the
   same order as the difference between decent and poor tactics. Selection
   must tolerate noise: rank-based, with elites re-scored.
7. **Determinism risk: BLAS threads.** numpy `@` on small matrices is
   single-threaded today, but OpenBLAS can split larger products across threads
   with a different summation order. Workers must pin `OPENBLAS_NUM_THREADS=1`
   (and MKL / OMP equivalents) so a fight is bit-identical under any worker
   count.
8. **No torch on the box** (numpy only). Everything below is numpy. Behaviour
   cloning is a small numpy Adam loop over about 10⁵ samples, which takes
   minutes on CPU.

## Model

- **Policy** = `MLPPolicy(weights: np.ndarray[float64], arch)`, a callable
  mapping `(n, 231) → (n, 13)` with no state between calls. It plugs into
  `NetworkBlueController` unchanged.
- **Genome** = `NetGenome(weights, sigma, lineage)`:
  - `weights`: the flat vector;
  - `sigma`: its own mutation step size;
  - `lineage`: id, parent id, birth generation and origin ("random" or
    "clone").
- **One generation:**
  1. Draw the evaluation set (spec 4 N).
  2. Evaluate each member on the 24 presentations (process pool; weights sent
     once per worker).
  3. Compute each member's behaviour descriptor (**L**).
  4. Rank members by fitness and novelty.
  5. Keep the elites.
  6. Pick parents by truncation.
  7. Mutate to make the children.
  8. Score the generation's best on the benchmark.
  9. Update the champion, the novelty archive and the hall of fame.
  10. Write the checkpoint.

## Decisions

**A — Network architecture and size.** *(approved by Rusty 2026-09-26)*
- Options:
  1. MLP 231-32-32-13 (**8,909** parameters);
  2. **MLP 231-64-64-13 (19,853)**;
  3. MLP 231-128-128-13 (47,885);
  4. one hidden layer, 231-128-13 (31,373).
- **Recommendation: 231-64-64-13, tanh hidden layers** (the network already
  timed in spec 5).
- *Why:*
  - Two layers let it combine "which slot" with "which contact" features.
  - At 20k weights a mutation-only GA is known to work (deep GA results reach
    millions). 47k would slow a population-50 search with no evidence it is
    needed.
  - 8.9k may be too tight for 6 contacts × 28 features with slot-dependent
    behaviour.
  - The cost is the same for all four (fact 3).
  - tanh keeps activations bounded, so a large mutation never produces huge
    outputs. ReLU dead units are a risk with mutation-only training.

**B — Contact encoder: flat vs set / attention.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **flat**: the 6 slots concatenated into the MLP input (A);
  2. **shared per-contact encoder + pooling** (Deep-Sets style): one small
     28→32 net applied to each slot, mean- and max-pooled, then concatenated
     with own and wingman inputs. Target logits are scored per slot from that
     slot's embedding plus the trunk. About 14k parameters;
  3. a small self-attention layer over the 6 contacts (about 20–25k
     parameters).
- **Recommendation: flat (1) for spec 6**, and build the set encoder (2) as a
  switchable `arch` with its own test. Use it if the first nights show the
  flat net cannot learn to use slots 1–5 (for example, it always targets
  slot 0).
- *Why:*
  - Spec 5 pins the current target to slot 0 and orders the rest by range, so
    slot position carries meaning. A permutation-invariant encoder would have
    to get that back from a flag.
  - Flat is simplest to mutate and to debug with `obs-dump`.
  - The set encoder's advantage (one learned "contact reader" reused 6 times)
    is real, so we keep it ready but unproven.
  - Attention is overkill at K = 6 and awkward for mutation.

**C — Recurrence / memory.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **feedforward only**;
  2. a small GRU / Elman state (for example 16 units) carried between 1 s
     decisions;
  3. a hand-made memory in the observation (spec 5 deferred "time since
     contact lost").
- **Recommendation: feedforward (1).** Recurrence is deferred.
- *Why:*
  - The observation already carries the memory that matters: missile states,
    time since the last launch, pair open, age of each track, kills and time.
  - Recurrence doubles the weight count and makes the policy's behaviour depend
    on its history, which is harder to replay, test and read for Rusty.
  - Mutation of recurrent weights is fragile.
  - Revisit only if champions visibly fail at things that need memory (such as
    re-acquiring a dropped track).

**D — Output heads and squashing.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - heading, altitude and speed: `tanh`, straight into spec 5's linear decode
    maps;
  - 7 target logits: linear, with argmax over present slots plus "none"
    (spec 5 I);
  - fire, radar and pair: `tanh`, with > 0 meaning on.
  - Fully deterministic, with no sampling.
  - Weights initialised with scale 1/√fan_in and hidden biases with 0.1 noise,
    exactly as `RandomMLPPolicy`.
- Options: sigmoid for bits (same thing up to scale); stochastic actions (spec
  5 deferred).
- *Why:* this matches the tested smoke network. With deterministic outputs,
  same weights + same presentation = same fight, which the whole
  reproducibility chain depends on.

**E — Shared network, slot input, parameter encoding.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - One network for all four jets, as already decided. It is batched as one
    `(n, 231)` call per decision; the slot one-hot is already input.
  - Genome = a flat **float64** vector of 19,853 weights in a fixed layer order
    (`W1, b1, W2, b2, W3, b3`), plus `sigma` and lineage.
  - The architecture is written with the genome, alongside hashes of
    `OBS_SPEC.names()` and `ACTION_SPEC`, so an old champion is refused (not
    misread) if the interface changes.
- Options: float32 (half the size; only a concern if we ever send many);
  per-slot heads (4 × the output layer).
- *Why:*
  - float64 keeps every fight bit-exact with today's float64 sim.
  - Size is not an issue (159 KB per network).
  - Per-slot heads would weaken the shared-network idea for little gain,
    since the slot one-hot already lets the net specialise.
- **Addition (Rusty, 2026-09-26):**
  - Per-element networks (2 nets: one for the lead element, one for the
    wing element) and per-jet networks (4 nets) go on the deferred list as
    future options to explore.
  - The genome and checkpoint format carries `n_networks` (1 for now); the
    genome holds a list of weight vectors, one per network. The
    jet-to-network map (`jet_network`, `(0, 0, 0, 0)` today) lives in the
    config (`NeuroConfig`), not in the genome. So a later move to 2 or 4
    networks needs no format redesign.
  - Multi-network evolution is not implemented. The loader and the config
    refuse any count other than 1 (and any map other than all zeros) with a
    clear message: "genome has n_networks=N; this build supports only 1
    shared network (spec 6 E). Per-element (2) and per-jet (4) networks are
    deferred ...".

**F — Evolution algorithm.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **Simple GA with Gaussian mutation, truncation selection and elitism**
     (the "deep neuroevolution" GA of Such et al., 2017);
  2. **OpenAI-ES** (Salimans et al., 2017): a mean vector, a mirrored Gaussian
     population, rank-shaped fitness, and an Adam step on the estimated
     gradient;
  3. **CMA-ES**. Full covariance is 19,853² ≈ 3.9 × 10⁸ entries (3 GB) and
     learns on an n² time scale, so it is impossible here. Separable
     (diagonal) sep-CMA-ES is feasible;
  4. **NEAT** (evolves topology from a minimal net).
- **Recommendation: (1), a mutation-only GA with elitism**, plus the diversity
  mechanism in **L**.
- *Why:*
  - **Budget fit.** About 190 generations × 50 members is about 9,500
    evaluated networks per night. OpenAI-ES estimates a gradient from each
    generation. With 50 samples in 20k dimensions that estimate is very noisy;
    published ES runs use hundreds to thousands per step. A GA at population
    50 is in the regime where it has been shown to work, and it tolerates
    noisy fitness (fact 6) because elites are re-scored every generation.
  - **Local minima (Rusty's worry).** ES and CMA-ES collapse to one search
    point, and if that point sits in the "drag and never commit" basin the
    whole search sits there. A GA keeps a spread of different solutions, and
    **L** makes that spread deliberate.
  - **NEAT** is poorly suited to 231 fixed inputs: evolving connections from
    a minimal net would take far more generations than a night allows.
  - **Simplicity.** It reuses the existing GA loop, champion reproduction and
    benchmark logic almost as is.
  - OpenAI-ES stays a documented fallback (same genome, same evaluation). If
    the GA plateaus with healthy diversity, spec 6.1 can try it at the same
    budget.

**G — Population size and evaluation budget.** *(approved by Rusty 2026-09-26)*

Cost per generation = (P × presentations + 64 benchmark) × 0.148 s on 8
workers (measured, fact above). Night = 10 h.

| population × presentations | engagements / gen | s / gen | generations / night | networks evaluated / night |
|---|---|---|---|---|
| 32 × 24 | 832 | 123 | 292 | 9,350 |
| **50 × 24** | **1,264** | **187** | **192** | **9,600** |
| 64 × 24 | 1,600 | 237 | 152 | 9,730 |
| 100 × 24 | 2,464 | 365 | 99 | 9,880 |
| 100 × 12 | 1,264 | 187 | 192 | 19,200 (noisier fitness) |
| 200 × 12 | 2,464 | 365 | 99 | 19,750 (noisier fitness) |

- **Recommendation: 50 × 24** (spec 4 N, as costed).
- *Why:*
  - The number of networks evaluated per night is nearly flat at about 9,600
    for 24 presentations, so the choice is breadth (population) against depth
    (generations).
  - 50 gives about 190 generations for mutations to accumulate, and enough
    members for **L** to keep several lineages alive.
  - Halving the presentations to 12 doubles the networks evaluated, but
    doubles the fitness variance, and that noise is already the main enemy
    (fact 6).
  - Spec 8 may revisit this once real per-generation times (evolved nets fight
    differently from random ones) are measured.

**H — Selection and elitism.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - **Elites: 2**, copied unchanged and **re-scored** on the new
    generation's 24 presentations (spec 4 N; no carried-over fitness).
  - **Parents: truncation**, the top **T = 10** (20 %) by the combined
    fitness and novelty rank (**L**), each chosen uniformly.
  - 48 children, each a mutated copy of one parent.
- Options: tournament k = 3 (today's GA); fitness-proportional; elites 1–5.
- *Why:*
  - Truncation plus Gaussian mutation is the simplest form shown to work at
    this size, and uses ranks only, so it is robust to fitness scale and
    outliers.
  - Two elites guard against losing the best to a bad mutation without
    freezing the population.
  - Re-scoring on fresh presentations stops a lucky 24-fight draw from
    staying on top for ever.

**I — Mutation.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - Gaussian noise added to **every** weight: `w' = w + σ·N(0, 1)`.
  - Each individual carries its own **self-adaptive σ**:
    `σ' = σ · exp(τ·N(0,1))`, with τ = 0.2 on the log scale. Children inherit σ and σ is clipped to [0.002, 0.2].
  - Initial σ = 0.02. Hidden weights are about ±0.07 (1/√231), so a step moves
    the first layer by about 30 % of its typical weight, and later layers by
    less.
- Options:
  1. a fixed σ (Such et al. used a single tuned σ);
  2. a per-layer σ scaled by 1/√fan_in;
  3. a 1/5th-success rule on the population;
  4. "safe mutation" that scales per-weight noise by output sensitivity
     (deferred).
- *Why:*
  - No one knows the right σ for this problem yet, and a wrong fixed σ wastes
    a night.
  - With self-adaptation, good step sizes hitch-hike with good weights, at the
    cost of one float per genome.
  - The clip stops σ collapsing to zero (freezing the search) or exploding
    (random search).
  - **L**'s stagnation response also raises σ when progress stops.

**J — Crossover.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **none** (mutation only);
  2. uniform weight crossover;
  3. layer-wise or unit-wise crossover (swap whole hidden units with their
     in and out weights).
- **Recommendation: none (1)**. Unit-wise crossover (3) is kept as an option
  behind a flag, off by default.
- *Why:*
  - Two networks can compute the same thing with hidden units in a different
    order (permutation symmetry), so averaging or mixing their weights usually
    breaks both.
  - Deep-GA results were obtained without crossover.
  - Unit-wise swaps respect that structure somewhat, but only help between
    close relatives; worth a test once lineages exist.

**K — Initialization: random vs behaviour-cloning warm start.** *(approved by Rusty 2026-09-26)*
- Options:
  1. **all random** (the `RandomMLPPolicy` scale, one seed per member);
  2. **all cloned**: behaviour-clone `HandBlue`, then perturb each copy;
  3. **mixed**: a few clones plus mostly random members.
- How cloning works:
  - Run `HandBlue` on about 200 presentations (not the benchmark ones) and log
    about 10⁵ (observation, action) pairs.
  - Fit the MLP in numpy with Adam: MSE on heading, altitude and speed in the
    tanh domain, cross-entropy on the target logits, and a bit-match loss on
    fire, radar and pair.
  - Accept the clone only if it reaches ≥ 80 % of `HandBlue`'s mean kills on
    100 held-out presentations.
- Pros of cloning:
  - It skips the long phase where random nets must discover "fly toward
    contacts, shoot when shot_ready" (fact 4). Early fitness is less noisy and
    the night is spent refining tactics instead of discovering flight.
- Cons of cloning (novelty):
  - It starts the search inside the human tactic's basin: a hand-made
    commit-bracket-shoot. Selection may just polish it, which is exactly the
    local minimum Rusty worries about.
  - `HandBlue` is itself a simplified tactic, including a fixed 55 km commit
    and an egress after two losses.
- **Recommendation: mixed (3).** 10 of 50 are clones (one clone perturbed with
  σ = 0.02 under 10 seeds) and 40 are random, tagged in `lineage.origin`.
  Novelty (**L**) protects the random lineages while they are still worse.
  Each night's report says whether the champion descends from a clone or from
  a random member. Rusty may prefer a **random-only control night** first to
  see what emerges with no human prior (Question 3).
- *Why:* it gets a working baseline from generation 0 without betting the
  whole population on the human tactic, and the lineage tag makes it
  measurable whether the prior helped or trapped the search.

**L — Diversity and novelty preservation (local-minimum guard).** *(approved by Rusty 2026-09-26)*
- Options:
  1. none;
  2. fitness sharing / speciation in weight space;
  3. **novelty-weighted ranking** with a behaviour descriptor and an archive
     (novelty search with local competition, in simplified form);
  4. MAP-Elites quality-diversity grid;
  5. island model (for example 2 × 25 with migration);
  6. stagnation restarts.
- **Recommendation: (3) + (6) + a hall of fame.**
  - **Behaviour descriptor (BD)** per member, the mean over its 24 fights of
    about 8 normalised numbers:
    1. fraction of time radar-silent;
    2. mean launch range / Rmax;
    3. shots per fight;
    4. mean closest approach to any Red;
    5. mean altitude;
    6. fraction of time heading away from the nearest contact
       (beam / drag);
    7. formation spread (mean distance between jets);
    8. time of the first shot.

    It is computed from the fight record (`SimResult` events and frames at 1 s)
    by one function, so spec 7 can change it.
  - **Novelty** = mean distance to the k = 10 nearest neighbours among the
    current population plus an **archive**. The archive adds 2 random members
    per generation and is capped at 1,000 (oldest out). Cost is trivial.
  - **Combined rank** for parent selection = rank(fitness) + w · rank(novelty),
    with **w = 0.5**. Elites stay by fitness only, so the best is never lost.
  - **Stagnation response.** If the benchmark champion has not improved for
    **25 generations**:
    - raise w to 1.0 for 10 generations;
    - multiply every σ by 2;
    - log a `stagnation` event.
  - **Hall of fame.** For a coarse 3 × 3 grid over two BD axes (radar-silent
    fraction × mean launch range / Rmax), keep the best-benchmark network of
    each cell. Rusty then gets a menu of different tactics, not just one
    champion.
- *Why:*
  - Pure fitness selection on a noisy, deceptive objective converges on the
    first tactic that works, for example "everyone fires at max range and
    drags".
  - Novelty on behaviour (not weights; weight distance is meaningless under
    permutation symmetry) rewards trying different ways of fighting, and the
    archive stops the population cycling through the same few.
  - MAP-Elites is the fuller version, but needs many more evaluations than
    9,600 a night to fill a grid. The hall of fame gives its reporting benefit
    for free, because it reuses benchmark scores already computed.
  - Islands halve the effective population. Kept as a later option.

**M — Re-evaluation and champion selection (ties to spec 4 N).** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - Keep spec 4 N as is: 24 fresh presentations per generation for everyone,
    elites re-scored, and only the generation's best by 24-fight fitness goes
    to the 64-presentation benchmark. **Champion = best benchmark score.**
  - Add one change: send the **top 3** by 24-fight fitness to the benchmark
    **every 5th generation** (+128 engagements, about +19 s, on those
    generations; about 2 % on average).
  - At the end of the night, re-score the hall of fame and the top 5 by
    benchmark on a **held-out test set of 256** presentations (seeded
    separately; up to 14 networks × 256 = 3,584 engagements, about 9 minutes), and report that as the
    unbiased score.
- Options: benchmark every member (+64 × 50 fights per generation, about
  6× the cost: no); a larger per-generation set; a rolling mean of an elite's
  past scores.
- *Why:*
  - The best of 50 on 24 noisy fights is biased upward (winner's curse), so
    sometimes the true best is second or third. Occasional top-3 benchmarking
    catches that cheaply.
  - Choosing the champion by the benchmark also biases the benchmark score
    upward over about 190 picks, so a set never used for selection is needed
    for an honest number.

**N — Determinism, seeding and parallel evaluation.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - All randomness comes from `SeedSequence([master_seed, gen, purpose])` with
    purposes such as "init", "mutation" and "novelty archive". A child's noise
    is drawn from `SeedSequence([master_seed, gen, "mutation", child_index])`,
    so a result never depends on worker order.
  - Presentations follow spec 4 M.
  - Workers get the generation's population once, through the pool
    initializer or a per-generation broadcast. Jobs are
    `(member_index, presentation)`, and results are gathered by index.
  - Workers set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`
    before importing numpy (fact 7).
  - Guarantee: same master seed + same code → a byte-identical run history,
    champion and ACMI, with **1 or 8 workers**.
- *Why:* reproducibility is how Rusty can trust a surprising champion. It must
  be re-playable on any machine and after any resume.

**O — Checkpointing and resume.** *(approved by Rusty 2026-09-26)*
- **Recommendation:** after every generation, write `checkpoint.npz` plus
  `checkpoint.json` atomically (write to a temporary file, then rename),
  keeping the last 3. They hold:
  - every member's weights, σ and lineage;
  - the elite list, novelty archive and hall of fame;
  - the champion and its benchmark score;
  - the history rows;
  - the stagnation counter;
  - the RNG state (implied by the seeds in **N**, but stored anyway);
  - a config hash, the git commit, and the `OBS_SPEC` / `ACTION_SPEC` hashes.

  `evolve --resume RUN_DIR` refuses a mismatched config or interface hash.
  Resuming at generation g gives **the same bytes** as an uninterrupted run.
  Size is about 50 × 159 KB + archive (1,000 × BD) ≈ 8 MB per checkpoint.
- *Why:* a 10 h night will be interrupted sometimes (reboots, power, Rusty
  needing the machine). Spec 8 decides when and how resume happens; spec 6
  makes it exact.

**P — Champion storage and replay.** *(approved by Rusty 2026-09-26)*
- **Recommendation:** extend today's `champion.json` (spec 4) with:
  - `champion_weights.npz` (the weights, σ, lineage and the `arch` block);
  - the interface hashes;
  - the 24 stored presentations and per-fight fitness of the generation where
    it won;
  - its benchmark and test-set scores.

  `replay-champion RUN_DIR` rebuilds the `MLPPolicy`, re-runs the stored
  presentations (byte-exact check, as today) and exports the ACMI of the best
  and the worst fight. The hall-of-fame entries are stored the same way under
  `hall_of_fame/<cell>/`.
- *Why:* the existing reproduction check becomes the network's too. Keeping
  the interface hashes means a replay fails loudly after an interface change
  instead of silently fighting differently.

**Q — What goes to spec 7 and spec 8.** *(approved by Rusty 2026-09-26)*
- **Spec 7 (fitness):**
  - the fitness formula (spec 6 uses today's `fitness_of` mean per fight as a
    placeholder so the loop runs);
  - penalties for refused fire requests and radar use;
  - the escape rule;
  - whether novelty enters fitness or stays in selection only;
  - final behaviour-descriptor definitions and weights.
- **Spec 8 (runner):**
  - overnight scheduling, auto-resume, logging and monitoring;
  - disk layout and retention;
  - the GPU question (spec 5 P: not useful at this batch size);
  - a lock-step batched runner if ever needed;
  - an e-mail or report at the end of the night.
- **Spec 6 delivers** the pieces spec 8 calls: `NeuroGA.step()`,
  `save_checkpoint`, `load_checkpoint`, and the champion and hall-of-fame
  writers.

**R — Scope and where the code lives.** *(approved by Rusty 2026-09-26)*
- **Recommendation:**
  - `stealth_tactics/neuro/` (new):
    - `mlp.py`: `MLPPolicy`, `arch`, init, flat ↔ layers;
    - `genome.py`: `NetGenome`, mutation, σ adaptation;
    - `neuroga.py`: the loop, selection and elites; it reuses the spec 4
      evaluation-set and benchmark builders;
    - `novelty.py`: descriptor, archive, hall of fame;
    - `clone.py`: `HandBlue` data collection and the numpy Adam fit;
    - `checkpoint.py`.
  - CLI:
    - `evolve-net --pop 50 --gens N --workers 8 --out RUN_DIR [--resume]
      [--init mixed|random|clone]`;
    - `clone-hand` (writes the clone weights plus a report);
    - `replay-champion` accepts network runs.
  - The existing `evolve` (genome GA) is untouched and keeps its tests.
- *Why:* a clean parallel path. The genome GA stays as the scripted
  baseline for comparisons in spec 9.

## Deferred / out of scope

- Recurrence / memory (**C**); attention encoder (**B** 3); stochastic actions.
- Per-element networks (2 nets) and per-jet networks (4 nets) (**E**, Rusty's
  addition). The format already carries `n_networks` and the config the
  jet-to-network map; only the count of 1 is accepted today.
- OpenAI-ES / sep-CMA-ES as alternatives (**F**), in spec 6.1 if the GA
  plateaus.
- Crossover beyond the unit-wise flag (**J**); safe mutation (**I** 4).
- Full MAP-Elites; island model (**L**).
- Co-evolving Red; curriculum on presentation difficulty (for example easy
  bands first).
- Fitness (spec 7); overnight runner, GPU, monitoring (spec 8); demo outputs
  (spec 9).

## What changes

- New `stealth_tactics/neuro/` package and CLI commands (**R**).
- `NetworkBlueController` is used as is. The pool workers pin BLAS threads
  (**N**). Instead of a broadcast initializer, the population is written once
  per generation to an `.npz` that each worker loads and caches by path
  (the same effect: no weights are pickled per job).
- Docs: this spec (with an Implementation section after the build), README and
  `design.md`.

## Test plan

1. **Network:**
   - `MLPPolicy` output shape and heads (tanh ranges, linear logits);
     flat ↔ layers round trip; 19,853 parameters;
   - batch-order independence; identical outputs from a genome saved and
     loaded.
2. **Toy task (fitness rises), no sim:**
   - fitness = −MSE between the network's outputs and `HandBlue`'s actions on
     a fixed set of 2,000 logged observations;
   - pop 50, 40 generations: best fitness never decreases (elites) and
     improves by at least 50 % from generation 0;
   - same run with σ adaptation: σ stays inside its clip.
3. **Sim smoke:** pop 8 × 4 presentations × 5 generations with benchmark 8.
   It runs end to end and writes a checkpoint, champion, hall of fame and
   ACMI. `replay-champion` reproduces the champion fights byte-exactly.
4. **Reproducibility:**
   - the same seed twice gives an identical history, champion weights and
     ACMI;
   - 1 worker vs 8 workers gives identical results;
   - the thread-pinning environment is set in workers.
5. **Checkpoint / resume:** 6 uninterrupted generations vs 3 + resume + 3 give
   byte-identical checkpoint, history and champion. Resume refuses a changed
   config or interface hash.
6. **Selection and novelty:**
   - elites are copied unchanged and re-scored;
   - parents come only from the top T;
   - novelty ranks identical behaviour descriptors lowest;
   - the archive cap holds;
   - the stagnation trigger fires on a synthetic flat history and raises σ
     and w;
   - hall-of-fame cells keep the best benchmark score.
7. **Cloning:** the clone reaches ≥ 80 % of `HandBlue`'s kills on 100 held-out
   presentations, and the clone data never uses benchmark or test seeds.
8. **Mutation statistics:**
   - the empirical std of `w' − w` matches σ;
   - σ′ follows log-normal adaptation;
   - children from the same parent and generation seed are identical;
     different child indices differ.
9. **Timing:** one generation at pop 50 × 24 + 64 on 8 workers (expected about
   187 s ± 15 %), reported like spec 5 `timing.md`.

## Questions for Rusty — recorded answers (Rusty, 2026-09-26)

1. **Algorithm (F):** simple mutation-only GA with elitism and novelty,
   instead of OpenAI-ES or CMA-ES?
   **Answer (Rusty, 2026-09-26): yes — mutation-only GA with novelty (not ES,
   CMA-ES or NEAT).**
2. **Population (G):** 50 × 24, 100 × 24 or 100 × 12?
   **Answer: 50 × 24.**
3. **Warm start (K):** mixed, all random or all cloned; a random-only control
   night first?
   **Answer: mixed start, 10 clones + 40 random. A random-only control night
   is offered (`evolve-net --init random`) but not run first.**
4. **Novelty (L):** novelty term w = 0.5, stagnation boost after 25
   generations, the ~8 behaviour-descriptor axes?
   **Answer: yes — w = 0.5, boost after 25 stalled generations, the 8
   behaviour measures as proposed.**
5. **Hall of fame (L):** 3 × 3 over radar-silent fraction × launch range / Rmax?
   **Answer: yes, 3 × 3 on radar-off share × launch range / Rmax.**
6. **Architecture (A/B):** flat 231-64-64-13 MLP with the set encoder built but
   off?
   **Answer: yes — flat net; the per-contact (set) encoder is built behind a
   switch (`--encoder set`), off by default.**
7. **Recurrence (C):** defer memory?
   **Answer: yes, memory deferred.**
8. **Crossover (J):** none by default?
   **Answer: no crossover; the whole-unit swap is built behind a switch
   (`--unit-swap`), off by default.**
9. **Champion checks (M):** top-3 benchmark every 5th generation and a
   256-presentation held-out test at night's end?
   **Answer: yes.**
10. **Mutation (I):** self-adaptive σ or a fixed tuned σ?
    **Answer: self-adaptive σ starting at 0.02, clipped to 0.002–0.2.**

## Implementation (2026-09-26)

Built as specified in `stealth_tactics/neuro/`: `mlp.py` (policy, set
encoder behind a switch, interface fingerprint), `genome.py` (network genome,
`n_networks`, mutation, unit swap, save/load), `novelty.py` (8 behaviour
measures, kNN novelty, hall-of-fame cells), `evaluate.py` (spawn pool, BLAS
threads pinned to 1), `neuroga.py` (`NeuroConfig`, `NeuroGA`, checkpoints,
champion storage, `replay_network_champion`), `clone.py` (behaviour cloning
and gate), `toy.py` (no-sim toy task), `checkpoint.py` (deterministic npz and
json). CLI: `clone-hand`, `evolve-net`, and `replay-champion` detects network
runs. The genome GA (`ga/`, `evolve`) is untouched. Fitness is today's
placeholder (`fitness_of`); spec 7 owns it.

### Test plan results (19 new spec 6 tests; whole suite 228 passed)

1. **Network:** 19,853 parameters (flat); tanh heads in range, linear
   logits; flat ↔ layers round trip exact; batch order independent (a
   single-row forward differs from a batched one only in the last bit,
   1e-12, from BLAS gemv vs gemm; fights always use the same batching, so
   they stay bit-exact); save → load gives identical outputs; set encoder is
   slot-equivariant. `n_networks` round-trips (1, and 2 / 4 are written
   correctly); loading 2 or 4 is refused with the "supports only 1 shared
   network … deferred" message, as are `n_networks=2` and a non-zero
   jet-to-network map in the config; a changed interface fingerprint
   is refused.
2. **Toy task:** best MSE 0.634 → 0.198 in 40 generations (−68.7 %, needs
   ≥ 50 %); best never decreased; σ stayed in the clip (mean 0.0069 at the
   end, range 0.0029–0.0133).
3. **Sim smoke** (8 × 4 × 5, benchmark 8, test 8): runs end to end, writes
   checkpoints, champion, hall of fame and ACMIs; `replay-champion`
   reproduces the stored fights exactly and both ACMIs byte-identically.
4. **Reproducibility:** the same seed twice and 1 vs 8 workers give identical
   history, champion weights and ACMI bytes; workers run with
   `OPENBLAS/OMP/MKL_NUM_THREADS=1`.
5. **Checkpoint / resume:** 6 straight vs 3 + resume + 3 give byte-identical
   checkpoints, history and champion (sim and toy); resume refuses a changed
   config hash.
6. **Selection and novelty:** elites copied unchanged and re-scored; every
   child's parent is in the top T; identical descriptors get the lowest
   novelty; archive cap holds; a flat history triggers the boost at the
   expected generations (w 0.5 → 1.0, σ × 2); hall-of-fame cells hold the
   best benchmark score; top-3 benchmarking on the configured generations.
7. **Cloning (full config):** 200 presentations → 199,257 (observation,
   action) pairs; gate on 100 held-out presentations: HandBlue 2.74 kills,
   clone 2.37 → **86 %** (pass). With the draft's 30 epochs the clone reached
   only 77 % (2.12 kills, fail), so the default is now **60 epochs**. The
   clone data uses its own seed salts, disjoint from benchmark and test.
8. **Mutation statistics:** std(w′ − w)/σ′ = 0.998, mean/σ′ = 0.003; std of
   log(σ′/σ) = 0.198 (τ = 0.2); same seed → identical child, different child
   index → different child.
9. **Timing (measured, not extrapolated):** real 50 × 24 + 64 benchmark on 8
   workers: 198 s for generation 0 (inside 187 s ± 15 %), mean 226 s over 10
   generations (221 s on top-1 generations, 239–254 s on top-3 generations).
   That is about 21 % above the estimate: fights get longer as the
   population improves (fewer early Blue losses). An 8-hour night would be
   about 127 generations, not 192.

### Demo evolve (50 × 24, mixed start, 10 generations, 8 workers)

About 42 minutes of wall time: 37.7 minutes of generations and about
4 minutes of final checks and held-out test.

| gen | best fitness | mean fitness | gen best on benchmark | champion benchmark | clone-origin members |
|----:|------:|------:|------:|------:|---:|
| 0 | 134.6 | −110.4 | 103.5 | 103.5 | 10 |
| 1 | 194.5 | 5.5 | 138.9 | 138.9 | 19 |
| 2 | 219.1 | 79.5 | 163.4 | 163.4 | 34 |
| 3 | 244.7 | 107.2 | 185.6 | 185.6 | 43 |
| 4 | 251.4 | 134.3 | 270.9 | 270.9 | 43 |
| 5 | 271.2 | 165.7 | 270.9 | 270.9 | 44 |
| 6 | 271.0 | 161.9 | 286.6 | 286.6 | 50 |
| 7 | 349.5 | 183.4 | 355.3 | 355.3 | 50 |
| 8 | 360.9 | 175.3 | 322.0 | 355.3 | 50 |
| 9 | 441.9 | 178.1 | 426.8 | 426.8 | 50 |

| policy | benchmark (64): fitness / kills / losses | held-out test (256): fitness / kills / losses |
|---|---|---|
| HandBlue | 98.7 / 2.89 / 3.17 | 125.7 / 3.03 / 3.09 |
| unperturbed clone | 98.3 / 2.67 / 2.91 | 90.9 / 2.61 / 2.93 |
| champion (gen 9, clone lineage) | 426.8 / 3.77 / 0.59 | 415.6 / 3.62 / 0.53 |

- The champion comes from the clone lineage. Random lineages were extinct
  by generation 6, and every generation's best was clone-origin.
- The champion beats HandBlue on the held-out test (3.62 vs 3.03 kills,
  0.53 vs 3.09 losses), though only on the placeholder fitness. Its
  behaviour: radar always on, launch r/Rmax 0.61, mean altitude about
  13.6 km, closest approach about 16.9 km. So it fights high and at range.
  This may exploit the placeholder models (high-altitude missile kinematics,
  Red's reach) rather than show a real tactic. It needs a look in TacView
  before anyone trusts it.
- Hall of fame: 2 of 9 cells filled (r0_l0 and r0_l1, both radar-on). The
  radar-off cells are empty after 10 generations. No stagnation boost fired.
