# V4 three-arm comparison — proposed execution plan

Status: **awaiting user approval**. This document specifies the next experiment;
training has not started. The bootstrap, checkpoint-export, and split-safe
evaluation prerequisites are implemented; the end-to-end pilot and cost audit
remain. After approval, run that pilot before the main comparison. Do not
spend the main-run budget if a gate fails.

## Questions and arms

| Arm | Initial actor | Training | Question |
| --- | --- | --- | --- |
| A | Random V4 weights | Reward-only, match-start PPO | How well does RL work without a teacher? |
| B | Rules-teacher bootstrap | Same PPO configuration as A | Does teacher initialization help PPO? |
| C | Same bootstrap checkpoint as B | Outcome-search distillation | How does search distillation compare with PPO? |

The teacher is a potentially useful prior, not an optimal policy. Correct
sampling weights reproduce its behavior more faithfully; they cannot correct
bad teacher decisions. A versus B tests whether that prior helps. B versus C
compares downstream methods at a shared starting point. A versus C compares
the complete pipelines. Do not select the best random seed or omit failed runs.

Use three independent replicate seeds: **0, 1, 2**. For each seed, save the
random initial actor before bootstrap optimization; A starts from those exact
weights, and B/C share the resulting bootstrap actor. All arms use the same
V4 architecture, learner deck, public observation contract, ruleset, and source
revision. Keep heuristic opponent controllers fixed. Frozen learned opponents
and self-play are a later experiment.

## 1. Implement and verify the prerequisites

1. **Correct bootstrap sampling weights.** Keep sampling across full matches
   and retaining plentiful PLAY examples. Record every eligible teacher label
   before sampling. For match `e` and mode `m`, retain the original count
   `N[e,m]` and sampled count `n[e,m]`; attach weight `N[e,m] / n[e,m]` to each
   retained record. Apply these weights only to the PLAY/WAIT cross-entropy,
   replacing its current class balancing. Normalize weights once against the
   full retained dataset and average weighted losses in uniformly shuffled
   minibatches. Continue to normalize card/placement losses over PLAY records.
   This targets the natural decision distribution of the collected teacher
   trajectories; it does not prescribe a fixed action frequency.
2. **Preserve sampling support.** Predeclare the collection batch of complete
   matches and retain at least one record from every observed match/mode group.
   A nonempty group with zero retained records cannot be repaired with a
   weight. Treat 50/50 as a coverage preference, not a hard requirement that
   discards an entire group. Record any resulting departure from 1,024 samples.
   Start with three complete matches per bootstrap archetype, 1,440 decisions
   maximum per match, and at most 32 retained records per mode per match.
3. **Separate development and final evaluation in the runners.** Learning
   curves use training-catalog opponents with fresh development seeds and
   optional development frontier diagnostics. They reject reserved deck
   identities and select only from complete-match results. The separate
   `--final-evaluation` path accepts frozen actors, defaults to the reserved
   catalog, and performs no training or checkpoint selection. Keep the final
   path unused until the selected actor hashes are frozen.
4. **Audit deck separation.** Check bootstrap opponent decks as well as PPO
   and search decks against the ten reserved evaluation identities. If a
   bootstrap deck overlaps, replace that bootstrap opponent with a compatible
   training-catalog deck before sealing the protocol. Save the actual lists.
5. **Record cost and checkpoints consistently.** Record actual simulator
   decisions, completed/truncated matches, search branch decisions, optimizer
   examples/steps, worker CPU time, GPU time where used, hardware, and wall time.
   Distillation must export plain actors at the scheduled development points.
   Save the initial random actor explicitly; do not assume that equal seed
   numbers alone prove identical weights.

Required checks: weighted loss reconstructs a known unsampled example;
unequal episode lengths and missing modes are handled; full-match reservoirs
retain late PLAY/WAIT states; action labels remain legal; identical seeds are
deterministic; train/development/final overlap is rejected; checkpoint loading
and a real optimizer update succeed for every arm. Verify frozen B/C anchors
have identical hashes and A matches the bootstrap's pre-training actor.

The bootstrap now predeclares three complete matches per archetype, records
original and retained match/mode counts, uses dataset-normalized
inverse-retention weights for mode loss, preserves each observed group's
support, saves the initial random actor, and rejects overlap with reserved
evaluation decks. Distillation now exports hashed plain actors at the planned
development epochs. The curve runner now keeps development matches on the
training catalog and has a frozen-only final path; scripted frontier entries
require an explicit opt-in. An end-to-end comparison pilot, measured cost audit,
and source freeze remain prerequisites. Resolve
relevant test failures; record unrelated environment/dependency skips
separately. Commit the experiment source before the main run and use fresh
output directories.

## 2. Freeze the data and seed protocol

Use the existing 40-deck catalog: **30 training decks, ten final evaluation
decks**. The learner continues to use its fixed base Hog-cycle deck. Real
opponent decks do not imply human-quality opponent behavior.

For replicate `s` in `0,1,2`:

| Purpose | Seed allocation | Decks / count |
| --- | --- | --- |
| Training openings | pool seed `20261011 + 100*s` | 24 matches per archetype: 120 total, four per training deck |
| Search development collection | pool seed `20261012 + 100*s` | 12 matches per archetype: 60 total, two per training deck |
| Development complete matches | seeds `20263000..20263003` | Four fresh seeds for each of the 30 training decks |
| Final complete matches | seeds `20264000..20264019` | Twenty fresh seeds for each of the ten reserved decks |

Assert actual match identities and deck coverage, not just different numeric
seed labels. Development shares training deck identities but uses fresh match
seeds; final evaluation tests unseen complete deck identities. Match the
deck/seed pairs across all actors, using controlled actor/opponent RNG streams.

Both PPO arms receive the same **search-free match openings**. Search collects
trajectories from the corresponding bootstrap actor at those same openings.
PPO never consumes frontier states, search verdicts, or teacher targets.
Development trajectories and final matches are never training inputs.

Store under `outputs/v4/comparison_v1/seed_<s>/`: `random/`, `bootstrap/`,
`starts/`, `search_train/`, `search_dev/`, `ppo_random/`, `ppo_bootstrap/`,
`distill/`, and `development/`. Reserve a separate `final/` directory until
checkpoint selection is frozen. Record source, deck-catalog, checkpoint, and
artifact hashes in a top-level experiment manifest.

## 3. Run a bounded pilot, then freeze the main settings

Use a separate pilot seed **900**, train/development pool seeds **20269001**
and **20269002**, and no final decks. Run bootstrap collection and fitting, one actual PPO update for each
initialization, and search/distillation on one match per training archetype.
The pilot only checks legality, finite losses/gradients, checkpoint exports,
coverage, accepted-label yield, and runtime. Do not choose methods by pilot wins.

Require both bootstrap modes and usable later-match observations; correctly
weighted counts; valid exported actors; nonempty independent search corrections
in training and development; and clean evaluation pairing. A search pool with
too few corrections is a collection failure, not evidence against the method.
Use a predeclared main-run minimum of 100 training and 30 development
corrections per replicate, with every training archetype represented in both.
If this fails, stop and revise collection rather than weakening acceptance
criteria or examining final scores.

The initial fixed-work study uses these settings, which must be recorded
explicitly rather than inherited from changing CLI defaults:

| Component | Proposed settings |
| --- | --- |
| Bootstrap B/C | Full-match collection above; eight epochs, batch 64, learning rate `3e-4` |
| PPO A/B | 32 updates; eight lanes; 1,440 rollout decisions and episode cap; `training-source=match-start`, `bc-coef=0`, `reward-mode=tower-delta`; actor LR `3e-4`, critic LR `1e-4`, gamma `.99`, GAE lambda `.95`, one PPO epoch |
| Search C | 16 candidates/match; branch cap seven (current planner can emit at most six); horizon 128; eight discovery plus eight independent confirmation replicas; match cap 1,440; three workers with `--torch-threads 1` |
| Distillation C | `soft-kl` only; context 32; 20 epochs; batch 8; LR `1e-5`; retention weight 1; temperature `.25`; one optimizer seed `s` per replicate |
| Development points | PPO updates `0,1,2,4,8,16,32`; distillation epochs `0,1,2,4,8,16,20` |
| Evaluation | 1,440 decisions maximum; seeded stochastic sampling from each masked policy for all arms; no search at inference |

The runner's default distillation method/context/seed sweep must be overridden;
it would otherwise create extra variants and unequal selection opportunities.
Record all remaining search acceptance, retention, entry-cap, and optimizer
settings in the manifest. Derive explicit worker/device settings and a runtime
estimate from the pilot before launching the main jobs. Hardware and monetary
budget have not yet been supplied; do not infer authorization to rent compute.

## 4. Train and select using development data only

For each replicate, train A from the saved random actor and B from the
bootstrap actor using identical PPO settings and openings. Generate and audit
independently confirmed search labels, then train C from B's exact bootstrap
checkpoint. Keep bootstrap generation/fitting costs for both B and C, even if
the physical artifact is shared. Include all search and label-validation costs
in C. Include failed attempts and retries in cost reporting.

Evaluate the seven scheduled checkpoints of each arm on the same development
complete matches. Select the checkpoint with the highest mean match score
over the 30 development decks; break ties by lower cumulative training cost,
then earlier checkpoint. Include point zero so regressions can select the
initial actor. Distillation label losses and frontier regret are diagnostics,
not an extra route to selecting on final data. Keep all three replicate seeds.

Freeze selected checkpoint hashes and the selection report before any final
evaluation. Keep the final endpoint checkpoint too, so the report can show
whether early selection hid a later regression.

## 5. Run final evaluation once and report uncertainty

Evaluate the nine selected actors (three arms by three replicates) on the same
200 deck/seed matches each: **1,800 complete-match attempts**. This is an initial
comparison capable of finding substantial differences, not a guarantee of
resolving small gains. Do not increase the final sample after seeing scores.

Primary outcome: mean match score, with win=1, draw=.5, loss=0, equal weight per
reserved deck. Report paired A/B, B/C, A/C differences, results for every
replicate and deck, and 95% uncertainty intervals. Respect pairing and cluster
by training replicate and match seed when estimating uncertainty; do not treat
all 1,800 games as independent training runs. Three training seeds provide
limited evidence about training variability. Treat the three contrasts as
exploratory unless a multiple-comparison rule is declared before evaluation.

Report truncations separately; never turn them into draws or silently discard
them. Show score bounds for unresolved matches and make no winner claim if
censoring can reverse the ranking. Invalid actions, crashes, or inconsistent
pairing trigger investigation rather than a clean performance conclusion.
Report crown/HP outcomes and action counts as secondary diagnostics, without
assuming an action-rate target or inventing a reliable "premature play" label.

Shared sealed frontier regret may be reported as a secondary diagnostic after
selection, using the same frozen states for all arms and recording the policy
that generated those states. It is a search-selected state distribution and
must not replace the complete-match comparison.

## 6. Interpret quality and cost separately

This first study fixes **work settings**, not equal total compute. A/B have the
same PPO update budget, but B additionally pays for bootstrap. C has different
collection and fitting costs. Report development quality against cumulative
measured cost and final quality alongside total end-to-end cost. Report
evaluation cost separately and shared costs both physically paid and charged
to each complete pipeline. Equal updates or epochs across methods are not a
fair efficiency comparison.

Do not call the higher-scoring arm more compute-efficient unless its cost
supports that claim. If curves leave the choice unclear, the next experiment
must predeclare matched total budgets using pilot measurements; do not choose
budgets retrospectively to favor an arm. An inconclusive comparison is a valid
result. Simulator-only results also do not change `training_ready: false` or
establish performance against real players.

Approval covers this protocol and its prerequisite work. The deliverables are
the corrected bootstrap weighting, split-safe evaluation, test evidence,
versioned run manifest, costed pilot report, three-arm training artifacts, and
paired final report. Main runs remain gated on prerequisite/pilot success and
an identified compute environment within the user's authorized budget.

## Runtime estimate from local probes (2026-09-28)

These timing probes were requested separately; they do not constitute approval
of the main experiment. Measurements used the local i9-13900H CPU, one Torch
thread, the compact V4 model, training-catalog opponents, and a smoke-trained
actor. No reserved-deck results were examined. Scripts and raw measurements are
local artifacts under `outputs/v4/comparison_runtime_probe_20260928/`.

| Measured workload | Wall time |
| --- | --- |
| PPO collection, eight lanes × 128 decisions | 4.46 seconds |
| PPO optimizer update on that rollout | 5.35 seconds |
| Search, four distinct branches × eight replicas × 128 decisions, early state | 23.69 seconds |
| Same search workload, later state, discovery / confirmation | 57.07 / 60.72 seconds |
| Five complete stochastic evaluation matches, one per training archetype | 5.62–20.73 seconds each; mean 10.39 seconds |
| Synthetic distillation optimizer probe, context 32, batch eight, eight updates plus small validation | 5.82 seconds |
| Earlier full bootstrap collection, 1,024 records from 18 matches | 420 seconds |

The complete proposal has 8,640 candidate evaluations if every match supplies
16 candidates. The measured four-branch search costs extrapolate to roughly
**114–283 single-worker hours** for discovery plus confirmation alone.
More branches or denser late-game boards can raise this. Confirmation currently
reruns all branches, even though its selected target is frozen. Source replay,
prefix-cache generation, and independent distillation outcome checks add work.

Development evaluation comprises 7,560 matches, final evaluation another
1,800: **9,360 matches**, or about **27 hours** at the measured average before
allowing for stronger actors and longer/denser games. The two PPO arms across
three seeds extrapolate to roughly six hours from short opening rollouts;
allow 6–15 hours for planning. Full 1,440-step optimizer memory/throughput has
not been benchmarked, so linear scaling is provisional. Distillation optimizer
timing used synthetic cached inputs; it excludes cache construction and paired
simulator validation.

The following pre-optimization estimate is superseded by the measurements at
the end of this section. The earlier budget was **about 5–10 days of elapsed
local compute** for the initial proposed
comparison, assuming three search workers with imperfect scaling, sequential
evaluation/training, and allowances for cache/validation work. Parallel
throughput itself was not measured. A serial execution is roughly 1–2 weeks,
potentially longer at the expensive end of the samples. This is a broad
planning estimate, not a deadline or an estimate of time to convergence.
It excludes prerequisite implementation/debugging and any later extension
beyond 32 PPO updates. No GPU/cloud speedup is inferred from these CPU tests.

### Local RTX 2050 follow-up

The user requested a GPU comparison. CUDA was inaccessible inside the sandbox
but usable outside it on the local RTX 2050 (4 GiB). One Torch CPU thread,
identical small workloads, warm-up, and CUDA synchronization gave:

| Workload | CPU | GPU | Speedup |
| --- | ---: | ---: | ---: |
| PPO collection, eight lanes × 128 steps (two-run mean) | 4.58 s | 3.59 s | 1.27× |
| PPO update on that rollout (two-run mean) | 6.06 s | 1.11 s | 5.47× |
| Search, one replica, four branches × 128 steps, including state replay | 10.19 s | 10.11 s | 1.01× |
| Synthetic context-32 distillation, eight updates plus small validation | 4.74 s | 0.84 s | 5.62× |

Unmodified CUDA PPO failed with `cudnn RNN backward can only be called in
training mode`: the updater deliberately uses evaluation mode. The successful
GPU timings disable cuDNN only around PPO updates in the benchmark script;
this is not yet a repository fix. Add a CUDA regression test and resolve this
before GPU PPO training. Distillation required no such workaround.

PPO collection plus update was about 2.26× faster overall in this short probe.
Peak allocated GPU memory reached about 1.1 GiB at 128 steps; the planned
1,440-step update has not been tested and must not be assumed to fit 4 GiB.
Do not silently shorten rollouts to fit memory, since that changes the study.
The frontier-generation CLI currently hardcodes CPU despite its underlying
evaluation functions accepting a device. The measured search workload gains
essentially nothing from GPU, so retain CPU search for now. GPU training may
save hours, but does not justify reducing the overall 5–10-day planning range
when search and evaluation dominate. Full-match GPU evaluation and sustained
parallel throughput have not been measured. Raw GPU results and both benchmark
scripts are saved beside the CPU probes.

### Result-preserving replay reuse

The frontier generator now replays each source match once and caches the exact
simulator state and actor memory at selected candidate decisions. Discovery,
independent confirmation, and accepted-entry materialization reuse that cache.
It still evaluates every original branch and replica with the same seeds and
continuation horizon. A 512-decision, 16-candidate local probe reduced prefix
replay from 4,246 to 506 decisions, taking **2.93 instead of 21.35 seconds**
for the replay portion (7.29×). Exact state hashes, recurrent tensors,
complete discovery/confirmation verdicts, and serialized entry state matched
the original path in regression checks. Continuation rollouts remain the
dominant cost, so do not apply 7.29× to the whole search estimate. The prior
5–10-day planning range remains conservative until full-match throughput is
measured after this change.

### Search-stage profile after replay reuse

Fresh CPU measurements on 2026-09-28 used one Torch thread, the same 512-step
aggressive-pressure source match, 16 selected candidates, four actual branches,
eight discovery plus eight confirmation replicas, and a 128-decision horizon.
These measurements precede the navigation optimization below.
One source match took 5.70 seconds to collect; caching all 16 candidate states
took 2.90 seconds. Two representative candidate evaluations took:

| Candidate | Discovery | Confirmation | Total | Simulator stepping | Actor sampling | Observation construction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early (`t=90`) | 26.71 s | 26.23 s | 52.94 s | 31.48 s (59%) | 15.12 s (29%) | 3.96 s (7%) |
| Late (`t=487`) | 61.17 s | 61.13 s | 122.30 s | 94.86 s (78%) | 16.52 s (14%) | 5.74 s (5%) |

Both evaluations executed 64 branch rollouts and 8,192 continuation decisions.
The later board cost more mainly because simulator stepping became slower. A
single late branch profiled with `cProfile` spent 3.17 of 4.08 seconds inside
the core simulator step; movement and route planning dominated it, including
404 uncached route plans and roughly 170,000 walkability checks. Profiler
overhead affects absolute times, so use the unprofiled component timings above
for wall-clock estimates. These are small representative samples, not a
measured distribution across all decks and matches. The new full candidate
timings overlap earlier pre-cache measurements, so a total-search speedup has
not yet been established; the reliably measured saving is prefix replay.

### Simulator stepping optimization

The route planner now computes fixed waypoint edges and goal visibility only
when search reaches the corresponding waypoint. It avoids point-by-point river
checks when all integer samples fit inside one bridge, and moving ground units
reuse valid routes before rebuilding obstacle lists. Collision separation skips
square roots for nonoverlapping troop pairs. Target acquisition calculates
in-flight projectile reservations once per physics phase instead of repeating
the same scan for every attacker.

Against the original implementation, a cold-cache probe of four distinct
128-decision branches at each candidate state measured:

| Candidate | Original simulator stepping | Optimized stepping | Reduction |
| --- | ---: | ---: | ---: |
| Early (`t=90`) | 2.14 s | 1.37 s | 36% |
| Late (`t=487`) | 6.09 s | 2.86 s | 53% |
| Air-beatdown (`t=314`) | 2.64 s | 1.81 s | 31% |

All 1,536 post-step simulator state hashes and branch results matched the
original code. The route planner also matched 1,000 randomized route queries
against the prior eager-graph implementation and 12,000 randomized visibility
queries against the original segment checker. These probes do not establish a
full-search speedup across every deck and board state. Repeating one late
branch with warm caches reduced its stepping time from 0.705 to 0.541 seconds;
warm-cache gains depend on how often the search revisits the same layout.

### Updated comparison runtime estimate after stepping optimization

A fresh full candidate evaluation on the optimized code (four actual branches,
eight discovery and eight confirmation replicas, horizon 128) took **35.14 s**
at an early state and **67.54 s** at a later state, versus 52.94 s and 122.30 s
before the stepping changes. This includes actor sampling, observations, forks,
and simulator stepping. The source match took 4.08 s and caching its 16
candidate states took 2.16 s; neither is repeated per candidate.

The protocol has 540 source matches across three seeds, or at most 8,640
candidate evaluations at 16 per match. At four branches, the measured
candidate range extrapolates to **84–162 single-worker search hours**. Of the
16 candidates in the measured source match, 11 had four branches and five had
five; applying that mix gives roughly **91–175 hours**. The current branch
builder can emit at most six branches, despite the configured cap of seven;
six throughout would instead be about **126–243 hours**. Full-length matches
may have denser states than the sampled 512-decision source match.

With three CPU search workers, budget **about 3–8 days of local elapsed time**
for the complete three-arm, three-seed comparison, including search, PPO,
distillation, and development/final evaluation. Ideal three-worker search
scaling would turn the 91–243 serial search hours into 30–81 elapsed hours;
actual scaling has not been measured. The previously measured 9,360-match
evaluation workload was about 27 hours before the simulator speedup, and the
two PPO arms were provisionally 6–15 hours. Those stages, cache construction,
validation, and imperfect parallel scaling account for the rest of the range.
Serial search would likely push the complete comparison toward **5–12 days**.
This remains a planning range, not a measured end-to-end run or a time to
convergence; the prerequisite pilot must still confirm sustained throughput.

### Next exact-protocol throughput checks

After the stepping change, five stochastic complete matches on the same
training-deck sample took 11.34, 7.08, 6.77, 6.68, and 4.85 seconds: **7.34 s
mean**, versus 10.39 s before the change. Scaling that small sample to the
protocol's 9,360 development/final matches gives **about 19 serial hours**,
replacing the earlier 27-hour point estimate. Three of those matches took
25.2 s sequentially and 14.1 s when run in separate one-thread processes;
their match results were identical. A persistent, ordered process pool should
be tested on the full evaluation path before claiming this 1.8× group speedup
at study scale.

The frontier-generation CLI already schedules independent source matches in
three processes, but does not pin PyTorch's intra-op threads in each worker.
In one late-branch probe, one thread took 1.74–1.78 s versus 2.03–2.16 s with
eight threads; all 256 step-state hashes and branch values matched. The pilot
should sweep one, two, and three or more workers with one or two Torch threads
per worker, checking exact verdict hashes and sustained matches/hour. Avoid
CPU oversubscription before raising the worker count.

These throughput changes preserve the declared matches, branches, replicas,
seeds, horizons, and checkpoint-selection rule. The current confirmation path
reruns all discovery branches; reducing it to only the selected action would
change retained confirmation diagnostics and the merge contract, so it is not
included in these estimates. Sealed-choice diagnostics can separately skip
duplicate branch rollouts when two policies choose the same action and cache
fixed champion rollouts across checkpoints, subject to exact output tests.

### Search optimization after the throughput review

Source collection, candidate-prefix replay, and active branch continuations
now advance simulator physics without constructing two V2 observations that
their callers discard. The actor still builds its public observation at each
decision. Prefix replay catches up the second viewer's public memory at every
retained candidate state. A 256-decision source match produced byte-equivalent
JSON in 1.55 instead of 1.88 seconds; replay of eight retained states took
1.00 instead of 1.17 seconds with matching state hashes, actor hidden tensors,
and both viewer memories.

An A/B run in the same process compared complete discovery plus confirmation
for four actual branches, eight replicas per stage, and horizon 128. The early
candidate fell from 46.31 to 38.26 seconds (17%); the later candidate fell
from 76.78 to 55.28 seconds (28%). Their complete merged verdicts were equal.
Applying only those relative savings to the earlier 91–175 serial-hour search
range gives roughly **75–126 serial hours** for the observed branch mix. The
recent absolute measurements were slower than the earlier machine baseline,
which widens a prudent planning range to **about 75–145 serial CPU hours**.
This remains an extrapolation from a 512-decision source match; full matches
and denser boards need a sustained pilot.

The search CLI now defaults to one PyTorch CPU thread per process. A two-replica
candidate verdict was identical with one and eight threads. In a concurrent
three-process probe, the one-thread group finished in 17.2 seconds; the
eight-thread group remained unfinished after 150 seconds and was stopped.
That shows severe oversubscription in this environment, but it is not a
completed throughput ratio for full source matches. Keep the thread count
explicit in the run manifest and benchmark sustained matches/hour in the pilot.

This comparison uses rule-based search opponents. The optional learned V4
opponent path now isolates cold checkpoint loading from the learner's Torch
RNG and seeds branch sampling after opponent construction. Cold and warm
cache paths have a regression check; learned opponents remain outside this
comparison protocol.

Terrain visibility now checks the exact continuous bridge interval with
integer geometry, so it no longer depends on finite samples. The non-jumping
Knight path from `(5000, 14750)` to `(2500, 17150)` is rejected; Hog Rider
can jump the river and can now retarget while airborne outside a bridge.
Navigation visibility is symmetric under endpoint reversal, and movement rounding is
mirror-symmetric. Frontier reuse
rejects source-code and match-config mismatches, and distillation prefix
replay verifies recorded state hashes, including rejected entries. Generate
fresh source matches and verdicts under these physics changes; do not feed
pre-fix records into a new distillation run. The 75–145-hour range above is
provisional until a new sustained throughput pilot measures this revision.
