# V4 training and evaluation

V4 has a shared evaluation protocol for three distinct training arms: pure
reward-only PPO from a random actor, reward-only PPO from the bootstrap actor,
and search distillation from that same bootstrap actor. The bootstrap trains a
compatible actor from public simulator observations; frontier generation mines
difficult states for distillation. Frontier-initialized PPO remains a separate
experiment and must not be reported as raw RL. Historical Stage-1/Stage-2
generators, timing curricula, and serialized search-dataset builders are
retired.

The next experiment is specified in the
[three-arm comparison plan](v4_comparison_plan.md), **awaiting user approval**.
It defines the prerequisites, seeds, work budgets, development-only selection,
and final evaluation. The commands below document existing runner interfaces;
they are not an approved launch sequence for that comparison. In particular,
the predeclared bootstrap match batch and strict development/final separation
still need an end-to-end pilot before the planned main run.

## Teacher bootstrap for the bootstrap-initialized arms

Run the bootstrap with the project environment:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_bootstrap.py \
  --states 1024 --epochs 8 --out outputs/v4/bootstrap_seed0
```

It writes a plain `actor.pt` state dict, the exact pre-training
`random_actor.pt` state dict, and a `manifest.json`. The runner
requires a clean tracked checkout, seals the Git revision, tracked worktree
hash, engine version, ruleset hash, and observation contract into the
manifest, and checks that the source does not change during the run. Use
`--allow-dirty` only for a development smoke run; the resulting manifest marks
that it came from a dirty tracked worktree. The manifest also records the
simulator-data generator version, model configuration, seed, sample-family
counts, loss curve, and actor hash. The actor is compatible with
`simulator.rl.v4.v4_ppo.load_frozen_anchor` and is the required input for the
frontier stage. The bootstrap uses only the public V4 observation, a
deterministic opponent pool, and the retained rules teacher. It does not read
the opponent's private hand or authoritative elixir.
The runner predeclares three complete matches for each of the six opponent
archetypes (`--episodes-per-archetype 3`); smoke uses one and caps each mode
at four records per match. It retains at least
one record from each observed match/mode group, so `--states` is a preferred
sample count rather than an exact fixed-batch output size.
It rolls each match to termination or the 1,440-decision cap (360 simulated
seconds), then retains a deterministic uniform reservoir per action mode.
The default `--samples-per-episode 64` permits at most 32 records per mode
from one match. Reaching the preferred count never truncates a trajectory.
The main fixed batch requires each match to terminate; smoke may stop at its
short decision cap and records that truncation. The batch fails if it observes
no PLAY or no WAIT labels; inspect mode
coverage and collect more matches before using the actor. The bootstrap uses six
scripted opponent archetypes; the real-deck catalog is used downstream.
The manifest records collection settings and per-archetype match, mode,
phase, and elapsed-time coverage alongside the loss history. Check these
before using its actor for collection. Balanced sampling deliberately changes
the observed PLAY/WAIT frequency; it is not an estimate of natural play rate.
For each match/mode group, the collector records eligible and retained counts
and weights each retained mode label by their ratio. It normalizes weights
once across the dataset; minibatches do not rebalance PLAY/WAIT. This restores
the collected teacher's natural decision frequencies in the mode objective
while retaining enough PLAY examples for card and placement training. The
manifest records these counts, both actor hashes, and the bootstrap opponent
decks; collection rejects a deck identical to a reserved final-evaluation
deck. Neither corrected weights nor teacher agreement establish optimal
actions. The random-initialized PPO arm must load `random_actor.pt` directly.

For a dependency and wiring check:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_bootstrap.py \
  --smoke --allow-dirty --out /tmp/v4-bootstrap-smoke
```

Smoke output is an integration check and makes no quality claim.

## Frontier and search training

Use the freshly produced actor explicitly when collecting frontier states:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_frontier_generation.py \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --out outputs/v4/frontier_seed0
```

The frontier runner records the checkpoint hash in its manifest. Generate a
new output directory after changing simulator, observation, model, or teacher
code. The train split is used for fitting and the sealed split is reserved for
final evaluation. Check `summary.json` before training: if either split has
too few accepted frontiers, collect more matches rather than treating an
empty or narrow pool as a performance result.
The generator replays each source match once to cache its selected decision
states and recurrent actor memory. Discovery, independent confirmation, and
entry materialization reuse those verified states; branch choices, replica
seeds, continuation lengths, and scoring are unchanged. This only removes
repeated prefix work, so full continuation rollouts still dominate runtime.

Search distillation also requires a separate development-validation frontier
run. Use different pool seeds so its match states are disjoint from the
training run; its `frontier_train.pt` supplies validation records. Its sealed
split is not a substitute for development validation:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_frontier_generation.py \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --train-pool-seed 20261003 --sealed-pool-seed 20261004 \
  --out outputs/v4/frontier_validation
```

Inspect the frontier verdicts with the teacher audit before using them for
search distillation. This audit is diagnostic; the distillation runner reads
the generator's verdict files, not the audit report. The audit's
`--first-count` must not exceed the number of accepted training entries
(the default is 300):

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_teacher_audit.py \
  --source outputs/v4/frontier_seed0:train \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --first-count 100 \
  --out outputs/v4/teacher_audit_seed0
```

The active search-distillation runner consumes disjoint frontier match records
and their generator verdicts:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_search_distillation.py \
  --dataset-root outputs/v4/frontier_seed0 \
  --validation-root outputs/v4/frontier_validation \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --out outputs/v4/search_distillation_seed0
```

Search distillation is an outcome-search experiment. Its corrections require
independent paired simulator replicas; a single replica cannot certify a
change. The package under `simulator/rl/v4/search_distillation/` owns cached
prefix records, target construction, validation, and resumable training. Each
completed run exports a plain actor state dict under
`outputs/v4/search_distillation_seed0/actors/<label>.pt`; the similarly named
file under `checkpoints/` also contains optimizer state and cannot be passed
directly to the V4 evaluator. It also exports plain actors at epochs
`0, 1, 2, 4, 8, 16, 20` under `actors/<label>_epochN.pt`, with hashes in the
run manifest. A completed run is reused only when its scheduled actors still
match those hashes. Freeze method/context settings using development
data before looking at sealed outcomes; report all predeclared replicate seeds
rather than selecting the best seed. The comparison plan fixes one method and
context in advance.

## Raw RL versus search distillation

Create search-free match-opening entries for PPO. The default pool seed and
match count reproduce the training match openings used by the frontier
generator, without its policy trajectories, search verdicts, or teacher
targets:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_raw_ppo_starts.py \
  --pool-seed 20261001 --matches-per-archetype 20 \
  --out outputs/v4/raw_match_starts_seed0.pt
```

The `pinned` profile above is the controlled six-archetype baseline. For the
main deck-distribution experiment, use `--profile real` for PPO and
`--train-profile real` for frontier generation. Both use the same versioned
[real deck catalog](../simulator/rl/v4/real_decks.json), seeded match openings,
and five training archetypes. The catalog currently contains 30 complete
training decks and ten disjoint evaluation decks from RoyaleAPI and Deck Shop
(six training and two evaluation decks per archetype),
captured on 2026-09-28. Every source URL encodes the exact eight card forms.
The loader rejects card-list mismatches, duplicates, unsupported cards, and
Evo/Hero variants; it never converts special forms to base cards. The source
pages include both tracked RoyaleAPI decks and published Deck Shop deck pages,
so these are source-linked real deck lists rather than a claim that every one
is currently popular or strong. RoyaleAPI's displayed usage changes over time;
the catalog pins deck identity, not a live popularity statistic.

Use the same train pool seed and match count for the two methods. The real
frontier runner can also generate a sealed split from the catalog's ten
evaluation decks, but those records must stay out of development curves and
checkpoint selection. The learning-curve runner now defaults to all 30
training-deck identities with fresh evaluation seeds. Its explicit
`--final-evaluation` path uses the ten reserved decks only after checkpoints
are frozen. Changing the catalog requires new output directories and
regenerated raw starts/frontiers.
The current raw loader requires `raw-ppo-starts-v3` and verifies the catalog
hash and train-deck identities in real-profile entries.

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_frontier_generation.py \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --train-profile real --train-pool-seed 20261011 \
  --out outputs/v4/frontier_real_seed0
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_raw_ppo_starts.py \
  --profile real --pool-seed 20261011 --matches-per-archetype 20 \
  --out outputs/v4/real_match_starts_seed0.pt
```

For the real-deck comparison, also generate development-validation frontiers
with `--train-profile real` and different pool seeds. Pass the real frontier
run as `--dataset-root`, the real validation run as `--validation-root`, and
`outputs/v4/real_match_starts_seed0.pt` as PPO `--train-entries`. Keep PPO
`--training-source match-start --bc-coef 0` and select the distillation
checkpoint on the separate real development-validation run before final
comparison. Optional frontier diagnostics in a development curve must come
from the development run's `frontier_train.pt`, never `frontier_sealed.pt`.

The optional `mixed` profile fills archetype decks with random legal cards;
some combinations are strategically incoherent. Use it for simulator stress
tests and action-coverage checks. `pinned`, `mixed`, and `real` are distinct
experiments. Keep the profile and opponent list fixed within each comparison.

After the controlled comparison, add frozen V4 actors to a real-deck PPO pool
from previous runs. Each `--policy-opponent` adds one mirror-match cohort of
the configured match count; repeat it for distinct older checkpoints. The
runner records an absolute path and SHA-256 for each actor, verifies the hash
when loading it, and keeps the actor frozen for the whole match. Store these
checkpoints in stable local paths throughout training:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_raw_ppo_starts.py \
  --pool-seed 20261011 --matches-per-archetype 20 --profile real \
  --policy-opponent outputs/v4/search_distillation_seed0/actors/context32_soft-kl_seed0.pt \
  --out outputs/v4/real_with_policy_match_starts_seed0.pt
```

The checkpoint path above is illustrative: use a plain V4 actor state dict
that actually exists, such as an exported search-distillation actor. A saved
PPO optimizer wrapper is not a valid opponent checkpoint. Use the new starts
with `--training-source match-start --bc-coef 0`. Freeze the opponent list for
each comparison run, then add a new actor only between runs. Frozen V4 actors
currently play the learner's deck in mirror matches, while scripted opponents
cover the other decks.

First run reward-only PPO from the bootstrap actor. The example comparison
checkpoint below is illustrative; replace it with the actor selected from the
distillation development-validation reports. `--training-source match-start`
rejects search-labelled entries, and `--bc-coef 0` removes anchor imitation.
Use a full-match rollout and episode cap so updates include late decisions:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_learning_curve.py \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --train-entries outputs/v4/raw_match_starts_seed0.pt \
  --training-source match-start --bc-coef 0 \
  --rollout-decisions 1200 --episode-cap 1200 --rollout-lanes 8 \
  --eval-seed-start 20263000 --eval-matches 4 --eval-max-decisions 1440 \
  --compare-checkpoint outputs/v4/search_distillation_seed0/actors/context32_soft-kl_seed0.pt \
  --updates 0,1,2,4,8 \
  --out outputs/v4/bootstrap_raw_ppo_vs_distill_seed0
```

For the controlled random-initialized RL baseline, use the exact pre-training
actor saved by the bootstrap run in an otherwise identical match-start PPO
run. A newly generated actor with the same numeric seed is not a proof of
matching initial weights:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_learning_curve.py \
  --checkpoint outputs/v4/bootstrap_seed0/random_actor.pt \
  --train-entries outputs/v4/raw_match_starts_seed0.pt \
  --training-source match-start --bc-coef 0 \
  --rollout-decisions 1200 --episode-cap 1200 --rollout-lanes 8 \
  --eval-seed-start 20263000 --eval-matches 4 --eval-max-decisions 1440 \
  --compare-checkpoint outputs/v4/search_distillation_seed0/actors/context32_soft-kl_seed0.pt \
  --updates 0,1,2,4,8 \
  --out outputs/v4/random_raw_ppo_vs_distill_seed0
```

The `--compare-checkpoint` actor and every PPO curve point use the same
development opponents and fresh evaluation seeds within each report.
The commands pin four development seeds for each training deck; the runner's
general-purpose default of 24 seeds would be substantially more expensive.
Pin the opponent set with `--opponents-json` for a larger controlled study. Compare
paired complete-match results, truncations, and per-opponent outcomes in each
`learning_curve.json`. Development frontier regret is an optional secondary
diagnostic if you supply disjoint development entries. The
`comparison_checkpoints[].versus_final_curve_point.full_match` field directly
pairs the selected distillation actor with the final raw PPO point by opponent
and seed; a positive score delta favors distillation. The bootstrap-start comparison
isolates the training method; the random-start arm measures the value of the
bootstrap as well. Report PPO environment decisions, distillation optimizer
examples, counterfactual search replicas/horizon, and wall time alongside
quality. Do not draw a performance conclusion from an empty or heavily
truncated set of paired matches. Equal update counts alone are not an equal
compute budget.

The separate final-evaluation path uses the catalog's ten reserved real decks.
Evaluation still shares heuristic controllers with training, so it
measures transfer to unseen complete decks under known controller styles.
It does not establish robustness against learned adversaries or real players.
Keep this evaluation pool fixed
while comparing actors, and report per-opponent results so weakness against
one style is visible. A future learned-opponent pool needs a player-one V4
actor pool with diverse decks; the current frozen V4 opponent adapter supports
mirror matches and maintains player-one public memory and recurrent state per
episode. The `--opponents-json` evaluation argument can include pinned
`{"kind":"v4-checkpoint","archetype":"name","checkpoint":"/absolute/path/actor.pt","sha256":"...","deck":[...]}`
specs, but their deck must match the actor's trained hand vocabulary. For the
current V4 actor, use the learner deck. Keep any learned evaluation opponents
out of the training starts and predeclare their checkpoints before examining
final scores.

The existing frontier-initialized PPO curve is a separate search-conditioned
experiment. Run it only after training frontier entries exist; any optional
frontier diagnostic entries must come from a disjoint development run:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_learning_curve.py \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --train-entries outputs/v4/frontier_seed0/frontier_train.pt \
  --allow-scripted-frontier-entries \
  --updates 0,1,2,4,8 \
  --out outputs/v4/frontier_ppo_curve_seed0
```

The opt-in above is for the separate pinned/scripted experiment. Omit it in
the controlled real-deck comparison, where training and development entries
must resolve to the pinned training catalog.

The runner evaluates update zero before applying PPO. A match reaching the
full-match decision cap is recorded as truncated rather than as a draw. No
checkpoint is promoted automatically.

The runtime executes one simulator decision per WAIT. Therefore WAIT duration
is retained in the supervised bootstrap target but is excluded from the PPO
action distribution and entropy. Public history is synchronized through
`PublicStateEstimator` before each V4 observation.

## Contracts and package boundaries

| Package | Responsibility |
| --- | --- |
| `v4/bootstrap.py` | Real-simulator public-state collection and bootstrap batches |
| `v4/hand_features.py` | The V4 learner-deck hand-token contract |
| `v4/supervised.py` | Finite legality-masked bootstrap loss |
| `v4/raw_ppo.py` | Search-free match-opening entries for reward-only PPO |
| `v4/real_decks.py` and `real_decks.json` | Validated source-linked training/evaluation deck catalog |
| `v4/policy_opponent.py` | Frozen V4 mirror opponents with player-one public memory |
| `v4/simulator_teacher.py` | Small public-state rules labeler used for bootstrap and audit |
| `v4/model_v4.py` | Recurrent actor, action heads, and legality masks |
| `v4/v4_ppo/` | Checkpoint loading, rollouts, updates, diagnostics, and value helpers |
| `v4/frontier_ppo/` | Frontier-entry collection and paired evaluation |
| `v4/search_distillation/` | Audited search targets and resumable distillation |
| `foundation/opponent_pool.py` | Deterministic opponent decks and controllers |

The simulator remains the authority for action legality and state transitions.
The actor receives public observations; opponent controllers receive
authoritative simulator state. Shared feature preparation lives in the V4
runtime so live rollouts, frontier evaluation, bootstrap data, and PPO use the
same tensor contract.

## Reproducibility and validation

The bootstrap seed controls episode selection, opponent sampling, and training
batch order. Its manifest and actor hash identify the exact starting point for
frontier generation. Do not reuse frontier outputs after changing source
identity or observation semantics.

Focused V4 checks can be run with:

```bash
PYTHONPATH=.:src outputs/venv/bin/pytest \
  simulator/tests/test_search_distillation.py \
  simulator/tests/test_v4_ppo_adapter.py \
  simulator/tests/test_frontier.py \
  simulator/tests/test_frontier_ppo.py \
  simulator/tests/test_public_state_estimator_v4.py
```

The broad project test command remains:

```bash
pytest --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py
```

Generated checkpoints, frontier datasets, caches, match replays, and reports
under `outputs/` are local artifacts. They are not part of a clean source
checkout and are never silently treated as a promoted model.
