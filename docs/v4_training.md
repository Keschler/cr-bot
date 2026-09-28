# V4 training and evaluation

V4 has one from-scratch path and one evaluation path. The bootstrap trains a
compatible actor from public observations collected in the deterministic
simulator. Frontier generation then mines difficult states with that actor;
teacher audit, search distillation, and PPO are separate experiments that can
be evaluated independently. Historical Stage-1/Stage-2 generators, timing
curricula, and serialized search-dataset builders are retired.

## Start from random initialization

Run the bootstrap with the project environment:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_bootstrap.py \
  --states 1024 --epochs 8 --out outputs/v4/bootstrap_seed0
```

It writes a plain `actor.pt` state dict and a `manifest.json`. The runner
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
Collection allocates states evenly across the six opponent archetypes; the
supervised mode loss balances PLAY and WAIT classes because ordinary matches
contain many more WAIT decisions. Check the manifest's target counts and loss
history before using its actor for collection.

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

Audit accepted frontier labels before using them for search distillation:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_teacher_audit.py \
  --source outputs/v4/frontier_seed0:train \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --out outputs/v4/teacher_audit_seed0
```

The active search-distillation runner consumes frontier match records and
their audited verdicts:

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
prefix records, target construction, validation, and resumable training.

## PPO evaluation

Run a controlled PPO curve only after train and sealed frontier entries exist:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_learning_curve.py \
  --checkpoint outputs/v4/bootstrap_seed0/actor.pt \
  --train-entries outputs/v4/frontier_seed0/frontier_train.pt \
  --sealed-entries outputs/v4/frontier_seed0/frontier_sealed.pt \
  --updates 0,1,2,4,8 \
  --out outputs/v4/learning_curve_seed0
```

The runner evaluates update zero before applying PPO and uses the same sealed
entries, opponent specifications, and evaluation seeds at every point. A
match reaching the full-match decision cap is recorded as truncated rather
than as a draw. No checkpoint is promoted automatically.

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
