# V4 training and evaluation

## Active V4 path

The current training path starts with a compatible V4 checkpoint, collects
on-policy frontier states with `run_v4_frontier_generation.py`, checks search
labels with `run_v4_teacher_audit.py`, and trains search-derived targets with
`run_v4_search_distillation.py`. `run_v4_learning_curve.py` is the separate
controlled PPO comparison. The budgeted CPU chain is
`run_v4_budget_guard.py` → `run_v4_overnight_cpu.sh`.

`run_v4_stage2_proof.py` remains the bootstrap/reproduction path for the
historical Stage-2 checkpoint. It still uses `run_v4_timing_proof.py` for
timing metrics and the simulator-generated timing curriculum; neither is part
of the new on-policy search-distillation loop. The one-off Stage-1
distillation-proof runner has been removed. Earlier public-actor checkpoints
remain loadable by the live-controller inference reader; the retired V1/V3
training package and simulator CLI command have been removed.

## Checkpoint and dataset status (September 2026)

There is no promoted V4 checkpoint from the expanded experiments. The local
`outputs/v4/stage2_12k_v2_e12_s12.pt` is the restored reference checkpoint
used for the comparisons below; it is an ignored local artifact, not a model
shipped with the repository. The most recently trained local V4 weights are
`outputs/v4/targeted_fit_20260920/actor.pt`, a diagnostic fit of 16 selected
search actions. It fitted 15/16 training targets but matched 0/28 sealed
targets and did not improve held-out action outcomes overall. Do not promote
that checkpoint as the new policy.

The expanded PPO comparison used 107 training frontier states, 28 sealed
states, and 48 full matches per curve. The reference actor scored 11-36-1;
neither the terminal-reward nor tower-delta eight-update curve established an
improvement. The later Runpod collection produced 227 training and 93 sealed
frontier entries from 1,920 and 768 searched candidates, respectively. These
are search data, not evidence that a new model was trained or validated.
The generated [PPO report](../outputs/v4/expanded_20260916/REPORT.md) and
[targeted-fit report](../outputs/v4/targeted_fit_20260920/REPORT.md) contain
the detailed results and limits. Their files under `outputs/` are local and
are not part of a fresh checkout.

## Correctness fixes and artifact compatibility

The September 22 review found failures in search/distillation and PPO diagnostics.
New simulator distillation samples synchronize the public estimator after the
scenario prelude (`sim-v4-distill-2`). Earlier samples with blank V4 event/belief
channels should be regenerated; changing the trainer cannot repair those inputs.
PPO quarantine measures entropy after the optimizer update. WAIT-only diagnostics
return null placement statistics, and return quantiles follow the tensor device.

The follow-up review also corrected frontier record preparation and retention
sample limits. Prefix caches and completed distillation runs now include dataset
settings and source match/verdict hashes in their identity, so changed labels or
sample limits cannot silently reuse old results. Existing caches rebuild; an
incompatible completed run requires a fresh output directory. PPO policy-movement
diagnostics and quarantine now measure the candidate after the optimizer step.
The return helper propagates terminal rewards to preceding actions, and destroyed
friendly Princess Tower sites become available in both deployment-mask paths.
Re-evaluate checkpoints after these simulator changes before comparing scores.

Keep older experiment outputs for provenance. Do not mix changed source code with
an existing frontier manifest or interpret older audit summaries as independent
confirmation. Use a new output directory for a changed protocol. Saved match
replays can be retained as inputs, but corrected labels and audit decisions need
their own versioned artifacts.

### Budgeted remote jobs

Run `scripts/run_v4_overnight_cpu.sh` through `scripts/run_v4_budget_guard.py`.
The guard stores an absolute deadline in its state file; restarting with that
file does not grant a new time budget. It reserves time for artifact transfer,
terminates the process group at the compute deadline, and runs optional archive
and provider-stop commands on success, failure, or timeout. Commands are JSON
argument arrays, not shell text.

For a wall-clock limit without provider billing control:

```bash
V4_REPO="$PWD" V4_PYTHON="$PWD/outputs/venv/bin/python" \
V4_CONFIRMED_VALIDATION_POOL=/path/to/confirmed-validation \
outputs/venv/bin/python scripts/run_v4_budget_guard.py \
  --state outputs/v4/budget/state.json \
  --max-runtime-seconds 3600 --reserve-seconds 300 \
  -- bash scripts/run_v4_overnight_cpu.sh
```

Set `V4_REPO`, `V4_PYTHON`, and the `V4_*_POOL`/`V4_RUN_DIR` variables to the
appropriate paths when using a different checkout. A dollar limit uses
`--budget-usd` and `--hourly-rate`, and **requires** `--stop-command-json` with
an authorized provider command that actually stops billing. Supply
`--archive-command-json` to copy results to durable storage before that stop.
Stopping Python alone leaves the pod billable. Use the current remaining balance
and rate when starting the guard; it is a wall-clock estimate, not an account-wide
billing meter. The guard cannot protect against its own SIGKILL or machine loss.

The chain requires `V4_CONFIRMED_VALIDATION_POOL` to point at a validation
dataset with held-out confirmation replicas. The old discovery-only validation
pool is insufficient. The audit must use `teacher-audit-v2` and pass the
fixed-action H28/H56 comparisons against H112 as well as the H128 first pass.
Confirmation adds simulator work; earlier runtime estimates for eight discovery
replicas do not include it. Re-estimate the runtime before allocating a budget.
The distillation runner also checks the number of independently confirmed
corrections after preparation. The budgeted chain requires at least 30 training
and 10 validation corrections; its frontier-entry count gate alone does not
establish that those labels passed confirmation.

The overnight CPU chain does not complete Expert Iteration, architecture sweeps,
PPO refinement, or final full-match confirmation. Its completion status names
only the implemented CPU phases.

`scripts/run_v4_learning_curve.py` runs a reproducible frontier PPO curve and
writes one artifact directory. It evaluates the supplied actor before any
updates and after each point in `--updates`. Every point uses the same sealed
entries, opponent specifications, and fresh evaluation seeds. Frontier train
entries are passed to the rollout collector only; they are never used as the
held out evaluation set.

The runner is an experiment harness around the existing APIs:

* `simulator.rl.v4.frontier_ppo.collect_frontier_rollout` collects a fixed number
  of decisions over the configured rollout lanes.
* `simulator.rl.v4.v4_ppo.update_v4_ppo` applies the PPO update and returns its
  diagnostics.
* `simulator.rl.v4.frontier_ppo.sealed_choice_regret` evaluates changed actions
  with common random numbers and reports per-entry paired uncertainty.
* `simulator.rl.v4.frontier_ppo.full_match_record` evaluates the fixed opponent
  pool on deterministic full matches.

The V4 runtime is configured before training with
`configure_v4_runtime`; WAIT always advances one simulator decision. The runner
also accepts injectable hooks in `simulator.rl.v4.learning_curve` so protocol
tests do not need model or frontier artifacts.

New curves use the explicit tower-delta reward (`--reward-mode tower-delta`),
which provides the change in tower-HP differential at each frontier decision.
Its undiscounted sum telescopes; with discounting it is a different objective
from terminal-only rewards, not policy-invariant reward shaping. Use
`--reward-mode terminal` for the sparse terminal ablation or `truncation` for
the terminal objective plus a final tower-difference lump. The default rollout
budget is 128 decisions across 8 lanes so the training episode cap can be
reached and the truncation semantics are exercised.

## Real run

The historical simx12 checkpoint is not checked into this repository. Supply a
real, compatible V4 checkpoint and both frontier files. The paths below are
placeholders for a fresh checkout:

```bash
PYTHONPATH=src outputs/venv/bin/python scripts/run_v4_learning_curve.py \
  --checkpoint /path/to/v4_actor.pt \
  --train-entries outputs/v4/frontier/frontier_train.pt \
  --sealed-entries outputs/v4/frontier/frontier_sealed.pt \
  --updates 0,1,2,4,8 \
  --eval-seed-start 10000 --eval-matches 24 \
  --out outputs/v4/curves/frontier_seed0
```

If the checkpoint is missing, the command fails with an explicit message. It
does not create a random replacement and does not claim an improvement.

The training episode cap (`--episode-cap`, default 128) is separate from the
full-match cap (`--eval-max-decisions`, default 1200). A match reaching the
latter cap is recorded as `truncated`, not as a draw. Truncated matches count
toward censoring metadata and are excluded from completed W/L/D score deltas.

## Evaluation-only baseline

To run the held-out protocol without collecting or updating:

```bash
PYTHONPATH=src outputs/venv/bin/python scripts/run_v4_learning_curve.py \
  --evaluation-only-checkpoint /path/to/v4_actor.pt \
  --sealed-entries outputs/v4/frontier/frontier_sealed.pt \
  --updates 0 --out outputs/v4/curves/baseline
```

The report marks `config.evaluation_only=true`; no PPO rollout or update hook
is called.

## Controlled protocol comparisons

The rollout duration and parallel lane count are explicit config fields:
`--rollout-decisions` (`--decisions-per-rollout`) and `--rollout-lanes`
(`--n-lanes`). To compare them, keep the checkpoint, entry hashes, seed,
update schedule, opponent specs, and evaluation seeds fixed, then run two
separate output directories while changing one field. The resulting reports
can be compared by update point and paired evaluation seed.

Model-size comparisons must use independently trained compatible checkpoints.
Pass additional compatible files with repeated `--compare-checkpoint`; each
is loaded and evaluated against the same baseline. For another architecture,
use a separate run with `--model-config` containing its actual ModelConfigV4
fields as JSON (or a JSON file), and its matching checkpoint.

## Smoke protocol

`--smoke` uses a randomly initialized lightweight synthetic policy and
generated entries. It exercises schedules, checkpoints, artifact hashes,
paired match rows, and truncation handling without quality claims:

```bash
python scripts/run_v4_learning_curve.py \
  --smoke --updates 0,1,2 --eval-matches 6 \
  --out /tmp/v4-learning-curve-smoke
```

This mode is an orchestration check and is not a simulator or policy-quality
benchmark.

## Artifacts and provenance

Each output directory contains:

* `config.json`: all protocol knobs, including rollout duration and lanes;
* `checkpoint_update_000000.pt`, then one plain actor checkpoint per requested
  update (with an optional critic sidecar from the real backend);
* `learning_curve.json`: point evaluations, update diagnostics, full paired
  match records, sealed choice rows, and automatic-promotion status;
* `provenance.json`: git revision, runner source hash, input artifact hashes,
  generated seed/opponent hashes, dirty source fingerprints, observation/runtime
  versions, and train/sealed separation metadata. Splits must have disjoint
  states and source matches; evaluation seeds must also be held out.

No `latest`, champion, or promotion pointer is created. Selecting a model for
deployment remains an explicit review decision after inspecting the curve.

## Runtime corrections and checkpoint compatibility

PPO now bootstraps from the actual successor observation before a truncation
reset, including the rollout tail. The frozen anchor produces its own target
logits with its own recurrent state. Full-match evaluation retains GRU state;
frontier training rotates through entries. WAIT duration is excluded from PPO
probabilities and entropy because the runtime executes one decision per WAIT.

The placement bias is registered before optimizer construction. The compact
model has 118,343 trainable parameters; the historical 117,767 count omitted
its 576 placement biases before their first forward pass.

Public observation schema `public-v3-2` feeds recent public events, stable
metadata card IDs, observed four-play rotation, and phase-aware elixir bounds.
It never reads the opponent's private hand. Champion deployments conservatively
disable the four-play exclusion because their lifecycle needs separate modeling.
Old tensors can remain shape-compatible, but their observation semantics and
runtime have changed: regenerate datasets and re-evaluate checkpoints. Old
reported scores are not directly comparable.

## Active search and recurrent distillation

Add these flags to the existing `scripts/run_v4_stage2_proof.py` workflow:

```text
--continuation-mode active --continuation-horizon 64
--active-match-decisions 64 --placement-target-mode tested --sequence-length 32
```

Active mode captures sequential states from two-sided controller play, starting
from the existing mechanics scenarios. Each candidate receives the same opponent
seed, and both sides continue playing after the forced initial decision.
The learner continuation uses the public rule teacher. A custom public policy
callback is supported by `search_state(..., continuation_mode="active-policy",
learner_controller=...)`; the dataset CLI exposes the implemented rule mode.

Placement targets put probability only on cells actually searched (or an
explicit one-hot legacy fallback). Geometric neighbors no longer inherit a
positive target without simulation evidence. Recurrent training preserves state
within contiguous sequences and resets at gaps, duplicates, and window boundaries.
The legacy wait-only teacher remains available as an ablation.

This is an outcome-search controller baseline, not an expert Hog 2.6 teacher.
Coverage statistics should be inspected before a substantial training run;
longer active horizons alone do not ensure card preservation, outcycling, or
endgame competence. The later frontier collection is a local generated
dataset; it does not include a newly validated pretrained model.

## Local validation

A real, tiny simulator curve using a random actor completed evaluation at updates
0 and 1 in `outputs/v4/runtime_validation_curve_final/`. An active dataset also completed
one recurrent training epoch in `outputs/v4/active_teacher_validation/`.
These ignored local artifacts exercise integration only. Their small budgets and
random starting model cannot establish an improvement over the historical bot.

The focused runtime/history/curve suite covers successor bootstrapping, frozen
anchors, public history, active sequence training, schedule independence,
quarantine, censoring, and artifact integrity. The existing teacher,
search dataset, distillation, and timing suites passed 34 tests in that earlier
validation. The follow-up review adds regression coverage for record preparation,
cache invalidation, policy movement, terminal returns, placement decoding, and
destroyed tower deployment. The Tornado assertions now match the pinned single
damage pulse. Repository-wide test discovery can include local external projects
and archived worktrees; a focused suite pass does not establish a full-suite pass.

Frontier candidate selection includes placement disagreements and applies minimum
spacing against every previously selected state, regardless of score order.
Alternative policy actions reuse the replayed recurrent state. Lost positions
are classified before the low-regret “solved” label, and verdicts retain individual
replica outcomes. A “solved” label only describes the tested alternatives and
horizon; it does not establish strategic mastery.

The restored-checkpoint pilot produced a 2/24 full-match win result and six entries
in each of the training and held-out splits. Its protocol and per-opponent results
are in the local generated
[baseline report](../outputs/v4/restored_baseline_20260915_reviewed/REPORT.md).
The expanded frontier and both PPO reward curves have since been run; see the
status above before planning another experiment.

## Next V4 experiment

Audit the search labels with independent paired replicas at multiple horizons,
then obtain a disjoint development-validation pool. The Runpod collection's
sealed split is reserved for final evaluation and must not become the
development-validation split. The search-distillation runner requires both
`--dataset-root` and `--validation-root`; its four methods compare hard and
soft search targets with or without conservative retention, and its context
settings compare zero-start with replayed GRU history. Use fresh output paths
for a changed protocol, select a candidate on development validation, and
open sealed states only after that selection. Full-match confirmation follows
the search-based checks. PPO refinement and architecture sweeps remain later
work, not completed steps of the current local dataset collection.

## Package layout and retired runners

Search distillation is organized under `simulator/rl/v4/search_distillation/`:

| Module | Responsibility |
| --- | --- |
| `records.py` | Actions, distributions, cached prefixes, and named correction records |
| `targets.py` | Paired confidence checks and hard/soft search targets |
| `preparation.py` | Source-match replay and correction/retention selection |
| `batching.py` | Tensor batches and recurrent forward inputs |
| `losses.py` | Correction and retention objectives |
| `evaluation.py` | Validation metrics and paired simulator outcomes |
| `training.py` | Optimizer loop and epoch-boundary resume |
| `artifacts.py` | Deadlines, checkpoint writes, and provenance helpers |

The package exports the existing public API, including record classes needed to
load older serialized caches. Preparation has one implementation; no runtime
replacement or permissive legacy legality fallback is used. The runner separates
cache preparation from each experiment's resume/report lifecycle and fingerprints
every implementation module. Old cache records remain readable, but the changed
source identity requires a fresh output directory for previously completed runs.

Refactor validation compares deterministic two-epoch runs of all four methods
against the preceding implementation, with identical weights and training metrics.
Regression tests also cover old cache class paths, exact training resume, changed
input rejection, and failure/interruption handling in multi-run sweeps.

`simulator.rl.v4` owns current training and search. Shared opponent pools,
scenario environments, models, and tensor utilities are in
`simulator.rl.foundation`. The `simulator.rl.legacy` package and the
`recurrent-prototype` CLI command have been removed, together with the old
generalized, league, rotation, shadow, and prototype benchmark runners.

Existing live public-actor checkpoints load through
`simulator.rl.foundation.inference_checkpoint`; the reader preserves their
architecture and validation contract. It is an inference compatibility path,
not a V4 checkpoint converter.
