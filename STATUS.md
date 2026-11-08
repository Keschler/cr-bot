# Project goal and status

Updated: 2026-11-08. This is the canonical current goal and status. Technical
guides describe how the software works; experiment reports preserve dated
evidence rather than defining another current plan.

## Goal

Build a very strong policy for the fixed Hog 2.6 deck against all enemy decks
and diverse opponent styles. Choose training methods and simulator work from
measured progress toward that goal. Simulator wins alone do not establish
live-game or human-level strength.

## Selected policy

The local selection remains `targeted-control-20261019-epoch4`, selected by
the user on October 20:

```text
outputs/v4/tactical_learning_20261019/fit_control/actor_epoch_004.pt
SHA256: 123f3ab04c10c19157e1d37f0f05b4c669bdb5d7478c654fe6d942d5319d01bd
```

The local receipt is `outputs/v4/selected_actor.json`. The checkpoint bytes
were checked against that receipt. Inference uses public observations,
carried recurrent memory, native stochastic sampling, and fixed 250 ms
decisions. Checkpoints and generated evaluation recordings are local
artifacts, so a fresh checkout needs those assets separately.

The user selected this actor despite its original failed Bait safeguard;
the receipt retains `engineering_gate_passed: false`. Its original screen
won **38/120** games versus the initializer's **27/120**, across 30 training
deck identities. The ten reserved decks were not evaluated for this actor.
See the [historical tactical results](docs/experiments/v4_tactical_learning_results.md)
and [checkpoint/playback guide](docs/workflows/checkpoints-and-replays.md).

## Latest completed experiment

The first replay-correction experiment found four independently validated
training corrections: three Cannon responses against X-Bow and one Fireball
response against Lavaloon. It found no accepted calibration corrections.

| Fresh conditions | Games per actor | Selected control wins | Candidate wins |
|---|---:|---:|---:|
| Fixed opponents | 30 | 8 | 8 |
| Adaptive and split-pressure | 60 | 14 | 12 |
| Total | 90 | **22** | **20** |

This run did not establish improvement. The selected actor is unchanged;
confirmation and reserved-deck evaluation did not run. The local evidence is
under `outputs/v4/replay_corrections_20261029/`; the
[dated results](docs/experiments/v4_replay_correction_results.md) explain the
coverage and training limitations. These rates use different conditions from
the older 120-game screen and should be compared with their paired control.

## Active work

The requested next improvement is broader, more informative correction data
and stronger corrective practice:

1. Filter origins with no useful alternative before assigning review slots;
   emphasize actual Princess/Cannon placements, early siege responses,
   spending before defensive shortages, and competing threats.
2. Validate short saving/defense sequences and retain their actual branch
   observations and follow-up actions as training evidence.
3. Compare ordinary and increased correction exposure with matched optimizer
   and retention budgets, preserving complete recurrent histories.
4. Measure PLAY, card, and placement behavior on training and held-out origins
   before spending on fresh complete-match comparisons.

The factual-correction trainer supports weighted corrections, focused
exposure within the existing optimizer budget, complete-prefix memory
reconstruction, and sealed before/after diagnostics and learning-gate results.
Saving-plan control, validation and capture components are tested separately.
Whole-sequence collection/training, the revised sampler, and pipeline-level
learning-gate scheduling still need complete orchestration; nonempty sequence
training manifests are explicitly rejected rather than silently ignored.
**No completed second correction experiment or new policy improvement is
recorded.** See the
[correction workflow](docs/workflows/v4_replay_corrections.md) for the supported
commands and integration boundaries.

## System boundaries

The deterministic simulator has 124 definitions, a 109-card eligible opponent
roster, and a source-linked 40-deck catalog split into 30 training and ten
reserved evaluation decks. Ruleset metadata remains `training_ready: false`
because fidelity evidence is incomplete. The current catalog and scripted
opponents do not cover every human deck or playing style.

Video analysis, live phone observation, policy inspection, evaluation
playback, and calibrated device control are implemented. Vision errors and
simulator-to-real transfer remain limitations. The
[physical-lab guide](simulator/physical_lab/README.md) describes controlled
capture, extraction and comparison; physical captures need their provenance
and fidelity checks before qualifying as evidence.

No curated expert demonstrations are available. Human-data training remains
disabled. The [policy-learning design](docs/design/policy-learning.md) records
the broader strategy and available experimental mechanisms.

## Repository conventions

Validation on 2026-11-08: **2,103 Python tests passed, 45 skipped, and three
existing failures remain** (annotation maximum-cardinality matching, missing
local spell-video media, and the KataCR tracker compatibility import).
All 12 frontend JavaScript checks passed. Local validation logs are under
`outputs/cleanup_20261108/`. No training or physical experiment ran during the
repository cleanup.

- [README.md](README.md): setup, entry points, and source layout.
- [docs/README.md](docs/README.md): technical guides and experiment index.
- `outputs/`: local environments, models, recordings, reports and run artifacts.
- `outputs/local/`: preserved standalone media/models and former root scratch
  files; these are intentionally outside source control.

Keep current status here. Preserve useful dated evidence under
`docs/experiments/` and reusable instructions under `docs/workflows/`.
