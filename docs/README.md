# Documentation

[Project status](../STATUS.md) is the source for the strategic goal, current
selected policy and latest experiment decision. [Repository setup](../README.md)
and the [simulator guide](../simulator/README.md) cover installation and runtime
contracts.

## Workflows and design

- [V4 training and evaluation](workflows/v4_training.md): bootstrap, PPO,
  search distillation, reproducibility and evaluation boundaries.
- [Tactical learning](workflows/v4_tactical_learning.md): matched consequence,
  plan and adaptive-practice experiments.
- [Replay corrections](workflows/v4_replay_corrections.md): fresh histories,
  branch confirmation, sparse training and full-match checks.
- [Checkpoints and evaluated replays](workflows/checkpoints-and-replays.md):
  receipt use, viewer playback and original-runtime export verification.
- [Policy-learning design](design/policy-learning.md): evidence-driven choices
  for teachers, opponent diversity, value learning and transfer.
- [Physical-fidelity lab](../simulator/physical_lab/README.md): controlled
  measurement, device preparation and evidence admission.
- [Action evaluation](eval/README.md): ground truth, annotation and scoring.

## Experiment records

These are dated evidence, including failed gates and superseded selections.
Their proposed follow-ups describe the decision at the time.

| Period (2026) | Record |
| --- | --- |
| Sep 30–Oct 1 | [Bounded comparison plan](experiments/v4_comparison_plan.md) · [pilot results](experiments/v4_pilot_results.md) |
| Oct 4 | [Evaluation audit](experiments/v4_evaluation_audit.md) · [40-deck results](experiments/v4_catalog_results.md) · [failure diagnosis](experiments/v4_catalog_diagnosis.md) |
| Oct 4–6 | [Initializer recovery and PPO](experiments/v4_recovery_results.md) |
| Oct 6 | [Catalog initializer](experiments/v4_initializer_coverage_results.md) · [remaining losses](experiments/v4_remaining_losses_diagnosis.md) · [memory and defense repair](experiments/v4_defense_repair_results.md) |
| Oct 7–18 | [Teacher, style and value-learning results](experiments/v4_human_strength_results.md) |
| Oct 20 | [Tactical-learning results](experiments/v4_tactical_learning_results.md) · [Bait diagnosis](experiments/v4_bait_regression_diagnosis.md) |
| Oct 26 | [Replay review](experiments/v4_replay_review_20261026.md) |
| Oct 29 | [First replay-correction results](experiments/v4_replay_correction_results.md) |

Links into `outputs/` refer to local generated receipts and media; a fresh
checkout may not contain them. Reports retain their dated hashes and results
without turning those artifacts into committed source or a current selection.
