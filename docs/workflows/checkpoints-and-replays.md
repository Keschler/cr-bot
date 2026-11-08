# Checkpoints and evaluated replays

The current checkpoint path, SHA-256, selection receipt and decision are in
[project status](../../STATUS.md). Existing experiment plans and raw results keep
their original actor identities. Pass `--actor-json outputs/v4/selected_actor.json`
explicitly to runners that support it; a runner's historical default may refer
to an earlier experiment.

Actors require the exact architecture from the fit receipt, public
observations, carried recurrent memory, native action sampling and fixed
250 ms decisions. Preserve the checkpoint bytes and complete model
configuration when changing a local selection pointer.

The phone/video frontend and prototype CLI take an explicit checkpoint or
`CR_BOT_V4_CHECKPOINT`; they do not automatically read actor JSON receipts.
Resolve the path from the receipt before using those workflows. The
[live prototype guide](../../simulator/RUN_PROTOTYPE_LIVE.md) documents calibration
and device execution.

## Watch the evaluated games

From the repository root:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m sim_viewer
```

Open **http://127.0.0.1:8010/**, choose **Evaluation**, and search for a deck,
opponent style or match seed. Click a match and press **Play**. Player zero
at the bottom is the policy. Use the timeline, arrow keys, speed control and
policy heatmap to inspect decisions; **Result** filters wins/losses and
**Order** groups them. The [viewer guide](../../src/sim_viewer/README.md) describes
playback and recording.

## Historical October control screen

The October 20 control screen contains 120 matches: 60 fixed-opponent games
and 60 adaptive/split-pressure games across all 30 training deck identities.
It scored 38 wins and 82 losses; the ten reserved deck identities were not
played for this screen. The [tactical report](../experiments/v4_tactical_learning_results.md),
[Bait diagnosis](../experiments/v4_bait_regression_diagnosis.md) and
[October 26 replay review](../experiments/v4_replay_review_20261026.md) retain the evidence and
its limits. These results do not establish human-level play.

Verified recordings are under `outputs/v4/sim_viewer/evaluated_control/`.
Their manifest and per-match receipts bind each JSONL to the original raw
result, model and source hashes. Export reconstructs the original native job
and publishes only after the entire native record and diagnostics match.

The following historical rebuild command requires the original sealed
simulator runtime. Added replay-correction modules change that source seal;
use the original runtime to rebuild October recordings. Existing verified
recordings still play. Export new evaluations from their own screen directories
and sealed runtime.

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m sim_viewer.export_evaluations \
  --screen outputs/v4/tactical_learning_20261019/screen_train/control \
  --screen outputs/v4/tactical_learning_20261019/screen_dev/control \
  --out outputs/v4/sim_viewer/evaluated_control --workers 3
```

All 120 exports reproduced the original native records and diagnostics exactly.
Recording and original-result hashes were checked; X-Bow, Bait and Lavaloon
examples opened with heatmaps and terminal frames. The dated
[validation receipt](../../outputs/v4/promotions/20261020_targeted_control/viewer_validation.json)
and [test log](../../outputs/v4/promotions/20261020_targeted_control/default_tests.log)
preserve the checks and environment failures from that work.

The [replay-correction workflow](v4_replay_corrections.md) collects fresh full
actor histories and tests alternative actions under its own source seal.
Its new results must not be substituted for the historical export verification.
