# First replay-correction experiment

> Historical record (2026-10-29). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The replay-branch correction pipeline is implemented and the first bounded
experiment is complete. **The candidate won 20/90 fresh games versus 22/90
for the unchanged selected policy.** This run did not establish improvement;
this experiment did not promote a replacement. No confirmation or reserved-deck
evaluation ran.

[Workflow and commands](../workflows/v4_replay_corrections.md) describe the reusable
implementation. [Generated report](../../outputs/v4/replay_corrections_20261029/REPORT.md)
and [complete results](../../outputs/v4/replay_corrections_20261029/result.json)
contain the sealed experiment outputs.

## Fresh match results

Each actor played the same 90 conditions across all 30 training-catalog decks:
one fresh seed with fixed opponents, and another fresh seed with each of the
two development styles. These are 90 matched pairs / 180 complete games, not
a new 40-deck final benchmark.

| Conditions | Games per actor | Selected control wins | Correction candidate wins |
|---|---:|---:|---:|
| Fixed opponents | 30 | 8 | 8 |
| Adaptive and split-pressure | 60 | 14 | 12 |
| Total | 90 | **22 (24.4%)** | **20 (22.2%)** |

| Family | Games per actor | Control wins | Candidate wins | Net |
|---|---:|---:|---:|---:|
| X-Bow | 18 | 8 | 6 | -2 |
| Bait | 18 | 1 | 2 | +1 |
| Lavaloon | 18 | 1 | 2 | +1 |
| Golem | 15 | 1 | 0 | -1 |
| Giant | 3 | 1 | 0 | -1 |
| Hog | 9 | 7 | 7 | 0 |
| PEKKA bridge | 9 | 3 | 3 | 0 |

All matches genuinely terminated, both players had zero invalid actions,
and no pairs were excluded. This small screen does not establish a systematic
two-win regression, but it supplies no reason to replace the selected actor.
Its win rate should be compared with its paired control, not with the older
120-game screen using different seeds and condition weights.

## What search established

Fresh actor-only collection produced 30 complete training and 30 complete
calibration episodes, with separate match origins and full carried memory.
The bounded search reviewed 36 training and 12 calibration decisions. Four
training corrections passed independent validation: **three Cannon plays
against X-Bow and one Fireball against Lavaloon**. No calibration correction
passed. Nineteen reviewed states had a ranged-ground tag; none became a
corrective label. Tags overlap and do not classify an action as a mistake.

The clearest example is `train_actor_06_00_xbow-30`, decision 488, simulator
elapsed time 122.00s. With 7.285 elixir, the recorded action is WAIT. Cannon
at `(column=5,row=18)` wins all six independent 64-second continuations;
the recorded-action continuations yield one win, two losses and three
nonterminal outcomes within that horizon. Both players continue normally
after the changed first action. This establishes a useful local correction,
not general X-Bow competence.

The air correction replaces Musketeer with Fireball at 140.00s against
Lavaloon Barbarians. It is a counter-selection correction; **no accepted
target explicitly teaches WAIT/saving**. A pre-starvation mining tag does
not by itself establish that the better action saves elixir.

Thirteen reviewed states had only the factual legal branch. These produced
no corrections and show that candidate mining should reject uninformative
origins before spending the search quota.

## What training established

The candidate received two epochs over 30 complete actor histories: 46,068
frame visits and 799 optimizer steps. Four correction labels were presented
twice each: **eight supervised presentations**. Other valid recorded rows
used full-action KL retention against the selected actor.

All four target probabilities increased on their recorded histories. Mean
target NLL fell from **10.804 to 9.884**. Most improvement came from choosing
the target card; conditional placement probabilities barely changed for the
three Cannon labels. In the X-Bow Rascals example, PLAY probability moved
from 0.047% to only 0.073%; Cannon probability conditional on PLAY moved
from 1.47% to 2.30%. That decision remains largely unrepaired.

Exact-action probability includes mode, card and cell factors. Small joint
probability alone does not establish a refusal to play: the other three
examples already had much higher PLAY probability. The
[factor-by-factor diagnostics](../../outputs/v4/replay_corrections_20261029/action_agreement.json)
retain these distinctions and nearby-cell probability mass.

With zero accepted calibration labels, there is no held-out measurement of
correction learning. Low average retention KL on recorded states also does
not guarantee preserved outcomes after the student visits different states.
This run does not identify a model-capacity or memory limitation.

## Follow-up proposed at the time

Expand independently validated corrections across fresh origins and all four
failure classes, especially actual Cannon/Princess placement decisions and
genuine upstream saving choices. Filter origins with no legal alternative,
and retain successful cheap tanks, stalls and cycles as positive controls.
Then predeclare a small comparison of correction exposure/weight, measuring
PLAY, card and placement separately while preserving full histories and broad
retention. More epochs over these four origins alone would not provide the
missing behavioral coverage.

The evaluated implementation changed one action followed by the
frozen actor. Longer saving plans need their own validated follow-up targets;
the initial WAIT must not receive credit for a scripted plan the actor never
executed.

## Artifacts and verification

- Origin actor SHA: `123f3ab04c10c19157e1d37f0f05b4c669bdb5d7478c654fe6d942d5319d01bd`.
- Candidate: `outputs/v4/replay_corrections_20261029/fit/actor_epoch_002.pt`,
  SHA `dc455296c5b44abeff8bf30ea145c058a0cbb8087c59bec2f0822672c8b01bd9`.
- Experiment source SHA: `86d4d514f7ef386d7e19b0a1c550db60ef27bf60dd3bccfc225cfb8182318485`.
- Implemented modules: `correction_proposals.py`, `replay_corrections.py`,
  `correction_training.py`, with CLI `scripts/run_v4_replay_corrections.py`.
- Independent audit rechecked all acceptance gates and all 180 raw match
  records. Replaying the four accepted origins covered 1,603 chronological
  actor steps with exact state hashes and every public input matching.
- **71 new regression tests passed.** The full non-audio suite finished with
  **2,047 passed, 45 skipped and the same eight existing failures**: five
  sandbox forkserver permission errors, annotation matching, missing spell
  media and KataCR tracker import. [Validation receipt and log](../../outputs/v4/replay_corrections_validation_20261029/validation.json).

The run left the selected-policy receipt and original checkpoint bytes unchanged.
