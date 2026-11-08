# Tactical-learning results, October 20, 2026

> Historical record (2026-10-20). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The basic targeted student was the strongest observed variant: **38/120 wins
(31.7%)**, versus **27/120 (22.5%)** for the unchanged initializer. Prediction
heads scored 36/120 and adaptive practice 32/120. Neither addition beat the
matched targeted control. All three students failed the predeclared matchup
safeguards, so **this experiment did not promote an actor**. Confirmation and the
ten reserved decks were not run; this is not a new 40-deck benchmark.

The [implementation and protocol](../workflows/v4_tactical_learning.md) and
[sealed experiment report](../../outputs/v4/tactical_learning_20261019/RESULTS.md)
describe the reproducible workflow. The strategic goal remains very strong
Hog 2.6 against all enemy decks. These experimental gates are research choices,
not a universal definition of success.

## Controlled comparison

All students reloaded the same hashed compact initializer. The shared corpus
contained 90 full training episodes across all 30 training deck identities,
with actor/cycle/plans drivers and balanced training styles. Thirty further
episodes used disjoint calibration origins. Development styles and the ten
reserved deck identities were excluded from collection and gradients.

Each arm received four epochs, **288,612 real-frame visits and 4,972 optimizer
calls**, with the same episode order and grouping. Training used 32-step TBPTT,
two episodes per batch, Adam at `5e-5`, gradient clipping at one, and fixed
250 ms WAIT. Student and frozen anchor each carried their own chronological
recurrent memory. All 56,660 inactive training rows per epoch received the
full masked action-distribution retention objective; 15,493 active air/X-Bow
rows received tactical supervision. Calibration selected no epoch.

Fresh evaluation used 60 fixed-controller games per actor and 60 games against
the two withheld development styles. These are the same 30 training deck
identities with fresh origins and different opponent decisions, rather than
held-out deck identities.

| Actor | Fixed controllers | Withheld styles | Combined | Win rate |
|---|---:|---:|---:|---:|
| Original initializer | 17/60 | 10/60 | 27/120 | 22.5% |
| Targeted control | 18/60 | 20/60 | 38/120 | 31.7% |
| Control + prediction heads | 19/60 | 17/60 | 36/120 | 30.0% |
| Control + adaptive practice | 15/60 | 17/60 | 32/120 | 26.7% |

Prediction heads finished **two wins below control**; adaptive practice
finished **six below control**. The initial fixed-controller advantage of the
prediction variant did not survive the combined comparison. There is only one
fitting seed, two fixed-controller match seeds and one shared development match
seed. The coarse paired seed-cluster ranges include zero for both additions
versus control; this pilot does not establish a general ranking of methods.

## Matchup gains and regressions

| Family | Games per actor | Initializer wins | Control | Prediction heads | Adaptive practice |
|---|---:|---:|---:|---:|---:|
| X-Bow | 24 | 2 | 11 | 10 | 8 |
| Lavaloon | 24 | 2 | 4 | 3 | 1 |
| Golem | 20 | 3 | 7 | 4 | 3 |
| Giant | 4 | 0 | 2 | 2 | 1 |
| Bait | 24 | 8 | 6 | 7 | 5 |
| Hog | 12 | 10 | 8 | 7 | 10 |
| P.E.K.K.A bridge | 12 | 2 | 0 | 3 | 4 |

The control's main gain was X-Bow: **+9 wins**. It also gained four Golem and
two Lavaloon wins, while losing two Bait, two Hog and two P.E.K.K.A games.
It failed the Bait nonloss safeguard. Prediction heads failed Bait retention
and lost three Hog games. Adaptive practice failed Bait retention, the general
family-loss threshold and the required Lavaloon gain. All met the overall
screen gain, but none met every declared condition.

The Bait regressions cannot be explained simply as direct tactical teaching
on all Bait decks. Only skeleton-barrel supplied active Bait training contexts:
501 rows, with 462 teacher WAITs and 39 PLAYs. The other five Bait identities
had no active rows and were fully anchored on their collected histories.
Their losses show that finite retention on recorded states does not guarantee
retention after the student generates different trajectories. Aggregate play
counts do not identify the decisive timing or placement error, and do not
support a universal excessive-waiting explanation.

## What the additional mechanisms contributed

**Prediction heads did not become useful forecasters in this fit.** On all
30 chronological calibration episodes, the mean tower-damage MSE was
**0.01332785**, versus **0.00405909** for constant predictions using training
means: **3.28 times worse**. All six horizon/side outputs lost to that baseline
and had negative explained variance. Outputs varied with the inputs, but the
variation was noisy and poorly calibrated. Lower auxiliary training loss was
therefore not evidence of tactical prediction skill.

This pilot supervised factual own/enemy tower damage at 2, 6 and 12 seconds.
Threat/defender survival targets remain masked because the public tensor
contract lacks reliable stable identities. The labels describe the original
actor/cycle/plans drivers, not counterfactual action values or the new student's
on-policy future. See the [head diagnostics](../../outputs/v4/tactical_learning_20261019/head_diagnostics.md).

**Adaptive practice genuinely changed the objective**, rather than silently
repeating control. It covered 35 public tactical contexts, kept every episode
and inactive retention row, and maintained population-mean-one weights. Actual
weights ranged from 0.410 to 1.298 across the later epochs. This train-only
weighted curriculum did not improve the outcome screen over uniform control.
It is not a test of every possible curriculum or arbitrary snapshot replay.

**Coordinated plans executed correctly but added no teacher wins.** The
paired teacher screen tied at 37/60 for both cycle and plans, with no changed
match outcome. The controller made 106 accepted planned plays, with zero
rejections. Its student fit was skipped because the teacher missed the
predeclared improvement threshold. Plan execution alone does not establish
better strategy.

## Verification and artifacts

The overnight evaluation stopped at 184 completed actor games. The resume
validated and reused those records, then completed the remaining grid without
retraining or changing source. No independent pre-resume byte-hash snapshot
was captured; the final complete raw digest manifests were independently
verified.

The [independent audit](../../outputs/v4/tactical_learning_20261019/independent_audit.md)
verified the 153-file source seal, initializer, all 120 episode files, all
three fits and exports, all **600 teacher/actor raw games**, and completion
seals. There were zero rejected learner actions or card-acceptance mismatches.
Two additional fresh-process cold replays reproduced their native records and
diagnostics exactly. All three saved calibration reports were independently
recomputed. No actor was promoted and no external device was used.

The 215 focused tests passed. The final default suite reported **1,959 passed,
45 skipped and eight failures**, matching the earlier failure list: five
sandbox forkserver socket PermissionErrors, annotation matching, spell scenario
media and KataCR tracker import. No tactical test failed. Automatic approval
review of an earlier attempt to rerun the five subprocess checks outside the
sandbox timed out; those checks remain blocked in this environment.

Artifacts are under `outputs/v4/tactical_learning_20261019/`:

- `completion_receipt.json`, `results.json` and `RESULTS.md`: completed screen
  and eligibility decisions.
- `policy_analysis.md` and `.json`: paired family/style comparisons, fit
  diagnostics, Bait flips and uncertainty.
- `head_diagnostics.md`, `.json` and `.py`: factual forecasting measurements
  and reproducible read-only helper.
- `independent_audit.md` and `.json`: provenance, legality and cold replay
  verification.
- `default_tests_20261020.log`: full test output.

The completion result SHA256 is
`fa76d9bfbf1aaf4c1056ab5067b24cfdf0d21f7f7223eccba606354f655b58c9`.
At this experiment's decision, the retained initializer had SHA256
`ba15ded00742ec68b509e89da8ad88bcc12b0e0c7bf2f76d8edd25e869903c3b`.

## Continuation proposed at the time

Use the targeted control as the research reference and diagnose its exact Bait
loss pairs while preserving the X-Bow gain. Record the full public trajectories
to locate the consequential divergence before choosing a retention change;
the current evaluation summaries contain aggregate behavior and trace hashes,
not the scene evidence needed to identify that error. A repair should be
tested against control on fresh matches with the same family safeguards.

Require a revised consequence predictor to beat simple factual baselines on
disjoint trajectories before sharing its gradients with the actor. Neither
larger prediction heads nor a larger curriculum is supported by this pilot.
The simulator's human transfer and strong play against arbitrary decks remain
unresolved.

The subsequent [Bait regression diagnosis](v4_bait_regression_diagnosis.md)
reproduced all eight changed-outcome pairs exactly and ran a separate,
predeclared 180-game check. On those 90 fresh Bait pairs, the initializer won
21 games and control 25; the old overall Bait deficit did not repeat.
Single-action interventions rescued three of the five old lost-win pairs,
including a Cannon placement that exposed the wrong lane to Skeleton Barrel.
These diagnostics establish tactical fragility and substantial seed variation,
without changing the original gate result or promoting a checkpoint.

On October 20, the user subsequently [selected the targeted control](../../STATUS.md)
as the local policy. That separate explicit selection preserves the historical
screen qualification reported here. Verified playback exports cover all 120
original control matches.
