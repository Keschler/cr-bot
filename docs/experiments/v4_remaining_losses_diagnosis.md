# Why the October 6 initializer lost 77% of catalog games

> Historical record (2026-10-06). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The October 6 initializer lost **308/400 games**. The clearest demonstrated weaknesses are air and siege defense, imperfect imitation of a limited rules teacher, and a mismatch between recurrent training and deployment. PPO has not supplied a confirmed improvement. The evidence does not assign a causal percentage of losses to each mechanism.

This is a read-only diagnosis of the [completed catalog experiment](v4_initializer_coverage_results.md), performed on 2026-10-06. It analyzed all 800 preserved final records, recalibrated both fitted actors without optimization, exactly replayed 34 predeclared cases, and ran the revised teacher on the same 300 training-deck match cells as the selected actor. No policy, opponent, simulator source, training objective, or checkpoint selection was changed. The strategic goal remains a very strong Hog 2.6 policy against all enemy decks.

Checkpoint selected in this experiment: `outputs/v4/initializer_coverage_20261006/fit_saving/actor_epoch_128.pt`, SHA256 `ba15ded00742ec68b509e89da8ad88bcc12b0e0c7bf2f76d8edd25e869903c3b`.

The subsequent [memory and defense repair experiment](v4_defense_repair_results.md)
is complete. Fresh memory-reset confirmation retains carried state. Revised
defense labels produce a promising X-Bow gain, but the trained candidate fails
the Bait safeguard and does not replace this actor. The historical diagnosis
below remains a record of the original catalog evaluation.

**The failures are concentrated in air and siege defense.**

| Actual deck family | Wins / games | Win rate | Losses |
|---|---:|---:|---:|
| Hog | 30/50 | 60.0% | 20 |
| Bait | 29/80 | 36.3% | 51 |
| Golem | 16/70 | 22.9% | 54 |
| P.E.K.K.A. bridge | 4/30 | 13.3% | 26 |
| Giant | 1/10 | 10.0% | 9 |
| X-Bow | 8/80 | 10.0% | 72 |
| Lavaloon | 4/80 | 5.0% | 76 |
| Total | 92/400 | 23.0% | 308 |

Lavaloon and X-Bow account for **148/308 losses, or 48.1%**. Eleven decks have zero wins in their ten matches. Actual families matter: Hog and P.E.K.K.A. share the `aggressive-pressure` controller strategy, but their outcomes are very different. The catalog has 40 deck identities implemented by five scripted controller strategies.

The training decks yield 68/300 wins (22.7%), and the ten reserved decks yield 24/100 (24.0%). The weakness is already present on decks included in training. This does not identify an aggregate collapse specific to reserved decks, although individual deck results remain noisy.

All 400 selected-policy games finished without draws, truncation, invalid learner actions or rejected plays, and all native action traces match the saved diagnostic traces. The previously documented single scripted X-Bow placement rejection occurs in the recovered comparator. Both policies lose that paired cell; it cannot account for the selected policy's 308 losses.

**The teacher itself is too weak to supply a strong final policy.**

The revised rules teacher won **99/300 (33.0%)**, compared with the student's **68/300 (22.7%)**, on exactly the same 30 training decks and ten final match seeds. The paired difference is 31 wins: 61 student losses become teacher wins and 30 student wins become teacher losses. Teacher actions are deterministic and student actions are stochastic; their trajectories diverge. This is a controlled capability comparison, not a teacher upper bound or a causal decomposition.

| Training-deck family | Teacher wins | Student wins | Games each |
|---|---:|---:|---:|
| Hog | 21 | 20 | 30 |
| Bait | 19 | 20 | 60 |
| Golem | 26 | 11 | 50 |
| P.E.K.K.A. bridge | 10 | 4 | 30 |
| Giant | 5 | 1 | 10 |
| X-Bow | 5 | 8 | 60 |
| Lavaloon | 13 | 4 | 60 |

The teacher wins more games in Golem and air matchups, but still loses two-thirds of this grid and wins only 5/60 X-Bow games. More accurate cloning of the current teacher therefore leaves substantial strategic weaknesses.

There is a reproduced **remote X-Bow blind spot**. The teacher recognizes forward threats only at normalized `y > 0.45`. An enemy X-Bow at native `y=13.5` lies at `13.5/32=0.421875` and is excluded, even while shooting an own tower. A separate remote-cluster spell branch can respond at eight or more elixir, but there is no general individual siege-defense branch.

For example, against `xbow-30`, seed `2026100665`, at 50.50 seconds, X-Bow UID67 is at `(14.5,13.5)`, with 560 HP and target UID3, the right Princess Tower. At 4.035 elixir, both the teacher and student send Hog down the left lane; the public teacher context says zero threats. Similar snapshots appear for seven X-Bow instances across five selected cases. Authoritative target IDs are used only to diagnose the resulting combat; they were not supplied to the teacher or policy. X-Bow contributes **85.2% of actual incoming combat tower damage** in the seven selected X-Bow replays.

The teacher otherwise uses short rules for card choice and an anchor two rows behind a threat. Its ground coverage check approximates nearby healthy Cannon/Musketeer defense. These rules do not establish optimal targeting, pulling, kiting, elixir trades or deployment timing. The direct teacher benchmark measures their remaining limitations.

**The student has a substantial imitation and generalization gap.**

On revised-teacher PLAY examples, deterministic agreement on PLAY mode, exact card and a legal placement within one cell falls from **97.3% on training data to 61.5% on development data**. There are 1,459 train PLAY labels and 335 dev PLAY labels. Of the 335 dev labels, 43 fail at PLAY/WAIT, a further 41 fail at card choice, and a further 45 fail at placement; 206 pass all three.

| Dev target/cohort | PLAY labels | Joint agreement within one cell |
|---|---:|---:|
| All revised PLAY labels | 335 | 61.5% |
| Hog | 72 | 83.3% |
| Musketeer | 46 | 65.2% |
| Fireball | 13 | 23.1% |
| Air-threat states | 67 | 38.8% |

Fireball card accuracy is only 30.8% on those thirteen dev labels, despite conditional placement agreement of 92.3%. Choosing the appropriate spell is a problem before placement is considered. The denominator is small: the revised training set has only 54 Fireball labels, versus 356 Hog and 235 Musketeer labels.

These values describe retained teacher-driven observations. Frequency-weighted joint agreement is 68.0% on dev; that estimate still describes the collection mixture, rather than the actor's deployed state distribution. The previously reported high mode accuracy measures PLAY/WAIT alone and should not be interpreted as full tactical accuracy. Teacher agreement itself is also not optimal play.

The exact replays show concrete differences on student-visited states. These are **29 deliberately selected median-duration losses**, one per training deck with a loss, and **five median-duration wins**, one per controller strategy. Their tactical statistics are not representative estimates for all 400 games. Across these 34 cases, the teacher has an affordable air-capable recommendation at 763 correlated decision frames. There are 111 actual plays in that cohort: 70 use the teacher's card, 33 use Cannon, Skeletons, Ice Golem, Hog or Log, and eight use another air-capable card. Ground cards can distract or pull particular enemies, so these substitutions are not automatically worthless.

Against `lavaloon-lightning`, seed `2026100663`, at 25.25 seconds, a Mega Minion targets the right Princess Tower. Musketeer is in hand and affordable at 4.017 elixir, but the actor selects Cannon; its conditional Musketeer probability is only 2.9%. Later, from 124.50 through 131.25 seconds, it waits for seven seconds while a legal Fireball recommendation persists, then plays Hog at 131.50 seconds. It subsequently deploys Cannon on the left while the teacher recommends Musketeer against the right-side air threat.

Among 914 executed non-Hog defensive plays that match the teacher's card, 116 placements differ by more than five cells. In 31 of those, the actor's most likely cell is near the teacher target but sampling selects a distant cell. In 76, even its most likely placement is more than five cells away. Against `xbow-pump-snowball`, seed `2026100667`, at 84 seconds, a left-side Knight is met by a Cannon sampled on the right, thirteen cells from the teacher target; the sampled placement had only 0.10% conditional probability. These examples distinguish sampling tails from a learned placement distribution that already disagrees.

Most resolved spells do cause damage in these replays: **158/172 Logs and 47/48 Fireballs** reduce HP or shields. Resolved zero-damage casts are uncommon in these selected cases, but causing damage does not establish good targeting, timing, elixir value or a causal effect on winning. These counts use historical spawn, resolution and damage events; the native engine prunes spent projectiles, so end-state projectile counts from the initial replay helper are invalid history and are superseded by `spell_summary.json`.

**The initializer's recurrent training and deployment are inconsistent.**

Supervised fitting shuffles isolated observations, resets the GRU for every example, and unrolls only one timestep. Full-match inference carries hidden state across 250 ms decisions. Saved tensor comparison confirms that `recurrent.gru.weight_hh_l0` remains bitwise identical to the random starting actor: initialization never trained the hidden-to-hidden matrix subsequently used in deployment.

An observational probe on identical replay observations finds that zeroing hidden state changes PLAY/WAIT argmax in **319/13,271 legal-PLAY frames (2.4%)**. Mean PLAY probability rises from 11.4% to 13.8%. This establishes a prediction effect, but no reset-only win rate was measured, and the probe did not compare reset effects on card or placement distributions.

Repeated 250 ms waits must not be counted as independent tactical misses. The 3,815 teacher-PLAY frames include long stalls. Pooled mode recall is 32.6%, but recall on the first frame of a maximal teacher-PLAY run is 73.5%; zero-start inference raises that first-frame value to 78.1%. These observational groupings are still correlated and teacher recommendations can change. They do not establish that most losses come from the memory mismatch.

**The PPO attempt did not repair these weaknesses.**

The deployed final actor is the initializer, update zero. The 16-update PPO run made 46,080 transitions and sixteen actor optimizer steps, with only 48 completed training matches (11 wins and 37 losses). All 30 training decks were visited, but experience per deck was shallow.

The randomly initialized critic explains little of the rollout return targets: mean incoming explained variance is 0.061, negative on five updates and -0.149 at the last update. The direct GAE residual weighting has a 2.83-second half-life at the configured 250 ms cadence; a good critic could bridge longer consequences, but this run has weak value prediction. Limited experience and delayed defensive credit are supported explanations for the failed improvement, not proof that PPO cannot work.

Fresh confirmation finds **42/150 initializer wins versus 31/150 PPO update-four wins**. PPO increases spending and cheap-card plays while reducing Hog, Musketeer and Fireball plays. It was correctly rejected by the experiment's selection rule. Numerical checks do not indicate invalid masks, an action-coordinate swap, an unfixed WAIT duration or a PPO sequence-reset error.

**The next improvement should address the demonstrated gaps before another broad algorithm comparison.**

1. Run a frozen, train-only inference ablation: carried memory versus reset-every-decision, with the same checkpoint, masks, event history, stochastic sampler and opponents. A 60-game screen over all 30 training decks can reuse its paired carried records; any promising result needs fresh confirmation before changing the deployed policy.
2. Add a public-observation siege response and test it in full matches. Prioritize the reproduced X-Bow blind spot, air-response timing and Cannon/Musketeer placement. Measure a better teacher's outcomes before spending another long fit on its labels.
3. Train on student-visited defensive states and, if retaining persistent memory, contiguous sequences with real episode resets. The current sparse retained records cannot become proper sequences merely by sorting timestamps. Verify useful recurrent gradients, keep train/dev episodes disjoint, and increase useful rare-action coverage.
4. Assess paired full-match wins, incoming combat damage and card use across every training family. Resume PPO once its initializer and value learning can support these defensive decisions; training duration alone is not evidence of improvement.

These are evidence-based priorities, not fixed assumptions about the eventual best training method. The inspected final catalog is now diagnostic evidence; subsequent claims require fresh evaluations. The 40 source-linked deck identities still use scripted controllers, so these simulator results do not establish strength against real players or every possible enemy deck.

Detailed [machine-readable summary](../../outputs/v4/initializer_coverage_20261006/loss_diagnosis_20261006/summary.json), [tactical replay notes](../../outputs/v4/initializer_coverage_20261006/loss_diagnosis_20261006/replay_notes.md), [calibration](../../outputs/v4/initializer_coverage_20261006/loss_diagnosis_20261006/calibration_notes.md), [teacher comparison](../../outputs/v4/initializer_coverage_20261006/loss_diagnosis_20261006/teacher_on_final_train/summary.json), [recurrent review](../../outputs/v4/initializer_coverage_20261006/loss_diagnosis_20261006/recurrent_review_notes.md), and [PPO audit](../../outputs/v4/initializer_coverage_20261006/loss_diagnosis_20261006/ppo_training_notes.md) preserve inputs, counts, examples and limitations.

Validation: all 800 original records were checked, 34 instrumented matches reproduce their full native record and trace, and a second historical-spell replay reproduces all 34 again. All 300 teacher games finish with legal learner actions and native trace parity. Selected actor bytes, original final protocol and 201 sealed native source files remain unchanged. The only new executions were diagnostic matches and inference; no training was performed. The default source test suite was not rerun because this diagnosis changes no executable project source; the original experiment's test outcomes remain in its results report.
