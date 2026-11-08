# Human-strength strategy implementation and local results

> Historical record (2026-10-07–2026-10-18). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The [strategy](../design/policy-learning.md) is implemented as opt-in local
training and evaluation tools. It has not produced a newly selected actor.
Human-data training remains disabled: no curated expert demonstrations are
available. The goal of strong play against all decks and human styles remains
open.

## Implemented capabilities

- An air-cycle teacher that preserves a four-elixir reserve while cycling
  toward an answer to a public air threat.
- Three training opponent styles and two held-out development styles, with
  explicit parameters, seeds and identities; historical defaults are preserved.
- Full 30-training-deck × three-style opening generation, plus observed
  per-entry training visits and completed outcomes.
- Pinned historical learned-opponent populations, independent per-match
  memory and stochastic sampling, train-only feedback and member coverage.
  Learned opponents currently support Hog mirrors only.
- Explicit frozen-initializer retention in match-start PPO, and an optional
  critic-only warmup whose failed validation gate prevents actor updates.
- Paired complete-match screens with source/checkpoint hashes, exact card
  acceptance, resumable sealed records and family safeguards.

## October 7 evidence: engine 0.37.0

The first screen used all 30 training decks and two declared seeds per deck,
with identical opponents and initial states for both teachers.

| Teacher | All games | Lavaloon | X-Bow | Bait |
| --- | ---: | ---: | ---: | ---: |
| Defense | 30/60 | 2/12 | 10/12 | 3/12 |
| Cycle | 36/60 | 8/12 | 10/12 | 3/12 |

All six changed outcomes were additional cycle wins against Lavaloon; there
were no unfavorable pairs. Every other family was unchanged. The screen
passed its predeclared gate. Ground-only air-threat waiting fell from 183.25
to 86.25 seconds in total; the longest such interval fell from 25.5 to seven
seconds. The new rule made 33 rescue plays. These are teacher results, not
results from a trained neural actor.

The selected neural actor was also screened against distinct opponent styles:

| Style split | Style | Actor wins |
| --- | --- | ---: |
| Train | Tempo | 5/30 |
| Train | Patient | 6/30 |
| Train | Counterpush | 7/30 |
| Development | Adaptive | 5/30 |
| Development | Split-pressure | 8/30 |

That is 18/90 training-style wins and 13/60 held-out-style wins. Across these
150 games it won **0/30 Lavaloon** and **1/25 Golem** matches. These results
reinforce the air/beatdown weakness. One seed per deck/style is descriptive
evidence; these scores must not be treated as a causal comparison with an
older benchmark using different opponents or seeds.

All 270 teacher/style matches completed genuinely, with zero illegal learner
actions and no opponent warnings. Independent audits verified exact grids,
native contracts, per-card acceptance, paired identities and raw-result seals.
They remain historical evidence on the sealed October 7 source. The planned
cold replays were not completed before the source changed; no old-engine
replay on the current engine is claimed.

## Critic-only diagnostic

The warmup used disjoint training-catalog match origins, eight critic epochs,
and no actor updates. There were 90 available deck/style openings, split into
45 train and 45 validation origins. Actual collection visited only 11 train
and nine validation origins; seven and six episodes completed respectively.
Unfinished tails received no supervised terminal target.

| Measurement | Before | After |
| --- | ---: | ---: |
| Train return MSE | 0.26178 | 0.02208 |
| Validation return MSE | 0.54820 | 0.82997 |
| Validation explained variance | −0.08936 | −0.09300 |

All seven completed training episodes were losses. Validation had two wins
and four losses. The critic fit its small training sample but generalized
worse. It failed the outcome-coverage, validation-variance and validation-MSE
gates. The learning curve contains **update zero only**, and all 56 saved
actor tensors are bitwise equal to the selected input actor. Different actor
archive hashes result from serialization; there was no parameter update.

This rejects scaling this particular critic configuration and sample. It does
not establish that PPO cannot work. A further value experiment needs broader
completed-game coverage that includes wins and losses, followed by independent
return prediction before actor training.

The generated learned-opponent pool contains 30 Hog-mirror openings across
three historical snapshots: initializer 10, saving 13 and defense seven.
All members' exact architectures and checkpoint hashes were verified.
Those are generated starts, not 30 completed training matches.

## October 18 source transition

The resumed workspace includes engine 0.37.1, which lets spirits advance
toward Crown Towers. This affects learner Ice Spirit and enemy spirits, so
the old screen cannot establish improvement on the new engine. The transition
audit recovered the five changed historical source files and matched their
old hashes. It also verified that the old results remained sealed.

The repeated current-engine screen again passed: defense **30/60** versus
cycle **36/60**, including **3/12 versus 9/12 Lavaloon wins**. All other family
win totals were unchanged. Individual pairs switched seven times in favor of
cycle and once against it, so identical aggregate scores across versions do
not imply identical trajectories. An independent audit reconstructed all 60
openings and verified all 120 records. Two fresh-process cold replays matched
the saved native records and diagnostics exactly.

The unchanged neural actor's current-engine style screen produced:

| Style | Actor wins |
| --- | ---: |
| Tempo | 9/30 |
| Patient | 5/30 |
| Counterpush | 6/30 |
| Adaptive (development) | 4/30 |
| Split-pressure (development) | 6/30 |

Totals are **20/90 training-style wins** and **10/60 held-out-style wins**.
Lavaloon remains **1/30**, Golem **1/25**, and Giant **0/5**. All 150 matches
completed legally, without opponent warnings; the audit and one separate
process cold replay passed. The engine update is not a new policy-training
result, and no actor selection uses these version-to-version differences.

The eligible screen triggered the five-fresh-seed, 300-game confirmation:

| Teacher | All games | Lavaloon | X-Bow | Bait |
| --- | ---: | ---: | ---: | ---: |
| Defense | 93/150 | 16/30 | 24/30 | 13/30 |
| Cycle | 96/150 | 19/30 | 24/30 | 13/30 |

The **+3 net wins** meet the predeclared threshold exactly. All net gain was
against Lavaloon; every other family total was unchanged. There were five
favorable and two unfavorable paired switches. All matches completed with
zero illegal learner actions or opponent warnings. The independent audit
reconstructed all 150 openings, verified all 300 raw records, and a fresh
process cold replay reproduced its native record and diagnostics exactly.

Seed-level net gains were **+2, +2, −1, +1, −1**. An exploratory paired
seed-cluster bootstrap interval spans **−2 to +6 percentage points**; five
clusters give limited uncertainty resolution. The engineering gate passes,
but this is a modest, uncertain teacher gain rather than strong statistical
proof. Air-threat waiting fell from 437 to 169.5 seconds, with 50 rescue plays.
Neither that duration change nor teacher wins establish student improvement.

The cycle teacher is eligible for a controlled targeted student fit. The next
fit protocol is saved as
`outputs/v4/human_strength_20261018/targeted_student_protocol.md`: supervise
public air/X-Bow contexts, retain the initializer's full masked action
distribution elsewhere, unroll both recurrent states chronologically, and
evaluate a fixed final checkpoint against fresh paired games with family
safeguards. It also records why copying the previous broad teacher fit would
repeat known risks. That fit has **not** been launched in this bounded
implementation stage. Its estimated collection/fit/evaluation cost is
8–15 / 45–75 / 10–18 minutes on this machine; these are estimates, not timings
from a new student run.

Current-engine results and the final completion receipt are recorded under
`outputs/v4/human_strength_20261018/`. Original results remain under
`outputs/v4/human_strength_20261007/`.

## Artifacts and validation

The strategy document contains executable CLI examples. Each output directory
retains its plan, exact command arrays, source seal, raw matches, summaries,
audit receipts and measured costs. The October 7 stages ran concurrently;
their individual wall times cannot be summed as total elapsed time.

The current focused suite passes **361 tests, with seven skips**, including the new
teacher, styles, populations, critic targets/gates, CLI integration, exact
resume contracts and historical behavior. The earlier full default suite had
1,737 passes and 44 skips. Five socket-restricted multiprocessing failures
passed on rerun outside that restriction; three unrelated failures remained:
annotation matching, missing spell video media, and KataCR constructor
compatibility. Current-engine full-suite results are retained in the October
18 output directory. That full run had **1,806 passes, 45 skips and the same
eight failures**. The five local forkserver-socket failures again passed when
rerun outside the socket restriction. The same three unrelated failures
remain; no relevant new test failed.

All **570 current-engine teacher/style games** are complete and audited.
No new actor has qualified for reserved-deck evaluation or automatic
promotion. A targeted student fit, multi-deck learned adversaries, curated
human demonstrations and real-game transfer validation remain outstanding.
