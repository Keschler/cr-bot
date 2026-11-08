# Hog 2.6 memory and defense repair results

> Historical record (2026-10-06). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The experiment retained its starting actor with carried memory. The new defense-label actor won **48/150 fresh training-deck
games (32%)**, compared with **36/150 (24%)** for that baseline. X-Bow improved
substantially, but Bait regressed enough to fail the safeguard declared before
training. Neither candidate qualified, so the conditional final 40-deck
evaluation was not started. The goal of a very strong Hog 2.6 policy against all
enemy decks remains open.

This follows the [remaining-loss diagnosis](v4_remaining_losses_diagnosis.md).
Outputs and receipts are in
[`outputs/v4/defense_repair_20261006/`](../../outputs/v4/defense_repair_20261006).
No further experiment is queued.

## Fresh learned-policy confirmation

All three actors used the same 30 training deck identities and five fresh seeds,
`2026100741` through `2026100745`: **450 complete native matches**. Every actor
used carried recurrent state, native stochastic sampling and fixed 250 ms
decisions. The ten reserved deck identities were excluded from data collection,
training and checkpoint selection.

| Actual deck family | Games per actor | Frozen baseline wins | Old-label sequence wins | Defense-label sequence wins | Defense minus baseline |
|---|---:|---:|---:|---:|---:|
| Hog | 15 | 10 | 12 | 9 | -1 |
| P.E.K.K.A. bridge | 15 | 2 | 1 | 3 | +1 |
| Giant | 5 | 1 | 2 | 1 | 0 |
| Golem | 25 | 3 | 6 | 8 | +5 |
| X-Bow | 30 | 4 | 6 | 15 | +11 |
| Lavaloon | 30 | 3 | 2 | 3 | 0 |
| Bait | 30 | 13 | 6 | 9 | -4 |
| **Total** | **150** | **36** | **35** | **48** | **+12** |

There were no draws. The defense candidate has 34 favorable and 22 unfavorable
paired switches versus the baseline. Its eight percentage point gain is
promising, but uncertainty remains: paired 95% bootstrap intervals for score
gain are -2 to +18 points when resampling decks and -5.3 to +22 points when
resampling the five seed clusters. These are diagnostics, not proof of general
strength. Each deck receives the same number of games; families have different
numbers of catalog decks.

The predeclared eligibility rule required at least three net wins-equivalent
overall, a positive combined X-Bow/Lavaloon gain, complete legal execution, and
no actual family losing three or more net wins-equivalent. The defense candidate
passes the overall and target checks but fails Bait at -4. The old-label control
fails both overall gain (-1) and Bait (-7). The rule was not relaxed after seeing
these outcomes.

Changing the labels under the common training protocol adds **13 wins** versus
the old-label control, including X-Bow +9, Lavaloon +1 and Bait +3. That supports
the value of the revised labels. Both retrained actors regress on Bait versus
the starting actor, so the shared collection, fitting dose or objective is also
a plausible contributor. This experiment does not isolate those factors.
`valk-log-bait` loses three wins in both arms; it is a useful first replay target.

All 450 learner games reach genuine terminal states with zero invalid actions,
rejected plays or missed plays. Native and diagnostic decisions, action traces,
requested/accepted cards and WAIT counts agree. Paired initial states,
opponents, controller seeds, initial policy seeds and runtime contracts agree.
Matching initial RNG seeds does not imply matching trajectories after actions
diverge.

Two opponent warnings are retained under the declared primary policy: baseline
`0000_hog-26-zap_seed2026100741`, and old-label control
`0042_xbow-icewiz-rocket_seed2026100743`, each with one opponent invalid action.
Removing the union of those two cells from every actor gives baseline 35/148,
old-label 34/148 and defense 48/148. Defense gains 13 instead of 12, but Bait
remains -4; the selection decision is unchanged. No opponent controller was
changed.

## Memory and teacher experiments

| Memory comparison | Carry wins | Reset every decision wins | Interpretation |
|---|---:|---:|---|
| Old-seed screen, 30 decks | 11/60 | 17/60 | Justified fresh confirmation |
| Fresh confirmation, 30 decks | 41/150 | 38/150 | Retain carry |

Reset zeros the actor's recurrent state at every decision while preserving
public estimator history. The favorable screen did not repeat on fresh seeds:
reset has 16 favorable and 19 unfavorable switches. One reset opponent-warning
cell is a win for both policies; symmetric exclusion gives 40/149 versus 37/149
and leaves the -3 difference unchanged. This does not establish that memory
reset is universally worse.

The additive experimental `defense_teacher_label` uses public hand/entity
tokens, own elixir and legal-action masks. It responds to potential X-Bow tower
reach before the old forward-threat cutoff, keeps urgent defense ahead of
remote siege, answers air earlier, and adjusts Musketeer/Cannon placement and
selected spell priorities. Historical `teacher_label` and
`saving_teacher_label` code remains unchanged.

On a fresh two-seed screen over all 30 training decks, the old saving teacher
wins **19/60**, the defense teacher **35/60**, and the frozen student **18/60**.
X-Bow improves from **1/12 to 10/12**, while Lavaloon changes from **5/12 to
4/12**. The teacher passes the declared combined-target capability gate mainly
through siege defense. All 180 games are complete with zero invalid actions for
either player. Air improvement was not established by this screen.

## Matched sequence training

The collector retains complete 250 ms public histories for 90 fresh episodes:
60 train and 30 dev episodes on training deck identities only. The drivers are
the frozen stochastic student and the defense teacher. Train/dev match seeds
are unique and disjoint. Both teachers label the same observations.

There are **72,887 public frames** and **17,143 selected loss positions**:
47,710/11,360 train and 25,177/5,783 dev. Selection includes either teacher's
PLAY rows and systematic WAIT rows with shared inclusion weights. Every
intervening frame advances recurrent history; countdown placeholders and
padding receive no fabricated label loss. All 90 artifacts pass independent
hash, public-input, label, legality, cadence, reset, termination and card-parity
checks.

Both fits start from the exact same frozen actor and use the same data, seeded
episode order, Adam learning rate 0.0003, 32-step chunks, two episode lanes and
eight passes. The fixed dose was chosen from timing before candidate match
outcomes existed. Both finish **381,680 training frames, 90,880 loss positions,
240 episode groups and 6,676 optimizer updates**. State resets at real episode
starts and carries between chunks; gradients detach at chunk boundaries. This
is ordinary online truncated backpropagation, not prefix recomputation after
each optimizer update. The historical duration-head auxiliary objective is
shared; deployment still executes fixed WAIT.

Unlike the old isolated-observation fit, both new fits have finite nonzero
hidden-to-hidden gradients and actual recurrent weight movement. Checkpoints
and complete training schedules passed independent audit. The final epoch-eight
actors were the only candidates; dev diagnostics did not select an epoch.

Joint PLAY mode, exact card and placement within one legal cell of the defense
teacher on the same dev histories:

| Actor | Overall | Saved forward-air cohort | X-Bow visible |
|---|---:|---:|---:|
| Frozen baseline | 22.4% | 14.9% | 3.0% |
| Old-label sequence | 43.5% | 26.6% | 8.9% |
| Defense-label sequence | 46.8% | 31.4% | 27.7% |

These are agreement diagnostics on correlated recorded states, not win rates
or estimates of optimal play. The saved air flag retains the old y > 0.45
definition and omits early-only air frames newly addressed by the teacher.
Visible X-Bow does not establish tower lock. Better imitation by itself did not
make the old-label sequence control stronger in fresh matches.

## Preserved source-check failure and explicit recovery

During confirmation, concurrent edits changed two phone-capture files:
`simulator/physical_lab/workflow/prototype_controller.py` and its test. The
original evaluator correctly failed its broad 202-file source check **after
saving all 450 games**. Its failure log, cost receipt, source seal, execution
helpers, original manifest and raw records are preserved. The user edits were
preserved too. Broad source stability is not claimed.

A separate dependency audit found all other 200 source files and all seven
original helpers unchanged. Its conservative 92-file static/lazy import closure
contains neither changed file and no unresolved dynamic imports. Three cold
consistency replays, selected lexicographically without filtering outcomes,
reproduce the entire saved native record for a baseline Hog game, an old-label
Lavaloon game and a defense-label X-Bow game. Loaded project modules stay inside
the unchanged closure; no physical-lab module is imported. These three replays
are not added to the statistical sample.

Separate versioned analysis accepts exactly that two-file non-runtime
exception. It verifies the full source file set, unchanged reachable hashes,
all original helpers, replay receipts and raw execution contracts. Its summary
is explicitly `complete_with_nonruntime_source_change`. The new decision helper
changes only recovery paths/status and source-boundary validation; reversing
its six recorded substitutions reproduces the sealed original byte for byte.
Eligibility, ranking, warning policy and raw validation are unchanged. An
altered runtime dependency proof is rejected.

Independent workers audited the complete raw grid, outcome rule, fits and source
boundary. The coordinator reviewed the new recovery helper and performed a
separate score/family recount. The reviewer reached its usage limit before a
second review of that helper; this limitation is recorded in `final_review.json`.
The original selection rule and complete independent raw/source audits already
support retaining the baseline.

## Artifacts, cost and proposed follow-up

The actor retained by this experiment was
`outputs/v4/initializer_coverage_20261006/fit_saving/actor_epoch_128.pt`, SHA256
`ba15ded00742ec68b509e89da8ad88bcc12b0e0c7bf2f76d8edd25e869903c3b`, with carried
memory. New candidates remain available for research:

- Old-label sequence: `fit_saving_v1/actor_epoch_008.pt`, SHA256
  `20c78a3fdd05b3995c2c3aadcd3945540cd796282ceccfd63bb510c21f0c27c9`.
- Defense-label sequence: `fit_defense_v1/actor_epoch_008.pt`, SHA256
  `1889b1acc5acb152cef946fed05ef2c1ef493fdd4f2a1fce00e378343004c923`.

Actual local CPU stage wall times are 12.8 minutes for fresh memory confirmation,
5.6 minutes for the teacher screen, 5.0 minutes for collection including its
pilot, about 31 minutes per fit run in parallel, and 33.3 minutes for the
450-game confirmation. Recorded monotonic costs, rather than timestamp
differences, are authoritative. No GPU was used.

The next focused experiment should first replay the common Bait regressions and
preserve those defenses while testing the documented air-cycle rescue. Selected
teacher-driver collection losses show long waits with a ground-only hand even
at ten elixir. A cheap cycle that retains four elixir, followed by normal air
defense, is a concrete candidate. A second candidate is Fireball priority for
killable Skeleton Dragon pairs. Both are **unapplied proposals**, supported by
public trajectories rather than a causal claim about the 450 confirmation games;
see [`air_review.md`](../../outputs/v4/defense_repair_20261006/air_review.md). This
result does not justify repeating a broad algorithm comparison or extending
training without testing the failure mechanism.

The catalog contains 40 source-linked deck identities played by five scripted
controller strategies, not real-player policies or every possible enemy deck.
This confirmation covers its 30 training identities only. The prior 92/400
catalog result still describes the retained actor; 48/150 is a new training-grid
candidate result and must not be presented as a new 40-deck win rate.

Validation: focused teacher checks **63 passed**; sequence helper checks **4
passed**. The default suite produced 1,387 passed, 14 skipped and eight failures.
Five local-IPC sandbox failures passed on rerun with process access. Three
unrelated failures remain: annotation maximum-cardinality scoring, absent spell
evaluation media and KataCR tracker-constructor import. The default run preceded
the concurrent capture edits; those edits were outside this task. Detailed logs
and counts are in `validation.json`, `default_tests.log` and `ipc_tests.log`.

Authoritative local receipts:
[completion](../../outputs/v4/defense_repair_20261006/completion_receipt.json),
[final decision](../../outputs/v4/defense_repair_20261006/final_decision.json),
[recovered native summary](../../outputs/v4/defense_repair_20261006/training_confirmation_recovery_v1/summary.json),
[original recommended actor](../../outputs/v4/promotions/20261020_targeted_control/previous_selected_actor.json),
[source boundary](../../outputs/v4/defense_repair_20261006/source_boundary_audit.json),
[exact replay receipt](../../outputs/v4/defense_repair_20261006/source_boundary_replay.json),
and [review notes](../../outputs/v4/defense_repair_20261006/review_notes.md).
