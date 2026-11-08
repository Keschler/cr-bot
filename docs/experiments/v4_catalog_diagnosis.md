# Why the frozen policies lost against the 40-deck catalog

> Historical record (2026-10-04). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The subsequent [initializer recovery and bounded PPO curve](v4_recovery_results.md)
is complete. It records the repairs, fresh fitting, paired final results,
and the proposed next experiment. The diagnosis below concerns the original
historical checkpoints.

Analysis dated 2026-10-04. The strongest observed explanation is a failure to
learn useful spending decisions. Bootstrap chose PLAY on only three of 192
regenerated training states labeled PLAY. In matches, actors cycle cheap cards,
rarely accumulate enough elixir for Hog Rider or Musketeer, and have weak
placement distributions. Their tiny PPO and distillation updates did not
repair that behavior. The expanded benchmark also uses stronger repaired
opponent controllers than the historical pilot.

This conclusion concerns the five saved pilot checkpoints. It does not show
that PPO or distillation cannot produce a strong policy with a suitable
training setup and sufficient data. No training or checkpoint selection was
performed during this diagnosis.

## Evidence from the complete evaluation

All 2,000 original records were inspected. Each policy spends approximately
85–89 elixir per game, close to its opponent, and plays approximately 46–48
cards per game. It is not simply refusing to act or failing to execute legal
actions. Bootstrap and its distilled child each destroyed zero enemy towers
over their 400 games. All records terminated with zero illegal actions and
zero truncations; the evaluation did not censor unfinished matches as losses.

The per-policy aggregates are in
`outputs/v4/catalog_comparison_20261004/diagnosis/all_records.json`.
The original results are in [v4_catalog_results.md](v4_catalog_results.md).

## Reproduced action-level failures

Additional diagnosis used the five original training-deck opponents
`hog-26-zap`, `xbow-30`, `golem-megam-lightning`, `lavaloon-lightning`, and
`valk-log-bait`. Three fresh-evaluation seeds were used for bootstrap, and the
first seed for the other comparisons. These are diagnostic samples, not a
new held-out estimate or a complete catalog behavior audit.

Across 15 bootstrap games, the actor made **749 card plays**:

| Card | Plays |
| --- | ---: |
| Skeletons | 152 |
| Ice Spirit | 151 |
| Ice Golem | 147 |
| Log | 145 |
| Cannon | 141 |
| Fireball | 8 |
| Musketeer | 4 |
| Hog Rider | 1 |

All 15 games were losses. Bootstrap had less than four elixir at **98.48% of
observed decision points**. In the first five games, Hog Rider stayed in hand
at every decision but was affordable at only 13 of 4,077 decision points.
Every affordable instance also had legal deployment cells. That directly
supports elixir starvation rather than an absent card or a broken Hog mask.
The expensive held cards leave the remaining cheap cards circulating through
the hand, which explains the repeated five-card cycle and dominant slot in
the historical PPO diagnostics.

In one five-minute Hog matchup, bootstrap spent 195 elixir and played 109
cards, yet never played Hog Rider, Musketeer, or Fireball. Its first actions
were Skeletons and Log; later actions repeatedly spent near the cost of the
next cheap card. The trace is retained in
`diagnosis/bootstrap_hog-26-zap_2026100401_sampled_actions.jsonl`.

Across the first seed's five opponents:

| Actor | Total plays | Hog Rider | Musketeer | Fireball |
| --- | ---: | ---: | ---: | ---: |
| Bootstrap | 307 | 0 | 0 | 2 |
| Bootstrap + PPO | 234 | 0 | 1 | 4 |
| Bootstrap + distillation | 307 | 0 | 0 | 2 |

At states where PLAY was legal, bootstrap's mean PLAY probability was about
0.40, and the PPO child's was about 0.28. Repeated sampling every 250 ms can
spend cheap available cards before an expensive card becomes affordable.
The overall approximately 93% WAIT rate is therefore misleading by itself:
many waits occur because the actor has already depleted elixir.

Deterministic bootstrap inference was also checked on the first seed against
all five opponents. It chose **WAIT throughout every match**, played zero
cards, and lost all five. Changing sampling to argmax is not an observed fix.
This is evidence that the fitted mode head does not distinguish sufficiently
between states that require a play and states that require saving elixir.

Placement remains a separate weakness. Mean normalized placement entropy was
about 0.83 of the legal-cell maximum in the first five stochastic bootstrap
games: the distributions remain broad, though they are not uniform. Both
bootstrap Fireballs in those games dealt no damage. These small samples do
not establish the frequency of wasted spells across all 400 games.

There is useful contrary evidence in the single full-catalog PPO win. Its
reproduced trace played Hog Rider four times and Musketeer three times,
destroyed two enemy towers, and won with crowns 2–1. The runtime can execute
the key cards and credit a player-0 win. That one trace is consistent with
the starvation diagnosis but does not by itself isolate the cause of winning.

## Bootstrap did not learn its own PLAY labels

The six-match collection was regenerated from the saved configuration without
running an optimizer. Its episode records, 6,152 decisions, retained counts,
family counts and inverse-retention weights all matched the original manifest.
The frozen actor and its original random initialization were evaluated on the
384 retained labels as isolated one-step states, matching the training setup.
This is a training-fit diagnostic, not a held-out quality estimate.

| Check on retained labels | Bootstrap | Random initialization |
| --- | ---: | ---: |
| Correct PLAY decisions / 192 PLAY labels | 3 | 0 |
| Correct WAIT decisions / 192 WAIT labels | 192 | 192 |
| Hog labels choosing PLAY / 7 | 0 | 0 |
| Conditional placement argmax within one cell / 192 | 74 | 5 |
| Mean probability in teacher's placement neighborhood | 10.7% | 3.4% |

There were only **seven Hog and five Musketeer PLAY labels** in the retained
data. On Hog labels, bootstrap's average PLAY probability was approximately
18%, lower than the random initializer's 33%. This independently demonstrates
poor mode learning even before a carried recurrent state or unfamiliar
opponent is involved. The model made partial progress on placement; it did not
become a functioning imitation of the teacher.

The conditional card accuracy was 83.3% versus the random actor's 82.3%; masks
make many choices trivial. At least 114 of the 192 PLAY examples had target-card
probability exactly one even under random weights. That aggregate accuracy
does not establish learned card-selection skill.

These findings explain why a declining total fitting loss was insufficient.
The inverse-retention mode loss restores the naturally WAIT-heavy teacher
distribution, while card and placement losses are separately scaled and
averaged. Sparse useful PLAY states require specific mode/coverage diagnostics;
changing class weights indiscriminately could worsen cheap-card dumping.
The appropriate correction must teach *when* to play and *when* to save.

Evidence and the executable probe are in
`outputs/v4/catalog_comparison_20261004/bootstrap_calibration/summary.json`,
`sample_records.jsonl`, and `evaluate_bootstrap_calibration.py`.

## Why subsequent training did not repair the failure

Bootstrap was an initializer with **384 retained labels from six matches**,
eight fitting epochs, and 48 optimizer steps. Only 192 retained labels were
PLAY examples. The six deck families were all played by the same
`deterministic-cycle` opponent strategy during collection, rather than the
current family-specific strategies
([bootstrap.py](../../simulator/rl/v4/bootstrap.py:318)).

The teacher is explicitly a small rules baseline. It attacks with Hog only
when no advanced threat exists and elixir is at least six; defensive rules
can spend affordable cheaper units first
([simulator_teacher.py](../../simulator/rl/v4/simulator_teacher.py:157)).
A same-protocol diagnostic of the teacher itself lost all five checked games.
It did play 11 Hogs, four Musketeers and 18 Fireballs, so the student learned
less useful behavior than even that weak teacher in these checks. The teacher
is not evidence of expert Hog strategy.

Bootstrap supervises isolated one-step states with fresh recurrent state,
while deployed policies carry their GRU state through a match
([bootstrap.py](../../simulator/rl/v4/bootstrap.py:476)). An existing 108-state
context probe changed only one deterministic action between reset and
32-step-context inference, with a mean PLAY-probability change of 0.01084
(`outputs/v4/audit_20261004/bootstrap_recurrence_probe.json`). This remains a
training/inference difference to examine, but evidence points to poor mode
learning even with the training-time reset convention.

Each PPO arm used **one rollout/update: 2 lanes × 1,440 decisions**, one PPO
epoch, and zero completed training wins. Their stored diagnostics show small
card-distribution changes and no card argmax changes. The bootstrap update
primarily adjusted mode probabilities and left the observed key-card
starvation in place. Negative training returns and finite gradients show
that the update received a learning signal; they do not show it learned a
useful strategy. Full-match evaluation time should not be mistaken for
training time or training data.

At 250 ms cadence, the used discount `gamma=0.99` has a reward half-life of
about 17 seconds; `gamma × lambda=0.9405` gives a roughly four-second direct
GAE weighting scale. With an initially poor critic and delayed tower rewards,
these settings could make saving elixir and coordinating a push harder to
learn. This is a hypothesis for a controlled learning-curve experiment, not
a confirmed reward-sign or implementation bug.

Distillation received **five unique accepted training corrections**, 64
retention examples, one epoch and nine optimizer steps at learning rate
`1e-5`. Its mean development KL movement was approximately `4.26e-6`, with
zero deterministic action changes in the saved local checks. Retention
deliberately preserves the parent policy. Five local corrections at that
dose did not change the parent's repeated cheap-card cycle; this is visible
in the reproduced card counts above. The procedure never had broad enough
correction coverage to demonstrate an all-deck strategy.

Training details and costs remain in
[v4_pilot_results.md](v4_pilot_results.md) and the accepted learning-curve
and distillation artifacts under `outputs/v4/pilot_20260930/`.

## Why the old pilot looked less poor

The current controllers save for their win conditions, deploy offensive
X-Bows in range, use Balloon, and support tanks in the same lane. Earlier
controllers could spend away those opportunities. A separate controller
activity probe recorded substantially greater tower HP reduction after those
fixes. Exact policy sampling conditions also changed during the evaluation
audit. The old five-deck win rates and current 40-deck rates therefore differ
in opponents and execution conditions as well as deck coverage.

No matched old-versus-new-controller policy experiment was performed here,
so the fraction of the win-rate drop attributable to each change is unknown.
The current failure occurs against the same five deck identities in the
diagnostic samples; unseen decks alone do not explain it.

## Smaller teacher and feature defects

Two concrete defects were found in the baseline labels/features:

- Spell logic detects a nearby enemy pair but places the spell at the centroid
  of **all** enemy entities, including distant ones
  ([simulator_teacher.py](../../simulator/rl/v4/simulator_teacher.py:185)). A
  synthetic three-enemy probe reproduced a target away from the nearby pair;
  its result is in `diagnosis/teacher_cluster_probe.json`. This can produce
  poor spell labels. Its impact on the 2,000 outcomes was not isolated.
- The fixed hand encoder marks Ice Spirit as not an air attacker
  ([hand_features.py](../../simulator/rl/v4/hand_features.py:35)), while the
  executed ruleset allows air and ground targets. That can prevent the
  teacher's preferred air-defense branch from considering it when a card
  flagged as an air attacker is available.

The declared `next_card_indicator` hand feature also remains zero because the
encoder does not populate it. It supplies no useful cycle signal as currently
implemented; both training and inference share that omission. These findings
do not explain the failed PLAY recall by themselves. No fixes were applied in
this analysis, so historical artifact identities remain intact.

## Work recommended at the time

First correct the confirmed teacher/feature defects if using this bootstrap
path, and establish whether the initializer learns its intended behavior:
inspect mode calibration, key-card availability/use, useful placements, and
full-match outcomes against a teacher baseline. Treat both all-WAIT argmax
and expensive-card starvation under stochastic sampling as concrete failures
to resolve, rather than accepting a declining aggregate supervised loss as
proof of a useful actor. Better teacher behavior or different initialization
is an option; bootstrap is not a required method.

Then run a bounded PPO learning curve with fresh sources under the repaired
controllers, enough successive updates to reveal a trend, and frequent cheap
diagnostics of Hog use, air defense, elixir saving and tower damage. Test the
discount/cadence and initialization choices if the behavior does not improve.
Broader search distillation becomes a meaningful competitor when it produces
enough validated corrections to change these behaviors and whole-match results.
Use the 30 training decks for development; keep the ten reserved decks for a
subsequent fixed comparison.

Another large evaluation of unchanged actors would not resolve the training
failure. Nor do these results justify choosing a winner between PPO and
distillation. The goal remains a strong Hog 2.6 policy against all enemy decks.

## Verification and limits

Diagnosis inspected all 2,000 existing records and ran 41 policy/baseline
diagnostic matches plus six bootstrap collection replays. **26 instrumented
stochastic matches reproduced their original
complete records exactly**, including action trace, final state and event
hashes. Fifteen additional checks compared deterministic bootstrap, the
teacher and a scripted Hog baseline on training decks. The latter won one of
five checks; it is a sanity check, not a strong policy or a fair learned-method
comparison.

Artifacts are under `outputs/v4/catalog_comparison_20261004/diagnosis/` and
`diagnosis_additional/`, including `summary.json`, `probe_summary.json`,
`probes.json`, complete per-decision action logs and `hand_replay.json`.
No runtime source or saved checkpoint was modified. No training was performed.
The default test suite was not rerun for this analysis-only work; the preceding
evaluation's verification remains 1,313 passed, 14 skipped and three existing
failures. Simulator live-game fidelity and side-swap checks remain limitations,
but no evidence currently identifies them as the main cause of these losses.
