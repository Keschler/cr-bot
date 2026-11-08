# Hog 2.6 initialization recovery and PPO curve

> Historical record (2026-10-04–2026-10-06). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

Experiment started on 2026-10-04 and resumed on 2026-10-05. Status:
complete, including all 800 final catalog games. The final stage completed
on 2026-10-05 at 22:36:59 UTC (2026-10-06 in Europe/Berlin).

The recovered initializer won 14/400 final games, compared with 1/400 for
the historical bootstrap under the same repaired runtime. Reward-only PPO
did not improve the development curve, so update zero was selected. This is
a modest recovery from a weak baseline, not a strong Hog 2.6 policy.

This is an experiment toward a very strong Hog 2.6 policy against all enemy decks;
its method, budget, and catalog are experimental choices rather than new
requirements for the strategic goal.

The [failure diagnosis](v4_catalog_diagnosis.md) found poor PLAY/WAIT learning,
elixir starvation, weak placements, and very small subsequent training doses.
This recovery tests whether improving the initializer's fit and running more
PPO updates improves complete matches. It does not repeat the three-arm
method comparison or establish that bootstrap is the best initialization.

## Corrections and initializer fitting

Two demonstrated defects were corrected, with regression tests:

- The rules teacher targets the densest local enemy group when casting a
  cluster spell. A distant enemy no longer pulls the target away from the
  group that triggered the spell decision.
- Ice Spirit's hand feature identifies its ability to attack air targets,
  matching the executed ruleset.

Fresh teacher collection produced 1,152 retained training examples from 18
complete matches and 384 development examples from six complete matches.
Collection seeds were 1004 and 2004. State identities and episode/seed
origins are disjoint. The mode loss retained its inverse-retention weights;
no PLAY frequency override or Hog-specific reward was introduced.

The compact actor was fitted for up to 128 epochs, with batch size 64 and
Adam learning rate `3e-4`: 2,304 optimizer steps. Collection and fitting took
5.38 minutes on the existing CPU machine. Runtime source hashes were stable.

| Fitting epochs | Development PLAY recall | Development WAIT specificity | Mean teacher placement-region probability |
| --- | ---: | ---: | ---: |
| 8 | 13.5% | 99.5% | 34.5% |
| 32 | 53.6% | 93.8% | 54.5% |
| 64 | 66.1% | 92.2% | 55.0% |
| 128 | 84.9% | 82.8% | 61.0% |

These are isolated-state teacher-agreement measurements, not match strength.
The development set has only four Hog PLAY labels; training has 18 Hog and
six Musketeer PLAY labels. The teacher itself remains a weak rules baseline.

## Initializer match screen

Each of the 32-, 64-, and 128-epoch actors played one stochastic match against
each of the 30 training catalog decks, seed 2026101701. Selection excluded
the ten reserved decks. The predeclared rule maximizes completed match score,
then development PLAY/WAIT balanced accuracy, then prefers the lower epoch.

| Epoch | Wins / completed games | Hog plays | Musketeer plays | Fireball plays | Decision points below four elixir |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 0 / 30 | 40 | 19 | 45 | 89.8% |
| 64 | 1 / 30 | 48 | 23 | 38 | 91.3% |
| 128 | 1 / 30 | 31 | 27 | 35 | 92.3% |

All 90 games completed. The tie-break selects epoch 128; this is a recorded
choice, not evidence that it is reliably stronger than epoch 64. Key cards
are used more often than in the historical diagnosis, but frequent low
elixir and low win rates remain. Better fitting has not produced a strong
policy.

A separate paired inference probe used the same 30 training decks and fresh
seed 2026101702, without entering checkpoint selection. Deterministic
inference lost all 30 games; stochastic inference won one. Deterministic
versus stochastic card counts were 41 versus 36 Hogs and seven versus six
Musketeers. Respectively, 91.0% and 92.3% of decision points were below four
elixir. Removing sampling did not resolve the weak spending behavior in this
one-seed diagnostic. Both modes' action traces matched the native evaluator,
with no illegal actions or unfinished matches.

## Bounded PPO and final evaluation protocol

Reward-only PPO starts from the selected initializer and uses 30 fresh raw
match openings, one per training catalog deck, pool seed 2026101401.
Each update collects two lanes of 1,440 decisions and performs one actor and
one critic optimizer step. All 16 planned updates completed, reported at
updates 0, 4, 8, 12, and 16. Initial source pairs rotate across the 30 decks;
actual reset identities are recorded and checked.

Actor and critic learning rates are `1e-4`, gamma is 0.99, GAE lambda is 0.95,
and the reward combines stepwise tower-HP changes with the simulator's
terminal outcome. Behavior-cloning weight is zero. Native update diagnostics,
quarantine, and rollback remain enabled. A rejected update is not exported
as an accepted policy. All 16 updates were accepted, totaling 46,080
transitions. Actual source identities cover all 30 training decks, with
every initial-pair rotation check passing. Training recorded three wins and
51 losses across 79 episode starts; episode starts are not completed games.

The native objective also uses clipping 0.20, value coefficient 0.50, entropy
coefficient 0.01, value clipping 0.20, advantage normalization, and total-norm
gradient clipping at 1.0. These implicit defaults and the executed guard
source hashes are recorded in `native_objective.json`; no objective change
was made while the curve ran.

Checkpoint selection uses completed stochastic games against the 30 training
decks with fresh seeds 2026101501 and 2026101502. The native rule selects the
largest completed match score, breaking ties by lower training cost and then
lower update. Reserved-deck results do not enter selection. The exact
baseline/update-zero duplicate evaluation is reused only after its actor
tensors, model configuration, opponents, seeds, and execution conditions
match.

| Accepted PPO updates | Development wins / completed games | Losses |
| --- | ---: | ---: |
| 0 | 1 / 60 | 59 |
| 4 | 1 / 60 | 59 |
| 8 | 1 / 60 | 59 |
| 12 | 0 / 60 | 60 |
| 16 | 0 / 60 | 60 |

All 300 development games completed, with no draws, illegal actions, or
truncations. The predeclared tie-break selects update zero. Its 56 actor
state tensors are exactly equal to the recovered 128-epoch initializer;
the checkpoint files differ only in serialization. This bounded PPO curve
did not improve development outcomes. It does not establish that PPO is
incapable of learning or choose a winner among training methods.

The critic's explained variance was negative on 15 of 16 updates, ranged
from -0.241 to 0.024, and ended at -0.123. Return prediction remained weak,
despite the updates passing native guards. Mean advantages were negative
throughout. Conditional card, mode, and placement entropies declined from
0.0309 to 0.0061, 0.0318 to 0.0136, and 0.591 to 0.253, respectively;
these are measured on different on-policy batches, not one fixed probe.
Rollout WAIT fractions remained around 93–94% at the 250 ms cadence.
Placement-to-mode gradient warnings occurred at updates 10 and 13, without
triggering quarantine. Guard acceptance establishes bounded updates, not
effective learning or healthy exploration.

The curve took approximately 41.7 minutes on CPU. A server restart
interrupted its cost supervisor, so exact total CPU time and peak memory
are unavailable; completed curve, source, and input checks are preserved.

After selection, the frozen selected actor and historical bootstrap actor
are evaluated against all 40 decks with ten fresh seeds, 2026101601 through
2026101610: 400 games per actor, 800 total. The protocol would add the
recovered initializer if a positive PPO update were selected; update zero
already has those weights, so a duplicate third evaluation is unnecessary.
Truncations remain separate from losses. Both actors run under the same
repaired runtime, so their paired comparison measures the recovery bundle,
not a causal separation of extra fitting from the two semantic fixes.

The catalog contains source-linked base-card deck identities played by
scripted controllers. It does not establish performance against human
players, every possible deck, or unsupported special card forms.

## Final 40-deck results

| Frozen actor | All 40 decks | 30 training decks | 10 reserved decks |
| --- | ---: | ---: | ---: |
| Recovered initializer, selected update 0 | 14 / 400 (3.50%) | 10 / 300 (3.33%) | 4 / 100 (4.00%) |
| Historical bootstrap | 1 / 400 (0.25%) | 1 / 300 (0.33%) | 0 / 100 (0.00%) |

Counts are wins divided by completed games. All 800 games completed, with
no draws, truncations, invalid actions, play misses, or rejected plays.
Every instrumented action trace matches the native evaluator. The raw
deck/seed grid, actor configuration, stochastic mode, decision cap, catalog,
and source start/end fingerprints were validated before summarizing.
An independent raw-record audit reproduced the paired results and intervals;
its completion record is `final_audit.json`.

The paired overall improvement is 3.25 percentage points: 13 conditions
changed from a historical loss to a recovered win, none changed in the
opposite direction, and one condition was a shared win. A 20,000-resample
95% bootstrap interval is +1.75 to +4.75 points when resampling the ten
shared match seeds, and +0.75 to +6.75 points when resampling the 40 decks.
These are sensitivity estimates for this fixed catalog, not evidence of
performance across every enemy deck. The reserved-deck delta is +4 points;
its intervals include zero (seed: 0 to +8; deck: 0 to +10 points), so the
four reserved wins do not establish a reliable generalization gain.

The recovered actor won against only six deck identities:

| Deck | Wins / games | Split |
| --- | ---: | --- |
| `hog-mini-pekka` | 6 / 10 | Training |
| `hog-26-zap` | 2 / 10 | Training |
| `hog-icegolem-snowball` | 2 / 10 | Reserved |
| `golem-guards-fireball` | 2 / 10 | Reserved |
| `golem-clone` | 1 / 10 | Training |
| `pekka-poison-snowball` | 1 / 10 | Training |

The historical bootstrap's sole win was also against `hog-26-zap`. The
recovered actor lost every tested seed against the other 34 deck identities.

## Remaining behavior and follow-up proposed at the time

| Final-match telemetry | Recovered initializer | Historical bootstrap |
| --- | ---: | ---: |
| Mean elixir at decision points | 1.728 | 1.541 |
| Decision points below four elixir | 93.40% | 97.40% |
| Accepted Hog plays / 400 games | 576 | 162 |
| Accepted Fireball plays / 400 games | 562 | 155 |
| Accepted Musketeer plays / 400 games | 135 | 177 |

The recovery increased Hog and Fireball use and reduced low-elixir exposure,
but spending still usually prevents expensive decisions. Hog remained in
hand at 86.36% of recovered-policy decision points and was affordable at
only 5.26% of the points where it was held. Musketeer use did not improve.
All Hog attempts were accepted, so missed or illegal actions do not explain
the remaining weakness. The stored aggregate `hog_attempt_conversion` sums
per-game ratios; it is not an aggregate probability and is not used here.

The evidence favors a targeted initializer-data experiment before a larger
PPO investment. Collect state-diverse examples of useful Hog attacks,
Musketeer defenses, Fireball targets, and WAIT decisions that preserve
elixir, using only training decks. Audit label quality first: the current
rules teacher is weak, and merely duplicating cheap-card PLAY labels could
reinforce the observed spending. Keep the model, eligibility weighting,
PPO settings, and matched development protocol fixed to measure the data
change. Compare the new update-zero policy with this initializer, then
check whether PPO adds match strength.

This is a proposed next experiment, not an executed run or a required final
training method. If improved key-card and saving behavior fails to improve
matched games, data coverage was not sufficient. A separate longer-credit
arm (`gamma=0.997`, `gae_lambda=0.99`) would then test the credit-assignment
hypothesis: the current direct GAE weighting has an approximately 4.1-second
e-fold at the 250 ms cadence, versus about 19 seconds for that proposed pair.
Short credit and weak value fitting are plausible contributors, not proven
causes. This recovery does not rank PPO against distillation; distillation
was not rerun here, and the earlier tiny training doses remain inconclusive.

![Observed development and final win rates](../../outputs/v4/recovery_20261004/recovery_win_rates.png)

## Measured cost

Collection and fitting took 5.38 minutes. The 16-update PPO curve took
approximately 41.7 minutes, with the cost-monitor limitation described above.
The final 800-game evaluation took 28.44 minutes of monotonic wall time and
85.05 aggregate CPU minutes on three workers. Sampled process-group RSS
peaked at about 998 MiB; shared pages may be counted more than once.
No GPU was used. Pause time is not training time, and the supervisor's
monotonic duration is used rather than subtracting UTC timestamps.

## Artifacts and validation

All experiment artifacts are local under `outputs/v4/recovery_20261004/`:
`execution_plan.json`, `fit_natural/report.json`, `initializer_selection.json`,
the resumable raw initializer matches, `ppo_curve/learning_curve.json`,
`selection.json`, `ppo_diagnostic_summary.json`, `zero_checkpoint_identity.json`,
`ppo_curve_completion.json`, `final_catalog/recovery_summary.json`, the 800
raw final matches, stage cost files, and the generated run/evaluation helpers.
Source, input actor, catalog, and helper hashes accompany the runs.
No model promotion or commit was performed.

`run_state.json` records completion, with no remaining experiment processes.
`continuation_plan.json` records the proposed next experiment; it has not
been started. The older pause files preserve the historical stopping point.

The checkpoint selected in this experiment was
`outputs/v4/recovery_20261004/ppo_curve/checkpoint_update_000000.pt`, SHA-256
`4a54afbc84a5240fcb3e896b5cccf78e8a3760ff2aa32a95f34492849320ffd1`.
The raw-match manifest and completed report share source SHA-256
`56d300d530a5664f4f9b9142153b20dc110a150f911ac2a4ca5640b541ac236a`
and catalog SHA-256
`cae33cab0a16e82868b05f093f33e96b2bf23d27287fa68dd6efc1621642d708`.

Focused checks passed again after resumption: 39 tests, two CUDA-dependent
skips, with the log saved as `validation/focused_resumed.log`. The pre-pause
default suite result was 1,315 passing tests, 14 skips, and three previously
observed unrelated failures: annotation matching, missing spell-evaluation
media, and the KataCR tracker constructor. Multiprocessing checks required
local sockets; that full suite was rerun with those enabled. Its counts are
recorded in `pause_state.json`; the original temporary full-suite log was
lost during the server restart. Core source has not changed since those
checks, and unrelated failures were not repaired as part of this experiment.
