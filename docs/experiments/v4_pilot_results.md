# Bounded Hog 2.6 pilot results

> Historical record (2026-10-01). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The pilot completed after the user resumed PPO and distillation on
2026-10-01. All three paths produced exported, evaluated actors. None showed
an established policy improvement at this small training budget. The goal
remains a very good Hog 2.6 policy against all enemy decks; the pilot does
not achieve that goal or identify the best training method.

Local artifacts are preserved under `outputs/v4/pilot_20260930/`, including
`report.json`, `comparison.json`, exact commands, source/checkpoint manifests,
source matches, verdicts, exports, logs, measured costs, and stop/resume
records. These are generated artifacts, not committed datasets.

## Policy outcomes

Each actor was evaluated on the same five training-deck opponents and ten
fresh match seeds: 50 stochastic complete games. No draws or truncations
occurred. No reserved evaluation decks or sealed pool were used.

| Actor | Wins / 50 | Win rate |
| --- | ---: | ---: |
| Random initialization | 4 | 8% |
| Random initialization + one PPO update | 3 | 6% |
| Rules-teacher bootstrap | 6 | 12% |
| Bootstrap + one PPO update | 4 | 8% |
| Bootstrap + one distillation epoch | 4 | 8% |

The paired change from random-start PPO was −2 percentage points (match-seed
clustered standard error 2 points). Bootstrap-start PPO and distillation
each changed by −4 points relative to bootstrap (standard error 2.67
points). Those differences do not establish a ranking. There is only one
training seed, five scripted opponent decks, one PPO update per arm, and
one distillation epoch. These are exploratory development results, not
evidence of live-game or all-deck strength.

## Measured runtime on the local machine

| Workload | Wall time |
| --- | ---: |
| Bootstrap collection and eight fitting epochs | 31.6 s |
| Bootstrap PPO, including before/after evaluation | 16.5 min |
| Random PPO attempt, including baseline evaluation, then rejected update | 17.8 min |
| Accepted random PPO replay and post-update evaluation, reusing baseline | 6.2 min |
| Verified bootstrap replay and repeated post-update evaluation, reusing baseline | 4.1 min |
| Search training pool | 38.0 min |
| Search development pool | 35.7 min |
| Distillation replay, one training epoch, and local outcome checks | 5.0 min |
| Distilled actor's 50 complete-match evaluations | 4.1 min |

These workloads overlapped, so summed stage wall time is not elapsed time.
Costs include the earlier failed attempts and any partial resume work in
the artifact report. Future runs need their own measured budget; this does
not estimate convergence to a strong policy. PPO training fits at roughly
5 GiB peak process RSS with two 1,440-decision lanes. The original eight-lane
attempts exceeded the 12 GiB guard. Distillation peaked around 1.6 GiB.
The machine used an i7-1260P, 16 logical CPUs, approximately 31 GiB RAM,
and CPU-only Torch 2.2.2 with one Torch thread per process.

## Search and distillation evidence

Both search pools used the same bootstrap actor, separate pool seeds
`20269001` / `20269002`, five training-deck source matches each, and 16
candidates per match. Each candidate used eight discovery replicas and
eight independent confirmation replicas, a 128-decision horizon, and a
branch cap of seven. Each pool had three workers.

| Pool | Candidates | Exported frontier entries | Independently accepted corrections |
| --- | ---: | ---: | ---: |
| Training | 80 | 19 | 5 |
| Development | 80 | 13 | 7 |

Accepted counts use the actual `validated_correction` confidence gate;
exported frontier entries alone are not accepted labels. Training coverage
was beatdown (2), defensive cycle (2), and air beatdown (1). Development
coverage was defensive cycle (4), beatdown (2), and siege/bait (1). Neither
pool yielded accepted aggressive-pressure corrections.

Distillation verified source-state hashes by prefix replay and checked
disjoint source match seeds. It prepared 5 corrections plus 64 retention
examples for training, and 7 corrections plus 32 retention examples for
development. One soft-KL/context-32 epoch made nine optimizer steps at
learning rate `1e-5`, with finite mean loss 0.01252. The exported actor
reloaded successfully. All 39 local development outcome checks completed;
their deterministic selected actions were unchanged and mean KL was
approximately `4.26e-6`. This is evidence of a functioning pipeline with
very little policy movement at this dose.

## Runtime fixes and recovered work

The updater fixes handle GRU training mode during autograd and the absent
gradient-clipping configuration attribute. The random PPO update was
initially rolled back because near-tied placement logits moved their
argmax despite negligible placement KL. Placement movement is now telemetry;
probability-drift, numeric, and entropy guards remain. A regression covers
this case. The retry reproduced the exact rollout and numerical update,
and reused only the verified initial actor's baseline games.

The earlier bootstrap job actually completed shortly after the stop
request, despite an intermediate cancellation record. Its final cost and
curve show successful completion. Resuming recovered that result and
reproduced exactly the same post-update actor and critic weights. The resumed
job also finished its repeated evaluation before cancellation; the accepted
curve uses the original completed records and counts the extra runtime.
Stop-time documentation was corrected; no
cancelled job is counted as evidence of an accepted update by itself.

Focused tests passed: 74 tests, two CUDA skips. The default suite yielded
1,259 passed, 14 skipped, and three existing failures: annotation matching,
missing spell-video fixtures, and the KataCR tracker import. No new failure
category appeared.

## Follow-up supported at the time

The pilot establishes local feasibility and cost, while showing sparse
correction coverage and no demonstrated improvement after minimal training.
The next useful experiment is a bounded learning-curve screen with more
training and broader correction coverage, checking complete-match outcomes
as the budget increases. A larger method comparison becomes more useful
once at least one path shows an improvement trend. Keep method choice,
training dose, and any future fidelity work open to evidence toward the
all-deck goal; do not turn these pilot settings into new requirements.
