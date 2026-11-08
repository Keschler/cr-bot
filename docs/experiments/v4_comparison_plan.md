# V4 comparison and bounded pilot

> Historical record (2026-09-30–2026-10-01). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The bounded pilot completed after resuming. Both PPO arms and
search distillation produced evaluated checkpoints. No best method was
established. See [the recorded results](v4_pilot_results.md). The pilot was a one-day-scale
feasibility study for the Hog 2.6 policy goal. The former large fixed
comparison protocol is superseded. Training methods, budgets, match counts,
and correction-label targets remain choices to test against the end goal.

The subsequent [40-deck benchmark](v4_catalog_results.md) exposed severe
policy weaknesses. Its [failure diagnosis](v4_catalog_diagnosis.md) led to a
separate [initializer recovery and bounded PPO experiment](v4_recovery_results.md),
which is also complete. The subsequent
[catalog initializer experiment](v4_initializer_coverage_results.md) improved
final wins to 92/400 versus 13/400 for the recovered baseline; PPO lost fresh
confirmation. That report records a scripted-opponent placement warning and
paired exclusion sensitivity. The design below records the historical pilot;
each later report records its own protocol and evidence.

The pilot must use a fresh source revision and fresh output directories. Older
frontier, verdict, and distillation outputs are historical artifacts; do not
reuse them after simulator, physics, observation, search, or checkpoint
changes. The actor must receive public observations and legal-action masks
only. A privileged critic or opponent controller may use simulator state
during training, but those features must not enter actor inference.

## Recorded pilot design

The pilot asks whether the training paths can produce valid, measurable data
quickly enough to guide the next investment toward a strong Hog 2.6 policy
against a wider opponent set. It does not promise that one match per
archetype can rank methods reliably.

Use the current source-linked real-deck catalog: 30 training decks and 10
reserved evaluation decks across five training archetypes. The pilot uses
training identities only; it has no sealed or final-evaluation pool.

Use these predeclared pilot inputs:

- pilot seed `900`;
- fresh training-pool seed `20269001`;
- fresh development-pool seed `20269002`;
- one complete source match per training archetype in each pool;
- no sealed matches and no final evaluation.

Run bootstrap collection and fitting, one actual PPO update from both the
random and bootstrap initializations, and search plus distillation from the
bootstrap actor. The three pilot arms are:

| Arm | Initial actor | Pilot question |
| --- | --- | --- |
| A | Saved random V4 actor | Does reward-only PPO execute and learn from the public match-start stream? |
| B | Rules-teacher bootstrap actor | Does the bootstrap provide a useful starting point for PPO? |
| C | The same bootstrap actor as B | Can outcome-search corrections be collected and distilled? |

For a bounded run, use 16 search candidates per source match, eight discovery
replicas, eight independent confirmation replicas, horizon 128, a 1,440
decision match cap, and the current three-worker/one-Torch-thread setting.
Use `soft-kl`, context 32, one distillation seed, batch 8, learning rate
`1e-5`, and one optimizer epoch. Keep the initial and trained actor exports. Override the distillation
runner defaults; they otherwise run four methods, two contexts, and three
seeds. Set the frontier sealed-match count to zero explicitly.

The pilot records:

- public-observation and legal-action checks, including late-match states;
- terminal versus truncated matches and both bootstrap action modes;
- finite losses and gradients, actor state-dict hashes, and export manifests;
- eligible and retained labels, independent search corrections, and their
  archetype coverage in both pools;
- simulator decisions, search branches and replicas, optimizer work, worker
  settings, hardware, wall time, and CPU/GPU time where available.

An empty correction pool, missing action mode, invalid export, source overlap,
or integrity mismatch is a collection or contract failure. It is not evidence
that a training method is weak. There is no current minimum correction count
or universal simulator-fidelity threshold for continuing simulator learning.
If the pilot is healthy, use its measured yield and cost to choose a bounded
method screen or a larger comparison. Keep that choice separate from this
pilot and predeclare it before examining held-out results.

The pilot should use external timing or the repository budget guard in
addition to manifests. Existing runners record useful counts and hashes, but
not every per-candidate timing needed for a sustained cost estimate. A pilot
cost claim is incomplete without the measured wall-time and worker context.

## Quality and interpretation

The target is policy strength against broad enemy decks, not a preferred loss
curve or a fixed action frequency. Compare complete matches, per-opponent
outcomes, crowns and tower damage, truncations, invalid actions, and paired
uncertainty. Keep simulator reward and search-label diagnostics separate from
policy quality.

The current catalog contains real deck identities, not a claim that every deck
is popular, strong, or played by a human. Its controllers are scripted. A
pilot or simulator comparison can guide training and expose regressions, but
it does not establish live-game strength, robustness to learned adversaries,
or coverage of every enemy deck. Those claims require evidence appropriate to
their scope.

## Data and checkpoint integrity

Every run must keep its source revision, ruleset and observation contracts,
catalog hash, actor hashes, seed allocation, deck identities, match paths,
split, and output directory together. Save the random actor before bootstrap
optimization so arm A and the bootstrap's pre-training state are identical.
Regenerate source matches and search verdicts after a simulator or search
change. Never mix old physics records with new distillation targets.

Bootstrap retention may balance PLAY and WAIT coverage, but it must retain
eligible counts and apply the inverse-retention weights to the mode loss. A
nonempty match/mode group cannot be reconstructed after it was discarded.
Search corrections require independent paired simulator replicas; a single
replica is a diagnostic, not a certified correction.

## Historical runtime probes

The following measurements are preserved because they explain why the large
protocol was expensive. They were local probes, not an end-to-end run of the
current pilot, and should not be treated as current throughput guarantees.

| Workload | Historical wall time |
| --- | ---: |
| PPO collection, eight lanes × 128 decisions | 4.46 s |
| PPO optimizer update on that rollout | 5.35 s |
| Search, four branches × eight replicas × 128 decisions, early state | 23.69 s |
| Same search workload, later discovery / confirmation | 57.07 / 60.72 s |
| Five complete stochastic evaluation matches | 5.62–20.73 s each; 10.39 s mean |
| Synthetic context-32 distillation probe | 5.82 s |
| Earlier bootstrap collection, 1,024 records from 18 matches | 420 s |

Later stepping and search changes preserved verdict and state hashes. A small
complete-match sample then averaged 7.34 s, and an A/B candidate probe reduced
four-branch search from 46.31 to 38.26 s early and from 76.78 to 55.28 s late.
Prefix replay alone fell from 21.35 to 2.93 s in a 512-decision probe. These
are component measurements, not sustained full-study throughput.

The superseded full search extrapolation was roughly 75–145 serial CPU-hours,
with about 3–8 days of elapsed local time at an assumed three-worker scale.
The full evaluation point estimate after the stepping change was about 19
serial hours. Full 1,440-step PPO optimizer throughput, sustained process-pool
scaling, and convergence time were not measured. The range remains a
historical planning warning, not a deadline or a current budget.

The RTX 2050 follow-up measured approximately 1.27× speedup for short PPO
collection, 5.47× for a short PPO update, 1.01× for search, and 5.62× for
synthetic distillation. Unmodified CUDA PPO hit
`cudnn RNN backward can only be called in training mode`; the successful probe
disabled cuDNN only around the update. Planned 1,440-step memory fit and
sustained GPU evaluation were not measured. Keep search on CPU unless a fresh
pilot demonstrates a benefit.

Physics and stepping changes require fresh source matches and verdicts. A
pre-fix output can document a historical probe, but it must not be fed into a
new search-distillation run. The historical timing range remains provisional
until a sustained pilot measures the current revision.
