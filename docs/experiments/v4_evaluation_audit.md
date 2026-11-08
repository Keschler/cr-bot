# V4 40-deck evaluation audit — 2026-10-04

> Historical record (2026-10-04). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

**Historical verdict: READY after the fixes in the audited worktree**, for a
fresh comparison of the five pinned, frozen actors against the documented
simulator and scripted controllers. The original execution path required
fixes. This verdict does not establish live-game fidelity, strong opponents,
policy quality, or the relative merits of training methods.

The requested 2,000-game experiment was **not run**. Artifact preflight played
zero games. Targeted checks used synthetic scenarios, short controller-only
probes, and repeated small policy evaluations. No checkpoint, model binary,
curated ground truth, or KataCR revision was changed. Pre-existing worktree
changes were preserved. Python and pytest used `outputs/venv/` throughout.

The audit started at HEAD `74fb0a326b3ca15dd960f1596ba70c9b16514862` with a
dirty worktree. Existing pilot reports describe five training decks and 50
matches per actor; they do not cover the requested 40-deck experiment. The
October 4 user request defined the audit's protocol even though older planning
documents have moved to a bounded pilot.

## Verified experiment and artifacts

The proposed manifest is
[comparison_manifest.json](../../outputs/v4/audit_20261004/comparison_manifest.json).
Its successful zero-game result is
[preflight.json](../../outputs/v4/audit_20261004/preflight.json).
The ten proposed fresh match seeds are `2026100401` through `2026100410`.
They do not overlap the inspected bootstrap episode seeds, PPO source-match
seeds, search training/development match seeds, or historical pilot evaluation
seeds. The user did not specify seed values; these values make the proposed
experiment concrete and reproducible.

All five actors strict-load through `load_frozen_anchor`, using the explicit
frontier model configuration. Each loaded tensor was checked against its saved
state dictionary; all 56 state entries are finite. The file hashes and actual
parameter digests are distinct. PPO update-zero weights equal their respective
random/bootstrap starting actors; serialization can give equal tensors
different file hashes.

Paths below are relative to `outputs/v4/pilot_20260930/`.

| Named policy | Checkpoint | SHA-256 |
| --- | --- | --- |
| Random initialization | `bootstrap/random_actor.pt` | `98144dd0fefad3ea45288be33d425e79526b46345d67cbb689c525b5561f237a` |
| Random initialization + PPO | `ppo_random_accepted_curve/checkpoint_update_000001.pt` | `b7451270b7aff20e934aa819bf56dfae363e441c18adb5540aae03cc29c1f0cd` |
| Rules-teacher bootstrap | `bootstrap/actor.pt` | `cdb48ffc1fc8ac72ebb3ac7a3081ebed2bf6487e35891537ddb8dbfd8bab993f` |
| Bootstrap + PPO | `ppo_bootstrap_accepted_curve/checkpoint_update_000001.pt` | `34b7d2a93d50440c5d0596bcdd3d8b8cdd72700305849737675451d532d836a2` |
| Bootstrap + search distillation | `distill/actors/context32_soft-kl_seed900.pt` | `05d05cad6a603a41c567078d0fab1f40856197b25d277da3ad0ce6a5fc17291b` |

The model uses a 21×32×18 raster, 768 global features, 128×32 entity tokens,
four 16-feature hand tokens, 128-card beliefs, and 16×8 event history.
Its fusion and one-layer GRU have width 64; the transformer has width 32,
four heads, one layer, and feed-forward width 64. Dropout is zero. Saved
configuration has `fixed_wait=false`; the executed V4 simulator runtime
explicitly sets `fixed_wait=true`, preserving weights while removing an
unexecuted duration factor from sampling and PPO probabilities.

Learner player is **0**, with the exact deck
`hog-rider, cannon, musketeer, skeletons, ice-golem, ice-spirit, fireball, log`.
Opponents are player 1. Actions are **seeded stochastic** for every actor,
using the same deck/seed-derived actor sampling seed across policies. Each
decision advances five 50 ms physics ticks: **250 ms**, with an explicit
**1,440-decision cap**. Match seeds control shuffled initial hands; controller
seeds are separately pinned per deck in preflight and honored at construction.

Catalog SHA-256 is
`cae33cab0a16e82868b05f093f33e96b2bf23d27287fa68dd6efc1621642d708`.
The following are separate reporting pools, with six train and two reserved
evaluation decks per controller family:

| Controller family | Training decks — 30 total | Reserved evaluation decks — 10 total |
| --- | --- | --- |
| aggressive-pressure | hog-26-zap, pekka-bridge-fireball, hog-knight-log, hog-mini-pekka, pekka-poison-snowball, pekka-ghost-minions | hog-icegolem-snowball, hog-knight-zap |
| defensive-cycle | xbow-30, xbow-pump-snowball, xbow-icewiz-rocket, xbow-valk-cycle, xbow-rascals-cycle, xbow-icegolem-rocket | xbow-megam, xbow-hunter |
| beatdown | golem-megam-lightning, giant-witch-arena3, golem-clone, golem-dark-prince, golem-standard-barbbarrel, golem-cannon-cart | golem-nw-pump, golem-guards-fireball |
| air-beatdown | lavaloon-lightning, lavaloon-skdragons, lavaloon-skdragons-fireball, lavaloon-barbarians, lavaloon-inferno-dragon, lavaloon-valk-flying-machine | lavaloon-fm-poison, lavaloon-miner-horde |
| siege-bait | valk-log-bait, bait-knight-dart, bait-prince-rascals, bait-valk-guards, bait-skeleton-barrel, bait-knight-tesla | rascals-princess-bait, bait-prince-guards |

Preflight retains the exact ordered eight-card deck, source URL, strategy,
and controller seed for all 40 identities. Catalog validation requires unique
IDs and unique eight-card sets, supported base cards, disjoint train/evaluation
sets, and card-set agreement with each stored source URL's deck slug
([real_decks.py](../../simulator/rl/v4/real_decks.py:59)). This is local provenance
validation; it does not verify current website popularity or controller skill.

## Material findings and fixes

### Evaluation conditions were incompletely executed and recorded

`full_match_record` constructed the opponent using the match seed, ignoring
an explicit `controller_seed`. It also reset the caller's global Torch RNG.
Raw rows omitted controller identity, explicit truncation reason, simulator
hashes, and illegal-action counts. The paired statistics key could accept
different declared controller/sampling conditions as comparable.

Fixed in [evaluation.py](../../simulator/rl/v4/frontier_ppo/evaluation.py:212)
and [learning_curve.py](../../simulator/rl/v4/learning_curve.py:1323): declared
controller seeds are honored; controller construction and sampling occur in
a restoring RNG context; the actor seed incorporates the opponent identity;
pairing includes controller, sampling, side, deck, cap, timing, and simulator
contract identities. Legacy rows only pair with rows carrying the same
legacy conditions. New raw rows retain termination/truncation, reason, crowns,
tower HP, actions played, elixir spent, invalid actions, action-trace hash,
initial/final state hashes, and event-log hash.

Reproduce with
`outputs/venv/bin/python -m pytest -q simulator/tests/test_v4_evaluation_audit.py`.
Tests check seed forwarding, caller RNG restoration, reversed evaluation order,
win/loss/draw on the last allowed decision, simulator truncation even when both
flags are set, and rejection of mismatched pairing conditions.

### Existing runners did not directly implement this frozen 2,000-game grid

`run_final_evaluation` is reserved-evaluation-only;
`run_learning_curve` evaluates its baseline separately from point zero.
Using either without carefully separate orchestration could miss training
decks or duplicate baseline games. The historical pilot covers only five
training deck identities.

Added [catalog_evaluation.py](../../simulator/rl/v4/catalog_evaluation.py:63)
and [run_v4_catalog_evaluation.py](../../scripts/run_v4_catalog_evaluation.py:15).
Preflight enforces the five names, pinned files, distinct actual weights,
explicit model/catalog, ten unique seeds, stochastic actions, and cap 1,440.
Execution performs **300 train + 100 reserved games per actor**, exactly
2,000 games overall, without training or checkpoint selection. It uses one
serial worker, writes ten raw JSONL files, and reports per-deck outcomes and
all ten policy pairs separately for each split. It records the source closure
at start/end and marks source drift or illegal actions as technically invalid.
Fully censored data remain reviewable and cannot support a winner claim.

`simulator/tests/test_v4_catalog_evaluation.py` verifies the entire grid using
stubbed match execution, so that test plays no large simulation workload.
The actual five-checkpoint preflight succeeded with zero games.

### Several assigned strategies could starve their own win conditions

Confirmed failures in
[opponent_pool.py](../../simulator/rl/foundation/opponent_pool.py:637):

- Quiet-board beatdown spent support elixir instead of saving for a held Golem
  or Lava Hound. Balloon was absent from primary/follow-up priorities.
- The defensive-cycle X-Bow strategy omitted a deliberate offensive X-Bow
  commitment and could keep cycling below six elixir. Siege X-Bow placements
  could be legal but outside enemy tower range.
- Aggressive Hog and siege/bait primary cards could also lose every quiet-board
  opportunity to cheaper cards. Minimal cases are Hog at three elixir with a
  one-elixir spirit, and Goblin Barrel at two elixir with Log.
- Beatdown follow-up selection could alternate away from the live tank's lane.
  Miner in the generic fallback targeted the own half instead of the enemy.

Fixed quiet-board elixir reservation for held primary cards, Balloon priorities,
same-lane tank support, Miner placement, and range-connected legal X-Bow cells.
Crossed enemy troops still trigger reactive defense first. Both seats are
covered by deterministic regressions in
[test_opponent_strategy_activity.py](../../simulator/tests/test_opponent_strategy_activity.py:91).

Reproduce with
`outputs/venv/bin/python -m pytest -q simulator/tests/test_opponent_strategy_activity.py simulator/tests/test_rl_opponent_pool.py`
(**34 passed**). The retained
[controller_probe.py](../../outputs/v4/audit_20261004/controller_probe.py) compares
HEAD controller code with the fixed code on the same 40 decks, hands, match
seeds and controller seeds, for at most 60 seconds per deck against WAIT.
It uses the actual 250 ms decision cadence and no policy actor.

[controller_probe.json](../../outputs/v4/audit_20261004/controller_probe.json)
records all placements and elixir usage. The fixed controllers played legally
in all 40 cases; all eight X-Bow decks deployed X-Bow, all eight Lava Hound
decks played same-lane Balloon, all seven Golem decks played Golem, and Miner
landed on the enemy side. The real Hog trace changed from nine cheaper plays
while holding an unaffordable Hog on a quiet board to zero such plays.
Unopposed terminal matches rose from 1 to 14; enemy tower HP reduction rose
from 196,886 to 314,030. These activity checks establish functioning win
conditions, not competitive opponent competence. HP reduction includes
automatic tower collapse and is not a pure combat-damage statistic.

### Standalone PPO could export a quarantined update

The standalone, initially untracked
`run_v4_frontier_ppo.py` (since retired) stepped optimizers,
then detected quarantine, but could still save the changed actor. Its stale
pre-V4 imports also prevented execution. Fixed imports and added accepted-state
snapshots/rollback for actor, critic, and both optimizers before final save.
Rollback covers updater quarantine and outer all-WAIT/card-collapse guards,
and is recorded in checkpoint/report metadata.

The historical standalone-runner quarantine test passed; that runner and
its dedicated test have since been retired. The synthetic test performs real Adam updates before forcing rejection,
then checks saved weights and restored model/optimizer states. The controlled
learning-curve runner already restores quarantined state. The five pinned
pilot actors were not produced by the affected standalone runner.

### Search certification accepted two malformed provenance cases

Duplicate discovery/confirmation replica IDs could certify nominally independent
corrections. Entries' claimed match seeds/state hashes were checked for overlap
without first binding them to their actual source decision, allowing false
metadata to bypass split checks.

Fixed replica uniqueness in
[targets.py](../../simulator/rl/v4/search_distillation/targets.py:105), and entry to
source-match/offset/seed/state-hash binding in
[run_v4_search_distillation.py](../../scripts/run_v4_search_distillation.py:231).
Reproduce with
`outputs/venv/bin/python -m pytest -q simulator/tests/test_search_distillation_replica_integrity.py tests/test_v4_search_distillation_source_provenance.py tests/test_v4_search_distillation_runner.py`.
The inspected pilot contains no malformed replica IDs or mismatched entries.
These guards therefore do not invalidate its saved actor.

## Other execution-path checks

**Outcomes and mechanics.** Tower death, king collapse, same-tick crown caps,
terminal tick accounting, tiebreak winner attribution and equal-lowest-HP draws
were traced through `engine/deaths.py`,
[engine/match.py](../../simulator/engine/match.py:80), and
[env.py](../../simulator/env.py:632). Player-0 wins score 1, losses score 0,
completed draws score 0.5. Truncations are censored, never draws; cap expiry and
simulator truncation remain explicit. A completed last-cap decision is still
completed. Reported tower HP reduction includes tiebreak drain/king collapse.

Focused deterministic checks and the default suite exercised mirrored Hog
deployment/damage, building-only targeting and Cannon pulls, deployment delays,
ground/air targeting, ranged damage, Fireball area/king activation, Log ground
collision and knockback, Ice Spirit damage/freeze, exact elixir accumulation
and phase change, four-play cycling/refill cooldown, and destroyed-tower
deployment zones. Representative tests are in
[test_engine.py](../../tests/simulator/test_engine.py:238),
[test_state_interaction_mechanics.py](../../simulator/tests/test_state_interaction_mechanics.py:449),
and `simulator/tests/test_remaining_base_card_mechanics.py`.
No evaluation-invalidating physics or winner/side bug was confirmed; engine
mechanics and observation schema were not modified by this audit.

**Actor input and recurrence.**
[runtime.py](../../simulator/rl/v4/v4_ppo/runtime.py:81) decodes model `(row, col)`
to world `(col, row)`, uses current compacted hand slots, leaves absent refill
slots masked, and rejects unsupported WAIT durations. Each observation precedes
the action it masks. Player 0's own hand/elixir are permitted actor inputs;
opponent hand/elixir come from public-history estimates rather than hidden
simulator state. `PublicStateEstimator.sync_from_env` uses viewer-visible events,
including concealment filtering. The actor receives public observation tensors;
the privileged critic is separate. Full matches create fresh estimator/GRU state,
keep recurrence within each match, and use eval mode with gradients disabled.
Existing privacy, mask, hand, log-probability and recurrent boundary tests pass.

**PPO and bootstrap.** Reward signs in
[env.py](../../simulator/env.py:1031) favor enemy tower damage and the learner's win,
and penalize own tower damage/loss. Collection computes successor values before
reset, bootstraps truncation without crossing resets, and zeroes true-terminal
bootstraps. Masked rollout/recomputed log probabilities agree. Returns,
advantages, gradient signs/finite checks, recurrent backward, and optimizer
movement are covered by the PPO/runtime regressions. Both pinned one-update PPO
actors are accepted, with nonzero actor and critic gradients and zero illegal
actions in their saved diagnostics. Pre-existing modifications to PPO update
and diagnostics were inspected and preserved. The random run records current
near-tie placement telemetry semantics; the bootstrap run records the earlier
diagnostic implementation but has zero placement moves, so this difference
does not alter that update's acceptance.

The bootstrap manifest records six completed episodes, 384 retained labels,
and eight training epochs. Eligible labels contain 462 PLAY and 5,288 WAIT;
each episode retains 32 of each mode. Inverse reservoir probabilities restore
natural mode proportions for the **mode loss**, as documented. Conditional
card/placement/duration targets remain episode-balanced deliberately. Teacher
actions respect public inputs and masks. All six bootstrap families use the
unchanged deterministic-cycle opponent, so the strategy fixes do not alter
those labels.

**Bootstrap recurrent-training limitation.** The retained bootstrap labels are
trained as shuffled sequences of length one, with every label resetting the
GRU ([bootstrap.py](../../simulator/rl/v4/bootstrap.py:476),
[run_v4_bootstrap.py](../../scripts/run_v4_bootstrap.py:158)). Evaluation carries
hidden state across the match. With zero initial hidden state and only one
training transition, the recurrent weight matrix receives no training signal.
The saved `recurrent.gru.weight_hh_l0` is exactly equal to random initialization;
it is the only unchanged bootstrap state tensor. This confirms that bootstrap
learns a one-step public-state mapping without training its recurrent kernel.

The independent
[bootstrap_recurrence_probe.json](../../outputs/v4/audit_20261004/bootstrap_recurrence_probe.json)
compares reset-state inference with 32 cached observations of history on all
108 saved correction/retention decisions. One deterministic action changes;
mean absolute PLAY-probability change is 0.01084, maximum 0.04771, and mean
factorized KL is 0.00219. This establishes a train/deployment context difference
but does not measure its game-level effect. Reproduce with
`outputs/venv/bin/python outputs/v4/audit_20261004/bootstrap_recurrence_probe.py`.
The frozen-policy comparison retains the deployed recurrent actor consistently
for all five policies. Sequence-aware bootstrap fitting would change training
semantics and require a new bootstrap and all dependent PPO/search/distillation
artifacts; it is a separate training improvement, not a reason to relabel or
reset the saved actor during evaluation.

**Bootstrap card-coverage limitation.** Recollecting the six predeclared
bootstrap episodes under the unchanged deterministic-cycle controller reproduces
all historical PLAY/WAIT counts and all 6,152 simulator decisions. The bounded
check in
[bootstrap_label_probe.json](../../outputs/v4/audit_20261004/bootstrap_label_probe.json)
finds the following actual card targets:

| Card | Eligible PLAY labels | Retained PLAY labels |
| --- | ---: | ---: |
| Hog Rider | 10 | 7 |
| Musketeer | 7 | 5 |
| Cannon | 76 | 32 |
| Ice Golem | 87 | 37 |
| Ice Spirit | 91 | 37 |
| Skeletons | 92 | 39 |
| Fireball | 43 | 15 |
| Log | 56 | 20 |
| Total | 462 | 192 |

The teacher's defense and spell rules precede pressure. Its pressure rule
requires no immediate threat and at least six elixir
([simulator_teacher.py](../../simulator/rl/v4/simulator_teacher.py:201)). Mode-balanced
reservoirs do not ensure card coverage. Consequently the bootstrap's principal
win condition and ranged defender receive little direct supervision, despite
legally and correctly encoded hand slots. This is a measured teacher/data
limitation rather than a corrupted mask or incorrect Hog encoding. The saved
manifest records mode counts but not per-card counts; new training manifests
should retain both. Reproduce the six-episode data check, without fitting an
actor or running a policy evaluation, with
`outputs/venv/bin/python outputs/v4/audit_20261004/bootstrap_label_probe.py`.
Broader teacher trajectories or a stronger training procedure need fresh
bootstrap-derived artifacts. The measured coverage does not justify claiming
that these actors represent well-trained Hog 2.6 policies.

[checkpoint_audit.json](../../outputs/v4/audit_20261004/checkpoint_audit.json)
measures actual parameter movement, rather than relying on new file hashes:

| Parent → child | Changed state tensors | Maximum absolute change | Total absolute change |
| --- | ---: | ---: | ---: |
| Random → bootstrap | 55 | 0.01723779 | 505.41352 |
| Random → random + PPO | 53 | 0.00030005 | 25.78387 |
| Bootstrap → bootstrap + PPO | 54 | 0.00030005 | 25.92965 |
| Bootstrap → distillation | 54 | 0.00006052 | 1.43031 |

**Distillation.** Before audit edits, the saved distillation implementation
closure matched 226 source files. Bootstrap/source/checkpoint hashes and prefix
cache identity agreed. All 19 training and 13 development frontier entries
matched actual source decisions; seeds and entry state hashes were disjoint.
All 160 saved verdicts had disjoint discovery/confirmation replicas. Replayed
training offsets 39/112 and development offsets 48/40 reproduced saved state
hashes. Cached recorded actions match source decisions, and correction and
retention actions are legal under cached masks. Prefix recurrence precedes the
target decision; soft targets align with mode, card slot, and placement.

There are five unique train corrections plus 64 retention examples, and seven
development corrections plus 32 retention examples. The student ran one
context-32 soft-KL epoch at learning rate `1e-5`, nine optimizer steps, with
finite loss and the small parameter changes above. Minibatch padding accounts
for reported exposure of six correction and 66 retention examples; these are
not extra unique source examples. Correction and retention losses are separately
averaged and use the explicit retention coefficient
([losses.py](../../simulator/rl/v4/search_distillation/losses.py:94)).

Training and development sources use the **same five training deck identities**,
with different seeds, because each selected `match_index=0`. This is seed-disjoint
development, not held-out deck development. The ten reserved deck identities
were not used in the inspected training/search sources.

**Reproducibility and pairing.** The frozen runner is serial; it does not rely
on thread-local isolation of Torch's process-global generator. Engine and
controller randomness use match-local/controller-local RNGs. Existing process
vector-backend parity checks pass with local socket access. Two independent CPU
processes repeated the bootstrap actor against `hog-26-zap` and `xbow-30`, using
two fresh seeds, stochastic sampling and the full cap. Deck/seed iteration order
was reversed and external Torch RNG seeds differed. Sorted complete records,
action traces, initial/final state hashes and event hashes were identical, and
caller RNG was restored. All four cases were losses, with zero illegal actions
and zero censored outcomes; paired score delta was exactly zero. This verifies
reproducibility, not strength.

Raw records and statistics are retained in
[repeat_a.json](../../outputs/v4/audit_20261004/repeat_a.json),
[repeat_b.json](../../outputs/v4/audit_20261004/repeat_b.json), and
[repeat_summary.json](../../outputs/v4/audit_20261004/repeat_summary.json).
Pairing rejects mismatched deck/seed/controller/contract identities and retains
censored pairs. Uncertainty is clustered by shared match seed: ten clusters
for the planned experiment, rather than treating all deck-seed cells as
independent replicates. Per-deck records remain available for interpretation.

## Verification and reproduction

Run from the repository root. This preflight command plays **zero games**:

```bash
outputs/venv/bin/python scripts/run_v4_catalog_evaluation.py \
  --manifest outputs/v4/audit_20261004/comparison_manifest.json \
  --preflight-only --device cpu --torch-threads 1
```

The small repeated-match check can be reproduced with:

```bash
outputs/venv/bin/python outputs/v4/audit_20261004/repeat_policy_probe.py /tmp/audit_repeat_a.json
outputs/venv/bin/python outputs/v4/audit_20261004/repeat_policy_probe.py /tmp/audit_repeat_b.json reversed
```

The original focused training, bootstrap/public-input, distillation and mechanics
suites passed; CUDA-specific checks skipped because CUDA was unavailable.
The final default suite covers all new regressions:

```bash
outputs/venv/bin/python -m pytest \
  --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py
```

| Run | Passed | Failed | Skipped | Explanation |
| --- | ---: | ---: | ---: | --- |
| Before audit edits | 1,254 | 8 | 14 | Three existing failures plus five sandbox-denied process socket tests |
| Process socket tests with local socket access | 5 | 0 | 0 | Same tests, no implementation change |
| Final default suite with local socket access | 1,313 | 3 | 14 | Same three existing failures; no new failures; 166.71 s |

The three existing failures are:

- `tests/test_annotation_batch.py::test_semantic_and_location_scoring_use_maximum_cardinality_matching`: one match instead of the expected two.
- `tests/test_run_action_eval_scenarios.py::test_spell_scenarios_use_matching_videos_and_ground_truth`: missing local `dataset_generation/data/video_clips/spell/spell.mp4`.
- `simulator/tests/test_vision_compat.py::test_katacr_tracker_constructor_supports_current_and_legacy_signatures`: the pinned vendored tracker lacks `_build_tracker`.

The audit did not modify those areas or the submodule. Audio-specific files
were excluded as required by AGENTS.md. Logs and machine-readable verification:
[baseline_pytest.log](../../outputs/v4/audit_20261004/baseline_pytest.log),
[worker_pytest.log](../../outputs/v4/audit_20261004/worker_pytest.log),
[final_pytest.log](../../outputs/v4/audit_20261004/final_pytest.log), and
[verification.json](../../outputs/v4/audit_20261004/verification.json).
`git diff --check` passed.

For the later authorized large evaluation, use this dedicated frozen runner
with a fresh output directory; **this command was not executed during the audit**:

```bash
outputs/venv/bin/python scripts/run_v4_catalog_evaluation.py \
  --manifest outputs/v4/audit_20261004/comparison_manifest.json \
  --out outputs/v4/catalog_comparison_20261004 --device cpu --torch-threads 1
```

## Artifact regeneration and interpretation limits

| Artifact or claim | Effect of this audit |
| --- | --- |
| Five pinned actors | Unchanged, finite, distinct, load-compatible; usable for the frozen-policy benchmark. Label them as historical pilot actors. |
| Historical full-match/development comparisons | Rerun all compared actors. Controller behavior and actor sampling conditions changed; old and new rows must not be mixed. |
| New PPO training under repaired heuristic opponents | Generate fresh rollout/start sources as appropriate and fresh PPO runs. Existing pilot PPO weights remain valid historical actors, not products of the repaired training opponents. |
| New search/distillation under repaired opponents | Regenerate controller-dependent source collection, continuation verdicts, frontier classifications, targets, prefix caches and student training. Do not resume an old implementation-identity cache as if it used current semantics. Recorded old action prefixes can still replay exactly because physics/observations did not change. |
| Bootstrap labels and initial actors | Deterministic-cycle behavior, actor inputs and physics are unchanged; these parent artifacts need no regeneration for this benchmark. |
| Future sequence-aware or differently covered bootstrap | Recollect suitable sequences/labels, refit bootstrap, and rerun all bootstrap-derived PPO and search/distillation arms. Existing actors continue to represent the earlier one-step bootstrap procedure. |
| Guard-only search fixes | Existing checked entries/replicas are valid; guards alone do not require replacing their saved student. New runs/cache identities must describe the new code. |
| Standalone quarantined PPO outputs | Any output saved after a rejected update by the old standalone runner is suspect and should be rebuilt from the last accepted state. None of the five pinned actors comes from that path. |

The catalog covers 40 **base-card decks in five scripted families**, not all
possible decks, evolutions, heroes, skilled human play, or adaptive learned
opponents. Source-linked card lists and functional win conditions do not prove
strong controllers. The scripts remain simple priorities; `air-beatdown` shares
the beatdown controller implementation. Report controller competence separately
from deck provenance.

The ruleset is a documented approximation and still has `training_ready=false`.
The Log's placement/width calibration remains explicitly uncertain
([v1.json](../../simulator/rulesets/v1.json:7435)). Equal-lowest-tower tiebreak draw
handling is a tested engine convention, not independently established live
fidelity. Public simulator damage/HP is more precise than vision estimates
([simulator/README.md](../../simulator/README.md:209)). These can change strategy
rankings and transfer; the audit did not conceal them by altering physics.

Reference engine is `reference-0.37.0`, ruleset hash
`sha256:02ab25da274367671ced1593ad0ff99efd3d537e61331d697454dac64bde8d76`,
observation schema `public-v3-2`, contract hash
`sha256:13f44f8e01d76211ba598954ee114a4210eab5a7613f94103a707db0c62f8a8c`.
The tested execution profile is CPU with one Torch thread; CUDA reproducibility
was not established. Future source/ruleset/config changes require fresh
provenance and revalidation.

Finally, one training seed, one PPO update, and five distillation corrections
make these procedure-scale actors. Four hundred games per actor can measure
their strength against this fixed benchmark with uncertainty; it cannot show
which training method is generally best. Keep 30-deck training results and
ten-deck reserved results separate, retain draws/truncations and raw records,
and avoid selecting or tuning actors on the reserved results.
