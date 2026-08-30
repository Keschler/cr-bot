# Simulator V1 Main Goal

## Mission

Build a deterministic, headless, versioned Level-11 Clash Royale simulator
that can provide trustworthy rollouts for a production reinforcement-learning
policy.

The player uses one fixed base Hog-cycle deck. The opponent may use every
eligible base card released before **2025-12-01**. Unknown or weakly supported
behavior remains provisional and fails the relevant readiness gate; it is never
silently presented as accurate.

The controlled `physical_lab` is the primary sim-to-real evidence path. Public
video is supplementary discovery and validation evidence, not a substitute for
a controlled probe when the action boundary or causal behavior matters.

## Current status

- Latest validated implementation revision: `87f96304bfe2ad7e0539298859e8ff953f42e6c7`;
  its tracked worktree is clean and the current ruleset hash is
  `sha256:992ead6f14016917b5a108eaa1ca370c10a48b70d6a87ad69c7de50a8b020d7a`.
- `rulesets/v1.json` contains 124 definitions, including all 109 eligible
  opponent cards, and remains `training_ready: false`.
- Generated coverage is deterministic and executable: a fresh strict run of
  1,182 cases passed execution, repeated hashes, complete roster coverage, and
  behavioral obligations (`1,182/1,182`, two repeats, per-tick validation,
  eight workers).
- The focused RL/diagnostics/vector suite is green (`95 passed`). The full
  non-audio/non-mining suite is `528 passed, 9 failed, 11 warnings`; eight
  failures require card-image assets from the separate capture tree and one
  requires the unavailable `katacr` package. Roster completeness is clean,
  but no card is fidelity-ready, the metadata has nine source conflicts, and
  three training blockers remain declared.
- Phase-0 physical-lab software is implemented, but physical evidence is
  intentionally deferred for the current RL-first execution path. No
  connected run has yet satisfied the evidence/readiness gates.
- The policy is a provisional research harness, not an any-deck player. The
  retained checkpoints under `outputs/training/` were produced on stale
  `cd22` ruleset/revision hashes and must not be used as current evidence. The
  current engine `reference-0.37.0` is used for new runs, which must be
  revision-pinned.
- On revision `87f9630`, batch-16 CUDA inference measured 4.69k actor-only
  decisions/s and 3.77k full actor-selection decisions/s on the RTX 2050.
  The same workload on CPU measured 0.84k actor-only and 0.74k full
  decisions/s. The current reference simulator measured 1.63k environment
  steps/s at 16 lanes; CPU policy parity is therefore still open.
- The current vector benchmark measured 1,627.7 reference, 878.2 process,
  45.9 packed-process, and 624.5 persistent-process environment steps/s at
  16 lanes. All optimized runs matched the reference state-hash sequence;
  backend replay/event parity is covered by the regression suite. Packed
  process is correct but currently a throughput outlier.
- The retained historical trainer baseline is 590 decisions/s on the RTX 2050
  with 48 lanes, eight rollout workers, and overlap. The current memory-bounded
  two-lane PPO path measures 96.4 decisions/s end-to-end over 1,536 transitions
  (98.0 best repeat); current batched matrix evaluation measures 377 decisions/s
  over 4,753 decisions. Normal evaluation skips replay-hash serialization;
  `rl.generalized evaluate --replay-hashes` enables it for differential audits.
  These are throughput results, not strength claims.
- The current action contract has `WAIT`/`PLAY`, card slot, and placement but
  no learned wait-duration head; `WAIT` advances the fixed simulator decision
  interval. The proposed timing head remains a future architecture change.

These results establish plumbing and performance, not game strength or
sim-to-real fidelity.

The previously recorded generalized actor and reports are generated local
artifacts; they are retained only for provenance and are not current strength
evidence. They do not establish the mission's any-deck capability.

## Current execution priority

For the current development pass, proceed directly with RL improvements using
the executable but provisional V1 simulator. Do not wait for or fabricate
physical-lab evidence. This is a research-path waiver only: provisional
training and evaluation may run, but the deferred physical-lab and fidelity
gates still block `training_ready` and full-V1 release claims.

## Development workflow

Use `../capture/.venv-train/` from `simulator/` (or
`capture/.venv-train/` from the repository root). Keep `README.md` current
when behavior, architecture, setup, evidence, or benchmarks change.

Run focused tests first, then the default suite when practical:

```bash
pytest tests/test_rl_model.py tests/test_rl_prototype.py
pytest --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py
```

Generated outputs, datasets, caches, media, and unrelated worktree changes
are never part of an implementation commit.

The simulator may receive independent bug-fix commits while RL work is in
progress. Treat every new `HEAD` as a new experiment revision: record the
commit and tracked-worktree state before collection, require the run to finish
on that same revision, and rerun preflight, parity, and exploit checks after
any intervening commit. Do not promote or compare an artifact produced across
mixed simulator revisions.

## Scope

### Player

The only player deck is:

```text
Hog Rider, Cannon, Musketeer, Skeletons,
Ice Golem, Ice Spirit, Fireball, The Log
```

### Opponent

The pinned V1 manifest contains eligible ground and air troops, buildings,
spawners, spells, swarms, split/death forms, and status effects. Release date,
card ID, canonical ID, card kind, provenance, and eligibility must be
machine-readable.

V1 uses one immutable Level-11 ruleset and one engine version. A data or
behavior change creates a new version; it does not mutate V1 in place.

### Exclusions

- Evolutions, Heroes, activated abilities, and unsupported Champion forms.
- Alternative tower troops, game modes, arenas, special events, and draft.
- Alternative balance versions, card levels, and player decks.

An excluded feature must fail explicitly rather than being approximated by a
different mechanic.

## Policy and simulator boundary

The simulator is separate from the policy. It owns mechanics, authoritative
state, action validation, legal masks, rewards, terminal outcomes, and replay
hashes. The actor consumes only public observations. A separate training-only
critic may consume full simulator state; its features never enter actor
inference.

```text
authoritative simulator
        |
        +--> public observation --> recurrent actor --> legal action
        |                              |
        +--> legal-action masks --------+
        |
        +--> full state -----------> privileged critic (training only)
```

Only impossible actions may be masked: unavailable cards, insufficient elixir,
illegal cells or targets, and terminal states. Strategic choices remain
learned. Fixed counter trees and teachers may be opponents, regression
baselines, or short-lived auxiliary-training data, but they must not execute
the learner's actions in ordinary PPO.

## PPO regression diagnosis gate

Do not respond to a PPO regression with blind hyperparameter tuning. Diagnostic
training traces must record the public state, units/positions, hand, elixir,
towers, legal mask, actor action and alternatives, label-only teacher action,
agreement, critic value, return/advantage, PPO ratio/clipping, and per-update
KL, entropy, clip fraction, value/policy loss, explained variance, advantage
distribution, gradient norms, per-head gradients, action distributions, and
teacher/head entropies. `rl.diagnose` compares the last good, regressed, and
recovery checkpoints on identical state streams/seeds and reports concrete
category and consequence evidence. The actor remains the environment action
source, and every run is subject to the simulator-exploit audit.

The current `cd22` evidence identifies the verified failure as the
class-balanced factor behavior-cloning auxiliary loss destabilizing
mixed-PPO mode/card/placement updates. The same defensive stream changed from
311 bad-vs-good decisions with the term enabled to 8 with it disabled. The
smallest fix is therefore to apply that auxiliary loss only in explicit
`imitation_only` warm-starts; mixed PPO records but does not apply its raw
factor loss. This fixes the diagnosed decision failure, not overall policy
strength. Resume larger training only after the identical held-out decision
check and exploit audit are clean.

The first repaired continuation (segments 33–34, 4,096 actor decisions) was
clean and actor-controlled. On the exact defensive comparison it reduced the
bad checkpoint's 311 divergent decisions to 5, with zero additional
follow-on self-tower damage; the identical six-cell held-out matrix still
scored 0/6, so this is not a strength promotion. Update 130 also showed value
loss 0.0149 and explained variance 0.082, which requires continued
monitoring rather than a convergence claim.

## Actor architecture

The active public actor is factorized and recurrent:

1. Build public raster/global features and public entity rows.
2. Project hand-card table features independently for each of the four hand
   slots.
3. Run the Transformer over public entity tokens. Hand cards are not
   Transformer entity tokens; `card_embedding` represents the four slot
   positions, not card identities.
4. Feed the encoded public history to a GRU of approximately 256 units.
5. Decode masked `WAIT`/`PLAY`, card slot, and card-conditioned placement.
   `WAIT` currently advances the fixed simulator decision interval; timing is
   not yet a learned head.

The critic has a separate encoder and may use exact hidden opponent state.
Belief heads are optional low-weight auxiliaries and must use public actor
inputs. Mechanics and legality are simulator facts; card choice, timing, lane,
defense, kiting, and placement are policy decisions.

## RL phases

Decision counts are planning ranges. A segment is promoted only after the
preflight, exploit audit, and declared quality gates pass.

1. **Basic mechanics — approximately 1–5M decisions.** Use short generated
   scenarios rather than complete games:
   - 25% isolated offense/tower pressure;
   - 25% ground defense;
   - 20% air defense;
   - 15% spell situations;
   - 15% kiting/cycling/elixir situations.

   Randomize lane, card order, elixir, tower HP, troop timing, and placement.
   Define success from the resulting game state, never from a presumed
   “correct card” label.

2. **Scripted curriculum — additional 10–30M decisions.** Sample 20% Phase-1
   rehearsal, 20% passive/random-legal opponents, 20% simple win-condition
   scripts, 20% reactive defensive/aggressive scripts, and 20% randomized
   tempo/placement scripts. Scripts control only the opponent.

3. **Meta-deck training — additional 30–100M decisions.** Expand from 5
   representative archetypes to 20 decks, 50 decks, and the full validated
   meta pool. Sample 35% uniform archetypes, 30% weakness-prioritized
   matchups, 20% earlier-curriculum rehearsal, and 15% randomized variants.
   Keep unseen validation decks separate.

4. **Historical self-play — additional 100–300M decisions.** Sample 30%
   scripted/meta anchors, 30% PFSP historical policies, 20% the newest frozen
   main policy, 10% random historical checkpoints, and 10%
   exploiters/adversarial policies. Train both sides. Do not use only the
   latest-policy mirror.

5. **Small league — roughly 300M–1B+ cumulative decisions.** Maintain a main
   learner, main exploiter, league exploiter, 16–32 frozen historical
   policies, a payoff matrix, PFSP matchmaking, periodic exploiter resets, and
   fixed scripted anchors. Add PSRO only for demonstrated non-transitive
   cycles that PFSP cannot represent.

6. **Frozen evaluation.** Use paired seeds, both sides, per-deck results,
   confidence intervals, tower/crown outcomes, action traces,
   rejected/fallback counts, historical regression checks, and
   simulator-perturbation tests. A finite 100% result against one script is
   not an any-deck claim.

   Matrix robustness runs accept a `DomainRandomizationConfig`, force
   sequential execution, and record the declared profile plus sampled episode
   variant separately from baseline throughput and promotion evidence.

   When no cap is supplied, evaluation derives the complete
   regulation-plus-overtime horizon. A non-divisible training sequence length
   is discarded only from the temporary collector configuration; recurrent
   hidden state remains continuous per decision.

The executable strategic curriculum switches the default phases on cumulative
learner decisions at 5M, 35M, 135M, and 435M. It stores each phase mix as
deterministic `sampling_mix` slots and records observed source counts and
decision cursors per segment. Custom schedules without decision boundaries
use the segment cursor. These labels select opponent/scenario provenance only;
they never select the learner's card, timing, lane, or placement. Generalized
reports persist the cumulative decision cursor so resume does not infer phase
progress from a changed lane count or horizon.

Side-balanced self-play is executable with matched generalized runs using
`--target-player 0` and `--target-player 1`. The trainer swaps only world deck
ordering, keeps the learner's public observation contract unchanged, and
records the actor/opponent side in each report.

The league coordinator now retains directional payoff/rating state, provides
cursor-seeded PFSP sampling, and requires an explicit callback for configured
periodic exploiter resets; reset state is serialized with the league cursor.

## Implementation loop for RL scaling and frozen evaluation

Run this loop for each curriculum or league segment. Frozen evaluation uses
the same loop with learning and parameter selection disabled.

1. **Seal the run.** Record code, ruleset/engine, observation/action contract,
   architecture, optimizer, device/backend, decks/opponents, seeds, budget,
   and splits. Reports and checkpoints record the Git `code_revision` and
   tracked-worktree state. Create the held-out split before collection. If an
   independent simulator fix changes `HEAD`, stop the run, quarantine its
   artifact, and reseal on the new revision.
2. **Run preflight.** Check reset/save/restore, recurrent resets, public-only
   actor inputs, legality masks, reference/optimized parity, and fixed
   regression scenarios.
3. **Collect actor-controlled rollouts.** Ordinary PPO keeps
   `expert_execution_probability=0`. Preserve recurrent state boundaries,
   terminal versus truncated outcomes, masks, action sources, opponent
   assignments, and reproducible state/event hashes.
4. **Train and checkpoint.** Train only on the training split. Record outcomes,
   censoring, entropy/KL, rejected/fallback actions, assignments, throughput,
   and the checkpoint/report pair.
5. **Audit for simulator exploitation.** Check for reward hacking, truncation
   stalling, reset/terminal bugs, stale observations or masks, hidden
   information leaks, collision/navigation/timing loopholes, opponent
   controller artifacts, illegal-action accounting errors, and
   reference/optimized divergence. High reward without terminal wins,
   abnormal cap-time outcomes, impossible transitions, action collapse, or
   controller-specific wins are flags.
6. **Quarantine and fix flags.** Mark the run
   `simulation_exploit`/`invalid`; preserve the smallest reproducing seed and
   trace; quarantine contaminated data; classify the bug; add a regression
   test; fix the engine or harness; regenerate hashes; and rerun preflight and
   the audit. Resume only from the last clean checkpoint.
7. **Freeze a clean candidate.** Fingerprint it and add it to the league only
   after a clean audit. Exploiters remain explicitly labeled opponents.
8. **Evaluate without adaptation.** Use disjoint decks, controllers,
   checkpoints, paired seeds, and simulator perturbations. Do not tune on
   held-out results. The matrix `quality_gate` must verify the held-out split,
   complete matches, zero rejected actions, public-only actor inputs, and a
   clean simulator-exploit audit; win rate is evidence, not a pass criterion.
9. **Promote or loop back.** Promote only when quality, reproducibility,
   fidelity, and performance gates pass. A quality failure changes sampling;
   an exploit or parity failure returns to step 6.
10. **Commit the completed loop.** After tests, benchmark, exploit audit,
    artifact review, and documentation pass, create exactly one focused git
    commit and record its ID in the manifest/report. If remediation was
    required, commit only the simulator/harness fix after its reproducer and
    rerun are clean. Never commit an invalid checkpoint or contaminated data
    as a success.

```text
sealed manifest
    -> preflight
    -> actor-controlled rollout/training
    -> simulator-exploit audit
    -> clean frozen checkpoint
    -> no-adaptation held-out evaluation
    -> promote, or quarantine/fix/repeat
```

No league or frozen-evaluation result is valid with an unresolved simulator
exploit.

## Current retained RL audit

The older generalized checkpoints and held-out results are stale because they
were produced on earlier simulator revisions; they are not current strength
evidence. The actor uses public observations, while the critic is
training-only. The matrix records `actor_controls_actions=true` for neural
actor runs;
`tower_hp_before`, `tower_hp_after`, and `tower_hp_end` have different
per-decision versus terminal/cap-time meanings. The prototype `--trace-out` contains every decision; `troop_positions_end` and `tower_hp_end` are only terminal/cap-time snapshots. A finite `all_wins=true` result is not a universal-win claim.

The latest revision-pinned provisional PPO smoke on `87f9630` promoted a
checkpoint after 2,048 actor-controlled transitions with a stable revision
guard, public-only actor inputs, privileged training-only critic, full
decision tracing, and a clean simulator-exploit audit. It did not complete a
match, so no held-out strength claim is attached to it.

## Simulator requirements

The authoritative state is integer, serializable, replayable, deterministic,
and independent of wall-clock or thread order. V1 must support:

- match phases, elixir, hands/cycle, legal deployment, crowns, overtime,
  tiebreaks, King activation, terminal outcomes, and both-player actions;
- ground/air movement, bridges, terrain, buildings, repathing, steering,
  collision, mass, knockback, pulls, targeting, retargeting, and concealment;
- deploy/first-hit/attack/projectile timing, piercing, chains, splash, damage,
  status effects, death effects, spawns, splits, transformations, and
  lifetimes;
- spells and buildings with correct target classes, boundaries, decay,
  persistent areas, and temporal re-evaluation;
- public observations and authoritative legal-action masks.

Prefer shared data-driven components. Every engine failure becomes a minimized
deterministic regression scenario.

## Fidelity gates

Readiness is per card and per mechanic, never one aggregate accuracy score.
It requires:

- complete schema, rule, determinism, replay, invariant, and backend-parity
  coverage;
- no missing eligible cards or mechanic edges and no unresolved critical
  source conflicts;
- independent held-out evidence from at least two capture groups per
  version-sensitive mechanic, with predeclared sample counts and tolerances;
- controlled physical-lab provenance for causal, version-sensitive behavior;
- at least 98% held-out agreement for decision-critical outcomes, normally
  99%+ for common Hog interactions;
- confidence intervals, group counts, oracle error, exclusions, and
  first-divergence reasons beside every metric;
- no `training_ready` status while a required mechanic is missing, provisional,
  under-sampled, split-leaked, or below threshold.

Candidate, inferred-only, rejected, fake/offline-lab, and uninspected
observations cannot satisfy a held-out gate.

## Performance and backend

Keep the object-based Python engine as the reference oracle. The optimized
backend must support batched reset/step, legal masks, terminal/truncation,
deterministic seeding, and asynchronous reset while producing identical
canonical state and public-event hashes on the parity corpus.

The neural fast path is above the historical simulator benchmark on an RTX
2050 (about 4.69k actor-only and 3.77k full actor-selection decisions/s at
batch 16). On the current CPU host it reaches about 0.84k actor-only and
0.74k full decisions/s versus about 1.63k simulator steps/s, so CPU parity is
not yet met. The bounded trainer remains physics-bound; rollout-process and
larger-lane variants are benchmarked separately and are not enabled by
default when they trade memory or behavior-policy freshness. The
deployment-only decoding/layout changes preserve the PPO forward path and
exact selected-action parity on the regression workload. The preferred
large-self-play target is 5k–10k simulator environment steps/s.
The vector backend regression also checks state, event-log, and replay hashes
across consecutive card-play steps with privileged info disabled; both process
transports pass that parity check.

## Delivery order

1. Freeze V1 roster, ruleset, observation/action contract, and mechanic graph.
2. Improve the public actor, recurrent PPO training path, evaluator, and
   behavior-preserving inference fast path.
3. Run the RL phases with held-out splits, exploit audits, and reproducible
   checkpoints; promote only clean candidates.
4. Prove optimized/reference parity and meet the trainer-derived throughput
   gate.
5. Resume physical-lab evidence and close per-mechanic fidelity gates before
   declaring `training_ready` or completing the full V1 release.

## Definition of done

V1 is complete when the fixed player deck can play deterministic matches
against every legal eight-card deck in the eligible manifest; all required
mechanics pass generated, regression, parity, and independent held-out gates;
the physical-lab evidence chain is reproducible; the optimized backend is
hash-identical to the reference; and video, reference simulation, and
optimized simulation share the same versioned observation/action contract.
