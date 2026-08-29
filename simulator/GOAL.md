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

- `rulesets/v1.json` contains the 109-card opponent roster but remains
  `training_ready: false`.
- Generated coverage is deterministic and executable: the latest 1,135-case
  strict run passes execution, repeated hashes, complete roster coverage, and
  behavioral obligations.
- The RL suite is green (`158 passed`); the default non-audio/non-mining suite
  is `410 passed, 11 failed`, with failures limited to unavailable physical-lab
  assets, sandboxed forkserver sockets, and the absent KataCR submodule.
- Phase-0 physical-lab software is implemented, but physical evidence is
  intentionally deferred for the current RL-first execution path. No
  connected run has yet satisfied the evidence/readiness gates.
- The policy is a provisional research harness, not an any-deck player. The
  latest retained generalized actor is
  `outputs/simulator/training/generalized-coverage-ppo-v2.pt`, with reports
  `outputs/simulator/training/generalized-coverage-ppo-v2.json` and
  `outputs/simulator/training/generalized-coverage-ppo-v2-heldout-smoke.json`.
  It uses engine `reference-0.31.0`.
- On the benchmark host, batch-16 actor-only deterministic inference reaches
  about 3.61k decisions/s on the RTX 2050 versus about 1.54k simulator
  environment steps/s. The full actor-selection path reaches about 2.69k/s.
  The actor-only path now uses direct masked argmax decoding, a CPU
  channels-last raster layout, and tail-padding removal for sparse entity
  batches. In the current CPU-only training environment it reaches about
  255 decisions/s at batch 16, versus about 1.23k simulator steps/s; CPU
  fallback remains a diagnostic path, not the production throughput target.
- The current action contract has `WAIT`/`PLAY`, card slot, and placement but
  no learned wait-duration head; `WAIT` advances the fixed simulator decision
  interval. The proposed timing head remains a future architecture change.

These results establish plumbing and performance, not game strength or
sim-to-real fidelity.

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

The executable strategic curriculum stores each phase mix as deterministic
`sampling_mix` slots and records observed source counts per segment. These
labels select opponent/scenario provenance only; they never select the
learner's card, timing, lane, or placement.

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
   tracked-worktree state. Create the held-out split before collection.
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
   held-out results.
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

The latest retained generalized actor is
`outputs/simulator/training/generalized-coverage-ppo-v2.pt`; its training
report is `outputs/simulator/training/generalized-coverage-ppo-v2.json`, and
its six-deck held-out smoke report is
`outputs/simulator/training/generalized-coverage-ppo-v2-heldout-smoke.json`.
The actor uses public observations, while the critic is training-only.

The fixed deterministic-cycle regression is 8 wins, 0 losses, 0 draws,
0 truncated. The six held-out archetype smoke is 1 win, 5 losses, 0 draws,
0 truncated. These results do not establish the mission's any-deck policy
goal. The matrix records `actor_controls_actions=true` for neural actor runs;
`tower_hp_before`, `tower_hp_after`, and `tower_hp_end` have different
per-decision versus terminal/cap-time meanings. The prototype `--trace-out` contains every decision; `troop_positions_end` and `tower_hp_end` are only terminal/cap-time snapshots. A finite `all_wins=true` result is not a universal-win claim.

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

The neural fast path is already above the simulator on the benchmark host:
about 3.61k actor-only decisions/s and 2.69k full actor-selection decisions/s
versus about 1.54k simulator environment steps/s at batch 16. Continue
profiling full matches, dense swarms, projectile-heavy states, and observation
construction. The deployment-only decoding/layout changes preserve the PPO
forward path and exact selected-action parity on the regression workload. The
preferred large-self-play target is 5k–10k simulator environment steps/s.

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
