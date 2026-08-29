# Deterministic Level-11 Simulator

This directory contains the headless, versioned Clash Royale simulator used
for deterministic testing, sim-to-real fidelity work, and reinforcement
learning. [GOAL.md](GOAL.md) is the authoritative roadmap and release
definition.

The current `v1` ruleset declares the complete 109-card eligible opponent
roster, but remains `training_ready: false`. It is an executable research
ruleset, not a claim that every live-game mechanic has been measured.

The current execution priority is RL-first: physical-lab evidence is deferred
while the actor, training loop, evaluation, and simulator throughput improve.
This permits provisional experiments only; it does not satisfy the fidelity
gates or authorize a `training_ready` claim.

## Setup and tests

From `simulator/`, use the repository training environment:

```bash
../capture/.venv-train/bin/python -m pip install -e ..
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator --help
```

Run the focused RL tests and the default suite:

```bash
pytest tests/test_rl_model.py tests/test_rl_prototype.py
pytest --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py
```

The two audio/mining tests are excluded unless the change concerns those
systems. Generated outputs under `outputs/` are local artifacts and should not
be committed.

## Quick start

Inspect the ruleset, run a deterministic match, and verify replay stability:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 ruleset
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 run --seed 7
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 check-determinism --seed 7
```

Generate and validate roster scenarios:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 generate-scenarios --per-mechanic 1 \
  --json-out outputs/simulator/generated-roster-scenarios.json
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 validate-generated \
  outputs/simulator/generated-roster-scenarios.json \
  --json-out outputs/simulator/generated-roster-validation.json \
  --workers 4 --require-complete
```

The strict matrix is fail-closed on execution, determinism, roster coverage,
and behavioral-obligation failures.

### Physical-fidelity lab

Physical-lab evidence is deferred for the current RL-first pass. The software
harness remains available for the later release gate and can be exercised
offline before connecting devices:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator lab plan \
  --hog-cannon-only \
  --json-out outputs/simulator/fidelity_media/physical_lab/plan.jsonl
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator lab run \
  --experiment outputs/simulator/fidelity_media/physical_lab/plan.jsonl \
  --mode offline \
  --json-out outputs/simulator/fidelity_media/physical_lab/offline-summary.json
```

Offline runs remain `candidate_only`. Device preparation, calibration,
capture, ingest, and the connected two-phone workflow are documented in
[PHYSICAL_FIDELITY_LAB.md](PHYSICAL_FIDELITY_LAB.md).

### Fidelity readiness

Run the fail-closed readiness report only on sealed, provenance-carrying
reports:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator readiness \
  outputs/sim/calibration-fidelity.json \
  outputs/sim/heldout-fidelity.json \
  --json-out outputs/sim/training-readiness.json \
  --min-heldout-observations 20 \
  --min-heldout-agreement-rate 0.98 \
  --min-heldout-groups 2
```

The controlled physical lab is primary for causal, version-sensitive
measurements. Public-video mining is supplementary discovery and
validation; candidates and inferred-only observations never satisfy a
held-out gate.

## What is implemented

- Integer authoritative state: microseconds, milli-elixir, milli-tiles,
  integer HP/damage, stable UIDs, canonical state/event/replay hashes, and
  deterministic RNG.
- Regulation, overtime, elixir phases, crowns, King activation, tiebreaks,
  hands/cycle, simultaneous actions, legal placement, rejection events, and
  complete unattended matches.
- Ground/air navigation, bridges, terrain, buildings, repathing, targeting,
  retargeting, collision, mass, knockback, pulls, projectiles, splash,
  piercing, chains, statuses, death effects, spawns, splits, transformations,
  and building lifetimes.
- Data-driven card components for the current V1 roster, including swarms,
  passive spawners, persistent areas, shields, stealth, burrow, charges,
  dashes, hooks, jumps, reflections, and compound death streams.
- Public V1/V2 observation contracts, authoritative legality masks, a
  two-player `SimulatorEnv`, and deterministic vector/process wrappers.
- Generated card/mechanic scenarios, pairwise interaction coverage,
  invariant/fuzz/soak checks, replay-cache mining, fidelity corpora, and
  first-divergence reports.

Implemented means dispatchable. `fidelity_ready` and `training_ready` require
the evidence gates below.

## Architecture

```text
ruleset -> authoritative BattleState/engine -> SimulatorEnv
                         |                         |
                         v                         v
                  public observations        actions/rewards
                         |
                         v
                 recurrent public actor

full simulator state -> separate training-only critic
```

`BattleState` is separate from
`cr_bot.domain.game_state.GameState`, which is a lossy observed DTO produced
from video. The actor never receives exact opponent hand/elixir/cycle,
targets, cooldowns, RNG state, or other privileged physics fields.

The object-based Python engine remains the reference oracle. Optimized
backends must preserve canonical state and public-event hashes across reviewed
scenarios, randomized streams, save/reload boundaries, and long seeded matches.

## Ruleset and truth contract

The fixed V1 player deck is:

```text
Hog Rider, Cannon, Musketeer, Skeletons,
Ice Golem, Ice Spirit, Fireball, The Log
```

The opponent manifest covers eligible base cards released before
`2025-12-01`. It stores card IDs, release provenance, Level-11 data, mechanic
dependencies, and unresolved conflicts. The ruleset and engine are immutable
within a version; behavior-changing edits require a new version.

Check roster coverage and source reconciliation with:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 roster --require-coverage \
  --json-out outputs/simulator/roster-contract.json
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 reconcile-data \
  --json-out outputs/simulator/card-data-reconciliation.json
```

Unresolved behavior is recorded in
[UNKNOWN_BEHAVIORS.md](UNKNOWN_BEHAVIORS.md) and
[unknown_behaviors.json](unknown_behaviors.json). Do not fill an evidence
gap with a guessed constant.

## Policy boundary and observations

`PolicyObservationV1` preserves the sealed vision contract:

```text
board          float32 [21, 32, 18]
global_vector  float32 [768]
spatial_masks  bool    [4, 32, 18]
legal_play     bool    [4, 32, 18]
legal_wait     bool
```

`spatial_masks` preserves the imported V1 tensor. New training code should use
the simulator-provided, center-based `legal_play` mask, which also accounts
for elixir, dynamic territory, footprints, and occupied structures.

`PolicyObservationV2` adds public entity rows and masks without changing V1.
It does not expose private opponent state. The public action is either
`Wait` or `Play(card_slot=0..3, cell=(column, row))`.

The active actor's Transformer processes public entity tokens. Hand cards are
represented as one-hot card-table features projected independently for each
hand slot. `card_embedding` is an embedding of the four slot positions, not a
learned embedding of card identities, and hand cards are not Transformer
entity tokens.

## Neural policy and training

The mainline policy is a factorized recurrent PPO actor:

```text
public raster/global/entity features
        -> entity Transformer
        -> GRU (~256 hidden units)
        -> mode -> card slot -> card-conditioned placement
```

Legality masks remove impossible actions only. A separate privileged critic
may use full simulator state during training. The actor controls every
ordinary PPO action; counter policies and teachers are explicit opponents,
baselines, or short-lived auxiliary labels.

The current action contract has no learned wait-duration head: `WAIT` advances
the fixed simulator decision interval. Adding learned timing remains a future
behavior-changing architecture step.

The runnable neural prototype is in [rl/prototype.py](rl/prototype.py).
The generalized runner adds curriculum sampling, held-out provenance,
historical checkpoints, and PFSP bookkeeping:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../capture/.venv-train/bin/python -m rl.prototype train \
  --allow-provisional --updates 1 --envs 2 --horizon 128 \
  --device auto \
  --checkpoint-out outputs/simulator/training/recurrent-prototype.pt

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../capture/.venv-train/bin/python -m rl.prototype evaluate \
  --checkpoint outputs/simulator/training/recurrent-prototype.pt \
  --episodes 1 --policy actor --device auto
```

The NumPy trainer remains a lightweight simulator-regression smoke test. It
is not the mainline neural training path.

The generalized runner supports explicit public-feature switches including
`--explicit-hand-features`, `--direct-public-action-features`,
`--direct-public-card-features`, `--contextual-public-card-features`,
`--direct-public-mask-features`, `--direct-public-context-features`, and
`--direct-public-slot-card-features`. The last option is for compatible fresh
architectures and must not be added when resuming an incompatible checkpoint.

Historical/self-play training can be side-balanced by running matched
generalized segments with `--target-player 0` and `--target-player 1`. The
trainer changes only world deck ordering, keeps the public actor contract
unchanged, and records `target_player`, `actor_player`, and `opponent_player`.

### Generalized opponent training and held-out evaluation

The generalized runner and evaluation commands below reproduce the recorded
RL audit. The generated checkpoint and reports are local artifacts and are not
included in this checkout; rerun training before using those paths. The
recorded run used engine `reference-0.31.0`.

Evaluate the six-deck held-out smoke audit with the same checkpoint, seed,
strategy, cap, and exclusion report every time:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../capture/.venv-train/bin/python -m rl.generalized evaluate \
  --checkpoint outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  --policy actor --device auto --batch-size 6 --seed 0 \
  --max-decisions 1200 \
  --archetypes aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal \
  --strategies deterministic-cycle --seeds 10000 \
  --training-report outputs/simulator/training/generalized-coverage-ppo-v2.json \
  --no-match-results \
  --json-out outputs/simulator/training/generalized-coverage-ppo-v2-heldout-smoke.json
```

Training and evaluation artifacts include the Git `code_revision` used to
produce them, plus whether tracked source was dirty.

The retained six-deck held-out smoke audit is six archetype variants × one
strategy × one seed. The `deterministic-cycle` archetype is deliberately absent
because its template is the fixed learner deck; evaluate that fixed regression
separately. A broader smoke audit is `6 × 6 × 4 = 144` matches.

Use `--training-report` to prove split exclusions, and require
`held_out_audit.disjointness_verified=true` before calling a result held out.
`--no-match-results` keeps large reports compact. When resuming a sidecar whose
cursor starts at segment 16, use `--segment-offset 16` if sidecar inference is
unavailable.

The matrix report is schema version 2 with kind
`recurrent_public_ppo_evaluation_matrix`. `policy_mode=actor` and
`actor_controls_actions=true` identify neural actor evidence;
`actor_controls_actions` is `true` only for the neural actor. Counter-policy
rows are diagnostics, not checkpoint quality.

For robustness evaluation, pass a `DomainRandomizationConfig` as
`domain_randomization` to `evaluate_checkpoint_matrix`. Such runs are forced
through the sequential path, record both the declared profile and sampled
episode variant, and remain separate from baseline throughput and promotion
evidence.

When `--max-decisions` is omitted, the evaluator derives the complete
regulation-plus-overtime cap from the ruleset. If that cap is not divisible by
the checkpoint's training sequence length, evaluation drops only that
training-only chunking setting; recurrent hidden state is still carried at
every decision, so full-match evaluation is not artificially truncated.

Every prototype and matrix report includes `simulation_exploit_audit`. To audit
an existing report and optional full decision trace, run:

```bash
PYTHONPATH=.:..:../src \
../capture/.venv-train/bin/python -m rl.exploit_audit \
  outputs/simulator/training/evaluation.json \
  --trace outputs/simulator/training/evaluation-trace.json
```

Matrix reports also include a fail-closed `quality_gate`: only verified
held-out actor runs with complete matches, zero rejected actions, public-only
inputs, and a clean exploit audit can pass it. Win rate remains evidence, not
a gate. Exit status 2 means the artifact is flagged, invalid, or failed this
gate and must be quarantined before promotion.

`target_play_trace` contains target `PLAY` attempts only. Prototype
`--trace-out` contains every decision. `tower_hp_before`, `tower_hp_after`, `tower_hp_end`
distinguish per-decision snapshots from final or cap-time
snapshots; `crowns_end` reports final world-player crown totals; and
`troop_positions_end` is likewise terminal/cap-time data, not a continuous
trajectory.

Recorded RL results:

| Evaluation | Result |
| --- | --- |
| Fixed deterministic-cycle regression | 8 wins, 0 losses, 0 draws, 0 truncated; `actor_controls_actions=true` |
| Six held-out archetype variants | 1 win, 5 losses, 0 draws, 0 truncated; `held_out_audit.disjointness_verified=true` |
| Previous actor on the same six held-out cells | 0 wins, 6 losses, 0 draws, 0 truncated |
| Six archetypes × six strategies × one seed | 1 win, 35 losses, 0 draws, 0 truncated |

The self-play matrix uses the fixed prototype player deck and reports
`held_out=false`; it is a separate identity/regression check, not diverse-deck
evidence. A finite 100% result against one script is not a universal-win
claim.

## RL phases

Apply the gated loop below after every segment. The scripts control only the
opponent and never choose the learner's move.

1. **Basic mechanics — approximately 1–5M decisions.** Use short generated
   scenarios. Sample 25% isolated offense/tower pressure, 25% ground defense,
   20% air defense, 15% spell situations, and 15% kiting/cycling/elixir
   situations. Randomize lane, card order, elixir, tower HP, troop timing, and
   placement. Judge success from the resulting game state, not a “correct card”
   label.
2. **Scripted curriculum — additional 10–30M decisions.** Sample 20% Phase-1
   rehearsal, 20% passive/random-legal, 20% simple win-condition, 20%
   reactive defensive/aggressive, and 20% randomized tempo/placement scripts.
3. **Meta-deck training — additional 30–100M decisions.** Expand from 5
   representative archetypes to 20 decks, 50 decks, and the full validated
   meta pool. Sample 35% uniform archetypes, 30% weakness-prioritized
   matchups, 20% curriculum rehearsal, and 15% randomized variants. Keep
   unseen validation decks separate.
4. **Historical self-play — additional 100–300M decisions.** Sample 30%
   scripted/meta anchors, 30% PFSP historical policies, 20% newest frozen
   main, 10% random historical checkpoints, and 10% exploiters/adversarial
   policies. Train both world sides with matched `--target-player` runs; do
   not use only latest-policy mirror play.
5. **Small league — roughly 300M–1B+ cumulative decisions.** Use a main
   learner, main exploiter, league exploiter, 16–32 frozen historical
   policies, a payoff matrix, PFSP matchmaking, periodic exploiter resets, and
   fixed scripted anchors. Add PSRO only for demonstrated non-transitive
   cycles.
6. **Frozen evaluation.** Run paired seeds on both sides across unseen decks,
   controllers, historical checkpoints, and simulator perturbations. Report
   per-deck outcomes, confidence intervals, tower/crown outcomes, traces,
   rejected/fallback counts, and regressions.

The default strategic curriculum switches on cumulative learner decisions at
5M, 35M, 135M, and 435M, serializes the phase percentages as deterministic
`sampling_mix` slots, and records source counts plus decision cursors per
segment. Custom schedules without decision boundaries retain segment-cursor
fallback. A source is opponent/scenario provenance only; it never chooses the
learner's action. Generalized reports persist the cumulative decision cursor,
so resume remains correct even if the next run changes lane count or horizon.

`LeagueOrchestrator` retains directional payoff and rating state, exposes PFSP
sampling at the current cursor, and fails closed when a configured periodic
exploiter reset has no callback. The callback owns the actual learner reset;
the league records the completed reset in its serializable run state.

### Implementation loop for RL scaling and frozen evaluation

```text
seal manifest and held-out split
    -> preflight reset/replay/public-boundary/mask/parity checks
    -> actor-controlled rollout and PPO update
    -> simulator-exploit audit
    -> freeze and fingerprint clean checkpoint
    -> paired, no-adaptation held-out evaluation
    -> promote, or quarantine/reproduce/fix and repeat
```

For every loop, record the code/ruleset/engine hashes, observation/action
contract, architecture, optimizer, device/backend, decks/opponents, seeds,
budget, split, terminal versus truncated outcomes, action sources, masks,
rejected/fallback actions, and throughput. Create the held-out split before
collection and keep ordinary PPO at `expert_execution_probability=0`.

The simulator-exploit audit is mandatory. Check reward hacking, truncation
stalling, reset/terminal errors, stale observations or masks, hidden
information leaks, collision/navigation/timing loopholes, opponent-controller
artifacts, illegal-action accounting, and reference/optimized divergence.
Reward without terminal wins, impossible transitions, action collapse, abnormal
cap-time behavior, or controller-specific wins are flags.

A flag marks the run `simulation_exploit`/`invalid`. Preserve the smallest
reproducing seed and trace, quarantine contaminated data, classify and fix the
engine or harness, add a regression test, regenerate hashes, and rerun
preflight plus the audit. Resume only from the last clean checkpoint.

After a clean loop, run tests, benchmarks, exploit audit, artifact review, and
the documentation check, then create exactly one focused git commit and record
its ID in the run manifest/report. If remediation was needed, commit only the
simulator/harness fix after its reproducer and rerun are clean. Never promote
or commit an invalid checkpoint or contaminated training data as a success.

## Deterministic tick order

Each physics step is ordered as follows:

1. elixir regeneration;
2. actions and deployments;
3. deployment, status, persistent-area, and lifetime clocks;
4. target invalidation, acquisition, and retargeting;
5. movement and collision separation;
6. attacks and projectile creation;
7. projectile movement and impacts;
8. damage, statuses, deaths, and death effects;
9. towers, crowns, and victory resolution;
10. phase and match-clock transitions.

Changing this order is a mechanics change and requires new regression evidence.

## Scenarios and replay

Scenarios contain immutable schema/ruleset/engine identities, a seed, decks,
scheduled actions, a bounded stop tick, an evidence split, and oracle metadata.
Mined mid-match cases may start from a complete canonical `BattleState`.

Every failure is minimized into a deterministic regression scenario. A
repeated hash is not enough for a mechanic case: the required branch event or
final-state obligation must also occur.

## Sim-to-real fidelity

Evidence roles are assigned before evaluation:

```text
calibration -> fit provisional mechanics
validation  -> select implementations
regression  -> preserve reviewed failures
heldout     -> final unbiased reporting
```

Each observation carries source, method, confidence, capture group, split,
media/cache hashes, measurement tolerance, and frame/time provenance. Groups
are split before inspection; material used for tuning or debugging can never
become held out.

Use `mine-video-truth`/`compile-video-truth` for supplementary public-video
discovery and the `physical_lab` workflow for controlled deployment, timing,
damage, targeting, collision, spell, spawn, and lifecycle measurements.
Ambiguous or rejected observations remain auditable and cannot be promoted.

Readiness requires, per card and mechanic:

- complete rule/schema, determinism, replay, invariant, and backend-parity
  coverage;
- no missing required mechanic edge or critical source conflict;
- at least two independent held-out groups with predeclared counts and
  tolerances;
- physical-lab provenance for causal, version-sensitive behavior;
- at least 98% held-out agreement for decision-critical outcomes (normally
  99%+ for common Hog interactions);
- confidence intervals, oracle error, exclusions, and first-divergence data;
- no unresolved provisional or split-leaked requirement.

There is no single meaningful “overall accuracy” score.

## Performance

Benchmark the simulator and actor on the same host and batch/workload:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 benchmark-vector --envs 16 --steps 100 \
  --backend reference
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m rl.benchmark_policy \
  --batch-size 16 --device auto
```

The current benchmark host measured approximately:

| Path | Throughput |
| --- | ---: |
| Simulator, 16 lanes, CPU reference | 3.06k environment steps/s |
| Actor-only deterministic fast path, batch 16, CPU | 1.44k decisions/s |
| Actor-only deterministic fast path, batch 16, RTX 2050 | 3.61k decisions/s |
| Full actor selection, batch 16, RTX 2050 | 2.69k decisions/s |

The historical accelerated-host actor result clears the historical simulator
rate. On the current CPU host, the actor is about 1.44k decisions/s versus
3.06k simulator steps/s, so CPU parity remains an open performance gate. The
deployment path avoids belief heads and distribution normalization, resolves
`WAIT` before card/placement decoding, uses a channels-last raster, removes
masked entity tails, and caps CPU intra-op parallelism at four threads. The
single-observation deployment callers also bypass one-element host stacking
and prepare the raster layout at the observation boundary. The
PPO/reference forward path and selected-action parity are unchanged. The
vector-backend regression checks state, event-log, and replay hashes for both
process transports across consecutive steps with privileged info disabled.

## Automation and current limits

Recommended checks:

```bash
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 audit --seeds 16 --max-ticks 6000 \
  --json-out outputs/simulator/audit.json
PYTHONPATH=.:..:../src ../capture/.venv-train/bin/python -m simulator \
  --ruleset v1 soak --seeds 32 --tick-budget 1000000 \
  --json-out outputs/simulator/soak.json
```

The following remain explicitly provisional: internal tick timing, many
projectile and attack timings, exact collision/steering geometry, Cannon decay,
some tower targeting/activation behavior, and excluded Evolution/Hero/
Champion/tower-troop/ability mechanics. Missing evidence must produce a
failed or empty metric with recorded uncertainty.
