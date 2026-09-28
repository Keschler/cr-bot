# Deterministic Level-11 Simulator

This directory contains the headless, versioned Clash Royale simulator used
for deterministic testing, sim-to-real fidelity work, and reinforcement
learning. [GOAL.md](GOAL.md) is the authoritative roadmap and release
definition.

The current `v1` ruleset contains 124 definitions, including the complete
109-card eligible opponent roster, but remains `training_ready: false`. It is
an executable research ruleset, not a claim that every live-game mechanic has
been measured.

The current execution priority is RL-first: physical-lab evidence is deferred
while the actor, training loop, evaluation, and simulator throughput improve.
This permits provisional experiments only; it does not satisfy the fidelity
gates or authorize a `training_ready` claim.

The active V4 workflow compares reward-only match-start PPO with search
distillation from the same simulator bootstrap, plus a randomly initialized
PPO baseline. Frontier-initialized PPO is a separate experiment. A fresh
checkout has no promoted model checkpoint or measured policy strength. See
the [V4 training guide](../docs/v4_training.md) for the
reproducible commands and evaluation boundary.
The [concrete three-arm comparison plan](../docs/v4_comparison_plan.md) awaits
approval and lists the remaining weighting and evaluation-split prerequisites.

## Setup and tests

From `simulator/`, use the repository training environment:

```bash
../outputs/venv/bin/python -m pip install -e ..
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator --help
```

Run the focused RL tests and the default suite:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m pytest \
  tests/test_v4_model.py tests/test_v4_search_review_fixes.py
(cd .. && outputs/venv/bin/python -m pytest --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py)
```

The two audio/mining tests are excluded unless the change concerns those
systems. Generated outputs under `outputs/` are local artifacts and should not
be committed.

Treat each new `HEAD` as a new experiment revision: pin and record the commit
before training or evaluation, require the run to finish on that revision, and
rerun preflight, parity, and exploit checks after any intervening commit. Never
promote an artifact produced across mixed revisions.

## Quick start

Inspect the ruleset, run a deterministic match, and verify replay stability:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 ruleset
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 run --seed 7
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 check-determinism --seed 7
```

Generate and validate roster scenarios:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 generate-scenarios --per-mechanic 1 \
  --json-out outputs/simulator/generated-roster-scenarios.json
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
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
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator lab plan \
  --hog-cannon-only \
  --json-out outputs/simulator/fidelity_media/physical_lab/plan.jsonl
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator lab run \
  --experiment outputs/simulator/fidelity_media/physical_lab/plan.jsonl \
  --mode offline \
  --json-out outputs/simulator/fidelity_media/physical_lab/offline-summary.json
```

Offline runs remain `candidate_only`. The current two-phone campaign and
connected workflow are documented in
[the physical-lab goal](physical_lab/PHYSICAL_FIDELITY_GOAL.md).

### V4 actor on video or a phone

The visual frontend and the simulator use the same V4 actor checkpoint format.
First produce a checkpoint with the [bootstrap](../docs/v4_training.md), then
validate observation and action decoding on a recorded video:

```bash
../outputs/venv/bin/python run_prototype_live.py \
  --checkpoint ../outputs/v4/bootstrap_seed0/actor.pt \
  --video /absolute/path/to/gameplay.mp4 \
  --max-frames 30
```

A phone dry run requires an exact ADB serial and records decisions without
tapping the device:

```bash
../outputs/venv/bin/python run_prototype_live.py \
  --checkpoint ../outputs/v4/bootstrap_seed0/actor.pt \
  --serial YOUR_PHONE_SERIAL \
  --max-frames 100
```

Live taps require a reviewed phone calibration and the explicit execution
flags. See [RUN_PROTOTYPE_LIVE.md](RUN_PROTOTYPE_LIVE.md) for the device and
safety workflow. The earlier released prototype binary and checkpoint use a
different model format and do not represent the V4 actor.

### Fidelity readiness

Run the fail-closed readiness report only on sealed, provenance-carrying
reports:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator readiness \
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

Formal paired evaluation uses a durable manifest registry at
`outputs/simulator/evaluation/manifest_registry.json` by default. Once a
cell set is issued, it cannot be issued for another challenger; retiring it
also prevents reuse after a process restart. The stratified bootstrap requires a stratum
label for each block when reporting macro-averaged match improvement.

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
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 roster --require-coverage \
  --json-out outputs/simulator/roster-contract.json
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
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

Current search, distillation, PPO, and evaluation code lives in [rl/v4/](rl/v4/).
Shared tensor utilities and opponent pools live
in [rl/foundation/](rl/foundation/). See the [V4 training guide](../docs/v4_training.md)
for the bootstrap, dataset splits, recurrent context, and evaluation gates.
The reward-only PPO and frontier paths can use a pinned catalog of complete
base-card decks from RoyaleAPI and Deck Shop. Its train and evaluation decks
are disjoint. The random mixed pool remains a stress-test option, and frozen
V4 actors can serve as mirror opponents in PPO.

The earlier prototype and NumPy PPO learners, historical Stage-2 proof,
synthetic/timing curricula, and experiment-specific runners have been retired.
Use the V4 scripts under `scripts/` for training and evaluation; the simulator
CLI contains simulation, fidelity, and audit commands.

The phone/video frontend consumes V4 actor state dictionaries through a public
observation adapter. Supply a compatible checkpoint explicitly; the old
prototype checkpoint format is retired.

Legality masks remove impossible actions only. A privileged critic may use full
simulator state during training; the actor receives public observations. Teacher
and search outputs supply training targets. Ordinary PPO actions come from the
actor. WAIT advances the fixed simulator decision interval.

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

Benchmark the simulator on the current host from the repository root:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator \
  --ruleset v1 benchmark-vector --envs 16 --steps 100 --backend reference
```

## Automation and current limits

Recommended checks:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 audit --seeds 16 --max-ticks 6000 \
  --json-out outputs/simulator/audit.json
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator \
  --ruleset v1 soak --seeds 32 --tick-budget 1000000 \
  --json-out outputs/simulator/soak.json
```

The following remain explicitly provisional: internal tick timing, many
projectile and attack timings, exact collision/steering geometry, Cannon decay,
some tower targeting/activation behavior, and excluded Evolution/Hero/
Champion/tower-troop/ability mechanics. Missing evidence must produce a
failed or empty metric with recorded uncertainty.
