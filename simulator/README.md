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

The older live-phone prototype actor is
`outputs/simulator/training/prototype-public-primary-current/prototype.pt`.
It uses the public hand/current-state card head as the primary card policy,
fixing the verified stale recurrent cheap-cycle failure. It scores 4/6 on the
exact matrix and 16/24 (66.7%) on a paired shuffled-hand matrix, versus 2/24
(8.3%) for the former baseline. Evaluation is actor-controlled and public-only;
all candidate quality and simulation-exploit gates pass. The checkpoint's
`evaluation-exact6.json` and `evaluation-summary.json` contain the sealed
evidence. This is simulator prototype evidence, not live-game strength or a
V4 model-quality result. See the [V4 training guide](../docs/v4_training.md)
for the current V4 checkpoint and dataset status.

The retired prototype PPO experiment used 8 GPU lanes, a 512-decision rollout,
recurrent minibatches of 8, and a `2e-6` resume learning rate. It completed 69,632 clean
Phase-1 decisions without changing the retained held-out trajectory, at 336
end-to-end decisions/s with bulky decision traces disabled. Full diagnostics
measure about 128–145 decisions/s. Set
`PYTORCH_ALLOC_CONF=expandable_segments:True` for multi-segment runs on the
4 GB GPU; this fixes the observed allocator-fragmentation OOM. Exact evaluation
remains memory-safe at batch 6.

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

An independent bug-fix agent may commit simulator changes at any time. Treat
each new `HEAD` as a new experiment revision: pin and record the commit before
training or evaluation, require the run to finish on that revision, and rerun
preflight, parity, and exploit checks after any intervening commit. Never
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

Offline runs remain `candidate_only`. Device preparation, calibration,
capture, ingest, and the connected two-phone workflow are documented in
[PHYSICAL_FIDELITY_LAB.md](PHYSICAL_FIDELITY_LAB.md).

### Prototype actor on a phone

`dist/prototype-live-linux-x86_64` is a standalone Linux x86-64 CPU
executable. It contains the live controller, visual extractor, neural-network
runtime and assets, default policy, ADB, and FFmpeg.

```text
phone → live stream → visual extractor → public state
      → policy → card + placement → ADB → phone action
```

> [!NOTE]
> The binary is approximately **800 MB** because it bundles the runtime and
> all required dependencies. It uses **CPU inference only**, which is
> substantially slower than GPU inference. An RTX 2050 / Intel i9-13900H
> machine reached up to **4 FPS** during a live-action run. A GPU build is not
> distributed because its additional dependencies would make the download
> significantly larger.

#### 1. Make it executable

```bash
chmod +x dist/prototype-live-linux-x86_64
```

#### 2. Test on a video

For this older live-phone prototype, use its released checkpoint,
[`prototype.pt`](https://github.com/Keschler/cr-bot/releases/download/v0.5/prototype.pt):

```bash
./dist/prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --video /absolute/path/to/gameplay.mp4 \
  --max-frames 20
```

`--checkpoint` is optional when the bundled `prototype-fast-current` policy is
sufficient.

The repository includes a [1080×2400 sample gameplay video](../assets/pictures/gameplay.mp4).
It starts with roughly five seconds of loading and the versus animation, so
this smoke test samples every tenth source frame to reach the in-game section
without preprocessing the video:

```bash
./dist/prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --video ../assets/pictures/gameplay.mp4 \
  --frame-stride 10 \
  --max-frames 30 \
  --jsonl-out /tmp/prototype-gameplay.jsonl
```

`--max-frames` counts processed frames; with `--frame-stride 10`, these 30
frames cover approximately the first 10 seconds. Use `--frame-stride 1` to
process every frame, allowing enough frames to pass the opening animation.

#### 3. Dry run on a phone

Enable USB debugging, connect and authorize the phone, then use its exact ADB
serial:

```bash
./dist/prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --serial YOUR_PHONE_SERIAL \
  --max-frames 100 \
  --jsonl-out /tmp/prototype-live.jsonl
```

> [!NOTE]
> This is a dry run by default: it observes the game and records decisions,
> but never taps the phone. The default transport is a low-latency H.264
> stream; use `--adb-transport screenshot` for diagnosis.

#### 4. Enable live actions

> [!WARNING]
> Live actions are calibrated only for a **1080×2400 phone**. Do not enable
> taps on another resolution or layout without a separately reviewed
> calibration.

Validate dry runs first. Real taps require a phone-specific calibration file
and both confirmation flags:

```bash
./dist/prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --serial YOUR_PHONE_SERIAL \
  --calibration /absolute/path/to/phone-calibration.json \
  --execute \
  --confirm-live
```

Before every `PLAY`, the controller verifies the selected card and applies the
calibrated card and arena taps. `WAIT` never taps the phone. Press `Ctrl-C` to
stop the controller.

See [RUN_PROTOTYPE_LIVE.md](RUN_PROTOTYPE_LIVE.md) for the complete option
reference, troubleshooting, safety gates, and native-build instructions.

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
Shared models, tensor utilities, opponent pools, and scenario environments live
in [rl/foundation/](rl/foundation/). See the [V4 training guide](../docs/v4_training.md)
for the roadmap runners, dataset splits, recurrent context, and evaluation gates.

The earlier prototype/generalized training, league, rotation, shadow, diagnosis,
and benchmark runners have been retired. The `recurrent-prototype` CLI command
and `rl.legacy` package are removed. Use the V4 scripts under `scripts/` for
training and evaluation.

The phone/video frontend still supports deployed public-actor checkpoints through
[the inference reader](rl/foundation/inference_checkpoint.py). This reader keeps
checkpoint architecture, engine, ruleset, and public-observation validation; it
does not expose a training command. This compatibility path does not make old
checkpoints V4 checkpoints or change the live observation contract.

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

Use the V4 roadmap reports for current policy throughput. Historical prototype
throughput is not a prediction for the current search and training workload.

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
