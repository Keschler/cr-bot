# Strategy for a strong Hog 2.6 policy

This design describes public-observation learning tools and evidence-driven
choices for Hog 2.6. The strategic goal, current policy identity and latest
decisions are maintained in [project status](../../STATUS.md). The October 2026
experiments below motivate the design; their gates and budgets are local
research choices.

## What the evidence says

The [catalog experiment](../experiments/v4_initializer_coverage_results.md) reached 92/400
wins, or 23%, against 40 source-linked decks controlled by five scripts. Deck
variety did not provide equivalent variety in decision making. The later
[memory and defense repair](../experiments/v4_defense_repair_results.md) retained recurrent
memory. Its defense-label student won 48/150 fresh training-deck games versus
36/150 for the initializer, but lost four additional Bait games and failed the
declared retention rule. Neither new student replaced the selected actor.

The earlier reward-only PPO pilot regressed and its critic explained little
return variance. At 250 ms per decision, the old gamma/lambda combination
gave direct GAE credit a half-life of about 2.8 seconds. That evidence did not
support scaling the same configuration.

Actors use the exact model configuration recorded in their fit receipt,
public observations, carried recurrent memory and fixed 250 ms WAIT semantics.
Use the [checkpoint and replay guide](../workflows/checkpoints-and-replays.md) to resolve a receipt
and inspect the evaluated games.

Implementation results, failed value validation, and the engine-version
transition are recorded in the [local results](../experiments/v4_human_strength_results.md).
The subsequent [tactical-learning experiment](../workflows/v4_tactical_learning.md) adds
separately controlled consequence-head, coordinated-plan and adaptive-practice
arms with full-history retention and conditional reserved-catalog evaluation.
The [completed tactical results](../experiments/v4_tactical_learning_results.md) favor the
targeted control within that pilot; prediction heads and adaptive practice
did not beat it, and Bait regressions prevented any actor from qualifying.
The October 18 teacher confirmation passed narrowly (96/150 versus
93/150), qualifying a targeted student experiment under its declared gate.
The tactical-fit protocol tested retention of the previous Bait behavior
while preserving full recurrent and fixed-WAIT semantics.

## Implemented learning options

1. **Repair a measured tactical failure.** The opt-in `cycle_teacher_label`
   keeps the existing defense teacher and adds a cheap legal cycle when air
   is threatening, Musketeer/Fireball/Ice Spirit are absent from hand, and
   the play leaves at least four elixir. This tests the observed air-defense
   hand stall. It does not change historical teachers or ordinary actor
   inference. First compare teachers, then confirm on fresh seeds. Fit a
   student only after the teacher passes both screens. A useful teacher is
   not sufficient evidence for a useful student.
2. **Train against different decisions as well as decks.** The implemented
   styles vary reserves, commitment timing, formations, counterpushes and
   responses to visible commitments. Train styles are `tempo`, `patient`,
   and `counterpush`; held-out development styles are `adaptive` and
   `split-pressure`. Training openings cover every one of the 30 training
   decks under each train style. Ten reserved deck identities remain out of
   training and checkpoint selection. The historical benchmark stays
   reproducible when no style is supplied. These are still scripts, not
   simulated humans or proof of greater opponent strength.
   Generated opening coverage is not completed training coverage: inspect the
   collector's per-entry visits and complete outcomes before claiming that a
   finite training run reached every matchup.
3. **Preserve a population of learned opponents.** The implemented population
   pins every snapshot's bytes, complete architecture, deck, inference mode,
   sampling seed, generation and train/dev membership. Historical snapshots
   are append-only. A finite schedule visits every eligible member before
   weighted extras; optional difficulty feedback accepts only completed train
   matches and retains a uniform coverage floor. Per-match hidden state,
   public estimator and sampling generator are independent. Current learned
   opponents support Hog 2.6 mirrors only. Multi-deck learned opponents need
   a general hand encoder and separately trained policies; that extension
   remains work to do, rather than relabeling a Hog actor as a Lava actor.
4. **Make outcome-driven learning viable before scaling it.** The new
   `anchored-match-start` mode explicitly combines PPO with a frozen
   initializer action anchor. This retention term is not expert human data.
   Optional critic warmup fits only detached actor features from genuinely
   completed reset-to-terminal episodes. Separate match origins validate the
   value fit; cuts, incomplete prefixes and unfinished tails receive no
   supervised terminal target. A failed value gate prevents actor updates.
   Longer gamma/lambda values are experimental settings, not new universal
   defaults. For example, `.999/.999` gives direct GAE credit a half-life of
   roughly 87 seconds, while value bootstrapping still matters at boundaries.
5. **Measure transfer.** Freeze a candidate before fresh development matches;
   retain per-family safeguards and report uncertainty, illegal actions,
   genuine completion, and opponent warnings. Only an eligible frozen actor
   proceeds to reserved-deck evaluation. Later assess different human decks,
   styles and visual input errors in real-game transfer tests. Simulator
   mechanics, unusual card forms and deployment latency must be validated
   for the claims being made. External phone experiments require the concrete
   reviewed protocol and confirmation specified by `simulator/AGENTS.md`.

The user confirmed on 2026-10-07 that **no curated expert demonstrations are
available**. Human-data training is disabled. Future demonstrations must
identify the player/deck, reconstruct public observations and legal timing,
validate card/placement labels against video, split by player and match, and
retain full recurrent sequences. Unverified recordings must not be called
expert labels.

## Recorded October 2026 protocol

The pre-outcome plan is
`outputs/v4/human_strength_20261007/plan.json`. The first teacher screen uses
all 30 training decks and two fresh seeds: 60 complete games per teacher.
Cycle must gain at least two win-equivalents overall and one against
Lavaloon, with no family losing three or more. Passing triggers a five-seed
confirmation, 150 games per teacher, requiring at least three net wins and
one Lavaloon gain under the same family rule. Learner legality and native
termination are mandatory. These are bounded engineering gates, not proof
of a statistically established improvement across arbitrary opponents.

The October 18 engine update changes spirit movement toward towers. The
resumed current-engine screen and conditional confirmation are separately
sealed under `outputs/v4/human_strength_20261018/`; the October 7 outputs are
preserved as historical evidence. Use a new output directory when the source
differs, rather than attempting to resume the old commands below.

The screening runner records paired initial states, controller and policy
seeds, public/runtime contracts, requested and accepted card identities,
action traces, checkpoint hashes, per-file source hashes and sealed raw-result hashes.
Opponent warnings remain in the primary result and are removed symmetrically
only for a sensitivity report. Results resume only under the same protocol
and source identity. The source scope explicitly excludes capture/frontend
code and source tests, and includes the native runtime's local dependencies.

## Runner examples

Use the existing `outputs/venv/` environment. Commands below illustrate the
October protocol. Use new output directories for a new source revision;
historical sealed directories are not reusable under changed source. These
commands do not promote an actor.

Generate all 90 deck/style training openings:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_raw_ppo_starts.py \
  --profile diverse-real --pool-seed 2026100820 --matches-per-deck 1 \
  --out outputs/v4/diverse_starts_seed2026100820.pt
```

Compare the opt-in teacher with the existing defense teacher; save the
declared `cycle_screen.gate` object as a JSON file before running:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_policy_development.py \
  --experiment cycle --seeds 2026100801 2026100802 --workers 3 \
  --gate-json outputs/v4/human_strength_20261007/cycle_screen_gate.json \
  --out outputs/v4/human_strength_20261007/cycle_screen
```

Evaluate a pinned actor under the new train styles:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_policy_development.py \
  --experiment diversity --styles tempo patient counterpush --style-split train \
  --actor-json outputs/v4/selected_actor.json \
  --seeds 2026100811 --workers 3 \
  --out outputs/v4/human_strength_20261007/diversity_train
```

The actor JSON above is the selected actor receipt, including its hash and
complete model configuration. `--style-split dev --styles adaptive
split-pressure` provides the separate development style screen on fresh
training-deck matches; do not place these styles in training openings.

For critic-only diagnostics, use the learning-curve runner with `--updates 0`,
`--training-source anchored-match-start`, a positive `--bc-coef`,
`--reward-mode terminal`, `--episode-cap 1440`, and
`--critic-warmup-epochs 8`. Supply the exact actor `--model-config`, diverse
`--train-entries`, declared fresh evaluation opponents/seeds and a new output
directory. Warmup defaults to four lanes and 1,440 decisions; its default
gate requires at least four complete episodes and both wins and losses in
each origin-disjoint subset, validation explained variance at least .05,
and lower validation MSE. Actor updates remain disabled in this diagnostic.
With an update schedule, failure of the same gate still stops before PPO.

For learned-opponent starts, serialize an `OpponentPopulation.to_dict()`
manifest with explicit train/dev members, then run:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_opponent_population.py \
  --population /absolute/path/population.json --pool-seed 2026100821 \
  --episodes 30 --out outputs/v4/population_starts_seed2026100821.pt
```

`--train-feedback` optionally reads completed typed train results. This
command guarantees member coverage in the generated schedule, verifies
checkpoint contracts and writes its hash manifest. It does not train the
opponents. Mix these entries with broad catalog/style openings using an
explicit recorded workload; Hog mirrors cannot replace broad deck coverage.

Choose any scale-up from fresh screens under a newly sealed protocol. Fresh wins and family
retention decide whether to invest in a student fit or PPO; training loss,
reward and synthetic test success alone do not select a stronger actor.
