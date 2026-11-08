# Tactical learning experiments

This workflow tests consequence prediction, coordinated teacher plans and
adaptive practice against a matched targeted student fit. For the strategic
goal, current actor and later selection decisions, see [project status](../../STATUS.md).
The [October 20 experiment report](../experiments/v4_tactical_learning_results.md) records the
completed comparison and its failed family safeguards.

## Implemented additions

1. `tactical_outcomes.py` provides action-conditioned heads predicting public
   consequences at 2, 6 and 12 seconds. The native pilot supervises own/enemy
   tower damage following the **executed** action. The generic target builder
   supports tracked threat/defender survival, but those labels are masked in
   this pilot: the current public tensor contract has no stable track IDs.
   It never substitutes engine target locks or hidden hand information.
   True terminal measurements absorb later horizons; truncated tails do not.
2. `tactical_plans.py` offers bounded Cannon-to-Musketeer and air-intercept
   sequences with public geometry, current hand/elixir, fresh legal masks,
   conditional followups and interruption. It observes every 250 ms, resolves
   the current slot each time and advances only after actual acceptance.
   The existing cycle teacher handles other situations. The teacher itself
   must pass a paired match screen before a full plan student is fitted.
3. `tactical_training.py` offers an adaptive supervision curriculum. Every
   training episode is still visited once each epoch. Prior training-epoch
   losses and improvement change the weights of public tactical contexts;
   no calibration, development or reserved results set these weights.
   Priorities emphasize learnable contexts and retain a positive coverage
   floor. Population-normalized weights are bounded in `[0.25,4]`.
   This implementation does **not** reset arbitrary mid-match snapshots or
   feed stale trajectories into PPO.

Research precedents are
[future-measurement prediction](https://arxiv.org/abs/1611.01779),
[tactical portfolio search](https://davechurchill.ca/publications/pdf/combat13.pdf)
and [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html).
These motivate the adaptations; their benchmark gains are not predictions
for Clash Royale.

## Controlled local pilot

The experiment initializer is pinned by checkpoint bytes and full architecture.
The default corpus contains 90 complete training episodes: all 30 training
decks, with actor/cycle/plans drivers and three balanced training styles.
Each deck sees all three training styles across the drivers; this is not
the complete 30-deck × three-style × three-driver Cartesian product.
Thirty further calibration episodes have disjoint match origins and receive
no gradients. The ten reserved identities and both development styles are
excluded from collection.

All observations, legal masks, inactive rows, initial countdown and terminal
measurements are retained. Targets for alternative teacher actions are kept
separate from factual consequence labels. Plan continuations are acknowledged
only on trajectories that actually execute them. Raw public arrays, card
acceptance, requested/accepted parity, source hashes and origin identities
are validated on collection, resume and load.

Four independent arms reload the same initializer:

| Arm | Supervision |
|---|---|
| `control` | Cycle-teacher joint action NLL in public air/X-Bow contexts; initializer KL elsewhere |
| `outcomes` | Control plus 0.1 × factual tactical-outcome loss on active rows |
| `plans` | Coordinated-plan labels on the same corpus; same inactive KL |
| `curriculum` | Control with adaptive train-only tactical weights |

Every fitted arm has four epochs, Adam `5e-5`, 32-step TBPTT, two episodes per
batch and gradient clipping at 1. Both student and frozen initializer carry
their own independently unrolled recurrent states through every observation.
The anchor uses exact masked autoregressive KL at **every valid inactive
timestep**, including all legal card-conditioned placement distributions.
There is no duration objective: native WAIT remains one 250 ms decision.
Prelude and padding have no supervised loss. A zero-dropout actor is required
to keep the matched workloads deterministic. Every arm must receive exactly
the same real-frame and optimizer-step dose.

The sole candidate is epoch four; calibration never selects an epoch.
Auxiliary heads are exported separately, so ordinary actor inference and the
plain actor checkpoint contract remain compatible. PPO stays disabled.

## Evaluation and commands

Use `outputs/venv/`. The dated paths below identify the historical protocol;
choose a new directory for a new source revision. First run the small diagnostic:

```bash
outputs/venv/bin/python scripts/run_v4_tactical_learning.py \
  --out outputs/v4/tactical_learning_smoke_20261019 --smoke --epochs 1
```

Run the bounded full comparison in a new directory:

```bash
outputs/venv/bin/python scripts/run_v4_tactical_learning.py \
  --out outputs/v4/tactical_learning_20261019 --workers 3 --fit-workers 3
```

The paired teacher screen first compares cycle/plans on all 30 training decks
and two fresh seeds. Plans require one net additional win-equivalent, no net
Lavaloon/Bait/X-Bow loss and no other family losing three or more games.
Failure skips the plan student, with the reason preserved. The other arms
remain valid experiments on the same corpus.

The actor screen uses 60 fresh fixed-controller games and 60 held-out-style
games per actor. Qualification requires at least three net wins overall,
one Lavaloon gain, no net Bait/X-Bow loss, no other family losing three or
more, and no overall loss against development styles. Report comparisons
against both the initializer and the trained control. Qualified frozen actors
receive a separate five-seed, 150-game confirmation on training decks under
the same family safeguards. Only independently confirmed actors receive the
ten reserved decks under those same five seeds, completing a fresh 40-deck
catalog evaluation. Reserved scores never tune the models or select epochs.

These are engineering gates, not statistical proof. If no actor qualifies,
the reserved evaluation is explicitly not run. The runner does not promote
an actor. No external phone experiment is part of this workflow.

Resume the exact sealed experiment after interruption:

```bash
outputs/venv/bin/python scripts/run_v4_tactical_learning.py \
  --out outputs/v4/tactical_learning_20261019 --resume
```

Use `--stage prepare|teachers|collect|fit|evaluate` for a single stage.
`--stage fit --arm outcomes --resume` resumes one predeclared fit. Each
completed epoch saves optimizer/actor/auxiliary state; an interrupted epoch
restarts from the preceding completed epoch. Source or protocol changes
require a new directory.

Check progress and completion:

```bash
outputs/venv/bin/python scripts/run_v4_tactical_learning.py \
  --out outputs/v4/tactical_learning_20261019 --stage status
```

`completion_receipt.json`, `results.json` and `RESULTS.md` identify a completed
comparison. Intermediate teacher or training files alone do not.
