# Replay-branch policy corrections

See [project status](../../STATUS.md) for the current actor and latest experiment
decision. The [October 29 report](../experiments/v4_replay_correction_results.md) preserves the
first completed run: four accepted local corrections, with no full-match
improvement. Commands here describe the reusable pipeline; a new source or
protocol requires a new output directory.

The October replay review identified four overlapping skills to repair:

| Skill | Examples | General correction mechanism |
|---|---|---|
| Response timing | X-Bow connects while a defender is affordable | Review the first public siege response opportunity, with alternative interception cells |
| Elixir and cycle planning | A cheap play delays the needed defense | Review spending before public threats and before an in-hand air answer becomes unaffordable |
| Counter effectiveness and placement | Cannon never reaches Princess | Propose reachable ranged-ground counters and compare their actual consequences |
| Coordinating threats | A ground response consumes the air-defense budget | Compare alternatives with both players continuing, preserving legitimate tanks, stalls and emergency ground defenses |

These flags select review opportunities, not proven mistakes. The pipeline
does not ban low-elixir Ice Golem, Cannon tanking, cheap cycling or Spirit
stalls. It makes no permanent strategic changes to the legal action masks.

## Workflow

Use the selected actor and current runtime to collect **fresh actor-only,
complete trajectories**. Default collection covers all 30 training decks:
30 training episodes and 30 calibration episodes with separate origins and
training-style rotation. The ten reserved decks are excluded from collection
and candidate selection.

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_replay_corrections.py run \
  --out outputs/v4/replay_corrections_20261029 --workers 3
```

The stages can also run separately: `prepare`, `collect`, `search`, `fit`,
`evaluate`, `report`. Continue an interrupted run with its exact source,
checkpoint and settings:

```bash
PYTHONPATH=.:src outputs/venv/bin/python scripts/run_v4_replay_corrections.py run \
  --out outputs/v4/replay_corrections_20261029 --resume
```

Inspect completion without importing models:

```bash
outputs/venv/bin/python scripts/run_v4_replay_corrections.py status \
  --out outputs/v4/replay_corrections_20261029
```

Preparation seals source bytes, actor bytes and complete model configuration,
job origins, search budget and training settings. Changes require a new output
directory. Original evaluation seals are preserved; adding new simulator
modules does not retroactively qualify the old runtime under a new source hash.

## Search and evidence

The bounded default reviews up to 36 training and 12 calibration decisions.
It balances categories, catalog archetypes and origins with seeded tie breaks.
Candidate records include deck and context coverage. Factual public tower
damage helps prioritize review, but never establishes that an action is wrong.
Public siege onsets reserve an early affordable defensive opportunity; these
are not privileged target-lock labels. Earlier spending is selected from
public threat onsets and resource-starvation transitions. Future public
observations can select examples; they are never appended to actor inputs.

Before evaluating any alternatives, the pipeline replays the complete prefix
from match start. **Every state hash and every public input field** must match
the collected episode. Every observation, including the initial countdown,
advances the actor's GRU exactly once. Each branch receives an isolated fork
and cloned carried hidden state. Opponents keep the exact original scripted
style and controller seed. Learned opponents are rejected until their own
recurrent history can also be restored.

At each origin, legal proposals include the exact recorded action, WAIT,
existing teachers and additional public geometric alternatives. Proposals
use visible positions and card metadata, without enemy hand contents,
deployment timers, target IDs or exact target locks.

Discovery uses three matched sampling replicas at 128 decisions (32 seconds).
It selects one action, then freezes that identity. Six **disjoint** sampling
replicas compare that frozen action against the factual action at 256
decisions (64 seconds). Branches share seeds within each replica; discovery
and confirmation use different actual seeds. The paired advantage must have
`mean - 1.96 * standard_error > 0.05` in tower-fraction/terminal-value units.
This is a local, nominal uncertainty gate; replicas from one origin are not
independent matches and do not establish population-wide strength.

Targets certify changing **one decision**, followed by the frozen actor's
normal continuation. WAIT receives no credit for a scripted saving plan that
the actor did not subsequently execute. Rejected proposals remain in the
report and receive no corrective training label.

## Training and fresh evaluation

Only independently accepted actions become sparse targets. This path accepts
ground corrections as well as air/siege corrections, without the older
air-or-X-Bow-only teaching gate. Other valid rows preserve the selected actor's
full legal action distribution through KL retention. Factual observations,
actions and chronology remain unchanged. The default fit is two epochs at
learning rate `1e-5`, using 32-step truncated backpropagation while carrying
hidden state through complete episodes. Calibration origins receive no
gradients and do not select an epoch. No accepted training labels means no fit.

The Python `train_corrections` API also accepts `correction_weight` and
`correction_exposures` through `CorrectionTrainingConfig`. Extra factual-target
presentations are accumulated into existing optimizer updates, with full
current-student prefix reconstruction; ordinary retention visits and update
counts stay matched. Before/after factor diagnostics, the learning gate and
actual exposure/burn-in counts travel with the sealed training result. These
controls are not exposed by the replay runner's CLI yet.

The separate `correction_sequences.py` module provides public saving-plan
control, independent sequence validation, and branch-observation capture.
The current trainer explicitly rejects a nonempty `sequences` manifest before
creating fit output. Collection/search/training integration for those captures
is tracked in [STATUS.md](../../STATUS.md).

The single predeclared final checkpoint is compared with the unchanged actor
on fresh full games against **all 30 training decks**, first fixed opponents,
then the two development styles. Positive overall screen performance with no
negative family net and no development-style regression triggers confirmation
on new match origins. This local gate controls additional evaluation work;
it does not certify human strength or promote a policy. Reserved-deck final
evaluation remains separate, after candidate selection.

Reports live under the chosen output directory:

- `plan.json`: immutable workload and source/checkpoint identity.
- `dataset/manifest.json`: complete episodes and byte hashes.
- `search_selection.json`, `search/*.json`, `corrections.json`: reviewed
  decisions, actual replica evidence, rejection reasons and accepted targets.
- `fit/result.json`: trained checkpoint, full-history calibration and dose.
- `evaluation.json`: fresh screen and any independent confirmation.
- `REPORT.md`, `result.json`, `progress.json`: human and machine summaries.

The selected actor receipt is never modified by this runner. Improved label
agreement alone is insufficient evidence to replace it.
