# V4 Arena Lab

A separate local browser application for watching the deterministic simulator.
It runs independently of Arena Replay Analyst and does not load vision models.
The backend uses the existing Python dependencies; the native JavaScript UI
needs no Node installation or build step.

From the repository root:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m sim_viewer
```

Open **http://127.0.0.1:8010/**. Use `--port 8011` to select another port.
If Uvicorn is missing from the existing environment, install the dependency
already declared in `requirements.txt`:

```bash
outputs/venv/bin/python -m pip install uvicorn
```

## Watching matches

- **Recorded frontier matches:** choose a `*_match.json` from the match library.
  The backend reconstructs physics from the seed, decks, and both players'
  recorded actions. Every recorded decision state hash must match; an old file
  that needs an older simulator revision stops with an explicit error.
  These records preserve only top placement cells, so a complete probability
  heatmap is unavailable for them.
- **Exhibitions:** select a plain V4 actor checkpoint, scripted or checkpoint
  opponent, opponent deck, seed, and decision limit. Matches run in a background
  worker and are recorded automatically for later playback. One simulation
  worker runs at a time; viewing existing JSONL files needs no simulation worker.
  Exhibitions sample the policy with the selected seed. They are new matches,
  separate from the episodes that produced a training checkpoint.
- **Training samples:** enable recording before starting a V4 PPO run, then
  open its recording in the library. New frames appear as training writes them.
  Existing training processes cannot acquire the recorder retroactively.

The tactical arena shows both players' authoritative state, hands, next cards,
elixir, troops, towers, projectiles, and persistent effects. Enable labels,
targets, ranges, or the policy placement heatmap; click a unit to inspect it.
Playback supports pause, single-step, speed, seeking, and live follow.
**← / →** steps one recorded frame; **Shift + ← / →** steps ten.
These shortcuts also work when a playback button or timeline slider has focus.
Typing fields, dropdowns, and open dialogs retain their normal keyboard input.
Seeking or pausing the viewer does not pause training or the match worker. Stop cancels
an exhibition or reconstructed replay; stopping a training-file view only
detaches the viewer and leaves training running.

V4's retained learner encoder accepts the fixed eight-card Hog-cycle deck.
Learner deck order can change within that vocabulary. Checkpoint opponents
have the same restriction; scripted opponents can use other valid decks from
the local real-deck catalog. Plain actor exports are required; datasets,
optimizer checkpoints, and critic weights cannot run an exhibition.

## Understanding the policy heatmap

The map shows the learner's placement probabilities **assuming it chooses
Play**, before the action happens. It combines all legal hand slots:

```text
P(cell | Play) = sum over cards P(card | Play) × P(cell | card, Play)
```

Illegal cards and cells have zero probability. The highlighted hand slot is
the sampled card; the heatmap still includes the other playable cards. A
bright enemy-side region can come from a spell. The Wait/Play bars show the
separate chance of playing at all: a bright map can accompany 99% Wait.
For example, 20% Play and a cell with 30% conditional placement probability
give a 6% chance of playing any card at that cell on this decision.

Brightness is relative to the strongest cell in **each frame**, using
`0.62 × (cell_probability / peak_probability)^0.6`. It emphasizes weaker
preferences and cannot be used to compare absolute probabilities across
frames. The map describes the actor's placement preference; it is neither a
board-value estimate nor a prediction of winning. Historical frontier traces
store unmasked softmax probabilities and only top cells; the Policy panel
labels those raw values separately, and a dense heatmap is unavailable.

## Recording a training run

For a **new** invocation of your existing V4 training command, prefix it with:

```bash
export CR_SIM_VIEWER_DIR="$PWD/outputs/v4/my_run/viewer"
export CR_SIM_VIEWER_EVERY=5
# Run your usual scripts/run_v4_learning_curve.py command in this shell.
```

This records the first and then every fifth episode start on lane zero across
rollouts in that process. Other lanes are not recorded. Each sample stores
detached pre-action snapshots, both actions, the learner's masked policy
distribution, and the reward caused by the displayed action. The last frame
has no action or reward. A footer distinguishes a completed match from an
episode cap, environment truncation, or rollout boundary; samples can begin
mid-match when training uses frontier entries.

Recording adds JSON serialization and disk I/O. Increase the sampling interval
for less overhead. All frames are retained on disk until you remove the
generated recordings; no automatic eviction or background cleanup runs.
Partial trailing lines are ignored until complete. Interrupted files remain
viewable; absence of an end marker does not prove training is still running.

PPO also writes atomic `progress.json` after updates and evaluation points.
The training dashboard reads these live and reads existing
`learning_curve.json`, bootstrap manifests, distillation arm reports, and
`training_history.json` for completed runs. Distillation publishes epoch
progress when the same recording directory is set. Unavailable measurements
remain gaps; evaluations with no completed matches have no win rate, and
truncated matches are counted separately. The return chart uses recorded PPO
returns, rather than labeling missing rewards as zero. Run comparisons overlay
available measurements; they do not perform a new statistical evaluation or
claim that unlike experiment protocols are comparable.

## Files and configuration

By default the library scans `outputs/v4/`. Generated exhibitions and
reconstructed matches live in `outputs/v4/sim_viewer/matches/`. These are local
artifacts and should not be committed. To browse another local V4 directory,
set roots before starting the **viewer**:

```bash
CR_SIM_VIEWER_ROOTS="$PWD/outputs/v4:/path/to/other/v4" \
  PYTHONPATH=.:src outputs/venv/bin/python -m sim_viewer
```

`CR_SIM_VIEWER_DIR`, if set on the viewer process too, adds that directory to
its discovery roots. Paths outside configured roots and symlinks escaping
them are rejected. Checkpoint discovery reads filenames, not Torch payloads;
incompatible checkpoints produce an error when launched. GPU/CV imports remain
lazy. The exhibition worker uses CPU inference in the viewer process, separate
from training, and restores the process's Torch RNG after the match.

New JSONL recordings contain a schema header with engine and ruleset identity,
consecutive frame records, and an explicit end marker. Their saved snapshots
can be viewed without rerunning the historical engine or loading a checkpoint.
The frame `index`, physics `tick`, and match `time_s` are distinct. Coordinates
are in tiles on the simulator's 18 × 32 arena, with player zero at the bottom.

## API and validation

Interactive API documentation is at `/docs`. The main endpoints are
`GET /api/catalog`, `/api/library`, `/api/geometry`, `/api/runs/{id}`;
`POST /api/replays`, `/api/exhibitions`; and
`GET /api/matches/{id}`, `/api/matches/{id}/frames`,
`/api/matches/{id}/stream` (SSE), plus `POST /api/matches/{id}/stop`.

Focused tests:

```bash
outputs/venv/bin/python -m pytest \
  tests/test_sim_viewer_library.py tests/test_sim_viewer_app.py \
  tests/test_sim_viewer_matches.py tests/test_sim_viewer_keyboard.py \
  simulator/tests/test_viewer_recording.py
```

HTTP tests need the optional `httpx` test client. The default non-audio suite
also covers existing simulator behavior and training regressions. The keyboard
regression uses Node.js when available and skips when it is absent.
