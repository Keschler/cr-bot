# Clash Royale Vision Bot

Clash Royale Vision Bot reads Clash Royale gameplay from recorded video, screenshots, or live capture and turns it into structured game state and card-play events. The project is currently focused on extracting state and actions from expert gameplay for evaluation and future imitation-learning datasets.

<img width="1600" height="900" alt="cr-bot-banner-v1-dashboard" src="https://github.com/user-attachments/assets/1d225b86-79a3-4477-b524-2274faa92692" />

## Features

- Detects battlefield objects with KataCR's best-performance dual YOLO detector setup.
- Reads important HUD values such as timer, elixir, hand cards, next card, and tower HP.
- Classifies hand cards and the next card with two self-trained MobileNetV3-Small models.
- Tracks visible units, teams, positions, confidence scores, and estimated HP.
- Estimates enemy card plays and enemy unit HP from detected troops, buildings, health bars, and spell effects.
- Estimates enemy elixir over time from match clock and confirmed enemy plays.
- Extracts own card plays from hand-slot changes and confirms deploy locations from elixir-change flashes, troop clocks, spell elixir-cost overlays, and rolling spell tracks.
- Tracks projectile and rolling-spell direction to distinguish own and enemy Fireballs and Logs more reliably.
- Supports live capture through a video device such as `v4l2loopback`.
- Supports frame-analysis replay caches for fast tracker-only iteration without rerunning YOLO and OCR.
- Includes an action-evaluation harness with built-in scenarios, focused time windows, ground-truth JSON, timing reports, and cell visualization overlays.
- Includes scripts for detector training, inference, annotation preparation, and dataset processing.

## Demo

[Download the executable demo from GitHub Releases](https://github.com/keschler/cr-bot/releases/latest)

[Demo Video](https://youtu.be/QRP_nLJWApM)

The release asset is a Linux CPU executable bundle. It does not require setting up Python, installing project dependencies, or cloning the repository.

## How It Works

The project processes each frame in a few steps:

1. It crops the arena and runs the KataCR dual-detector model.
2. It remaps detections back onto the full game frame.
3. It extracts HUD information with OCR, template matching, and card classifiers.
4. It builds a structured `GameState`.
5. It updates trackers for own actions, enemy cards, enemy elixir, and match state.

The hand HUD detector can distinguish an active evolution from its base card and emits names such as `evo-cannon`. Battlefield YOLO support is more limited: only evolution classes present in the KataCR training set can be detected, and enemy evolution units are currently normalized to their base card by the card-play tracker.

Own action extraction is based on the card hand changing first, then confirming the deploy location from the best available in-game cue:

- Normal troops and buildings use deploy-clock detections when available.
- Recent allied unit tracks are kept briefly so a unit can still be matched after the clock has disappeared.
- The elixir digit flash is used as the action placement timestamp, then the action is emitted only after later confirmation. This avoids reporting the later confirmation frame as the placement time.
- Circle/radius spells use the white spell radius ellipse only as an aiming candidate.
- The purple elixir-cost overlay is required to confirm that a spell was actually released. It is scored from the middle of the top half of the detected spell ellipse, so the crop is tied to the actual spell-radius candidate rather than a loose rectangle around the approximate center.
- Rolling spells such as The Log and Barbarian Barrel use their first visible rolling unit track plus elixir confirmation.

## Install From Source

Python 3.12 or newer is required.

```bash
git clone https://github.com/Keschler/cr-bot.git
cd cr-bot
git submodule update --init vendor/external/KataCR
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Run the test suite with:

```bash
pytest
```

## Use The Executable

Download `capture` from the latest GitHub Release:

```text
https://github.com/keschler/cr-bot/releases/latest
```

Run it on a screenshot:

```bash
./capture --debug-frame /path/to/screenshot.png
```

Run it on live video from a Linux video device:

```bash
VIDEO_DEVICE=/dev/video37 ./capture
```

Run it on a recorded video:

```bash
./capture --video /path/to/gameplay.mp4
```

Limit recorded-video analysis to the first N seconds:

```bash
./capture --video /path/to/gameplay.mp4 --video-duration 198
```

Analyze every fifth frame within an absolute video-time window:

```bash
./capture --video /path/to/gameplay.mp4 \
  --frame-stride 5 \
  --video-start-time 120 \
  --video-end-time 180
```

The same arguments are available from a source installation through `cr-bot`.

### Replay Cache

Vision processing is the expensive part of repeated tracker debugging. Write a replay cache during a normal video run:

```bash
cr-bot --video /path/to/gameplay.mp4 \
  --write-replay-cache data/eval/replay_cache/gameplay.pkl.gz
```

Replay the stored frame analyses without rerunning YOLO, OCR, or the hand-card classifiers:

```bash
cr-bot --replay-cache data/eval/replay_cache/gameplay.pkl.gz
```

For Android live capture on Linux, create a loopback device first:

```bash
sudo modprobe v4l2loopback video_nr=37 card_label=scrcpy exclusive_caps=1
```

Then stream the phone screen into that device with `scrcpy`:

```bash
scrcpy --video-source=display --v4l2-sink=/dev/video37
```

Use `--yolo-detections` to derive tower-HP crops from detected health bars instead of fixed tower ROIs. Use `--alternative-rois` for gameplay layouts where the bottom HUD is vertically shifted.

Frames are normalized to `1080x2400` internally by default so the existing ROIs keep matching the game UI. Use `--no-normalize` only when intentionally processing the raw capture size. Currently, only `1080x2400` or resolutions with the same aspect ratio are expected to work.

## Project Structure

```text
src/cr_bot/
  app/                        CLI, live capture, video replay, frame pipeline
  domain/                     game state, constants, ROIs, card metadata
  vision/                     YOLO runtime and frame extractors
  trackers/                   own actions, enemy cards/elixir, and match clock
  features/                   board and global feature builders
  replay/                     serialized frame-analysis cache support
  eval/                       action evaluation and cell visualizer
  debug/                      debug rendering and reporting helpers

assets/                       detector/classifier checkpoints, templates, local media
configs/                      detector training configs
data/                         local datasets, labels, and evaluation ground truth
scripts/                      training, inference, and dataset helper scripts
scripts/debug/                local debug renderers for action and spell detection
vendor/external/KataCR/       patched KataCR detector dependency
outputs/                      generated runs, caches, debug images, and videos

dataset_generation/
  scripts/process_frame.py    offline frame-state dataset generation
  data/                       local generated dataset outputs

docs/                         documentation
```

## Debugging

Most local visual debugging scripts live under `scripts/debug/`.

Examples:

```bash
# Render purple elixir-cost detector crops for failed spell confirmations.
outputs/venv/bin/python scripts/debug/debug_spell_purple_detector.py \
  --preset failed-wrong-detections \
  --video assets/pictures/10_fps_wrong_detections.mp4 \
  --output-dir outputs/debug/spell_purple_failed_wrong_detections

# Render the confirmed purple elixir-cost cases from the wrong-detections clip.
outputs/venv/bin/python scripts/debug/debug_spell_purple_detector.py \
  --preset confirmed-wrong-detections \
  --video assets/pictures/10_fps_wrong_detections.mp4 \
  --output-dir outputs/debug/spell_purple_confirmed_wrong_detections
```

Debug outputs are written under `outputs/debug/` and are intentionally not part of the runtime pipeline.

### Profile Video Analysis

Use the profiling script to analyze the first 10 seconds of the 3400-ladder
video and the 10 FPS hog-cycle video. It reports detector startup, warmup,
overall throughput, and per-stage timings:

```bash
PYTHONPATH=src outputs/venv/bin/python scripts/profile_video_analysis.py
```

Pass `--duration` or explicit video paths to run a shorter smoke test or
profile different clips.

## Action Evaluation

The evaluation tools compare detected own and enemy actions against hand-labeled
ground truth. They report precision, recall, F1, missed actions, false
positives, timing error, and placement-cell distance.

For the two built-in hog-cycle evaluation runs, use:

```bash
PYTHONPATH=src python3 -m cr_bot.eval.run_action_eval_scenarios
```

Use `--mode summary` for only the aggregate totals, `--scenario 2hog-cycle` or
`--scenario 3400ladder` to run one clip, `--run-detection` to regenerate the
capture txt first, and `--explain-misses` to run the missing-action explainer.

To isolate one card occurrence and avoid processing the whole video:

```bash
PYTHONPATH=src python3 -m cr_bot.eval.run_action_eval_scenarios \
  --scenario 2hog-cycle \
  --run-detection \
  --focus-card fireball \
  --focus-side enemy \
  --focus-video-time 183.7 \
  --focus-window-before 5 \
  --focus-window-after 10
```

Use `--build-replay-cache` once and `--replay-cache` on later scenario runs when changing tracker logic but not frame analysis. The scenario runner stores one cache per built-in scenario under `data/eval/replay_cache/`.

The cell visualizer renders the action grid over labeled frames so ground-truth
cells can be checked or filled in from video frames. See `docs/eval/README.md`
for the ground-truth format and script options.

## Devlogs

Development logs are available online:

```text
Devlogs: https://flavortown.hackclub.com/projects/16627
```


## Planned Work

- Improve player-action extraction accuracy across more decks, cards, and layouts.
- Build behavior cloning datasets from expert gameplay.
- Train models that imitate expert decisions from extracted game state.
- Explore reinforcement learning on top of the extracted state and action pipeline.

## Current Limitations

- The hand-card and next-card UI classifiers reached 100% accuracy on the tested 2.6 Hog Cycle workflow, but have not been fully validated across every other deck and troop/card combination.
- Heroes/champions are not currently included in detection, either in the YOLO battlefield detector setup or in the self-trained hand-card and next-card UI classifiers.
- Goblinstein and Three Musketeers are not added to the hand-card and next-card detection model.
- Active evolutions can appear as `evo-*` in hand state, but evolution names are not yet represented by `CARD_METADATA` IDs and therefore are not encoded in the card one-hot feature vector.
- The battlefield detector only recognizes evolution classes included in the older KataCR model. Newer evolutions such as Cannon Evolution do not have dedicated YOLO classes.
- Mirror cannot be identified from its spawned battlefield unit; it requires hand/cycle inference.
- Timer, elixir, and tower HP extraction can still be noisy in some frames.
- Enemy elixir is an estimate, not a value shown by the game.
- Spell detection can be ambiguous when multiple radius effects overlap or when the purple elixir-cost overlay is outside the selected spell ellipse crop.
- The current action ground truth is strongest for the 2.6 Hog Cycle champion clip; enemy-action labels and broader deck coverage are still incomplete.

## Tech Stack

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- KataCR
- NumPy
- v4l2loopback / live video capture tooling

## Credits

- Battlefield unit, tower, spell, and part of the health-bar detection is based on the KataCR project and its best-performance dual YOLO detector setup.
- Hand-card and next-card classifiers are self-trained project models based on `mobilenet_v3_small`.
