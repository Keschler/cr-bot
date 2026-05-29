# Clash Royale Vision Bot

Clash Royale Vision Bot is a computer vision project that reads Clash Royale gameplay from video or live capture and turns it into structured game state. Planned is to extract state and action from expert 2.6 hog cycle gameplay and use that to do behavior cloning and reinforcement learning.
<img width="1600" height="900" alt="cr-bot-banner-v1-dashboard" src="https://github.com/user-attachments/assets/1d225b86-79a3-4477-b524-2274faa92692" />

## Features

- Detects battlefield objects with KataCR's best-performance dual YOLO detector setup.
- Reads important HUD values such as timer, elixir, hand cards, next card, and tower HP.
- Classifies hand cards and the next card with two self-trained MobileNetV3-Small models.
- Tracks visible units, teams, positions, confidence scores, and estimated HP.
- Estimates enemy card plays and enemy unit HP from detected troops, buildings, health bars, and spell effects.
- Estimates enemy elixir over time from match clock and confirmed enemy plays.
- Extracts own card plays from hand-slot changes and confirms deploy locations from elixir-change flashes, troop clocks, spell elixir-cost overlays, and rolling spell tracks.
- Supports live capture through a video device such as `v4l2loopback`.
- Supports offline dataset generation from recorded gameplay clips.
- Includes an action-evaluation harness with ground-truth JSON, timing error reports, and cell visualization overlays.
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

Own action extraction is based on the card hand changing first, then confirming the deploy location from the best available in-game cue:

- Normal troops and buildings use deploy-clock detections when available.
- Recent allied unit tracks are kept briefly so a unit can still be matched after the clock has disappeared.
- The elixir digit flash is used as the action placement timestamp, then the action is emitted only after later confirmation. This avoids reporting the later confirmation frame as the placement time.
- Circle/radius spells use the white spell radius ellipse only as an aiming candidate.
- The purple elixir-cost overlay is required to confirm that a spell was actually released. It is scored from the middle of the top half of the detected spell ellipse, so the crop is tied to the actual spell-radius candidate rather than a loose rectangle around the approximate center.
- Rolling spells such as The Log and Barbarian Barrel use their first visible rolling unit track plus elixir confirmation.

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

For Android live capture on Linux, create a loopback device first:

```bash
sudo modprobe v4l2loopback video_nr=37 card_label=scrcpy exclusive_caps=1
```

Then stream the phone screen into that device with `scrcpy`:

```bash
scrcpy --video-source=display --v4l2-sink=/dev/video37
```

**Use** `--yolo-detections` if screenshot or phone screen isn't `1080x2400` -> uses yolo dections for the extraction of tower-hp

Frames are normalized to `1080x2400` internally by default so the existing ROIs keep matching the game UI. Use `--no-normalize` only when intentionally processing the raw capture size. Currently, only `1080x2400` or resolutions with the same aspect ratio are expected to work.

## Project Structure

```text
src/cr_bot/
  app/                        CLI, live capture, video replay, frame pipeline
  domain/                     game state, constants, ROIs, card metadata
  vision/                     YOLO runtime and frame extractors
  trackers/                   enemy cards, match clock, and stateful tracking
  features/                   board and global feature builders
  audio/                      audio classifier support
  eval/                       action evaluation and cell visualizer
  debug/                      debug rendering and reporting helpers

assets/                       detector/classifier checkpoints, templates, local media
configs/                      detector training configs
data/                         local datasets, labels, and evaluation ground truth
scripts/                      training, inference, and dataset helper scripts
scripts/debug/                local debug renderers for action and spell detection
vendor/                       external KataCR dependency
outputs/                      generated runs, caches, debug images, and videos
capture/                      temporary compatibility wrappers for old entrypoints

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

## Action Evaluation

The evaluation tools compare detected own and enemy actions against hand-labeled
ground truth. They report precision, recall, F1, missed actions, false
positives, timing error, and placement-cell distance.

The cell visualizer renders the action grid over labeled frames so ground-truth
cells can be checked or filled in from video frames. See `docs/eval/README.md`
for the ground-truth format and script options.

## Devlogs

Development logs are available online:

```text
Devlogs: https://flavortown.hackclub.com/projects/16627
```


## Planned Work

- Extract player actions from gameplay videos, including card choice and deployment location.
- Build behavior cloning datasets from expert gameplay.
- Train models that imitate expert decisions from extracted game state.
- Explore reinforcement learning on top of the extracted state and action pipeline.

## Current Limitations

- The hand-card and next-card UI classifiers reached 100% accuracy on the tested 2.6 Hog Cycle workflow, but have not been fully validated across every other deck and troop/card combination.
- Heroes/champions are not currently included in detection, either in the YOLO battlefield detector setup or in the self-trained hand-card and next-card UI classifiers.
- Goblinstein and Three Musketeers are not added to the hand-card and next-card detection model.
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
