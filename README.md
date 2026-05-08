# Clash Royale Vision Bot

Clash Royale Vision Bot is a computer vision project that reads Clash Royale gameplay from video or live capture and turns it into structured game state.
<img width="1600" height="900" alt="cr-bot-banner-v1-dashboard" src="https://github.com/user-attachments/assets/1d225b86-79a3-4477-b524-2274faa92692" />

## Features

- Detects battlefield objects with KataCR's best-performance dual YOLO detector setup.
- Reads important HUD values such as timer, elixir, hand cards, next card, and tower HP.
- Classifies hand cards and the next card with two self-trained MobileNetV3-Small models.
- Tracks visible units, teams, positions, confidence scores, and estimated HP.
- Estimates enemy card plays and enemy unit HP from detected troops, buildings, health bars, and spell effects.
- Estimates enemy elixir over time from match clock and confirmed enemy plays.
- Supports live capture through a video device such as `v4l2loopback`.
- Supports offline dataset generation from recorded gameplay clips.
- Includes scripts for detector training, inference, annotation preparation, and dataset processing.

## Demo

[Download the executable demo from GitHub Releases](https://github.com/keschler/cr-bot/releases/latest)

[Demo Video](https://youtu.be/qxFdI4DDtCA)

The release asset is a Linux CPU executable bundle. It does not require setting up Python, installing project dependencies, or cloning the repository.

## How It Works

The project processes each frame in a few steps:

1. It crops the arena and runs the KataCR dual-detector model.
2. It remaps detections back onto the full game frame.
3. It extracts HUD information with OCR, template matching, and card classifiers.
4. It builds a structured `GameState`.
5. It updates trackers for enemy cards, enemy elixir, and match state.

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
capture/
  main.py                     live capture loop
  extractors/                 timer, elixir, card, unit, tower HP extraction
  trackers/                   enemy cards, match clock, and stateful tracking
  vision/                     YOLO/KataCR runtime helpers
  features/                   board and global feature builders
  scripts/                    training, inference, and dataset helper scripts
  models/                     detector and classifier checkpoints

dataset_generation/
  scripts/process_frame.py    offline frame-state dataset generation
  data/                       local generated dataset outputs

docs/                         documentation
```

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
- Spell detection can be ambiguous.

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
