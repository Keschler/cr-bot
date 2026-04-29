# Clash Royale Vision Bot

Clash Royale Vision Bot is a computer vision project that reads Clash Royale gameplay from video or live capture and turns it into structured game state.

## Features

- Detects battlefield objects with a YOLO/KataCR-based detector.
- Reads important HUD values such as timer, elixir, hand cards, next card, and tower HP.
- Tracks visible units, teams, positions, confidence scores, and estimated HP.
- Estimates enemy card plays and enemy unit HP from detected troops, buildings, health bars, and spell effects.
- Estimates enemy elixir over time from match clock and confirmed enemy plays.
- Supports live capture through a video device such as `v4l2loopback`.
- Supports offline dataset generation from recorded gameplay clips.
- Includes scripts for detector training, inference, annotation preparation, and dataset processing.

## Demo

Add your shipped demo link here:

```text
Demo: https://example.com/your-demo
```

The recommended demo is a short video showing the project running on a Clash Royale match, with detections and extracted state visible on screen.

## How It Works

The project processes each frame in a few steps:

1. It crops the arena and runs the object detector.
2. It remaps detections back onto the full game frame.
3. It extracts HUD information with OCR and template matching.
4. It builds a structured `GameState`.
5. It updates trackers for enemy cards, enemy elixir, and match state.

## Setup

Create and activate a Python environment:

```bash
cd capture
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The runtime expects trained detector weights in `capture/models/`. The current default paths are configured in `capture/vision/yolo_runtime.py`.

## Run Live Capture

Start your screen capture pipeline so that Clash Royale frames are available through a video device. By default the project looks for a dummy video device, or falls back to `/dev/video37`.

Create a V4L2 loopback device first:

```bash
sudo modprobe v4l2loopback video_nr=37 card_label=scrcpy exclusive_caps=1
```

Start the phone stream with the included helper:

```bash
cd capture
./start_stream.sh
```

The helper uses `scrcpy` and sends the phone screen into the loopback video device. It auto-detects a dummy video device when possible, otherwise it uses `/dev/video37`.

In another terminal, run the vision pipeline:

```bash
cd capture
source venv/bin/activate
python main.py
```

To choose a specific video device:

```bash
VIDEO_DEVICE=/dev/video37 python main.py
```

You can also pass the same device to the stream helper:

```bash
VIDEO_DEVICE=/dev/video37 ./start_stream.sh
```

## Run Dataset Generation

The dataset generation script reads a gameplay clip, processes sampled frames, saves frame images, and writes JSONL state rows.

```bash
source capture/venv/bin/activate
python dataset_generation/scripts/process_frame.py
```

The script currently uses paths inside `dataset_generation/data/`. Edit those paths in `dataset_generation/scripts/process_frame.py` for your own clips.

## Training And Inference Scripts

Most detector workflow scripts live in `capture/scripts/`.

Common commands:

```bash
cd capture
source .venv-train/bin/activate
python scripts/setup_seed_detectors.py
python scripts/train_seed_baseline.py --detector 1 --device 0
python scripts/train_seed_baseline.py --detector 2 --device 0
```

Run detector inference:

```bash
python scripts/run_seed_inference.py \
  --weights runs/detector1_baseline_seed/weights/best.pt runs/detector2_baseline_seed/weights/best.pt \
  --source data/video_clips/clip.mp4 \
  --video-interval 3
```

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

docs/
  DEVLOG.md                   local development notes
```

## Devlogs

Development logs are available online:

```text
Devlogs: https://example.com/your-devlogs
```

There is also a local development summary in `docs/DEVLOG.md`.

## Planned Work

- Extract player actions from gameplay videos, including card choice and deployment location.
- Build behavior cloning datasets from expert gameplay.
- Train models that imitate expert decisions from extracted game state.
- Explore reinforcement learning on top of the extracted state and action pipeline.

## Current Limitations

- Live capture depends on the correct screen resolution and video device setup.
- Timer, elixir, and tower HP extraction can still be noisy in some frames.
- Enemy elixir is an estimate, not a value shown by the game.
- Spell and spawned-unit detection can be ambiguous.
- Model weights and large datasets may need to be provided separately from the repository.

## Tech Stack

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- KataCR
- NumPy
- v4l2loopback / live video capture tooling
