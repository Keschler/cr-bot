# Clash Royale Vision Bot

Clash Royale Vision Bot is a computer vision project that reads Clash Royale gameplay from video or live capture and turns it into structured game state.

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

Add your shipped demo link here:

```text
Demo: https://example.com/your-demo
```

The recommended demo is a short video showing the project running on a Clash Royale match, with detections and extracted state visible on screen.

## How It Works

The project processes each frame in a few steps:

1. It crops the arena and runs the KataCR dual-detector model.
2. It remaps detections back onto the full game frame.
3. It extracts HUD information with OCR, template matching, and card classifiers.
4. It builds a structured `GameState`.
5. It updates trackers for enemy cards, enemy elixir, and match state.

## Run Locally From A Fresh Clone

Clone the repository and enter it:

```bash
git clone <repo-url>
cd cr-bot
```

Install the system tools needed for live Android capture on Linux:

```bash
sudo apt install scrcpy v4l2loopback-dkms v4l2loopback-utils
```

Create and activate a Python environment:

```bash
cd capture
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

Make sure the external project folders are present. The current repository tracks these folders as gitlinks, but `.gitmodules` is missing, so a fresh clone may not fetch them automatically:

```text
capture/vendor/external/KataCR
capture/vendor/external/Clash-Royale-Detection-Dataset
capture/templates/cr-api-assets
capture/data/seed_labels/cvat
```

For runtime, `capture/vendor/external/KataCR` is required. `capture/templates/cr-api-assets` is used by the card template fallback path. The Clash Royale detection dataset and CVAT checkout are mainly needed for dataset/training workflows.

If those folders are missing or empty after cloning, restore at least the runtime dependencies manually:

```bash
git clone https://github.com/wty-yy/KataCR.git capture/vendor/external/KataCR
git clone https://github.com/RoyaleAPI/cr-api-assets.git capture/templates/cr-api-assets
```

Optional training/dataset dependencies:

```bash
git clone https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git capture/vendor/external/Clash-Royale-Detection-Dataset
git clone https://github.com/cvat-ai/cvat capture/data/seed_labels/cvat
```

The model weights are committed in `capture/models/`:

```text
detector1_v0.7.13.pt
detector2_v0.7.13.pt
hand_classifier_best.pt
next_classifier_best.pt
```

Run a single debug frame:

```bash
cd capture
source venv/bin/activate
python capture.py --debug-frame pictures/screen.png
```

Run live capture in two terminals. First, create the loopback video device:

```bash
sudo modprobe v4l2loopback video_nr=37 card_label=scrcpy exclusive_caps=1
```

Then start the phone stream:

```bash
cd capture
VIDEO_DEVICE=/dev/video37 ./bin/start_stream.sh
```

In a second terminal, run the vision pipeline:

```bash
cd capture
source venv/bin/activate
VIDEO_DEVICE=/dev/video37 python capture.py
```

Frames are normalized to `1080x2400` internally by default so the existing ROIs keep matching the game UI. Use `--no-normalize` only when you intentionally want to process the raw capture size.

## Setup

Create and activate a Python environment:

```bash
cd capture
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The runtime expects trained detector and classifier weights in `capture/models/`. The current default detector paths are configured in `capture/vision/yolo_runtime.py` and use the KataCR best-performance combo detector weights `detector1_v0.7.13.pt` and `detector2_v0.7.13.pt`.

Hand-card and next-card recognition use two self-trained classifiers, `hand_classifier_best.pt` and `next_classifier_best.pt`. The training script in `capture/scripts/train_card_classifier.py` fine-tunes `mobilenet_v3_small` with a replacement classifier head for the local Clash Royale card classes.

## Run Live Capture

Start your screen capture pipeline so that Clash Royale frames are available through a video device. By default the project looks for a dummy video device, or falls back to `/dev/video37`.

Create a V4L2 loopback device first:

```bash
sudo modprobe v4l2loopback video_nr=37 card_label=scrcpy exclusive_caps=1
```

Start the phone stream with the included helper:

```bash
cd capture
./bin/start_stream.sh
```

The helper uses `scrcpy` and sends the phone screen into the loopback video device. It auto-detects a dummy video device when possible, otherwise it uses `/dev/video37`.

In another terminal, run the vision pipeline:

```bash
cd capture
source venv/bin/activate
python capture.py
```

To choose a specific video device:

```bash
VIDEO_DEVICE=/dev/video37 python capture.py
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
- Large generated datasets may need to be provided separately from the repository.

## Tech Stack

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- KataCR
- NumPy
- v4l2loopback / live video capture tooling

## Credits

- Battlefield unit, tower, spell, and health-bar detection is based on the KataCR project and its best-performance dual YOLO detector setup.
- Hand-card and next-card classifiers are self-trained project models based on `mobilenet_v3_small`.
