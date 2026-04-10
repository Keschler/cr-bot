# Capture Workspace

This workspace now uses a clearer split between local data, upstream vendor code, and workflow scripts.

## Layout

- `data/`
  - `video_clips/`: raw local recordings
  - `part2/`: extracted local frames before merging into a dataset
  - `seed_dataset/`: merged training dataset built from the upstream annotations plus local clip imports
  - `seed_labels/`: CVAT-related local assets
- `vendor/`
  - `external/Clash-Royale-Detection-Dataset/`: upstream dataset repo clone
  - `external/KataCR/`: upstream KataCR repo clone
- `scripts/`
  - `prepare_seed_dataset.py`: import local frames into the merged seed dataset
  - `build_seed_annotations.py`: rebuild YOLO annotations and split files from JSON labels
  - `setup_seed_detectors.py`: generate KataCR detector configs against the merged seed dataset
- `configs/`
  - `katacr_train_baseline.yaml`: smaller baseline training config for a single low-VRAM GPU
- `models/`
  - detector and classifier checkpoints used by the runtime and training scripts
- `docs/`
  - workspace notes, exports, and reference documents kept out of the runtime root
- `bin/`
  - shell helpers such as `start_stream.sh`
- `.venv-train/`
  - dedicated training environment for the KataCR detector workflow

## Compatibility

Legacy top-level paths such as `external/`, `part2/`, `seed_dataset/`, `seed_labels/`, and `video_clips/` can remain as symlinks so existing commands continue to work.

## Baseline Workflow

1. `python scripts/prepare_seed_dataset.py`
2. Annotate new frames in `data/seed_dataset/images/part2/capture_clip/1`
3. `python scripts/build_seed_annotations.py`
4. `python scripts/setup_seed_detectors.py`
5. Train with KataCR using the generated detector configs and `configs/katacr_train_baseline.yaml`

## Baseline Commands

Use the dedicated training environment:

```bash
source .venv-train/bin/activate
```

Generate detector configs:

```bash
python scripts/setup_seed_detectors.py
```

Train detector 1:

```bash
python scripts/train_seed_baseline.py --detector 1 --device 0
```

Train detector 2:

```bash
python scripts/train_seed_baseline.py --detector 2 --device 0
```

Run inference after training:

```bash
python scripts/run_seed_inference.py \
  --weights runs/detector1_baseline_seed/weights/best.pt runs/detector2_baseline_seed/weights/best.pt \
  --source data/video_clips/clip.mp4 \
  --video-interval 3
```
