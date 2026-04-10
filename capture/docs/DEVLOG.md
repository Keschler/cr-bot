# Devlog

## 2026-03-17

Started from a local Clash Royale capture/OCR workspace focused on HUD extraction:
- existing code handled timer, elixir, tower HP, and hand-card recognition from screen captures
- local video input was added under `video_clips/clip.mp4`

Reviewed the upstream Clash Royale dataset workflow:
- confirmed that `Clash-Royale-Detection-Dataset` contains manually annotated detection data, sliced assets, and small classification datasets
- confirmed that `KataCR` contains the dataset conversion, synthetic generation, detector setup, and training code
- confirmed that battlefield objects are manually annotated and converted into YOLO-style labels through `KataCR`

Prepared the local clip for annotation:
- extracted `clip.mp4` into sampled frames
- created a dataset-style folder layout for local annotation
- installed `labelme` into a local `.venv` after the system Python install path was blocked by the Arch externally-managed environment rules

Set up upstream dependencies locally:
- cloned `Clash-Royale-Detection-Dataset`
- cloned `KataCR`

Built a merged seed dataset:
- created a merged local dataset rooted at `seed_dataset` and later moved to `data/seed_dataset`
- imported the upstream `images/part2` annotated dataset
- added local clip frames under `capture_clip/1`
- preserved a mapping file for imported local frames

Added local dataset workflow scripts:
- `scripts/prepare_seed_dataset.py`
  - imports local frames into the merged seed dataset
- `scripts/build_seed_annotations.py`
  - rebuilds `.txt` labels and annotation index files from `.json` annotations

Patched `KataCR` for local use:
- updated `katacr/build_dataset/constant.py` to allow overriding `path_dataset` through `KATACR_DATASET_PATH`
- avoided relying on the original author’s hardcoded filesystem paths

Generated dataset annotations and split files:
- rebuilt YOLO-style `.txt` labels from the upstream and local seed dataset
- generated `annotation.txt`, `train_annotation.txt`, `val_annotation.txt`, `yolo_annotation.txt`
- generated `ClashRoyale_detection.yaml`
- generated `version_info/dataset.py`

Result at this stage:
- a local trainable seed dataset was available
- the project had a reproducible path from raw local frames to trainable detector data

## 2026-03-19

Verified GPU and training prerequisites:
- confirmed GPU visibility outside the sandbox on an NVIDIA GeForce RTX 2050
- verified that the default local environment was not suitable for detector training
- created a dedicated training environment in `.venv-train`

Restructured the repository:
- moved upstream repos under `vendor/external/`
- moved local mutable assets under `data/`
- kept backward-compatible symlinks:
  - `external -> vendor/external`
  - `part2 -> data/part2`
  - `seed_dataset -> data/seed_dataset`
  - `seed_labels -> data/seed_labels`
  - `video_clips -> data/video_clips`

Added project organization and baseline config:
- added `README.md` documenting structure and training workflow
- added `configs/katacr_train_baseline.yaml`
  - single-GPU-friendly baseline config for the RTX 2050

Installed the detector training stack in `.venv-train`:
- `torch`
- `torchvision`
- `ultralytics==8.1.24`
- `jax`, `jaxlib`
- `flax`, `optax`, `orbax-checkpoint`
- supporting Python packages required to make the KataCR YOLO path and import chain work

Generated detector-specific configs:
- added `scripts/setup_seed_detectors.py`
- ran `KataCR` detector config generation against the merged seed dataset
- produced:
  - `vendor/external/KataCR/katacr/yolov8/detector1/data.yaml`
  - `vendor/external/KataCR/katacr/yolov8/detector2/data.yaml`
  - combined detector config

Added local wrappers around the upstream detector workflow:
- `scripts/train_seed_baseline.py`
  - simplified detector training entry point
  - supports detector selection and later resume support
- `scripts/run_seed_inference.py`
  - simplified merged inference runner for one or more detectors

Resolved compatibility issues:
- redirected Ultralytics config storage into the workspace via `YOLO_CONFIG_DIR`
- forced trusted checkpoint loading behavior via `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`
- patched around NumPy 2.x / Ultralytics `np.trapz` incompatibility by mapping `np.trapz` to `np.trapezoid` in local wrappers

Smoke-tested training:
- confirmed CUDA was accessible from `.venv-train`
- verified that the baseline detector training path reached real GPU training batches on the RTX 2050

Result at this stage:
- the project had a working single-GPU detector training pipeline
- the workspace layout was cleaner and easier to maintain

## 2026-03-22

Started actual detector training on the merged seed dataset.

Tracked detector 1 training quality over early epochs:
- epoch 1 showed weak but expected initial performance
- epoch 2 showed meaningful improvement
- epoch 3 confirmed healthy learning
- epoch 4 and epoch 5 pushed the detector to a clearly usable pre-labeling baseline

Observed and interpreted key metrics:
- monitored `mAP50`, `mAP50-95`, precision, recall, and loss trends
- concluded detector 1 had become good enough for visual inference testing

Generated detector evaluation artifacts:
- F1-confidence curve
- recall-confidence curve
- confusion matrix and normalized confusion matrix
- used these to tune expectations around practical inference confidence thresholds

Result at this stage:
- detector 1 was no longer just training successfully; it had become useful for inference experiments

## 2026-03-23

Completed detector 2 training path and clarified checkpoint handling:
- distinguished between local resumable checkpoints and upstream pretrained weights
- confirmed the current detector 2 best checkpoint path under `runs/detector2_baseline_seed2/weights/best.pt`
- clarified that `--resume` is only for local interrupted training runs, not for evaluating arbitrary pretrained detector files

Added resume support to the local training wrapper:
- `scripts/train_seed_baseline.py` now supports `--resume`
- makes it possible to continue training directly from `last.pt`

Confirmed that battlefield side ownership is part of the model output:
- the detector predicts both object class and affiliation (`belong`)
- output is represented internally as `0/1` and rendered with different colors

Result at this stage:
- both detector checkpoints were trainable and manageable through local wrapper scripts

## 2026-03-24

Finished combined testing workflow for both detectors:
- confirmed best checkpoints:
  - `runs/detector1_baseline_seed6/weights/best.pt`
  - `runs/detector2_baseline_seed2/weights/best.pt`
- ran combined inference through the local merged inference wrapper

Added inference-time class suppression:
- patched `scripts/run_seed_inference.py` to support `--exclude-classes`
- made it possible to suppress classes like:
  - `clock`
  - `emote`
  - `tower-bar`
  - `king-tower-bar`
  - `bar`
  - `bar-level`
- this allowed visual testing to focus on battlefield units and spells rather than UI clutter

Documented detector class split:
- detector 1:
  - mostly smaller troop-like units and various light battlefield objects
- detector 2:
  - heavier units, buildings, and many spell/projectile classes
- both:
  - shared tower and HUD-adjacent classes used by the original KataCR detector design

Result at this stage:
- the combined detector pipeline was practical to run and easier to inspect visually

## 2026-03-25

Focused on dataset improvement workflow and pre-annotation automation.

Documented dataset structure and curation process:
- clarified the structure of `data/seed_dataset`
- clarified how real detection images are stored under `images/part2`
- clarified how synthetic assets live under `images/segment`
- clarified which generated files must be rebuilt after dataset edits

Clarified dataset editing rules:
- to add new data:
  - add numbered `.jpg` frames to `images/part2/<video>/<round>/`
  - annotate them
  - rebuild annotations
- to delete bad data:
  - remove `.jpg`, `.json`, and `.txt` together for a sample
  - rebuild annotations afterward
- to fix bad labels:
  - edit the `.json`
  - rebuild annotations afterward

Explained data improvement strategy:
- emphasized that targeted high-quality frames matter more than random quantity
- recommended focusing on weak classes, hard scenes, local domain mismatch, and crowded combat

Added pre-annotation export support:
- patched `scripts/run_seed_inference.py` with `--export-labelme`
- this exports:
  - extracted frames as numbered `.jpg`
  - matching pre-annotation `.json` files in Labelme-compatible format
- labels are exported using the same class-plus-affiliation style used in the original dataset, e.g. `archer0`, `archer1`

Note on annotation tooling:
- the local pre-annotation export currently writes Labelme-style JSONs
- the active annotation workflow later shifted toward CVAT usage
- the current exporter is still useful as an automatic pre-label generation step, but would need a separate export path if full native CVAT import/export is desired

Result at this stage:
- the project now supports automatic pre-annotation export in addition to training and inference
- the dataset curation workflow is defined end-to-end

## Current Project State

The project now includes:
- local Clash Royale HUD extraction code
- a cleaned workspace structure with `data/`, `vendor/`, `scripts/`, and `configs/`
- a merged Clash Royale seed detection dataset
- reproducible annotation rebuild scripts
- a dedicated GPU training environment
- a working two-detector KataCR-based training pipeline
- resume support for interrupted local runs
- merged inference over both detectors
- inference-time class suppression
- pre-annotation export for downstream annotation correction

Core local paths:
- workspace root: [capture](/home/keschler/Documents/Coding/cr-bot/capture)
- dataset root: [data/seed_dataset](/home/keschler/Documents/Coding/cr-bot/capture/data/seed_dataset)
- local videos: [data/video_clips](/home/keschler/Documents/Coding/cr-bot/capture/data/video_clips)
- upstream repos: [vendor/external](/home/keschler/Documents/Coding/cr-bot/capture/vendor/external)
- training env: [`.venv-train`](/home/keschler/Documents/Coding/cr-bot/capture/.venv-train)

Core local scripts:
- [scripts/prepare_seed_dataset.py](/home/keschler/Documents/Coding/cr-bot/capture/scripts/prepare_seed_dataset.py)
- [scripts/build_seed_annotations.py](/home/keschler/Documents/Coding/cr-bot/capture/scripts/build_seed_annotations.py)
- [scripts/setup_seed_detectors.py](/home/keschler/Documents/Coding/cr-bot/capture/scripts/setup_seed_detectors.py)
- [scripts/train_seed_baseline.py](/home/keschler/Documents/Coding/cr-bot/capture/scripts/train_seed_baseline.py)
- [scripts/run_seed_inference.py](/home/keschler/Documents/Coding/cr-bot/capture/scripts/run_seed_inference.py)
