# Mined Audio Dataset

`data/audio_classifier/mined/` is the working dataset root for mined real
gameplay audio used by the audio card classifier.

## Overview

The pipeline stores:

- source-video metadata
- downloaded and converted videos
- extracted full-match audio
- per-frame state streams
- mined enemy-card candidate events
- train/val/test manifests
- extracted per-event audio windows

## Top-Level Files

`sources.jsonl`

- One row per accepted source video.
- Includes metadata such as `video_id`, `upload_date`, `download_path`,
  `analysis_video_path`, `gameplay_wav_path`, `fps`, and `span_count`.

`skipped_sources.jsonl`

- Videos considered during mining but skipped.
- Includes the stage and reason for the skip.

`coverage.json`

- Aggregate card and quality-tier coverage across all exported manifest rows.

`spell_coverage.json`

- Spell-focused subset of the aggregate coverage report.

## Per-Video Directories

`downloads/`

- Downloaded source videos.
- May contain both the original downloaded file and an OpenCV-friendly
  converted file such as `*.opencv.mp4`.

`audio/`

- Full gameplay mono WAV extracted from each accepted source video.

`states/`

- Dense per-analysis-step JSONL state stream for each video.
- Each row represents a processed gameplay state snapshot and includes inferred
  enemy plays seen so far.

`spans/`

- Per-video JSON files describing continuous usable gameplay spans.

`candidates/`

- Per-video JSONL of mined enemy card-play candidates.
- These rows are the event candidates that later become train/val/test
  examples.

## Training Manifests

`manifests/train.jsonl`

- Final training manifest rows.

`manifests/val.jsonl`

- Final validation manifest rows.

`manifests/test.jsonl`

- Final test manifest rows.

These manifest files are the inputs consumed by the real-data stage of
`scripts/train_audio_classifier.py`.

## Extracted Audio Windows

`windows/train/`

- Per-event WAV snippets for training rows.

`windows/val/`

- Per-event WAV snippets for validation rows.

`windows/test/`

- Per-event WAV snippets for test rows.

These files are derived from the full-match gameplay WAVs in `audio/`. The
manifest files are the source of truth for which window files are active.

## End-to-End Flow

The mining pipeline works roughly like this:

1. Select source videos and write accepted rows to `sources.jsonl`.
2. Download source videos into `downloads/`.
3. Extract full gameplay WAV files into `audio/`.
4. Analyze frames and write per-step states into `states/`.
5. Detect continuous gameplay spans and write them into `spans/`.
6. Export mined enemy event candidates into `candidates/`.
7. Split candidate rows into `manifests/train.jsonl`,
   `manifests/val.jsonl`, and `manifests/test.jsonl`.
8. Extract per-event WAV windows into `windows/train/`, `windows/val/`, and
   `windows/test/`.

## What Matters For Training

For model training, the most important files are:

- `manifests/train.jsonl`
- `manifests/val.jsonl`
- `manifests/test.jsonl`
- the WAV files referenced by those manifests under `windows/`

If there are stale files in `windows/`, they do not matter as long as they are
not referenced by the manifests. The manifests define the active dataset.
