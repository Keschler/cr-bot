# Repository Guidance

## Project Overview

This is a Python 3.12+ computer-vision and tracking project that extracts
Clash Royale game state and card-play events from screenshots, recorded video,
or live capture. In the future this should be used for a behavior-cloning bot.

The installable package uses a `src` layout:

- `src/cr_bot/app/`: CLI, capture, replay, session, and frame pipeline
- `src/cr_bot/domain/`: shared data models, constants, card metadata, and ROIs
- `src/cr_bot/vision/`: detector, OCR, classifier, and frame extraction code
- `src/cr_bot/trackers/`: stateful own-action and enemy-action tracking
- `src/cr_bot/audio/`: audio models, datasets, features, and metrics
- `src/cr_bot/replay/`: serialized frame-analysis cache support
- `src/cr_bot/eval/`: action evaluation and visualization tools
- `src/cr_bot/mining/`: dataset mining and retention utilities
- `tests/`: pytest suite
- `scripts/`: training, evaluation, mining, and debugging entry points
- `vendor/external/KataCR/`: patched git submodule used by the detector runtime

## Setup

Create and activate a Python 3.12 virtual environment, then run:

```bash
git submodule update --init vendor/external/KataCR
python -m pip install --upgrade pip
pip install -e .
```

Dependencies are declared through `requirements.txt` and loaded by
`pyproject.toml`. Do not add a second dependency-management system unless the
task explicitly requires it.

When using an existing project environment, use `outputs/venv/`. Do not use
`outputs/venv-cpu/`.

## Development Commands

Run the full test suite:

```bash
pytest --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py
```

Run a focused test while iterating:

```bash
pytest tests/test_name.py
pytest tests/test_name.py::test_case_name
```

Do not run `tests/test_audio_dataset.py` or `tests/test_mining_pipeline.py`
unless the task concerns audio or audio-mining behavior. For audio-related
work, run them explicitly:

```bash
pytest tests/test_audio_dataset.py tests/test_mining_pipeline.py
```

Run the CLI from an editable installation:

```bash
cr-bot --help
```

Run the action-evaluation scenarios:

```bash
PYTHONPATH=src python -m cr_bot.eval.run_action_eval_scenarios
```

Prefer focused unit tests for tracker, domain, replay, audio, and mining
changes. Full video evaluation is substantially more expensive and may depend
on local media or model assets.

## Implementation Guidelines

- Follow the existing module boundaries. Keep shared value objects and events
  in `domain`, frame-local extraction in `vision`, and stateful temporal logic
  in `trackers`.
- Preserve the distinction between video time, match time, and frame index.
  Do not substitute one for another without an explicit conversion.
- Keep tracker state transitions deterministic. Tests should use small
  synthetic observation sequences where possible.
- Extend existing dataclasses and typed models rather than passing new
  unstructured dictionaries between pipeline stages.
- Keep CLI parsing in `src/cr_bot/app/cli.py` and runtime orchestration in the
  app/session/pipeline modules.
- Use `cr_bot.paths` or existing path helpers for repository assets rather than
  depending on the process working directory.
- Avoid loading heavyweight YOLO, Torch, JAX, or OCR models at module import
  time. Preserve lazy initialization patterns.
- Do not silently change coordinate systems, normalized frame dimensions, ROI
  conventions, card-name normalization, or action timing semantics.
- Add or update tests for behavior changes. Regression tests should reproduce
  the smallest observation sequence that demonstrates the bug.

## Data, Models, and Generated Output

- Treat files under `assets/models/` as large binary artifacts. Do not replace
  or regenerate them unless explicitly requested.
- Treat datasets, replay caches, videos, debug renders, and files under
  `outputs/` as local/generated artifacts unless the task specifically targets
  them.
- Do not commit mined datasets, caches, or generated media merely because a
  command produced them.
- Ground-truth files under `data/eval/ground_truth/` are curated evaluation
  inputs. Preserve their schema and avoid broad rewrites.
- When a test needs media or a model, skip cleanly if the repository already
  follows that pattern; otherwise prefer a lightweight fixture.

## KataCR Submodule

`vendor/external/KataCR` is a patched external dependency and a git submodule.
Do not edit, update, or replace its pinned revision unless the requested change
specifically concerns KataCR integration. Keep compatibility logic in
`src/cr_bot/vision/katacr_runtime.py` or the nearest existing adapter.

## Change Discipline

- Inspect `git status` before editing. The worktree may contain unrelated user
  changes; preserve them and do not revert or reformat them.
- Keep changes scoped to the requested behavior. Avoid opportunistic
  refactors, mass formatting, and generated-file churn.
- When committing changes, create separate commits for different kinds of
  work. Keep each commit coherent; for example, do not combine implementation,
  unrelated refactoring, documentation, generated data, or model updates in a
  single commit.
- Update README or files under `docs/` when public CLI behavior, setup,
  evaluation workflows, or architecture changes.
- Before finishing, run the narrowest relevant tests and then the default
  test command above when practical. Include the two audio-related test files
  only when working on audio or audio mining. Report any tests that could not
  be run and why.
