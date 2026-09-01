# Clash Royale RL Bot

This project runs a recurrent Clash Royale policy from live Android gameplay.
It captures the newest phone frame, extracts public game state with computer
vision, converts it to `PolicyObservationV2`, and lets the policy choose
`WAIT` or `PLAY(card_slot, (column, row))`.

The primary entry point is the standalone Linux executable documented in
[`simulator/RUN_PROTOTYPE_LIVE.md`](simulator/RUN_PROTOTYPE_LIVE.md). It bundles
the Python runtime, CPU PyTorch, visual extractor, KataCR inference, the default
prototype checkpoint, card assets, ADB, and FFmpeg. Users do not need Python,
pip, a virtual environment, or a repository checkout.

<img width="1600" height="900" alt="cr-bot-banner-v1-dashboard" src="https://github.com/user-attachments/assets/1d225b86-79a3-4477-b524-2274faa92692" />

## Quick start

Download `prototype-live-linux-x86_64` from the
[latest GitHub release](https://github.com/keschler/cr-bot/releases/latest),
make it executable, and inspect its options:

```bash
chmod +x prototype-live-linux-x86_64
./prototype-live-linux-x86_64 --help
```

The executable is CPU-only and targets Linux x86-64. It includes the
`prototype-fast-current` checkpoint by default.
### Current best trained neural network

The current best neural network produced by training is
[`prototype.pt`](https://github.com/Keschler/cr-bot/releases/download/v0.5/prototype.pt).

You can use it with the live executable by passing it with `--checkpoint`:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt
```

### Test with a recorded video

```bash
./prototype-live-linux-x86_64 \
  --video /absolute/path/to/gameplay.mp4 \
  --max-frames 20 \
  --jsonl-out /tmp/prototype-video-dry-run.jsonl
```

Use another compatible recurrent checkpoint with `--checkpoint`:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --video /absolute/path/to/gameplay.mp4
```

### Run an ADB dry run

Enable USB debugging, authorize the computer, and pass the phone's exact ADB
serial:

```bash
./prototype-live-linux-x86_64 \
  --serial R7AIB700D744BX7 \
  --max-frames 100 \
  --jsonl-out /tmp/prototype-live-dry-run.jsonl
```

Dry run is the default: it observes and records decisions but never taps the
device. The binary deliberately does not guess which connected phone to use.

### Execute policy actions

Validate dry-run behavior first. Real taps require a reviewed, phone-specific
calibration artifact and both execution confirmations:

```bash
./prototype-live-linux-x86_64 \
  --serial R7AIB700D744BX7 \
  --calibration /absolute/path/to/phone-a-candidate.json \
  --execute \
  --confirm-live
```

Before every real play, the controller verifies the selected card and then
uses the calibrated card and arena coordinates. `WAIT` never taps the phone.
Stop safely with `Ctrl-C`; the controller does not force-stop the game or
delete device recordings or storage.

## Live pipeline

```text
newest Android frame
        |
        v
KataCR + HUD/card extraction
        |
        v
PolicyObservationV2 + legal-action masks
        |
        v
recurrent public actor
        |
        v
WAIT or verified card/arena taps
```

Live mode uses a persistent, serial-scoped H.264 screen stream decoded by the
bundled FFmpeg. Only the newest frame is retained, preventing slow CPU
inference from acting on a stale backlog. Use `--adb-transport screenshot` for
the diagnostic per-frame screenshot transport.

The default YOLO input size is `896`. `--yolo-image-size 640` is faster but can
reduce small-object detection quality and should be validated in dry runs
before live use.

## Development

Python 3.12 or newer is required for source development:

```bash
git clone https://github.com/Keschler/cr-bot.git
cd cr-bot
git submodule update --init vendor/external/KataCR
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The source launcher accepts the same options as the packaged executable:

```bash
python simulator/run_prototype_live.py \
  --video /absolute/path/to/gameplay.mp4 \
  --max-frames 20
```

Run the default test suite with:

```bash
pytest --ignore=tests/test_audio_dataset.py --ignore=tests/test_mining_pipeline.py
```

See [`simulator/RUN_PROTOTYPE_LIVE.md`](simulator/RUN_PROTOTYPE_LIVE.md) for
complete ADB setup, option reference, troubleshooting, safety gates, and native
PyInstaller build instructions.

## Simulator and RL status

The policy is trained against a deterministic, versioned Level-11 simulator.
The actor receives public observations only; privileged simulator state is
restricted to the training critic. The RL stack includes recurrent factorized
PPO, deterministic curriculum states, held-out evaluation, checkpoint
regression diagnosis, and simulation-exploit auditing.

The V1 research ruleset contains 124 definitions and the complete 109-card
eligible opponent roster, but remains explicitly `training_ready: false` until
its physical-fidelity evidence gates are satisfied. Current checkpoint results
are simulator evidence, not proof of live-game strength.

Detailed simulator architecture, training commands, evaluation gates, and
performance notes are in [`simulator/README.md`](simulator/README.md).

## Repository layout

```text
cr-bot/
├── simulator/
│   ├── run_prototype_live.py       primary source launcher
│   ├── RUN_PROTOTYPE_LIVE.md       distribution and live-operation guide
│   ├── prototype_live.spec         one-file PyInstaller build
│   ├── physical_lab/               phone control, calibration, and safety gates
│   ├── rl/                         policy, PPO, curriculum, and evaluation
│   ├── engine/                     deterministic battle engine
│   ├── rulesets/                   versioned card and mechanic definitions
│   ├── rosters/                    supported opponent roster data
│   ├── scenarios/                  deterministic simulator scenarios
│   └── tests/                      simulator-local tests
├── src/cr_bot/
│   ├── app/                        frame pipeline and runtime orchestration
│   ├── vision/                     detector, OCR, and frame extraction
│   ├── domain/                     shared game-state and card models
│   ├── features/                   policy feature and action-space builders
│   ├── trackers/                   temporal action and state tracking
│   ├── audio/                      audio models and features
│   ├── replay/                     serialized frame-analysis caches
│   └── eval/                       action evaluation tools
├── capture/                        capture tooling and packaged live dependencies
├── assets/                         detector, classifier, and template assets
├── configs/                        detector training configuration
├── data/                           evaluation inputs and local datasets
├── scripts/                        training, evaluation, and debugging scripts
├── tests/                          project tests, including tests/simulator/
├── vendor/external/KataCR/         patched detector dependency
└── pyproject.toml                  Python package and dependency configuration
```

## Current limitations

- The packaged executable currently targets Linux x86-64 and CPU inference.
- Live execution requires an accurate calibration for the exact phone/layout.
- Vision errors in cards, entities, elixir, timer, or tower HP can affect policy
  decisions; validate a dry run before enabling taps.
- The simulator is provisional and does not yet claim complete live-game
  mechanical fidelity.
- The current policy is an experimental prototype and has not demonstrated
  general live-game strength.

## Credits

Battlefield detection uses the patched KataCR dual-YOLO runtime. Hand-card and
next-card classification use project-trained MobileNetV3-Small models.
