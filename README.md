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

## Run the packaged binary

Download `prototype-live-linux-x86_64` from the
[latest GitHub release](https://github.com/keschler/cr-bot/releases/latest).

```text
phone → live stream → visual extractor → public state
      → policy → card + placement → ADB → phone action
```

> [!NOTE]
> The Linux x86-64 binary is approximately **800 MB** because it bundles the
> Python runtime, visual-extractor assets, policy, ADB, FFmpeg, and their
> dependencies. It uses **CPU inference only**, which is substantially slower
> than GPU inference. An RTX 2050 / Intel i9-13900H machine reached up to
> **4 FPS** during a live-action run. A GPU build is not distributed because
> its additional dependencies would make the download significantly larger.

### 1. Make it executable

```bash
chmod +x prototype-live-linux-x86_64
```

### 2. Test on a video

Use the current best trained policy,
[`prototype.pt`](https://github.com/Keschler/cr-bot/releases/download/v0.5/prototype.pt):

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --video /absolute/path/to/gameplay.mp4 \
  --max-frames 20
```

`--checkpoint` is optional when the bundled `prototype-fast-current` policy is
sufficient.

### 3. Dry run on a phone

Enable USB debugging, connect and authorize the phone, then use its exact ADB
serial:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --serial YOUR_PHONE_SERIAL \
  --max-frames 100 \
  --jsonl-out /tmp/prototype-live.jsonl
```

> [!NOTE]
> This is a dry run by default: it observes the game and records decisions,
> but never taps the phone. The default transport is a low-latency H.264
> stream; use `--adb-transport screenshot` for diagnosis.

### 4. Enable live actions

> [!WARNING]
> Live actions are calibrated only for a **1080×2400 phone**. Do not enable
> taps on another resolution or layout without a separately reviewed
> calibration.

Validate dry runs first. Real taps require a phone-specific calibration file
and both confirmation flags:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/prototype.pt \
  --serial YOUR_PHONE_SERIAL \
  --calibration /absolute/path/to/phone-calibration.json \
  --execute \
  --confirm-live
```

Before every `PLAY`, the controller verifies the selected card and applies the
calibrated card and arena taps. `WAIT` never taps the phone. Press `Ctrl-C` to
stop the controller.

## Development

Python 3.12 is required for source development because the pinned PyTorch and
JAX wheels do not support the newer system Python versions:

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
