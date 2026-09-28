# Build and run the V4 live binary

Build a new `dist/prototype-live-linux-x86_64` after producing a V4 checkpoint.
Earlier prototype release binaries use a different checkpoint format and do
not contain the V4 actor. The new build is a single Linux x86-64 executable
containing the Python runtime, the `cr_bot` visual extractor, KataCR inference
code, CPU PyTorch, the selected V4 checkpoint, card assets, and the
`adb`/`ffmpeg` helpers used by the live stream.

Recipients do not need Python, pip, a virtual environment, the repository, or
any Python packages. The binary is CPU-only and is built for Linux x86-64; a
different operating system or CPU architecture needs its own native build.

## Quick start

After building, copy the executable to the target machine, make it executable,
and check it:

```bash
chmod +x prototype-live-linux-x86_64
./prototype-live-linux-x86_64 --help
```

The binary embeds the V4 checkpoint selected at build time. To use another
compatible V4 state dict, pass its path with `--checkpoint`:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/v4_actor.pt \
  --video /absolute/path/to/gameplay.mp4 \
  --max-frames 20
```

The checkpoint must be compatible with the V4 actor loader. The
external checkpoint is the only model file needed; the extractor assets are
already inside the binary.

## ADB dry run

Enable USB debugging, authorize the computer on the phone, and use the exact
ADB serial. The binary includes its own ADB and FFmpeg, so they do not need to
be installed separately. The default `stream` transport uses a persistent H.264
screen stream and processes only the newest decoded frame.

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/v4_actor.pt \
  --serial R7AIB700D744BX7 \
  --max-frames 100 \
  --jsonl-out /tmp/prototype-live-dry-run.jsonl
```

The default is a dry run: it never sends taps. The original screenshot
transport remains available for diagnosis:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/v4_actor.pt \
  --serial R7AIB700D744BX7 \
  --adb-transport screenshot
```

The serial must be explicitly supplied for safety. If it is unknown, obtain it
from the Android device-management tooling or an existing `adb devices`
installation; the binary deliberately does not guess which phone to control.

## Execute on a phone

First validate a dry run. Real control additionally requires a reviewed,
phone-specific calibration artifact and both confirmation flags:

```bash
./prototype-live-linux-x86_64 \
  --checkpoint /absolute/path/to/v4_actor.pt \
  --serial R7AIB700D744BX7 \
  --calibration /absolute/path/to/phone-a-candidate.json \
  --execute \
  --confirm-live
```

The controller extracts each newest frame, creates the simulator's
`PolicyObservationV2`, carries the recurrent state forward, and dispatches
only the policy's `PLAY` actions. Before a real play it re-checks the selected
card and applies the calibrated card and arena taps. `WAIT` actions do not tap
the phone.

Stop with `Ctrl-C`. The controller does not force-stop the game or delete
recordings/storage.

## Useful options

- `--checkpoint PATH` selects the V4 recurrent neural network. If omitted, the
  checkpoint embedded at build time is used.
- `--yolo-image-size 896` is the default extractor resolution. `640` is a
  faster experimental setting that can reduce small-object detection quality.
- `--interval-s 0.25` controls the minimum live polling interval.
- `--jsonl-out PATH` records one extracted state/action record per processed
  frame.
- `--template-root PATH` overrides the bundled card-art template directory
  when a custom template set is required.
- `--adb-transport screenshot` switches from the default H.264 stream to the
  older per-frame screenshot path.

## Troubleshooting

- `Permission denied`: run `chmod +x prototype-live-linux-x86_64`.
- `selected ADB device is not connected`: check USB debugging, authorization,
  the cable, and the exact serial. The binary's embedded ADB still needs host
  permission to access the USB device.
- `V4 checkpoint does not exist`: pass an absolute path with
  `--checkpoint`, or omit it to use the embedded model.
- Calibration or card-template errors mean a live safety gate failed. Fix the
  artifact or path instead of bypassing the gate.
- The one-file executable extracts its bundled runtime to a temporary
  directory at startup. Set `TMPDIR` to a location with sufficient free disk
  space if the default temporary directory is too small.

## Building a new native binary

Builds are native. For the compact CPU artifact, use a build environment with
the CPU-only PyTorch wheels and the project's PyInstaller installation, then
run from the simulator directory:

```bash
CR_BOT_V4_CHECKPOINT=/absolute/path/to/v4_actor.pt \
  ../outputs/venv/bin/python -m PyInstaller --clean --noconfirm \
  prototype_live.spec
```

Set `CR_BOT_V4_CHECKPOINT` to a compatible plain V4 state-dict checkpoint to
embed that model in the binary. There is no historical checkpoint default.

The result is written to `dist/prototype-live-linux-x86_64`. The spec locates
the checkpoint, extractor assets, and the `adb`/`ffmpeg` binaries from the
checkout at build time. Build separately on each target OS/architecture.

## Source launcher

For development, the repository launcher `run_prototype_live.py` remains
available and has the same arguments. Pass `--checkpoint` or set
`CR_BOT_V4_CHECKPOINT`; the source launcher has no bundled model. It requires
the checkout's Python dependencies and is not the distribution artifact.
