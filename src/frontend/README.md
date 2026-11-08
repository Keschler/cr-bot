# cr-bot frontend

Vanilla HTML/CSS/JS dashboard for the Arena Replay Analyst. All data comes
from the real backend API — there are no mocks or bundled fixtures.

## Layout

Backend (`src/frontend/`, `src` layout so tests import both `frontend.*`
and `src.frontend.*`):

- `app.py` — `create_app()` factory (CORS, routers, static mount) + `app`.
  `server.py` is a thin back-compat shim re-exporting `app` and handlers.
- `api/` — one `APIRouter` per domain: `system`, `video`, `live`,
  `frames`, `roi`, `assets`, `corrections`, `stream` (`deps` shares the
  session manager).
- `models/` — `frames` (`FrontendFrame`), `session` (`FrontendSession`
  store only), `requests` (Pydantic request/response models).
- `services/` — `paths`, `devices`, `checkpoints`, `uploads`, `card_art`,
  `session_manager` (background-thread lifecycle + frame JSON).
- `runners/` — `pump` (frame-pump loop + tracker summarizers), `video`,
  `live`. `session.py` is a shim re-exporting these for old imports.
- `corrections/` — what-if label-correction engine (`_common`, `vocab`,
  `edits`, `observation`, `reevaluate`).
- `scoring.py` — pure helpers for ranking V4 actor suggestions.
- `imaging.py` — JPEG encode + frame-dimension helpers.

No heavy imports (`torch`/`cv2`/`cr_bot`) at module top: they stay lazy
inside functions so the server imports without GPU/CV deps.

Frontend (`static/`, native ES modules, no bundler — entry
`<script type="module" src="js/main.js">`):

- `js/main.js` — boot: binds all feature events, runs init loads.
- `js/state/store.js` — single shared mutable `state` + constants.
- `js/utils/` — `elements` (DOM/canvas handles), `dom`, `format`,
  `geometry` (canvas/coordinate mapping).
- `js/api/client.js` — fetch wrappers.
- `js/features/` — `frames` (frame accessors), `roi` (pure payload
  helpers), `roi-editor` (proposal review), `session` (polling, image,
  video/live controls), `panels` (side panels), `overlay` (canvas
  layers), `corrections` (drafts, edit mode, gestures), `timeline`
  (history, markers, `renderCurrent` fan-out).
- `styles/*.css` — topical stylesheets (`styles.css` is an `@import`
  shim so the old `<link>` keeps working).

## Run

From the repository root:

```bash
uvicorn src.frontend.server:app
```

Then open `http://127.0.0.1:8000/` (or the host/port your server binds).
The page opens an SSE stream (`GET /api/stream`) for ~0.25 s frame latency
while a session runs, with `GET /api/frames` polling as fallback and stall
re-sync. Both loops back off exponentially (to 30 s /
15 s) while the backend is unreachable; the status pill doubles as a
retry button, and a sticky toast offers Retry with reconnect confirmation.
Frame images load via fetch: frames evicted from the bounded server
history (HTTP 204) show a "frame evicted" badge with a one-time toast
instead of stale pixels, and overlays stay withheld.

UI preferences (overlay toggles, speed, device, checkpoint, transport)
persist in `localStorage` (`ara-settings-v1`) and are restored on reload.
Transient confirmations (uploads with progress bar, exports, link copies,
reconnects) appear as dismissible toasts; blocking form/start errors stay
in the persistent error bar.

## Video mode

1. Click the **Video** tab (or **Open Replay** in the top bar).
2. Choose a video file with the native file picker (it is uploaded to
   `POST /api/upload` first; the label then shows frame count/duration
   from `GET /api/video/info`), pick a V4 checkpoint from the configured
   checkpoint list,
   plus start frame (`0` = from the beginning), `frame_stride` and `max_frames`.
   The session panel collapses automatically once analysis runs with frames.
3. If the probed video size differs from the native `1080x2400` ROI space,
   an **Adapt ROIs to this video's format** checkbox appears (auto-checked)
   with a notice such as `1080x1920 detected — fixed ROIs assume 1080x2400.`.
   Checking it fetches the proposal automatically
   (`GET /api/roi-preview?path=`) and shows it in the middle player: the
   probe frame with one box per ROI (green = landmark, blue = scaled
   fallback, orange dashed = your adjustment, yellow = selected). Drag a
   box to move it, a corner to resize it (`Reset adjustments` reverts).
   If the probe frame is bad (menus, emotes, transitions), **Another
   frame** re-probes at a different point of the video — edits are reset.
   Pressing **Start** is the confirmation: it sends `adapt_rois` + `roi_set`
   (proposal plus your adjustments). Start aborts with an error when Adapt
   is on but no proposal is ready yet (still loading or failed — try
   Another frame, or uncheck the adapt checkbox).
4. Press **Start** (`POST /api/video/start`).
5. Scrub history with the transport bar or the bottom timeline.
   Seeks pause playback (polling continues so the timeline stays fresh);
   dragging to the far right — or the ● Live button — resumes live
   follow instead.
   Keyboard: `Space` play/pause, `←`/`→` or `j`/`l` step (`Shift` = 10),
   `Home`/`End` jump, `e` edit mode, `1`–`3` inspect a suggestion, `?`
   opens the shortcut help, `Esc` closes it. Keys no-op while typing.
   The transport bar also holds share actions: **JSON** (session frames +
   suggestions), **CSV** (confirmed own/foe plays), **Frame** (current
   frame JPEG), and **Link** (copies a `#f=<frame>&r=<rank>&v=<video>`
   deep link; opening it jumps back to that frame once frames arrive).

Every video start is recorded in the **Recent replays** list (backed by
`uploads/sessions.json` via `GET/POST /api/sessions` and
`DELETE /api/sessions/{name}`, newest first, capped at 50). **Resume**
re-runs the replay with its saved parameters and jumps back to the saved
cursor once analysis catches up. Replays that used adapted ROIs are not
auto-resumed — re-check Adapt ROIs and Start manually.

## Live mode

1. Click the **Live** tab (or **Dashboard** in the top bar).
2. Enter the ADB `serial`, `transport` (`stream` / `screenshot`), pick a
   V4 checkpoint and optional `calibration` profile. Set
   `CR_BOT_V4_CHECKPOINT` to expose one explicit default.
   Leaving calibration blank uses
   `simulator/physical_lab/calibrations/phone-a-candidate.json` for the ASUS
   AI2302 (1080×2400). Enter a device-specific JSON path for another layout;
   relative paths resolve from the repository root.
3. Press **Start** (`POST /api/live/start`).
4. Press **Stop** in either mode to call `POST /api/stop`.

H.264 **Stream capture width** defaults to **Native**, preserving image detail.
Choosing **720 px** on a 1080×2400 phone makes the encoder and FFmpeg process
720×1600 frames, about 56% fewer pixels; **480 px** reduces capture load further.
This setting changes capture resolution, independently of YOLO input
size and live JPEG size. Frames are restored to the phone's original dimensions
before analysis and action verification, preserving ROI and tap coordinates;
reduced capture resolution can still affect OCR and small-object accuracy.
Screenshot transport ignores this setting. The frontend API accepts
`stream_max_width_px` (default `null`, meaning native). CLI capture retains
native resolution; direct Python callers can set a cap explicitly.

With H.264 streaming, the live picture reads the newest decoded capture
independently of analysis, with one image request at a time (up to about
10 updates per second). The live JPEG is at most 480 pixels wide, with fast
resizing and JPEG quality 80; analyzed history keeps the 720-pixel, quality-85
images. Capture starts during model loading. The picture
badge says gameplay age is unknown: elementary H.264 has no trusted phone
capture timestamp. Boxes and suggestions are shown only on their matching
pixels. **Pause** or scrub the timeline
to inspect an analyzed frame with its detections and suggestions, and
**Live** returns to the current picture.

The session's **history frames** count updates when analyzed frames arrive,
alongside their metadata. **Frame ID** is the selected capture frame's index;
it can jump because analysis samples the newest capture. It is independent
of the number of analyzed frames retained on the timeline. Pausing holds the
selected Frame ID while new history frames can continue arriving.
Every completed analysis is published, including outside battles; history
updates are no longer throttled to once per second in the lobby.

The decoder uses reduced H.264 probing to avoid holding back incoming
frames. Analysis takes the newest available frame whenever it is ready;
the frontend live runner no longer inserts the previous 250 ms minimum
interval. Actual analysis throughput still depends on the detector, HUD
extraction, policy, and hardware. Screenshot transport uses its latest
analyzed picture. Frame ages use workstation decode timestamps and do
not measure delay inside the phone or ADB transport.

Raw H.264 frames are read into private writable buffers without joining
pipe chunks or making a second full-frame image copy. The extraction pipeline
shares one timer localization between text and overtime checks. Card crops
retain their original classifier inputs while avoiding Torch CPU thread-pool
work for small normalization operations and copying output scores in batches.

### Measure headless live throughput

Stop the frontend before measuring to avoid competing GPU work. The headless
benchmark uses native H.264 capture, both YOLO detectors on CUDA, and the selected
V4 policy on CUDA, with no preview JPEGs, browser, or device taps. It excludes
initial model loading and five warmup frames; leave a battle/replay visible to
measure policy calls, since lobby frames skip policy inference.

```bash
outputs/venv-gpu/bin/python scripts/benchmark_live_stream.py \
  --serial YOUR_SERIAL --checkpoint PATH_TO_ACTOR.pt \
  --seconds 45 --yolo-image-size 896 \
  --output outputs/headless-gpu-benchmark.json
```

Use the same CUDA environment as the frontend; `outputs/venv-gpu` is the active
frontend environment on the benchmarked workstation. The JSON reports completed
analysis FPS, decoded capture FPS, policy-call count, and vision-stage medians.
`--capture-only` measures capture without loading models. These FPS values do
not measure phone-to-display age.

### Diagnose slow live capture

Stop the backend and restart it with diagnostics enabled, then hard-refresh
the browser to load the current JavaScript:

```bash
CR_BOT_LIVE_DIAGNOSTICS=1 outputs/venv/bin/python -m uvicorn src.frontend.server:app --reload 2>&1 | tee outputs/live-debug.log
```

Run live mode until the slowdown occurs, for at least 30 seconds, then press
**Stop**. `LIVE_DIAGNOSTICS` JSON lines appear every two seconds, including
during model loading and an active analysis stall. They report decoder
output FPS and read waits, stream restarts and bounded ADB/FFmpeg errors,
per-stage vision and policy wall times, selected devices and CPU thread
counts, preview handler/dispatch times, and browser request/body/decode times
and changed-picture FPS. Browser reports are collected only while following
live preview, with at most one diagnostic report request in flight.
Logs also identify `in_game`, whether action execution is enabled, and time
spent dispatching actions or waiting afterward, so these waits are distinct
from inference time.

`GET /api/live/diagnostics` exposes the same snapshot and retains it after
stopping. `X-Frame-Clock: workstation-decode-publication` identifies the
preview age headers' clock; `X-Capture-Age-Ms` starts when a decoded picture
is published on the workstation. These values can stay small while the
phone stream is delayed. `phone_to_display_age_ms` is therefore explicitly
unknown (`null`). Measurements identify workstation bottlenecks; proving
phone-to-display delay still requires comparing the phone and preview.

### Safety note for execute

Checking **Execute actions on device** lets the backend play cards on the
connected device. The UI blocks start unless the **confirm live control**
checkbox is also ticked, and shows a warning banner. Use a test account
and a dedicated device; never enable execute on an unattended phone.

## API table

| Method | Endpoint            | Purpose                                              |
|--------|---------------------|------------------------------------------------------|
| GET    | `/api/health`       | Liveness probe → `{ok: true}`                        |
| GET    | `/api/checkpoints`  | List V4 `*.pt` candidates + configured default       |
| POST   | `/api/upload`       | Upload a video file (`multipart/form-data`)          |
| GET    | `/api/video/info?path=` | Probe frames/fps/duration/size of a video          |
| GET    | `/api/roi-preview?path=&frame=&overlay=` | Preview adapted ROIs → `{probe_frame, native_size, adapted, rois, warnings, image}` (404/422/501 on failures; `overlay=0` returns the raw probe frame so clients can draw/edit boxes themselves) |
| GET    | `/api/status`       | `{running, mode, error, summary, frame_count, latest_frame_index}` |
| POST   | `/api/video/start`  | `{video_path, checkpoint, start_frame, frame_stride, max_frames, device, adapt_rois, roi_set}` |
| POST   | `/api/live/start`   | `{serial, transport, stream_max_width_px, checkpoint, device, calibration, execute, confirm_live}` |
| GET    | `/api/live/diagnostics` | Opt-in live performance snapshot; disabled unless `CR_BOT_LIVE_DIAGNOSTICS=1` |
| POST   | `/api/live/preview-metrics` | Opt-in browser timing sample for the matching running diagnostic session (204) |
| POST   | `/api/stop`         | Stop the current session                             |
| GET    | `/api/sessions`       | Saved recent replays (newest first, max 50)          |
| POST   | `/api/sessions`       | Upsert a replay entry `{name, video_path, params, cursor_frame, frame_count}` |
| DELETE | `/api/sessions/{name}` | Forget one saved replay                             |
| GET    | `/api/frames?since=N&limit=50` | `{frames: [{frame_index, timestamp_s, in_game, emitted, record: {visual_state, action, result}, suggestions, diagnostics}]}` |
| GET    | `/api/frame/latest` | Latest analyzed frame as `image/jpeg`; `X-Frame-Index` identifies its pixels |
| GET    | `/api/frame/preview` | Newest decoded live capture as `image/jpeg` (analyzed-frame fallback); `X-Frame-Index`, live decoder-publication age headers and `X-Frame-Clock`; no-store; opt-in `X-Preview-Server-Ms` |
| GET    | `/api/frame/{index}` | One history frame as `image/jpeg` (204 if evicted) |
| GET    | `/api/labels` | Unit-class vocabulary + teams for label correction |
| POST   | `/api/frame/{index}/reevaluate` | What-if re-score with corrected labels `{updates, deletes, adds}` (404 evicted / 409 no actor / 400 malformed / 422 unbuildable) |
| DELETE | `/api/frame/{index}/reevaluate` | Revert a frame's what-if correction |
| GET    | `/api/stream`       | SSE frame stream (UI primary; polling fallback + stall re-sync) |

Fixed ROIs assume `NATIVE_SIZE = [1080, 2400]`. The adapt UI applies only
when the probed video dims differ.

`device` is one of `auto` (default), `cpu`, or `cuda`, and pins both the
YOLO detector and the policy network. Explicit `cpu`/`cuda` override the
`YOLO_DEVICE` environment; `auto` keeps the existing auto-selection.
Requesting `cuda` without CUDA in the server environment fails with 400.
The resolved devices are reported in the session `summary.devices` and
shown in the reasoning card, alongside the per-frame pipeline time
(`diagnostics.timing_ms`, full breakdown on hover). The arena grid also
shades the river band and bridge columns served by `GET /api/grid`.
CPU-only installs cannot use `cuda`. Use the project environment at
`outputs/venv/` with a CUDA-enabled PyTorch installation when GPU inference is
needed; measure throughput on the target machine.

Label correction is a stateless what-if: the DETECTED OBJECTS panel lets
you relabel a detection's class/team, delete false positives, or draw a
box for a missed unit, then re-score that frame with the live policy
(hidden state restored afterwards). The result overwrites only the
frame's displayed suggestions (badge + Revert); a diff row shows the
original top action → the corrected one. Trackers, timeline
markers, history, and live execution are never touched. Frames expose
their raw `detections` (box + class + team + track) for this; only
emitted in-game frames within the bounded history can be revised.

The Edit labels button (top right, under the arena badge) enters a
drag-and-drop mode and pauses playback: drag a box to move it, drag a
corner handle to resize, click a box to select it (floating Team/Delete
actions, `Delete`/`t`/`Esc` shortcuts), or drag empty canvas to add a
missed unit with the picked label. Moves and resizes are updates carrying
the new box; everything else follows the same what-if pipeline.
