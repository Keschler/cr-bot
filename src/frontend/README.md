# cr-bot frontend (static UI v1)

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
- `scoring.py` — pure policy-scoring helpers, unchanged.
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
The page polls `GET /api/status` every 2 s and `GET /api/frames` every
1 s (playback speed steps the cursor through buffered frames instead of
changing the poll rate). The center image refreshes from `GET /api/frame/latest`.

UI preferences (overlay toggles, speed, device, checkpoint, transport)
persist in `localStorage` (`ara-settings-v1`) and are restored on reload.

## Video mode

1. Click the **Video** tab (or **Open Replay** in the top bar).
2. Choose a video file with the native file picker (it is uploaded to
   `POST /api/upload` first; the label then shows frame count/duration
   from `GET /api/video/info`), pick a checkpoint (defaults to `prototype.pt`),
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
   Seeks pause playback; polling continues so the timeline stays fresh.
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
   checkpoint (defaults to `prototype.pt`) and optional `calibration` profile.
3. Press **Start** (`POST /api/live/start`).
4. Press **Stop** in either mode to call `POST /api/stop`.

### Safety note for execute

Checking **Execute actions on device** lets the backend play cards on the
connected device. The UI blocks start unless the **confirm live control**
checkbox is also ticked, and shows a warning banner. Use a test account
and a dedicated device; never enable execute on an unattended phone.

## API table

| Method | Endpoint            | Purpose                                              |
|--------|---------------------|------------------------------------------------------|
| GET    | `/api/health`       | Liveness probe → `{ok: true}`                        |
| GET    | `/api/checkpoints`  | List `*.pt` candidates + default (`prototype.pt`)    |
| POST   | `/api/upload`       | Upload a video file (`multipart/form-data`)          |
| GET    | `/api/video/info?path=` | Probe frames/fps/duration/size of a video          |
| GET    | `/api/roi-preview?path=&frame=&overlay=` | Preview adapted ROIs → `{probe_frame, native_size, adapted, rois, warnings, image}` (404/422/501 on failures; `overlay=0` returns the raw probe frame so clients can draw/edit boxes themselves) |
| GET    | `/api/status`       | `{running, mode, error, summary, frame_count, latest_frame_index}` |
| POST   | `/api/video/start`  | `{video_path, checkpoint, start_frame, frame_stride, max_frames, device, adapt_rois, roi_set}` |
| POST   | `/api/live/start`   | `{serial, transport, checkpoint, device, calibration, execute, confirm_live}` |
| POST   | `/api/stop`         | Stop the current session                             |
| GET    | `/api/sessions`       | Saved recent replays (newest first, max 50)          |
| POST   | `/api/sessions`       | Upsert a replay entry `{name, video_path, params, cursor_frame, frame_count}` |
| DELETE | `/api/sessions/{name}` | Forget one saved replay                             |
| GET    | `/api/frames?since=N&limit=50` | `{frames: [{frame_index, timestamp_s, in_game, emitted, record: {visual_state, action, result}, suggestions, diagnostics}]}` |
| GET    | `/api/frame/latest` | Current frame as `image/jpeg`                        |
| GET    | `/api/frame/{index}` | One history frame as `image/jpeg` (204 if evicted) |
| GET    | `/api/labels` | Unit-class vocabulary + teams for label correction |
| POST   | `/api/frame/{index}/reevaluate` | What-if re-score with corrected labels `{updates, deletes, adds}` (404 evicted / 409 no actor / 400 malformed / 422 unbuildable) |
| DELETE | `/api/frame/{index}/reevaluate` | Revert a frame's what-if correction |
| GET    | `/api/stream`       | Optional SSE stream (polling `/api/frames` is enough for v1) |

Fixed ROIs assume `NATIVE_SIZE = [1080, 2400]`. The adapt UI applies only
when the probed video dims differ.

`device` is one of `auto` (default), `cpu`, or `cuda`, and pins both the
YOLO detector and the policy network. Explicit `cpu`/`cuda` override the
`YOLO_DEVICE` environment; `auto` keeps the existing auto-selection.
Requesting `cuda` without CUDA in the server environment fails with 400.
The resolved devices are reported in the session `summary.devices`.
CPU-only installs cannot use `cuda` (see `outputs/venv-gpu` for a CUDA
build); video throughput is ~3x higher on GPU.

Label correction is a stateless what-if: the DETECTED OBJECTS panel lets
you relabel a detection's class/team, delete false positives, or draw a
box for a missed unit, then re-score that frame with the live policy
(hidden state restored afterwards). The result overwrites only the
frame's displayed suggestions (badge + Revert); trackers, timeline
markers, history, and live execution are never touched. Frames expose
their raw `detections` (box + class + team + track) for this; only
emitted in-game frames within the bounded history can be revised.

The Edit labels button (top right, under the arena badge) enters a
drag-and-drop mode and pauses playback: drag a box to move it, drag a
corner handle to resize, click a box to select it (floating Team/Delete
actions, `Delete`/`t`/`Esc` shortcuts), or drag empty canvas to add a
missed unit with the picked label. Moves and resizes are updates carrying
the new box; everything else follows the same what-if pipeline.
