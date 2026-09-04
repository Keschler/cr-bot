# cr-bot frontend (static UI v1)

Vanilla HTML/CSS/JS dashboard for the Arena Replay Analyst. All data comes
from the real backend API — there are no mocks or bundled fixtures.

## Run

From the repository root:

```bash
uvicorn src.frontend.server:app
```

Then open `http://127.0.0.1:8000/` (or the host/port your server binds).
The page polls `GET /api/status` every 2 s and `GET /api/frames` every
`1 s / speed`. The center image refreshes from `GET /api/frame/latest`.

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
   Press **Preview ROIs** (`GET /api/roi-preview?path=`), inspect the debug
   image plus the `Frame N · X landmark / Y scaled` meta line, then tick
   **Use adapted ROIs for this run**. Start (`POST /api/video/start`) sends
   `adapt_rois` + `roi_set` and aborts with an error unless the preview was
   accepted (or the adapt checkbox is unchecked).
4. Press **Start** (`POST /api/video/start`).
5. Scrub history with the transport bar or the bottom timeline.
   Pausing stops frame polling; the status poll keeps running.

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
| GET    | `/api/roi-preview?path=&frame=` | Preview adapted ROIs → `{probe_frame, native_size, adapted, rois, warnings, image}` (404/422/501 on failures) |
| GET    | `/api/status`       | `{running, mode, error, summary, frame_count, latest_frame_index}` |
| POST   | `/api/video/start`  | `{video_path, checkpoint, start_frame, frame_stride, max_frames, adapt_rois, roi_set}` |
| POST   | `/api/live/start`   | `{serial, transport, checkpoint, calibration, execute, confirm_live}` |
| POST   | `/api/stop`         | Stop the current session                             |
| GET    | `/api/frames?since=N&limit=50` | `{frames: [{frame_index, timestamp_s, in_game, emitted, record: {visual_state, action, result}, suggestions, diagnostics}]}` |
| GET    | `/api/frame/latest` | Current frame as `image/jpeg`                        |
| GET    | `/api/frame/{index}` | One history frame as `image/jpeg` (204 if evicted) |
| GET    | `/api/stream`       | Optional SSE stream (polling `/api/frames` is enough for v1) |

Fixed ROIs assume `NATIVE_SIZE = [1080, 2400]`. The adapt UI applies only
when the probed video dims differ.
