"""FastAPI server for the cr-bot frontend preview UI (light import surface).

Only fastapi/pydantic/threading/pathlib (plus stdlib) are imported here.
Heavy CV/torch/cr_bot imports stay inside ``session.py`` workers and are
loaded lazily per background thread.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .session import FrontendSession, run_live_session, run_video_session


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = REPO_ROOT / "uploads"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
CARD_ART_CANDIDATES = (
    REPO_ROOT / "assets/templates/cr-api-assets/cards-gold",
    REPO_ROOT / "capture/templates/cr-api-assets/cards-gold",
)
# Extractor/ruleset names that differ from the card-art file names.
CARD_ART_ALIASES = {"log": "the-log"}
_CARD_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def _card_art_dir() -> Path | None:
    for candidate in CARD_ART_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def _card_art_candidates(name: str) -> list[str]:
    norm = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    norm = _CARD_SAFE_RE.sub("", norm).strip("-")
    if not norm:
        return []
    options = [norm]
    alias = CARD_ART_ALIASES.get(norm)
    if alias and alias not in options:
        options.append(alias)
    return options
CHECKPOINT_GLOBS = (
    "prototype.pt",
    "*.pt",
    "simulator/outputs/**/*.pt",
    "outputs/**/*.pt",
)


class VideoStartRequest(BaseModel):
    video_path: str
    frame_stride: int = 1
    start_frame: int = 0
    max_frames: int | None = None
    checkpoint: str | None = None
    device: str = "cpu"
    yolo_image_size: int = 896
    # Any + manual 400 checks so wrong JSON types map to 400 (not 422).
    adapt_rois: Any = False
    roi_set: Any = None


class LiveStartRequest(BaseModel):
    serial: str
    transport: str = "stream"
    checkpoint: str | None = None
    device: str = "cpu"
    calibration: str | None = None
    execute: bool = False
    confirm_live: bool = False


class StopResponse(BaseModel):
    stopped: bool = True
    running: bool = False


app = FastAPI(title="cr-bot frontend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_lock = threading.Lock()
_session = FrontendSession(mode="idle")
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None


def _get_session() -> FrontendSession:
    with _lock:
        return _session


def _resolve_against_repo(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def _default_checkpoint() -> str:
    preferred = REPO_ROOT / "prototype.pt"
    if preferred.is_file():
        return str(preferred)
    try:
        from simulator.physical_lab.prototype_controller import DEFAULT_CHECKPOINT
    except ImportError:
        from physical_lab.prototype_controller import DEFAULT_CHECKPOINT  # type: ignore
    return str(DEFAULT_CHECKPOINT)


def _list_checkpoints(limit: int = 20) -> tuple[list[dict[str, Any]], str | None]:
    """List candidate checkpoint files; default prefers repo-root prototype.pt."""
    seen: dict[str, None] = {}
    candidates: list[Path] = []
    for pattern in CHECKPOINT_GLOBS:
        try:
            paths = sorted(REPO_ROOT.glob(pattern))
        except (NotImplementedError, ValueError):
            continue
        for path in paths:
            if not path.is_file() or path.suffix != ".pt":
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen[key] = None
            candidates.append(path)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    default = _default_checkpoint()
    items = [
        {
            "path": str(path),
            "name": path.name
            if path.parent == REPO_ROOT
            else str(path.relative_to(REPO_ROOT)),
            "default": str(path) == default,
        }
        for path in candidates
    ]
    # Ensure the default is listed even when the glob missed it.
    if default and not any(item["path"] == default for item in items):
        path = Path(default)
        items.insert(
            0,
            {
                "path": default,
                "name": path.name,
                "default": True,
                "missing": not path.is_file(),
            },
        )
    if not any(item.get("default") for item in items) and items:
        items[0]["default"] = True
        default = items[0]["path"]
    return items, default


def _sanitize_upload_filename(filename: str | None) -> str:
    raw = (filename or "upload").strip() or "upload"
    name = Path(raw).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "upload"
    if Path(cleaned).suffix.lower() not in VIDEO_EXTENSIONS:
        cleaned += ".mp4"
    return cleaned[:128]


def _stop_current_locked() -> None:
    global _thread, _stop_event
    event = _stop_event
    thread = _thread
    if event is not None:
        try:
            event.set()
        except Exception:
            pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
    _thread = None
    _stop_event = None


def stop_current_session() -> FrontendSession:
    """Stop the background worker (if any) and return the current session."""

    with _lock:
        _stop_current_locked()
        session = _session
        session.running = False
        return session


def _start_worker(
    *, mode: str, target: Any, kwargs: dict[str, Any]
) -> FrontendSession:
    global _session, _thread, _stop_event
    with _lock:
        _stop_current_locked()
        session = FrontendSession(mode=mode)
        event = threading.Event()
        kwargs = dict(kwargs)
        kwargs["stop_event"] = event

        def _runner() -> None:
            try:
                target(session, **kwargs)
            except Exception:
                # run_* already records session.error fail-closed.
                pass

        thread = threading.Thread(target=_runner, name=f"frontend-{mode}", daemon=True)
        _session = session
        _stop_event = event
        _thread = thread
        session.running = True
        thread.start()
        return session


def _frame_to_json(frame: Any) -> dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "timestamp_s": frame.timestamp_s,
        "in_game": frame.in_game,
        "emitted": frame.emitted,
        "record": frame.record,
        "suggestions": frame.suggestions,
        "diagnostics": frame.diagnostics,
        "has_image": frame.jpeg_bytes is not None,
        "frame_width": getattr(frame, "frame_width", None),
        "frame_height": getattr(frame, "frame_height", None),
        "own_actions": getattr(frame, "own_actions", []),
        "enemy_plays": getattr(frame, "enemy_plays", []),
    }


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return _get_session().to_status_dict()


@app.get("/api/checkpoints")
def api_checkpoints() -> dict[str, Any]:
    items, default = _list_checkpoints()
    return {"checkpoints": items, "default": default}


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = _sanitize_upload_filename(getattr(file, "filename", None))
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail=f"could not create upload directory: {error}"
        ) from error
    destination = UPLOAD_DIR / filename
    stem, suffix = destination.stem, destination.suffix
    counter = 1
    while destination.exists():
        counter += 1
        destination = UPLOAD_DIR / f"{stem}-{counter}{suffix}"
        if counter > 999:
            raise HTTPException(status_code=409, detail="upload name collision")
    try:
        with destination.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail=f"could not store upload: {error}"
        ) from error
    finally:
        try:
            file.file.close()
        except (AttributeError, OSError):
            pass
    try:
        display = str(destination.relative_to(REPO_ROOT))
    except ValueError:
        display = str(destination)
    return {
        "path": display,
        "filename": destination.name,
        "size": destination.stat().st_size,
    }


@app.post("/api/video/start")
def api_video_start(request: VideoStartRequest) -> dict[str, Any]:
    video_path = _resolve_against_repo(request.video_path)
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path is required")
    if not Path(video_path).is_file():
        raise HTTPException(
            status_code=404, detail=f"video file does not exist: {video_path}"
        )
    if type(request.frame_stride) is not int or request.frame_stride <= 0:
        raise HTTPException(status_code=400, detail="frame_stride must be positive")
    if type(request.start_frame) is not int or request.start_frame < 0:
        raise HTTPException(status_code=400, detail="start_frame must be non-negative")
    if request.max_frames is not None and (
        type(request.max_frames) is not int or request.max_frames <= 0
    ):
        raise HTTPException(status_code=400, detail="max_frames must be positive")
    if type(request.yolo_image_size) is not int or request.yolo_image_size <= 0:
        raise HTTPException(status_code=400, detail="yolo_image_size must be positive")
    if type(request.adapt_rois) is not bool:
        raise HTTPException(status_code=400, detail="adapt_rois must be a bool")
    if request.roi_set is not None and not isinstance(request.roi_set, dict):
        raise HTTPException(status_code=400, detail="roi_set must be an object or null")
    checkpoint = _resolve_against_repo(request.checkpoint) or _default_checkpoint()
    if not Path(checkpoint).is_file():
        raise HTTPException(
            status_code=404, detail=f"checkpoint file does not exist: {checkpoint}"
        )
    if request.adapt_rois:
        try:
            import cv2  # lazy

            _cap = cv2.VideoCapture(video_path)
            try:
                if not _cap.isOpened():
                    raise ValueError(f"could not open video: {video_path}")
                _nw = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                _nh = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                try:
                    _cap.release()
                except Exception:
                    pass
            if _nw <= 0 or _nh <= 0:
                raise ValueError(f"could not probe video size: {video_path}")
            try:
                from cr_bot.vision.roi_adapt import validate_and_merge
            except ImportError as error:
                raise HTTPException(
                    status_code=501, detail="roi adaptation requires OpenCV"
                ) from error
            validate_and_merge(request.roi_set, _nw, _nh)
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        session = _start_worker(
            mode="video",
            target=run_video_session,
            kwargs={
                "video_path": video_path,
                "checkpoint": checkpoint,
                "device": request.device or "cpu",
                "frame_stride": request.frame_stride,
                "start_frame": request.start_frame,
                "max_frames": request.max_frames,
                "yolo_image_size": request.yolo_image_size,
                "adapt_rois": request.adapt_rois,
                "roi_set": request.roi_set,
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return session.to_status_dict()


@app.post("/api/live/start")
def api_live_start(request: LiveStartRequest) -> dict[str, Any]:
    serial = (request.serial or "").strip()
    if not serial:
        raise HTTPException(status_code=400, detail="serial must be non-empty")
    if request.transport not in ("stream", "screenshot"):
        raise HTTPException(
            status_code=400, detail="transport must be 'stream' or 'screenshot'"
        )
    if request.execute and not request.confirm_live:
        raise HTTPException(
            status_code=400,
            detail="live execution requires confirm_live=True",
        )
    if request.execute and not (request.calibration or "").strip():
        raise HTTPException(
            status_code=400,
            detail="live execution requires a calibration artifact path",
        )
    checkpoint = _resolve_against_repo(request.checkpoint) or _default_checkpoint()
    if not Path(checkpoint).is_file():
        raise HTTPException(
            status_code=404, detail=f"checkpoint file does not exist: {checkpoint}"
        )
    calibration = _resolve_against_repo(request.calibration)
    try:
        session = _start_worker(
            mode="live",
            target=run_live_session,
            kwargs={
                "serial": serial,
                "transport": request.transport,
                "checkpoint": checkpoint,
                "device": request.device or "cpu",
                "calibration": calibration,
                "execute": bool(request.execute),
                "confirm_live": bool(request.confirm_live),
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return session.to_status_dict()


@app.post("/api/stop", response_model=StopResponse)
def api_stop() -> StopResponse:
    session = stop_current_session()
    _ = session
    return StopResponse(stopped=True, running=False)


@app.get("/api/card-icon")
def api_card_icon(name: str = Query(default="")) -> Response:
    options = _card_art_candidates(name)
    if not options:
        raise HTTPException(status_code=400, detail="card name is required")
    art_dir = _card_art_dir()
    if art_dir is None:
        raise HTTPException(status_code=404, detail="card art is not available")
    for option in options:
        path = art_dir / f"{option}.png"
        if path.is_file():
            return Response(
                content=path.read_bytes(),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )
    raise HTTPException(
        status_code=404, detail=f"no card art for {options[0]!r}"
    )


@app.get("/api/grid")
def api_grid() -> dict[str, Any]:
    """Serve the authoritative action-grid spec (mirrors ACTION_GRID)."""
    try:
        from cr_bot.features.action_space import (
            ACTION_GRID,
            BRIDGE_COLS,
            OWN_SIDE_FIRST_ROW,
            RIVER_ROWS,
        )
        from cr_bot.domain.constants import KING_TOWER_HP, PRINCESS_TOWER_HP
    except ImportError as error:
        raise HTTPException(
            status_code=404, detail="grid spec is not available"
        ) from error
    return {
        "cols": ACTION_GRID.cols,
        "rows": ACTION_GRID.rows,
        "x0": ACTION_GRID.x0,
        "y0": ACTION_GRID.y0,
        "x1": ACTION_GRID.x1,
        "y1": ACTION_GRID.y1,
        "river_rows": list(RIVER_ROWS),
        "bridge_cols": list(BRIDGE_COLS),
        "own_side_first_row": OWN_SIDE_FIRST_ROW,
        "tower_hp": {"princess": PRINCESS_TOWER_HP, "king": KING_TOWER_HP},
    }


@app.get("/api/video/info")
def api_video_info(path: str = Query(default="")) -> dict[str, Any]:
    """Probe a server-local video file (frame count, fps, size)."""
    video_path = _resolve_against_repo(path)
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(
            status_code=404, detail=f"video file does not exist: {path}"
        )
    try:
        import cv2
    except ImportError as error:
        raise HTTPException(
            status_code=501, detail="video probing requires OpenCV"
        ) from error

    def finite_or_none(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        import math

        return result if math.isfinite(result) and result > 0 else None

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise HTTPException(
                status_code=422, detail=f"could not open video: {path}"
            )
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = finite_or_none(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    try:
        display = str(Path(video_path).relative_to(REPO_ROOT))
    except ValueError:
        display = str(video_path)
    duration_s = frames / fps if frames and fps else None
    return {
        "path": display,
        "filename": Path(video_path).name,
        "frames": frames or None,
        "fps": fps,
        "duration_s": duration_s,
        "width": width or None,
        "height": height or None,
    }


@app.get("/api/roi-preview")
def api_roi_preview(
    path: str = Query(default=""),
    frame: int | None = Query(default=None),
) -> dict[str, Any]:
    """Preview adapted ROIs for one video frame."""
    video_path = _resolve_against_repo(path)
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(
            status_code=404, detail=f"video file does not exist: {path}"
        )
    try:
        import cv2
    except ImportError as error:
        raise HTTPException(
            status_code=501, detail="video probing requires OpenCV"
        ) from error
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise HTTPException(
                status_code=422, detail=f"could not open video: {path}"
            )
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame is None:
            probe_index = total // 2 if total > 0 else 0
        else:
            try:
                probe_index = int(frame)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422, detail=f"invalid frame index: {frame}"
                )
            if probe_index < 0:
                raise HTTPException(
                    status_code=422, detail=f"invalid frame index: {frame}"
                )
            if total and probe_index >= total:
                raise HTTPException(
                    status_code=422, detail=f"frame {probe_index} past EOF ({total})"
                )
        capture.set(cv2.CAP_PROP_POS_FRAMES, probe_index)
        ok, native = capture.read()
        if not ok or native is None or getattr(native, "size", 0) == 0:
            raise HTTPException(
                status_code=422, detail=f"could not read frame {probe_index}: {path}"
            )
        try:
            native_h, native_w = native.shape[:2]
        except Exception:
            raise HTTPException(
                status_code=422, detail=f"could not read frame {probe_index}: {path}"
            )
    finally:
        try:
            capture.release()
        except Exception:
            pass
    try:
        from cr_bot.vision.roi_adapt import adapt_rois_for_probe
    except ImportError as error:
        raise HTTPException(
            status_code=501, detail="roi adaptation requires OpenCV"
        ) from error
    try:
        roi_entries, overlay_jpeg, warnings, (nw, nh) = adapt_rois_for_probe(native)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        display = str(Path(video_path).relative_to(REPO_ROOT))
    except ValueError:
        display = str(video_path)
    image_url: str | None = None
    if overlay_jpeg is not None:
        import base64

        image_url = "data:image/jpeg;base64," + base64.b64encode(overlay_jpeg).decode(
            "ascii"
        )
    return {
        "video": display,
        "probe_frame": int(probe_index),
        "native_size": [int(nw), int(nh)],
        "adapted": [int(nw), int(nh)] != [1080, 2400],
        "rois": roi_entries,
        "warnings": list(warnings),
        "image": image_url,
    }


@app.get("/api/frames")
def api_frames(
    since: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=512),
) -> dict[str, Any]:
    session = _get_session()
    frames = session.get_since(int(since), int(limit))
    return {
        "frames": [_frame_to_json(f) for f in frames],
        "count": len(frames),
        "since": int(since),
    }


@app.get("/api/frame/latest")
def api_frame_latest() -> Response:
    session = _get_session()
    latest = session.get_latest()
    if latest is None or latest.jpeg_bytes is None:
        return Response(status_code=204)
    return Response(content=latest.jpeg_bytes, media_type="image/jpeg")


@app.get("/api/frame/{frame_index}")
def api_frame_at(frame_index: int) -> Response:
    """Serve one history frame's JPEG so scrubbing shows matching imagery.

    Frame images are immutable per index and cacheable. Unknown or evicted
    indices (bounded history) answer 204; the UI then keeps its last image
    and withholds frame-bound overlays instead of misaligning them.
    """
    session = _get_session()
    for frame in session.history:
        try:
            match = int(frame.frame_index) == int(frame_index)
        except (TypeError, ValueError, OverflowError):
            match = False
        if match and frame.jpeg_bytes is not None:
            return Response(
                content=frame.jpeg_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"},
            )
    return Response(status_code=204)


@app.get("/api/stream")
async def api_stream(since: int = Query(default=0, ge=0)):
    import asyncio

    session = _get_session()
    start_index = int(since)
    latest = session.get_latest()
    if start_index <= 0 and latest is not None:
        # Default SSE cursor: only new frames to avoid replaying history.
        start_index = int(latest.frame_index)

    async def _event_generator():
        cursor = start_index
        while True:
            frames = session.get_since(cursor, 20)
            for frame in frames:
                cursor = max(cursor, int(frame.frame_index))
                payload = json.dumps(_frame_to_json(frame), default=str)
                yield f"data: {payload}\n\n"
            if await _client_disconnected():
                break
            await asyncio.sleep(0.25)

    async def _client_disconnected() -> bool:
        return False

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve the static UI at `/` after API routes so `/api/*` keeps precedence.
if STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    def _static_placeholder() -> JSONResponse:
        return JSONResponse(
            {"ok": True, "message": "frontend static bundle not present"}
        )


__all__ = ["app"]
