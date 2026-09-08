from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..models.requests import VideoStartRequest
from ..runners.video import run_video_session
from ..services.checkpoints import _default_checkpoint
from ..services.paths import REPO_ROOT, UPLOAD_DIR
from ..services.paths import _resolve_against_repo
from ..services.devices import resolve_inference_devices
from ..services.uploads import _sanitize_upload_filename
from .deps import _start_worker

router = APIRouter()


@router.post("/api/upload")
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


@router.post("/api/video/start")
def api_video_start(request: VideoStartRequest) -> dict[str, Any]:
    video_path = _resolve_against_repo(request.video_path)
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path is required")
    try:
        resolve_inference_devices(request.device)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
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
                "device": request.device or "auto",
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


@router.get("/api/video/info")
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
