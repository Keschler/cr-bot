from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..services.paths import REPO_ROOT, _resolve_against_repo

router = APIRouter()


@router.get("/api/roi-preview")
def api_roi_preview(
    path: str = Query(default=""),
    frame: int | None = Query(default=None),
    overlay: int = Query(default=1),
) -> dict[str, Any]:
    """Preview adapted ROIs for one video frame.

    ``overlay=0`` returns the raw probe frame JPEG so interactive clients
    can draw (and edit) the proposed boxes themselves.
    """
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
        roi_entries, overlay_jpeg, warnings, (nw, nh) = adapt_rois_for_probe(
            native, draw_overlay=(overlay != 0)
        )
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
