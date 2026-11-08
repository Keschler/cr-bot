from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Query, Request, Response

from .deps import _frame_to_json, _get_session

router = APIRouter()


@router.get("/api/frames")
def api_frames(
    since: int = Query(default=0, ge=-1),
    limit: int = Query(default=50, ge=1, le=512),
) -> dict[str, Any]:
    session = _get_session()
    frames = session.get_since(int(since), int(limit))
    return {
        "frames": [_frame_to_json(f) for f in frames],
        "count": len(frames),
        "since": int(since),
    }


@router.get("/api/frame/latest")
def api_frame_latest() -> Response:
    session = _get_session()
    latest = session.get_latest()
    if latest is None or latest.jpeg_bytes is None:
        return Response(status_code=204)
    return Response(
        content=latest.jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Frame-Index": str(latest.frame_index)},
    )


@router.get("/api/frame/preview")
def api_frame_preview(request: Request = None) -> Response:
    """Serve the latest live capture without waiting for analysis to finish."""
    session = _get_session()
    diagnostics = session.live_diagnostics
    if request is not None:
        request.scope["cr_bot.live_diagnostics"] = diagnostics
    started_s = time.monotonic() if diagnostics is not None else None
    preview = session.get_live_preview() if session.mode == "live" and session.running else None
    analyzed = session.get_latest()
    image_frame = preview if preview is not None else analyzed
    if image_frame is None or image_frame.jpeg_bytes is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    headers = {
        "Cache-Control": "no-store",
        "X-Frame-Index": str(image_frame.frame_index),
    }
    if session.mode == "live":
        now_s = time.monotonic()
        # Backwards-compatible stage ages: neither includes phone/ADB/codec
        # buffering before publication on this workstation.
        headers["X-Frame-Clock"] = session.live_frame_clock
        headers["X-Capture-Age-Ms"] = str(max(0.0, (now_s - image_frame.timestamp_s) * 1000.0))
        if analyzed is not None:
            headers["X-Analysis-Age-Ms"] = str(max(0.0, (now_s - analyzed.timestamp_s) * 1000.0))
    if diagnostics is not None:
        diagnostics.record_preview(
            frame_index=int(image_frame.frame_index),
            handler_ms=(time.monotonic() - started_s) * 1000.0,
            size_bytes=len(image_frame.jpeg_bytes),
        )
        headers["X-Live-Diagnostics"] = diagnostics.session_id
    return Response(content=image_frame.jpeg_bytes, media_type="image/jpeg", headers=headers)


@router.get("/api/frame/{frame_index}")
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
