from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

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
    return Response(content=latest.jpeg_bytes, media_type="image/jpeg")


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
