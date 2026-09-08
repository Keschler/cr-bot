from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .deps import _frame_to_json, _get_session

router = APIRouter()


@router.get("/api/stream")
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
