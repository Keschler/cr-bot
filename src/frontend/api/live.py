from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import LiveStartRequest, StopResponse
from ..runners.live import run_live_session
from ..services.checkpoints import _default_checkpoint
from ..services.paths import _resolve_against_repo
from ..services.devices import resolve_inference_devices
from .deps import _start_worker, stop_current_session

router = APIRouter()


@router.post("/api/live/start")
def api_live_start(request: LiveStartRequest) -> dict[str, Any]:
    serial = (request.serial or "").strip()
    if not serial:
        raise HTTPException(status_code=400, detail="serial must be non-empty")
    try:
        resolve_inference_devices(request.device)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
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
                "device": request.device or "auto",
                "calibration": calibration,
                "execute": bool(request.execute),
                "confirm_live": bool(request.confirm_live),
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return session.to_status_dict()


@router.post("/api/stop", response_model=StopResponse)
def api_stop() -> StopResponse:
    session = stop_current_session()
    _ = session
    return StopResponse(stopped=True, running=False)
