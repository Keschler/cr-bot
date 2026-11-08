from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from ..models.requests import LivePreviewMetricsRequest, LiveStartRequest, StopResponse
from ..runners.live import run_live_session
from ..services.checkpoints import _default_checkpoint
from ..services.paths import _resolve_against_repo
from ..services.devices import resolve_inference_devices
from ..services.live_diagnostics import LivePreviewClientMetrics
from .deps import _get_session, _start_worker, stop_current_session

router = APIRouter()
DEFAULT_LIVE_CALIBRATION = "simulator/physical_lab/calibrations/phone-a-candidate.json"


@router.get("/api/live/diagnostics")
def api_live_diagnostics() -> dict[str, Any]:
    session = _get_session()
    if session.live_diagnostics is None:
        return {"enabled": False, "enable_with": "CR_BOT_LIVE_DIAGNOSTICS=1"}
    return session.live_diagnostics.snapshot()


@router.post("/api/live/preview-metrics", status_code=204)
def api_live_preview_metrics(request: LivePreviewMetricsRequest) -> Response:
    session = _get_session()
    diagnostics = session.live_diagnostics
    if session.running and diagnostics is not None and request.session_id == diagnostics.session_id:
        diagnostics.record_client(LivePreviewClientMetrics(**request.model_dump(exclude={"session_id"})))
    return Response(status_code=204)


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
    checkpoint = _resolve_against_repo(request.checkpoint) or _default_checkpoint()
    if not checkpoint:
        raise HTTPException(
            status_code=400,
            detail="checkpoint is required; configure a V4 checkpoint or set CR_BOT_V4_CHECKPOINT",
        )
    if not Path(checkpoint).is_file():
        raise HTTPException(
            status_code=404, detail=f"checkpoint file does not exist: {checkpoint}"
        )
    calibration = _resolve_against_repo(
        (request.calibration or "").strip() or DEFAULT_LIVE_CALIBRATION
    )
    try:
        session = _start_worker(
            mode="live",
            target=run_live_session,
            kwargs={
                "serial": serial,
                "transport": request.transport,
                "stream_max_width_px": request.stream_max_width_px,
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
