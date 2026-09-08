from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .deps import _get_session
from ..services.checkpoints import _list_checkpoints

router = APIRouter()


@router.get("/api/health")
def api_health() -> dict[str, Any]:
    from ..services.devices import INFERENCE_DEVICES, _cuda_available

    cuda = _cuda_available()
    return {"ok": True, "cuda_available": cuda, "inference_devices": list(INFERENCE_DEVICES)}


@router.get("/api/status")
def api_status() -> dict[str, Any]:
    return _get_session().to_status_dict()


@router.get("/api/checkpoints")
def api_checkpoints() -> dict[str, Any]:
    items, default = _list_checkpoints()
    return {"checkpoints": items, "default": default}
