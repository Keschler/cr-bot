from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VideoStartRequest(BaseModel):
    video_path: str
    frame_stride: int = 1
    start_frame: int = 0
    max_frames: int | None = None
    checkpoint: str | None = None
    device: str = "auto"
    yolo_image_size: int = 896
    # Any + manual 400 checks so wrong JSON types map to 400 (not 422).
    adapt_rois: Any = False
    roi_set: Any = None


class LiveStartRequest(BaseModel):
    serial: str
    transport: str = "stream"
    stream_max_width_px: int | None = Field(default=None, gt=0, strict=True)
    checkpoint: str | None = None
    device: str = "auto"
    calibration: str | None = None
    execute: bool = False
    confirm_live: bool = False


class LivePreviewMetricsRequest(BaseModel):
    model_config = {"allow_inf_nan": False}

    session_id: str = Field(min_length=1, max_length=64)
    window_s: float = Field(gt=0)
    displayed_frames: int = Field(ge=0)
    changed_frames: int = Field(ge=0)
    last_frame_index: int = Field(ge=0)
    response_headers_ms: float = Field(ge=0)
    body_ms: float = Field(ge=0)
    decode_ms: float = Field(ge=0)
    request_to_display_ms: float = Field(ge=0)


class StopResponse(BaseModel):
    stopped: bool = True
    running: bool = False


class ReevaluateRequest(BaseModel):
    updates: Any = None
    deletes: Any = None
    adds: Any = None
