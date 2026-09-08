from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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
    checkpoint: str | None = None
    device: str = "auto"
    calibration: str | None = None
    execute: bool = False
    confirm_live: bool = False


class StopResponse(BaseModel):
    stopped: bool = True
    running: bool = False


class ReevaluateRequest(BaseModel):
    updates: Any = None
    deletes: Any = None
    adds: Any = None
