from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol

from ..imaging import encode_preview_jpeg
from ..models.frames import LivePreviewFrame


class CapturedFrame(Protocol):
    image: Any
    frame_index: int
    timestamp_s: float


class LivePreviewProvider:
    """Encode only the newest capture, caching repeat requests by frame index."""

    def __init__(
        self,
        latest_frame: Callable[[], CapturedFrame | None],
        *,
        normalize_frame: Callable[[Any], Any] | None = None,
    ) -> None:
        self._latest_frame = latest_frame
        self._normalize_frame = normalize_frame
        self._cached: LivePreviewFrame | None = None
        self._lock = threading.Lock()

    def __call__(self) -> LivePreviewFrame | None:
        with self._lock:
            frame = self._latest_frame()
            if frame is None:
                return None
            if self._cached is not None and self._cached.frame_index == frame.frame_index:
                return self._cached
            image = frame.image
            if self._normalize_frame is not None:
                image = self._normalize_frame(image)
            jpeg = encode_preview_jpeg(image)
            if jpeg is None:
                return None
            self._cached = LivePreviewFrame(
                frame_index=int(frame.frame_index),
                timestamp_s=float(frame.timestamp_s),
                jpeg_bytes=jpeg,
            )
            return self._cached
