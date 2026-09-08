from __future__ import annotations

import collections
import threading
from typing import Any

from .frames import FrontendFrame


class FrontendSession:
    """Thread-safe bounded store of recent :class:`FrontendFrame` objects."""

    def __init__(self, *, mode: str = "idle", maxlen: int = 512) -> None:
        self._lock = threading.Lock()
        self._frames: collections.deque[FrontendFrame] = collections.deque(
            maxlen=maxlen
        )
        self.latest: FrontendFrame | None = None
        self.running: bool = False
        self.mode: str = mode
        self.error: str | None = None
        self.summary: dict | None = None
        # Live policy actor owned by the worker thread (None when idle).
        # Re-evaluation borrows it under ``actor_lock`` with hidden-state
        # save/restore, so the running session is never disturbed.
        self.actor: Any | None = None
        self.actor_lock = threading.Lock()

    @property
    def history(self) -> list[FrontendFrame]:
        with self._lock:
            return list(self._frames)

    @property
    def maxlen(self) -> int | None:
        return self._frames.maxlen

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self.latest = None

    def push(self, frame: FrontendFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            self.latest = frame

    def get_latest(self) -> FrontendFrame | None:
        with self._lock:
            return self.latest

    def find_frame(self, frame_index: Any) -> FrontendFrame | None:
        """Return the retained frame with ``frame_index`` (None if evicted)."""
        try:
            wanted = int(frame_index)
        except (TypeError, ValueError):
            return None
        with self._lock:
            for frame in self._frames:
                try:
                    if int(frame.frame_index) == wanted:
                        return frame
                except (TypeError, ValueError):
                    continue
        return None

    def set_correction(self, frame_index: Any, correction: dict | None) -> bool:
        """Attach (or with None, clear) a what-if correction. Thread-safe."""
        frame = self.find_frame(frame_index)
        if frame is None:
            return False
        with self._lock:
            frame.corrected = dict(correction) if correction is not None else None
        return True

    def get_since(
        self, since: int = 0, limit: int | None = 50
    ) -> list[FrontendFrame]:
        """Return frames with ``frame_index`` strictly greater than ``since``.

        Results are chronological (oldest first).  When ``limit`` is set,
        at most ``limit`` frames are returned.
        """

        try:
            since_int = int(since)
        except (TypeError, ValueError):
            since_int = 0
        with self._lock:
            matched = [f for f in self._frames if int(f.frame_index) > since_int]
        if limit is not None:
            try:
                limit_int = int(limit)
            except (TypeError, ValueError):
                limit_int = 50
            if limit_int is not None and limit_int >= 0:
                matched = matched[:limit_int]
        return matched

    def to_status_dict(self) -> dict[str, Any]:
        with self._lock:
            latest = self.latest
            count = len(self._frames)
            running = self.running
            mode = self.mode
            error = self.error
            summary = dict(self.summary) if isinstance(self.summary, dict) else None
        if latest is not None:
            latest_info: dict[str, Any] | None = {
                "frame_index": latest.frame_index,
                "timestamp_s": latest.timestamp_s,
                "in_game": latest.in_game,
                "emitted": latest.emitted,
                "has_image": latest.jpeg_bytes is not None,
                "record": latest.record,
                "suggestions": latest.suggestions,
                "diagnostics": latest.diagnostics,
                "frame_width": latest.frame_width,
                "frame_height": latest.frame_height,
                "own_actions": latest.own_actions,
                "enemy_plays": latest.enemy_plays,
            }
            latest_index: int | None = latest.frame_index
        else:
            latest_info = None
            latest_index = None
        return {
            "running": running,
            "mode": mode,
            "error": error,
            "summary": summary,
            "frames": count,
            "frame_count": count,
            "latest_frame_index": latest_index,
            "latest": latest_info,
        }
