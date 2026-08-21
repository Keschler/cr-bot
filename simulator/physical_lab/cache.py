"""Sealing and conservative recognition of existing replay-cache artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .artifacts import hash_file
from .schema import PhysicalLabError


class ReplayCacheError(PhysicalLabError):
    """Raised when a replay cache is missing, opaque, or incomplete."""


@dataclass(frozen=True, slots=True)
class ReplayCacheSeal:
    path: str
    sha256: str
    frame_count: int
    first_frame_index: int
    last_frame_index: int
    recognized: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "frame_count": self.frame_count,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "recognized": self.recognized,
        }


def seal_replay_cache(path: str | Path, *, minimum_frames: int = 1) -> ReplayCacheSeal:
    """Validate the repository replay-cache reader boundary and hash the file.

    The cache is an observation artifact, not truth.  The reader is imported
    lazily because it depends on OpenCV/NumPy and those dependencies must not
    load for core simulator callers.
    """

    source = Path(path)
    if not source.is_file():
        raise ReplayCacheError(f"replay cache does not exist: {source}")
    if type(minimum_frames) is not int or minimum_frames <= 0:
        raise ReplayCacheError("minimum_frames must be positive")
    try:
        from cr_bot.replay.cache import ReplayCacheReader

        rows = ReplayCacheReader(source)
        frame_indices: list[int] = []
        for row in rows:
            frame_index = getattr(row, "frame_idx", None)
            video_time_s = getattr(row, "video_time_s", None)
            frame_png = getattr(row, "frame_png", None)
            if type(frame_index) is not int or frame_index < 0:
                raise ReplayCacheError("replay frame index is invalid")
            if type(video_time_s) not in (int, float) or not math.isfinite(float(video_time_s)):
                raise ReplayCacheError("replay frame video time is invalid")
            if not isinstance(frame_png, bytes) or not frame_png:
                raise ReplayCacheError("replay frame has no encoded image")
            frame_indices.append(frame_index)
        if len(frame_indices) < minimum_frames:
            raise ReplayCacheError(
                f"replay cache contains {len(frame_indices)} frames; {minimum_frames} required"
            )
        if frame_indices != sorted(set(frame_indices)):
            raise ReplayCacheError("replay cache frame indices are not sorted and unique")
    except ReplayCacheError:
        raise
    except Exception as error:
        raise ReplayCacheError(f"replay cache is not recognized by the existing reader: {error}") from error
    return ReplayCacheSeal(
        path=str(source),
        sha256=hash_file(source),
        frame_count=len(frame_indices),
        first_frame_index=frame_indices[0],
        last_frame_index=frame_indices[-1],
        recognized=True,
    )


__all__ = ["ReplayCacheError", "ReplayCacheSeal", "seal_replay_cache"]
