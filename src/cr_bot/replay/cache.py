from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
from pathlib import Path
import pickle

import cv2
import numpy as np

from cr_bot.domain.frame_analysis import FrameAnalysisResult


SCHEMA_VERSION = 1


@dataclass(slots=True)
class ReplayFrame:
    frame_idx: int
    video_time_s: float
    analysis: FrameAnalysisResult
    frame_png: bytes

    def decode_frame(self) -> np.ndarray:
        frame = cv2.imdecode(
            np.frombuffer(self.frame_png, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise ValueError(f"Could not decode replay frame {self.frame_idx}")
        return frame


class ReplayCacheWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "ReplayCacheWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.open(self.path, "wb")
        pickle.dump({"schema_version": SCHEMA_VERSION}, self._handle)
        return self

    def write(
        self,
        *,
        frame_idx: int,
        video_time_s: float,
        analysis: FrameAnalysisResult,
        frame: np.ndarray,
    ) -> None:
        if self._handle is None:
            raise RuntimeError("ReplayCacheWriter must be used as a context manager")
        ok, encoded = cv2.imencode(
            ".png",
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not ok:
            raise ValueError(f"Could not encode replay frame {frame_idx}")

        compact_analysis = replace(
            analysis,
            rendered=None,
            yolo_boxes=None,
            tower_hp_debug_steps={},
            timer_debug_steps={},
        )
        pickle.dump(
            ReplayFrame(
                frame_idx=frame_idx,
                video_time_s=video_time_s,
                analysis=compact_analysis,
                frame_png=encoded.tobytes(),
            ),
            self._handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is not None:
            self.path.unlink(missing_ok=True)


class ReplayCacheReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self):
        with gzip.open(self.path, "rb") as handle:
            header = pickle.load(handle)
            if header.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported replay cache schema in {self.path}: {header!r}"
                )
            while True:
                try:
                    yield pickle.load(handle)
                except EOFError:
                    return
