"""Fail-closed observation boundaries for autonomous physical experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Protocol, Sequence

from .devices import Frame
from .schema import PhysicalLabError


HOG_CROSSES_Y_EVENT = "hog_crosses_y_mtile"
HOG_BRIDGE_BOUNDARY_Y_MTILE = 17_000
DEFAULT_BRIDGE_X_RANGES_MTILE = ((2_000, 5_000), (13_000, 16_000))


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One normalized detector observation on the workstation clock."""

    track_id: int
    card_id: str
    team: str
    confidence: float
    x_mtile: int
    y_mtile: int
    workstation_monotonic_us: int

    def __post_init__(self) -> None:
        if type(self.track_id) is not int or self.track_id < 0:
            raise PhysicalLabError("track observation requires a non-negative integer track_id")
        if not self.card_id or not self.team:
            raise PhysicalLabError("track observation card and team are required")
        if type(self.confidence) not in (int, float) or not math.isfinite(float(self.confidence)):
            raise PhysicalLabError("track observation confidence must be finite")
        if not 0 <= float(self.confidence) <= 1:
            raise PhysicalLabError("track observation confidence must be between zero and one")
        for field_name in ("x_mtile", "y_mtile", "workstation_monotonic_us"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise PhysicalLabError(f"track observation {field_name} must be non-negative")


class FrameRecorder(Protocol):
    def record_frame(self, frame: Frame) -> None: ...


class LiveTrackObservationSource:
    """Serialize screenshot, capture recording, and lazy detector inference."""

    def __init__(
        self,
        screenshot: Callable[[], Frame],
        *,
        recorder: FrameRecorder | None = None,
        analyze_frame: Callable[[Frame], Sequence[TrackObservation]] | None = None,
        detector_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.screenshot = screenshot
        self.recorder = recorder
        self._analyze_frame_override = analyze_frame
        self._detector_factory = detector_factory
        self._detector: Any | None = None

    def _analyze_frame(self, frame: Frame) -> tuple[TrackObservation, ...]:
        if frame.payload is None:
            raise PhysicalLabError("live Hog observation frame has no PNG payload")
        # Heavy CV/runtime imports remain behind the first live observation.
        import cv2
        import numpy as np

        from cr_bot.app.pipeline import normalize_frame, process_frame
        from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD
        from cr_bot.vision.yolo_runtime import build_detector
        from simulator.mining import _detection_world_position

        encoded = np.frombuffer(frame.payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise PhysicalLabError("live Hog observation frame is not a decodable PNG")
        image = normalize_frame(image)
        if self._detector is None:
            factory = self._detector_factory or build_detector
            self._detector = factory()
        try:
            analysis = process_frame(image, self._detector)
        except (RuntimeError, ValueError, OSError) as error:
            raise PhysicalLabError(f"live Hog detector inference failed: {error}") from error

        observations: list[TrackObservation] = []
        for match in analysis.matches:
            detection = match.troop
            card_id = DIRECT_UNIT_TO_CARD.get(detection.class_name)
            if card_id is None or detection.track_id is None:
                continue
            x_mtile, y_mtile = _detection_world_position(
                detection,
                analysis.arena_px,
                ground_anchor=True,
            )
            observations.append(
                TrackObservation(
                    track_id=int(detection.track_id),
                    card_id=card_id,
                    team=str(detection.team),
                    confidence=float(detection.confidence),
                    x_mtile=x_mtile,
                    y_mtile=y_mtile,
                    workstation_monotonic_us=frame.workstation_monotonic_us,
                )
            )
        return tuple(observations)

    def __call__(self) -> tuple[TrackObservation, ...]:
        frame = self.screenshot()
        if self.recorder is not None:
            self.recorder.record_frame(frame)
        analyze = self._analyze_frame_override or self._analyze_frame
        return tuple(analyze(frame))


class HogBridgeObservationWaiter:
    """Confirm an ally Hog entering the river through a reviewed bridge.

    The returned integer is the workstation monotonic timestamp of the first
    at/through-boundary sample.  It is deliberately not labelled match time;
    callers must use a separately reviewed clock mapping for direct timing.
    """

    def __init__(
        self,
        observe: Callable[[], Sequence[TrackObservation]],
        *,
        confidence_threshold: float = 0.98,
        bridge_x_ranges_mtile: Sequence[tuple[int, int]] = DEFAULT_BRIDGE_X_RANGES_MTILE,
        confirmation_samples: int = 3,
        monotonic_clock_us: Callable[[], int] = lambda: time.monotonic_ns() // 1_000,
        poll: Callable[[], None] = lambda: time.sleep(0.05),
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise PhysicalLabError("Hog bridge confidence threshold must be in (0, 1]")
        if type(confirmation_samples) is not int or confirmation_samples < 1:
            raise PhysicalLabError("Hog bridge confirmation_samples must be positive")
        ranges = tuple((int(start), int(end)) for start, end in bridge_x_ranges_mtile)
        if not ranges or any(start < 0 or end < start for start, end in ranges):
            raise PhysicalLabError("Hog bridge x ranges are invalid")
        self.observe = observe
        self.confidence_threshold = float(confidence_threshold)
        self.bridge_x_ranges_mtile = ranges
        self.confirmation_samples = confirmation_samples
        self.monotonic_clock_us = monotonic_clock_us
        self.poll = poll

    def _inside_bridge(self, x_mtile: int) -> bool:
        return any(start <= x_mtile <= end for start, end in self.bridge_x_ranges_mtile)

    def __call__(self, event: str, value: int, timeout_us: int) -> int:
        if event != HOG_CROSSES_Y_EVENT or value != HOG_BRIDGE_BOUNDARY_Y_MTILE:
            raise PhysicalLabError(
                "Hog bridge waiter only supports hog_crosses_y_mtile at y=17000"
            )
        if type(timeout_us) is not int or timeout_us <= 0:
            raise PhysicalLabError("Hog bridge waiter timeout_us must be positive")

        started = self.monotonic_clock_us()
        selected_track_id: int | None = None
        saw_pre_boundary = False
        crossing_samples: list[TrackObservation] = []
        while self.monotonic_clock_us() - started <= timeout_us:
            rows = tuple(self.observe())
            if self.monotonic_clock_us() - started > timeout_us:
                raise TimeoutError("timed out waiting for a verified Hog bridge crossing")
            candidates = tuple(
                row
                for row in rows
                if row.card_id == "hog-rider"
                and row.team == "ally"
                and row.confidence >= self.confidence_threshold
            )
            if len(candidates) > 1:
                raise PhysicalLabError("Hog bridge observation is ambiguous")
            if not candidates:
                self.poll()
                continue
            current = candidates[0]
            if selected_track_id is None:
                selected_track_id = current.track_id
            elif current.track_id != selected_track_id:
                raise PhysicalLabError("Hog bridge observation switched track identity")

            if current.y_mtile > value:
                if crossing_samples:
                    raise PhysicalLabError("Hog bridge trajectory reversed after crossing")
                saw_pre_boundary = True
                self.poll()
                continue
            if not saw_pre_boundary:
                raise PhysicalLabError("Hog bridge crossing lacks a pre-boundary sample")
            if not self._inside_bridge(current.x_mtile):
                raise PhysicalLabError("Hog crossed the river boundary outside a reviewed bridge")
            if crossing_samples:
                previous = crossing_samples[-1]
                if current.workstation_monotonic_us <= previous.workstation_monotonic_us:
                    raise PhysicalLabError("Hog bridge timestamps are not strictly monotonic")
                if current.y_mtile > previous.y_mtile:
                    raise PhysicalLabError("Hog bridge trajectory is not monotonic toward the opponent")
            crossing_samples.append(current)
            if len(crossing_samples) >= self.confirmation_samples:
                return crossing_samples[0].workstation_monotonic_us
            self.poll()
        raise TimeoutError("timed out waiting for a verified Hog bridge crossing")


__all__ = [
    "DEFAULT_BRIDGE_X_RANGES_MTILE",
    "HOG_BRIDGE_BOUNDARY_Y_MTILE",
    "HOG_CROSSES_Y_EVENT",
    "HogBridgeObservationWaiter",
    "LiveTrackObservationSource",
    "TrackObservation",
]
