"""Capture-clock alignment while preserving video time, match time, and frame index."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping

from .devices import CaptureManifest, Frame
from .schema import PhysicalLabError


class SynchronizationError(PhysicalLabError):
    """Raised when a capture cannot meet a measurement's timing contract."""


@dataclass(frozen=True, slots=True)
class SyncMarker:
    """A visible/common edge observed by one device capture."""

    marker_id: str
    device_id: str
    workstation_monotonic_us: int
    frame_index: int | None = None
    device_time_us: int | None = None
    uncertainty_us: int = 0

    def __post_init__(self) -> None:
        if not self.marker_id or not self.device_id:
            raise SynchronizationError("sync marker IDs are required")
        for name in ("workstation_monotonic_us", "uncertainty_us"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SynchronizationError(f"marker.{name} must be non-negative")
        for name in ("frame_index", "device_time_us"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise SynchronizationError(f"marker.{name} must be non-negative when present")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "marker_id": self.marker_id,
            "device_id": self.device_id,
            "workstation_monotonic_us": self.workstation_monotonic_us,
            "uncertainty_us": self.uncertainty_us,
        }
        if self.frame_index is not None:
            result["frame_index"] = self.frame_index
        if self.device_time_us is not None:
            result["device_time_us"] = self.device_time_us
        return result


@dataclass(frozen=True, slots=True)
class DeviceClockAlignment:
    device_id: str
    offset_us: int
    uncertainty_us: int
    marker_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "offset_us": self.offset_us,
            "uncertainty_us": self.uncertainty_us,
            "marker_count": self.marker_count,
        }


@dataclass(frozen=True, slots=True)
class SynchronizationResult:
    """Immutable result of aligning both streams on the workstation clock."""

    reference_device: str
    alignments: tuple[DeviceClockAlignment, ...]
    common_marker_count: int
    uncertainty_us: int
    accepted: bool
    declared_tolerance_us: int
    rejection_reasons: tuple[str, ...] = ()

    def alignment_for(self, device_id: str) -> DeviceClockAlignment:
        for alignment in self.alignments:
            if alignment.device_id == device_id:
                return alignment
        raise SynchronizationError(f"no clock alignment for device {device_id!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_device": self.reference_device,
            "alignments": [item.to_dict() for item in self.alignments],
            "common_marker_count": self.common_marker_count,
            "uncertainty_us": self.uncertainty_us,
            "accepted": self.accepted,
            "declared_tolerance_us": self.declared_tolerance_us,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _median_int(values: Iterable[int]) -> int:
    values = tuple(values)
    if not values:
        raise SynchronizationError("at least one synchronization sample is required")
    return int(round(float(median(values))))


def _residual_uncertainty(values: Iterable[int], center: int, marker_uncertainty: int = 0) -> int:
    values = tuple(values)
    return max([abs(value - center) for value in values] + [marker_uncertainty])


def estimate_clock_alignment(
    markers: Iterable[SyncMarker],
    *,
    device_ids: Iterable[str] = ("A", "B"),
    reference_device: str = "A",
    declared_tolerance_us: int = 10_000,
) -> SynchronizationResult:
    """Estimate per-device clock offsets from common visible markers.

    Workstation monotonic timestamps are the primary alignment clock.  When a
    device timestamp is available, ``workstation - device`` is also retained
    as a transport offset.  Multiple common markers are required for a timing
    gate; a single marker is accepted only when its explicitly declared
    uncertainty is already within the threshold.
    """

    if type(declared_tolerance_us) is not int or declared_tolerance_us < 0:
        raise SynchronizationError("declared_tolerance_us must be non-negative")
    device_ids = tuple(dict.fromkeys(str(device_id) for device_id in device_ids))
    if reference_device not in device_ids:
        raise SynchronizationError("reference_device must be one of device_ids")
    grouped: dict[str, list[SyncMarker]] = {device_id: [] for device_id in device_ids}
    for marker in markers:
        if marker.device_id in grouped:
            grouped[marker.device_id].append(marker)
    if any(not grouped[device_id] for device_id in device_ids):
        missing = [device_id for device_id in device_ids if not grouped[device_id]]
        reasons = (f"missing synchronization markers for {','.join(missing)}",)
        alignments = tuple(
            DeviceClockAlignment(device_id, 0, declared_tolerance_us + 1, 0)
            for device_id in device_ids
        )
        return SynchronizationResult(
            reference_device=reference_device,
            alignments=alignments,
            common_marker_count=0,
            uncertainty_us=declared_tolerance_us + 1,
            accepted=False,
            declared_tolerance_us=declared_tolerance_us,
            rejection_reasons=reasons,
        )

    by_id: dict[str, dict[str, SyncMarker]] = {
        device_id: {marker.marker_id: marker for marker in grouped[device_id]}
        for device_id in device_ids
    }
    common_ids = set(by_id[reference_device])
    for device_id in device_ids:
        common_ids &= set(by_id[device_id])
    common_marker_count = len(common_ids)
    reasons: list[str] = []
    if common_marker_count == 0:
        reasons.append("captures do not share a visible synchronization marker")

    ref_markers = by_id[reference_device]
    alignments: list[DeviceClockAlignment] = []
    all_uncertainties: list[int] = []
    for device_id in device_ids:
        if device_id == reference_device:
            offsets = [0]
            residuals = [marker.uncertainty_us for marker in grouped[device_id]]
            count = len(grouped[device_id])
        else:
            offsets = []
            residuals = []
            for marker_id in sorted(common_ids):
                target = by_id[device_id][marker_id]
                reference = ref_markers[marker_id]
                offsets.append(target.workstation_monotonic_us - reference.workstation_monotonic_us)
                residuals.append(target.uncertainty_us + reference.uncertainty_us)
            if not offsets:
                # Keep a record for every device even when the pair is not
                # alignable; the oversized uncertainty makes rejection clear.
                offsets = [0]
                residuals = [declared_tolerance_us + 1]
            count = len(offsets)
        offset = _median_int(offsets)
        uncertainty = max(
            _residual_uncertainty(offsets, offset),
            max(residuals, default=declared_tolerance_us + 1),
        )
        alignment = DeviceClockAlignment(device_id, offset, uncertainty, count)
        alignments.append(alignment)
        all_uncertainties.append(uncertainty)

    uncertainty_us = max(all_uncertainties, default=declared_tolerance_us + 1)
    if uncertainty_us > declared_tolerance_us:
        reasons.append(
            f"synchronization uncertainty {uncertainty_us}us exceeds tolerance {declared_tolerance_us}us"
        )
    return SynchronizationResult(
        reference_device=reference_device,
        alignments=tuple(alignments),
        common_marker_count=common_marker_count,
        uncertainty_us=uncertainty_us,
        accepted=not reasons,
        declared_tolerance_us=declared_tolerance_us,
        rejection_reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class TimeMapping:
    """Explicit conversion for one stream; frame index remains an integer key."""

    device_id: str
    capture_start_workstation_us: int
    match_anchor_video_time_us: int = 0
    match_anchor_match_time_us: int = 0
    device_offset_us: int = 0

    def __post_init__(self) -> None:
        for name in (
            "capture_start_workstation_us",
            "match_anchor_video_time_us",
            "match_anchor_match_time_us",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SynchronizationError(f"{name} must be a non-negative integer")
        if type(self.device_offset_us) is not int:
            raise SynchronizationError("device_offset_us must be an integer")

    def video_time_from_workstation(self, workstation_monotonic_us: int) -> int:
        if type(workstation_monotonic_us) is not int:
            raise SynchronizationError("workstation time must be an integer")
        return workstation_monotonic_us - self.capture_start_workstation_us

    def video_time_from_frame(self, frame: Frame) -> int:
        if frame.source_device != self.device_id:
            raise SynchronizationError("frame source does not match time mapping")
        if frame.presentation_time_us is not None:
            return frame.presentation_time_us
        return self.video_time_from_workstation(frame.workstation_monotonic_us)

    def match_time_from_video(self, video_time_us: int) -> int:
        if type(video_time_us) is not int:
            raise SynchronizationError("video time must be an integer")
        return self.match_anchor_match_time_us + (video_time_us - self.match_anchor_video_time_us)

    def video_time_from_match(self, match_time_us: int) -> int:
        if type(match_time_us) is not int:
            raise SynchronizationError("match time must be an integer")
        return self.match_anchor_video_time_us + (match_time_us - self.match_anchor_match_time_us)

    def to_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "capture_start_workstation_us": self.capture_start_workstation_us,
            "match_anchor_video_time_us": self.match_anchor_video_time_us,
            "match_anchor_match_time_us": self.match_anchor_match_time_us,
            "device_offset_us": self.device_offset_us,
        }


def markers_from_captures(captures: Mapping[str, CaptureManifest]) -> tuple[SyncMarker, ...]:
    """Create conservative start-edge markers from capture manifests.

    This is enough for the software harness.  A physical run should add a
    visible countdown/sync-marker pair; the larger frame uncertainty remains
    visible and can fail a timing-sensitive measurement.
    """

    markers: list[SyncMarker] = []
    for device_id, capture in sorted(captures.items()):
        first_frame = capture.frames[0] if capture.frames else None
        markers.append(
            SyncMarker(
                marker_id="capture-start",
                device_id=device_id,
                workstation_monotonic_us=(
                    first_frame.workstation_monotonic_us
                    if first_frame is not None
                    else capture.started_at_monotonic_us
                ),
                frame_index=first_frame.frame_index if first_frame is not None else None,
                device_time_us=first_frame.device_time_us if first_frame is not None else None,
                uncertainty_us=(
                    first_frame.capture_uncertainty_us if first_frame is not None else 10_000
                ),
            )
        )
    return tuple(markers)


__all__ = [
    "DeviceClockAlignment",
    "SynchronizationError",
    "SynchronizationResult",
    "SyncMarker",
    "TimeMapping",
    "estimate_clock_alignment",
    "markers_from_captures",
]
