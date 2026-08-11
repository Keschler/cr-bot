from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Iterable

import cv2
import numpy as np

from cr_bot.domain.rois import ROIS


ENEMY_ARENA_ROI = (30, 150, 1010, 1644)


@dataclass(frozen=True)
class TeamMarker:
    frame_index: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int


@dataclass
class MarkerTrack:
    track_id: int
    observations: list[TeamMarker] = field(default_factory=list)

    @property
    def first(self) -> TeamMarker:
        return self.observations[0]

    @property
    def last(self) -> TeamMarker:
        return self.observations[-1]


@dataclass(frozen=True)
class MarkerBurst:
    burst_id: int
    start_frame: int
    end_frame_exclusive: int
    tracks: tuple[MarkerTrack, ...]


def marker_burst_candidate_frame(
    burst: MarkerBurst,
    *,
    segment_start_frame: int,
) -> int:
    """Estimate onset without letting one early effect dominate a burst.

    Multi-component deployment effects can start several frames before the
    unit's own level badge. The median track start is robust to that early
    component, while the one-frame backoff accounts for the badge lag.
    """

    median_start = statistics.median_low(
        track.first.frame_index for track in burst.tracks
    )
    return max(segment_start_frame, int(median_start) - 1)


def marker_review_frame_indices(
    *,
    anchor_frame: int,
    marker_end_frame_exclusive: int,
    segment_start_frame: int,
    segment_end_frame_exclusive: int,
) -> list[int]:
    """Sample compact before/after evidence with a two-second future horizon."""
    offsets = (-4, -2, -1, 0, 2, 5, 10, 15, 20)
    frames = {
        anchor_frame + offset
        for offset in offsets
        if (
            segment_start_frame
            <= anchor_frame + offset
            < segment_end_frame_exclusive
        )
    }
    last_marker_frame = marker_end_frame_exclusive - 1
    if segment_start_frame <= last_marker_frame < segment_end_frame_exclusive:
        frames.add(last_marker_frame)
    return sorted(frames)


def detect_enemy_team_markers(
    frame: np.ndarray,
    *,
    frame_index: int,
) -> list[TeamMarker]:
    """Detect compact red level-badge components, without identifying a card."""
    x0, y0, width, height = ENEMY_ARENA_ROI
    view = frame[y0 : y0 + height, x0 : x0 + width]
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    red_high = cv2.inRange(hsv, (160, 105, 75), (179, 255, 255))
    red_low = cv2.inRange(hsv, (0, 145, 100), (6, 255, 255))
    mask = cv2.bitwise_or(red_high, red_low)
    _mask_fixed_tower_health(mask, offset=(x0, y0))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    clock_centers = _clock_like_centers(hsv, offset=(x0, y0))
    markers: list[TeamMarker] = []
    for label in range(1, count):
        left, top, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if not (
            45 <= area <= 2200
            and 15 <= component_width <= 45
            and 15 <= component_height <= 50
        ):
            continue
        fill = area / max(1, component_width * component_height)
        if fill < 0.18:
            continue
        center_x, center_y = centroids[label]
        markers.append(
            TeamMarker(
                frame_index=frame_index,
                bbox=(
                    left + x0,
                    top + y0,
                    component_width,
                    component_height,
                ),
                center=(float(center_x + x0), float(center_y + y0)),
                area=area,
            )
        )
    return [
        marker
        for marker in markers
        if any(
            abs(center_x - marker.center[0]) <= 130
            and -55 <= center_y - marker.center[1] <= 155
            and math.dist((center_x, center_y), marker.center) <= 145
            for center_x, center_y in clock_centers
        )
    ]


def track_enemy_team_markers(
    observations: Iterable[tuple[int, list[TeamMarker]]],
    *,
    max_gap_frames: int = 6,
    max_distance_px: float = 70.0,
) -> list[MarkerTrack]:
    tracks: list[MarkerTrack] = []
    next_track_id = 0
    for frame_index, markers in observations:
        active = [
            track
            for track in tracks
            if 0 < frame_index - track.last.frame_index <= max_gap_frames + 1
        ]
        unmatched_tracks = set(track.track_id for track in active)
        unmatched_markers = set(range(len(markers)))
        pairs: list[tuple[float, int, int]] = []
        for track in active:
            gap = frame_index - track.last.frame_index
            threshold = max_distance_px + 25.0 * max(0, gap - 1)
            for marker_index, marker in enumerate(markers):
                distance = math.dist(track.last.center, marker.center)
                if distance <= threshold:
                    pairs.append((distance, track.track_id, marker_index))
        by_id = {track.track_id: track for track in tracks}
        for _, track_id, marker_index in sorted(pairs):
            if track_id not in unmatched_tracks or marker_index not in unmatched_markers:
                continue
            by_id[track_id].observations.append(markers[marker_index])
            unmatched_tracks.remove(track_id)
            unmatched_markers.remove(marker_index)
        for marker_index in sorted(unmatched_markers):
            tracks.append(
                MarkerTrack(
                    track_id=next_track_id,
                    observations=[markers[marker_index]],
                )
            )
            next_track_id += 1
    return tracks


def deployment_candidate_tracks(
    tracks: Iterable[MarkerTrack],
    *,
    segment_start_frame: int,
    persistence_observations: int = 3,
    persistence_window_frames: int = 5,
    enemy_deployment_max_y: float = 1070.0,
) -> list[MarkerTrack]:
    candidates = []
    for track in tracks:
        first = track.first
        if first.frame_index <= segment_start_frame:
            continue
        if first.center[1] > enemy_deployment_max_y:
            continue
        if _is_fixed_ui_marker(first.center):
            continue
        persistent = sum(
            observation.frame_index
            <= first.frame_index + persistence_window_frames - 1
            for observation in track.observations
        )
        if persistent < persistence_observations:
            continue
        candidates.append(track)
    return candidates


def group_candidate_tracks(
    tracks: Iterable[MarkerTrack],
    *,
    max_burst_span_frames: int = 6,
    max_center_distance_px: float = 260.0,
) -> list[MarkerBurst]:
    ordered = sorted(tracks, key=lambda track: track.first.frame_index)
    groups: list[list[MarkerTrack]] = []
    for track in ordered:
        matching_group = None
        for group in reversed(groups):
            frame_delta = (
                track.first.frame_index - group[0].first.frame_index
            )
            if frame_delta > max_burst_span_frames:
                break
            if any(
                math.dist(track.first.center, member.first.center)
                <= max_center_distance_px
                for member in group
            ):
                matching_group = group
                break
        if matching_group is None:
            groups.append([track])
        else:
            matching_group.append(track)
    return [
        MarkerBurst(
            burst_id=index,
            start_frame=min(track.first.frame_index for track in group),
            end_frame_exclusive=max(
                track.first.frame_index for track in group
            )
            + 1,
            tracks=tuple(group),
        )
        for index, group in enumerate(groups)
    ]


def _is_fixed_ui_marker(center: tuple[float, float]) -> bool:
    """Suppress recurring red artwork around fixed arena/UI structures."""
    x, y = center
    return (
        (90 <= x <= 135 and 285 <= y <= 335)
        or (485 <= x <= 535 and 280 <= y <= 335)
        or (930 <= x <= 1040 and 145 <= y <= 195)
        or (930 <= x <= 1040 and 280 <= y <= 335)
    )


def _mask_fixed_tower_health(
    mask: np.ndarray,
    *,
    offset: tuple[int, int],
) -> None:
    offset_x, offset_y = offset
    for name in (
        "opponent_king_health_bar",
        "opponent_left_support_health_bar",
        "opponent_right_support_health_bar",
        "opponent_king_health_text",
        "opponent_left_support_health_text",
        "opponent_right_support_health_text",
    ):
        x, y, width, height = ROIS[name]
        left = max(0, x - offset_x - 8)
        top = max(0, y - offset_y - 8)
        right = min(mask.shape[1], x + width - offset_x + 8)
        bottom = min(mask.shape[0], y + height - offset_y + 8)
        if left < right and top < bottom:
            mask[top:bottom, left:right] = 0


def _clock_like_centers(
    hsv: np.ndarray,
    *,
    offset: tuple[int, int],
) -> list[tuple[float, float]]:
    white = cv2.inRange(hsv, (0, 0, 170), (179, 85, 255))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(white)
    offset_x, offset_y = offset
    centers = []
    for label in range(1, count):
        left, top, width, height, area = (
            int(value) for value in stats[label]
        )
        if not (
            450 <= area <= 2600
            and 25 <= width <= 95
            and 24 <= height <= 95
        ):
            continue
        fill = area / max(1, width * height)
        if not 0.3 <= fill <= 0.82:
            continue
        center_x, center_y = centroids[label]
        centers.append(
            (float(center_x + offset_x), float(center_y + offset_y))
        )
    return centers
