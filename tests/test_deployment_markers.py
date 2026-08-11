from __future__ import annotations

import cv2
import numpy as np

from cr_bot.vision.deployment_markers import (
    MarkerBurst,
    MarkerTrack,
    marker_review_frame_indices,
    TeamMarker,
    deployment_candidate_tracks,
    detect_enemy_team_markers,
    group_candidate_tracks,
    marker_burst_candidate_frame,
    track_enemy_team_markers,
)


def test_marker_review_frames_are_sparse_and_extend_two_seconds():
    assert marker_review_frame_indices(
        anchor_frame=100,
        marker_end_frame_exclusive=104,
        segment_start_frame=0,
        segment_end_frame_exclusive=200,
    ) == [96, 98, 99, 100, 102, 103, 105, 110, 115, 120]


def test_marker_review_frames_are_clamped_to_segment():
    assert marker_review_frame_indices(
        anchor_frame=2,
        marker_end_frame_exclusive=3,
        segment_start_frame=0,
        segment_end_frame_exclusive=12,
    ) == [0, 1, 2, 4, 7]


def _marker_frame(*, include_clock: bool = True) -> np.ndarray:
    frame = np.zeros((1800, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (280, 480), (307, 508), (0, 0, 255), -1)
    if include_clock:
        cv2.circle(frame, (294, 570), 19, (255, 255, 255), -1)
    return frame


def test_enemy_marker_requires_nearby_clock_like_component():
    with_clock = detect_enemy_team_markers(
        _marker_frame(),
        frame_index=10,
    )
    without_clock = detect_enemy_team_markers(
        _marker_frame(include_clock=False),
        frame_index=10,
    )
    assert len(with_clock) == 1
    assert with_clock[0].frame_index == 10
    assert without_clock == []


def test_marker_tracks_require_persistence_and_split_distant_bursts():
    observations = [
        (
            frame,
            [
                TeamMarker(
                    frame_index=frame,
                    bbox=(280 + frame, 480, 28, 29),
                    center=(294.0 + frame, 494.0),
                    area=700,
                )
            ],
        )
        for frame in (1, 2, 3)
    ]
    tracks = track_enemy_team_markers(observations)
    candidates = deployment_candidate_tracks(
        tracks,
        segment_start_frame=0,
    )
    assert len(candidates) == 1

    second = track_enemy_team_markers(
        [
            (
                frame,
                [
                    TeamMarker(
                        frame_index=frame,
                        bbox=(600, 500, 25, 25),
                        center=(612.0, 512.0),
                        area=600,
                    )
                ],
            )
            for frame in (5, 6, 7)
        ]
    )[0]
    bursts = group_candidate_tracks([candidates[0], second])
    assert len(bursts) == 2
    assert bursts[0].start_frame == 1

    nearby = track_enemy_team_markers(
        [
            (
                frame,
                [
                    TeamMarker(
                        frame_index=frame,
                        bbox=(400, 500, 25, 25),
                        center=(412.0, 512.0),
                        area=600,
                    )
                ],
            )
            for frame in (5, 6, 7)
        ]
    )[0]
    assert len(group_candidate_tracks([candidates[0], nearby])) == 1


def test_marker_track_survives_short_detection_gap():
    observations = [
        (
            frame,
            [
                TeamMarker(
                    frame_index=frame,
                    bbox=(200 + frame, 500, 25, 25),
                    center=(212.0 + frame, 512.0),
                    area=600,
                )
            ],
        )
        for frame in (1, 2, 3, 8, 9, 10)
    ]
    tracks = track_enemy_team_markers(observations)
    assert len(tracks) == 1
    assert [row.frame_index for row in tracks[0].observations] == [
        1,
        2,
        3,
        8,
        9,
        10,
    ]


def test_burst_candidate_uses_median_track_start_with_one_frame_backoff():
    tracks = []
    for track_id, start in enumerate((10, 12, 12, 16, 16)):
        tracks.append(
            MarkerTrack(
                track_id=track_id,
                observations=[
                    TeamMarker(
                        frame_index=start,
                        bbox=(200, 900, 20, 20),
                        center=(210.0, 910.0),
                        area=300,
                    )
                ],
            )
        )
    burst = MarkerBurst(
        burst_id=0,
        start_frame=10,
        end_frame_exclusive=17,
        tracks=tuple(tracks),
    )
    assert marker_burst_candidate_frame(
        burst, segment_start_frame=0
    ) == 11


def test_fixed_tower_art_is_not_a_deployment_candidate():
    observations = [
        (
            frame,
            [
                TeamMarker(
                    frame_index=frame,
                    bbox=(100, 300, 25, 25),
                    center=(114.0, 314.0),
                    area=600,
                )
            ],
        )
        for frame in (1, 2, 3)
    ]
    tracks = track_enemy_team_markers(observations)
    assert deployment_candidate_tracks(
        tracks,
        segment_start_frame=0,
    ) == []

    right_tower_observations = [
        (
            frame,
            [
                TeamMarker(
                    frame_index=frame,
                    bbox=(965, 290, 25, 25),
                    center=(978.0, 303.0),
                    area=600,
                )
            ],
        )
        for frame in (1, 2, 3)
    ]
    right_tracks = track_enemy_team_markers(right_tower_observations)
    assert deployment_candidate_tracks(
        right_tracks,
        segment_start_frame=0,
    ) == []


def test_marker_track_at_segment_start_is_not_a_new_deployment():
    observations = [
        (
            frame,
            [
                TeamMarker(
                    frame_index=frame,
                    bbox=(300, 500, 25, 25),
                    center=(312.0, 512.0),
                    area=600,
                )
            ],
        )
        for frame in (10, 11, 12)
    ]
    tracks = track_enemy_team_markers(observations)
    assert (
        deployment_candidate_tracks(
            tracks,
            segment_start_frame=10,
        )
        == []
    )
