from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
from cr_bot.domain.frame_analysis import FrameAnalysisResult
from cr_bot.replay import ReplayCacheReader, ReplayCacheWriter


def _analysis() -> FrameAnalysisResult:
    return FrameAnalysisResult(
        rendered=np.zeros((2, 2, 3), dtype=np.uint8),
        elixir={"estimated_value": 4.0, "displayed_digit": 1.0},
        elixir_change={"covered": False},
        towers_hp={
            "own_support_left": 1,
            "own_king": 1,
            "own_support_right": 1,
            "enemy_support_left": 1,
            "enemy_king": 1,
            "enemy_support_right": 1,
        },
        time="2:30",
        time_left_s=150.0,
        total_remaining_s=150.0,
        overtime=False,
        hand_state={
            "card_1": "fireball",
            "card_2": None,
            "card_3": None,
            "card_4": None,
            "next_card": None,
        },
        yolo_boxes=np.array([[1, 2, 3]], dtype=np.float32),
        clock_boxes=[],
        emote_boxes=[],
        matches=[],
        arena_px=(0, 0, 2, 2),
        tower_hp_debug_steps={"unused": {}},
        timer_debug_steps={"unused": object()},
    )


def test_replay_cache_round_trip_is_lossless(tmp_path: Path, monkeypatch):
    path = tmp_path / "sample.pkl.gz"
    frame = np.array(
        [
            [[0, 10, 255], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )
    encoded_frame = frame.tobytes()
    monkeypatch.setattr(
        "cr_bot.replay.cache.cv2.imencode",
        lambda extension, image, options: (
            True,
            np.frombuffer(encoded_frame, dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        "cr_bot.replay.cache.cv2.imdecode",
        lambda encoded, mode: np.frombuffer(encoded.tobytes(), dtype=np.uint8)
        .reshape(frame.shape)
        .copy(),
    )

    with ReplayCacheWriter(path) as writer:
        writer.write(
            frame_idx=7,
            video_time_s=12.5,
            analysis=_analysis(),
            frame=frame,
        )

    records = list(ReplayCacheReader(path))

    assert len(records) == 1
    assert records[0].frame_idx == 7
    assert records[0].video_time_s == 12.5
    assert np.array_equal(records[0].decode_frame(), frame)
    assert records[0].analysis.time_left_s == 150.0
    assert records[0].analysis.rendered is None
    assert records[0].analysis.yolo_boxes is None
