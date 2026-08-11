from __future__ import annotations

import numpy as np
import pytest

from cr_bot.domain.events import TemporalSpellDetection
from cr_bot.temporal_spells.config import TemporalSpellConfig
from cr_bot.temporal_spells.dataset import (
    normalize_manifest_row,
    overlaps_event_window,
    split_rows_by_session,
)
from cr_bot.temporal_spells.features import clip_to_tensor
from cr_bot.trackers.enemy_cards import EnemyCardTracker


def test_clip_tensor_shape_and_first_difference():
    frames = [np.full((20, 10, 3), index, dtype=np.uint8) for index in range(8)]
    tensor = clip_to_tensor(frames, TemporalSpellConfig())
    assert tensor.shape == (8, 6, 288, 192)
    assert np.all(tensor[0, 3:] == 0)


def test_session_split_never_leaks_session():
    rows = [
        {"video": f"v{index}", "recording_session": f"s{index // 2}"}
        for index in range(20)
    ]
    splits = split_rows_by_session(rows, val_fraction=0.2, test_fraction=0.2)
    session_sets = {
        name: {row["recording_session"] for row in split_rows}
        for name, split_rows in splits.items()
    }
    assert not session_sets["train"] & session_sets["val"]
    assert not session_sets["train"] & session_sets["test"]
    assert not session_sets["val"] & session_sets["test"]


def test_negative_window_excludes_events_inside_causal_clip():
    config = TemporalSpellConfig()
    assert overlaps_event_window(10.0, [9.5], config)
    assert not overlaps_event_window(10.0, [8.0], config)


@pytest.mark.parametrize("ownership", ["own", "enemy"])
def test_own_and_enemy_casts_keep_same_visual_class(ownership):
    row = normalize_manifest_row({"card": "zap", "ownership": ownership})
    assert row["card"] == "zap"
    assert row["ownership"] == ownership


def test_positive_cast_requires_ownership():
    with pytest.raises(ValueError, match="must have ownership"):
        normalize_manifest_row({"card": "zap"})


def test_background_uses_background_ownership():
    row = normalize_manifest_row({"card": "background"})
    assert row["ownership"] == "background"


def test_model_shapes():
    torch = pytest.importorskip("torch")
    from cr_bot.temporal_spells.model import TemporalSpellCNN

    events, heatmaps = TemporalSpellCNN()(torch.zeros(2, 8, 6, 288, 192))
    assert events.shape == (2, 5)
    assert heatmaps.shape == (2, 4, 32, 18)


def test_tracker_records_temporal_only_without_track_id():
    tracker = EnemyCardTracker(debug=False)
    tracker.start_match(180.0, 180.0, now_s=0.0)
    detection = TemporalSpellDetection("zap", 0.95, 10.0, (8, 20), 0.8)
    tracker.update(170.0, [], now_s=10.0, temporal_spell_detections=[detection])
    play = tracker.detected_card_plays[-1]
    assert play.card == "zap"
    assert play.track_id is None
    assert play.played_via == "temporal-vision"


def test_tracker_vetoes_matching_own_temporal_spell():
    tracker = EnemyCardTracker(debug=False)
    tracker.start_match(180.0, 180.0, now_s=0.0)
    detection = TemporalSpellDetection("zap", 0.95, 10.0, (8, 20), 0.8)
    tracker.update(
        170.0,
        [],
        now_s=10.0,
        own_actions=[{"card": "zap", "video_time_s": 9.8, "cell": (9, 20)}],
        temporal_spell_detections=[detection],
    )
    assert tracker.detected_card_plays == []
