from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

sys.modules.setdefault("cv2", MagicMock())

from cr_bot.features.action_space import ACTION_GRID
from cr_bot.trackers.enemy_cards import EnemyCardTracker
from cr_bot.trackers.enemy_cards.models import RecentSpellTargetObservation


def _match(track_id, *, class_name="musketeer", center_x=500.0, center_y=700.0):
    troop = SimpleNamespace(
        class_name=class_name,
        track_id=track_id,
        center_x=center_x,
        center_y=center_y,
        confidence=0.9,
        team="enemy",
    )
    return SimpleNamespace(troop=troop)


def _ally_match(track_id, *, class_name="musketeer", center_x=500.0, center_y=700.0):
    match = _match(track_id, class_name=class_name, center_x=center_x, center_y=center_y)
    match.troop.team = "ally"
    return match


def _clock(*, track_id=10, center_x=500.0, center_y=780.0):
    return {
        "track_id": track_id,
        "team": "enemy",
        "confidence": 0.9,
        "center_x": center_x,
        "center_y": center_y,
    }


def _tracker():
    tracker = EnemyCardTracker()
    tracker.start_match(time_left_s=180, total_remaining_s=300, now_s=0.0)
    return tracker


def _obs(card, time_left_s, cell, arena_px, *, phase="release", quality=1.0):
    center_x, center_y = ACTION_GRID.cell_to_pixel_center(*cell, arena_px)
    return RecentSpellTargetObservation(
        card=card,
        time_left_s=time_left_s,
        cell=cell,
        phase=phase,
        quality=quality,
        center_x=center_x,
        center_y=center_y,
        key=f"{card}:{phase}:{time_left_s}:{cell}",
    )


def test_enemy_repeat_before_full_cycle_is_marked_as_mirror():
    tracker = _tracker()

    tracker.update(177, [_match(1)], clock_boxes=[_clock(track_id=11)], now_s=1.0)
    tracker.update(176, [_match(2)], clock_boxes=[_clock(track_id=12)], now_s=2.0)

    first, mirrored = tracker.detected_card_plays
    assert first["card"] == "musketeer"
    assert first["played_via"] is None
    assert mirrored["card"] == "musketeer"
    assert mirrored["played_via"] == "mirror"
    assert mirrored["cost"] == 5


def test_enemy_repeat_after_four_other_plays_is_normal_cycle():
    tracker = _tracker()
    cards = ["musketeer", "knight", "golem", "royal-giant", "hog-rider", "musketeer"]

    for idx, card in enumerate(cards):
        tracker.update(
            177 - idx,
            [_match(idx + 1, class_name=card)],
            clock_boxes=[_clock(track_id=20 + idx)],
            now_s=float(idx + 1),
        )

    repeated = tracker.detected_card_plays[-1]
    assert repeated["card"] == "musketeer"
    assert repeated["played_via"] is None
    assert repeated["cost"] == 4


def _raised_cell(cell, rows=2):
    col, row = cell
    return col, max(0, row - rows)


def _frame_with_burst(arena_px, cell=None, *, color=(40, 160, 255)):
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    if cell is None:
        return frame
    center_x, center_y = ACTION_GRID.cell_to_pixel_center(*cell, arena_px)
    cell_w = arena_px[2] * ACTION_GRID.width / 18
    cell_h = arena_px[3] * ACTION_GRID.height / 32
    radius_x = max(8, int(round(cell_w * 0.8)))
    radius_y = max(8, int(round(cell_h * 0.8)))
    x0 = max(0, int(round(center_x - radius_x)))
    y0 = max(0, int(round(center_y - radius_y)))
    x1 = min(frame.shape[1], int(round(center_x + radius_x)))
    y1 = min(frame.shape[0], int(round(center_y + radius_y)))
    frame[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)
    return frame


def test_enemy_clock_track_can_confirm_only_one_current_troop_track():
    tracker = _tracker()
    clock = _clock()

    tracker.update(289, [_match(1)], clock_boxes=[clock], now_s=1.0)
    tracker.update(288, [_match(2, center_y=720.0)], clock_boxes=[clock], now_s=1.1)

    assert [play["track_id"] for play in tracker.detected_card_plays] == [1]
    assert tracker.tracks[1].clock_confirmed is True
    assert tracker.tracks[2].clock_confirmed is False


def test_consumed_recent_enemy_clock_cannot_confirm_replacement_troop_track():
    tracker = _tracker()

    tracker.update(289, [_match(1)], clock_boxes=[_clock()], now_s=1.0)
    tracker.update(288, [_match(2, center_y=720.0)], clock_boxes=[], now_s=1.1)

    assert [play["track_id"] for play in tracker.detected_card_plays] == [1]
    assert tracker.tracks[2].clock_confirmed is False


def test_enemy_clock_is_single_use_within_update_without_monotonic_time():
    tracker = _tracker()

    tracker.update(289, [_match(1), _match(2)], clock_boxes=[_clock()])

    assert [play["track_id"] for play in tracker.detected_card_plays] == [1]
    assert tracker.tracks[2].clock_confirmed is False


def test_enemy_clock_candidate_debug_includes_positions(capsys):
    tracker = _tracker()

    tracker.update(
        289,
        [_match(1, center_x=500.0, center_y=700.0)],
        clock_boxes=[_clock(center_x=800.0, center_y=780.0)],
        now_s=1.0,
    )

    out = capsys.readouterr().out
    assert "clock candidate track=1 class=musketeer source=current status=rejected" in out
    assert "troop_center=(500.0,700.0)" in out
    assert "clock_center=(800.0,780.0)" in out
    assert "dx=300.0 dy=80.0" in out
    assert "reject=clock horizontal gap 300.0 > 90" in out


def test_enemy_play_cell_uses_claimed_clock_center_instead_of_troop_center():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    clock = _clock(center_x=500.0, center_y=780.0)

    tracker.update(
        289,
        [_match(1, center_x=500.0, center_y=700.0)],
        clock_boxes=[clock],
        now_s=1.0,
        arena_px=arena_px,
    )

    play = tracker.detected_card_plays[0]
    assert play["cell"] == _raised_cell(ACTION_GRID.pixel_to_cell(500.0, 780.0, arena_px))
    assert play["cell"] != _raised_cell(ACTION_GRID.pixel_to_cell(500.0, 700.0, arena_px))


def test_enemy_play_cell_uses_remembered_clock_center():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    clock = _clock(center_x=500.0, center_y=780.0)

    tracker.update(290, [], clock_boxes=[clock], now_s=1.0, arena_px=arena_px)
    tracker.update(
        289,
        [_match(1, center_x=500.0, center_y=700.0)],
        clock_boxes=[],
        now_s=1.1,
        arena_px=arena_px,
    )

    play = tracker.detected_card_plays[0]
    assert play["cell"] == _raised_cell(ACTION_GRID.pixel_to_cell(500.0, 780.0, arena_px))


def test_enemy_cell_calibration_raises_rows_by_two():
    tracker = _tracker()

    assert tracker._raise_cell_rows((7, 7), rows=2) == (7, 5)
    assert tracker._raise_cell_rows((7, 1), rows=2) == (7, 0)
    assert tracker._raise_cell_rows(None, rows=2) is None


def test_clock_confirmed_enemy_skeleton_records_skeletons_card():
    tracker = _tracker()

    tracker.update(
        289,
        [_match(1, class_name="skeleton")],
        clock_boxes=[_clock()],
        now_s=1.0,
    )

    assert tracker.detected_card_plays[0]["card"] == "skeletons"
    assert tracker.detected_card_plays[0]["track_id"] == 1


def test_frame_confirm_troop_exception_records_without_clock():
    tracker = _tracker()

    for idx in range(3):
        tracker.update(
            289 - idx,
            [_match(1, class_name="electro-wizard")],
            clock_boxes=[],
            now_s=1.0 + idx * 0.1,
        )

    assert tracker.tracks[1].frame_confirmed is True
    assert tracker.detected_card_plays[0]["card"] == "electro-wizard"
    assert tracker.detected_card_plays[0]["track_id"] == 1


def test_ally_matches_do_not_poison_enemy_track_memory():
    tracker = _tracker()

    tracker.update(290, [_ally_match(1)], clock_boxes=[_clock()], now_s=1.0)
    tracker.update(289, [_match(1)], clock_boxes=[_clock()], now_s=1.1)

    assert tracker.tracks[1].best_team == "enemy"
    assert tracker.detected_card_plays[0]["card"] == "musketeer"


def test_enemy_play_video_time_uses_first_seen_video_time_not_match_clock():
    tracker = _tracker()

    tracker.update(
        177,
        [_match(1, class_name="golem")],
        clock_boxes=[_clock()],
        now_s=123.0,
    )

    play = tracker.detected_card_plays[0]
    assert play["card"] == "golem"
    assert play["video_time_s"] == 123.0


def test_recorded_enemy_play_is_relabelled_when_track_class_later_flips():
    tracker = _tracker()

    tracker.update(
        177,
        [_match(1, class_name="royal-giant")],
        clock_boxes=[_clock()],
        now_s=123.0,
    )
    for idx in range(3):
        tracker.update(
            176 - idx,
            [_match(1, class_name="golem")],
            clock_boxes=[],
            now_s=180.0 + idx,
        )

    play = tracker.detected_card_plays[0]
    assert play["card"] == "golem"
    assert play["best_class"] == "golem"
    assert play["event_id"].startswith("golem_")
    assert play["cost"] == 8
    assert play["class_votes"]["golem"] >= 3
    assert play["video_time_s"] == 123.0


def test_enemy_fireball_event_revises_from_late_observation_without_track_continuity():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)
    tracker.update(
        200.0,
        [_match(1, class_name="fireball", center_x=560.0, center_y=360.0)],
        arena_px=arena_px,
    )
    tracker.update(
        199.5,
        [_match(1, class_name="fireball", center_x=610.0, center_y=470.0)],
        arena_px=arena_px,
    )
    tracker.update(
        199.0,
        [_match(1, class_name="fireball", center_x=655.0, center_y=610.0)],
        arena_px=arena_px,
    )
    tracker.update(
        198.8,
        [_ally_match(7, class_name="fireball", center_x=target_x, center_y=target_y)],
        arena_px=arena_px,
    )

    assert tracker.detected_card_plays[0]["track_id"] == 1

    play = tracker.detected_card_plays[0]
    assert play["cell"] == (12, 22)
    assert tracker.projectile_spell_events[0].finalized is True
    assert tracker.projectile_spell_events[0].first_track_id == 1


def test_enemy_fireball_fragment_is_merged_before_late_ally_endpoint():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)

    tracker.update(200.0, [_match(1, class_name="fireball", center_x=850.0, center_y=220.0)], arena_px=arena_px)
    tracker.update(199.5, [_match(1, class_name="fireball", center_x=860.0, center_y=240.0)], arena_px=arena_px)
    tracker.update(199.0, [_match(1, class_name="fireball", center_x=870.0, center_y=260.0)], arena_px=arena_px)

    tracker.update(198.5, [_match(7, class_name="fireball", center_x=580.0, center_y=300.0)], arena_px=arena_px)
    tracker.update(198.0, [_match(7, class_name="fireball", center_x=620.0, center_y=440.0)], arena_px=arena_px)
    tracker.update(197.5, [_match(7, class_name="fireball", center_x=660.0, center_y=590.0)], arena_px=arena_px)
    tracker.update(197.2, [_ally_match(9, class_name="fireball", center_x=target_x, center_y=target_y)], arena_px=arena_px)

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 1
    assert tracker.detected_card_plays[0]["cell"] == (12, 22)


def test_enemy_fireball_event_can_revise_from_observation_seen_just_before_frame_confirm():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)
    tracker.update(
        200.0,
        [_match(1, class_name="fireball", center_x=570.0, center_y=390.0)],
        arena_px=arena_px,
    )
    tracker.update(
        199.5,
        [_match(1, class_name="fireball", center_x=620.0, center_y=500.0)],
        arena_px=arena_px,
    )
    tracker.update(
        199.0,
        [
            _match(1, class_name="fireball", center_x=660.0, center_y=640.0),
            _ally_match(7, class_name="fireball", center_x=target_x, center_y=target_y),
        ],
        arena_px=arena_px,
    )

    assert tracker.detected_card_plays[0]["cell"] == (12, 22)
    assert tracker.projectile_spell_events[0].finalized is True


def test_enemy_fireball_trajectory_filter_rejects_late_impossible_candidate():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    tracker.update(200.0, [_match(1, class_name="fireball", center_x=500.0, center_y=200.0)], arena_px=arena_px)
    tracker.update(199.5, [_match(1, class_name="fireball", center_x=520.0, center_y=260.0)], arena_px=arena_px)
    tracker.update(199.0, [_match(1, class_name="fireball", center_x=540.0, center_y=320.0)], arena_px=arena_px)

    original_cell = tracker.detected_card_plays[0]["cell"]
    tracker.update(198.6, [_ally_match(7, class_name="fireball", center_x=120.0, center_y=720.0)], arena_px=arena_px)

    assert tracker.detected_card_plays[0]["cell"] == original_cell


def test_own_fireball_action_blocks_enemy_projectile_reuse_without_changing_own_cell():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)
    own_action = {"card": "fireball", "time_left_s": 199.1, "cell": (12, 22)}
    tracker.update(200.0, [_match(1, class_name="fireball", center_x=560.0, center_y=360.0)], arena_px=arena_px)
    tracker.update(199.5, [_match(1, class_name="fireball", center_x=610.0, center_y=470.0)], arena_px=arena_px)
    tracker.update(
        199.0,
        [
            _match(1, class_name="fireball", center_x=655.0, center_y=610.0),
            _ally_match(7, class_name="fireball", center_x=target_x, center_y=target_y),
        ],
        arena_px=arena_px,
        own_actions=[own_action],
    )

    assert tracker.detected_card_plays[0]["cell"] != (12, 22)
    assert own_action["cell"] == (12, 22)


def test_downward_ally_labelled_fireball_without_recent_own_action_records_enemy_spell():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            now_s=7.9 + idx * 0.1,
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    play = tracker.detected_card_plays[0]
    assert play["card"] == "fireball"
    assert play["track_id"] == 19
    assert play["video_time_s"] == 7.9


def test_recent_own_fireball_at_same_cell_suppresses_ally_labelled_enemy_fallback():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)
    own_action = {"card": "fireball", "time_left_s": 200.1, "cell": (12, 22)}

    for idx in range(3):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=target_x, center_y=target_y)],
            own_actions=[own_action],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []
    assert own_action["cell"] == (12, 22)


def test_recent_own_fireball_at_different_cell_keeps_ally_labelled_enemy_fallback():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    own_action = {"card": "fireball", "time_left_s": 200.1, "cell": (3, 8)}

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            own_actions=[own_action],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "fireball"
    assert own_action["cell"] == (3, 8)


def test_upward_fireball_is_claimed_internally_even_when_yolo_labels_it_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((700.0, 620.0, 530.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_upward_giant_snowball_is_suppressed_when_yolo_labels_it_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((700.0, 620.0, 530.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="giant-snowball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_downward_enemy_giant_snowball_records_projectile_spell():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 360.0, 430.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="giant-snowball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "giant-snowball"


def test_ally_labelled_giant_snowball_does_not_use_fireball_fallback():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="giant-snowball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_direction_only_projectiles_suppress_upward_enemy_detections():
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for card in ("goblin-barrel", "arrows", "rocket"):
        tracker = _tracker()
        for idx, center_y in enumerate((700.0, 620.0, 530.0)):
            tracker.update(
                200.0 - idx * 0.1,
                [_match(19, class_name=card, center_x=600.0, center_y=center_y)],
                arena_px=arena_px,
            )

        assert tracker.detected_card_plays == []


def test_direction_only_projectiles_record_downward_enemy_detections():
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for card in ("goblin-barrel", "arrows", "rocket"):
        tracker = _tracker()
        for idx, center_y in enumerate((300.0, 360.0, 430.0)):
            tracker.update(
                200.0 - idx * 0.1,
                [_match(19, class_name=card, center_x=600.0, center_y=center_y)],
                arena_px=arena_px,
            )

        assert len(tracker.detected_card_plays) == 1
        assert tracker.detected_card_plays[0]["card"] == card


def test_direction_only_projectiles_ignore_ally_labels():
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for card in ("goblin-barrel", "arrows", "rocket"):
        tracker = _tracker()
        for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0)):
            tracker.update(
                200.0 - idx * 0.1,
                [_ally_match(19, class_name=card, center_x=600.0, center_y=center_y)],
                arena_px=arena_px,
            )

        assert tracker.detected_card_plays == []


def test_reconcile_skips_legacy_direction_only_projectile_events():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    event = tracker.projectile_spell_events[0]
    event.card = "goblin-barrel"
    tracker.projectiles.reconcile(
        arena_px=arena_px,
        claimed_spell_observation_keys=set(),
    )

    assert event.card == "goblin-barrel"


def test_rocket_cell_advances_with_projectile_trajectory():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((120.0, 220.0, 340.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="rocket", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )
    initial_cell = tracker.detected_card_plays[0]["cell"]

    tracker.update(
        199.6,
        [_match(20, class_name="rocket", center_x=610.0, center_y=760.0)],
        arena_px=arena_px,
    )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["cell"][1] > initial_cell[1]


def test_projectile_event_keeps_early_and_late_trajectory_samples():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx in range(12):
        tracker.update(
            200.0 - idx * 0.1,
            [
                _match(
                    19,
                    class_name="rocket",
                    center_x=500.0 + idx * 5.0,
                    center_y=100.0 + idx * 60.0,
                )
            ],
            arena_px=arena_px,
        )

    samples = tracker.projectile_spell_events[0].observed_centers
    assert len(samples) == 8
    assert max(center_y for _, _, center_y in samples) >= 700.0


def test_goblin_barrel_cell_refines_to_spawned_enemy_goblin():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((100.0, 180.0, 260.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="goblin-barrel", center_x=700.0, center_y=center_y)],
            arena_px=arena_px,
        )
    goblin_cell = (14, 24)
    goblin_x, goblin_y = ACTION_GRID.cell_to_pixel_center(*goblin_cell, arena_px)

    tracker.update(
        199.4,
        [_match(20, class_name="goblin", center_x=goblin_x, center_y=goblin_y)],
        arena_px=arena_px,
    )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["cell"] == goblin_cell


def test_royal_delivery_cell_refines_to_spawned_enemy_recruit():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx in range(3):
        tracker.update(
            200.0 - idx * 0.1,
            [_match(19, class_name="royal-delivery", center_x=250.0, center_y=180.0)],
            arena_px=arena_px,
        )
    recruit_cell = (4, 7)
    recruit_x, recruit_y = ACTION_GRID.cell_to_pixel_center(*recruit_cell, arena_px)

    tracker.update(
        199.4,
        [_match(20, class_name="royal-recruit", center_x=recruit_x, center_y=recruit_y)],
        arena_px=arena_px,
    )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["cell"] == recruit_cell


def test_compact_upward_fireball_uses_direction_before_explosion_fallback():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((700.0, 680.0, 660.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_compact_explosion_is_not_published_before_later_direction_resolves_own():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((700.0, 698.0, 696.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []

    for idx, center_y in enumerate((670.0, 640.0), start=3):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_compact_downward_fireball_uses_direction_before_explosion_fallback():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 320.0, 340.0, 360.0, 380.0, 400.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "fireball"


def test_duplicate_same_frame_fireball_boxes_preserve_direction_history():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        matches = [
            _ally_match(
                19,
                class_name="fireball",
                center_x=600.0 + duplicate_idx,
                center_y=center_y + duplicate_idx,
            )
            for duplicate_idx in range(6)
        ]
        tracker.update(
            200.0 - idx * 0.1,
            matches,
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "fireball"


def test_fireball_direction_uses_video_time_when_match_clock_is_unchanged():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        tracker.update(
            111.0,
            [_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            now_s=183.0 + idx * 0.1,
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "fireball"


def test_pending_own_fireball_suppresses_different_cell_enemy_fallback():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    pending_target = {
        "card": "fireball",
        "time_left_s": 200.1,
        "cell": (3, 8),
    }

    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=600.0, center_y=center_y)],
            pending_own_spell_targets=[pending_target],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_reused_non_fireball_track_id_does_not_create_fireball_play():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    tracker.update(
        200.0,
        [_match(32, class_name="mega-knight")],
        clock_boxes=[_clock(track_id=63)],
        arena_px=arena_px,
    )
    for idx, center_y in enumerate((300.0, 360.0, 430.0, 500.0, 570.0, 640.0)):
        tracker.update(
            199.9 - idx * 0.1,
            [_ally_match(32, class_name="fireball", center_x=600.0, center_y=center_y)],
            arena_px=arena_px,
        )

    assert [play["card"] for play in tracker.detected_card_plays] == ["mega-knight"]


def test_compact_fireball_explosion_without_matching_own_target_records_enemy_spell():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, center_x in enumerate((440.0, 450.0, 445.0)):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=center_x, center_y=500.0)],
            arena_px=arena_px,
        )
    tracker.update(199.3, [], arena_px=arena_px)

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "fireball"


def test_compact_fireball_explosion_matching_pending_own_target_is_consumed():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    target_cell = (8, 15)
    target_x, target_y = ACTION_GRID.cell_to_pixel_center(*target_cell, arena_px)
    pending_target = {
        "card": "fireball",
        "time_left_s": 200.1,
        "cell": target_cell,
    }

    for idx in range(3):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=target_x, center_y=target_y)],
            pending_own_spell_targets=[pending_target],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_compact_fireball_explosion_at_different_pending_target_is_provisionally_owned():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)
    pending_target = {
        "card": "fireball",
        "time_left_s": 200.1,
        "cell": (3, 8),
    }

    for idx in range(3):
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(19, class_name="fireball", center_x=target_x, center_y=target_y)],
            pending_own_spell_targets=[pending_target],
            arena_px=arena_px,
        )
    tracker.update(
        199.3,
        [],
        pending_own_spell_targets=[pending_target],
        arena_px=arena_px,
    )

    assert tracker.detected_card_plays == []


def test_log_moving_toward_increasing_rows_records_enemy_despite_team_labels():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    teams = ("ally", "enemy", "ally", "ally")

    for idx, row in enumerate((13, 13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        match = _ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)
        match.troop.team = teams[idx]
        tracker.update(
            200.0 - idx * 0.1,
            [match],
            now_s=40.3 + idx * 0.1,
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    play = tracker.detected_card_plays[0]
    assert play["card"] == "log"
    assert play["track_id"] == 24
    assert play["video_time_s"] == 40.3


def test_log_moving_toward_decreasing_rows_is_not_recorded_as_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((22, 21, 20, 19)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(8, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_stationary_log_team_noise_is_not_recorded_as_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 13, 13, 13)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_log_row_jitter_is_not_recorded_as_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 14, 13, 14, 13)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []


def test_fragmented_enemy_log_is_suppressed_within_log_duplicate_window():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )
    for idx, row in enumerate((16, 17, 18)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            198.8 - idx * 0.1,
            [_match(25, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 24


def test_later_enemy_log_in_different_lane_is_not_suppressed():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for track_id, lane, start_time in ((24, 4, 200.0), (25, 13, 198.8)):
        for idx, row in enumerate((13, 14, 15)):
            center_x, center_y = ACTION_GRID.cell_to_pixel_center(
                lane, row, arena_px
            )
            tracker.update(
                start_time - idx * 0.1,
                [
                    _match(
                        track_id,
                        class_name="the-log",
                        center_x=center_x,
                        center_y=center_y,
                    )
                ],
                arena_px=arena_px,
            )

    assert [play["track_id"] for play in tracker.detected_card_plays] == [24, 25]


def test_log_fragment_history_survives_recorded_play_reclassification():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )
    tracker.detected_card_plays[0].card = "ice-golem"

    for idx, row in enumerate((16, 17, 18)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            198.8 - idx * 0.1,
            [_match(25, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1


def test_increasing_short_fragment_at_own_log_time_is_suppressed():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    own_action = {"card": "log", "time_left_s": 199.9, "cell": (13, 16)}

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            own_actions=[own_action],
            arena_px=arena_px,
        )

    assert tracker.detected_card_plays == []
    assert own_action["cell"] == (13, 16)


def test_enemy_log_started_before_nearby_own_log_remains_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    own_action = {"card": "log", "time_left_s": 199.0, "cell": (13, 16)}
    tracker.reconcile_own_actions([own_action], arena_px=arena_px)

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 24


def test_increasing_shared_own_log_track_can_also_record_enemy_log():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    own_action = {
        "card": "log",
        "time_left_s": 199.9,
        "cell": (13, 16),
        "rolling_spell_track_id": 24,
    }

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            own_actions=[own_action],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 24


def test_late_own_log_confirmation_does_not_remove_enemy_direction_log():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(4, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )
    assert len(tracker.detected_card_plays) == 1

    own_action = {"card": "log", "time_left_s": 197.2, "cell": (4, 17)}
    tracker.reconcile_own_actions([own_action], arena_px=arena_px)

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 24
    assert tracker.log_trajectory_candidates[24].counted_as_card is True

    for idx, row in enumerate((17, 16, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(4, row, arena_px)
        tracker.update(
            197.0 - idx * 0.1,
            [_ally_match(25, class_name="the-log", center_x=center_x, center_y=center_y)],
            own_actions=[own_action],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 24


def test_simultaneous_log_in_different_lane_remains_enemy():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)
    own_action = {"card": "log", "time_left_s": 200.1, "cell": (3, 16)}

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.0 - idx * 0.1,
            [_ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            own_actions=[own_action],
            arena_px=arena_px,
        )

    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["card"] == "log"


def test_own_log_claim_selects_closest_same_lane_track_only():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    for idx, row in enumerate((13, 14, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            203.0 - idx * 0.1,
            [_ally_match(24, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )
    for idx, row in enumerate((17, 16, 15)):
        center_x, center_y = ACTION_GRID.cell_to_pixel_center(13, row, arena_px)
        tracker.update(
            200.2 - idx * 0.1,
            [_ally_match(25, class_name="the-log", center_x=center_x, center_y=center_y)],
            arena_px=arena_px,
        )

    own_action = {"card": "log", "time_left_s": 200.1, "cell": (13, 17)}
    tracker.reconcile_own_actions([own_action], arena_px=arena_px)

    assert tracker.own_log_claims[(200.1, (13, 17))] == 25
    assert len(tracker.detected_card_plays) == 1
    assert tracker.detected_card_plays[0]["track_id"] == 24


def test_two_nearby_enemy_fireballs_resolve_in_time_order():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    x1, y1 = ACTION_GRID.cell_to_pixel_center(7, 22, arena_px)
    x2, y2 = ACTION_GRID.cell_to_pixel_center(11, 22, arena_px)
    tracker.update(200.0, [_match(1, class_name="fireball", center_x=340.0, center_y=260.0)], arena_px=arena_px)
    tracker.update(199.5, [_match(1, class_name="fireball", center_x=380.0, center_y=430.0)], arena_px=arena_px)
    tracker.update(199.0, [_match(1, class_name="fireball", center_x=420.0, center_y=600.0)], arena_px=arena_px)

    tracker.update(198.7, [_ally_match(9, class_name="fireball", center_x=x1, center_y=y1)], arena_px=arena_px)

    tracker.update(195.0, [_match(2, class_name="fireball", center_x=610.0, center_y=260.0)], arena_px=arena_px)
    tracker.update(194.5, [_match(2, class_name="fireball", center_x=630.0, center_y=430.0)], arena_px=arena_px)
    tracker.update(194.0, [_match(2, class_name="fireball", center_x=650.0, center_y=600.0)], arena_px=arena_px)
    tracker.update(193.7, [_ally_match(10, class_name="fireball", center_x=x2, center_y=y2)], arena_px=arena_px)

    assert [play["cell"] for play in tracker.detected_card_plays[:2]] == [(7, 22), (11, 22)]


def test_finalized_enemy_projectile_cell_does_not_regress_during_track_sync():
    tracker = _tracker()
    arena_px = (0.0, 0.0, 1000.0, 1000.0)

    target_x, target_y = ACTION_GRID.cell_to_pixel_center(12, 22, arena_px)
    tracker.update(200.0, [_match(1, class_name="fireball", center_x=560.0, center_y=360.0)], arena_px=arena_px)
    tracker.update(199.5, [_match(1, class_name="fireball", center_x=610.0, center_y=470.0)], arena_px=arena_px)
    tracker.update(199.0, [_match(1, class_name="fireball", center_x=655.0, center_y=610.0)], arena_px=arena_px)
    tracker.update(198.8, [_ally_match(7, class_name="fireball", center_x=target_x, center_y=target_y)], arena_px=arena_px)

    tracker.update(198.0, [_match(1, class_name="fireball", center_x=550.0, center_y=350.0)], arena_px=arena_px)

    assert tracker.detected_card_plays[0]["cell"] == (12, 22)
