from types import SimpleNamespace

from cr_bot.features.action_space import ACTION_GRID
from cr_bot.trackers.enemy_cards import EnemyCardTracker


def _match(track_id, *, center_x=500.0, center_y=700.0):
    troop = SimpleNamespace(
        class_name="musketeer",
        track_id=track_id,
        center_x=center_x,
        center_y=center_y,
        confidence=0.9,
        team="enemy",
    )
    return SimpleNamespace(troop=troop)


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
    assert play["cell"] == ACTION_GRID.pixel_to_cell(500.0, 780.0, arena_px)
    assert play["cell"] != ACTION_GRID.pixel_to_cell(500.0, 700.0, arena_px)


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
    assert play["cell"] == ACTION_GRID.pixel_to_cell(500.0, 780.0, arena_px)
