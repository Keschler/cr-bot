from types import SimpleNamespace

from cr_bot.features.action_space import ACTION_GRID
from cr_bot.trackers.enemy_cards import EnemyCardTracker


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


def _raised_cell(cell, rows=2):
    col, row = cell
    return col, max(0, row - rows)


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
