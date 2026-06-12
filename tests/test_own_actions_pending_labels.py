from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

sys.modules.setdefault("cv2", MagicMock())

from cr_bot.trackers.own_actions import OwnActionTracker, PendingOwnPlay


def test_mirror_drop_reuses_last_confirmed_card_and_costs_one_more():
    tracker = OwnActionTracker()
    tracker._append_action(
        now=200.0,
        card="fireball",
        slot_idx=1,
        cell=(12, 22),
    )
    tracker.last_hand = ["mirror", None, None, None]

    tracker._detect_slot_drops(
        [None, None, None, None],
        elixir=7.0,
        now=198.0,
    )

    pending = tracker.pending[0]
    assert pending.card == "fireball"
    assert pending.played_via == "mirror"
    assert tracker._required_numeric_elixir_drop(pending) == 3.5
    assert tracker._append_action(
        now=198.0,
        card=pending.card,
        slot_idx=pending.slot_idx,
        cell=(12, 22),
        played_via=pending.played_via,
    )
    assert tracker.actions[-1]["played_via"] == "mirror"


def _match(class_name, *, track_id=1, center_x=500.0, center_y=1000.0):
    troop = SimpleNamespace(
        class_name=class_name,
        track_id=track_id,
        center_x=center_x,
        center_y=center_y,
        confidence=0.9,
        team="ally",
    )
    return SimpleNamespace(troop=troop)


def _clock(*, center_x=500.0, center_y=1080.0):
    return {
        "team": "ally",
        "confidence": 0.9,
        "center_x": center_x,
        "center_y": center_y,
    }


def test_pending_skeletons_ignore_unrelated_ally_tracks():
    tracker = OwnActionTracker()

    cell = tracker._infer_cell_from_clock(
        [_match("cannon")],
        arena_px=(0, 0, 1080, 2400),
        clock_boxes=[_clock()],
        card="skeletons",
    )

    assert cell is None


def test_pending_skeletons_accept_skeleton_tracks():
    tracker = OwnActionTracker()

    cell = tracker._infer_cell_from_clock(
        [_match("skeleton")],
        arena_px=(0, 0, 1080, 2400),
        clock_boxes=[_clock()],
        card="skeletons",
    )

    assert cell is not None


def test_confirmed_pending_action_uses_elixir_change_time():
    tracker = OwnActionTracker()
    tracker.pending.append(
        PendingOwnPlay(
            card="hog-rider",
            slot_idx=0,
            started_at_s=289.0,
            elixir_before=5.0,
            elixir_change_time_s=288.7,
            elixir_change_video_time_s=3.1,
        )
    )
    tracker._infer_pending_cell = lambda *args, **kwargs: (1, 17)
    tracker._recent_tracks_for_pending = lambda *args, **kwargs: []

    game_state = SimpleNamespace(
        own_units=[],
        hud=SimpleNamespace(hand_cards=["hog-rider", None, None, None]),
    )

    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=1.0,
        now=287.5,
        frame=None,
        clock_boxes=[],
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["time_left_s"] == 288.7
    assert tracker.actions[0]["video_time_s"] == 3.1


def test_pending_cell_without_elixir_evidence_does_not_append_action():
    tracker = OwnActionTracker()
    tracker.pending.append(
        PendingOwnPlay(
            card="ice-golem",
            slot_idx=3,
            started_at_s=212.0,
            elixir_before=6.0,
        )
    )
    tracker._infer_pending_cell = lambda *args, **kwargs: (9, 17)
    tracker._recent_tracks_for_pending = lambda *args, **kwargs: []

    game_state = SimpleNamespace(
        own_units=[],
        hud=SimpleNamespace(hand_cards=["fireball", "log", "hog-rider", "skeletons"]),
    )

    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=6.0,
        now=211.5,
        frame=None,
        clock_boxes=[],
    )

    assert tracker.actions == []
    assert len(tracker.pending) == 1


def test_pending_troop_reuses_latched_numeric_elixir_drop_when_cell_arrives_later():
    tracker = OwnActionTracker()
    tracker.last_elixir = 6.08
    tracker.pending.append(
        PendingOwnPlay(
            card="ice-spirit",
            slot_idx=2,
            started_at_s=200.0,
            elixir_before=6.08,
        )
    )
    cells = iter([None, (14, 21)])
    tracker._infer_pending_cell = lambda *args, **kwargs: next(cells)
    tracker._recent_tracks_for_pending = lambda *args, **kwargs: []

    game_state = SimpleNamespace(
        own_units=[],
        hud=SimpleNamespace(hand_cards=["fireball", "log", "ice-spirit", "skeletons"]),
    )

    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=4.86,
        now=199.9,
        frame=None,
        clock_boxes=[],
        video_time_s=12.83,
    )

    assert tracker.actions == []
    assert len(tracker.pending) == 1
    assert tracker.pending[0].numeric_elixir_drop_time_s == 199.9
    assert tracker.pending[0].numeric_elixir_drop_video_time_s == 12.83

    tracker.last_elixir = 4.86
    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=4.88,
        now=198.7,
        frame=None,
        clock_boxes=[],
        video_time_s=14.0,
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "ice-spirit"
    assert tracker.actions[0]["cell"] == (14, 21)
    assert tracker.actions[0]["time_left_s"] == 199.9
    assert tracker.actions[0]["video_time_s"] == 12.83


def test_pending_troop_can_confirm_from_own_elixir_before_when_log_consumes_global_drop():
    tracker = OwnActionTracker()
    tracker.last_elixir = 8.15
    tracker.pending.extend(
        [
            PendingOwnPlay(
                card="ice-spirit",
                slot_idx=0,
                started_at_s=185.0,
                elixir_before=8.30,
            ),
            PendingOwnPlay(
                card="log",
                slot_idx=0,
                started_at_s=184.0,
                elixir_before=8.15,
            ),
        ]
    )
    tracker._infer_pending_cell = lambda pending, *args, **kwargs: (
        (17, 30) if pending.card == "ice-spirit" else None
    )
    tracker._recent_tracks_for_pending = lambda *args, **kwargs: []

    game_state = SimpleNamespace(
        own_units=[_match("the-log", track_id=13, center_x=797.2, center_y=1152.1)],
        hud=SimpleNamespace(hand_cards=[None, "fireball", "cannon", "musketeer"]),
    )

    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=5.83,
        now=183.0,
        frame=None,
        clock_boxes=[],
        video_time_s=111.9,
    )

    assert [action["card"] for action in tracker.actions] == ["ice-spirit", "log"]
    assert tracker.actions[0]["cell"] == (17, 30)
    assert tracker.actions[0]["time_left_s"] == 183.0
    assert tracker.actions[0]["video_time_s"] == 111.9
    assert tracker.actions[1]["cell"] is not None


def test_old_musketeer_drop_is_normalized_to_musketeer():
    tracker = OwnActionTracker()
    tracker.last_hand = ["old-musketeer", None, None, None]

    tracker._detect_slot_drops(
        hand=[None, None, None, None],
        elixir=5.0,
        now=200.0,
    )

    assert len(tracker.pending) == 1
    assert tracker.pending[0].card == "musketeer"


def test_elixir_change_video_time_is_calibrated():
    tracker = OwnActionTracker()
    tracker.pending.append(
        PendingOwnPlay(
            card="hog-rider",
            slot_idx=0,
            started_at_s=289.0,
            elixir_before=5.0,
        )
    )

    tracker._attach_elixir_change_to_pending(now=288.7, video_time_s=3.1)

    assert tracker.pending[0].elixir_change_time_s == 288.7
    assert tracker.pending[0].elixir_change_video_time_s == 3.2


def test_barbarian_barrel_uses_rolling_spell_detection():
    tracker = OwnActionTracker()
    tracker.last_elixir = 5.0
    tracker.pending.append(
        PendingOwnPlay(
            card="barbarian-barrel",
            slot_idx=1,
            started_at_s=200.0,
            elixir_before=5.0,
        )
    )

    game_state = SimpleNamespace(
        own_units=[
            _match("barbarian-barrel", track_id=12, center_x=500.0, center_y=1000.0)
        ],
        hud=SimpleNamespace(hand_cards=["hog-rider", "barbarian-barrel", None, None]),
    )

    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=3.0,
        now=199.5,
        frame=None,
        clock_boxes=[],
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "barbarian-barrel"
    assert tracker.actions[0]["slot_idx"] == 1
    assert tracker.actions[0]["cell"] is not None
    assert 12 in tracker.consumed_log_track_ids


def test_log_uses_rolling_spell_detection():
    tracker = OwnActionTracker()
    tracker.last_elixir = 5.0
    tracker.pending.append(
        PendingOwnPlay(
            card="log",
            slot_idx=1,
            started_at_s=200.0,
            elixir_before=5.0,
        )
    )

    game_state = SimpleNamespace(
        own_units=[_match("the-log", track_id=13, center_x=500.0, center_y=1000.0)],
        hud=SimpleNamespace(hand_cards=["hog-rider", "log", None, None]),
    )

    tracker._confirm_pending(
        game_state,
        arena_px=(0, 0, 1080, 2400),
        elixir=3.0,
        now=199.5,
        frame=None,
        clock_boxes=[],
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "log"
    assert 13 in tracker.consumed_log_track_ids


def test_confirm_pending_spell_behavior_is_unchanged_after_shared_helper_extraction():
    tracker = OwnActionTracker()
    pending = PendingOwnPlay(
        card="fireball",
        slot_idx=1,
        started_at_s=200.0,
        elixir_before=5.0,
    )

    tracker.spell_deploy_locator.locate = lambda *args, **kwargs: SimpleNamespace(
        center_x=540.0,
        center_y=1200.0,
    )
    tracker.spell_deploy_locator.locate_released = lambda *args, **kwargs: SimpleNamespace(
        center_x=560.0,
        center_y=1240.0,
    )

    cell, keep_pending = tracker._confirm_pending_spell(
        pending,
        arena_px=(0, 0, 1080, 2400),
        frame=object(),
        elixir_confirms=True,
        now=199.8,
    )

    assert keep_pending is False
    assert cell == pending.spell_target_cell
    assert pending.spell_aim_seen is True
    assert pending.spell_release_seen is True
    assert pending.spell_elixir_confirmed is True
    assert tracker.claimed_spell_target_observations
