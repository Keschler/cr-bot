from types import SimpleNamespace

from cr_bot.domain.game_state import Detection, Match
from cr_bot.trackers.own_actions import OwnActionTracker, PendingOwnPlay


ARENA_PX = (0, 0, 568, 896)


def _state(*, hand, elixir, own_units=None, now=100.0):
    return SimpleNamespace(
        total_remaining_s=now,
        hud=SimpleNamespace(
            hand_cards=hand,
            elixir_self=elixir,
        ),
        own_units=own_units or [],
    )


def _own_log(center_x=284.0, center_y=500.0, confidence=0.9, track_id=42):
    return Match(
        troop=Detection(
            track_id=track_id,
            class_name="the-log",
            team="ally",
            confidence=confidence,
            x1=center_x - 15,
            y1=center_y - 45,
            x2=center_x + 15,
            y2=center_y + 45,
            center_x=center_x,
            center_y=center_y,
        ),
        bar=None,
    )


def _tracker_with_log_in_hand():
    tracker = OwnActionTracker()
    tracker.last_hand = ["log", "ice-spirit", "hog-rider", "musketeer"]
    tracker.last_elixir = 5.0
    for _ in range(5):
        tracker._remember_slot_history(tracker.last_hand)
    return tracker


def test_log_slot_change_confirms_from_first_visible_log_and_elixir_drop():
    tracker = _tracker_with_log_in_hand()

    tracker.update(
        _state(
            hand=["skeletons", "ice-spirit", "hog-rider", "musketeer"],
            elixir=3.0,
            own_units=[_own_log()],
        ),
        ARENA_PX,
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "log"
    assert tracker.actions[0]["slot_idx"] == 0
    assert tracker.actions[0]["cell"] is not None


def test_log_visible_without_sufficient_numeric_drop_stays_pending():
    tracker = _tracker_with_log_in_hand()

    tracker.update(
        _state(
            hand=["skeletons", "ice-spirit", "hog-rider", "musketeer"],
            elixir=4.6,
            own_units=[_own_log()],
        ),
        ARENA_PX,
    )

    assert tracker.actions == []
    assert len(tracker.pending) == 1
    assert tracker.pending[0].card == "log"
    assert tracker.pending[0].rolling_spell_first_cell is not None


def test_log_confirms_when_yolo_appears_after_elixir_drop():
    tracker = _tracker_with_log_in_hand()

    tracker.update(
        _state(
            hand=["skeletons", "ice-spirit", "hog-rider", "musketeer"],
            elixir=3.0,
            own_units=[],
            now=100.0,
        ),
        ARENA_PX,
    )
    tracker.update(
        _state(
            hand=["skeletons", "ice-spirit", "hog-rider", "musketeer"],
            elixir=3.0,
            own_units=[_own_log()],
            now=99.8,
        ),
        ARENA_PX,
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "log"


def test_first_visible_log_uses_recent_pending_slot_and_cancels_old_jitter():
    tracker = OwnActionTracker()
    tracker.last_hand = ["clone", "giant-snowball", None, "zap"]
    tracker.last_elixir = 8.0
    tracker.pending = [
        PendingOwnPlay(
            card="log",
            slot_idx=1,
            started_at_s=102.0,
            elixir_before=8.0,
        ),
        PendingOwnPlay(
            card="log",
            slot_idx=2,
            started_at_s=100.0,
            elixir_before=8.0,
        ),
    ]

    tracker.update(
        _state(
            hand=["clone", "giant-snowball", None, "zap"],
            elixir=6.0,
            own_units=[_own_log(track_id=12)],
            now=100.0,
        ),
        ARENA_PX,
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "log"
    assert tracker.actions[0]["slot_idx"] == 2
    assert tracker.pending == []


def test_consumed_log_track_does_not_confirm_later_pending():
    tracker = _tracker_with_log_in_hand()

    tracker.update(
        _state(
            hand=["skeletons", "ice-spirit", "hog-rider", "musketeer"],
            elixir=3.0,
            own_units=[_own_log(track_id=12)],
            now=100.0,
        ),
        ARENA_PX,
    )
    tracker.pending.append(PendingOwnPlay(
        card="log",
        slot_idx=0,
        started_at_s=99.8,
        elixir_before=5.0,
        spell_elixir_confirmed=True,
    ))
    tracker.update(
        _state(
            hand=["skeletons", "ice-spirit", "hog-rider", "musketeer"],
            elixir=3.0,
            own_units=[_own_log(track_id=12)],
            now=99.8,
        ),
        ARENA_PX,
    )

    assert len(tracker.actions) == 1


class _FakeSpellDeployLocator:
    def locate(self, frame, arena_px, card_name, elixir_cost):
        return None

    def locate_released(self, frame, arena_px, card_name, elixir_cost):
        return SimpleNamespace(center_x=284.0, center_y=500.0)


def test_log_elixir_drop_is_not_reused_by_stale_clone_pending():
    tracker = OwnActionTracker()
    tracker.last_hand = ["clone", "giant-snowball", None, "zap"]
    tracker.last_elixir = 8.0
    tracker.spell_deploy_locator = _FakeSpellDeployLocator()
    tracker.pending = [
        PendingOwnPlay(
            card="clone",
            slot_idx=3,
            started_at_s=102.0,
            elixir_before=8.0,
        ),
        PendingOwnPlay(
            card="log",
            slot_idx=2,
            started_at_s=100.0,
            elixir_before=8.0,
        ),
    ]

    tracker.update(
        _state(
            hand=["clone", "giant-snowball", None, "zap"],
            elixir=6.0,
            own_units=[_own_log(track_id=12)],
            now=100.0,
        ),
        ARENA_PX,
        frame=object(),
    )

    assert len(tracker.actions) == 1
    assert tracker.actions[0]["card"] == "log"
    assert tracker.actions[0]["slot_idx"] == 2

    tracker.update(
        _state(
            hand=["clone", "giant-snowball", None, "zap"],
            elixir=6.0,
            own_units=[_own_log(track_id=12)],
            now=99.8,
        ),
        ARENA_PX,
        frame=object(),
    )
    assert len(tracker.actions) == 1

    tracker.update(
        _state(
            hand=["clone", "giant-snowball", None, "zap"],
            elixir=3.0,
            own_units=[_own_log(track_id=12)],
            now=99.6,
        ),
        ARENA_PX,
        frame=object(),
    )
    assert [action["card"] for action in tracker.actions] == ["log", "clone"]
