"""Own-action tracker must see the RAW hand, not the filtered hand.

The hand-state filter deliberately hides the brief empty-slot transition of
a played card (it re-emits the previous card for policy stability). The
tracker creates pending plays exactly from that transition, so
MatchSession must feed it the raw hand while policy/display keep the
filtered one. Synthetic observations, no models (cv2 stubbed).

The tracker additionally scrubs the raw hand against the session deck
(HOG_26_CYCLE_DECK): the classifier labels empty/dealing slots with
arbitrary card names ("skeleton-dragons" sliding in), and anything outside
the deck reads as an empty slot so the card -> None drop edge fires.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

sys.modules.setdefault("cv2", MagicMock())

from cr_bot.app.match_session import MatchSession, tracker_hand_card
from cr_bot.domain.game_state import GameState, HudState, PrincessTowerState


def _game_state(hand, own_units=None, elixir=5.0):
    return GameState(
        hud=HudState(
            time_left_s=200.0,
            overtime=False,
            elixir_self=elixir,
            hand_cards=list(hand),
            next_card="giant",
            tower_hp_self=[4424.0, 7032.0, 4424.0],
            tower_hp_enemy=[4424.0, 7032.0, 4424.0],
            princess_towers=PrincessTowerState(
                own_left_alive=True,
                own_right_alive=True,
                enemy_left_alive=True,
                enemy_right_alive=True,
            ),
        ),
        total_remaining_s=200.0,
        own_units=list(own_units) if own_units else [],
        enemy_units=[],
        seen_enemy_cards=[],
        elixir_enemy_est=0.0,
        own_king_active=False,
        enemy_king_active=False,
        started=True,
    )


def _analysis():
    return SimpleNamespace(
        arena_px=(0, 0, 1080, 2400),
        clock_boxes=[],
        emote_boxes=[],
        elixir_change=None,
    )


def _analysis(clock_boxes=None, elixir_change=None):
    return SimpleNamespace(
        arena_px=(0, 0, 1080, 2400),
        clock_boxes=list(clock_boxes) if clock_boxes else [],
        emote_boxes=[],
        elixir_change=elixir_change,
    )


def _raw_hand(first):
    return {
        "card_1": first,
        "card_2": ("musketeer", 90.0),
        "card_3": ("cannon", 90.0),
        "card_4": ("ice-spirit", 90.0),
        "next_card": ("fireball", 90.0),
    }


def _primed_session():
    session = MatchSession(tracker_debug=False)
    tracker = session.own_action_tracker
    tracker.last_hand = ["hog-rider", "musketeer", "cannon", "ice-spirit"]
    tracker.last_elixir = 5.0
    return session


def test_drop_visible_only_in_raw_hand_creates_pending():
    session = _primed_session()
    # Filtered view (policy/display): slot still shows the old card.
    game_state = _game_state(["hog-rider", "musketeer", "cannon", "ice-spirit"])
    # Raw extractor view: slot 0 went empty (card was played).
    raw = _raw_hand(None)
    session._update_own_actions(
        game_state, _analysis(), frame=None, now_s=200.0, raw_hand_state=raw
    )
    assert [p.card for p in session.own_action_tracker.pending] == ["hog-rider"]
    # The policy-facing hand is untouched.
    assert game_state.hud.hand_cards[0] == "hog-rider"


def test_stable_raw_hand_creates_nothing():
    session = _primed_session()
    game_state = _game_state(["hog-rider", "musketeer", "cannon", "ice-spirit"])
    raw = _raw_hand(("hog-rider", 92.0))
    session._update_own_actions(
        game_state, _analysis(), frame=None, now_s=200.0, raw_hand_state=raw
    )
    assert session.own_action_tracker.pending == []
    assert session.own_action_tracker.actions == []


def test_missing_raw_hand_falls_back_to_filtered():
    session = _primed_session()
    game_state = _game_state(["hog-rider", "musketeer", "cannon", "ice-spirit"])
    session._update_own_actions(
        game_state, _analysis(), frame=None, now_s=200.0, raw_hand_state=None
    )
    assert session.own_action_tracker.pending == []


def test_tracker_hand_card_passes_deck_members():
    for raw in (
        "hog-rider",
        "musketeer",
        "old-musketeer",
        "cannon",
        "ice-spirit",
        "ice-golem",
        "skeletons",
        "fireball",
        "log",
        "the-log",
        ("skeletons", 73.0),
        ("musketeer-ev1", 80.0),
        ("hog-rider-evolution", 80.0),
    ):
        assert tracker_hand_card(raw) is not None


def test_tracker_hand_card_rejects_off_deck_names():
    assert tracker_hand_card(None) is None
    assert tracker_hand_card(("None", 73.0)) is None
    for raw in (
        "skeleton-dragons",
        "clone",
        "archers",
        "knight",
    ):
        assert tracker_hand_card(raw) is None, raw


def test_off_deck_transition_reads_as_empty_drop():
    # Mirrors gameplay.mp4 ~7.1s: slot 0 shows skeletons, then the
    # classifier labels the dealing slot "skeleton-dragons" (never None).
    # The deck scrub turns that into an empty slot, arming the drop.
    session = MatchSession(tracker_debug=False)
    tracker = session.own_action_tracker
    tracker.last_hand = ["skeletons", "ice-spirit", "fireball", "ice-golem"]
    tracker.last_elixir = 9.0
    game_state = _game_state(
        ["skeletons", "ice-spirit", "fireball", "ice-golem"], elixir=9.0
    )
    raw = {
        "card_1": ("skeleton-dragons", 70.0),
        "card_2": ("ice-spirit", 90.0),
        "card_3": ("fireball", 90.0),
        "card_4": ("ice-golem", 90.0),
        "next_card": ("hog-rider", 90.0),
    }
    session._update_own_actions(
        game_state, _analysis(), frame=None, now_s=200.0, raw_hand_state=raw
    )
    assert [p.card for p in session.own_action_tracker.pending] == ["skeletons"]


def test_off_deck_drop_confirms_with_clock_and_elixir():
    # End-to-end replay of the 7.1s skeletons play: drop via deck scrub,
    # numeric elixir drop 9.1 -> 7.7 (>= 0.5 for cost 1), ally deploy clock
    # next to fresh skeleton tracks -> confirmed action with a cell.
    session = MatchSession(tracker_debug=False)
    tracker = session.own_action_tracker
    tracker.last_hand = ["skeletons", "ice-spirit", "fireball", "ice-golem"]
    tracker.last_elixir = 9.1
    raw = {
        "card_1": ("skeleton-dragons", 70.0),
        "card_2": ("ice-spirit", 90.0),
        "card_3": ("fireball", 90.0),
        "card_4": ("ice-golem", 90.0),
        "next_card": ("hog-rider", 90.0),
    }
    game_state = _game_state(
        ["skeletons", "ice-spirit", "fireball", "ice-golem"], elixir=9.1
    )
    session._update_own_actions(
        game_state, _analysis(), frame=None, now_s=200.0, raw_hand_state=raw
    )
    assert [p.card for p in session.own_action_tracker.pending] == ["skeletons"]

    track = SimpleNamespace(
        troop=SimpleNamespace(
            class_name="skeleton",
            team="ally",
            track_id=13,
            center_x=472.0,
            center_y=1771.0,
            confidence=0.9,
        )
    )
    clock = {
        "team": "ally",
        "confidence": 0.928,
        "center_x": 514.0,
        "center_y": 1777.0,
    }
    game_state = _game_state(
        ["hog-rider", "ice-spirit", "fireball", "ice-golem"],
        own_units=[track],
        elixir=7.7,
    )
    session._update_own_actions(
        game_state,
        _analysis(clock_boxes=[clock], elixir_change={"covered": True}),
        frame=None,
        now_s=200.0,
        raw_hand_state=raw,
    )
    assert [a.card for a in session.own_action_tracker.actions] == ["skeletons"]
    assert session.own_action_tracker.actions[0].cell is not None
