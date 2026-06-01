import pytest


pytest.importorskip("cv2")

from cr_bot.vision.match_state import game_end_from_result


def in_game_result(*, time, time_left_s):
    return {
        "time": time,
        "time_left_s": time_left_s,
        "elixir": {
            "displayed_digit": 5,
            "estimated_value": 0.0,
        },
        "state": {
            "card_1": ("ice-golem", 99.0),
            "card_2": ("fireball", 99.0),
            "card_3": ("log", 99.0),
            "card_4": ("hog-rider", 99.0),
            "next_card": ("musketeer", 99.0),
        },
        "towers_hp": {
            "own_support_left": 4424,
        },
    }


def test_timer_ocr_gap_does_not_end_game_while_filtered_clock_is_positive():
    result = in_game_result(time="0:76", time_left_s=30.5)

    assert not game_end_from_result(result)


def test_missing_timer_digits_do_not_end_game_while_filtered_clock_is_positive():
    result = in_game_result(time="9", time_left_s=28.5)

    assert not game_end_from_result(result)


def test_invalid_timer_can_end_game_after_filtered_clock_reaches_zero():
    result = in_game_result(time="9", time_left_s=0.0)

    assert game_end_from_result(result)


def test_visible_timer_keeps_game_active_when_hand_classification_degrades():
    result = in_game_result(time="1:28", time_left_s=88.0)
    result["state"] = {
        "card_1": ("None", 73.0),
        "card_2": ("baby-dragon", 20.0),
        "card_3": ("cannon", 44.0),
        "card_4": ("old-musketeer", 49.0),
        "next_card": ("ice-spirit", 82.0),
    }

    assert not game_end_from_result(result)


def test_filtered_timer_alone_does_not_keep_game_active_after_timer_disappears():
    result = in_game_result(time="9", time_left_s=88.0)
    result["state"] = {}

    assert game_end_from_result(result)
