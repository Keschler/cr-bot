from __future__ import annotations

import re
import sys
from pathlib import Path
import cv2


CAPTURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPTURE_ROOT))

from constants import (
    CARD_CONFIDENCE_THRESHOLD,
    GAME_START_MIN_SECONDS,
    IN_GAME_SCORE_THRESHOLD,
    MAX_ELIXIR,
    HAND_CARD_CONFIDENCE_THRESHOLD,
)
from extractors.cards import extract_hand_state
from extractors.elixir import extract_elixir
from extractors.timer import extract_time
from image_utils import detect_if_king_tower_activated, detect_if_support_tower_alive


_TIMER_RE = re.compile(r"^\d{1,2}:\d{2}$")


def _parse_timer(timer_text: str) -> int | None:
    text = str(timer_text).strip()
    if not _TIMER_RE.fullmatch(text):
        return None

    minutes_str, seconds_str = text.split(":", 1)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    if seconds >= 60 or minutes > 3:
        return None
    return minutes * 60 + seconds


def _is_confident_card(card, min_confidence: float) -> bool:
    if not isinstance(card, (tuple, list)) or len(card) != 2:
        return False

    name, confidence = card
    return isinstance(name, str) and bool(name) and float(confidence) >= min_confidence


def _count_confident_hand_cards(hand_state: dict) -> int:
    hand_slots = ("card_1", "card_2", "card_3", "card_4")
    return sum(_is_confident_card(hand_state.get(slot), HAND_CARD_CONFIDENCE_THRESHOLD) for slot in hand_slots)


def _count_distinct_hand_cards(hand_state: dict) -> int:
    names = set()
    for slot in ("card_1", "card_2", "card_3", "card_4"):
        card = hand_state.get(slot)
        if _is_confident_card(card, CARD_CONFIDENCE_THRESHOLD):
            names.add(card[0])
    return len(names)


def _has_plausible_elixir(frame) -> bool:
    elixir = extract_elixir(frame)
    digit = elixir["displayed_digit"]
    estimated_value = float(elixir["estimated_value"]) + float(digit)
    return 0.0 <= estimated_value <= MAX_ELIXIR


def _has_visible_tower_bars(frame) -> bool:
    king_state = detect_if_king_tower_activated(frame)
    support_state = detect_if_support_tower_alive(frame)

    support_visible = sum(bool(value) for value in support_state.values())
    king_visible = sum(bool(value) for value in king_state.values())

    return support_visible >= 2 or (support_visible >= 1 and king_visible >= 1)


def _extract_non_card_signals(frame) -> dict:
    timer_text = extract_time(frame)
    timer_seconds = _parse_timer(timer_text)

    elixir = extract_elixir(frame)
    elixir_digit = elixir["displayed_digit"]
    estimated_elixir = float(elixir["estimated_value"]) + float(elixir_digit)
    plausible_elixir = 0.0 <= estimated_elixir <= MAX_ELIXIR

    king_state = detect_if_king_tower_activated(frame)
    support_state = detect_if_support_tower_alive(frame)
    support_visible = sum(bool(value) for value in support_state.values())
    king_visible = sum(bool(value) for value in king_state.values())
    visible_tower_bars = support_visible >= 2 or (support_visible >= 1 and king_visible >= 1)

    return {
        "timer_text": timer_text,
        "timer_seconds": timer_seconds,
        "elixir_digit": elixir_digit,
        "estimated_elixir": estimated_elixir,
        "plausible_elixir": plausible_elixir,
        "king_state": king_state,
        "support_state": support_state,
        "support_visible": support_visible,
        "king_visible": king_visible,
        "visible_tower_bars": visible_tower_bars,
    }


def in_game_debug(frame) -> dict:
    non_card = _extract_non_card_signals(frame)
    hand_state = extract_hand_state(frame)
    confident_hand_cards = _count_confident_hand_cards(hand_state)
    distinct_hand_cards = _count_distinct_hand_cards(hand_state)
    next_card_confident = _is_confident_card(hand_state.get("next_card"), CARD_CONFIDENCE_THRESHOLD)

    score = 0
    score_reasons = []
    if confident_hand_cards >= 4:
        score += 2
        score_reasons.append("4 confident hand cards")
    elif confident_hand_cards >= 3 and next_card_confident:
        score += 1
        score_reasons.append("3 confident hand cards + confident next card")

    if distinct_hand_cards >= 3:
        score += 1
        score_reasons.append("3 distinct hand cards")

    if non_card["plausible_elixir"]:
        score += 1
        score_reasons.append("plausible elixir")

    if non_card["visible_tower_bars"]:
        score += 1
        score_reasons.append("visible tower bars")

    return {
        **non_card,
        "hand_state": hand_state,
        "confident_hand_cards": confident_hand_cards,
        "distinct_hand_cards": distinct_hand_cards,
        "next_card_confident": next_card_confident,
        "score": score,
        "score_reasons": score_reasons,
        "threshold": IN_GAME_SCORE_THRESHOLD,
        "in_game": non_card["timer_seconds"] is not None and score >= IN_GAME_SCORE_THRESHOLD,
    }


def game_start(frame) -> bool:
    timer_seconds = _parse_timer(extract_time(frame))
    if timer_seconds is None:
        return False
    return in_game(frame) and timer_seconds >= GAME_START_MIN_SECONDS

def game_end(frame):
    return not in_game(frame)


def in_game(frame) -> bool:
    non_card = _extract_non_card_signals(frame)
    if non_card["timer_seconds"] is None:
        return False

    cheap_score = int(non_card["plausible_elixir"]) + int(non_card["visible_tower_bars"])
    if cheap_score + 3 < IN_GAME_SCORE_THRESHOLD:
        return False

    hand_state = extract_hand_state(frame)
    confident_hand_cards = _count_confident_hand_cards(hand_state)

    score = cheap_score
    if confident_hand_cards >= 4:
        score += 2
    elif confident_hand_cards >= 3 and _is_confident_card(hand_state.get("next_card"), CARD_CONFIDENCE_THRESHOLD):
        score += 1

    if score + 1 < IN_GAME_SCORE_THRESHOLD:
        return False

    score += int(_count_distinct_hand_cards(hand_state) >= 3)
    return score >= IN_GAME_SCORE_THRESHOLD

def game_end_from_result(result) -> bool:
    timer_seconds = _parse_timer(result["time"])
    if timer_seconds is None:
      return True

    elixir = result["elixir"]
    digit = elixir["displayed_digit"]
    estimated_elixir = float(elixir["estimated_value"]) + float(digit)
    plausible_elixir = 0.0 <= estimated_elixir <= MAX_ELIXIR

    hand_state = result["state"]
    confident_hand_cards = _count_confident_hand_cards(hand_state)
    distinct_hand_cards = _count_distinct_hand_cards(hand_state)
    next_card_confident = _is_confident_card(
      hand_state.get("next_card"),
      CARD_CONFIDENCE_THRESHOLD,
    )

    visible_tower_bars = any(hp is not None and hp > 0 for hp in result["towers_hp"].values())

    score = 0
    if confident_hand_cards >= 4:
      score += 2
    elif confident_hand_cards >= 3 and next_card_confident:
      score += 1

    if distinct_hand_cards >= 3:
      score += 1

    if plausible_elixir:
      score += 1

    if visible_tower_bars:
      score += 1

    return score < IN_GAME_SCORE_THRESHOLD



if __name__ == "__main__":
    frame1 = cv2.imread("/home/keschler/Documents/Coding/python/cr-bot/dataset_generation/data/frame_states/clip/frames3/010280.jpg")
    frame2 = cv2.imread("/home/keschler/Documents/Coding/python/cr-bot/dataset_generation/data/frame_states/clip/frames3/000000.jpg")
    frame3 = cv2.imread("/home/keschler/Documents/Coding/python/cr-bot/dataset_generation/data/frame_states/clip/frames4/000300.jpg")
    frame4 = cv2.imread("/home/keschler/Documents/Coding/python/cr-bot/dataset_generation/data/frame_states/clip/frames3/000080.jpg")
    for name, frame in (
        ("FRAME 1", frame1),
        ("FRAME 2", frame2),
        ("FRAME 3", frame3),
        ("FRAME_4", frame4)
    ):
        debug = in_game_debug(frame)
        print(f"{name}: {debug['in_game']}")
        print(f"  timer={debug['timer_text']!r} parsed={debug['timer_seconds']}")
        print(f"  score={debug['score']}/{debug['threshold']} reasons={debug['score_reasons']}")
        print(f"  confident_hand_cards={debug['confident_hand_cards']} distinct={debug['distinct_hand_cards']} next={debug['next_card_confident']}")
        print(f"  elixir digit={debug['elixir_digit']} est={debug['estimated_elixir']:.2f} plausible={debug['plausible_elixir']}")
        print(f"  king_state={debug['king_state']}")
        print(f"  support_state={debug['support_state']}")
        print()
    print(game_start(frame4))
