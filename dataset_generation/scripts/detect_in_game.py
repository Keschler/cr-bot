from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = ROOT / "capture"
sys.path.insert(0, str(CAPTURE_ROOT))
os.chdir(CAPTURE_ROOT)

from constants import (
    CARD_CONFIDENCE_THRESHOLD,
    ELIXIR_DIGIT_SCORE_THRESHOLD,
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
    digit, digit_score = elixir["displayed_digit"]
    estimated_value = float(elixir["estimated_value"]) + float(digit)
    return digit_score >= ELIXIR_DIGIT_SCORE_THRESHOLD and 0.0 <= estimated_value <= MAX_ELIXIR


def _has_visible_tower_bars(frame) -> bool:
    king_state = detect_if_king_tower_activated(frame)
    support_state = detect_if_support_tower_alive(frame)

    support_visible = sum(bool(value) for value in support_state.values())
    king_visible = sum(bool(value) for value in king_state.values())

    return support_visible >= 2 or (support_visible >= 1 and king_visible >= 1)


def game_start(frame) -> bool:
    timer_seconds = _parse_timer(extract_time(frame))
    if timer_seconds is None:
        return False
    return in_game(frame) and timer_seconds >= GAME_START_MIN_SECONDS


def in_game(frame) -> bool:
    timer_seconds = _parse_timer(extract_time(frame))
    if timer_seconds is None:
        return False

    hand_state = extract_hand_state(frame)
    confident_hand_cards = _count_confident_hand_cards(hand_state)
    distinct_hand_cards = _count_distinct_hand_cards(hand_state)
    next_card_confident = _is_confident_card(hand_state.get("next_card"), CARD_CONFIDENCE_THRESHOLD)

    score = 0
    if confident_hand_cards >= 4:
        score += 2
    elif confident_hand_cards >= 3 and next_card_confident:
        score += 1

    if distinct_hand_cards >= 3:
        score += 1

    if _has_plausible_elixir(frame):
        score += 1

    if _has_visible_tower_bars(frame):
        score += 1

    return score >= IN_GAME_SCORE_THRESHOLD
