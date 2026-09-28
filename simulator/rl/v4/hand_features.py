"""Encode the fixed V4 learner hand into policy input tokens.

The V4 checkpoint contract uses the legacy card IDs and class flags that were
defined for the 2.6 Hog learner deck.  Keeping that small, immutable mapping
next to the retained V4 input code avoids coupling live inference to any
retired dataset generator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from ...roster import PLAYER_DECK
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.roster import PLAYER_DECK


class HandEncodingError(ValueError):
    """Raised when a live learner hand is outside the V4 card contract."""


# (card key, legacy policy id, elixir cost, win condition, spell, building,
# swarm, air attacker).  These IDs are part of the V4 checkpoint interface.
V4_LEARNER_DECK: tuple[tuple[str, int, int, bool, bool, bool, bool, bool], ...] = (
    ("hog-rider", 49, 4, True, False, False, False, False),
    ("cannon", 114, 3, False, False, True, False, False),
    ("musketeer", 72, 4, False, False, False, False, True),
    ("skeletons", 96, 1, False, False, False, True, False),
    ("ice-golem", 51, 2, False, False, False, False, False),
    ("ice-spirit", 52, 1, False, False, False, False, False),
    ("fireball", 28, 4, False, True, False, False, False),
    ("log", 59, 2, False, True, False, False, False),
)


def build_hand_tokens(hand: Sequence[str], elixir: float) -> np.ndarray:
    """Encode four learner cards using the V4 ``[4, 16]`` layout."""

    if len(hand) != 4:
        raise HandEncodingError(f"V4 learner hand must contain four cards, got {len(hand)}")
    if not np.isfinite(float(elixir)):
        raise HandEncodingError("elixir must be finite")
    by_key = {row[0]: row for row in V4_LEARNER_DECK}
    canonical_order = {card: index for index, card in enumerate(PLAYER_DECK)}
    tokens = np.zeros((4, 16), dtype=np.float32)
    for slot, card in enumerate(hand):
        try:
            _, policy_id, cost, win, spell, building, swarm, air = by_key[card]
            cycle_position = canonical_order[card]
        except KeyError as error:
            raise HandEncodingError(
                f"hand card {card!r} is outside the V4 learner deck"
            ) from error
        affordable = cost <= float(elixir)
        tokens[slot, 0] = policy_id / 127.0
        tokens[slot, 1] = cost / 10.0
        tokens[slot, 2] = slot / 3.0
        tokens[slot, 3] = 1.0 if affordable else 0.0
        tokens[slot, 4] = 1.0 if affordable else 0.0
        tokens[slot, 5] = 1.0 if win else 0.0
        tokens[slot, 6] = 1.0 if spell else 0.0
        tokens[slot, 7] = 1.0 if building else 0.0
        tokens[slot, 8] = 1.0 if swarm else 0.0
        tokens[slot, 9] = 1.0 if air else 0.0
        tokens[slot, 10] = float(cycle_position) / 7.0
    return tokens


__all__ = ["HandEncodingError", "V4_LEARNER_DECK", "build_hand_tokens"]
