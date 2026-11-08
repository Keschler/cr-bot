"""Regression tests for the V4 teacher's spatial and air-defense semantics."""

from __future__ import annotations

import numpy as np

from simulator.rl.v4.hand_features import build_hand_tokens
from simulator.rl.v4.simulator_teacher import teacher_label


def _entities(*positions: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Build enemy entity rows from ``(y, x, is_air)`` positions."""

    tokens = np.zeros((128, 32), dtype=np.float32)
    mask = np.zeros((128,), dtype=bool)
    for index, (y, x, is_air) in enumerate(positions):
        tokens[index, 1:7] = [1.0, x, y, 1.0, is_air, 0.0]
        mask[index] = True
    return tokens, mask


def test_spell_target_uses_local_cluster_with_a_distant_third_enemy() -> None:
    hand = build_hand_tokens(
        ["fireball", "skeletons", "ice-golem", "ice-spirit"], 4.0
    )
    entities, entity_mask = _entities(
        (0.20, 0.20, 0.0),
        (0.20, 0.22, 0.0),
        (0.30, 0.80, 0.0),
    )
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[0] = True

    target = teacher_label(
        hand_tokens=hand,
        entity_tokens=entities,
        entity_mask=entity_mask,
        legal_play=legal,
        legal_wait=True,
        own_elixir=4.0,
    )

    assert (target.mode, target.card_slot) == (1, 0)
    # The close pair centers at (y=.20, x=.21) -> (row=6, col=3).
    assert (target.row, target.col) == (6, 3)


def test_air_threat_selects_ice_spirit_over_ground_only_cards() -> None:
    hand = build_hand_tokens(
        ["skeletons", "ice-spirit", "ice-golem", "fireball"], 2.0
    )
    entities, entity_mask = _entities((0.75, 0.50, 1.0))
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[:3] = True

    target = teacher_label(
        hand_tokens=hand,
        entity_tokens=entities,
        entity_mask=entity_mask,
        legal_play=legal,
        legal_wait=True,
        own_elixir=2.0,
    )

    assert hand[1, 9] == 1.0  # Ice Spirit is an air attacker.
    assert (target.mode, target.card_slot) == (1, 1)
    assert (target.row, target.col) == (26, 9)
