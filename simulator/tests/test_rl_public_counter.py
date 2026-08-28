from __future__ import annotations

import numpy as np


def _observation(
    *,
    hand: list[str],
    crossed: bool = False,
    air: bool = False,
    enemy_y: float = 0.5,
    own_cannon: bool = False,
    tower_left: float = 0.0,
    tower_right: float = 0.0,
    own_tower_left: float = 0.0,
    own_tower_right: float = 0.0,
):
    from cr_bot.domain.card_metadata import CARD_METADATA
    from cr_bot.features.channels import GLOBAL_SCALAR_IDX
    from cr_bot.features.global_features import CARD_COUNT
    from simulator.observation_v2 import (
        ENTITY_TOKEN_DIM,
        ENTITY_TOKEN_MAX,
        PolicyObservationV2,
    )

    global_vector = np.zeros((768,), dtype=np.float32)
    hand_offset = len(GLOBAL_SCALAR_IDX)
    for slot, card_name in enumerate(hand):
        global_vector[
            hand_offset + slot * CARD_COUNT + int(CARD_METADATA[card_name]["id"])
        ] = 1.0
    global_vector[GLOBAL_SCALAR_IDX["tower_hp_enemy_left"]] = tower_left
    global_vector[GLOBAL_SCALAR_IDX["tower_hp_enemy_right"]] = tower_right
    global_vector[GLOBAL_SCALAR_IDX["tower_hp_self_left"]] = own_tower_left
    global_vector[GLOBAL_SCALAR_IDX["tower_hp_self_right"]] = own_tower_right
    legal_play = np.ones((4, 32, 18), dtype=bool)
    tokens = np.zeros((ENTITY_TOKEN_MAX, ENTITY_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((ENTITY_TOKEN_MAX,), dtype=bool)
    if crossed:
        # Public entity-token indices: card_id, side, x, y, ...
        tokens[0, 1] = 1.0
        tokens[0, 2] = 0.2
        tokens[0, 3] = enemy_y
        tokens[0, 4] = 1.0
        tokens[0, 5] = float(air)
        tokens[0, 9] = 1.0
        mask[0] = True
    if own_cannon:
        max_card_id = max(
            int(metadata["id"])
            for metadata in CARD_METADATA.values()
        )
        tokens[1, 0] = float(CARD_METADATA["cannon"]["id"]) / max_card_id
        tokens[1, 4] = 1.0
        tokens[1, 6] = 1.0
        tokens[1, 9] = 1.0
        mask[1] = True
    return PolicyObservationV2(
        board=np.zeros((21, 32, 18), dtype=np.float32),
        global_vector=global_vector,
        entity_tokens=tokens,
        entity_mask=mask,
        legal_play=legal_play,
        legal_wait=True,
    )


def test_public_counter_uses_public_hand_and_legality_mask() -> None:
    from simulator.rl.public_counter import PublicCounterController

    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
    )
    controller = PublicCounterController()
    action = controller.choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 0
    assert action.cell == (3, 17)
    assert bool(observation.legal_play[action.card_idx, action.cell[1], action.cell[0]])


def test_public_counter_prioritizes_cannon_for_visible_crossed_troop() -> None:
    from simulator.rl.public_counter import PublicCounterController

    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
        crossed=True,
    )
    action = PublicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 1
    assert action.cell == (8, 21)


def test_public_counter_counts_observed_hand_changes_not_proposed_actions() -> None:
    from simulator.rl.public_counter import PublicCounterController

    controller = PublicCounterController()
    first = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
    )
    proposed = controller.choose_action(first)
    assert proposed.card_idx == 0
    assert controller.plays == 0

    # This is the public hand after some action was accepted.  The controller
    # must count the transition, even if the accepted action was chosen by an
    # actor during DAgger collection rather than by the teacher itself.
    second = _observation(
        hand=["cannon", "musketeer", "skeletons", "ice-spirit"],
    )
    controller.choose_action(second)
    assert controller.plays == 1


def test_strategic_counter_prioritizes_hog_over_opening_fireball() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    observation = _observation(
        hand=["fireball", "hog-rider", "cannon", "musketeer"],
    )
    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 1
    assert action.cell == (3, 17)


def test_strategic_counter_uses_fireball_after_public_tower_damage() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    opening = _observation(
        hand=["fireball", "hog-rider", "cannon", "musketeer"],
        tower_left=1.0,
        tower_right=1.0,
    )
    damaged = _observation(
        hand=["fireball", "hog-rider", "cannon", "musketeer"],
        tower_left=0.9,
        tower_right=1.0,
    )

    controller = StrategicCounterController()
    assert controller.choose_action(opening).card_idx == 1
    action = controller.choose_action(damaged)

    assert action.kind == "Play"
    assert action.card_idx == 0
    assert action.cell == (3, 6)


def test_strategic_counter_defends_a_crossing_before_attacking() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
        crossed=True,
    )
    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 1
    assert action.cell == (8, 21)


def test_strategic_counter_uses_musketeer_for_an_air_crossing() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
        crossed=True,
        air=True,
    )
    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 2
    assert action.cell == (3, 23)


def test_strategic_counter_answers_air_even_with_a_live_cannon() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    # A Cannon can cover a ground escort but cannot answer the air troop.  The
    # ranged answer must therefore not depend on the no-Cannon crossing path.
    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
        crossed=True,
        air=True,
        enemy_y=0.8,
        own_cannon=True,
    )
    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 2
    assert action.cell == (3, 23)


def test_strategic_counter_answers_a_crossed_air_threat_before_near_tower() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
        crossed=True,
        air=True,
        enemy_y=0.60,
        own_cannon=True,
    )
    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 2
    assert action.cell == (3, 23)


def test_strategic_counter_recovers_critical_tower_with_cheap_card() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    # A live Cannon does not answer every threat, and waiting for an enemy
    # token to reach the deepest threshold can lose the tower first.  The
    # public own-tower scalar must activate recovery and use a cheap stall
    # card when it is the only timely legal answer.
    observation = _observation(
        hand=["hog-rider", "cannon", "skeletons", "ice-spirit"],
        crossed=True,
        own_cannon=True,
        own_tower_left=0.50,
        own_tower_right=1.0,
    )
    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 3
    assert action.cell == (3, 22)


def test_strategic_counter_counterpushes_after_cannon_is_alive() -> None:
    from simulator.rl.public_counter import StrategicCounterController

    observation = _observation(
        hand=["hog-rider", "cannon", "musketeer", "skeletons"],
        crossed=True,
        own_cannon=True,
    )

    action = StrategicCounterController().choose_action(observation)

    assert action.kind == "Play"
    assert action.card_idx == 0
    assert action.cell == (3, 17)
