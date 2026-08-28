from __future__ import annotations

import numpy as np
import pytest

from cr_bot.domain.card_metadata import CARD_METADATA
from simulator.engine import BattleEngine
from simulator.env import SimulatorEnv
from simulator.geometry import mirror_position
from simulator.observation_v2 import ENTITY_TOKEN_FEATURES, PolicyObservationV2
from simulator.observation_v2_adapter import (
    build_policy_observation_v2,
    build_public_entity_rows,
)
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import StatusState


RULESET = load_ruleset("v1")
FEATURE = {name: index for index, name in enumerate(ENTITY_TOKEN_FEATURES)}


def _state():
    engine = BattleEngine(RULESET)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=1927, shuffle_decks=False)
    return engine, state


def _spawn(engine, state, card_id: str, owner: int, x: int, y: int):
    return engine._spawn_single_at(
        state,
        RULESET.card(card_id),
        owner=owner,
        x_mtile=x,
        y_mtile=y,
        deploy_remaining_us=0,
    )


def _card_value(card_id: str) -> float:
    max_card_id = max(int(metadata["id"]) for metadata in CARD_METADATA.values())
    return float(CARD_METADATA[card_id]["id"]) / max_card_id


def test_adapter_returns_public_normalized_rows() -> None:
    engine, state = _state()
    _spawn(engine, state, "hog-rider", 0, 4_000, 23_000)
    _spawn(engine, state, "musketeer", 1, 13_000, 8_000)

    observation = build_policy_observation_v2(state, RULESET, viewer=0)

    assert isinstance(observation, PolicyObservationV2)
    assert observation.entity_mask[:2].tolist() == [True, True]
    assert int(observation.entity_mask.sum()) == 2
    rows = observation.entity_tokens[:2]
    assert rows.shape == (2, 32)
    assert np.isfinite(rows).all()
    assert np.all((rows >= 0.0) & (rows <= 1.0))
    assert rows[0, FEATURE["card_id"]] == pytest.approx(_card_value("hog-rider"))
    assert rows[0, FEATURE["side"]] == 0.0
    assert rows[1, FEATURE["side"]] == 1.0
    assert rows[:, FEATURE["is_visible"]].tolist() == [1.0, 1.0]
    assert rows[:, FEATURE["is_targetable"]].tolist() == [1.0, 1.0]
    assert rows[:, FEATURE["confidence"]].tolist() == [1.0, 1.0]
    assert np.all(rows[:, FEATURE["state_invisible"]] == 0.0)
    assert np.all(rows[:, FEATURE["has_target"]] == 0.0)


def test_hidden_entities_are_removed_before_v1_and_v2_projection() -> None:
    engine, state = _state()
    _spawn(engine, state, "hog-rider", 0, 4_000, 23_000)
    royal_ghost = _spawn(engine, state, "royal-ghost", 1, 13_000, 8_000)
    miner = _spawn(engine, state, "miner", 1, 12_000, 9_000)
    tesla = _spawn(engine, state, "tesla", 1, 11_000, 10_000)
    own_hidden = _spawn(engine, state, "royal-ghost", 0, 5_000, 22_000)
    status_hidden = _spawn(engine, state, "musketeer", 1, 10_000, 11_000)
    status_hidden.statuses.append(StatusState(kind="invisibility", remaining_us=100_000))

    assert royal_ghost.stealth_active
    assert miner.burrow_active
    assert tesla.concealed_active
    assert own_hidden.stealth_active

    observation = build_policy_observation_v2(state, RULESET)

    assert int(observation.entity_mask.sum()) == 1
    visible_card_value = observation.entity_tokens[0, FEATURE["card_id"]]
    assert visible_card_value == pytest.approx(_card_value("hog-rider"))
    # In particular, hidden enemy forms never become public rows merely
    # because the authoritative simulator knows their exact card identity.
    assert not any(
        visible_card_value == pytest.approx(_card_value(card_id))
        for card_id in ("royal-ghost", "miner", "tesla")
    )


def test_viewer_one_uses_the_existing_rotated_public_frame() -> None:
    engine, state = _state()
    # The legacy public DTO quantizes positions to action-grid cell centers.
    # Use cell centers here so the two viewer-local projections test the
    # intended exact mirror relation rather than a one-cell rounding artifact
    # for an off-center authoritative position.
    own = _spawn(engine, state, "hog-rider", 0, 4_500, 23_500)
    enemy = _spawn(engine, state, "musketeer", 1, 12_500, 8_500)

    viewer_zero = build_policy_observation_v2(state, RULESET, viewer=0)
    viewer_one = build_policy_observation_v2(state, RULESET, viewer=1)

    # Rows are sorted by public track ID, so the same entity has the same row
    # index in both viewer-local observations.
    own_zero = viewer_zero.entity_tokens[0]
    own_one = viewer_one.entity_tokens[0]
    assert own_zero[FEATURE["card_id"]] == pytest.approx(own_one[FEATURE["card_id"]])
    assert own_zero[FEATURE["side"]] == 0.0
    assert own_one[FEATURE["side"]] == 1.0
    assert own_zero[FEATURE["x"]] + own_one[FEATURE["x"]] == pytest.approx(1.0)
    assert own_zero[FEATURE["y"]] + own_one[FEATURE["y"]] == pytest.approx(1.0)
    assert own_zero[FEATURE["distance_to_own_tower"]] == pytest.approx(
        own_one[FEATURE["distance_to_enemy_tower"]]
    )
    assert own_zero[FEATURE["distance_to_enemy_tower"]] == pytest.approx(
        own_one[FEATURE["distance_to_own_tower"]]
    )

    # Make the purpose of the IDs explicit and ensure the fixture was not
    # accidentally reordered by the projection.
    assert own.uid < enemy.uid


def test_private_simulator_fields_are_not_encoded_as_public_features() -> None:
    engine, state = _state()
    own = _spawn(engine, state, "hog-rider", 0, 4_000, 23_000)
    enemy = _spawn(engine, state, "musketeer", 1, 13_000, 8_000)
    enemy.target_uid = own.uid
    enemy.attack_cooldown_us = 999_999
    state.players[1].elixir_milli = 0

    observation = build_policy_observation_v2(state, RULESET)
    enemy_row = observation.entity_tokens[1]

    assert enemy_row[FEATURE["has_target"]] == 0.0
    assert enemy_row[FEATURE["velocity_x"]] == 0.0
    assert enemy_row[FEATURE["velocity_y"]] == 0.0
    assert enemy_row[FEATURE["recent_damage"]] == 0.0
    assert enemy_row[FEATURE["recent_deploy"]] == 0.0
    assert enemy_row[FEATURE["age"]] == 0.0
    expected_public_estimate = RULESET.match.initial_elixir_milli / RULESET.match.max_elixir_milli
    assert observation.global_vector[1] == pytest.approx(expected_public_estimate)


def test_direct_public_game_state_rows_do_not_require_authoritative_state() -> None:
    engine, state = _state()
    _spawn(engine, state, "hog-rider", 0, 4_000, 23_000)
    observation = build_policy_observation_v2(state, RULESET)

    rows = build_public_entity_rows(
        # This is the public DTO produced by the adapter itself.  The helper
        # must not need BattleState to normalize it a second time.
        __import__("simulator.observation", fromlist=["battle_state_to_observed_game_state"])
        .battle_state_to_observed_game_state(
            state,
            RULESET,
            viewer=0,
            memory=__import__("simulator.observation", fromlist=["ObservationMemory"])
            .ObservationMemory(viewer=0),
        ),
        viewer=0,
    )

    np.testing.assert_array_equal(rows, observation.entity_tokens[:1])


def test_invalid_viewer_and_uid_allow_list_fail_closed() -> None:
    engine, state = _state()
    _spawn(engine, state, "hog-rider", 0, 4_000, 23_000)
    with pytest.raises(ValueError, match="viewer"):
        build_policy_observation_v2(state, RULESET, viewer=2)
    with pytest.raises(TypeError, match="non-negative integers"):
        build_public_entity_rows(
            __import__("simulator.observation", fromlist=["battle_state_to_observed_game_state"])
            .battle_state_to_observed_game_state(
                state,
                RULESET,
                viewer=0,
                memory=__import__("simulator.observation", fromlist=["ObservationMemory"])
                .ObservationMemory(viewer=0),
            ),
            public_entity_uids=[True],
        )


def test_environment_exposes_v2_as_an_explicit_opt_in_boundary() -> None:
    environment = SimulatorEnv()
    environment.reset(seed=23, shuffle_decks=False)

    observations = environment.observe_v2()

    assert len(observations) == 2
    assert all(isinstance(item, PolicyObservationV2) for item in observations)
    assert all(int(item.entity_mask.sum()) == 0 for item in observations)
