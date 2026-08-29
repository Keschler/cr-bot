from __future__ import annotations

import numpy as np
import pytest

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.features.channels import DYNAMIC_CHANNEL_IDX, GLOBAL_SCALAR_FEATURES, STATIC_CHANNELS
from simulator.engine import BattleEngine
from simulator.events import SimEvent
from simulator.observation import ObservationMemory, build_policy_observation
from simulator.observation_v2 import ENTITY_TOKEN_FEATURES
from simulator.observation_v2_adapter import build_policy_observation_v2
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.soa import ObservationSoA
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


def _tower_hp_indices() -> list[int]:
    return [
        GLOBAL_SCALAR_FEATURES.index("tower_hp_enemy_left"),
        GLOBAL_SCALAR_FEATURES.index("tower_hp_enemy_king"),
        GLOBAL_SCALAR_FEATURES.index("tower_hp_enemy_right"),
    ]


def _dynamic_channel(observation, name: str) -> np.ndarray:
    return observation.board[len(STATIC_CHANNELS) + DYNAMIC_CHANNEL_IDX[name]]


@pytest.mark.parametrize("use_soa", [False, True])
def test_v1_and_soa_hide_active_enemy_concealment(use_soa: bool) -> None:
    baseline_engine, baseline = _state()
    _spawn(baseline_engine, baseline, "hog-rider", 0, 4_000, 23_000)

    engine, state = _state()
    _spawn(engine, state, "hog-rider", 0, 4_000, 23_000)
    royal_ghost = _spawn(engine, state, "royal-ghost", 1, 13_000, 8_000)
    miner = _spawn(engine, state, "miner", 1, 12_000, 9_000)
    tesla = _spawn(engine, state, "tesla", 1, 11_000, 10_000)
    status_hidden = _spawn(engine, state, "musketeer", 1, 10_000, 11_000)
    status_hidden.statuses.append(StatusState(kind="invisibility", remaining_us=100_000))

    assert royal_ghost.stealth_active
    assert miner.burrow_active
    assert tesla.concealed_active

    soa_state = ObservationSoA() if use_soa else None
    expected = build_policy_observation(
        baseline,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
    )
    actual = build_policy_observation(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=soa_state,
    )

    np.testing.assert_array_equal(actual.board, expected.board)
    np.testing.assert_array_equal(actual.global_vector, expected.global_vector)


@pytest.mark.parametrize("use_soa", [False, True])
def test_hidden_enemy_play_updates_seen_card_memory(use_soa: bool) -> None:
    engine, state = _state()
    _spawn(engine, state, "royal-ghost", 1, 13_000, 8_000)
    state.events = [
        SimEvent.create(
            tick=0,
            sequence=0,
            kind="card_played",
            player=1,
            card_id="royal-ghost",
            cost_milli=3_000,
        )
    ]
    state.event_sequence = 1
    memory = ObservationMemory(viewer=0)

    observation = build_policy_observation(
        state,
        RULESET,
        viewer=0,
        memory=memory,
        soa_state=ObservationSoA() if use_soa else None,
    )

    assert int(_dynamic_channel(observation, "enemy_ground_presence").sum()) == 0
    assert memory.seen_opponent_cards == ["royal-ghost"]


@pytest.mark.parametrize("use_soa", [False, True])
def test_crown_towers_use_entity_kind_and_buildings_remain_public(use_soa: bool) -> None:
    baseline_engine, baseline = _state()
    expected = build_policy_observation(
        baseline,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
    )

    engine, state = _state()
    _spawn(engine, state, "bomb-tower", 1, 8_000, 10_000)
    _spawn(engine, state, "inferno-tower", 1, 10_000, 10_000)
    soa_state = ObservationSoA() if use_soa else None
    actual = build_policy_observation(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=soa_state,
    )

    np.testing.assert_array_equal(
        actual.global_vector[_tower_hp_indices()],
        expected.global_vector[_tower_hp_indices()],
    )
    assert _dynamic_channel(actual, "enemy_ground_presence").sum() > 0.0

    v2 = build_policy_observation_v2(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=ObservationSoA() if use_soa else None,
    )
    assert int(v2.entity_mask.sum()) == 2
    rows = v2.entity_tokens[:2]
    assert np.all(rows[:, FEATURE["is_building"]] == 1.0)
    assert np.all(rows[:, FEATURE["is_tower"]] == 0.0)
    assert {
        int(round(float(row[FEATURE["card_id"]] * max(CARD_METADATA[card]["id"] for card in CARD_METADATA))))
        for row, card in zip(rows, ("bomb-tower", "inferno-tower"), strict=True)
    } == {CARD_METADATA["bomb-tower"]["id"], CARD_METADATA["inferno-tower"]["id"]}


@pytest.mark.parametrize("use_soa", [False, True])
def test_elixir_collector_has_finite_zero_threat_observations(use_soa: bool) -> None:
    engine, state = _state()
    _spawn(engine, state, "elixir-collector", 1, 10_000, 10_000)
    soa_state = ObservationSoA() if use_soa else None

    observation = build_policy_observation(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=soa_state,
    )

    assert np.isfinite(observation.board).all()
    assert _dynamic_channel(observation, "enemy_threat_mass").sum() == pytest.approx(0.0)

    v2 = build_policy_observation_v2(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=ObservationSoA() if use_soa else None,
    )
    assert np.isfinite(v2.entity_tokens).all()
    assert int(v2.entity_mask.sum()) == 1
    assert v2.entity_tokens[0, FEATURE["is_building"]] == 1.0


@pytest.mark.parametrize("use_soa", [False, True])
def test_finished_regulation_state_does_not_report_active_overtime(use_soa: bool) -> None:
    engine, state = _state()
    state.elapsed_us = RULESET.match.regulation_us
    state.phase = "ended"
    state.terminal = True
    state.winner = 0
    state.terminal_reason = "regulation_crowns"
    state.players[0].crowns = 1

    overtime_index = GLOBAL_SCALAR_FEATURES.index("overtime")
    observation = build_policy_observation(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=ObservationSoA() if use_soa else None,
    )
    assert observation.global_vector[overtime_index] == pytest.approx(0.0)

    v2 = build_policy_observation_v2(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
        soa_state=ObservationSoA() if use_soa else None,
    )
    assert v2.global_vector[overtime_index] == pytest.approx(0.0)
