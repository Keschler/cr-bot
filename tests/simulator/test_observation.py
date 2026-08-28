from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from cr_bot.domain.game_state import Action as PolicyAction
from cr_bot.features.action_masks import get_action_mask
from cr_bot.features.board_rasterizer import build_board
from cr_bot.features.channels import DYNAMIC_CHANNEL_IDX, GLOBAL_SCALAR_IDX, STATIC_CHANNELS
from cr_bot.features.global_features import CARD_COUNT, build_global_vector
from simulator.actions import PlayCardAction, UseAbilityAction, WaitAction
from simulator.events import SimEvent
from simulator.observation import (
    ACTION_MASK_SHAPE,
    BASE_POLICY_CARD_IDS,
    BOARD_SHAPE,
    GLOBAL_VECTOR_SHAPE,
    ObservationMemory,
    PINNED_OBSERVATION_CONTRACT_HASH,
    PolicyObservationV1,
    UnsupportedPolicyFormError,
    battle_state_to_observed_game_state,
    build_policy_observation,
    calculate_observation_contract_hash,
    decode_policy_action,
    encode_sim_action,
    policy_card_id,
    policy_card_name,
)
from simulator.state import BattleState, EntityState, PlayerState


@dataclass(frozen=True)
class _Card:
    card_id: str
    kind: str
    elixir_milli: int


@dataclass(frozen=True)
class _MatchRules:
    initial_elixir_milli: int = 6_000
    max_elixir_milli: int = 10_000
    regulation_us: int = 180_000_000
    overtime_us: int = 120_000_000
    normal_elixir_interval_us: int = 2_800_000
    double_elixir_interval_us: int = 1_400_000
    triple_elixir_interval_us: int = 900_000


class _Ruleset:
    ruleset_id = "test"
    content_hash = "sha256:test-observation"
    tick_us = 100_000
    match = _MatchRules()

    def __init__(self) -> None:
        self.cards = {
            "fireball": _Card("fireball", "spell", 4_000),
            "hog-rider": _Card("hog-rider", "troop", 4_000),
            "ice-golem": _Card("ice-golem", "troop", 2_000),
            "ice-spirit": _Card("ice-spirit", "troop", 1_000),
            "log": _Card("log", "spell", 2_000),
            "musketeer": _Card("musketeer", "troop", 4_000),
            "skeletons": _Card("skeletons", "troop", 1_000),
            "cannon": _Card("cannon", "building", 3_000),
        }

    def card(self, value: str) -> _Card:
        key = value.strip().lower().replace("_", "-").replace(" ", "-")
        if key == "the-log":
            key = "log"
        return self.cards[key]


RULESET = _Ruleset()


def _player(
    *,
    hand: list[str] | None = None,
    draw_pile: list[str] | None = None,
    elixir_milli: int = 6_000,
) -> PlayerState:
    deck = (
        "hog-rider",
        "musketeer",
        "ice-golem",
        "ice-spirit",
        "cannon",
        "skeletons",
        "fireball",
        "log",
    )
    return PlayerState(
        deck=deck,
        hand=list(hand or ["hog-rider", "cannon", "fireball", "the-log"]),
        draw_pile=list(draw_pile or ["musketeer", "ice-golem", "ice-spirit", "skeletons"]),
        elixir_milli=elixir_milli,
    )


def _entity(
    uid: int,
    card_id: str,
    owner: int,
    kind: str,
    x_mtile: int,
    y_mtile: int,
    hp: int,
    max_hp: int,
    *,
    role: str | None = None,
) -> EntityState:
    return EntityState(
        uid=uid,
        card_id=card_id,
        owner=owner,
        kind=kind,
        x_mtile=x_mtile,
        y_mtile=y_mtile,
        hp=hp,
        max_hp=max_hp,
        spawn_tick=0,
        role=role,
    )


def _tower_entities() -> dict[int, EntityState]:
    return {
        1: _entity(1, "princess-tower", 0, "tower", 3_500, 25_500, 3_052, 3_052, role="left"),
        2: _entity(2, "king-tower", 0, "tower", 9_000, 28_500, 4_824, 4_824, role="king"),
        3: _entity(3, "princess-tower", 0, "tower", 14_500, 25_500, 3_052, 3_052, role="right"),
        4: _entity(4, "princess-tower", 1, "tower", 14_500, 6_500, 3_052, 3_052, role="left"),
        5: _entity(5, "king-tower", 1, "tower", 9_000, 3_500, 4_824, 4_824, role="king"),
        6: _entity(6, "princess-tower", 1, "tower", 3_500, 6_500, 3_052, 3_052, role="right"),
    }


def _state(
    *,
    elapsed_us: int = 2_800_000,
    tick: int = 28,
    events: list[SimEvent] | None = None,
    event_sequence: int | None = None,
    player0: PlayerState | None = None,
    player1: PlayerState | None = None,
    extra_entities: list[EntityState] | None = None,
) -> BattleState:
    entities = _tower_entities()
    for entity in extra_entities or []:
        entities[entity.uid] = entity
    event_rows = list(events or [])
    return BattleState(
        schema_version=1,
        engine_version="test-engine",
        ruleset_id="test",
        ruleset_hash=RULESET.content_hash,
        seed=7,
        rng_state=11,
        tick=tick,
        elapsed_us=elapsed_us,
        phase="regulation",
        players=[player0 or _player(), player1 or _player(elixir_milli=123)],
        entities=entities,
        projectiles={},
        next_uid=max(entities) + 1,
        event_sequence=(len(event_rows) if event_sequence is None else event_sequence),
        events=event_rows,
    )


def test_policy_card_ids_are_stable_and_log_alias_is_explicit() -> None:
    assert calculate_observation_contract_hash() == PINNED_OBSERVATION_CONTRACT_HASH
    assert BASE_POLICY_CARD_IDS == {
        "fireball": 28,
        "hog-rider": 49,
        "ice-golem": 51,
        "ice-spirit": 52,
        "log": 59,
        "musketeer": 72,
        "skeletons": 96,
        "cannon": 114,
    }
    assert policy_card_name("The Log") == "log"
    assert policy_card_id("the-log") == 59
    assert policy_card_name("Hero Musketeer") is None
    assert policy_card_id("evo-cannon") is None


def test_exact_mode_has_declared_shapes_dtypes_and_legacy_builder_parity() -> None:
    state = _state(
        player0=_player(elixir_milli=3_000),
        extra_entities=[
            _entity(7, "hog-rider", 1, "troop", 4_500, 12_500, 848, 1_696),
            _entity(8, "musketeer", 0, "troop", 12_500, 22_500, 720, 720),
        ],
    )
    memory = ObservationMemory(viewer=0)
    observation = build_policy_observation(state, RULESET, memory=memory)
    observed = battle_state_to_observed_game_state(state, RULESET, viewer=0, memory=memory)

    assert observation.board.shape == BOARD_SHAPE
    assert observation.global_vector.shape == GLOBAL_VECTOR_SHAPE
    assert observation.spatial_masks.shape == ACTION_MASK_SHAPE
    assert observation.legal_play.shape == ACTION_MASK_SHAPE
    assert observation.board.dtype == np.float32
    assert observation.global_vector.dtype == np.float32
    assert observation.spatial_masks.dtype == np.bool_
    assert observation.legal_play.dtype == np.bool_
    assert not observation.board.flags.writeable
    assert not observation.legal_play.flags.writeable

    np.testing.assert_array_equal(observation.board, build_board(observed, (0.0, 0.0, 1.0, 1.0)))
    np.testing.assert_array_equal(observation.global_vector, build_global_vector(observed))
    for slot, card_name in enumerate(observed.hud.hand_cards):
        np.testing.assert_array_equal(observation.spatial_masks[slot], get_action_mask(card_name, observed))

    # Spatial masks preserve the old top-left building anchor convention.
    assert observation.spatial_masks[1, 17, 1]
    # Authoritative actions use center cells, so the conservative legal mask
    # closes the row-17 footprint that would cross the river boundary.
    assert not observation.legal_play[1, 17, 1]
    assert observation.legal_play[1, 20, 8]
    # Hog and Fireball are unaffordable at three elixir; Log remains playable.
    assert not observation.legal_play[0].any()
    assert not observation.legal_play[2].any()
    assert observation.legal_play[3].any()
    assert observation.legal_wait


def test_hidden_authoritative_fields_do_not_change_policy_observation() -> None:
    play = SimEvent.create(10, 0, "card_played", player=1, card_id="hog-rider")
    visible_hog = _entity(7, "hog-rider", 1, "troop", 4_500, 12_500, 848, 1_696)
    first = _state(events=[play], event_sequence=1, extra_entities=[visible_hog])
    second = deepcopy(first)

    second.players[1].hand[:] = ["fireball", "fireball", "fireball", "fireball"]
    second.players[1].draw_pile[:] = ["cannon"]
    second.players[1].deck = tuple(reversed(second.players[1].deck))
    second.players[1].elixir_milli = 9_999
    second.players[1].seen_enemy_cards[:] = ["private-marker"]
    second.entities[7].target_uid = 2
    second.entities[7].pending_target_uid = 3
    second.entities[7].attack_cooldown_us = 987_654
    second.entities[7].windup_remaining_us = 123_456

    first_memory = ObservationMemory(viewer=0)
    second_memory = ObservationMemory(viewer=0)
    first_observation = build_policy_observation(first, RULESET, memory=first_memory)
    second_observation = build_policy_observation(second, RULESET, memory=second_memory)

    np.testing.assert_array_equal(first_observation.board, second_observation.board)
    np.testing.assert_array_equal(first_observation.global_vector, second_observation.global_vector)
    np.testing.assert_array_equal(first_observation.spatial_masks, second_observation.spatial_masks)
    np.testing.assert_array_equal(first_observation.legal_play, second_observation.legal_play)

    # Six starting elixir + one regenerated by 2.8 s - four public Hog cost.
    assert first_memory.opponent_elixir_milli_est == 3_000
    assert first_memory.seen_opponent_cards == ["hog-rider"]
    assert first_observation.global_vector[GLOBAL_SCALAR_IDX["elixir_enemy_est"]] == pytest.approx(0.3)
    seen_offset = len(GLOBAL_SCALAR_IDX) + 5 * CARD_COUNT
    assert first_observation.global_vector[seen_offset + BASE_POLICY_CARD_IDS["hog-rider"]] == 1.0


def test_public_elixir_memory_is_independent_of_observation_frequency() -> None:
    play = SimEvent.create(10, 0, "card_played", player=1, card_id="hog-rider")
    before_play = _state(elapsed_us=900_000, tick=9, events=[], event_sequence=0)
    after_play = _state(elapsed_us=2_800_000, tick=28, events=[play], event_sequence=1)

    sparse = ObservationMemory(viewer=0)
    sparse.update(after_play, RULESET)
    dense = ObservationMemory(viewer=0)
    dense.update(before_play, RULESET)
    dense.update(after_play, RULESET)

    assert dense.opponent_elixir_milli_est == sparse.opponent_elixir_milli_est == 3_000
    assert dense.seen_opponent_cards == sparse.seen_opponent_cards == ["hog-rider"]


def test_viewer_one_rotates_units_towers_and_legality_callback_cells() -> None:
    own_hog = _entity(7, "hog-rider", 0, "troop", 2_500, 20_500, 1_696, 1_696)
    state = _state(
        player1=_player(hand=["hog-rider", "cannon", "fireball", "the-log"], elixir_milli=10_000),
        extra_entities=[own_hog],
    )
    callback_actions: list[PlayCardAction] = []

    def legal_only_one_cell(_: BattleState, action: PlayCardAction) -> bool:
        callback_actions.append(action)
        return action.card_slot == 0 and action.cell == (9, 11)

    viewer_zero = build_policy_observation(state, RULESET, viewer=0)
    viewer_one = build_policy_observation(
        state,
        RULESET,
        viewer=1,
        legality_callback=legal_only_one_cell,
    )

    ally_ground = len(STATIC_CHANNELS) + DYNAMIC_CHANNEL_IDX["ally_ground_presence"]
    enemy_ground = len(STATIC_CHANNELS) + DYNAMIC_CHANNEL_IDX["enemy_ground_presence"]
    assert viewer_zero.board[ally_ground, 20, 2] > 0
    assert viewer_one.board[enemy_ground, 11, 15] > 0
    assert viewer_one.legal_play[0, 20, 8]
    assert viewer_one.legal_play.sum() == 1
    assert any(action.player == 1 and action.cell == (9, 11) for action in callback_actions)


def test_unknown_forms_fail_closed_instead_of_aliasing_to_base_cards() -> None:
    state = _state(
        player0=_player(
            hand=["hero-musketeer", "evo-cannon", "the-log", "hog-rider"],
            elixir_milli=10_000,
        ),
    )
    with pytest.raises(UnsupportedPolicyFormError, match="hero-musketeer"):
        build_policy_observation(state, RULESET)

    visible_unknown = _state(
        extra_entities=[
            _entity(7, "hero-musketeer", 1, "troop", 4_500, 12_500, 1_000, 1_000),
        ]
    )
    with pytest.raises(UnsupportedPolicyFormError, match="visible entity"):
        build_policy_observation(visible_unknown, RULESET)

    public_unknown = SimEvent.create(10, 0, "card_played", player=1, card_id="hero-musketeer")
    future_ruleset = _Ruleset()
    future_ruleset.cards["hero-musketeer"] = _Card("hero-musketeer", "troop", 4_000)
    with pytest.raises(UnsupportedPolicyFormError, match="public opponent"):
        build_policy_observation(
            _state(events=[public_unknown], event_sequence=1),
            future_ruleset,
        )


def test_non_policy_opponent_cards_and_hidden_children_share_feature_boundary() -> None:
    """All opponent entities remain visible without changing policy tensor shapes.

    The fixed player deck still rejects unsupported hand/ability forms, but an
    opponent Baby Dragon or a spawned Golemite must be renderable by the
    existing aggregate vision channels.  Child aliases affect only the lossy
    feature profile; authoritative IDs stay in ``BattleState``.
    """

    state = _state(
        extra_entities=[
            _entity(7, "baby-dragon", 1, "troop", 4_500, 12_500, 1_000, 1_000),
            _entity(8, "golemite", 1, "troop", 6_500, 12_500, 700, 1_039),
            _entity(9, "goblin-brawler", 1, "troop", 8_500, 12_500, 900, 1_080),
            _entity(10, "rascal-boy", 1, "troop", 10_500, 12_500, 1_000, 1_940),
            _entity(11, "rascal-girl", 1, "troop", 11_500, 12_500, 180, 202),
            _entity(12, "cursed-hog", 1, "troop", 12_500, 12_500, 600, 1_696),
        ]
    )
    observation = build_policy_observation(state, RULESET)
    observed = battle_state_to_observed_game_state(
        state, RULESET, viewer=0, memory=ObservationMemory(viewer=0)
    )

    assert [match.troop.class_name for match in observed.enemy_units] == [
        "baby-dragon",
        "golem",
        "goblins",
        "barbarians",
        "spear-goblins",
        "hog-rider",
    ]
    enemy_ground = len(STATIC_CHANNELS) + DYNAMIC_CHANNEL_IDX["enemy_ground_presence"]
    enemy_air = len(STATIC_CHANNELS) + DYNAMIC_CHANNEL_IDX["enemy_air_presence"]
    assert observation.board[enemy_air].any()
    assert observation.board[enemy_ground].any()


def test_eligible_non_policy_opponent_play_is_charged_and_seen() -> None:
    """The fixed player action vocabulary must not reject eligible opponents."""

    ruleset = _Ruleset()
    ruleset.cards["baby-dragon"] = _Card("baby-dragon", "troop", 4_000)
    ruleset.interaction_set = ("baby-dragon",)
    event = SimEvent.create(10, 0, "card_played", player=1, card_id="baby-dragon")
    state = _state(events=[event], event_sequence=1)

    memory = ObservationMemory(viewer=0)
    build_policy_observation(state, ruleset, memory=memory)
    observed = battle_state_to_observed_game_state(
        state, ruleset, viewer=0, memory=memory
    )

    assert memory.seen_opponent_cards == ["baby-dragon"]
    assert observed.seen_enemy_cards == [3]
    assert memory.opponent_elixir_milli_est == 3_000


def test_action_conversion_is_lossless_for_wait_and_play_from_both_viewers() -> None:
    assert decode_policy_action(PolicyAction(kind="Wait"), viewer=0) == WaitAction(0)
    for viewer in (0, 1):
        policy = PolicyAction(kind="Play", card_idx=2, cell=(4, 23))
        simulator_action = decode_policy_action(policy, viewer=viewer)
        expected_cell = (4, 23) if viewer == 0 else (13, 8)
        assert simulator_action == PlayCardAction(viewer, 2, expected_cell)
        assert encode_sim_action(simulator_action, viewer=viewer) == policy

    with pytest.raises(ValueError, match="no lossless ability"):
        encode_sim_action(UseAbilityAction(player=0, entity_uid=7), viewer=0)
    with pytest.raises(ValueError, match="player 1 action"):
        encode_sim_action(WaitAction(player=1), viewer=0)


def test_policy_observation_rejects_wrong_shapes_and_unknown_modes() -> None:
    with pytest.raises(ValueError, match="unsupported compatibility mode"):
        build_policy_observation(_state(), RULESET, compatibility_mode="future")

    with pytest.raises(ValueError, match="board must have shape"):
        PolicyObservationV1(
            board=np.zeros((1, 1, 1), dtype=np.float32),
            global_vector=np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float32),
            spatial_masks=np.zeros(ACTION_MASK_SHAPE, dtype=bool),
            legal_play=np.zeros(ACTION_MASK_SHAPE, dtype=bool),
            legal_wait=True,
        )


def test_terminal_and_pregame_phases_close_all_actions() -> None:
    terminal = _state()
    terminal.terminal = True
    terminal.phase = "finished"
    terminal_observation = build_policy_observation(terminal, RULESET)
    assert not terminal_observation.legal_play.any()
    assert not terminal_observation.legal_wait

    pregame = _state(elapsed_us=0, tick=0)
    pregame.phase = "not_started"
    pregame_observation = build_policy_observation(pregame, RULESET)
    assert not pregame_observation.legal_play.any()
    assert not pregame_observation.legal_wait


def test_observation_rejects_a_state_from_a_different_ruleset_snapshot() -> None:
    wrong_id = _state()
    wrong_id.ruleset_id = "different"
    with pytest.raises(ValueError, match="ruleset ID"):
        build_policy_observation(wrong_id, RULESET)

    wrong_hash = _state()
    wrong_hash.ruleset_hash = "sha256:different"
    with pytest.raises(ValueError, match="ruleset hash"):
        build_policy_observation(wrong_hash, RULESET)
