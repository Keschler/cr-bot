"""Focused public-observation contract checks for the recurrent RL prototype.

These tests deliberately construct two authoritative states that differ only
in information the viewer is not allowed to observe.  The policy boundary and
the model are then compared end-to-end.  Checkpoint metadata is covered by the
separate train/resume/evaluate integration test.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from simulator.engine import BattleEngine
from simulator.events import SimEvent
from simulator.observation import ObservationMemory
from simulator.observation_v2 import PolicyObservationV2
from simulator.observation_v2_adapter import build_policy_observation_v2
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


RULESET = load_ruleset("v1")
requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _states_with_private_difference():
    engine = BattleEngine(RULESET)
    state = engine.new_battle(
        (PLAYER_DECK, PLAYER_DECK),
        seed=1911,
        shuffle_decks=False,
    )
    own = engine._spawn_single_at(
        state,
        RULESET.card("hog-rider"),
        owner=0,
        x_mtile=4_500,
        y_mtile=23_500,
        deploy_remaining_us=0,
    )
    enemy = engine._spawn_single_at(
        state,
        RULESET.card("musketeer"),
        owner=1,
        x_mtile=12_500,
        y_mtile=8_500,
        deploy_remaining_us=0,
    )

    private_state = copy.deepcopy(state)
    private_enemy = private_state.entities[enemy.uid]
    private_state.players[1].hand = (
        private_state.players[1].hand[1:] + private_state.players[1].hand[:1]
    )
    private_state.players[1].elixir_milli = 0
    private_enemy.target_uid = own.uid
    private_enemy.pending_target_uid = own.uid
    private_enemy.attack_cooldown_us = 777_000
    private_enemy.windup_remaining_us = 111_000
    private_enemy.navigation_target_uid = own.uid
    private_enemy.navigation_revision = 42
    private_enemy.attack_count = 9

    assert private_state.players[1].hand != state.players[1].hand
    assert private_state.players[1].elixir_milli != state.players[1].elixir_milli
    assert private_enemy.target_uid == own.uid
    assert private_enemy.attack_cooldown_us != enemy.attack_cooldown_us

    return state, private_state


def _public_observation(state) -> PolicyObservationV2:
    return build_policy_observation_v2(
        state,
        RULESET,
        viewer=0,
        memory=ObservationMemory(viewer=0),
    )


def test_v2_observation_is_invariant_to_hidden_opponent_state() -> None:
    public_state, private_state = _states_with_private_difference()

    public = _public_observation(public_state)
    changed = _public_observation(private_state)

    np.testing.assert_array_equal(public.board, changed.board)
    np.testing.assert_array_equal(public.global_vector, changed.global_vector)
    np.testing.assert_array_equal(public.entity_tokens, changed.entity_tokens)
    np.testing.assert_array_equal(public.entity_mask, changed.entity_mask)
    np.testing.assert_array_equal(public.legal_play, changed.legal_play)
    assert public.legal_wait == changed.legal_wait
    assert public.schema_version == changed.schema_version
    assert public.contract_hash == changed.contract_hash

    # The public resource estimate is derived from public time/events, not the
    # authoritative opponent elixir field.
    assert public.global_vector[1] == pytest.approx(
        RULESET.match.initial_elixir_milli / RULESET.match.max_elixir_milli
    )


def test_public_elixir_memory_uses_effective_cost_and_bonus_events() -> None:
    engine = BattleEngine(RULESET)
    state = engine.new_battle(
        (PLAYER_DECK, PLAYER_DECK),
        seed=2718,
        shuffle_decks=False,
    )
    memory = ObservationMemory(viewer=0)
    memory.reset(RULESET, battle_seed=state.seed)
    state.players[1].elixir_milli = 5_000
    state.events = [
        SimEvent.create(
            0,
            1,
            "card_played",
            player=1,
            card_id="mirror",
            cost_milli=3_000,
        ),
        SimEvent.create(
            0,
            2,
            "elixir_generated",
            player=1,
            amount_milli=1_000,
        ),
    ]
    state.event_sequence = 2

    memory.update(state, RULESET)

    # The public estimate starts at 5,000, pays Mirror's effective 3,000,
    # then receives the public 1,000 bonus.  It must not charge Mirror's
    # 1,000 base-card cost or inspect the authoritative 5,000 value above.
    assert memory.opponent_elixir_milli_est == 3_000


def _model_config():
    from rl.model import ModelConfig

    return ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
    )


def _model_inputs(observation: PolicyObservationV2):
    assert torch is not None
    return (
        torch.tensor(observation.board).unsqueeze(0).unsqueeze(0),
        torch.tensor(observation.global_vector).unsqueeze(0).unsqueeze(0),
        torch.tensor(observation.entity_tokens).unsqueeze(0).unsqueeze(0),
        torch.tensor(observation.entity_mask).unsqueeze(0).unsqueeze(0),
    )


def _action_masks(observation: PolicyObservationV2):
    from rl.trajectory import ActionMasks

    assert torch is not None
    mode, card, placement = observation.structured_action_masks()
    return ActionMasks(
        mode=torch.tensor(mode).unsqueeze(0).unsqueeze(0),
        card=torch.tensor(card).unsqueeze(0).unsqueeze(0),
        placement=torch.tensor(placement).unsqueeze(0).unsqueeze(0),
    )


@requires_torch
def test_model_actor_path_is_invariant_to_hidden_opponent_state() -> None:
    from rl.model import RecurrentHybridPolicy

    public_state, private_state = _states_with_private_difference()
    public = _public_observation(public_state)
    changed = _public_observation(private_state)
    config = _model_config()
    torch.manual_seed(23)
    policy = RecurrentHybridPolicy(config).eval()

    public_inputs = _model_inputs(public)
    changed_inputs = _model_inputs(changed)
    reset_mask = torch.ones((1, 1), dtype=torch.bool)
    hidden = policy.initial_hidden(1)

    with torch.inference_mode():
        public_output = policy(
            *public_inputs,
            reset_mask=reset_mask,
            hidden=hidden,
        )
        changed_output = policy(
            *changed_inputs,
            reset_mask=reset_mask,
            hidden=hidden,
        )

        for public_logits, changed_logits in (
            (public_output.logits.mode, changed_output.logits.mode),
            (public_output.logits.card, changed_output.logits.card),
            (public_output.logits.placement, changed_output.logits.placement),
            (public_output.encoded_features, changed_output.encoded_features),
            (public_output.recurrent_features, changed_output.recurrent_features),
        ):
            torch.testing.assert_close(public_logits, changed_logits)

        public_masked = policy.action_head.masked_log_probs(
            public_output.logits,
            _action_masks(public),
        )
        changed_masked = policy.action_head.masked_log_probs(
            changed_output.logits,
            _action_masks(changed),
        )
        torch.testing.assert_close(public_masked.mode, changed_masked.mode)
        torch.testing.assert_close(public_masked.card, changed_masked.card)
        torch.testing.assert_close(public_masked.placement, changed_masked.placement)
