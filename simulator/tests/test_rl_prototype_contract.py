"""Public-observation and V4 actor invariance under hidden-state changes."""

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
    state.events[:] = [
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


def test_public_memory_replays_when_a_processed_event_is_rewritten() -> None:
    engine = BattleEngine(RULESET)
    state = engine.new_battle(
        (PLAYER_DECK, PLAYER_DECK),
        seed=31415,
        shuffle_decks=False,
    )
    state.elapsed_us = RULESET.tick_us
    state.events[:] = [
        SimEvent.create(
            0,
            1,
            "card_played",
            player=1,
            card_id="fireball",
            cost_milli=4_000,
        )
    ]
    state.event_sequence = 1
    memory = ObservationMemory(viewer=0)
    memory.update(state, RULESET)
    first_estimate = memory.opponent_elixir_milli_est
    assert memory.seen_opponent_cards == ["fireball"]

    state.events[:] = [
        SimEvent.create(
            0,
            1,
            "card_played",
            player=1,
            card_id="ice-spirit",
            cost_milli=1_000,
        )
    ]
    memory.update(state, RULESET)

    expected = ObservationMemory(viewer=0)
    expected.update(state, RULESET)
    assert memory.seen_opponent_cards == ["ice-spirit"]
    assert memory.opponent_elixir_milli_est == expected.opponent_elixir_milli_est
    assert memory.opponent_elixir_milli_est != first_estimate


@requires_torch
def test_v4_actor_path_is_invariant_to_hidden_opponent_state() -> None:
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.model_v4 import RecurrentV4Policy
    from simulator.rl.v4.v4_ppo.runtime import _v4_step_inputs, batch_v4_inputs

    public_state, private_state = _states_with_private_difference()
    public = _public_observation(public_state)
    changed = _public_observation(private_state)
    hand = tuple(public_state.players[0].hand[:4])
    elixir = float(public_state.players[0].elixir_milli) / 1000.0
    snapshot = PublicStateEstimator().snapshot()
    first = batch_v4_inputs(_v4_step_inputs(public, hand, elixir, snapshot), "cpu")
    second = batch_v4_inputs(_v4_step_inputs(changed, hand, elixir, snapshot), "cpu")

    torch.manual_seed(23)
    policy = RecurrentV4Policy(frontier_model_config()).eval()
    reset = torch.ones((1, 1), dtype=torch.bool)
    with torch.inference_mode():
        public_logits, public_recurrent, _ = policy.forward_step(first, reset_mask=reset)
        changed_logits, changed_recurrent, _ = policy.forward_step(second, reset_mask=reset)

    for name in ("mode", "card", "placement", "wait_duration"):
        torch.testing.assert_close(
            getattr(public_logits, name), getattr(changed_logits, name)
        )
    torch.testing.assert_close(public_recurrent, changed_recurrent)
