from __future__ import annotations

import numpy as np
import pytest

from cr_bot.domain.game_state import Action as PolicyAction
from simulator.actions import WaitAction
from simulator.env import SimulatorEnv, VectorSimulatorEnv
from simulator.observation import ACTION_MASK_SHAPE, BOARD_SHAPE, GLOBAL_VECTOR_SHAPE


def _assert_observations_equal(first, second) -> None:
    np.testing.assert_array_equal(first.board, second.board)
    np.testing.assert_array_equal(first.global_vector, second.global_vector)
    np.testing.assert_array_equal(first.spatial_masks, second.spatial_masks)
    np.testing.assert_array_equal(first.legal_play, second.legal_play)
    assert first.legal_wait == second.legal_wait
    assert first.compatibility_mode == second.compatibility_mode


def test_reset_exposes_the_existing_policy_boundary_for_both_players() -> None:
    env = SimulatorEnv()

    observations = env.reset(seed=19, shuffle_decks=False)

    assert len(observations) == 2
    for observation in observations:
        assert observation.board.shape == BOARD_SHAPE
        assert observation.global_vector.shape == GLOBAL_VECTOR_SHAPE
        assert observation.spatial_masks.shape == ACTION_MASK_SHAPE
        assert observation.legal_play.shape == ACTION_MASK_SHAPE
        assert observation.board.dtype == np.float32
        assert observation.global_vector.dtype == np.float32
        assert observation.spatial_masks.dtype == np.bool_
        assert observation.legal_play.dtype == np.bool_
        assert observation.legal_wait


def test_policy_actions_advance_one_decision_interval_deterministically() -> None:
    first = SimulatorEnv()
    second = SimulatorEnv()
    first.reset(seed=42, shuffle_decks=False)
    second.reset(seed=42, shuffle_decks=False)
    actions = (PolicyAction(kind="Play", card_idx=0, cell=(3, 17)), None)

    first_step = first.step(actions)
    second_step = second.step(actions)

    assert first_step.info["physics_tick"] == first.decision_interval_ticks
    assert first.state is not None and second.state is not None
    assert first.state.state_hash() == second.state.state_hash()
    assert first_step.rewards == second_step.rewards == (0.0, 0.0)
    assert first_step.terminated == second_step.terminated is False
    assert first_step.truncated == second_step.truncated is False
    for first_observation, second_observation in zip(
        first_step.observations, second_step.observations, strict=True
    ):
        _assert_observations_equal(first_observation, second_observation)


def test_default_environment_does_not_expose_authoritative_state() -> None:
    env = SimulatorEnv(expose_privileged_info=False)
    env.reset(seed=0, shuffle_decks=False)

    result = env.step((None, None))

    assert "authoritative_state" not in result.info
    assert set(result.info) == {
        "ruleset_id",
        "ruleset_hash",
        "engine_version",
        "observation_contract_hash",
        "physics_tick",
        "decision_interval_ticks",
        "reward_version",
        "winner",
        "terminal_reason",
    }
    privileged = SimulatorEnv(expose_privileged_info=True)
    privileged.reset(seed=0, shuffle_decks=False)
    privileged_result = privileged.step((None, None))
    assert "authoritative_state" in privileged_result.info
    assert "state_hash" in privileged_result.info
    assert "event_log_hash" in privileged_result.info
    assert "events" in privileged_result.info


def test_viewer_observation_is_independent_of_opponent_private_state() -> None:
    baseline = SimulatorEnv()
    changed = SimulatorEnv()
    baseline.reset(seed=7, shuffle_decks=False)
    changed.reset(seed=7, shuffle_decks=False)
    assert baseline.state is not None and changed.state is not None

    opponent = changed.state.players[1]
    opponent.hand[:], opponent.draw_pile[:] = opponent.draw_pile[:], opponent.hand[:]
    opponent.deck = tuple(reversed(opponent.deck))
    opponent.elixir_milli = 0
    opponent.elixir_remainder = 123
    opponent.seen_enemy_cards[:] = ["private-opponent-memory"]
    enemy_king = next(
        entity
        for entity in changed.state.entities.values()
        if entity.owner == 1 and entity.role == "king"
    )
    enemy_king.attack_cooldown_us = 777_777
    enemy_king.windup_remaining_us = 222_222
    changed.state.rng_state ^= 0xDEADBEEF

    assert baseline.state.state_hash() != changed.state.state_hash()
    baseline_view = baseline.observe()[0]
    changed_view = changed.observe()[0]
    _assert_observations_equal(baseline_view, changed_view)


def test_environment_save_load_resume_preserves_outcomes_and_observations() -> None:
    first = SimulatorEnv()
    first.reset(seed=88, shuffle_decks=False)
    first.step((PolicyAction(kind="Play", card_idx=0, cell=(3, 17)), None))
    saved = first.save_state()

    resumed = SimulatorEnv()
    loaded_observations = resumed.load_state(saved)
    current_observations = first.observe()
    for current, loaded in zip(current_observations, loaded_observations, strict=True):
        _assert_observations_equal(current, loaded)

    first_step = first.step((None, None))
    resumed_step = resumed.step((None, None))
    assert first.state is not None and resumed.state is not None
    assert first.state.state_hash() == resumed.state.state_hash()
    assert first_step.rewards == resumed_step.rewards
    assert first_step.terminated == resumed_step.terminated
    for current, loaded in zip(
        first_step.observations, resumed_step.observations, strict=True
    ):
        _assert_observations_equal(current, loaded)


def test_environment_rejects_malformed_action_batches_and_uninitialized_use() -> None:
    env = SimulatorEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.observe()
    with pytest.raises(RuntimeError, match="reset"):
        env.step((None, None))

    env.reset(seed=0, shuffle_decks=False)
    with pytest.raises(ValueError, match="one action"):
        env.step((None,))
    with pytest.raises(ValueError, match="does not match"):
        env.step((WaitAction(1), None))


def test_environment_rejects_steps_after_terminal_transition() -> None:
    env = SimulatorEnv()
    env.reset(seed=0, shuffle_decks=False)
    assert env.state is not None
    env.state.terminal = True
    env.state.phase = "ended"
    env.state.terminal_reason = "test_terminal"

    with pytest.raises(RuntimeError, match="terminal environment"):
        env.step((None, None))


def test_vector_environment_keeps_independent_deterministic_states() -> None:
    vector = VectorSimulatorEnv.create(2)
    vector.reset((3, 4))

    results = vector.step(((None, None), (None, None)))

    assert len(results) == 2
    assert all(result.info["physics_tick"] == 5 for result in results)
    states = [environment.state for environment in vector.environments]
    assert all(state is not None for state in states)
    assert states[0].state_hash() != states[1].state_hash()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="one seed"):
        vector.reset((1,))


def test_process_vector_backend_matches_reference_hashes_observations_and_rewards() -> None:
    reference = VectorSimulatorEnv.create(2, expose_privileged_info=True)
    parallel = VectorSimulatorEnv.create(
        2,
        backend="process",
        workers=2,
        expose_privileged_info=True,
    )
    try:
        reference.reset((31, 32))
        parallel.reset((31, 32))
        action_rows = (
            ((PolicyAction(kind="Play", card_idx=0, cell=(3, 17)), None)),
            ((None, None)),
        )
        for rows in (action_rows, ((None, None), (None, None))):
            expected = reference.step(rows)
            actual = parallel.step(rows)
            assert len(actual) == len(expected) == 2
            for expected_step, actual_step in zip(expected, actual, strict=True):
                assert actual_step.rewards == expected_step.rewards
                assert actual_step.terminated == expected_step.terminated
                assert actual_step.truncated == expected_step.truncated
                assert actual_step.info == expected_step.info
                for expected_observation, actual_observation in zip(
                    expected_step.observations,
                    actual_step.observations,
                    strict=True,
                ):
                    _assert_observations_equal(expected_observation, actual_observation)
            for expected_env, actual_env in zip(
                reference.environments,
                parallel.environments,
                strict=True,
            ):
                assert expected_env.state is not None and actual_env.state is not None
                assert expected_env.state.state_hash() == actual_env.state.state_hash()
                assert expected_env.state.event_log_hash() == actual_env.state.event_log_hash()
    finally:
        parallel.close()


def test_vector_backend_rejects_unknown_backend_and_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="backend"):
        VectorSimulatorEnv.create(1, backend="cuda")
    with pytest.raises(ValueError, match="workers"):
        VectorSimulatorEnv.create(1, backend="process", workers=0)
