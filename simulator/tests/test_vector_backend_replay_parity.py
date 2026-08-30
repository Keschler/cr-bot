from __future__ import annotations

import multiprocessing

import pytest


@pytest.mark.parametrize(
    "backend",
    ["process", "packed-process", "persistent-process"],
)
def test_vector_backend_preserves_all_hashes_without_privileged_info(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker transport must retain replay history at the public boundary."""

    # The sandbox cannot create forkserver sockets. Forking here keeps the
    # differential assertion identical to the production worker contract.
    real_get_context = multiprocessing.get_context

    def fork_context(method: str | None = None):
        if method in {"forkserver", "spawn"}:
            method = "fork"
        return real_get_context(method)

    monkeypatch.setattr(multiprocessing, "get_context", fork_context)

    from simulator.actions import PlayCardAction
    from simulator.env import SimulatorEnv, VectorSimulatorEnv

    reference = VectorSimulatorEnv(tuple(SimulatorEnv() for _ in range(2)))
    parallel = VectorSimulatorEnv(
        tuple(SimulatorEnv() for _ in range(2)),
        backend=backend,
        workers=2,
    )
    try:
        reference.reset((701, 702))
        parallel.reset((701, 702))

        actions = []
        for environment in reference.environments:
            state = environment.state
            assert state is not None
            card_slot = 0
            card_id = state.players[0].hand[card_slot]
            legal_cells = environment.engine.legal_cells(state, 0, card_id)
            assert legal_cells
            actions.append(
                (PlayCardAction(0, card_slot, legal_cells[0]), None)
            )

        expected = reference.step(tuple(actions))
        actual = parallel.step(tuple(actions))
        for expected_step, actual_step in zip(expected, actual, strict=True):
            assert actual_step.rewards == expected_step.rewards
            assert actual_step.terminated == expected_step.terminated
            assert actual_step.truncated == expected_step.truncated
            assert actual_step.info == expected_step.info

        for expected_environment, actual_environment in zip(
            reference.environments,
            parallel.environments,
            strict=True,
        ):
            expected_state = expected_environment.state
            actual_state = actual_environment.state
            assert expected_state is not None
            assert actual_state is not None
            assert actual_state.state_hash() == expected_state.state_hash()
            assert actual_state.event_log_hash() == expected_state.event_log_hash()
            assert actual_state.replay_hash() == expected_state.replay_hash()
            assert actual_state.events

        # A second round catches implementations that preserve only the
        # current step's events rather than the accumulated replay history.
        waits = ((None, None), (None, None))
        reference.step(waits)
        parallel.step(waits)
        for expected_environment, actual_environment in zip(
            reference.environments,
            parallel.environments,
            strict=True,
        ):
            assert actual_environment.state is not None
            assert expected_environment.state is not None
            assert actual_environment.state.replay_hash() == expected_environment.state.replay_hash()
    finally:
        reference.close()
        parallel.close()


@pytest.mark.parametrize("packed", [False, True])
def test_worker_payload_honors_hidden_authoritative_state(packed: bool) -> None:
    """The isolated worker must not re-enable privileged state by default."""

    from simulator.env import (
        SimulatorEnv,
        _packed_parallel_env_step_worker,
        _parallel_env_step_worker,
    )
    from simulator.packed_batch import pack_state

    environment = SimulatorEnv(
        expose_privileged_info=True,
        include_authoritative_state=False,
    )
    environment.reset(seed=811)
    state = environment.state
    assert state is not None
    reward = environment.reward_config
    common = (
        state.ruleset_id,
        environment.decision_interval_ticks * environment.engine.ruleset.tick_us,
        reward.version,
        reward.tower_damage_weight,
        reward.crown_weight,
        reward.win_weight,
        True,
        False,
        environment.engine.validate_every_tick,
    )
    if packed:
        result = _packed_parallel_env_step_worker(
            (*common, pack_state(state).to_bytes(), (None, None))
        )
    else:
        result = _parallel_env_step_worker(
            (*common, state.to_primitive(include_events=False), (None, None))
        )
    info = result[-1]
    assert "state_hash" in info
    assert "event_log_hash" in info
    assert "authoritative_state" not in info


def test_simulator_observation_cache_is_contract_specific_and_invalidated_on_step() -> None:
    from simulator.env import SimulatorEnv

    environment = SimulatorEnv()
    environment.reset(seed=812)
    state = environment.state
    assert state is not None

    v1 = environment.observe()
    environment._persistent_observation_cache = (id(state), "v1", v1)
    assert environment.observe() is v1

    v2 = environment.observe_v2()
    environment._persistent_observation_cache = (id(state), "v2", v2)
    assert environment.observe_v2() is v2
    assert environment.observe() is not v2

    environment._persistent_observation_cache = (id(state), "v1", v1)
    step = environment.step((None, None))
    assert step.observations is not v1


@pytest.mark.parametrize(
    "backend",
    ["process", "packed-process", "persistent-process"],
)
def test_vector_backend_privileged_event_hash_includes_parent_history(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_get_context = multiprocessing.get_context

    def fork_context(method: str | None = None):
        if method in {"forkserver", "spawn"}:
            method = "fork"
        return real_get_context(method)

    monkeypatch.setattr(multiprocessing, "get_context", fork_context)

    from simulator.env import SimulatorEnv, VectorSimulatorEnv

    vector = VectorSimulatorEnv(
        (
            SimulatorEnv(
                expose_privileged_info=True,
                include_authoritative_state=False,
            ),
        ),
        backend=backend,
        workers=1,
    )
    try:
        vector.reset((813,))
        first = vector.step(((None, None),))[0]
        state = vector.environments[0].state
        assert state is not None
        assert first.info["event_log_hash"] == state.event_log_hash()

        second = vector.step(((None, None),))[0]
        state = vector.environments[0].state
        assert state is not None
        assert second.info["event_log_hash"] == state.event_log_hash()
    finally:
        vector.close()


def test_persistent_backend_syncs_in_place_parent_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_get_context = multiprocessing.get_context

    def fork_context(method: str | None = None):
        if method in {"forkserver", "spawn"}:
            method = "fork"
        return real_get_context(method)

    monkeypatch.setattr(multiprocessing, "get_context", fork_context)

    from simulator.env import SimulatorEnv, VectorSimulatorEnv

    reference = VectorSimulatorEnv((SimulatorEnv(),))
    persistent = VectorSimulatorEnv(
        (SimulatorEnv(),),
        backend="persistent-process",
        workers=1,
    )
    try:
        reference.reset((814,))
        persistent.reset((814,))
        reference.step(((None, None),))
        persistent.step(((None, None),))

        for vector in (reference, persistent):
            state = vector.environments[0].state
            assert state is not None
            state.players[0].elixir_milli = 0

        expected = reference.step(((None, None),))[0]
        actual = persistent.step(((None, None),))[0]

        assert actual.rewards == expected.rewards
        assert actual.info == expected.info
        expected_state = reference.environments[0].state
        actual_state = persistent.environments[0].state
        assert expected_state is not None and actual_state is not None
        assert actual_state.replay_hash() == expected_state.replay_hash()
    finally:
        reference.close()
        persistent.close()


def test_persistent_backend_fingerprint_does_not_hash_event_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_get_context = multiprocessing.get_context

    def fork_context(method: str | None = None):
        if method in {"forkserver", "spawn"}:
            method = "fork"
        return real_get_context(method)

    monkeypatch.setattr(multiprocessing, "get_context", fork_context)

    from simulator.env import (
        SimulatorEnv,
        VectorSimulatorEnv,
        _state_sync_fingerprint,
    )
    from simulator.state import BattleState

    def fail_if_hashed(_state):
        raise AssertionError("persistent sync hashed the complete event log")

    monkeypatch.setattr(BattleState, "replay_hash", fail_if_hashed)
    vector = VectorSimulatorEnv(
        (SimulatorEnv(),),
        backend="persistent-process",
        workers=1,
    )
    try:
        vector.reset((815,))
        vector.step(((None, None),))
        state = vector.environments[0].state
        assert state is not None
        before = _state_sync_fingerprint(state)
        state.events[:] = list(state.events)
        assert _state_sync_fingerprint(state) != before
        vector.step(((None, None),))
    finally:
        vector.close()
