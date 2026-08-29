from __future__ import annotations

import multiprocessing

import pytest


@pytest.mark.parametrize("backend", ["process", "packed-process"])
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
