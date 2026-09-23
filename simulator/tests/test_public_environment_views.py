"""Public V2 environment projection regression checks."""

import numpy as np


def test_v2_environment_reset_and_step_stay_on_public_boundary() -> None:
    from simulator.env import EnvStepV2, SimulatorEnv
    from simulator.observation_v2 import PolicyObservationV2

    environment = SimulatorEnv(decision_interval_us=1_000_000)
    observations = environment.reset_v2(seed=7, shuffle_decks=False)
    assert all(isinstance(observation, PolicyObservationV2) for observation in observations)
    step = environment.step_v2((None, None))
    assert isinstance(step, EnvStepV2)
    assert all(isinstance(observation, PolicyObservationV2) for observation in step.observations)


def test_v2_single_view_matches_full_view() -> None:
    from simulator.env import SimulatorEnv

    environment = SimulatorEnv(decision_interval_us=1_000_000)
    full_before = environment.reset_v2(seed=7, shuffle_decks=False)
    for single, full in zip(
        (environment.observe_v2_for_viewer(0), environment.observe_v2_for_viewer(1)),
        full_before,
        strict=True,
    ):
        assert np.array_equal(single.board, full.board)
        assert np.array_equal(single.global_vector, full.global_vector)
        assert np.array_equal(single.entity_tokens, full.entity_tokens)
        assert np.array_equal(single.entity_mask, full.entity_mask)
        assert np.array_equal(single.legal_play, full.legal_play)
        assert single.legal_wait == full.legal_wait

    full = environment.step_v2((None, None))
    single_environment = SimulatorEnv(decision_interval_us=1_000_000)
    single_environment.reset_v2(seed=7, shuffle_decks=False)
    single = single_environment.step_v2_for_viewer((None, None), viewer=0)
    assert single.rewards == full.rewards
    assert single.terminated == full.terminated
    assert single.truncated == full.truncated
    assert single.info == full.info
    assert np.array_equal(single.observations[0].board, full.observations[0].board)
    assert np.array_equal(
        single.observations[0].global_vector,
        full.observations[0].global_vector,
    )
    assert np.array_equal(
        single.observations[0].entity_tokens,
        full.observations[0].entity_tokens,
    )
    assert np.array_equal(
        single.observations[0].entity_mask,
        full.observations[0].entity_mask,
    )
    assert np.array_equal(
        single.observations[0].legal_play,
        full.observations[0].legal_play,
    )
    assert single.observations[0].legal_wait == full.observations[0].legal_wait
    assert single.observations[1] is None
