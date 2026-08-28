from __future__ import annotations

import numpy as np
import pytest

from rl.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizedEnv,
    DomainVariantSampler,
)
from simulator.env import SimulatorEnv
from simulator.roster import PLAYER_DECK


def test_domain_randomization_config_and_sampler_are_replayable() -> None:
    config = DomainRandomizationConfig(
        profile_id="interface-test",
        decision_interval_jitter_ticks=2,
        action_latency_max_steps=3,
        entity_observation_noise_std=0.01,
    )
    restored = DomainRandomizationConfig.from_mapping(config.as_dict())
    assert restored == config
    first = DomainVariantSampler(config, base_decision_interval_ticks=5, seed=17)
    second = DomainVariantSampler(config, base_decision_interval_ticks=5, seed=17)
    variants_a = [first.sample(index) for index in range(8)]
    variants_b = [second.sample(index) for index in range(8)]
    assert variants_a == variants_b
    assert all(1 <= item.decision_interval_ticks <= 7 for item in variants_a)
    assert all(0 <= item.action_latency_steps <= 3 for item in variants_a)


def _noisy_wrapper(seed: int) -> DomainRandomizedEnv:
    environment = SimulatorEnv(decision_interval_us=1_000_000)
    wrapper = DomainRandomizedEnv(
        environment,
        DomainRandomizationConfig(entity_observation_noise_std=0.02),
        seed=seed,
    )
    wrapper.reset(decks=(PLAYER_DECK, PLAYER_DECK), seed=31, shuffle_decks=False)
    assert wrapper.state is not None
    environment.engine._spawn_single_at(
        wrapper.state,
        environment.engine.ruleset.card("hog-rider"),
        owner=0,
        x_mtile=4_500,
        y_mtile=23_500,
        deploy_remaining_us=0,
    )
    return wrapper


def test_domain_randomized_v2_noise_is_seeded_and_does_not_touch_v1() -> None:
    first = _noisy_wrapper(99)
    second = _noisy_wrapper(99)
    first_v1 = first.observe()
    second_v1 = second.observe()
    first_v2 = first.observe_v2()
    second_v2 = second.observe_v2()

    np.testing.assert_array_equal(first_v1[0].board, second_v1[0].board)
    np.testing.assert_array_equal(first_v1[0].global_vector, second_v1[0].global_vector)
    np.testing.assert_array_equal(first_v2[0].entity_tokens, second_v2[0].entity_tokens)
    assert int(first_v2[0].entity_mask.sum()) == 1
    assert np.isfinite(first_v2[0].entity_tokens).all()


def test_domain_randomized_wrapper_requires_reset() -> None:
    wrapper = DomainRandomizedEnv(
        SimulatorEnv(),
        DomainRandomizationConfig(),
    )
    with pytest.raises(RuntimeError, match="reset"):
        wrapper.observe_v2()
    with pytest.raises(RuntimeError, match="reset"):
        wrapper.step((None, None))
