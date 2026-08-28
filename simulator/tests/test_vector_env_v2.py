from __future__ import annotations

import numpy as np
import pytest


def _observation_arrays(observation):
    return (
        observation.board,
        observation.global_vector,
        observation.entity_tokens,
        observation.entity_mask,
        observation.legal_play,
        observation.legal_wait,
    )


@pytest.mark.parametrize("backend", ["process", "packed-process"])
def test_process_backends_preserve_v2_step_results(backend: str) -> None:
    from simulator.env import SimulatorEnv, VectorSimulatorEnv

    reference_lanes = tuple(
        SimulatorEnv(decision_interval_us=1_000_000) for _ in range(2)
    )
    parallel_lanes = tuple(
        SimulatorEnv(decision_interval_us=1_000_000) for _ in range(2)
    )
    reference = VectorSimulatorEnv(reference_lanes)
    parallel = VectorSimulatorEnv(parallel_lanes, backend=backend, workers=2)
    try:
        reference.reset_v2((101, 202))
        parallel.reset_v2((101, 202))

        actions = ((None, None), (None, None))
        expected = reference.step_v2(actions)
        actual = parallel.step_v2(actions)

        assert len(actual) == len(expected) == 2
        for expected_step, actual_step in zip(expected, actual, strict=True):
            assert actual_step.rewards == expected_step.rewards
            assert actual_step.terminated == expected_step.terminated
            assert actual_step.truncated == expected_step.truncated
            for expected_observation, actual_observation in zip(
                expected_step.observations,
                actual_step.observations,
                strict=True,
            ):
                expected_arrays = _observation_arrays(expected_observation)
                actual_arrays = _observation_arrays(actual_observation)
                for expected_array, actual_array in zip(
                    expected_arrays[:-1], actual_arrays[:-1], strict=True
                ):
                    assert np.array_equal(actual_array, expected_array)
                assert actual_arrays[-1] == expected_arrays[-1]
    finally:
        reference.close()
        parallel.close()


def test_prototype_cli_accepts_process_backend() -> None:
    from rl.prototype import PrototypeConfig, _parser

    args = _parser().parse_args(
        [
            "train",
            "--env-backend",
            "process",
            "--env-workers",
            "3",
        ]
    )
    config = PrototypeConfig(env_backend=args.env_backend, env_workers=args.env_workers)
    assert config.env_backend == "process"
    assert config.env_workers == 3
