"""Regression coverage for public-memory features in V4 dataset builders."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simulator.public_state_estimator import PublicStateEstimator
from simulator.rl.foundation.opponent_pool import OpponentPool
from simulator.rl.v4.search_dataset.records import _new_sim_env, _record_from_env
from simulator.rl.v4.timing_curriculum import (
    _TimingContext,
    _capture,
    _deploy,
    _new_raw_env,
    _to_sample,
)
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_fixed_ruleset


def _snapshot() -> SimpleNamespace:
    event_history = np.zeros((16, 8), dtype=np.float32)
    event_history[-1, 2] = 1.0
    hand_probs = np.zeros((128,), dtype=np.float32)
    hand_probs[7] = 1.0
    out_of_cycle = np.zeros((128,), dtype=bool)
    out_of_cycle[11] = True
    return SimpleNamespace(
        event_history=event_history,
        opp_hand_probs=hand_probs,
        opp_out_of_cycle=out_of_cycle,
        opp_elixir_lo=2.25,
        opp_elixir_hi=4.75,
    )


def test_search_record_syncs_public_memory_from_simulator_prefix() -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=9)
    env, family, source = _new_sim_env(ruleset, pool, 9, 3, 9)
    expected = PublicStateEstimator().sync_from_env(env, viewer=0)

    record = _record_from_env(env, family, source, ruleset)

    assert record is not None
    np.testing.assert_array_equal(record["event_history"], expected.event_history)
    np.testing.assert_array_equal(record["opp_hand_probs"], expected.opp_hand_probs)
    np.testing.assert_array_equal(record["opp_out_of_cycle"], expected.opp_out_of_cycle)
    np.testing.assert_array_equal(
        record["opp_elixir_interval"],
        np.asarray([expected.opp_elixir_lo, expected.opp_elixir_hi], dtype=np.float32),
    )
    assert np.any(record["event_history"])
    assert not np.array_equal(record["opp_elixir_interval"], np.asarray([0.0, 10.0]))


def test_timing_capture_and_sample_keep_supplied_public_memory() -> None:
    context = _TimingContext.build(3)
    env = _new_raw_env(context, tuple(PLAYER_DECK), 3)
    memory = _snapshot()

    record = _capture(
        context,
        env,
        family="unit",
        seq_id="unit:0",
        offset=0,
        env_seed=3,
        source="unit",
        public_memory=memory,
    )

    assert record is not None
    np.testing.assert_array_equal(record["event_history"], memory.event_history)
    np.testing.assert_array_equal(record["opp_hand_probs"], memory.opp_hand_probs)
    np.testing.assert_array_equal(record["opp_out_of_cycle"], memory.opp_out_of_cycle)
    np.testing.assert_array_equal(
        record["opp_elixir_interval"], np.asarray([2.25, 4.75], dtype=np.float32)
    )
    sample = _to_sample(record, 0)
    np.testing.assert_array_equal(sample.event_history, memory.event_history)
    np.testing.assert_array_equal(sample.opp_hand_probs, memory.opp_hand_probs)
    np.testing.assert_array_equal(sample.opp_out_of_cycle, memory.opp_out_of_cycle)
    np.testing.assert_array_equal(
        sample.opp_elixir_interval, np.asarray([2.25, 4.75], dtype=np.float32)
    )


def test_timing_capture_default_sync_observes_public_deployments() -> None:
    context = _TimingContext.build(3)
    env = _new_raw_env(context, tuple(PLAYER_DECK), 3)
    _deploy(context, env.state, env.engine, 1, "skeletons", 9, 5)

    record = _capture(
        context,
        env,
        family="unit",
        seq_id="unit:0",
        offset=0,
        env_seed=3,
        source="unit",
    )

    assert record is not None
    assert np.any(record["event_history"])
    assert int(record["opp_out_of_cycle"].sum()) > 0
    assert not np.array_equal(record["opp_elixir_interval"], np.asarray([0.0, 10.0]))
