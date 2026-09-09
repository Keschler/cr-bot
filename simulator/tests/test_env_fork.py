"""Parity and isolation tests for SimulatorEnv.fork() (Stage 2, Step 1).

A fork must start from the exact same logical state as its parent
(observations, hashes, legal masks identical) and become fully independent
afterwards (same actions reproduce, different actions diverge, stepping the
child never mutates the parent, forking is deterministic).
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.engine import BattleEngine
from simulator.env import SimulatorEnv
from simulator.rl.basic_scenarios import (
    BasicMechanicsScenarioEnv,
    BasicScenarioConfig,
)
from simulator.rl.opponent_pool import OpponentPool
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_fixed_ruleset


def _scenario_env(seed: int = 123) -> BasicMechanicsScenarioEnv:
    ruleset = load_fixed_ruleset()
    opponent = OpponentPool(ruleset, seed=71).sample(
        0, archetype="beatdown", strategy="deterministic-cycle"
    )
    base = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    wrapped = BasicMechanicsScenarioEnv(
        base,
        BasicScenarioConfig(source="ground-defense", target_player=0, decision_limit=64),
    )
    wrapped.reset_v2(
        seed=seed,
        decks=(tuple(PLAYER_DECK), opponent.deck.cards),
        shuffle_decks=True,
    )
    return wrapped


def _v2_snapshot(env) -> dict[str, np.ndarray | bool]:
    obs = env.observe_v2_for_viewer(0)
    return {
        "board": np.array(obs.board),
        "global": np.array(obs.global_vector),
        "entities": np.array(obs.entity_tokens),
        "mask": np.array(obs.entity_mask),
        "legal": np.array(obs.legal_play),
        "wait": bool(obs.legal_wait),
    }


def _assert_same_snapshot(first: dict, second: dict) -> None:
    for key in ("board", "global", "entities", "mask", "legal"):
        assert np.array_equal(first[key], second[key]), key
    assert first["wait"] == second["wait"]


def test_fork_observations_hashes_and_masks_match() -> None:
    env = _scenario_env()
    child = env.fork()
    _assert_same_snapshot(_v2_snapshot(env), _v2_snapshot(child))
    assert env.state.state_hash() == child.state.state_hash()


def test_fork_same_actions_reproduce() -> None:
    env = _scenario_env()
    child = env.fork()
    for _ in range(4):
        env.step_v2((WaitAction(0), WaitAction(1)))
        child.step_v2((WaitAction(0), WaitAction(1)))
        assert env.state.state_hash() == child.state.state_hash()
        _assert_same_snapshot(_v2_snapshot(env), _v2_snapshot(child))


def test_fork_different_actions_diverge_independently() -> None:
    env = _scenario_env()
    child = env.fork()
    before = env.state.state_hash()
    hand = list(child.state.players[0].hand)
    slot = hand.index("cannon") if "cannon" in hand else 0
    cells = child.engine.legal_cells(child.state, 0, hand[slot])
    assert cells, "scenario must offer a legal learner cell"
    child.engine.apply_actions(child.state, (PlayCardAction(0, slot, cells[0]),))
    assert child.state.state_hash() != before
    assert env.state.state_hash() == before


def test_stepping_child_never_mutates_parent() -> None:
    env = _scenario_env()
    child = env.fork()
    parent_hash = env.state.state_hash()
    parent_snapshot = _v2_snapshot(env)
    for _ in range(6):
        child.step_v2((WaitAction(0), WaitAction(1)))
    assert env.state.state_hash() == parent_hash
    _assert_same_snapshot(parent_snapshot, _v2_snapshot(env))


def test_fork_is_deterministic() -> None:
    env = _scenario_env()
    first, second = env.fork(), env.fork()
    assert first.state.state_hash() == second.state.state_hash()
    _assert_same_snapshot(_v2_snapshot(first), _v2_snapshot(second))
    first.step_v2((WaitAction(0), WaitAction(1)))
    second.step_v2((WaitAction(0), WaitAction(1)))
    assert first.state.state_hash() == second.state.state_hash()


def test_fork_shares_no_mutable_state() -> None:
    env = _scenario_env()
    child = env.fork()
    assert child.engine is not env.engine
    assert child.state is not env.state
    assert child.state.events is not env.state.events
    for _ in range(3):
        child.step_v2((WaitAction(0), WaitAction(1)))
    assert env.engine._navigation_cache_state is not child.state


def test_fork_before_reset_is_rejected() -> None:
    ruleset = load_fixed_ruleset()
    fresh = SimulatorEnv(engine=BattleEngine(ruleset, validate_every_tick=False))
    with pytest.raises(RuntimeError, match="reset"):
        fresh.fork()
