"""Tests for the Stage-2 counterfactual search teacher.

Pins: all branches legal and deterministic, branch budget honored,
parent/fork isolation during search, tick-step rollout parity with
decision stepping, scoring sanity (defense/offense beat WAIT when they
connect, pointless cycling loses to WAIT), and full branch provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.engine import BattleEngine
from simulator.env import SimulatorEnv
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_fixed_ruleset
from simulator.rl.basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig
from simulator.rl.opponent_pool import OpponentPool
from simulator.rl.simulator_distillation import build_hand_tokens
from simulator.rl.simulator_teacher import teacher_label
from simulator.rl import search_teacher as st


def _env(source: str, archetype: str, seed: int):
    ruleset = load_fixed_ruleset()
    opponent = OpponentPool(ruleset, seed=0).sample(
        seed, archetype=archetype, strategy="deterministic-cycle"
    )
    base = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    wrapped = BasicMechanicsScenarioEnv(
        base, BasicScenarioConfig(source=source, target_player=0, decision_limit=64)
    )
    wrapped.reset_v2(
        seed=seed, decks=(tuple(PLAYER_DECK), opponent.deck.cards), shuffle_decks=True
    )
    return wrapped


def _labeled(env):
    from simulator.rl.timing_curriculum import _set_elixir, _set_hand

    return env, _set_hand, _set_elixir


def _search_inputs(env, hand, elixir_milli):
    from simulator.rl.timing_curriculum import _set_elixir, _set_hand

    _set_hand(env.state, 0, hand)
    _set_elixir(env.state, 0, elixir_milli)
    obs = env.observe_v2_for_viewer(0)
    state = env.state
    live_hand = list(state.players[0].hand)
    elixir = state.players[0].elixir_milli / 1000.0
    hand_tokens = build_hand_tokens(live_hand, elixir)
    legal = np.array(obs.legal_play, dtype=bool)
    rule = teacher_label(
        hand_tokens=hand_tokens,
        entity_tokens=np.asarray(obs.entity_tokens),
        entity_mask=np.asarray(obs.entity_mask),
        legal_play=legal,
        legal_wait=bool(obs.legal_wait),
        own_elixir=elixir,
    )
    return legal, rule, live_hand


def test_search_branches_are_all_legal_and_capped() -> None:
    env = _env("ground-defense", "beatdown", 5)
    legal, rule, hand = _search_inputs(
        env, ["cannon", "skeletons", "hog-rider", "musketeer"], 6000
    )
    branches = st.candidate_actions(
        legal_play=legal, rule_action=rule, hand_keys=hand, max_branches=24
    )
    assert 2 <= len(branches) <= 24
    assert branches[0].kind == "wait"
    for branch in branches:
        if branch.kind == "play":
            assert bool(legal[branch.slot, branch.row, branch.col])
            assert branch.card_key == hand[branch.slot]


def test_search_is_deterministic() -> None:
    def run():
        env = _env("ground-defense", "beatdown", 5)
        legal, rule, hand = _search_inputs(
            env, ["cannon", "skeletons", "hog-rider", "musketeer"], 6000
        )
        return st.search_state(
            env, legal_play=legal, rule_action=rule, hand_keys=hand, seed=3, source="t"
        )

    first, second = run(), run()
    assert first.state_hash == second.state_hash
    assert [b.score for b in first.branches] == [b.score for b in second.branches]
    assert [b.action for b in first.branches] == [b.action for b in second.branches]
    assert first.best_index == second.best_index


def test_search_never_mutates_parent() -> None:
    env = _env("ground-defense", "beatdown", 5)
    legal, rule, hand = _search_inputs(
        env, ["cannon", "skeletons", "hog-rider", "musketeer"], 6000
    )
    before = env.state.state_hash()
    result = st.search_state(env, legal_play=legal, rule_action=rule, hand_keys=hand)
    assert env.state.state_hash() == before
    assert result.branch_count >= 2
    assert result.state_hash == before


def test_tick_rollout_matches_decision_stepping() -> None:
    from simulator.actions import WaitAction

    env = _env("ground-defense", "beatdown", 5)
    ref = env.fork()
    child = env.fork()
    for _ in range(6):
        ref.step_v2((WaitAction(0), WaitAction(1)))
    st._rollout(child, 6)
    assert ref.state.state_hash() == child.state.state_hash()


def test_defense_beats_wait_when_it_connects() -> None:
    env = _env("ground-defense", "beatdown", 5)
    legal, rule, hand = _search_inputs(
        env, ["cannon", "skeletons", "hog-rider", "musketeer"], 6000
    )
    result = st.search_state(env, legal_play=legal, rule_action=rule, hand_keys=hand)
    by_kind = {}
    for branch in result.branches:
        key = "wait" if branch.action.kind == "wait" else branch.action.card_key
        by_kind.setdefault(key, []).append(branch.score)
    assert max(by_kind.get("cannon", [-1])) > by_kind["wait"][0]


def test_pointless_cycling_loses_to_wait() -> None:
    env = _env("isolated-offense", "aggressive-pressure", 5)
    legal, rule, hand = _search_inputs(
        env, ["skeletons", "ice-spirit", "cannon", "musketeer"], 9000
    )
    result = st.search_state(env, legal_play=legal, rule_action=rule, hand_keys=hand)
    wait_score = next(b.score for b in result.branches if b.action.kind == "wait")
    for branch in result.branches:
        if branch.action.kind == "play" and branch.action.card_key in ("skeletons", "ice-spirit"):
            assert branch.score < wait_score


def test_search_result_carries_full_provenance() -> None:
    env = _env("spell-situations", "siege-bait", 7)
    legal, rule, hand = _search_inputs(
        env, ["fireball", "log", "cannon", "musketeer"], 5000
    )
    result = st.search_state(
        env, legal_play=legal, rule_action=rule, hand_keys=hand, seed=9, source="probe"
    )
    assert result.state_hash == env.state.state_hash()
    assert result.horizon == st.HORIZON_DECISIONS
    assert result.second_gap >= 0.0
    for branch in result.branches:
        assert set(branch.terms) == {
            "tower_dealt",
            "tower_taken",
            "threat_removed",
            "deployed_survived",
            "elixir_spent",
        }
        assert all(np.isfinite(v) for v in branch.terms.values())
        assert np.isfinite(branch.score)
