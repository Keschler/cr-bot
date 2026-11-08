"""Regressions for reproducible, correctly paired full-match comparisons."""

from dataclasses import replace

import pytest
import torch

from simulator.rl.v4.frontier import frontier_model_config
from simulator.rl.v4.frontier_ppo import evaluation
from simulator.rl.v4.learning_curve import _default_opponents, paired_match_statistics
from simulator.rl.v4.model_v4 import RecurrentV4Policy


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return RecurrentV4Policy(frontier_model_config())


def test_full_match_honors_declared_controller_seed(policy, monkeypatch):
    spec = dict(_default_opponents()[0], controller_seed=123456)
    seeds = []
    original = evaluation.make_opponent

    def spy(spec, seed, **kwargs):
        seeds.append(seed)
        return original(spec, seed, **kwargs)

    monkeypatch.setattr(evaluation, "make_opponent", spy)
    rows = evaluation.full_match_record(
        policy=policy, opponent_specs=[spec], seeds=[7],
        deterministic=False, max_decisions=4,
    )
    assert seeds == [123456]
    assert rows[0]["opponent"]["controller_seed"] == 123456


def test_stochastic_matches_restore_rng_and_reproduce_traces_in_any_order(policy):
    specs = _default_opponents()[:2]
    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    one = evaluation.full_match_record(
        policy=policy, opponent_specs=specs, seeds=[7, 8],
        deterministic=False, max_decisions=16,
    )
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(37)
    two = evaluation.full_match_record(
        policy=policy, opponent_specs=list(reversed(specs)), seeds=[8, 7],
        deterministic=False, max_decisions=16,
    )
    identity = lambda row: (row["deck_id"], row["seed"])
    assert sorted(one, key=identity) == sorted(two, key=identity)
    assert all(row["result"] == "truncated" and row["truncated"] for row in one)
    assert all(row["cap_reached"] and row["terminal_reason"] == "decision_cap" for row in one)
    assert all(row["invalid_actions"] == [0, 0] for row in one)


@pytest.mark.parametrize("winner,result", [(0, "win"), (1, "loss"), (None, "draw")])
def test_completed_outcome_on_last_decision_is_not_censored(policy, monkeypatch, winner, result):
    make_env = evaluation.make_frontier_env

    def terminal_env():
        env = make_env()
        step = env.step_v2

        def finish(actions):
            actual = step(actions)
            env.state.terminal = True
            env.state.winner = winner
            env.state.terminal_reason = "test_terminal"
            return replace(actual, terminated=True, truncated=False,
                           info=dict(actual.info, winner=winner, terminal_reason="test_terminal"))

        env.step_v2 = finish
        return env

    monkeypatch.setattr(evaluation, "make_frontier_env", terminal_env)
    rows = evaluation.full_match_record(
        policy=policy, opponent_specs=_default_opponents()[:1], seeds=[7], max_decisions=1,
    )
    assert rows[0]["result"] == result
    assert rows[0]["terminated"] and not rows[0]["truncated"]
    assert not rows[0]["cap_reached"]


def test_simulator_runner_truncation_remains_identifiable(policy, monkeypatch):
    make_env = evaluation.make_frontier_env

    def limited_env():
        env = make_env()
        step = env.step_v2

        def finish(actions):
            actual = step(actions)
            return replace(actual, terminated=True, truncated=True,
                           info=dict(actual.info, winner=0, terminal_reason="runner_tick_limit"))

        env.step_v2 = finish
        return env

    monkeypatch.setattr(evaluation, "make_frontier_env", limited_env)
    row = evaluation.full_match_record(
        policy=policy, opponent_specs=_default_opponents()[:1], seeds=[7], max_decisions=4,
    )[0]
    assert row["result"] == "truncated" and row["winner"] is None
    assert row["truncated"] and not row["terminated"]
    assert row["terminal_reason"] == "runner_tick_limit"


@pytest.mark.parametrize("field,value", [
    ("controller_seed", 456), ("strategy", "defensive-cycle"),
    ("deterministic_policy", True), ("policy_seed", 99),
    ("learner_player", 1), ("max_decisions", 1200),
])
def test_pairing_rejects_changed_match_conditions(field, value):
    base = dict(archetype="aggressive-pressure", deck_id="hog-26-zap", seed=7,
                controller_seed=123, strategy="aggressive-pressure", result="win",
                deterministic_policy=False, policy_seed=9, learner_player=0, max_decisions=1440)
    with pytest.raises(ValueError, match="identities mismatch"):
        paired_match_statistics([base], [dict(base, **{field: value})], max_decisions=1440)
