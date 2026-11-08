"""Replay identity, frozen selection and upstream public-history mining."""
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rl.v4 import replay_corrections as repairs
from simulator.rl.v4.frontier_eval import BranchSpec


def test_search_config_rejects_insufficient_confirmation_and_invalid_budgets():
    for kwargs in (dict(discovery_replicas=1), dict(confirmation_replicas=1),
                   dict(confirmation_horizon=64), dict(min_lcb=float("nan")),
                   dict(max_branches=2), dict(train_budget=0), dict(seed=-1)):
        with pytest.raises(ValueError):
            repairs.CorrectionSearchConfig(**kwargs)


@pytest.fixture
def branching(monkeypatch):
    import simulator.rl.v4.frontier_eval as frontier
    calls = []
    env = SimpleNamespace(state=SimpleNamespace(state_hash=lambda: "origin"))
    branches = [BranchSpec("sim", 0), BranchSpec("siege-geometry", 1, 1, 18, 4), BranchSpec("other", 1, 2, 18, 5)]
    config = repairs.CorrectionSearchConfig(discovery_horizon=4, confirmation_horizon=8,
                                          discovery_replicas=2, confirmation_replicas=3)

    def stub(**kwargs):
        calls.append(kwargs)
        mode = kwargs["branch"].mode
        value = 0.0 if mode == 0 else 1.0 if kwargs["branch"].slot == 1 else .5
        return dict(value=value, winner=None, terminal=False, decisions_run=kwargs["horizon"])
    monkeypatch.setattr(frontier, "run_branch", stub)
    return env, branches, config, calls


def test_discovery_winner_is_frozen_and_confirmed_on_disjoint_paired_seeds(branching):
    env, branches, config, calls = branching
    result = repairs.evaluate_correction(env=env, hidden=None, policy=None,
        candidate=dict(t=4, state_hash="origin"), branches=branches,
        opponent_spec=dict(strategy="siege-bait", style="patient", controller_seed=7), config=config)
    assert result["accepted"] and result["target"] == [1, 1, 18, 4, 0]
    assert len(calls) == 2 * 3 + 3 * 2
    early = calls[:6]
    late = calls[6:]
    assert all(call["horizon"] == 4 for call in early)
    assert all(call["horizon"] == 8 for call in late)
    assert {call["branch"].slot for call in late if call["branch"].mode} == {1}
    assert not {call["replica_seed"] for call in early} & {call["replica_seed"] for call in late}
    for index in range(0, len(early), 3):
        assert len({call["replica_seed"] for call in early[index:index + 3]}) == 1
    assert all(call["opponent_spec"]["style"] == "patient" for call in calls)


def test_confirmation_cannot_reselect_a_different_action(branching, monkeypatch):
    env, branches, config, calls = branching
    import simulator.rl.v4.frontier_eval as frontier
    original = frontier.run_branch

    def reversed_value(**kwargs):
        result = original(**kwargs)
        if kwargs["horizon"] == 8 and kwargs["branch"].mode:
            result["value"] = -1
        return result
    monkeypatch.setattr(frontier, "run_branch", reversed_value)
    result = repairs.evaluate_correction(env=env, hidden=None, policy=None,
        candidate=dict(t=4, state_hash="origin"), branches=branches,
        opponent_spec=dict(strategy="siege-bait"), config=config)
    assert not result["accepted"] and result["target"] is None
    assert result["verdict"]["frozen_action"]["slot"] == 1
    assert result["statistics"]["mean"] == -1


def test_recorded_best_does_not_create_training_label(branching, monkeypatch):
    env, branches, config, _ = branching
    import simulator.rl.v4.frontier_eval as frontier
    monkeypatch.setattr(frontier, "run_branch", lambda **kwargs: dict(value=0.0))
    result = repairs.evaluate_correction(env=env, hidden=None, policy=None,
        candidate=dict(t=1, state_hash="origin"), branches=branches,
        opponent_spec=dict(strategy="siege-bait"), config=config)
    assert not result["accepted"]
    assert result["rejection"] == "discovery_kept_recorded_action"
    assert result["statistics"]["se"] is None


def test_origin_and_opponent_restoration_boundaries_fail_closed(branching):
    env, branches, config, _ = branching
    kwargs = dict(env=env, hidden=None, policy=None, candidate=dict(t=1, state_hash="origin"),
                  branches=branches, opponent_spec=dict(strategy="siege-bait"), config=config)
    with pytest.raises(ValueError, match="origin"):
        repairs.evaluate_correction(**dict(kwargs, candidate=dict(t=1, state_hash="wrong")))
    with pytest.raises(ValueError, match="distinct"):
        repairs.evaluate_correction(**dict(kwargs, branches=[branches[0], branches[0]]))
    with pytest.raises(ValueError, match="learned"):
        repairs.evaluate_correction(**dict(kwargs, opponent_spec=dict(kind="v4-checkpoint")))


def test_distinct_replica_ids_cannot_hide_colliding_actual_seeds(branching, monkeypatch):
    env, branches, config, _ = branching
    monkeypatch.setattr("simulator.rl.v4.frontier._stable_seed", lambda *_args: 1)
    with pytest.raises(ValueError, match="seeds collided"):
        repairs.evaluate_correction(env=env, hidden=None, policy=None,
            candidate=dict(t=4, state_hash="origin"), branches=branches,
            opponent_spec=dict(strategy="siege-bait"), config=config)


def test_upstream_review_selects_earlier_spend_without_adding_future_inputs(monkeypatch):
    from simulator.rl.v4 import correction_proposals
    length = 70
    arrays = {key: np.zeros((length, 1)) for key in repairs.PUBLIC_KEYS}
    arrays.update(entities=np.zeros((length, 128, 32)), entity_mask=np.zeros((length, 128), bool),
                  hand_tokens=np.zeros((length, 4, 16)), global_features=np.zeros((length, 768)))
    arrays.update(label_valid=np.ones(length, bool), own_elixir=np.ones(length),
                  state_hash=np.asarray([str(i) for i in range(length)]),
                  executed_actions=np.zeros((length, 5), np.int64), outcome_values=np.zeros((length, 3, 4)))
    arrays["executed_actions"][10] = [1, 0, 17, 3, 0]
    arrays["global_features"][50, 0] = 1
    monkeypatch.setattr(correction_proposals, "candidate_categories", lambda inputs, *_args, **_kwargs:
                        ("air",) if inputs["global_features"][0] else ())
    before = {key: value.copy() for key, value in arrays.items()}
    result = repairs.select_episode_candidates(arrays, dict(job_id="train_actor_0"), repairs.CorrectionSearchConfig())
    assert any(row["t"] == 10 and "upstream-spend" in row["tags"] for row in result)
    assert any(row["t"] == 50 and "air" in row["tags"] for row in result)
    assert all(np.array_equal(arrays[key], value) for key, value in before.items())


def test_first_affordable_siege_response_is_reserved_before_later_damage():
    from cr_bot.domain.card_metadata import CARD_METADATA
    from simulator.rl.v4.hand_features import build_hand_tokens
    length = 100
    arrays = {key: np.zeros((length, 1)) for key in repairs.PUBLIC_KEYS}
    arrays.update(entities=np.zeros((length, 128, 32)), entity_mask=np.zeros((length, 128), bool),
                  hand_tokens=np.tile(build_hand_tokens(("cannon", "ice-golem", "musketeer", "log"), 5), (length, 1, 1)),
                  global_features=np.zeros((length, 768)), legal_play=np.ones((length, 4, 32, 18), bool),
                  legal_wait=np.ones(length, bool), label_valid=np.ones(length, bool), own_elixir=np.full(length, 5.),
                  state_hash=np.asarray([str(i) for i in range(length)]),
                  executed_actions=np.zeros((length, 5), np.int64), outcome_values=np.zeros((length, 3, 4)))
    maximum = max(int(row["id"]) for row in CARD_METADATA.values())
    arrays["entities"][10:80, 0, :11] = [CARD_METADATA["x-bow"]["id"] / maximum,
                                        1, 4.5 / 18, 13.5 / 32, 1, 0, 1, 0, 0, 1, 1]
    arrays["entity_mask"][10:80, 0] = True
    arrays["outcome_values"][60, -1, 0] = .9
    result = repairs.select_episode_candidates(arrays, dict(job_id="siege"), repairs.CorrectionSearchConfig(max_per_episode=1))
    assert len(result) == 1 and result[0]["t"] == 10
    assert "interception-window" in result[0]["tags"]


def test_report_does_not_claim_an_unrun_experiment_is_complete(tmp_path):
    result = repairs.report(tmp_path, dict(plan_sha256="plan", checkpoint=dict(sha256="actor")))
    assert result["status"] == "partial"
    import json
    assert json.loads((tmp_path / "progress.json").read_text())["stage"] == "partial"


def test_exact_prefix_keeps_countdown_and_all_public_history():
    import torch
    from simulator.actions import WaitAction
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.rl.v4.frontier_eval import _fresh_env
    from simulator.rl.v4.tactical_data import build_collection_jobs
    from simulator.rl.v4.bootstrap import catalog_training_opponent_specs
    from simulator.rl.v4.v4_ppo.runtime import observe_v4_inputs

    job = build_collection_jobs(catalog_training_opponent_specs()[:1], 223, drivers=("actor",), calibration_matches=0)[0]
    from dataclasses import asdict
    metadata = dict(asdict(job), native_actor_memory="carry")
    arrays = dict(state_hash=np.asarray(["pending"]), executed_actions=np.zeros((1, 5), np.int64),
                  opponent_actions=np.zeros((1, 5), np.int64))
    env = _fresh_env(repairs.episode_match(arrays, metadata))
    estimator = PublicStateEstimator()
    values = {key: [] for key in repairs.PUBLIC_KEYS}
    hashes = []
    for _ in range(16):
        hashes.append(env.state.state_hash())
        _, inputs, _, _ = observe_v4_inputs(env, estimator)
        for key in values:
            values[key].append(np.array(inputs[key], copy=True))
        env.step_v2_for_viewer((WaitAction(0), WaitAction(1)), viewer=0)
    arrays = {key: np.asarray(value) for key, value in values.items()}
    arrays.update(state_hash=np.asarray(hashes), executed_actions=np.zeros((16, 5), np.int64),
                  opponent_actions=np.zeros((16, 5), np.int64))

    class Policy:
        config = SimpleNamespace()
        def __init__(self):
            self.calls = []
        def eval(self):
            return self
        def initial_hidden(self, batch_size, device=None):
            return torch.zeros(1, batch_size, 1)
        def forward_step(self, inputs, hidden, reset_mask):
            self.calls.append(bool(reset_mask.item()))
            return None, None, hidden + 1

    policy = Policy()
    from unittest.mock import patch
    with patch("simulator.rl.v4.v4_ppo.runtime.configure_v4_runtime"):
        kept = repairs.replay_episode_states(arrays, metadata, [0, 15], policy)
    assert policy.calls == [True] + [False] * 15
    assert kept[0][1].item() == 1 and kept[15][1].item() == 16
    assert kept[15][0].state.state_hash() == hashes[15]
    arrays["event_history"][5, 0, 0] += 1
    with patch("simulator.rl.v4.v4_ppo.runtime.configure_v4_runtime"):
        with pytest.raises(ValueError, match="public prefix input event_history"):
            repairs.replay_episode_states(arrays, metadata, [15], Policy())
