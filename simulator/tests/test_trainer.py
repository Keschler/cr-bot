from __future__ import annotations

import json

import numpy as np
import pytest

from simulator.cli import main
from simulator.env import EnvStep, SimulatorEnv
from simulator.env import RewardConfig
from simulator.trainer import (
    ACTION_COUNT,
    DEFAULT_DECISION_INTERVAL_US,
    WAIT_INDEX,
    FactorizedPolicy,
    PPOConfig,
    PPOTrainer,
    TrainingConfigurationError,
    action_index_to_policy_action,
    evaluate_policy,
    full_match_decisions,
    time_aware_discount,
)
from simulator.ruleset import load_ruleset


def test_action_mapping_and_masked_sampling_never_selects_an_illegal_cell() -> None:
    observation = SimulatorEnv().reset(seed=4, shuffle_decks=False)[0]
    policy = FactorizedPolicy(seed=4)
    sample = policy.sample(observation, np.random.default_rng(9))
    assert sample.action_index == WAIT_INDEX or bool(observation.legal_play.reshape(-1)[sample.action_index])
    assert ACTION_COUNT == 4 * 32 * 18 + 1
    assert action_index_to_policy_action(WAIT_INDEX).kind == "Wait"


def test_factorized_policy_checkpoint_round_trip_preserves_outputs(tmp_path) -> None:
    observation = SimulatorEnv().reset(seed=8, shuffle_decks=False)[0]
    policy = FactorizedPolicy(seed=8)
    path = policy.save(tmp_path / "policy.npz", metadata={"ruleset_id": "v1"})
    restored, metadata = FactorizedPolicy.load(path)
    np.testing.assert_allclose(policy.raw_logits_value(observation)[0], restored.raw_logits_value(observation)[0])
    assert policy.raw_logits_value(observation)[1] == restored.raw_logits_value(observation)[1]
    assert metadata["ruleset_id"] == "v1"


def test_short_smoke_training_writes_auditable_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "run.npz"
    trainer = PPOTrainer(
        PPOConfig(
            num_envs=2,
            rollout_steps=2,
            total_steps=4,
            update_epochs=1,
            checkpoint_every=4,
            eval_every=4,
            eval_episodes=1,
            eval_max_decisions=2,
            seed=12,
            checkpoint_out=checkpoint,
            allow_provisional_smoke=True,
        )
    )
    report = trainer.train()
    assert report["total_steps"] == 4
    assert report["updates"] == 1
    assert report["illegal_action_attempts"] == 0
    assert checkpoint.exists()


def test_smoke_training_honors_non_divisible_transition_budget(tmp_path) -> None:
    for requested_steps in (1, 3, 5):
        trainer = PPOTrainer(
            PPOConfig(
                num_envs=2,
                rollout_steps=2,
                total_steps=requested_steps,
                update_epochs=1,
                checkpoint_every=100,
                eval_every=100,
                eval_episodes=1,
                seed=21 + requested_steps,
                checkpoint_out=tmp_path / f"run-{requested_steps}.npz",
                allow_provisional_smoke=True,
            )
        )
        report = trainer.train()
        assert report["total_steps"] == requested_steps


def test_checkpoint_evaluation_is_json_safe(tmp_path) -> None:
    policy = FactorizedPolicy(seed=2)
    path = policy.save(tmp_path / "policy.npz", metadata={"ruleset_id": "v1"})
    restored, _ = FactorizedPolicy.load(path)
    report = evaluate_policy(restored, episodes=1, seed_start=20, max_decisions=1)
    assert report["episodes"] == 1
    assert report["completed"] == 0
    assert report["truncated"] == 1
    assert report["draws"] + report["wins"] + report["losses"] == 0
    assert report["win_rate"] == report["loss_rate"] == report["draw_rate"] == 0.0


def test_evaluation_rates_use_completed_matches_only(monkeypatch) -> None:
    import simulator.trainer as trainer_module

    observations = SimulatorEnv().reset(seed=33, shuffle_decks=False)
    episode_number = 0

    class FakeEnvironment:
        def __init__(self, completed: bool) -> None:
            self.completed = completed

        def reset(self, *, seed: int, shuffle_decks: bool) -> tuple:
            return observations

        def step(self, actions) -> EnvStep:
            return EnvStep(
                observations=observations,
                rewards=(0.0, 0.0),
                terminated=self.completed,
                truncated=False,
                info={"winner": 0} if self.completed else {},
            )

    def fake_new_environment(ruleset, config):
        nonlocal episode_number
        environment = FakeEnvironment(completed=episode_number == 0)
        episode_number += 1
        return environment

    monkeypatch.setattr(trainer_module, "_new_environment", fake_new_environment)
    report = evaluate_policy(
        FactorizedPolicy(seed=34),
        opponent="self-play",
        episodes=2,
        max_decisions=1,
    )

    assert report["completed"] == 1
    assert report["truncated"] == 1
    assert report["wins"] == 1
    assert report["losses"] == report["draws"] == 0
    assert report["win_rate"] == 1.0
    assert report["loss_rate"] == report["draw_rate"] == 0.0


def test_full_match_evaluation_budget_is_derived_from_ruleset_duration() -> None:
    ruleset = load_ruleset("v1")
    duration_us = ruleset.match.regulation_us + ruleset.match.overtime_us
    expected = (duration_us + DEFAULT_DECISION_INTERVAL_US - 1) // DEFAULT_DECISION_INTERVAL_US

    assert PPOConfig().eval_max_decisions is None
    assert full_match_decisions(ruleset) == expected
    assert full_match_decisions(ruleset, decision_interval_us=500_000) == duration_us // 500_000


def test_time_aware_discount_is_validated_and_can_drive_effective_gamma() -> None:
    expected = float(np.exp(-0.25))
    assert time_aware_discount(250_000, 1_000_000) == pytest.approx(expected)
    config = PPOConfig(discount_time_constant_us=1_000_000)
    assert config.effective_gamma == pytest.approx(expected)

    with pytest.raises(TrainingConfigurationError, match="discount_time_constant_us"):
        PPOConfig(discount_time_constant_us=0)


def test_serious_trainer_defaults_to_sparse_terminal_outcome_reward() -> None:
    config = PPOConfig(allow_provisional_smoke=True)
    reward = RewardConfig.terminal_outcome()

    assert config.reward_version == "terminal-outcome-v1"
    assert config.gamma == 1.0
    assert reward.version == config.reward_version
    assert reward.tower_damage_weight == reward.crown_weight == 0.0
    assert reward.win_weight == 1.0


def test_terminal_potential_reward_is_bounded_and_auditable() -> None:
    reward = RewardConfig.terminal_with_potential(0.1)

    assert reward.version == "terminal-potential-v1"
    assert reward.tower_damage_weight == pytest.approx(0.1)
    assert reward.crown_weight == pytest.approx(0.1)
    assert reward.win_weight == pytest.approx(1.0)
    assert reward.as_dict() == {
        "version": "terminal-potential-v1",
        "tower_damage_weight": 0.1,
        "crown_weight": 0.1,
        "win_weight": 1.0,
    }
    with pytest.raises(ValueError, match="potential shaping weight"):
        RewardConfig.terminal_with_potential(0.0)


def test_train_cli_is_fail_closed_for_provisional_ruleset(tmp_path) -> None:
    checkpoint = tmp_path / "must-not-exist.npz"

    with pytest.raises(SystemExit, match="not training-ready"):
        main(
            [
                "--ruleset",
                "v1",
                "train",
                "--steps",
                "1",
                "--envs",
                "1",
                "--rollout-steps",
                "1",
                "--checkpoint-out",
                str(checkpoint),
            ]
        )

    assert not checkpoint.exists()


def test_train_and_evaluate_cli_round_trip_is_deterministic(tmp_path) -> None:
    checkpoint = tmp_path / "cli-policy.npz"
    training_summary = tmp_path / "training.json"
    evaluation_a = tmp_path / "evaluation-a.json"
    evaluation_b = tmp_path / "evaluation-b.json"
    train_args = [
        "--ruleset",
        "v1",
        "train",
        "--steps",
        "2",
        "--envs",
        "1",
        "--rollout-steps",
        "1",
        "--update-epochs",
        "1",
        "--checkpoint-every",
        "2",
        "--eval-every",
        "2",
        "--eval-episodes",
        "1",
        "--eval-max-decisions",
        "1",
        "--seed",
        "22",
        "--allow-provisional-smoke",
        "--checkpoint-out",
        str(checkpoint),
        "--json-out",
        str(training_summary),
    ]
    assert main(train_args) == 0
    assert checkpoint.exists()
    trained = json.loads(training_summary.read_text(encoding="utf-8"))
    assert trained["kind"] == "simulator_ppo_smoke_training"
    assert trained["total_steps"] == 2
    assert trained["provisional_smoke"] is True

    evaluate_args = [
        "--ruleset",
        "v1",
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        "1",
        "--seed",
        "90210",
        "--max-decisions",
        "1",
    ]
    assert main([*evaluate_args, "--json-out", str(evaluation_a)]) == 0
    assert main([*evaluate_args, "--json-out", str(evaluation_b)]) == 0
    first = json.loads(evaluation_a.read_text(encoding="utf-8"))
    second = json.loads(evaluation_b.read_text(encoding="utf-8"))
    assert first == second
    assert first["kind"] == "simulator_ppo_checkpoint_evaluation"
    assert first["checkpoint"] == str(checkpoint)
    assert first["episodes"] == 1
    assert first["completed"] == 0
    assert first["truncated"] == 1
    assert first["wins"] + first["losses"] + first["draws"] == 0
