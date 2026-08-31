from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def test_collector_import_is_safe_without_torch() -> None:
    module = importlib.import_module("rl.collector")
    assert hasattr(module, "CollectorConfig")
    if torch is None:
        with pytest.raises(module.TorchUnavailableError, match="PyTorch"):
            module.RecurrentRolloutCollector()


def test_collector_config_rejects_invalid_rollout_shape() -> None:
    from rl.collector import CollectorConfig

    assert CollectorConfig(horizon=3, target_player=1).horizon == 3
    with pytest.raises(ValueError, match="horizon"):
        CollectorConfig(horizon=0)
    with pytest.raises(ValueError, match="target_player"):
        CollectorConfig(target_player=2)
    with pytest.raises(ValueError, match="two eight-card"):
        CollectorConfig(decks=(("hog-rider",), ("cannon",)))
    with pytest.raises(ValueError, match="expert_execution_probability"):
        CollectorConfig(expert_execution_probability=1.1)
    with pytest.raises(TypeError, match="expert_label_on_disagreement"):
        CollectorConfig(expert_label_on_disagreement=1)


def test_collector_defaults_to_actor_controlled_rollouts() -> None:
    from rl.collector import CollectorConfig

    config = CollectorConfig()

    assert config.expert_execution_probability == 0.0
    assert config.expert_label_on_threat_only is False
    assert config.expert_label_on_disagreement is False


def test_fireball_teacher_label_has_decisive_sparse_card_weight() -> None:
    from rl.collector import _expert_action_weight

    environment = SimpleNamespace(
        state=SimpleNamespace(
            players=(SimpleNamespace(hand=("fireball",)), SimpleNamespace(hand=())),
        ),
    )
    action = SimpleNamespace(kind="Play", card_slot=0, cell=(3, 17))

    assert _expert_action_weight(environment, action, 0) == 20.0


def test_action_agreement_compares_only_public_decision_fields() -> None:
    from actions import PlayCardAction, WaitAction
    from rl.collector import _actions_match

    assert _actions_match(WaitAction(0), WaitAction(1))
    assert _actions_match(PlayCardAction(0, 2, (3, 17)), PlayCardAction(1, 2, (3, 17)))
    assert not _actions_match(PlayCardAction(0, 2, (3, 17)), PlayCardAction(0, 2, (3, 18)))
    assert not _actions_match(WaitAction(0), PlayCardAction(0, 2, (3, 17)))


def test_public_threat_label_gate_uses_only_visible_observation() -> None:
    from cr_bot.features.channels import GLOBAL_SCALAR_IDX
    from simulator.observation_v2 import ENTITY_TOKEN_FEATURES, PolicyObservationV2
    from rl.public_counter import public_defensive_threat_observed

    board = np.zeros((21, 32, 18), dtype=np.float32)
    global_vector = np.zeros((768,), dtype=np.float32)
    entity_tokens = np.zeros((128, 32), dtype=np.float32)
    entity_mask = np.zeros((128,), dtype=bool)
    legal_play = np.zeros((4, 32, 18), dtype=bool)

    quiet = PolicyObservationV2(
        board=board,
        global_vector=global_vector,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=legal_play,
        legal_wait=True,
    )
    assert public_defensive_threat_observed(quiet) is False

    feature_index = {name: index for index, name in enumerate(ENTITY_TOKEN_FEATURES)}
    entity_tokens[0, feature_index["side"]] = 1.0
    entity_tokens[0, feature_index["y"]] = 0.5
    entity_tokens[0, feature_index["is_visible"]] = 1.0
    entity_mask[0] = True
    crossed = PolicyObservationV2(
        board=board,
        global_vector=global_vector,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=legal_play,
        legal_wait=True,
    )
    assert public_defensive_threat_observed(crossed) is True

    entity_tokens[0] = 0.0
    entity_mask[0] = False
    global_vector[GLOBAL_SCALAR_IDX["tower_hp_self_left"]] = 0.5
    damaged_tower = PolicyObservationV2(
        board=board,
        global_vector=global_vector,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=legal_play,
        legal_wait=True,
    )
    assert public_defensive_threat_observed(damaged_tower) is True


def test_rollout_stats_can_attribute_terminal_results_to_lanes() -> None:
    from rl.collector import RolloutStats

    stats = RolloutStats(
        completed_matches=2,
        wins=1,
        draws=1,
        match_outcomes=((0, "win"), (1, "draw")),
    )

    assert stats.as_dict()["match_outcomes"] == [
        {"lane": 0, "outcome": "win"},
        {"lane": 1, "outcome": "draw"},
    ]


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


@requires_torch
def test_single_observation_batching_matches_general_stack() -> None:
    from simulator.observation_v2 import PolicyObservationV2
    from rl.collector import _batch_observations

    observation = PolicyObservationV2(
        board=np.zeros((21, 32, 18), dtype=np.float32),
        global_vector=np.zeros((768,), dtype=np.float32),
        entity_tokens=np.zeros((128, 32), dtype=np.float32),
        entity_mask=np.zeros((128,), dtype=bool),
        legal_play=np.zeros((4, 32, 18), dtype=bool),
        legal_wait=True,
    )

    regular = _batch_observations([observation], device=torch.device("cpu"))
    inference = _batch_observations(
        [observation],
        device=torch.device("cpu"),
        inference=True,
    )

    for regular_value, inference_value in zip(regular[:4], inference[:4], strict=True):
        torch.testing.assert_close(regular_value, inference_value)
    for regular_value, inference_value in zip(
        (regular[4].mode, regular[4].card, regular[4].placement),
        (inference[4].mode, inference[4].card, inference[4].placement),
        strict=True,
    ):
        torch.testing.assert_close(regular_value, inference_value)
    assert inference[0][:, 0].is_contiguous(memory_format=torch.channels_last)


@requires_torch
@pytest.mark.parametrize(
    ("expert_execution_probability", "expected"),
    ((0.0, False), (1.0, True)),
)
def test_teacher_execution_is_disabled_by_default_and_explicitly_opt_in(
    expert_execution_probability: float,
    expected: bool,
) -> None:
    from rl.collector import CollectorConfig, _expert_should_execute

    config = CollectorConfig(
        seed=17,
        expert_execution_probability=expert_execution_probability,
    )

    assert _expert_should_execute(config, lane=2, timestep=5) is expected


@requires_torch
def test_decode_actions_batches_device_values_before_building_policy_actions() -> None:
    from rl.collector import _decode_actions
    from rl.trajectory import ActionBatch

    actions = ActionBatch(
        mode=torch.tensor([[0], [1], [1]], dtype=torch.long),
        card_slot=torch.tensor([[3], [2], [0]], dtype=torch.long),
        placement=torch.tensor(
            [
                [[7, 8]],
                [[11, 4]],
                [[2, 16]],
            ],
            dtype=torch.long,
        ),
    )

    decoded = _decode_actions(actions)

    assert [(action.kind, action.card_idx, action.cell) for action in decoded] == [
        ("Wait", None, None),
        ("Play", 2, (4, 11)),
        ("Play", 0, (16, 2)),
    ]


@requires_torch
def test_collector_builds_trajectory_from_v2_environment() -> None:
    # Kept as an integration smoke test for environments where the optional
    # torch dependency is installed.  The repository's default environment
    # intentionally remains NumPy-only.
    from simulator.env import SimulatorEnv
    from rl.collector import CollectorConfig, RecurrentRolloutCollector
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import ModelConfig, RecurrentHybridPolicy

    model_config = ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
    )
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(model_config),
        config=LearnerConfig(update_epochs=1, sequence_minibatch_size=1),
    )
    environment = SimulatorEnv(decision_interval_us=1_000_000)
    environment.reset(seed=11, shuffle_decks=False)
    result = RecurrentRolloutCollector(
        learner,
        CollectorConfig(horizon=2, shuffle_decks=False),
    ).collect([environment])

    assert result.trajectory.batch_size == 1
    assert result.trajectory.time_steps == 2
    assert result.bootstrap_values.shape == (1,)


@requires_torch
def test_nonterminal_truncation_bootstraps_from_pre_reset_observation() -> None:
    from dataclasses import replace
    from types import MethodType

    from simulator.env import SimulatorEnv
    from rl.collector import CollectorConfig, RecurrentRolloutCollector
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import ModelConfig, RecurrentHybridPolicy

    model_config = ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
    )
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(model_config),
        config=LearnerConfig(update_epochs=1, sequence_minibatch_size=1),
    )
    environment = SimulatorEnv(decision_interval_us=1_000_000)
    environment.reset(seed=37, shuffle_decks=False)

    def batch_step(actions):
        step = environment.step_v2(actions[0])
        return [replace(step, terminated=False, truncated=True)]

    collector = RecurrentRolloutCollector(
        learner,
        CollectorConfig(horizon=1, shuffle_decks=False),
        batch_step=batch_step,
    )

    def deterministic_bootstrap(
        self,
        environments,
        observations,
        rollout_state,
        reset_mask,
        last_done,
    ):
        del observations, reset_mask, last_done
        value = 7.0 if environments[0].state.elapsed_us > 0 else 0.0
        return torch.tensor([value], dtype=torch.float32), rollout_state.detach()

    collector._bootstrap = MethodType(deterministic_bootstrap, collector)
    result = collector.collect([environment])

    assert bool(result.trajectory.truncated[0, 0])
    assert result.learner_batch.next_values is not None
    assert result.learner_batch.bootstrap_values is None
    assert float(result.learner_batch.next_values[0, 0]) == pytest.approx(7.0)
    assert float(result.bootstrap_values[0]) == pytest.approx(7.0)


@requires_torch
def test_mixed_terminal_and_truncated_lanes_sanitize_terminal_bootstrap_rows() -> None:
    from dataclasses import replace

    from simulator.env import SimulatorEnv
    from rl.collector import CollectorConfig, RecurrentRolloutCollector
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import ModelConfig, RecurrentHybridPolicy

    model_config = ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
    )
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(model_config),
        config=LearnerConfig(update_epochs=1, sequence_minibatch_size=1),
    )
    environments = [
        SimulatorEnv(decision_interval_us=1_000_000),
        SimulatorEnv(decision_interval_us=1_000_000),
    ]
    for lane, environment in enumerate(environments):
        environment.reset(seed=50 + lane, shuffle_decks=False)

    def batch_step(actions):
        results = []
        for lane, environment in enumerate(environments):
            step = environment.step_v2(actions[lane])
            if lane == 0:
                terminal_observation = replace(
                    step.observations[0],
                    legal_play=np.zeros_like(step.observations[0].legal_play, dtype=bool),
                    legal_wait=False,
                )
                step = replace(
                    step,
                    observations=(terminal_observation, step.observations[1]),
                    terminated=True,
                    truncated=False,
                )
            else:
                step = replace(step, terminated=False, truncated=True)
            results.append(step)
        return results

    result = RecurrentRolloutCollector(
        learner,
        CollectorConfig(horizon=1, shuffle_decks=False),
        batch_step=batch_step,
    ).collect(environments)

    assert bool(result.trajectory.terminated[0, 0])
    assert bool(result.trajectory.truncated[1, 0])
    assert result.learner_batch.next_values is not None
    assert float(result.learner_batch.next_values[0, 0]) == pytest.approx(0.0)


@requires_torch
def test_collector_reports_completed_rollout_steps() -> None:
    from simulator.env import SimulatorEnv
    from rl.collector import CollectorConfig, RecurrentRolloutCollector
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import ModelConfig, RecurrentHybridPolicy

    model_config = ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
    )
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(model_config),
        config=LearnerConfig(update_epochs=1, sequence_minibatch_size=1),
    )
    environments = [
        SimulatorEnv(decision_interval_us=1_000_000),
        SimulatorEnv(decision_interval_us=1_000_000),
    ]
    for lane, environment in enumerate(environments):
        environment.reset(seed=20 + lane, shuffle_decks=False)
    completed: list[int] = []
    RecurrentRolloutCollector(
        learner,
        CollectorConfig(horizon=3, shuffle_decks=False),
    ).collect(environments, step_callback=completed.append)

    assert completed == [2, 4, 6]


@requires_torch
def test_collector_records_privileged_and_belief_targets_for_learner_update() -> None:
    from simulator.env import SimulatorEnv
    from rl.collector import CollectorConfig, RecurrentRolloutCollector
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import ModelConfig, RecurrentHybridPolicy

    model_config = ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
    )
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(model_config),
        config=LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=1,
            belief_coef=0.1,
            require_privileged_critic=True,
        ),
        privileged_dim=2,
    )
    environment = SimulatorEnv(decision_interval_us=1_000_000)
    environment.reset(seed=12, shuffle_decks=False)
    collector = RecurrentRolloutCollector(
        learner,
        CollectorConfig(horizon=1, shuffle_decks=False, collect_belief_targets=True),
        privileged_feature_fn=lambda env, _viewer: (
            env.state.players[0].elixir_milli / 10_000.0,
            env.state.players[1].elixir_milli / 10_000.0,
        ),
    )

    result = collector.collect([environment])

    assert result.learner_batch.privileged_features.shape == (1, 1, 2)
    assert result.learner_batch.belief_targets is not None
    metrics = learner.update(result.learner_batch)
    assert metrics.optimization_steps == 1
