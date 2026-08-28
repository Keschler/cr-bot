from __future__ import annotations

import importlib

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


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


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
