from __future__ import annotations

import importlib
import math

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def test_learner_import_is_safe_without_torch() -> None:
    module = importlib.import_module("rl.learner")
    assert hasattr(module, "LearnerConfig")
    if torch is None:
        with pytest.raises(module.TorchUnavailableError, match="PyTorch"):
            module.RecurrentPPOLearner()


def test_learner_config_validates_long_horizon_defaults() -> None:
    from rl.learner import LearnerConfig
    from rl.objectives import PPOObjectiveConfig

    config = LearnerConfig()
    assert config.gamma > 0.99
    assert config.sequence_minibatch_size > 0
    with pytest.raises(ValueError, match="sequence_minibatch_size"):
        LearnerConfig(sequence_minibatch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LearnerConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="imitation_only"):
        LearnerConfig(imitation_only="yes")
    with pytest.raises(ValueError, match="clip_epsilon"):
        PPOObjectiveConfig(clip_epsilon=float("inf"))


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _model_config():
    from rl.model import ModelConfig

    return ModelConfig(
        raster_channels=2,
        raster_height=6,
        raster_width=5,
        global_dim=4,
        entity_dim=3,
        max_entities=5,
        model_dim=8,
        encoder_dim=7,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=9,
        gru_layers=2,
        card_slots=3,
        belief_card_count=6,
        placement_rows=4,
        placement_cols=3,
    )


def _trajectory(config, *, batch=3, time=4):
    from rl.trajectory import ActionBatch, ActionMasks, RecurrentSequence, TrajectoryBatch

    raster = torch.randn(batch, time, config.raster_channels, config.raster_height, config.raster_width)
    global_features = torch.randn(batch, time, config.global_dim)
    entities = torch.randn(batch, time, 4, config.entity_dim)
    entity_mask = torch.ones(batch, time, 4, dtype=torch.bool)
    reset_mask = torch.zeros(batch, time, dtype=torch.bool)
    reset_mask[:, 0] = True
    sequence = RecurrentSequence(
        raster=raster,
        global_features=global_features,
        entities=entities,
        entity_mask=entity_mask,
        reset_mask=reset_mask,
    )
    # A WAIT-only mask keeps the fixture independent of card-placement rules
    # while still exercising the structured action log-probability path.
    mode = torch.zeros(batch, time, 2, dtype=torch.bool)
    mode[..., 0] = True
    card = torch.zeros(batch, time, config.card_slots, dtype=torch.bool)
    placement = torch.zeros(
        batch,
        time,
        config.card_slots,
        config.placement_rows,
        config.placement_cols,
        dtype=torch.bool,
    )
    actions = ActionBatch(
        mode=torch.zeros(batch, time, dtype=torch.long),
        card_slot=torch.zeros(batch, time, dtype=torch.long),
        placement=torch.zeros(batch, time, 2, dtype=torch.long),
    )
    masks = ActionMasks(mode=mode, card=card, placement=placement)
    return TrajectoryBatch(
        sequence=sequence,
        action_masks=masks,
        actions=actions,
        rewards=torch.zeros(batch, time),
        terminated=torch.zeros(batch, time, dtype=torch.bool),
        truncated=torch.zeros(batch, time, dtype=torch.bool),
        old_log_probs=torch.zeros(batch, time),
        values=torch.zeros(batch, time),
        advantages=torch.ones(batch, time),
        returns=torch.ones(batch, time),
    )


@requires_torch
def test_sequence_minibatches_preserve_rows_and_hidden_snapshots() -> None:
    from rl.learner import BeliefTargets, LearnerBatch, iter_sequence_minibatches

    config = _model_config()
    trajectory = _trajectory(config, batch=5, time=4)
    layers = config.gru_layers
    hidden = config.gru_hidden_dim
    trajectory = trajectory.__class__(
        sequence=trajectory.sequence.__class__(
            raster=trajectory.sequence.raster,
            global_features=trajectory.sequence.global_features,
            entities=trajectory.sequence.entities,
            entity_mask=trajectory.sequence.entity_mask,
            reset_mask=trajectory.sequence.reset_mask,
            hidden_states=torch.randn(5, 4, layers, hidden),
            initial_hidden=torch.randn(layers, 5, hidden),
        ),
        action_masks=trajectory.action_masks,
        actions=trajectory.actions,
        rewards=trajectory.rewards,
        terminated=trajectory.terminated,
        truncated=trajectory.truncated,
        old_log_probs=trajectory.old_log_probs,
        values=trajectory.values,
        advantages=trajectory.advantages,
        returns=trajectory.returns,
    )
    batch = LearnerBatch(
        trajectory=trajectory,
        privileged_features=torch.randn(5, 4, 2),
        belief_targets=BeliefTargets(enemy_elixir=torch.zeros(5, 4)),
    )

    minibatches = list(
        iter_sequence_minibatches(
            batch,
            minibatch_size=2,
            shuffle=False,
            sequence_length=2,
        )
    )
    assert [item.batch_size for item in minibatches] == [2, 2, 1, 2, 2, 1]
    assert all(item.time_steps == 2 for item in minibatches)
    assert minibatches[0].trajectory.sequence.initial_hidden.shape == (layers, 2, hidden)
    torch.testing.assert_close(
        minibatches[0].trajectory.sequence.initial_hidden,
        trajectory.sequence.initial_hidden[:, :2],
    )
    torch.testing.assert_close(
        minibatches[3].trajectory.sequence.initial_hidden,
        trajectory.sequence.hidden_states[:2, 2].permute(1, 0, 2),
    )


@requires_torch
def test_privileged_recurrent_update_belief_loss_and_checkpoint_round_trip(tmp_path) -> None:
    from rl.learner import BeliefTargets, LearnerBatch, LearnerConfig, RecurrentPPOLearner
    from rl.model import PrivilegedCritic, RecurrentHybridPolicy

    config = _model_config()
    policy = RecurrentHybridPolicy(config)
    critic = PrivilegedCritic(config.gru_hidden_dim, privileged_dim=2)
    learner = RecurrentPPOLearner(
        policy,
        critic,
        LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=2,
            shuffle_sequences=False,
            belief_coef=0.1,
        ),
    )
    trajectory = _trajectory(config, batch=3, time=4)
    privileged = torch.randn(3, 4, 2)
    targets = BeliefTargets(
        enemy_elixir=torch.rand(3, 4),
        enemy_hand=torch.zeros(3, 4, config.belief_card_count),
        enemy_next_card=torch.zeros(3, 4, dtype=torch.long),
    )
    metrics = learner.update(
        LearnerBatch(
            trajectory=trajectory,
            privileged_features=privileged,
            belief_targets=targets,
        )
    )
    assert metrics.update_index == 1
    assert metrics.minibatches == 2
    assert metrics.optimization_steps == 2
    assert metrics.belief_loss >= 0.0

    checkpoint = learner.checkpoint_state()
    assert {"policy", "critic", "optimizer", "update_count"}.issubset(checkpoint)
    restored = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        PrivilegedCritic(config.gru_hidden_dim, privileged_dim=2),
        learner.config,
    )
    restored.load_checkpoint_state(checkpoint)
    assert restored.update_count == learner.update_count
    for left, right in zip(learner.policy.parameters(), restored.policy.parameters()):
        torch.testing.assert_close(left, right)

    path = tmp_path / "learner.pt"
    learner.save_checkpoint(path)
    assert path.exists()


@requires_torch
def test_rollout_state_resets_hidden_before_next_observation() -> None:
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import RecurrentHybridPolicy
    from rl.trajectory import ActionMasks

    config = _model_config()
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=LearnerConfig(update_epochs=1),
    )
    state = learner.initial_rollout_state(2)
    mode = torch.zeros(2, 2, dtype=torch.bool)
    mode[:, 0] = True  # WAIT-only: no card/placement masks are legal here.
    card = torch.zeros(2, config.card_slots, dtype=torch.bool)
    placement = torch.zeros(
        2,
        config.card_slots,
        config.placement_rows,
        config.placement_cols,
        dtype=torch.bool,
    )
    masks = ActionMasks(mode=mode, card=card, placement=placement)
    reset = torch.tensor([True, False])
    reset_state = state.reset(reset)
    assert torch.count_nonzero(reset_state.hidden[:, 0]).item() == 0
    assert torch.equal(reset_state.hidden[:, 1], state.hidden[:, 1])
    step = learner.rollout_step(
        state,
        torch.randn(2, config.raster_channels, config.raster_height, config.raster_width),
        torch.randn(2, config.global_dim),
        torch.randn(2, 4, config.entity_dim),
        torch.ones(2, 4, dtype=torch.bool),
        masks,
        reset_mask=reset,
        deterministic=True,
    )
    assert step.actions.mode.shape == (2, 1)
    assert step.next_state.hidden.shape == state.hidden.shape


@requires_torch
def test_nonfinite_objective_is_rejected_before_ratio_can_overflow() -> None:
    from rl.objectives import ppo_objective

    tensors = {
        "old_log_probs": torch.zeros(1, 1),
        "new_log_probs": torch.zeros(1, 1),
        "advantages": torch.ones(1, 1),
        "values": torch.zeros(1, 1),
        "returns": torch.zeros(1, 1),
        "entropy": torch.zeros(1, 1),
    }
    with pytest.raises(FloatingPointError, match="non-finite"):
        ppo_objective(**{**tensors, "advantages": torch.full((1, 1), float("nan"))})
    with pytest.raises(FloatingPointError, match="probability ratio"):
        ppo_objective(**{**tensors, "old_log_probs": torch.full((1, 1), -1000.0)})


@requires_torch
def test_nonfinite_gradient_skips_optimizer_step_and_is_counted() -> None:
    from rl.learner import LearnerBatch, LearnerConfig, RecurrentPPOLearner
    from rl.model import RecurrentHybridPolicy

    config = _model_config()
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=1,
            shuffle_sequences=False,
        ),
    )
    trajectory = _trajectory(config, batch=1, time=2)
    parameter = learner.critic.value[-1].bias
    parameters = list(learner.policy.parameters()) + list(learner.critic.parameters())
    before = [value.detach().clone() for value in parameters]
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    try:
        metrics = learner.update(LearnerBatch(trajectory=trajectory))
    finally:
        hook.remove()

    assert metrics.optimization_steps == 0
    assert metrics.skipped_steps == 1
    assert all(bool(torch.isfinite(value).all()) for value in parameters)
    assert all(torch.equal(left, right) for left, right in zip(before, parameters))
    assert learner.optimizer.state == {}


@requires_torch
def test_gradient_norm_accumulates_large_finite_gradients_without_overflow() -> None:
    from rl.learner import _gradient_norm

    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1.0e20, -1.0e20])
    norm = _gradient_norm([parameter])

    assert math.isfinite(norm)
    assert norm == pytest.approx(math.sqrt(2.0) * 1.0e20, rel=1.0e-6)


@requires_torch
def test_corrupt_checkpoint_is_rejected_before_resume() -> None:
    from rl.learner import LearnerConfig, RecurrentPPOLearner
    from rl.model import RecurrentHybridPolicy

    config = _model_config()
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=LearnerConfig(update_epochs=1),
    )
    state = learner.checkpoint_state()
    key = next(name for name, value in state["policy"].items() if value.is_floating_point())
    state["policy"][key].reshape(-1)[0] = float("nan")

    restored = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=learner.config,
    )
    with pytest.raises(ValueError, match="non-finite"):
        restored.load_checkpoint_state(state)
