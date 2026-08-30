from __future__ import annotations

from dataclasses import replace
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


@requires_torch
def test_policy_device_auto_uses_visible_accelerator_or_cpu_fallback() -> None:
    from rl.learner import resolve_policy_device

    assert resolve_policy_device("cpu") == torch.device("cpu")
    selected = resolve_policy_device("auto")
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert selected == expected


@requires_torch
def test_policy_cpu_thread_cap_preserves_lower_setting_and_skips_accelerator() -> None:
    from rl.learner import configure_policy_cpu_threads

    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        assert configure_policy_cpu_threads(torch.device("cpu")) == 2
        assert torch.get_num_threads() == 2
        assert configure_policy_cpu_threads(torch.device("cuda")) is None
    finally:
        torch.set_num_threads(original_threads)


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
def test_compact_entity_tail_handles_empty_sparse_and_interior_masks() -> None:
    from rl.learner import _compact_entity_tail

    config = _model_config()
    sequence = _trajectory(config, batch=2, time=3).sequence

    empty = replace(
        sequence,
        entity_mask=torch.zeros_like(sequence.entity_mask),
    )
    compact_empty = _compact_entity_tail(empty)
    assert compact_empty.entities.shape[2] == 1
    assert not bool(compact_empty.entity_mask.any().item())

    sparse_mask = torch.zeros_like(sequence.entity_mask)
    sparse_mask[..., :2] = True
    sparse = replace(sequence, entity_mask=sparse_mask)
    compact_sparse = _compact_entity_tail(sparse)
    assert compact_sparse.entities.shape[2] == 2
    torch.testing.assert_close(compact_sparse.entity_mask, sparse_mask[..., :2])

    interior_mask = torch.zeros_like(sequence.entity_mask)
    interior_mask[..., :3] = True
    interior_mask[..., 1] = False
    interior = replace(sequence, entity_mask=interior_mask)
    compact_interior = _compact_entity_tail(interior)
    assert compact_interior.entities.shape[2] == 3
    torch.testing.assert_close(compact_interior.entity_mask, interior_mask[..., :3])
    torch.testing.assert_close(compact_interior.entities, sequence.entities[..., :3, :])


@requires_torch
def test_compact_entity_tail_preserves_policy_logprob_and_gradients() -> None:
    from rl.learner import (
        _compact_entity_tail,
        _joint_entropy,
        _joint_log_prob_and_entropy,
    )
    from rl.model import RecurrentHybridPolicy
    from rl.trajectory import ActionBatch, ActionMasks

    config = _model_config()
    trajectory = _trajectory(config, batch=2, time=3)
    sequence = trajectory.sequence
    mask = torch.zeros_like(sequence.entity_mask)
    mask[..., :3] = True
    mask[..., 1] = False
    sequence = replace(sequence, entity_mask=mask)
    compact = _compact_entity_tail(sequence)
    action_masks = ActionMasks(
        mode=torch.ones(2, 3, 2, dtype=torch.bool),
        card=torch.ones(2, 3, config.card_slots, dtype=torch.bool),
        placement=torch.zeros(
            2,
            3,
            config.card_slots,
            config.placement_rows,
            config.placement_cols,
            dtype=torch.bool,
        ),
    )
    action_masks.placement[..., 1, 2] = True
    actions = ActionBatch(
        mode=torch.ones(2, 3, dtype=torch.long),
        card_slot=torch.zeros(2, 3, dtype=torch.long),
        placement=torch.tensor([1, 2], dtype=torch.long).expand(2, 3, 2).clone(),
    )
    full_policy = RecurrentHybridPolicy(config)
    compact_policy = RecurrentHybridPolicy(config)
    compact_policy.load_state_dict(full_policy.state_dict())
    full_policy.train()
    compact_policy.train()

    full = full_policy(
        sequence.raster,
        sequence.global_features,
        sequence.entities,
        sequence.entity_mask,
        reset_mask=sequence.reset_mask,
        action_masks=action_masks,
    )
    reduced = compact_policy(
        compact.raster,
        compact.global_features,
        compact.entities,
        compact.entity_mask,
        reset_mask=compact.reset_mask,
        action_masks=action_masks,
    )
    for actual, expected in (
        (reduced.encoded_features, full.encoded_features),
        (reduced.recurrent_features, full.recurrent_features),
        (reduced.mode_logits, full.mode_logits),
        (reduced.card_logits, full.card_logits),
        (reduced.placement_logits, full.placement_logits),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    full_log_prob = full_policy.log_prob(
        full,
        actions,
        action_masks,
    )
    reduced_log_prob = compact_policy.log_prob(
        reduced,
        actions,
        action_masks,
    )
    torch.testing.assert_close(reduced_log_prob, full_log_prob, rtol=1e-5, atol=1e-6)
    full_joint_log_prob, full_joint_entropy = _joint_log_prob_and_entropy(
        full_policy,
        full,
        actions,
        action_masks,
    )
    reduced_joint_log_prob, reduced_joint_entropy = _joint_log_prob_and_entropy(
        compact_policy,
        reduced,
        actions,
        action_masks,
    )
    torch.testing.assert_close(full_joint_log_prob, full_log_prob)
    torch.testing.assert_close(reduced_joint_log_prob, reduced_log_prob)
    torch.testing.assert_close(
        full_joint_entropy,
        _joint_entropy(full_policy, full, action_masks),
    )
    torch.testing.assert_close(
        reduced_joint_entropy,
        _joint_entropy(compact_policy, reduced, action_masks),
    )

    full_loss = (
        full.mode_logits.square().mean()
        + full.card_logits.square().mean()
        + full.placement_logits.square().mean()
    )
    reduced_loss = (
        reduced.mode_logits.square().mean()
        + reduced.card_logits.square().mean()
        + reduced.placement_logits.square().mean()
    )
    full_loss.backward()
    reduced_loss.backward()
    full_gradients = dict(full_policy.named_parameters())
    reduced_gradients = dict(compact_policy.named_parameters())
    assert full_gradients.keys() == reduced_gradients.keys()
    for name in full_gradients:
        actual = reduced_gradients[name].grad
        expected = full_gradients[name].grad
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


@requires_torch
def test_dropout_policy_keeps_entity_tail_for_reproducible_evaluation(monkeypatch) -> None:
    from rl.learner import LearnerConfig, RecurrentPPOLearner, _joint_entropy
    from rl.model import RecurrentHybridPolicy

    config = replace(_model_config(), dropout=0.2)
    policy = RecurrentHybridPolicy(config)
    learner = RecurrentPPOLearner(
        policy,
        config=LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=2,
            shuffle_sequences=False,
        ),
    )
    trajectory = _trajectory(config, batch=2, time=3)
    mask = torch.zeros_like(trajectory.sequence.entity_mask)
    mask[..., :2] = True
    trajectory = replace(
        trajectory,
        sequence=replace(trajectory.sequence, entity_mask=mask),
    )

    # The transfer path must not discard stochastic masked rows before the
    # learner has a chance to evaluate the policy.
    prepared = learner.prepare_batch(trajectory)
    assert (
        prepared.trajectory.sequence.entities.shape[2]
        == trajectory.sequence.entities.shape[2]
    )

    captured: dict[str, torch.Tensor] = {}
    original_forward = policy.forward

    def capture_forward(*args, **kwargs):
        captured["entities"] = args[2]
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(policy, "forward", capture_forward)
    torch.manual_seed(17)
    optimized = learner.evaluate_sequence(
        trajectory.sequence,
        trajectory.actions,
        trajectory.action_masks,
        include_beliefs=False,
    )
    assert captured["entities"].shape[2] == trajectory.sequence.entities.shape[2]

    # With the same RNG state, the guarded learner path must be identical to a
    # direct full-width policy call, including dropout masks and all outputs.
    monkeypatch.setattr(policy, "forward", original_forward)
    torch.manual_seed(17)
    reference = policy(
        trajectory.sequence.raster,
        trajectory.sequence.global_features,
        trajectory.sequence.entities,
        trajectory.sequence.entity_mask,
        reset_mask=trajectory.sequence.reset_mask,
        hidden=trajectory.sequence.initial_hidden,
        action_masks=trajectory.action_masks,
        include_beliefs=False,
    )
    for actual, expected in (
        (optimized.output.encoded_features, reference.encoded_features),
        (optimized.output.recurrent_features, reference.recurrent_features),
        (optimized.output.mode_logits, reference.mode_logits),
        (optimized.output.card_logits, reference.card_logits),
        (optimized.output.placement_logits, reference.placement_logits),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    reference_values = learner._critic_values(reference, None)
    torch.testing.assert_close(optimized.values, reference_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        optimized.log_probs,
        policy.log_prob(reference, trajectory.actions, trajectory.action_masks),
    )
    torch.testing.assert_close(
        optimized.entropy,
        _joint_entropy(policy, reference, trajectory.action_masks),
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
def test_factor_behavior_cloning_is_not_applied_to_mixed_ppo() -> None:
    from rl.learner import LearnerBatch, LearnerConfig, RecurrentPPOLearner
    from rl.model import RecurrentHybridPolicy
    from rl.trajectory import ActionBatch, ActionMasks

    config = _model_config()
    trajectory = _trajectory(config, batch=1, time=2)
    trajectory = replace(
        trajectory,
        action_masks=ActionMasks(
            mode=torch.ones_like(trajectory.action_masks.mode),
            card=torch.ones_like(trajectory.action_masks.card),
            placement=torch.ones_like(trajectory.action_masks.placement),
        ),
        actions=ActionBatch(
            mode=torch.ones_like(trajectory.actions.mode),
            card_slot=torch.zeros_like(trajectory.actions.card_slot),
            placement=torch.zeros_like(trajectory.actions.placement),
        ),
    )
    teacher_weights = torch.ones_like(trajectory.rewards)
    batch = LearnerBatch(
        trajectory=trajectory,
        behavior_cloning_actions=trajectory.actions,
        behavior_cloning_weights=teacher_weights,
    )

    torch.manual_seed(1234)
    mixed = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=1,
            shuffle_sequences=False,
            bc_factor_coef=0.5,
        ),
    )
    torch.manual_seed(1234)
    reference = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=1,
            shuffle_sequences=False,
            bc_factor_coef=0.0,
        ),
    )

    mixed_metrics = mixed.update(batch, diagnostics=True)
    reference_metrics = reference.update(batch, diagnostics=True)

    assert mixed_metrics.effective_factor_behavior_cloning_coef == 0.0
    assert mixed_metrics.factor_behavior_cloning_loss > 0.0
    assert mixed_metrics.total_loss == pytest.approx(reference_metrics.total_loss)
    for mixed_parameter, reference_parameter in zip(
        mixed.policy.parameters(), reference.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(mixed_parameter, reference_parameter)


@requires_torch
def test_factor_behavior_cloning_remains_available_for_imitation_only() -> None:
    from rl.learner import LearnerBatch, LearnerConfig, RecurrentPPOLearner
    from rl.model import RecurrentHybridPolicy
    from rl.trajectory import ActionBatch, ActionMasks

    config = _model_config()
    trajectory = _trajectory(config, batch=1, time=2)
    trajectory = replace(
        trajectory,
        action_masks=ActionMasks(
            mode=torch.ones_like(trajectory.action_masks.mode),
            card=torch.ones_like(trajectory.action_masks.card),
            placement=torch.ones_like(trajectory.action_masks.placement),
        ),
        actions=ActionBatch(
            mode=torch.ones_like(trajectory.actions.mode),
            card_slot=torch.zeros_like(trajectory.actions.card_slot),
            placement=torch.zeros_like(trajectory.actions.placement),
        ),
    )
    learner = RecurrentPPOLearner(
        RecurrentHybridPolicy(config),
        config=LearnerConfig(
            update_epochs=1,
            sequence_minibatch_size=1,
            shuffle_sequences=False,
            bc_factor_coef=0.5,
            imitation_only=True,
        ),
    )
    metrics = learner.update(
        LearnerBatch(
            trajectory=trajectory,
            behavior_cloning_actions=trajectory.actions,
            behavior_cloning_weights=torch.ones_like(trajectory.rewards),
        ),
        diagnostics=True,
    )

    assert metrics.effective_factor_behavior_cloning_coef == pytest.approx(0.5)
    assert metrics.factor_behavior_cloning_loss > 0.0


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
