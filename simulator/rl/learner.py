"""Optional recurrent PPO learner and rollout-state foundation.

This module is intentionally additive to :mod:`rl.model`,
:mod:`rl.objectives`, and :mod:`rl.trajectory`.  It does not make PyTorch a
simulator dependency: importing this module succeeds when torch is absent,
while constructing or using the torch-backed learner raises the same targeted
``TorchUnavailableError`` used by the rest of the optional RL stack.

The learner treats one row of a :class:`~rl.trajectory.TrajectoryBatch` as a
contiguous recurrent sequence.  Sequence rows, rather than individual
timesteps, are shuffled into minibatches.  This prevents PPO updates from
silently mixing hidden states between unrelated episodes.  Optional temporal
chunks are supported when the trajectory retained ``hidden_states`` snapshots
for the beginning of every chunk.

The rollout helper is deliberately environment-agnostic.  It accepts the
already encoded observation tensors and action masks produced by the caller,
maintains the GRU state, applies reset masks before the corresponding model
step, and returns the next detached state plus a sampled action.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ._compat import TORCH_AVAILABLE, TorchUnavailableError

_TORCH_IMPORT_ERROR: BaseException | None = None

if TORCH_AVAILABLE:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F

        from .model import (
            PrivilegedCritic,
            RecurrentHybridPolicy,
            RecurrentPolicyOutput,
        )
        from .objectives import PPOObjectiveConfig, ppo_objective, compute_gae
        from .trajectory import (
            ActionBatch,
            ActionMasks,
            RecurrentSequence,
            TrajectoryBatch,
        )
    except TorchUnavailableError as exc:  # pragma: no cover - defensive path
        TORCH_AVAILABLE = False
        _TORCH_IMPORT_ERROR = exc
else:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    PrivilegedCritic = Any  # type: ignore[misc,assignment]
    RecurrentHybridPolicy = Any  # type: ignore[misc,assignment]
    RecurrentPolicyOutput = Any  # type: ignore[misc,assignment]
    PPOObjectiveConfig = Any  # type: ignore[misc,assignment]
    ActionBatch = Any  # type: ignore[misc,assignment]
    ActionMasks = Any  # type: ignore[misc,assignment]
    RecurrentSequence = Any  # type: ignore[misc,assignment]
    TrajectoryBatch = Any  # type: ignore[misc,assignment]


def _raise_torch_unavailable() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise TorchUnavailableError(
            "The recurrent PPO learner requires PyTorch. Install the optional "
            "torch dependency before using rl.learner."
        ) from _TORCH_IMPORT_ERROR
    raise TorchUnavailableError(
        "The recurrent PPO learner requires PyTorch. Install the optional "
        "torch dependency before using rl.learner."
    )


def resolve_policy_device(device: Any) -> Any:
    """Resolve a policy execution device, including the ``auto`` sentinel.

    ``auto`` selects CUDA when it is visible and otherwise falls back to CPU.
    An explicit device remains untouched, so callers can force CPU for
    reproducibility or select a particular accelerator.  Keeping this choice
    at learner construction means every inference entry point uses the same
    policy device resolution and does not need to duplicate CUDA checks.  CPU
    policy work is also capped at eight intra-op threads; this avoids the
    oversubscription penalty measured for the small batched actor workload.
    """

    if not TORCH_AVAILABLE:  # pragma: no cover - guarded by learner use
        _raise_torch_unavailable()
    if isinstance(device, str) and device.strip().lower() == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    configure_policy_cpu_threads(resolved)
    return resolved


def configure_policy_cpu_threads(device: Any, *, cap: int = 8) -> int | None:
    """Limit CPU intra-op parallelism for the small batched policy workload.

    The actor's convolution, recurrent step, and masked heads operate on
    relatively small tensors.  On the benchmark host, allowing the default
    thread pool to use more than eight workers makes each decision slower due
    to launch and synchronization overhead.  Preserve a caller's lower
    setting and leave accelerator execution untouched.  This is a runtime
    scheduling choice only; it does not change model parameters or numerical
    operations.

    Returns the active CPU thread count, or ``None`` for non-CPU devices.
    """

    if not TORCH_AVAILABLE:  # pragma: no cover - guarded by learner use
        _raise_torch_unavailable()
    resolved = torch.device(device)
    if resolved.type != "cpu":
        return None
    if type(cap) is not int or cap <= 0:
        raise ValueError("cap must be a positive integer")
    active = int(torch.get_num_threads())
    target = min(active, cap)
    if target != active:
        torch.set_num_threads(target)
    return target


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    """Numerical and batching settings for :class:`RecurrentPPOLearner`.

    ``sequence_minibatch_size`` counts recurrent sequence rows, not individual
    timesteps.  The default discount is expressed for a relatively long,
    real-time control horizon; callers should still tune it against their
    policy cadence.
    """

    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    update_epochs: int = 4
    sequence_minibatch_size: int = 8
    max_grad_norm: float = 0.5
    gamma: float = 0.9995
    gae_lambda: float = 0.98
    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    value_clip_epsilon: float | None = 0.20
    normalize_advantage: bool = True
    bc_coef: float = 0.0
    bc_factor_coef: float = 0.0
    imitation_only: bool = False
    belief_coef: float = 0.0
    require_privileged_critic: bool = False
    shuffle_sequences: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(float(self.adam_eps)) or float(self.adam_eps) <= 0.0:
            raise ValueError("adam_eps must be positive")
        for name in ("update_epochs", "sequence_minibatch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(self.max_grad_norm)) or float(self.max_grad_norm) < 0.0:
            raise ValueError("max_grad_norm must be non-negative")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= float(self.gae_lambda) <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not math.isfinite(float(self.clip_epsilon)) or float(self.clip_epsilon) <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        for name in (
            "value_coef",
            "entropy_coef",
            "bc_coef",
            "bc_factor_coef",
            "belief_coef",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.imitation_only) is not bool:
            raise ValueError("imitation_only must be boolean")
        if (
            self.value_clip_epsilon is not None
            and (
                not math.isfinite(float(self.value_clip_epsilon))
                or float(self.value_clip_epsilon) <= 0.0
            )
        ):
            raise ValueError("value_clip_epsilon must be positive or None")

    def objective_config(self) -> Any:
        """Build the existing objective configuration lazily."""

        if not TORCH_AVAILABLE:
            _raise_torch_unavailable()
        return PPOObjectiveConfig(
            clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            value_clip_epsilon=self.value_clip_epsilon,
            normalize_advantage=self.normalize_advantage,
            bc_coef=self.bc_coef,
        )


@dataclass(frozen=True, slots=True)
class BeliefTargets:
    """Optional training-only targets for the policy's opponent-belief heads.

    ``enemy_hand`` is a multi-label float/binary tensor with final dimension
    equal to the belief vocabulary.  ``enemy_next_card`` contains card indices
    and may use ``-100`` for ignored positions.  Any field may be omitted; the
    learner computes only the losses for fields that are present.
    """

    enemy_elixir: Any | None = None
    enemy_hand: Any | None = None
    enemy_next_card: Any | None = None

    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            return
        values = (
            self.enemy_elixir,
            self.enemy_hand,
            self.enemy_next_card,
        )
        present = [value for value in values if value is not None]
        if not present:
            raise ValueError("at least one belief target must be provided")
        prefix: tuple[int, int] | None = None
        for name, value in (
            ("enemy_elixir", self.enemy_elixir),
            ("enemy_hand", self.enemy_hand),
            ("enemy_next_card", self.enemy_next_card),
        ):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise ValueError(f"{name} must be a tensor with [batch, time, ...] shape")
            current_prefix = (int(value.shape[0]), int(value.shape[1]))
            if prefix is None:
                prefix = current_prefix
            elif current_prefix != prefix:
                raise ValueError("belief targets must share batch and time dimensions")
        if self.enemy_elixir is not None and self.enemy_elixir.ndim != 2:
            raise ValueError("enemy_elixir must have shape [batch, time]")
        if self.enemy_hand is not None and self.enemy_hand.ndim != 3:
            raise ValueError("enemy_hand must have shape [batch, time, card_count]")
        if self.enemy_next_card is not None and self.enemy_next_card.ndim != 2:
            raise ValueError("enemy_next_card must have shape [batch, time]")


@dataclass(frozen=True, slots=True)
class LearnerBatch:
    """A trajectory plus optional privileged and auxiliary training tensors."""

    trajectory: Any
    privileged_features: Any | None = None
    belief_targets: BeliefTargets | None = None
    next_values: Any | None = None
    bootstrap_values: Any | None = None
    behavior_cloning_log_probs: Any | None = None
    behavior_cloning_actions: Any | None = None
    behavior_cloning_weights: Any | None = None

    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            return
        if not isinstance(self.trajectory, TrajectoryBatch):
            raise TypeError("trajectory must be an rl.trajectory.TrajectoryBatch")
        prefix = tuple(self.trajectory.sequence.raster.shape[:2])
        if self.privileged_features is not None:
            _check_tensor_shape(
                "privileged_features",
                self.privileged_features,
                prefix,
                ndim=3,
            )
        for name, value, expected_ndim in (
            ("next_values", self.next_values, 2),
            ("behavior_cloning_log_probs", self.behavior_cloning_log_probs, 2),
            ("behavior_cloning_weights", self.behavior_cloning_weights, 2),
        ):
            if value is not None:
                _check_tensor_shape(name, value, prefix, ndim=expected_ndim)
        if self.behavior_cloning_actions is not None:
            if not isinstance(self.behavior_cloning_actions, ActionBatch):
                raise TypeError("behavior_cloning_actions must be an ActionBatch")
            if self.behavior_cloning_actions.prefix_shape != prefix:
                raise ValueError(
                    "behavior_cloning_actions must match trajectory batch and time dimensions"
                )
        if self.bootstrap_values is not None:
            _check_tensor_shape(
                "bootstrap_values",
                self.bootstrap_values,
                (prefix[0],),
                ndim=1,
            )
        if self.next_values is not None and self.bootstrap_values is not None:
            raise ValueError("provide next_values or bootstrap_values, not both")
        if self.belief_targets is not None:
            for name, value in (
                ("enemy_elixir", self.belief_targets.enemy_elixir),
                ("enemy_hand", self.belief_targets.enemy_hand),
                ("enemy_next_card", self.belief_targets.enemy_next_card),
            ):
                if value is not None and tuple(value.shape[:2]) != prefix:
                    raise ValueError(f"belief target {name} must match trajectory dimensions")

    @property
    def batch_size(self) -> int:
        return int(self.trajectory.sequence.raster.shape[0])

    @property
    def time_steps(self) -> int:
        return int(self.trajectory.sequence.raster.shape[1])


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Recomputed policy statistics for one recurrent sequence batch."""

    output: Any
    log_probs: Any
    entropy: Any
    values: Any


@dataclass(frozen=True, slots=True)
class RecurrentRolloutState:
    """Detached GRU state carried between one-step rollout calls."""

    hidden: Any

    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            return
        if not isinstance(self.hidden, torch.Tensor) or self.hidden.ndim != 3:
            raise ValueError("hidden must be a rank-3 torch.Tensor")

    def reset(self, reset_mask: Any) -> "RecurrentRolloutState":
        """Return a state with selected batch rows zeroed before a step."""

        if not TORCH_AVAILABLE:
            _raise_torch_unavailable()
        if not isinstance(reset_mask, torch.Tensor) or reset_mask.dtype != torch.bool:
            raise TypeError("reset_mask must be a boolean torch.Tensor")
        if reset_mask.ndim != 1 or reset_mask.shape[0] != self.hidden.shape[1]:
            raise ValueError("reset_mask must have shape [batch]")
        mask = reset_mask.reshape(1, -1, 1)
        return RecurrentRolloutState(self.hidden.masked_fill(mask, 0.0))

    def detach(self) -> "RecurrentRolloutState":
        if not TORCH_AVAILABLE:
            _raise_torch_unavailable()
        return RecurrentRolloutState(self.hidden.detach())


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """Output of :meth:`RecurrentPPOLearner.rollout_step`."""

    actions: Any
    log_probs: Any
    entropy: Any
    values: Any
    output: Any
    next_state: RecurrentRolloutState


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    """Aggregate scalar diagnostics from one learner update."""

    update_index: int
    epochs: int
    minibatches: int
    optimization_steps: int
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    belief_loss: float
    approx_kl: float
    clip_fraction: float
    gradient_norm: float
    skipped_steps: int = 0
    behavior_cloning_loss: float = 0.0
    factor_behavior_cloning_loss: float = 0.0
    effective_factor_behavior_cloning_coef: float = 0.0
    # Raw pre-clipping gradient norms grouped by actor head/encoder.  Empty in
    # ordinary runs so the hot path and old report shape remain unchanged.
    per_head_gradient_norms: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "update_index": self.update_index,
            "epochs": self.epochs,
            "minibatches": self.minibatches,
            "optimization_steps": self.optimization_steps,
            "total_loss": self.total_loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "belief_loss": self.belief_loss,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "gradient_norm": self.gradient_norm,
            "skipped_steps": self.skipped_steps,
            "behavior_cloning_loss": self.behavior_cloning_loss,
            "factor_behavior_cloning_loss": self.factor_behavior_cloning_loss,
            "effective_factor_behavior_cloning_coef": self.effective_factor_behavior_cloning_coef,
            "per_head_gradient_norms": dict(self.per_head_gradient_norms),
        }


def _check_tensor_shape(name: str, value: Any, prefix: tuple[int, ...], *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} torch.Tensor")
    if tuple(value.shape[: len(prefix)]) != prefix:
        raise ValueError(f"{name} must begin with shape {prefix}")


def _as_learner_batch(
    batch: LearnerBatch | Any,
    *,
    privileged_features: Any | None = None,
    belief_targets: BeliefTargets | None = None,
    next_values: Any | None = None,
    bootstrap_values: Any | None = None,
    behavior_cloning_log_probs: Any | None = None,
    behavior_cloning_actions: Any | None = None,
    behavior_cloning_weights: Any | None = None,
) -> LearnerBatch:
    if isinstance(batch, LearnerBatch):
        overrides = (
            privileged_features,
            belief_targets,
            next_values,
            bootstrap_values,
            behavior_cloning_log_probs,
            behavior_cloning_actions,
            behavior_cloning_weights,
        )
        if any(value is not None for value in overrides):
            raise ValueError("optional tensors cannot override an existing LearnerBatch")
        return batch
    return LearnerBatch(
        trajectory=batch,
        privileged_features=privileged_features,
        belief_targets=belief_targets,
        next_values=next_values,
        bootstrap_values=bootstrap_values,
        behavior_cloning_log_probs=behavior_cloning_log_probs,
        behavior_cloning_actions=behavior_cloning_actions,
        behavior_cloning_weights=behavior_cloning_weights,
    )


def _slice_sequence(
    sequence: Any,
    indices: Any,
    *,
    start: int = 0,
    end: int | None = None,
) -> Any:
    if end is None:
        end = sequence.time_steps
    time_slice = slice(start, end)
    hidden_states = None
    if sequence.hidden_states is not None:
        hidden_states = sequence.hidden_states[indices, time_slice]
    if start == 0:
        initial_hidden = (
            None
            if sequence.initial_hidden is None
            else sequence.initial_hidden[:, indices]
        )
    else:
        if sequence.hidden_states is None:
            raise ValueError(
                "time-chunked recurrent minibatches require hidden_states snapshots"
            )
        initial_hidden = sequence.hidden_states[indices, start].permute(1, 0, 2).contiguous()
    return RecurrentSequence(
        raster=sequence.raster[indices, time_slice],
        global_features=sequence.global_features[indices, time_slice],
        entities=sequence.entities[indices, time_slice],
        entity_mask=sequence.entity_mask[indices, time_slice],
        reset_mask=sequence.reset_mask[indices, time_slice],
        hidden_states=hidden_states,
        initial_hidden=initial_hidden,
    )


def _slice_trajectory(
    trajectory: Any,
    indices: Any,
    *,
    start: int = 0,
    end: int | None = None,
) -> Any:
    if end is None:
        end = trajectory.time_steps
    time_slice = slice(start, end)
    sequence = _slice_sequence(trajectory.sequence, indices, start=start, end=end)
    masks = ActionMasks(
        mode=trajectory.action_masks.mode[indices, time_slice],
        card=trajectory.action_masks.card[indices, time_slice],
        placement=trajectory.action_masks.placement[indices, time_slice],
    )
    actions = ActionBatch(
        mode=trajectory.actions.mode[indices, time_slice],
        card_slot=trajectory.actions.card_slot[indices, time_slice],
        placement=trajectory.actions.placement[indices, time_slice],
    )
    fields = {
        "sequence": sequence,
        "action_masks": masks,
        "actions": actions,
        "rewards": trajectory.rewards[indices, time_slice],
        "terminated": trajectory.terminated[indices, time_slice],
        "truncated": trajectory.truncated[indices, time_slice],
        "old_log_probs": trajectory.old_log_probs[indices, time_slice],
        "values": None
        if trajectory.values is None
        else trajectory.values[indices, time_slice],
        "advantages": None
        if trajectory.advantages is None
        else trajectory.advantages[indices, time_slice],
        "returns": None
        if trajectory.returns is None
        else trajectory.returns[indices, time_slice],
    }
    return TrajectoryBatch(**fields)


def _slice_belief_targets(targets: BeliefTargets | None, indices: Any, start: int, end: int) -> BeliefTargets | None:
    if targets is None:
        return None
    time_slice = slice(start, end)
    return BeliefTargets(
        enemy_elixir=None
        if targets.enemy_elixir is None
        else targets.enemy_elixir[indices, time_slice],
        enemy_hand=None
        if targets.enemy_hand is None
        else targets.enemy_hand[indices, time_slice],
        enemy_next_card=None
        if targets.enemy_next_card is None
        else targets.enemy_next_card[indices, time_slice],
    )


def _slice_learner_batch(batch: LearnerBatch, indices: Any, *, start: int = 0, end: int | None = None) -> LearnerBatch:
    if end is None:
        end = batch.time_steps
    time_slice = slice(start, end)
    return LearnerBatch(
        trajectory=_slice_trajectory(batch.trajectory, indices, start=start, end=end),
        privileged_features=None
        if batch.privileged_features is None
        else batch.privileged_features[indices, time_slice],
        belief_targets=_slice_belief_targets(batch.belief_targets, indices, start, end),
        next_values=None
        if batch.next_values is None
        else batch.next_values[indices, time_slice],
        bootstrap_values=None
        if batch.bootstrap_values is None
        else batch.bootstrap_values[indices],
        behavior_cloning_log_probs=None
        if batch.behavior_cloning_log_probs is None
        else batch.behavior_cloning_log_probs[indices, time_slice],
        behavior_cloning_actions=None
        if batch.behavior_cloning_actions is None
        else ActionBatch(
            mode=batch.behavior_cloning_actions.mode[indices, time_slice],
            card_slot=batch.behavior_cloning_actions.card_slot[indices, time_slice],
            placement=batch.behavior_cloning_actions.placement[indices, time_slice],
        ),
        behavior_cloning_weights=None
        if batch.behavior_cloning_weights is None
        else batch.behavior_cloning_weights[indices, time_slice],
    )


def iter_sequence_minibatches(
    batch: LearnerBatch | Any,
    minibatch_size: int,
    *,
    shuffle: bool = True,
    generator: Any | None = None,
    sequence_length: int | None = None,
) -> Iterator[LearnerBatch]:
    """Yield recurrent minibatches without breaking sequence boundaries.

    With the default ``sequence_length=None``, every trajectory row remains a
    complete sequence and only the batch dimension is shuffled/sliced.  When a
    fixed temporal chunk length is requested, the length must divide the
    sequence length and ``hidden_states`` must be present for non-initial
    chunks, so each chunk receives the correct pre-observation hidden state.
    """

    if not TORCH_AVAILABLE:
        _raise_torch_unavailable()
    learner_batch = _as_learner_batch(batch)
    if type(minibatch_size) is not int or minibatch_size <= 0:
        raise ValueError("minibatch_size must be a positive integer")
    batch_size = learner_batch.batch_size
    time_steps = learner_batch.time_steps
    if sequence_length is not None:
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        if time_steps % sequence_length:
            raise ValueError("sequence_length must divide the trajectory time dimension")
        if sequence_length < time_steps and learner_batch.trajectory.sequence.hidden_states is None:
            raise ValueError(
                "time-chunked recurrent minibatches require hidden_states snapshots"
            )
        groups: list[LearnerBatch] = []
        base_indices = torch.arange(batch_size, device=learner_batch.trajectory.sequence.raster.device)
        for start in range(0, time_steps, sequence_length):
            groups.append(
                _slice_learner_batch(
                    learner_batch,
                    base_indices,
                    start=start,
                    end=start + sequence_length,
                )
            )
        if shuffle and len(groups) > 1:
            order = torch.randperm(len(groups), generator=generator, device="cpu").tolist()
            groups = [groups[index] for index in order]
        for group in groups:
            yield from _split_sequence_group(group, minibatch_size, shuffle=shuffle, generator=generator)
        return

    indices = torch.arange(batch_size, device=learner_batch.trajectory.sequence.raster.device)
    if shuffle and batch_size > 1:
        indices = indices[torch.randperm(batch_size, generator=generator, device="cpu").to(indices.device)]
    for offset in range(0, batch_size, minibatch_size):
        yield _slice_learner_batch(
            learner_batch,
            indices[offset : offset + minibatch_size],
        )


def _split_sequence_group(
    batch: LearnerBatch,
    minibatch_size: int,
    *,
    shuffle: bool,
    generator: Any | None,
) -> Iterator[LearnerBatch]:
    indices = torch.arange(batch.batch_size, device=batch.trajectory.sequence.raster.device)
    if shuffle and batch.batch_size > 1:
        indices = indices[torch.randperm(batch.batch_size, generator=generator, device="cpu").to(indices.device)]
    for offset in range(0, batch.batch_size, minibatch_size):
        yield _slice_learner_batch(batch, indices[offset : offset + minibatch_size])


if TORCH_AVAILABLE:

    class RecurrentValueHead(nn.Module):
        """Simple actor-observation value fallback when no critic is supplied."""

        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            if type(hidden_dim) is not int or hidden_dim <= 0:
                raise ValueError("hidden_dim must be a positive integer")
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, recurrent_features: torch.Tensor) -> torch.Tensor:
            if recurrent_features.ndim != 3:
                raise ValueError("recurrent_features must have shape [batch, time, hidden]")
            return self.value(recurrent_features).squeeze(-1)


    class RecurrentPPOLearner:
        """PPO optimizer for the recurrent structured policy.

        ``critic`` may be the existing :class:`PrivilegedCritic`.  In that
        case pass ``privileged_features`` with every update/rollout call; the
        critic receives exact simulator features while the actor receives only
        the public observation tensors.  If no critic is supplied, a small
        actor-observation value head is created as a usable smoke fallback.
        """

        CHECKPOINT_FORMAT_VERSION = 1

        def __init__(
            self,
            policy: RecurrentHybridPolicy,
            critic: nn.Module | None = None,
            config: LearnerConfig | None = None,
            *,
            privileged_dim: int | None = None,
            privileged_critic: bool | None = None,
            device: torch.device | str | None = None,
        ) -> None:
            if not isinstance(policy, nn.Module):
                raise TypeError("policy must be a torch.nn.Module")
            self.config = LearnerConfig() if config is None else config
            if not isinstance(self.config, LearnerConfig):
                raise TypeError("config must be a LearnerConfig")
            if device is None:
                try:
                    device = next(policy.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
            self.device = resolve_policy_device(device)
            self.policy = policy.to(self.device)

            inferred_privileged = isinstance(critic, PrivilegedCritic)
            if critic is None and privileged_dim is not None:
                if type(privileged_dim) is not int or privileged_dim <= 0:
                    raise ValueError("privileged_dim must be a positive integer")
                critic = PrivilegedCritic(
                    recurrent_dim=self.policy.config.gru_hidden_dim,
                    privileged_dim=privileged_dim,
                )
                inferred_privileged = True
            elif critic is None:
                critic = RecurrentValueHead(self.policy.config.gru_hidden_dim)
            self.critic = critic.to(self.device)
            self.uses_privileged_critic = (
                inferred_privileged if privileged_critic is None else bool(privileged_critic)
            )
            if self.config.require_privileged_critic and not self.uses_privileged_critic:
                raise ValueError("configuration requires a privileged critic")
            if self.uses_privileged_critic and not isinstance(self.critic, nn.Module):
                raise TypeError("privileged critic must be a torch.nn.Module")

            parameters: list[nn.Parameter] = []
            seen: set[int] = set()
            for module in (self.policy, self.critic):
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                eps=self.config.adam_eps,
            )
            self.update_count = 0

        def initial_rollout_state(self, batch_size: int) -> RecurrentRolloutState:
            """Create a zero GRU state on the learner's device."""

            return RecurrentRolloutState(self.policy.initial_hidden(batch_size, device=self.device))

        def evaluate_sequence(
            self,
            sequence: RecurrentSequence,
            actions: ActionBatch,
            action_masks: ActionMasks,
            *,
            privileged_features: torch.Tensor | None = None,
            include_beliefs: bool = True,
        ) -> PolicyEvaluation:
            """Re-run one sequence with explicit initial hidden/reset semantics."""

            sequence = _move_sequence(sequence, self.device)
            actions = _move_actions(actions, self.device)
            action_masks = _move_masks(action_masks, self.device)
            if privileged_features is not None:
                privileged_features = privileged_features.to(self.device)
            output = self.policy(
                sequence.raster,
                sequence.global_features,
                sequence.entities,
                sequence.entity_mask,
                reset_mask=sequence.reset_mask,
                hidden=sequence.initial_hidden,
                action_masks=action_masks,
                include_beliefs=include_beliefs,
            )
            log_probs = self.policy.log_prob(output, actions, action_masks)
            entropy = _joint_entropy(self.policy, output, action_masks)
            values = self._critic_values(output, privileged_features)
            return PolicyEvaluation(output, log_probs, entropy, values)

        def rollout_step(
            self,
            state: RecurrentRolloutState,
            raster: torch.Tensor,
            global_features: torch.Tensor,
            entities: torch.Tensor,
            entity_mask: torch.Tensor,
            action_masks: ActionMasks,
            *,
            reset_mask: torch.Tensor | None = None,
            privileged_features: torch.Tensor | None = None,
            deterministic: bool = False,
            include_beliefs: bool = True,
            inference: bool = False,
            fast_sampling: bool = False,
        ) -> RolloutStep:
            """Run one recurrent policy step and return a detached next state.

            Inputs may omit the singleton time dimension; it is added for the
            caller.  ``reset_mask`` is applied by the GRU immediately before
            this observation, matching ``RecurrentSequence`` semantics.
            """

            state = state.detach()
            if type(inference) is not bool:
                raise TypeError("inference must be boolean")
            if type(fast_sampling) is not bool:
                raise TypeError("fast_sampling must be boolean")
            raster = _add_step_time(raster, expected_ndim=4, name="raster")
            global_features = _add_step_time(global_features, expected_ndim=2, name="global_features")
            entities = _add_step_time(entities, expected_ndim=3, name="entities")
            entity_mask = _add_step_time(entity_mask, expected_ndim=2, name="entity_mask")
            action_masks = _add_mask_time(action_masks)
            if reset_mask is None:
                reset_mask = torch.zeros(
                    (raster.shape[0], 1), dtype=torch.bool, device=self.device
                )
            elif reset_mask.ndim == 1:
                reset_mask = reset_mask.reshape(-1, 1)
            if reset_mask.shape != (raster.shape[0], 1):
                raise ValueError("reset_mask must have shape [batch] or [batch, 1]")
            reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
            if privileged_features is not None:
                privileged_features = _add_step_time(
                    privileged_features,
                    expected_ndim=2,
                    name="privileged_features",
                )
            model_masks = _move_masks(action_masks, self.device)
            policy_inputs = (
                raster.to(self.device),
                global_features.to(self.device),
                entities.to(self.device),
                entity_mask.to(self.device, dtype=torch.bool),
            )
            if fast_sampling and not deterministic:
                output, actions, log_probs, entropy = self.policy.rollout_sample(
                    *policy_inputs,
                    model_masks,
                    reset_mask=reset_mask,
                    hidden=state.hidden.to(self.device),
                    include_beliefs=include_beliefs,
                    inference=inference,
                )
            else:
                output = self.policy(
                    *policy_inputs,
                    reset_mask=reset_mask,
                    hidden=state.hidden.to(self.device),
                    action_masks=model_masks,
                    include_beliefs=include_beliefs,
                    inference=inference,
                )
                if deterministic:
                    actions, log_probs, entropy = _deterministic_action(
                        self.policy,
                        output,
                        model_masks,
                    )
                else:
                    actions, log_probs, entropy = self.policy.action_head.sample(
                        output.logits,
                        model_masks,
                    )
            values = self._critic_values(output, privileged_features)
            return RolloutStep(
                actions=actions,
                log_probs=log_probs,
                entropy=entropy,
                values=values,
                output=output,
                next_state=RecurrentRolloutState(output.final_hidden.detach()),
            )

        def prepare_batch(self, batch: LearnerBatch | Any, **kwargs: Any) -> LearnerBatch:
            """Fill old values and GAE targets when a rollout omitted them."""

            if not TORCH_AVAILABLE:  # pragma: no cover - class is unavailable then
                _raise_torch_unavailable()
            learner_batch = _as_learner_batch(batch, **kwargs)
            learner_batch = _move_learner_batch(learner_batch, self.device)
            trajectory = learner_batch.trajectory
            old_values = trajectory.values
            if old_values is None:
                with torch.no_grad():
                    evaluation = self.evaluate_sequence(
                        trajectory.sequence,
                        trajectory.actions,
                        trajectory.action_masks,
                        privileged_features=learner_batch.privileged_features,
                    )
                old_values = evaluation.values.detach()
            else:
                old_values = old_values.to(device=self.device, dtype=torch.float32)

            advantages = trajectory.advantages
            returns = trajectory.returns
            if (advantages is None) != (returns is None):
                raise ValueError("trajectory must provide both advantages and returns, or neither")
            if advantages is None:
                next_values = _resolve_next_values(learner_batch, old_values)
                advantages, returns = compute_gae(
                    trajectory.rewards.to(device=self.device, dtype=old_values.dtype),
                    old_values,
                    next_values,
                    trajectory.terminated.to(self.device),
                    trajectory.truncated.to(self.device),
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )
            else:
                advantages = advantages.to(device=self.device, dtype=old_values.dtype)
                returns = returns.to(device=self.device, dtype=old_values.dtype)

            prepared_trajectory = replace(
                trajectory,
                values=old_values.detach(),
                advantages=advantages.detach(),
                returns=returns.detach(),
            )
            return replace(learner_batch, trajectory=prepared_trajectory)

        def update(
            self,
            batch: LearnerBatch | Any,
            *,
            privileged_features: torch.Tensor | None = None,
            belief_targets: BeliefTargets | None = None,
            next_values: torch.Tensor | None = None,
            bootstrap_values: torch.Tensor | None = None,
            behavior_cloning_log_probs: torch.Tensor | None = None,
            behavior_cloning_actions: ActionBatch | None = None,
            behavior_cloning_weights: torch.Tensor | None = None,
            sequence_length: int | None = None,
            diagnostics: bool = False,
        ) -> UpdateMetrics:
            """Run configured PPO epochs over recurrent sequence minibatches."""

            if type(diagnostics) is not bool:
                raise TypeError("diagnostics must be boolean")

            prepared = self.prepare_batch(
                batch,
                privileged_features=privileged_features,
                belief_targets=belief_targets,
                next_values=next_values,
                bootstrap_values=bootstrap_values,
                behavior_cloning_log_probs=behavior_cloning_log_probs,
                behavior_cloning_actions=behavior_cloning_actions,
                behavior_cloning_weights=behavior_cloning_weights,
            )
            self.policy.train()
            self.critic.train()
            sums = {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "belief_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "gradient_norm": 0.0,
                "behavior_cloning_loss": 0.0,
                "factor_behavior_cloning_loss": 0.0,
                "effective_factor_behavior_cloning_coef": 0.0,
            }
            head_sums: dict[str, float] = {}
            minibatches = 0
            optimization_steps = 0
            skipped_steps = 0
            for _epoch in range(self.config.update_epochs):
                for minibatch in iter_sequence_minibatches(
                    prepared,
                    self.config.sequence_minibatch_size,
                    shuffle=self.config.shuffle_sequences,
                    sequence_length=sequence_length,
                ):
                    metrics = self._update_minibatch(minibatch, diagnostics=diagnostics)
                    minibatches += 1
                    optimization_steps += int(metrics.pop("_optimization_step", 0.0))
                    skipped_steps += int(metrics.pop("_skipped_step", 0.0))
                    for name in sums:
                        sums[name] += metrics[name]
                    if diagnostics:
                        for name, value in metrics.get("_per_head_gradient_norms", {}).items():
                            head_sums[name] = head_sums.get(name, 0.0) + float(value)
            if minibatches < 1:
                raise ValueError("PPO update produced no sequence minibatches")
            self.update_count += 1
            scale = 1.0 / minibatches
            return UpdateMetrics(
                update_index=self.update_count,
                epochs=self.config.update_epochs,
                minibatches=minibatches,
                optimization_steps=optimization_steps,
                skipped_steps=skipped_steps,
                per_head_gradient_norms={
                    name: value * scale for name, value in sorted(head_sums.items())
                },
                **{name: value * scale for name, value in sums.items()},
            )

        def _update_minibatch(
            self,
            batch: LearnerBatch,
            *,
            diagnostics: bool = False,
        ) -> dict[str, Any]:
            self.optimizer.zero_grad(set_to_none=True)
            parameters = self._optimizer_parameters()
            if not _parameters_are_finite(parameters):
                raise FloatingPointError("PPO parameters are already non-finite")
            trajectory = batch.trajectory
            evaluation = self.evaluate_sequence(
                trajectory.sequence,
                trajectory.actions,
                trajectory.action_masks,
                privileged_features=batch.privileged_features,
                include_beliefs=batch.belief_targets is not None,
            )
            behavior_cloning_actions = batch.behavior_cloning_actions
            behavior_cloning_evaluation_log_probs = evaluation.log_probs
            if behavior_cloning_actions is not None:
                behavior_cloning_evaluation_log_probs = self.policy.log_prob(
                    evaluation.output,
                    behavior_cloning_actions,
                    trajectory.action_masks,
                )
            # When behavior cloning is enabled, the rollout action is the
            # teacher action (the legacy collector behavior), or an explicit
            # teacher label when DAgger-style collection executes the actor's
            # sampled action. Re-evaluate the label with the current policy so
            # the cloning term remains differentiable. A caller may still
            # provide an explicit target tensor for older/off-policy datasets.
            behavior_cloning_log_probs = (
                behavior_cloning_evaluation_log_probs
                if self.config.bc_coef > 0.0
                and batch.behavior_cloning_log_probs is None
                else (
                    None
                    if batch.behavior_cloning_log_probs is None
                    else batch.behavior_cloning_log_probs.to(self.device)
                )
            )
            try:
                objective = ppo_objective(
                    old_log_probs=trajectory.old_log_probs.to(self.device),
                    new_log_probs=evaluation.log_probs,
                    advantages=trajectory.advantages.to(self.device),
                    values=evaluation.values,
                    returns=trajectory.returns.to(self.device),
                    entropy=evaluation.entropy,
                    old_values=trajectory.values.to(self.device),
                    behavior_cloning_log_probs=behavior_cloning_log_probs,
                    behavior_cloning_weights=None
                    if batch.behavior_cloning_weights is None
                    else batch.behavior_cloning_weights.to(self.device),
                    config=self.config.objective_config(),
                )
                factor_bc_loss = (
                    _factor_behavior_cloning_loss(
                        self.policy,
                        evaluation.output,
                        trajectory,
                        batch.behavior_cloning_weights,
                        actions=behavior_cloning_actions,
                    )
                    if self.config.imitation_only or diagnostics
                    else evaluation.output.recurrent_features.new_zeros(())
                )
                belief_loss = _belief_loss(evaluation.output, batch.belief_targets)
                # The factorized loss was introduced for supervised warm-starts.
                # In mixed actor-controlled PPO it class-balances the rare PLAY
                # labels and can move the mode/card/placement heads far more than
                # the outcome objective.  Keep measuring it in diagnostics, but
                # apply it only during the explicitly supervised phase.
                effective_factor_bc_coef = (
                    self.config.bc_factor_coef if self.config.imitation_only else 0.0
                )
                if self.config.imitation_only:
                    # Expert-guided rollouts contain teacher actions, not the
                    # actor's sampled actions.  During a warm-start phase the
                    # PPO advantage can therefore push the policy away from a
                    # useful teacher simply because a sparse outcome was
                    # assigned to a short segment.  Keep that phase a pure
                    # supervised update; the normal PPO objective remains the
                    # default for self-improvement after the warm start.
                    total_loss = (
                        self.config.bc_coef * objective.behavior_cloning_loss
                        + effective_factor_bc_coef * factor_bc_loss
                    )
                else:
                    total_loss = (
                        objective.total_loss
                        + effective_factor_bc_coef * factor_bc_loss
                        + self.config.belief_coef * belief_loss
                    )
                _require_finite_tensor("PPO total loss", total_loss)
                for name, value in (
                    ("PPO policy loss", objective.policy_loss),
                    ("PPO value loss", objective.value_loss),
                    ("PPO entropy", objective.entropy),
                    ("PPO factor behavior-cloning loss", factor_bc_loss),
                    ("PPO belief loss", belief_loss),
                    ("PPO approximate KL", objective.approx_kl),
                    ("PPO clip fraction", objective.clip_fraction),
                ):
                    _require_finite_tensor(name, value)
            except FloatingPointError:
                return _skipped_minibatch_metrics()

            total_loss.backward()
            gradient_norm = _gradient_norm(parameters)
            per_head_gradient_norms = (
                self._gradient_norm_by_head() if diagnostics else {}
            )
            if not math.isfinite(gradient_norm) or not _gradients_are_finite(parameters):
                self.optimizer.zero_grad(set_to_none=True)
                return _skipped_minibatch_metrics()

            if self.config.max_grad_norm and gradient_norm > self.config.max_grad_norm:
                scale = self.config.max_grad_norm / gradient_norm
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            if not _gradients_are_finite(parameters):
                self.optimizer.zero_grad(set_to_none=True)
                return _skipped_minibatch_metrics()
            if not _optimizer_state_is_finite(self.optimizer):
                raise FloatingPointError("PPO optimizer state is non-finite before step")

            self.optimizer.step()
            if not _parameters_are_finite(parameters) or not _optimizer_state_is_finite(self.optimizer):
                raise FloatingPointError("PPO optimizer step produced non-finite state")
            return {
                "total_loss": float(total_loss.detach().cpu().item()),
                "policy_loss": float(objective.policy_loss.detach().cpu().item()),
                "value_loss": float(objective.value_loss.detach().cpu().item()),
                "entropy": float(objective.entropy.detach().cpu().item()),
                "belief_loss": float(belief_loss.detach().cpu().item()),
                "behavior_cloning_loss": float(
                    objective.behavior_cloning_loss.detach().cpu().item()
                ),
                "factor_behavior_cloning_loss": float(
                    factor_bc_loss.detach().cpu().item()
                ),
                "effective_factor_behavior_cloning_coef": float(
                    effective_factor_bc_coef
                ),
                "approx_kl": float(objective.approx_kl.detach().cpu().item()),
                "clip_fraction": float(objective.clip_fraction.detach().cpu().item()),
                "gradient_norm": gradient_norm,
                "_optimization_step": 1.0,
                "_skipped_step": 0.0,
                "_per_head_gradient_norms": per_head_gradient_norms,
            }

        def _gradient_norm_by_head(self) -> dict[str, float]:
            """Return raw gradient norms partitioned by the causal model head."""

            grouped: dict[str, float] = {}
            modules = (
                ("actor", self.policy.named_parameters()),
                ("critic", self.critic.named_parameters()),
            )
            for module_kind, named_parameters in modules:
                for name, parameter in named_parameters:
                    if parameter.grad is None:
                        continue
                    lower = name.casefold()
                    if module_kind == "critic":
                        group = "critic"
                    elif "placement" in lower or "spatial" in lower:
                        group = "placement"
                    elif "card" in lower or "hand" in lower:
                        group = "card"
                    elif "mode" in lower:
                        group = "mode"
                    elif "core" in lower or "gru" in lower or "recurrent" in lower:
                        group = "recurrent"
                    elif "encoder" in lower:
                        group = "encoder"
                    else:
                        group = "other_actor"
                    norm = float(parameter.grad.detach().float().norm(2).cpu().item())
                    grouped[group] = grouped.get(group, 0.0) + norm * norm
            return {name: math.sqrt(value) for name, value in grouped.items()}

        def _optimizer_parameters(self) -> list[nn.Parameter]:
            parameters: list[nn.Parameter] = []
            for group in self.optimizer.param_groups:
                parameters.extend(group["params"])
            return parameters

        def _critic_values(
            self,
            output: RecurrentPolicyOutput,
            privileged_features: torch.Tensor | None,
        ) -> torch.Tensor:
            if self.uses_privileged_critic:
                if privileged_features is None:
                    raise ValueError(
                        "privileged_features are required by the configured privileged critic"
                    )
                privileged_features = privileged_features.to(
                    device=self.device,
                    dtype=output.recurrent_features.dtype,
                )
                values = self.critic(output.recurrent_features, privileged_features)
            else:
                values = self.critic(output.recurrent_features)
            if not isinstance(values, torch.Tensor) or values.shape != output.recurrent_features.shape[:2]:
                raise ValueError("critic must return values with shape [batch, time]")
            return values

        def checkpoint_state(self) -> dict[str, Any]:
            """Return policy, critic, optimizer, counters, and RNG state."""

            if (
                not _module_state_is_finite(self.policy)
                or not _module_state_is_finite(self.critic)
                or not _optimizer_state_is_finite(self.optimizer)
            ):
                raise FloatingPointError(
                    "cannot checkpoint non-finite PPO parameters or optimizer state"
                )
            state: dict[str, Any] = {
                "format_version": self.CHECKPOINT_FORMAT_VERSION,
                "learner_config": asdict(self.config),
                "uses_privileged_critic": self.uses_privileged_critic,
                "update_count": self.update_count,
                "policy": copy.deepcopy(self.policy.state_dict()),
                "critic": copy.deepcopy(self.critic.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "rng_state": torch.get_rng_state().clone(),
            }
            if torch.cuda.is_available():
                state["cuda_rng_state_all"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
            return state

        def state_dict(self) -> dict[str, Any]:
            """Alias for checkpoint serialization, including optimizer state."""

            return self.checkpoint_state()

        def load_checkpoint_state(
            self,
            state: Mapping[str, Any],
            *,
            strict: bool = True,
            restore_rng: bool = True,
        ) -> None:
            if not isinstance(state, Mapping):
                raise TypeError("checkpoint state must be a mapping")
            version = state.get("format_version")
            if version != self.CHECKPOINT_FORMAT_VERSION:
                raise ValueError(
                    f"unsupported learner checkpoint format {version!r}; "
                    f"expected {self.CHECKPOINT_FORMAT_VERSION}"
                )
            for key in ("policy", "critic", "optimizer"):
                if key not in state:
                    raise ValueError(f"checkpoint is missing {key!r} state")
                if not _nested_tensors_are_finite(state[key]):
                    raise ValueError(f"checkpoint {key!r} state contains non-finite values")
            self.policy.load_state_dict(state["policy"], strict=strict)
            self.critic.load_state_dict(state["critic"], strict=strict)
            self.optimizer.load_state_dict(state["optimizer"])
            _optimizer_to_device(self.optimizer, self.device)
            if (
                not _module_state_is_finite(self.policy)
                or not _module_state_is_finite(self.critic)
                or not _optimizer_state_is_finite(self.optimizer)
            ):
                raise ValueError("checkpoint loaded non-finite PPO state")
            self.update_count = int(state.get("update_count", 0))
            if restore_rng and state.get("rng_state") is not None:
                torch.set_rng_state(state["rng_state"].to(device="cpu"))
                if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
                    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        def load_state_dict(
            self,
            state: Mapping[str, Any],
            *,
            strict: bool = True,
            restore_rng: bool = True,
        ) -> None:
            self.load_checkpoint_state(state, strict=strict, restore_rng=restore_rng)

        def save_checkpoint(self, path: str | Path) -> None:
            """Serialize a complete learner checkpoint with ``torch.save``."""

            torch.save(self.checkpoint_state(), Path(path))

        def load_checkpoint(
            self,
            path: str | Path,
            *,
            map_location: torch.device | str | None = None,
            strict: bool = True,
            restore_rng: bool = True,
        ) -> None:
            """Load a complete learner checkpoint into this learner."""

            state = torch.load(
                Path(path),
                map_location=self.device if map_location is None else map_location,
            )
            self.load_checkpoint_state(state, strict=strict, restore_rng=restore_rng)


else:

    class RecurrentValueHead:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_torch_unavailable()


    class RecurrentPPOLearner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_torch_unavailable()


def _move_sequence(sequence: Any, device: Any) -> Any:
    if not TORCH_AVAILABLE:
        _raise_torch_unavailable()
    return replace(
        sequence,
        raster=sequence.raster.to(device),
        global_features=sequence.global_features.to(device),
        entities=sequence.entities.to(device),
        entity_mask=sequence.entity_mask.to(device),
        reset_mask=sequence.reset_mask.to(device),
        hidden_states=None
        if sequence.hidden_states is None
        else sequence.hidden_states.to(device),
        initial_hidden=None
        if sequence.initial_hidden is None
        else sequence.initial_hidden.to(device),
    )


def _move_masks(masks: Any, device: Any) -> Any:
    return ActionMasks(
        mode=masks.mode.to(device),
        card=masks.card.to(device),
        placement=masks.placement.to(device),
    )


def _move_actions(actions: Any, device: Any) -> Any:
    return ActionBatch(
        mode=actions.mode.to(device),
        card_slot=actions.card_slot.to(device),
        placement=actions.placement.to(device),
    )


def _move_learner_batch(batch: LearnerBatch, device: Any) -> LearnerBatch:
    trajectory = batch.trajectory
    moved_trajectory = replace(
        trajectory,
        sequence=_move_sequence(trajectory.sequence, device),
        action_masks=_move_masks(trajectory.action_masks, device),
        actions=_move_actions(trajectory.actions, device),
        rewards=trajectory.rewards.to(device),
        terminated=trajectory.terminated.to(device),
        truncated=trajectory.truncated.to(device),
        old_log_probs=trajectory.old_log_probs.to(device),
        values=None if trajectory.values is None else trajectory.values.to(device),
        advantages=None if trajectory.advantages is None else trajectory.advantages.to(device),
        returns=None if trajectory.returns is None else trajectory.returns.to(device),
    )
    targets = batch.belief_targets
    moved_targets = None
    if targets is not None:
        moved_targets = BeliefTargets(
            enemy_elixir=None if targets.enemy_elixir is None else targets.enemy_elixir.to(device),
            enemy_hand=None if targets.enemy_hand is None else targets.enemy_hand.to(device),
            enemy_next_card=None if targets.enemy_next_card is None else targets.enemy_next_card.to(device),
        )
    return replace(
        batch,
        trajectory=moved_trajectory,
        privileged_features=None
        if batch.privileged_features is None
        else batch.privileged_features.to(device),
        belief_targets=moved_targets,
        next_values=None if batch.next_values is None else batch.next_values.to(device),
        bootstrap_values=None
        if batch.bootstrap_values is None
        else batch.bootstrap_values.to(device),
        behavior_cloning_log_probs=None
        if batch.behavior_cloning_log_probs is None
        else batch.behavior_cloning_log_probs.to(device),
        behavior_cloning_actions=None
        if batch.behavior_cloning_actions is None
        else _move_actions(batch.behavior_cloning_actions, device),
        behavior_cloning_weights=None
        if batch.behavior_cloning_weights is None
        else batch.behavior_cloning_weights.to(device),
    )


def _resolve_next_values(batch: LearnerBatch, values: Any) -> Any:
    if batch.next_values is not None:
        return batch.next_values.to(device=values.device, dtype=values.dtype)
    if batch.bootstrap_values is not None:
        next_values = torch.zeros_like(values)
        if values.shape[1] > 1:
            next_values[:, :-1] = values[:, 1:]
        next_values[:, -1] = batch.bootstrap_values.to(
            device=values.device,
            dtype=values.dtype,
        )
        return next_values
    truncated = batch.trajectory.truncated.to(values.device)
    if bool(truncated.any().item()):
        raise ValueError(
            "truncated rollouts require explicit next_values or bootstrap_values"
        )
    next_values = torch.zeros_like(values)
    if values.shape[1] > 1:
        next_values[:, :-1] = values[:, 1:]
    return next_values


def _joint_entropy(policy: Any, output: Any, masks: Any) -> Any:
    """Compute entropy of the complete masked autoregressive action policy."""

    factors = policy.action_head.masked_log_probs(output.logits, masks)
    mode_entropy = _categorical_entropy(factors.mode)
    card_entropy = _categorical_entropy(factors.card)
    placement_shape = factors.placement.shape
    placement_logs = factors.placement.reshape(*placement_shape[:-2], -1)
    placement_entropy = _categorical_entropy(placement_logs)
    card_probabilities = torch.exp(factors.card)
    play_probability = torch.exp(factors.mode[..., 1])
    conditional_placement_entropy = (
        card_probabilities * placement_entropy
    ).sum(dim=-1)
    return mode_entropy + play_probability * (card_entropy + conditional_placement_entropy)


def _factor_behavior_cloning_loss(
    policy: Any,
    output: Any,
    trajectory: Any,
    weights: Any | None,
    *,
    actions: Any | None = None,
) -> Any:
    """Balance teacher gradients across mode, card, and placement factors.

    The ordinary joint log-probability is dominated by the 32x18 placement
    vocabulary.  During expert warm-up that can leave the smaller but more
    important WAIT/PLAY and card-slot heads passive.  This auxiliary loss
    gives each factor its own normalized contribution while retaining the
    original joint BC term in :func:`ppo_objective`.
    """

    if weights is None:
        return output.recurrent_features.new_zeros(())
    weights = weights.to(
        device=output.recurrent_features.device,
        dtype=output.recurrent_features.dtype,
    )
    if weights.shape != trajectory.actions.mode.shape:
        raise ValueError("behavior-cloning weights must match action dimensions")
    if bool((weights < 0.0).any().item()):
        raise ValueError("behavior-cloning weights must be non-negative")
    if not bool(torch.isfinite(weights).all().item()):
        raise FloatingPointError("behavior-cloning weights are non-finite")

    factors = policy.action_head.masked_log_probs(output.logits, trajectory.action_masks)
    target_actions = trajectory.actions if actions is None else actions
    if not isinstance(target_actions, ActionBatch):
        raise TypeError("behavior-cloning actions must be an ActionBatch")
    mode = target_actions.mode.to(dtype=torch.long)
    mode_log_probs = factors.mode.gather(-1, mode.unsqueeze(-1)).squeeze(-1)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    losses = [-(weights * mode_log_probs).sum() / denominator]

    # Normalize WAIT and PLAY separately.  A long simulator rollout contains
    # many more resource-saving waits than card plays; one global denominator
    # therefore teaches the actor to choose one mode everywhere (usually
    # WAIT).  Class balancing keeps both decisions visible while retaining the
    # caller's confidence weights within each class.
    mode_losses: list[Any] = []
    for mode_value in (policy.action_head.WAIT, policy.action_head.PLAY):
        mode_mask = mode == mode_value
        if not bool(mode_mask.any().item()):
            continue
        mode_weights = weights[mode_mask]
        mode_denominator = mode_weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        mode_losses.append(
            -(mode_weights * mode_log_probs[mode_mask]).sum() / mode_denominator
        )
    if mode_losses:
        losses = [torch.stack(mode_losses).mean()]

    play = mode == policy.action_head.PLAY
    if bool(play.any().item()):
        play_weights = weights[play]
        play_denominator = play_weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        card_indices = target_actions.card_slot[play].to(dtype=torch.long)
        card_log_probs = factors.card[play].gather(
            -1,
            card_indices.unsqueeze(-1),
        ).squeeze(-1)

        # Keep the caller's confidence weighting for card identity.  The
        # teacher already marks decisive cards (such as Hog Rider) more
        # strongly; making every rare card equally frequent would erase the
        # distinction between pressure, defense, and spell labels.
        losses.append(-(play_weights * card_log_probs).sum() / play_denominator)

        placement = factors.placement[play]
        sample_number = torch.arange(
            card_indices.shape[0],
            device=card_indices.device,
        )
        selected = placement[sample_number, card_indices].reshape(
            card_indices.shape[0], -1
        )
        rows = target_actions.placement[play][..., 0].to(dtype=torch.long)
        columns = target_actions.placement[play][..., 1].to(dtype=torch.long)
        placement_cols = int(factors.placement.shape[-1])
        cells = rows * placement_cols + columns
        placement_log_probs = selected.gather(-1, cells.unsqueeze(-1)).squeeze(-1)
        losses.append(-(play_weights * placement_log_probs).sum() / play_denominator)
    return torch.stack(losses).mean()


def _categorical_entropy(log_probs: Any) -> Any:
    finite = torch.isfinite(log_probs)
    safe_logs = torch.where(finite, log_probs, torch.zeros_like(log_probs))
    probabilities = torch.where(finite, torch.exp(log_probs), torch.zeros_like(log_probs))
    return -(probabilities * safe_logs).sum(dim=-1)


def _belief_loss(output: Any, targets: BeliefTargets | None) -> Any:
    if targets is None:
        return output.recurrent_features.new_zeros(())
    if output.belief_logits is None:
        raise ValueError("policy output does not contain opponent-belief logits")
    losses: list[Any] = []
    logits = output.belief_logits
    if targets.enemy_elixir is not None:
        losses.append(
            F.mse_loss(
                logits.enemy_elixir,
                targets.enemy_elixir.to(device=logits.enemy_elixir.device, dtype=logits.enemy_elixir.dtype),
            )
        )
    if targets.enemy_hand is not None:
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits.enemy_hand,
                targets.enemy_hand.to(device=logits.enemy_hand.device, dtype=logits.enemy_hand.dtype),
            )
        )
    if targets.enemy_next_card is not None:
        target = targets.enemy_next_card.to(device=logits.enemy_next_card.device, dtype=torch.long)
        valid = target != -100
        if bool(valid.any().item()):
            losses.append(
                F.cross_entropy(
                    logits.enemy_next_card[valid],
                    target[valid],
                )
            )
    if not losses:
        return output.recurrent_features.new_zeros(())
    return torch.stack(losses).mean()


def _require_finite_tensor(name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} is non-finite")


def _skipped_minibatch_metrics() -> dict[str, float]:
    """Return neutral metrics for a minibatch that must not update weights."""

    return {
        "total_loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "belief_loss": 0.0,
        "behavior_cloning_loss": 0.0,
        "factor_behavior_cloning_loss": 0.0,
        "effective_factor_behavior_cloning_coef": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
        "_optimization_step": 0.0,
        "_skipped_step": 1.0,
    }


def _nested_tensors_are_finite(value: Any) -> bool:
    """Check floating/complex tensors inside checkpoint or optimizer state."""

    if isinstance(value, torch.Tensor):
        if not (value.is_floating_point() or value.is_complex()):
            return True
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_nested_tensors_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_tensors_are_finite(item) for item in value)
    return True


def _module_state_is_finite(module: Any) -> bool:
    return _nested_tensors_are_finite(module.state_dict())


def _optimizer_state_is_finite(optimizer: Any) -> bool:
    return _nested_tensors_are_finite(optimizer.state_dict())


def _parameters_are_finite(parameters: Sequence[Any]) -> bool:
    return all(
        parameter is not None
        and (not parameter.is_floating_point() or bool(torch.isfinite(parameter.detach()).all().item()))
        for parameter in parameters
    )


def _gradients_are_finite(parameters: Sequence[Any]) -> bool:
    return all(
        parameter.grad is None
        or (not parameter.grad.is_floating_point())
        or bool(torch.isfinite(parameter.grad.detach()).all().item())
        for parameter in parameters
    )


def _gradient_norm(parameters: Sequence[Any]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            if not _gradients_are_finite((parameter,)):
                return math.inf
            # Accumulate in float64 so a finite collection of large float32
            # gradients cannot overflow before clipping is applied.
            contribution = float(
                gradient.to(dtype=torch.float64).square().sum().cpu().item()
            )
            squared += contribution
            if not math.isfinite(squared):
                return math.inf
    return math.sqrt(squared)


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _add_step_time(value: Any, *, expected_ndim: int, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == expected_ndim:
        return value.unsqueeze(1)
    if value.ndim == expected_ndim + 1:
        if value.shape[1] != 1:
            raise ValueError(f"{name} step input must have a singleton time dimension")
        return value
    raise ValueError(
        f"{name} must have rank {expected_ndim} or {expected_ndim + 1}, got {value.ndim}"
    )


def _add_mask_time(masks: Any) -> Any:
    if masks.mode.ndim == 2:
        return ActionMasks(
            mode=masks.mode.unsqueeze(1),
            card=masks.card.unsqueeze(1),
            placement=masks.placement.unsqueeze(1),
        )
    if masks.mode.ndim == 3 and masks.mode.shape[1] == 1:
        return masks
    raise ValueError("one-step action masks must have shape [batch, ...] or [batch, 1, ...]")


def _deterministic_action(policy: Any, output: Any, masks: Any) -> tuple[Any, Any, Any]:
    factors = policy.action_head.masked_log_probs(output.logits, masks)
    mode = factors.mode.argmax(dim=-1)
    card_slot = torch.zeros_like(mode, dtype=torch.long)
    placement = torch.zeros((*mode.shape, 2), dtype=torch.long, device=mode.device)
    play = mode == policy.action_head.PLAY
    if bool(play.any().item()):
        selected_cards = factors.card[play].argmax(dim=-1)
        card_slot[play] = selected_cards
        selected_placement = factors.placement[play]
        rows = selected_placement.shape[-2]
        cols = selected_placement.shape[-1]
        sample_number = torch.arange(selected_cards.shape[0], device=selected_cards.device)
        selected_logits = selected_placement[sample_number, selected_cards].reshape(
            selected_cards.shape[0], -1
        )
        cells = selected_logits.argmax(dim=-1)
        placement[play, 0] = torch.div(cells, cols, rounding_mode="floor")
        placement[play, 1] = cells.remainder(cols)
    actions = ActionBatch(mode=mode, card_slot=card_slot, placement=placement)
    log_probs = policy.action_head.log_prob(output.logits, actions, masks)
    entropy = _joint_entropy(policy, output, masks)
    return actions, log_probs, entropy


__all__ = [
    "BeliefTargets",
    "LearnerBatch",
    "LearnerConfig",
    "PolicyEvaluation",
    "RecurrentPPOLearner",
    "RecurrentRolloutState",
    "RecurrentValueHead",
    "RolloutStep",
    "configure_policy_cpu_threads",
    "resolve_policy_device",
    "TORCH_AVAILABLE",
    "TorchUnavailableError",
    "UpdateMetrics",
    "iter_sequence_minibatches",
]
