"""Recurrent PPO and behavior-cloning objectives for the optional RL stack.

The functions here are deliberately independent of environment orchestration:
rollout collectors provide masked joint log-probabilities, values, rewards and
episode-boundary flags.  Terminal matches stop bootstrapping; time-limit
truncations bootstrap the next value but stop the GAE trace, matching the
smoke trainer's corrected evaluation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._compat import TorchUnavailableError

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "rl.objectives requires PyTorch. Install torch to use PPO objectives."
        ) from exc
    raise


def _require_float(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_bool(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 torch.Tensor")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must use dtype torch.bool")


@dataclass(frozen=True, slots=True)
class PPOObjectiveConfig:
    """Numerical coefficients for one recurrent PPO update."""

    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    value_clip_epsilon: float | None = 0.20
    normalize_advantage: bool = True
    bc_coef: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.clip_epsilon)) or not 0.0 < float(self.clip_epsilon):
            raise ValueError("clip_epsilon must be positive")
        for name in ("value_coef", "entropy_coef", "bc_coef"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.value_clip_epsilon is not None
            and (
                not math.isfinite(float(self.value_clip_epsilon))
                or float(self.value_clip_epsilon) <= 0.0
            )
        ):
            raise ValueError("value_clip_epsilon must be positive or None")


@dataclass(frozen=True, slots=True)
class PPOObjectiveResult:
    """Loss components retained for metrics and reproducible diagnostics."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    behavior_cloning_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages/returns with separate terminal and truncation flags."""

    for name, value in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        _require_float(name, value)
        _require_finite(name, value)
    _require_bool("terminated", terminated)
    _require_bool("truncated", truncated)
    shape = values.shape
    if rewards.shape != shape or next_values.shape != shape:
        raise ValueError("rewards, values, and next_values must have identical shapes")
    if terminated.shape != shape or truncated.shape != shape:
        raise ValueError("episode flags must have the same shape as values")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    advantages = torch.zeros_like(values)
    running = torch.zeros(shape[0], dtype=values.dtype, device=values.device)
    terminal_float = terminated.to(dtype=values.dtype)
    boundary = terminated | truncated
    for timestep in range(shape[1] - 1, -1, -1):
        delta = rewards[:, timestep] + float(gamma) * (1.0 - terminal_float[:, timestep]) * next_values[:, timestep]
        delta = delta - values[:, timestep]
        running = delta + float(gamma * gae_lambda) * (~boundary[:, timestep]).to(values.dtype) * running
        advantages[:, timestep] = running
    returns = advantages + values
    _require_finite("advantages", advantages)
    _require_finite("returns", returns)
    return advantages, returns


def behavior_cloning_loss(
    action_log_probs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return confidence-weighted negative log likelihood for teacher actions."""

    _require_float("action_log_probs", action_log_probs)
    _require_finite("action_log_probs", action_log_probs)
    if weights is None:
        weights = torch.ones_like(action_log_probs)
    else:
        _require_float("behavior-cloning weights", weights)
        _require_finite("behavior-cloning weights", weights)
        if weights.shape != action_log_probs.shape:
            raise ValueError("behavior-cloning weights must match action log probabilities")
        if bool((weights < 0.0).any().item()):
            raise ValueError("behavior-cloning weights must be non-negative")
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    result = -(weights * action_log_probs).sum() / denominator
    _require_finite("behavior-cloning loss", result)
    return result


def ppo_objective(
    *,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    old_values: torch.Tensor | None = None,
    behavior_cloning_log_probs: torch.Tensor | None = None,
    behavior_cloning_weights: torch.Tensor | None = None,
    config: PPOObjectiveConfig = PPOObjectiveConfig(),
) -> PPOObjectiveResult:
    """Compute one clipped PPO objective over a recurrent minibatch."""

    tensors = {
        "old_log_probs": old_log_probs,
        "new_log_probs": new_log_probs,
        "advantages": advantages,
        "values": values,
        "returns": returns,
        "entropy": entropy,
    }
    for name, value in tensors.items():
        _require_float(name, value)
        _require_finite(name, value)
    shape = old_log_probs.shape
    if any(value.shape != shape for value in tensors.values()):
        raise ValueError("PPO objective tensors must have identical shapes")
    if old_values is None:
        old_values = values.detach()
    else:
        _require_float("old_values", old_values)
        _require_finite("old_values", old_values)
        if old_values.shape != shape:
            raise ValueError("old_values must match PPO objective tensor shapes")

    if config.normalize_advantage:
        centered = advantages - advantages.mean()
        advantages_for_loss = centered / advantages.std(unbiased=False).clamp_min(1e-8)
    else:
        advantages_for_loss = advantages

    log_ratio = new_log_probs - old_log_probs
    _require_finite("PPO log ratio", log_ratio)
    ratio = torch.exp(log_ratio)
    _require_finite("PPO probability ratio", ratio)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - float(config.clip_epsilon),
        1.0 + float(config.clip_epsilon),
    )
    surrogate = torch.minimum(ratio * advantages_for_loss, clipped_ratio * advantages_for_loss)
    policy_loss = -surrogate.mean()

    if config.value_clip_epsilon is None:
        value_error = (values - returns).square()
    else:
        clipped_values = old_values + torch.clamp(
            values - old_values,
            -float(config.value_clip_epsilon),
            float(config.value_clip_epsilon),
        )
        value_error = torch.maximum(
            (values - returns).square(),
            (clipped_values - returns).square(),
        )
    value_loss = 0.5 * value_error.mean()
    entropy_mean = entropy.mean()

    if behavior_cloning_log_probs is None:
        bc_loss = torch.zeros((), dtype=values.dtype, device=values.device)
    else:
        if behavior_cloning_log_probs.shape != shape:
            raise ValueError("behavior-cloning log probabilities must match PPO tensor shapes")
        bc_loss = behavior_cloning_loss(
            behavior_cloning_log_probs,
            behavior_cloning_weights,
        )

    total_loss = (
        policy_loss
        + float(config.value_coef) * value_loss
        - float(config.entropy_coef) * entropy_mean
        + float(config.bc_coef) * bc_loss
    )
    result = PPOObjectiveResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        behavior_cloning_loss=bc_loss,
        approx_kl=(old_log_probs - new_log_probs).mean(),
        clip_fraction=(torch.abs(ratio - 1.0) > float(config.clip_epsilon)).to(torch.float32).mean(),
    )
    for name, value in (
        ("PPO total loss", result.total_loss),
        ("PPO policy loss", result.policy_loss),
        ("PPO value loss", result.value_loss),
        ("PPO entropy", result.entropy),
        ("behavior-cloning loss", result.behavior_cloning_loss),
        ("PPO approximate KL", result.approx_kl),
        ("PPO clip fraction", result.clip_fraction),
    ):
        _require_finite(name, value)
    return result


__all__ = [
    "PPOObjectiveConfig",
    "PPOObjectiveResult",
    "behavior_cloning_loss",
    "compute_gae",
    "ppo_objective",
]
