"""Typed rollout payload and update report shared across PPO stages."""

from __future__ import annotations

from dataclasses import dataclass
from ..model_v4 import V4ActionBatch
from ...foundation.trajectory import TrajectoryBatch
import torch
from .config import V4_RUNTIME_VERSION


@dataclass(slots=True)
class V4RolloutBatch:
    """One collected rollout: PPO tensors plus anchor/diagnostic material."""

    trajectory: TrajectoryBatch
    anchor_actions: V4ActionBatch | None
    recurrent: torch.Tensor | None = None
    old_head_logits: dict[str, torch.Tensor] | None = None
    anchor_log_probs: torch.Tensor | None = None
    per_head_entropy: dict[str, torch.Tensor] | None = None
    wait_fraction: float = 0.0
    card_histogram: dict[str, float] | None = None
    illegal_attempts: int = 0
    next_values: torch.Tensor | None = None
    runtime_version: str = V4_RUNTIME_VERSION


@dataclass(slots=True)
class V4UpdateReport:
    """Diagnostics for one PPO update (losses, movement, quarantine flags)."""

    policy_loss: float
    value_loss: float
    entropy: float
    per_head_entropy: dict[str, float]
    behavior_cloning_loss: float
    approx_kl: float
    mean_abs_log_ratio: float
    clip_fraction: float
    grad_norms: dict[str, float]
    explained_variance: float
    mode_flip_fraction: float
    card_flip_fraction: float
    placement_move_fraction: float
    duration_flip_fraction: float
    per_head_kl: dict[str, float]
    policy_grad_norm: float
    bc_grad_norm: float
    returns_mean: float
    returns_std: float
    advantages_mean: float
    advantages_std: float
    anchor_mode_agree: float | None
    anchor_card_agree: float | None
    anchor_place_within1: float | None
    quarantined: bool
    quarantine_reasons: list[str]

