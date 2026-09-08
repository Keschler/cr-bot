"""Factorized V4 simulator-distillation loss (paper section 5.1).

``Lsup = wm*CE(mode) + ww*CE(duration | WAIT) + wc*CE(card | PLAY)
        + wp*CE(placement | card, PLAY)``

Sampling and weighting are factor-aware: rare decisive PLAY/card/placement
targets are upweighted so they are not drowned by WAIT states, while
natural-distribution evaluation stays separate from balanced training.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._compat import TorchUnavailableError

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "rl.losses_v4 requires PyTorch. Install torch to use the V4 "
            "distillation objectives."
        ) from exc
    raise

from .model_v4 import V4ActionBatch, V4Logits
from .trajectory import ActionMasks


@dataclass(frozen=True, slots=True)
class V4SupervisedWeights:
    """Loss weights; ``play_upweight`` rebalances rare decisive PLAY rows."""

    w_mode: float = 1.0
    w_wait_duration: float = 0.5
    w_card: float = 1.0
    w_placement: float = 1.0
    play_upweight: float = 2.0

    def __post_init__(self) -> None:
        for name in ("w_mode", "w_wait_duration", "w_card", "w_placement", "play_upweight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def _masked_nll(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = torch.where(
        mask, logits, torch.full_like(logits, float("-inf"))
    ).log_softmax(dim=-1)
    nll = -logp.gather(-1, target.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    # Rows with no legal action (e.g. card choice on an all-WAIT state) yield
    # NaN from log_softmax; they contribute nothing and are zeroed here.
    # Callers validate that *selected* rows always have legal mass.
    legal_row = mask.any(dim=-1)
    return torch.where(legal_row, nll, torch.zeros_like(nll))


def v4_supervised_loss(
    logits: V4Logits,
    masks: ActionMasks,
    actions: V4ActionBatch,
    *,
    soft_placement: torch.Tensor | None = None,
    weights: V4SupervisedWeights = V4SupervisedWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the factorized distillation loss and per-factor diagnostics.

    ``soft_placement`` optionally holds a ``[B, T, R, C]`` teacher
    distribution over the *selected* card's map (mass on legal cells);
    without it the hard target cell is used.  WAIT rows contribute mode +
    duration only; PLAY rows contribute mode + card + placement only.
    """

    if masks.mode.shape[:-1] != actions.mode.shape:
        raise ValueError("masks and actions batch/time dimensions must match")
    is_play = (actions.mode == 1).to(logits.mode.dtype)
    is_wait = (actions.mode == 0).to(logits.mode.dtype)
    n_play = int((actions.mode == 1).sum().item())
    n_wait = int((actions.mode == 0).sum().item())
    if n_play and not bool(masks.card.any(dim=-1)[actions.mode == 1].all()):
        raise ValueError("a PLAY target has no legal card (teacher/mask bug)")

    mode_loss = _masked_nll(logits.mode, masks.mode, actions.mode).mean()

    if n_wait:
        duration_nll = F.cross_entropy(
            logits.wait_duration.reshape(-1, logits.wait_duration.shape[-1]),
            actions.wait_duration.reshape(-1).clamp(0, logits.wait_duration.shape[-1] - 1),
            reduction="none",
        ).reshape(actions.mode.shape)
        duration_loss = (duration_nll * is_wait).sum() / max(n_wait, 1)
    else:
        duration_loss = torch.zeros((), device=logits.mode.device, dtype=logits.mode.dtype)

    play_scale = float(weights.play_upweight)
    if n_play:
        card_nll = _masked_nll(logits.card, masks.card, actions.card_slot)
        card_loss = (card_nll * is_play).sum() / max(n_play, 1) * play_scale
        rows, cols = logits.placement.shape[-2:]
        flat_logits = logits.placement.reshape(
            tuple(logits.placement.shape[:-3]) + (masks.card.shape[-1], rows * cols)
        )
        flat_mask = masks.placement.reshape(
            tuple(masks.placement.shape[:-3]) + (masks.card.shape[-1], rows * cols)
        )
        logp = torch.where(
            flat_mask, flat_logits, torch.full_like(flat_logits, float("-inf"))
        ).log_softmax(dim=-1)
        slot = actions.card_slot.clamp(0, masks.card.shape[-1] - 1)
        slot_index = slot.reshape(slot.shape + (1, 1)).expand(
            slot.shape + (1, rows * cols)
        )
        selected_logp = logp.gather(-2, slot_index).squeeze(-2)
        if soft_placement is not None:
            if tuple(soft_placement.shape) != tuple(actions.mode.shape) + (rows, cols):
                raise ValueError("soft_placement must have shape [B, T, R, C]")
            soft = soft_placement.reshape(tuple(actions.mode.shape) + (rows * cols,))
            # WAIT rows carry an all-zero soft map; multiplying zero mass by
            # -inf on illegal cells would yield NaN, so unsupported cells are
            # excluded before the product.  Mass on an illegal cell (a teacher
            # bug) still surfaces as +inf instead of being hidden.
            safe_logp = torch.where(
                soft > 0, selected_logp, torch.zeros_like(selected_logp)
            )
            placement_nll = -(soft * safe_logp).sum(dim=-1)
        else:
            cell = (
                actions.placement[..., 0].clamp(0, rows - 1) * cols
                + actions.placement[..., 1].clamp(0, cols - 1)
            ).clamp(0, rows * cols - 1)
            placement_nll = -selected_logp.gather(-1, cell.unsqueeze(-1)).squeeze(-1)
        placement_loss = (placement_nll * is_play).sum() / max(n_play, 1) * play_scale
    else:
        zero = torch.zeros((), device=logits.mode.device, dtype=logits.mode.dtype)
        card_loss, placement_loss = zero, zero

    total = (
        float(weights.w_mode) * mode_loss
        + float(weights.w_wait_duration) * duration_loss
        + float(weights.w_card) * card_loss
        + float(weights.w_placement) * placement_loss
    )
    detail = {
        "loss_mode": float(mode_loss.detach()),
        "loss_wait_duration": float(duration_loss.detach()),
        "loss_card": float(card_loss.detach()),
        "loss_placement": float(placement_loss.detach()),
        "n_play": float(n_play),
        "n_wait": float(n_wait),
    }
    return total, detail


__all__ = ["V4SupervisedWeights", "v4_supervised_loss"]
