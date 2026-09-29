"""Masked supervised objective for the retained V4 bootstrap path."""

from __future__ import annotations

from dataclasses import dataclass

from ..foundation._compat import TorchUnavailableError

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "simulator.rl.v4.supervised requires PyTorch to train V4"
        ) from exc
    raise

from ..foundation.trajectory import ActionMasks
from .model_v4 import V4ActionBatch, V4Logits


@dataclass(frozen=True, slots=True)
class V4SupervisedWeights:
    """Relative weights for V4 mode, duration, card, and placement targets."""

    w_mode: float = 1.0
    w_wait_duration: float = 0.5
    w_card: float = 1.0
    w_placement: float = 1.0
    play_upweight: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "w_mode", "w_wait_duration", "w_card", "w_placement", "play_upweight"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def _masked_nll(
    logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    logp = torch.where(mask, logits, torch.full_like(logits, float("-inf"))).log_softmax(-1)
    nll = -logp.gather(-1, target.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    legal_row = mask.any(dim=-1)
    return torch.where(legal_row, nll, torch.zeros_like(nll))


def v4_supervised_loss(
    logits: V4Logits,
    masks: ActionMasks,
    actions: V4ActionBatch,
    *,
    soft_placement: torch.Tensor | None = None,
    mode_weights: torch.Tensor | None = None,
    weights: V4SupervisedWeights = V4SupervisedWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute finite masked cross-entropy for one-step or sequence targets."""

    if masks.mode.shape[:-1] != actions.mode.shape:
        raise ValueError("masks and actions batch/time dimensions must match")
    if mode_weights is not None:
        if mode_weights.shape != actions.mode.shape:
            raise ValueError("mode_weights must match actions.mode shape")
        if not bool(torch.isfinite(mode_weights).all()) or not bool(
            (mode_weights >= 0).all()
        ):
            raise ValueError("mode_weights must be finite and non-negative")
    is_wait = (actions.mode == 0).to(logits.mode.dtype)
    play_rows = (actions.mode == 1).reshape(-1)
    wait_rows = (actions.mode == 0).reshape(-1)
    n_play = int(play_rows.sum().item())
    n_wait = int(wait_rows.sum().item())
    card_rows = masks.card.reshape(-1, masks.card.shape[-1])[play_rows]
    if n_play and not bool(card_rows.any(dim=-1).all()):
        raise ValueError("a PLAY target has no legal card")

    mode_nll = _masked_nll(logits.mode, masks.mode, actions.mode)
    # Episode/mode reservoirs change the retained PLAY/WAIT frequency. Their
    # weights are normalized once over the full dataset, so averaging the
    # weighted losses here reconstructs the unsampled teacher distribution.
    # Keep the unweighted caller's historical equal-mode objective.
    mode_loss = (
        (mode_nll * mode_weights).mean()
        if mode_weights is not None
        else (
            0.5 * (mode_nll[actions.mode == 0].mean() + mode_nll[actions.mode == 1].mean())
            if n_wait and n_play
            else mode_nll.mean()
        )
    )
    duration_nll = F.cross_entropy(
        logits.wait_duration.reshape(-1, logits.wait_duration.shape[-1]),
        actions.wait_duration.reshape(-1).clamp(0, logits.wait_duration.shape[-1] - 1),
        reduction="none",
    ).reshape(actions.mode.shape)
    duration_loss = (
        (duration_nll * is_wait).sum() / n_wait
        if n_wait
        else torch.zeros((), device=logits.mode.device, dtype=logits.mode.dtype)
    )
    if not n_play:
        card_loss = placement_loss = torch.zeros_like(mode_loss)
    else:
        play_scale = float(weights.play_upweight)
        card_logits = logits.card.reshape(-1, logits.card.shape[-1])[play_rows]
        card_masks = masks.card.reshape(-1, masks.card.shape[-1])[play_rows]
        card_targets = actions.card_slot.reshape(-1)[play_rows]
        card_loss = _masked_nll(card_logits, card_masks, card_targets).mean() * play_scale
        rows, cols = logits.placement.shape[-2:]
        cells = rows * cols
        slots = masks.card.shape[-1]
        flat_logits = logits.placement.reshape(-1, slots, cells)[play_rows]
        flat_masks = masks.placement.reshape(-1, slots, cells)[play_rows]
        logp = torch.where(
            flat_masks, flat_logits, torch.full_like(flat_logits, float("-inf"))
        ).log_softmax(-1)
        slots_target = actions.card_slot.reshape(-1)[play_rows].clamp(0, slots - 1)
        selected = logp.gather(
            -2, slots_target.reshape(-1, 1, 1).expand(-1, 1, cells)
        ).squeeze(-2)
        if soft_placement is None:
            cells_target = (
                actions.placement[..., 0].reshape(-1)[play_rows].clamp(0, rows - 1) * cols
                + actions.placement[..., 1].reshape(-1)[play_rows].clamp(0, cols - 1)
            )
            placement_nll = -selected.gather(-1, cells_target.unsqueeze(-1)).squeeze(-1)
        else:
            expected = tuple(actions.mode.shape) + (rows, cols)
            if tuple(soft_placement.shape) != expected:
                raise ValueError(f"soft_placement must have shape {expected}")
            soft = soft_placement.reshape(-1, cells)[play_rows]
            placement_nll = -(
                soft * torch.where(soft > 0, selected, torch.zeros_like(selected))
            ).sum(-1)
        placement_loss = placement_nll.mean() * play_scale

    total = (
        float(weights.w_mode) * mode_loss
        + float(weights.w_wait_duration) * duration_loss
        + float(weights.w_card) * card_loss
        + float(weights.w_placement) * placement_loss
    )
    return total, {
        "loss_mode": float(mode_loss.detach()),
        "loss_wait_duration": float(duration_loss.detach()),
        "loss_card": float(card_loss.detach()),
        "loss_placement": float(placement_loss.detach()),
        "n_play": float(n_play),
        "n_wait": float(n_wait),
    }


__all__ = ["V4SupervisedWeights", "v4_supervised_loss"]
