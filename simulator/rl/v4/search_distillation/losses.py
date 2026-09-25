"""Correction and conservative retention losses."""

from __future__ import annotations
from typing import Any, Mapping, Sequence

import torch

from ..model_v4 import RecurrentV4Policy, V4Logits
from .batching import _forward
from .records import ActionTarget, CachedPrefix, DEFAULT_TEMPERATURE, FactorizedTarget
from .targets import _record_soft_target, factorized_distributions


def _factorized_kl(
    old: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    new: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    masks: Any,
) -> torch.Tensor:
    """Mean factorized KL at the decision step.

    Prefixes are fed to the GRU to establish recurrent state, but retention
    supervision belongs to the selected decision.  Reducing over every
    prefix step accidentally turns burn-in into a second, implicit training
    objective and makes ``context=32`` differ from ``context=0`` even when the
    final distributions match.
    """

    old_mode, old_card, old_place = old
    new_mode, new_card, new_place = new
    if old_mode.ndim >= 3:
        old_mode, new_mode = old_mode[:, -1, ...], new_mode[:, -1, ...]
        old_card, new_card = old_card[:, -1, ...], new_card[:, -1, ...]
        old_place, new_place = old_place[:, -1, ...], new_place[:, -1, ...]
    eps = torch.finfo(old_mode.dtype).tiny
    mode = (old_mode * (old_mode.clamp_min(eps).log() - new_mode.clamp_min(eps).log())).sum(-1)
    play = old_mode[..., 1]
    card = (old_card * (old_card.clamp_min(eps).log() - new_card.clamp_min(eps).log())).sum(-1)
    place = (old_place * (old_place.clamp_min(eps).log() - new_place.clamp_min(eps).log())).sum((-1, -2))
    place = (place * old_card).sum(-1)
    card_mask = masks.card[:, -1, ...] if masks.card.ndim >= 3 else masks.card
    valid_card = card_mask.any(-1).to(mode.dtype)
    return (mode + play * (card + place) * valid_card).mean()


def _factorized_hard_nll(
    logits: V4Logits, masks: Any, actions: Sequence[ActionTarget]
) -> torch.Tensor:
    mode, card, place = factorized_distributions(logits, masks)
    eps = torch.finfo(mode.dtype).tiny
    mode_t = torch.tensor([a.mode for a in actions], device=mode.device, dtype=torch.long)
    card_t = torch.tensor([a.slot for a in actions], device=mode.device, dtype=torch.long)
    row_t = torch.tensor([a.row for a in actions], device=mode.device, dtype=torch.long)
    col_t = torch.tensor([a.col for a in actions], device=mode.device, dtype=torch.long)
    # The caller supplies final-step logits [B, 1, ...].
    mode = mode[:, -1]
    card = card[:, -1]
    place = place[:, -1]
    loss = -mode[torch.arange(len(actions), device=mode.device), mode_t].clamp_min(eps).log()
    play = mode_t == 1
    if bool(play.any()):
        idx = torch.where(play)[0]
        loss[idx] += -card[idx, card_t[idx]].clamp_min(eps).log()
        loss[idx] += -place[idx, card_t[idx], row_t[idx], col_t[idx]].clamp_min(eps).log()
    return loss.mean()


def _target_tensor(target: FactorizedTarget, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for value in (target.mode, target.card, target.placement))  # type: ignore[return-value]


def _factorized_soft_kl(
    old: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    new: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: FactorizedTarget,
) -> torch.Tensor:
    target_mode, target_card, target_place = _target_tensor(target, new[0].device)
    eps = torch.finfo(new[0].dtype).tiny
    mode = (target_mode * (target_mode.clamp_min(eps).log() - new[0][:, -1].clamp_min(eps).log())).sum(-1)
    play = target_mode[..., 1]
    card = (target_card * (target_card.clamp_min(eps).log() - new[1][:, -1].clamp_min(eps).log())).sum(-1)
    place = (target_place * (target_place.clamp_min(eps).log() - new[2][:, -1].clamp_min(eps).log())).sum((-1, -2))
    place = (place * target_card).sum(-1)
    return (mode + play * (card + place)).mean()


def compute_batch_loss(
    policy: RecurrentV4Policy,
    anchor: RecurrentV4Policy,
    batch: Mapping[str, Any],
    records: Sequence[CachedPrefix],
    *,
    method: str,
    temperature: float = DEFAULT_TEMPERATURE,
    retention_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one ablation loss on final target steps."""

    if method not in {"hard-none", "hard-replay", "soft-replay", "soft-kl"}:
        raise ValueError(f"unknown method {method!r}")
    logits, masks = _forward(policy, batch)
    with torch.no_grad():
        anchor_logits, _ = _forward(anchor, batch)
        old = factorized_distributions(anchor_logits, masks)
    corrections = [r for r in records if r.is_correction]
    retention = [r for r in records if not r.is_correction]
    # All batch records share one final-step index.  Build losses by subsets so
    # correction/retention cardinality cannot silently change weighting.
    losses: list[torch.Tensor] = []
    details: dict[str, float] = {"n_correction": float(len(corrections)), "n_retention": float(len(retention))}
    if corrections:
        indices = [i for i, r in enumerate(records) if r.is_correction]
        sub_logits = V4Logits(
            mode=logits.mode[indices], wait_duration=logits.wait_duration[indices],
            card=logits.card[indices], placement=logits.placement[indices],
        )
        sub_masks = type(masks)(mode=masks.mode[indices], card=masks.card[indices], placement=masks.placement[indices])
        if method.startswith("hard"):
            corr_loss = _factorized_hard_nll(sub_logits, sub_masks, [r.correction_action for r in corrections if r.correction_action is not None])
        else:
            corr_values = []
            sub_old = tuple(v[indices] for v in old)
            sub_new = factorized_distributions(sub_logits, sub_masks)
            for j, record in enumerate(corrections):
                # ``sub_old`` is already selected.  Indexing it with the
                # original batch index crashes as soon as retention and
                # correction records are interleaved (and silently selects
                # the wrong row when it happens to fit).
                corr_values.append(_factorized_soft_kl(
                    tuple(v[j:j + 1] for v in sub_old),
                    tuple(v[j:j + 1] for v in sub_new),
                    _record_soft_target(record, sub_old, j, temperature),
                ))
            corr_loss = torch.stack(corr_values).mean()
        losses.append(corr_loss)
        details["correction_loss"] = float(corr_loss.detach())
    if retention and method != "hard-none":
        indices = [i for i, r in enumerate(records) if not r.is_correction]
        sub_logits = V4Logits(
            mode=logits.mode[indices], wait_duration=logits.wait_duration[indices],
            card=logits.card[indices], placement=logits.placement[indices],
        )
        sub_masks = type(masks)(mode=masks.mode[indices], card=masks.card[indices], placement=masks.placement[indices])
        sub_new = factorized_distributions(sub_logits, sub_masks)
        sub_old = tuple(v[indices] for v in old)
        if method == "hard-replay":
            replay_loss = _factorized_hard_nll(sub_logits, sub_masks, [r.recorded_action for r in retention])
        elif method == "soft-replay":
            # Hard and soft replay compare the same recorded source action.
            # Using the anchor argmax here changes labels and confounds the
            # target softness ablation with a replay-target change.
            replay_loss = _factorized_hard_nll(sub_logits, sub_masks, [r.recorded_action for r in retention])
        else:
            replay_loss = _factorized_kl(sub_old, sub_new, sub_masks)
        losses.append(float(retention_weight) * replay_loss)
        details["retention_loss"] = float(replay_loss.detach())
    if not losses:
        raise ValueError("batch has no active loss terms")
    total = torch.stack(losses).sum()
    details["loss"] = float(total.detach())
    return total, details

