"""Policy movement, gradient measurements, and post-update quarantine."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from .runtime import _forward_sequence
from ..model_v4 import RecurrentV4Policy, V4ValueHead, V4ActionBatch
from ...foundation.trajectory import ActionMasks
from .records import V4RolloutBatch, V4UpdateReport
from ...foundation.objectives import PPOObjectiveConfig, PPOObjectiveResult


def _head_grad_norms(
    policy: RecurrentV4Policy, critic: V4ValueHead
) -> dict[str, float]:
    """Per-head gradient norms (the update-22 early-warning signal)."""

    def norm(modules: list[Any]) -> float:
        total = 0.0
        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    total += float(param.grad.detach().float().norm().item()) ** 2
        return total**0.5

    heads = policy.heads
    return {
        "mode": norm([heads.mode_head]),
        "duration": norm([heads.duration_head]),
        "card": norm([heads.card_query, heads.card_key, heads.card_score]),
        "placement": norm([heads.spatial_key, heads.placement_query]),
        "encoder": norm([policy.encoder]),
        "recurrent": norm([policy.recurrent]),
        "critic": norm([critic]),
    }


def per_head_kl(
    old_logits: dict[str, torch.Tensor],
    new_logits: Any,
    masks: ActionMasks,
) -> dict[str, torch.Tensor]:
    """Exact per-head KL(old || new) on identical masks, ``[B, T]`` each.

    Mode/duration/card compare full distributions; placement averages the
    per-card map KL over slots with a legal cell (empty maps contribute
    zero — they can never be selected).  Identical masks on both sides is
    required; the smoke reuses the stored batch masks by construction.
    """

    from ..model_v4 import _masked_log_softmax

    def kl(old_logp: torch.Tensor, new_logp: torch.Tensor) -> torch.Tensor:
        # Illegal entries hold -inf on both sides with zero mass; their
        # 0 * (-inf - -inf) NaN contributes nothing by definition of KL.
        old_prob = old_logp.exp()
        terms = old_prob * (old_logp - new_logp)
        return torch.where(
            torch.isfinite(terms), terms, torch.zeros_like(terms)
        ).sum(dim=-1)

    mode_old = _masked_log_softmax(old_logits["mode"], masks.mode)
    mode_new = _masked_log_softmax(new_logits.mode, masks.mode)
    out: dict[str, torch.Tensor] = {
        "mode": kl(mode_old, mode_new),
        "duration": kl(
            torch.log_softmax(old_logits["duration"], dim=-1),
            torch.log_softmax(new_logits.wait_duration, dim=-1),
        ),
    }
    # Broke WAIT rows may hold no legal card at all; those rows contribute
    # zero card KL (they can never select a card) instead of raising.
    card_ok = masks.card.any(dim=-1)
    card_term = torch.zeros_like(old_logits["card"][..., 0])
    if bool(card_ok.any().item()):
        card_term[card_ok] = kl(
            _masked_log_softmax(old_logits["card"][card_ok], masks.card[card_ok]),
            _masked_log_softmax(new_logits.card[card_ok], masks.card[card_ok]),
        )
    out["card"] = card_term
    rows, cols = new_logits.placement.shape[-2:]
    slots = masks.card.shape[-1]
    cells = rows * cols

    def selected(logits: torch.Tensor) -> torch.Tensor:
        flat = logits.reshape(logits.shape[:-3] + (slots, cells))
        mask = masks.placement.reshape(masks.placement.shape[:-3] + (slots, cells))
        safe = torch.where(
            mask.any(dim=-1, keepdim=True),
            torch.where(mask, flat, torch.full_like(flat, float("-inf"))),
            torch.zeros_like(flat),
        )
        return torch.log_softmax(safe, dim=-1)

    old_maps = selected(old_logits["placement"])
    new_maps = selected(new_logits.placement)
    per_cell = old_maps.exp() * (old_maps - new_maps)
    per_cell = torch.where(torch.isfinite(per_cell), per_cell, torch.zeros_like(per_cell))
    legal = masks.placement.reshape(masks.placement.shape[:-3] + (slots, cells)).any(dim=-1)
    base = (
        (per_cell.sum(dim=-1) * legal.to(per_cell.dtype)).sum(dim=-1)
        / legal.sum(dim=-1).clamp_min(1).to(per_cell.dtype)
    )
    out["placement"] = torch.where(
        legal.any(dim=-1),
        base,
        torch.zeros_like(base),
    )
    return out


def _term_grad_norm(loss: torch.Tensor, params: list[Any]) -> float:
    """L2 gradient norm of one loss term over actor parameters (measured)."""

    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().norm().item()) ** 2
    return total**0.5


def _anchor_agreement(
    decoded: Any, anchor_actions: Any
) -> dict[str, float | None]:
    """Argmax agreement of the updated policy against frozen anchor actions.

    Mode agreement over all rows; card agreement over rows PLAY on both
    sides; placement within-1 over same-card PLAY rows.  ``None`` when the
    denominator is empty.
    """

    mode_a = decoded.mode.reshape(-1)
    mode_b = anchor_actions.mode.reshape(-1)
    n_rows = int(mode_a.numel())
    if n_rows == 0:
        return {"mode": None, "card": None, "place": None}
    mode_agree = float((mode_a == mode_b).float().mean().item())
    both_play = (mode_a == 1) & (mode_b == 1)
    if bool(both_play.any().item()):
        card_a = decoded.card_slot.reshape(-1)[both_play]
        card_b = anchor_actions.card_slot.reshape(-1)[both_play]
        card_agree: float | None = float((card_a == card_b).float().mean().item())
        same_card = card_a == card_b
        if bool(same_card.any().item()):
            place_a = decoded.placement.reshape(-1, 2)[both_play][same_card]
            place_b = anchor_actions.placement.reshape(-1, 2)[both_play][same_card]
            chebyshev = (place_a - place_b).abs().max(dim=-1).values
            place_within1: float | None = float((chebyshev <= 1).float().mean().item())
        else:
            place_within1 = None
    else:
        card_agree = None
        place_within1 = None
    return {"mode": mode_agree, "card": card_agree, "place": place_within1}


def _decode_drift(first: Any, second: Any) -> dict[str, float]:
    """Behavioral movement between two deterministic decodes.

    Fractions over rows: mode flips; card flips (rows PLAY on both sides);
    placement Chebyshev moves >1 (rows PLAY on both sides with the same
    card, so pure placement drift is isolated from card switches);
    duration flips (rows WAIT on both sides, the only rows where the
    duration head executes).
    """

    import torch

    first_mode = first.mode.reshape(-1)
    second_mode = second.mode.reshape(-1)
    n_rows = int(first_mode.numel())
    if n_rows == 0:
        return {"mode_flip": 0.0, "card_flip": 0.0, "placement_move": 0.0, "duration_flip": 0.0}
    mode_flip = float((first_mode != second_mode).float().mean().item())
    both_play = (first_mode == 1) & (second_mode == 1)
    if bool(both_play.any().item()):
        first_card = first.card_slot.reshape(-1)[both_play]
        second_card = second.card_slot.reshape(-1)[both_play]
        card_flip = float((first_card != second_card).float().mean().item())
        same_card = first_card == second_card
        if bool(same_card.any().item()):
            first_place = first.placement.reshape(-1, 2)[both_play][same_card]
            second_place = second.placement.reshape(-1, 2)[both_play][same_card]
            chebyshev = (first_place - second_place).abs().max(dim=-1).values
            placement_move = float((chebyshev > 1).float().mean().item())
        else:
            placement_move = 0.0
    else:
        card_flip = 0.0
        placement_move = 0.0
    both_wait = (first_mode == 0) & (second_mode == 0)
    if bool(both_wait.any().item()):
        duration_flip = float(
            (first.wait_duration.reshape(-1)[both_wait]
             != second.wait_duration.reshape(-1)[both_wait])
            .float()
            .mean()
            .item()
        )
    else:
        duration_flip = 0.0
    return {
        "mode_flip": mode_flip,
        "card_flip": card_flip,
        "placement_move": placement_move,
        "duration_flip": duration_flip,
    }


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Fraction of return variance explained by the critic."""

    returns = returns.detach().float()
    values = values.detach().float()
    variance = returns.var(unbiased=False)
    if not bool(torch.isfinite(variance).item()) or float(variance.item()) <= 0.0:
        return 0.0
    residual = (returns - values).var(unbiased=False)
    return max(0.0, 1.0 - float(residual.item()) / float(variance.item()))


def build_update_report(
    policy: RecurrentV4Policy,
    batch: V4RolloutBatch,
    pre_decoded: V4ActionBatch,
    result: PPOObjectiveResult,
    values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    objective_config: PPOObjectiveConfig,
    grad_norms: dict[str, float],
    policy_grad_norm: float,
    bc_grad_norm: float,
) -> V4UpdateReport:
    """Measure the updated actor and evaluate quarantine gates after stepping."""
    trajectory = batch.trajectory
    sequence = trajectory.sequence
    device = sequence.raster.device
    with torch.no_grad():
        explained = _explained_variance(returns, values)
    per_head = {
        key: float(value.detach().mean().item())
        for key, value in (batch.per_head_entropy or {}).items()
    }
    with torch.no_grad():
        post_logits, _, _ = _forward_sequence(policy, sequence, device)
        post_decoded = policy.act_deterministic(
            post_logits, trajectory.action_masks
        )
        # The objective was evaluated before optimizer.step().  Measure
        # the actual candidate against the collected behavior policy.
        post_log_probs = policy.log_prob(
            post_logits, trajectory.action_masks, trajectory.actions
        )
        post_log_ratio = post_log_probs - trajectory.old_log_probs.to(device)
        post_approx_kl = float((-post_log_ratio).mean().item())
        post_mean_abs_log_ratio = float(post_log_ratio.abs().mean().item())
        post_clip_fraction = float(
            ((post_log_ratio.exp() - 1.0).abs() > objective_config.clip_epsilon)
            .float().mean().item()
        )
    drift = _decode_drift(pre_decoded, post_decoded)
    placement_ratio = grad_norms.get("placement", 0.0) / max(grad_norms.get("mode", 1e-8), 1e-8)
    reasons: list[str] = []
    if not bool(torch.isfinite(result.total_loss).item()):
        reasons.append("non-finite total loss")
    if not bool(torch.isfinite(post_log_ratio).all().item()):
        reasons.append("non-finite post-update log ratio")
    if post_approx_kl > 0.03:
        reasons.append(f"approx-kl {post_approx_kl:.4f} above 0.03")
    if post_mean_abs_log_ratio > 0.05:
        reasons.append(
            f"mean-abs-log-ratio {post_mean_abs_log_ratio:.4f} above 0.05"
        )
    if drift["mode_flip"] > 0.15:
        reasons.append(f"mode flip {drift['mode_flip']:.3f} above 0.15")
    if drift["card_flip"] > 0.25:
        reasons.append(f"card flip {drift['card_flip']:.3f} above 0.25")
    if drift["placement_move"] > 0.40:
        reasons.append(
            f"placement move {drift['placement_move']:.3f} above 0.40"
        )
    head_kl: dict[str, float] = {}
    if batch.old_head_logits is not None:
        with torch.no_grad():
            for key, value in per_head_kl(
                {k: v.to(device) for k, v in batch.old_head_logits.items()},
                post_logits,
                trajectory.action_masks,
            ).items():
                head_kl[key] = (0.0 if key == "duration" and policy.config.fixed_wait
                                else float(value.detach().mean().item()))
        for key, value in head_kl.items():
            if not np.isfinite(value):
                reasons.append(f"non-finite per-head KL on {key}")
                break
        placement_kl = head_kl.get("placement", 0.0)
        mode_kl = head_kl.get("mode", 0.0)
        if placement_kl > 20.0 * max(mode_kl, 1e-6) and placement_kl > 5e-3:
            reasons.append(
                f"placement KL {placement_kl:.4f} dominates mode KL "
                f"{mode_kl:.4f} (update-22 pattern)"
            )
    if placement_ratio > 10.0 and grad_norms.get("mode", 0.0) > 1e-6:
        # Landscape warning only (calibrated on the micro smoke): a
        # confident champion legitimately concentrates gradients where
        # uncertainty lives, and a near-zero mode gradient makes any
        # ratio explode mechanically.  It quarantines only alongside
        # movement, which is already covered above.
        per_head["placement_grad_warning"] = placement_ratio
    # ``logits`` still holds the pre-update forward pass.  Recompute the
    # entropy from the post-step logits so the quarantine gate measures the
    # update that just ran rather than reporting stale pre-update entropy.
    post_entropy = policy.action_entropy(post_logits, trajectory.action_masks)
    for key, value in post_entropy.items():
        pre = float(batch.per_head_entropy[key].detach().mean().item()) if batch.per_head_entropy is not None and key in batch.per_head_entropy else float("nan")
        post = float(value.detach().mean().item())
        if pre > 1e-6:
            collapse = 1.0 - post / pre
            per_head[f"{key}_collapse"] = collapse
            if key == "joint" and collapse > 0.50:
                reasons.append(
                    f"joint entropy collapse {collapse:.2f} above 0.50"
                )
    anchor_agree = (
        _anchor_agreement(post_decoded, batch.anchor_actions)
        if batch.anchor_actions is not None
        else {"mode": None, "card": None, "place": None}
    )
    return V4UpdateReport(
        policy_loss=float(result.policy_loss.item()),
        value_loss=float(result.value_loss.item()),
        entropy=float(result.entropy.item()),
        per_head_entropy=per_head,
        behavior_cloning_loss=float(result.behavior_cloning_loss.item()),
        approx_kl=post_approx_kl,
        mean_abs_log_ratio=post_mean_abs_log_ratio,
        clip_fraction=post_clip_fraction,
        grad_norms=grad_norms,
        explained_variance=explained,
        mode_flip_fraction=drift["mode_flip"],
        card_flip_fraction=drift["card_flip"],
        placement_move_fraction=drift["placement_move"],
        duration_flip_fraction=drift["duration_flip"],
        per_head_kl=head_kl,
        policy_grad_norm=policy_grad_norm,
        bc_grad_norm=bc_grad_norm,
        returns_mean=float(returns.detach().float().mean().item()),
        returns_std=float(returns.detach().float().std(unbiased=False).item()),
        advantages_mean=float(advantages.detach().float().mean().item()),
        advantages_std=float(advantages.detach().float().std(unbiased=False).item()),
        anchor_mode_agree=anchor_agree["mode"],
        anchor_card_agree=anchor_agree["card"],
        anchor_place_within1=anchor_agree["place"],
        quarantined=bool(reasons),
        quarantine_reasons=reasons,
    )
