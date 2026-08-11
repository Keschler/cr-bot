from __future__ import annotations

import torch
from torch.nn import functional as F


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    probability = logits.softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
    return (((1.0 - probability) ** gamma) * cross_entropy).mean()


def spatial_soft_target_loss(
    heatmap_logits: torch.Tensor,
    class_targets: torch.Tensor,
    target_heatmaps: torch.Tensor,
    has_target: torch.Tensor,
) -> torch.Tensor:
    selected = has_target.bool()
    if not selected.any():
        return heatmap_logits.sum() * 0.0
    class_indices = class_targets[selected] - 1
    predicted = heatmap_logits[selected, class_indices].flatten(1)
    target = target_heatmaps[selected].flatten(1)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(target * predicted.log_softmax(dim=1)).sum(dim=1).mean()
