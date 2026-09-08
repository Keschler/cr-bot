"""Placement quality metrics for V4 distillation evaluation.

Exact-cell top-1 accuracy is too crude to judge placement: with soft
supervision spread over neighboring cells, it marks a near-perfect
placement as completely wrong.  This module reports the metric set that
distinguishes "bad geometry" from "misleading measurement":

* top-1 grid error as Euclidean and Manhattan distance in cells;
* within-1 / within-2 accuracy under Chebyshev distance (within-1 is
  exactly the teacher's 3x3 soft region);
* ``soft_region_mass``: total predicted probability on cells the teacher
  considers good — "how much mass does the actor assign to placements
  the teacher considers good?";
* exact-cell accuracy kept as a secondary diagnostic only.

All functions are NumPy-only so the evaluation harness never depends on
torch; callers convert model tensors at the boundary.
"""

from __future__ import annotations

import numpy as np


def top1_cells(
    prob_maps: np.ndarray, card_slots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax cell per sample on the *selected* card's probability map.

    ``prob_maps`` is ``[N, K, H, W]`` (one distribution per card),
    ``card_slots`` is ``[N]``.  Returns integer ``(rows, cols)`` of shape
    ``[N]`` each.
    """

    prob_maps = np.asarray(prob_maps, dtype=np.float64)
    card_slots = np.asarray(card_slots)
    if prob_maps.ndim != 4:
        raise ValueError("prob_maps must have shape [N, K, H, W]")
    if card_slots.shape != prob_maps.shape[:1]:
        raise ValueError("card_slots must have shape [N]")
    count, slots, height, width = prob_maps.shape
    if count == 0:
        raise ValueError("prob_maps must not be empty")
    if bool((card_slots < 0).any()) or bool((card_slots >= slots).any()):
        raise ValueError("card_slots contains an out-of-range slot")
    selected = prob_maps[np.arange(count), card_slots.astype(int)]
    flat = selected.reshape(count, -1)
    rows = (flat.argmax(axis=1) // width).astype(int)
    cols = (flat.argmax(axis=1) % width).astype(int)
    return rows, cols


def _distances(
    pred_rows: np.ndarray,
    pred_cols: np.ndarray,
    true_rows: np.ndarray,
    true_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_rows = np.asarray(pred_rows, dtype=np.float64)
    pred_cols = np.asarray(pred_cols, dtype=np.float64)
    true_rows = np.asarray(true_rows, dtype=np.float64)
    true_cols = np.asarray(true_cols, dtype=np.float64)
    if not (pred_rows.shape == pred_cols.shape == true_rows.shape == true_cols.shape):
        raise ValueError("cell arrays must share one shape")
    if pred_rows.size == 0:
        raise ValueError("cell arrays must not be empty")
    dr = np.abs(pred_rows - true_rows)
    dc = np.abs(pred_cols - true_cols)
    euclidean = np.sqrt(dr**2 + dc**2)
    manhattan = dr + dc
    chebyshev = np.maximum(dr, dc)
    return euclidean, manhattan, chebyshev


def placement_error_stats(
    pred_rows: np.ndarray,
    pred_cols: np.ndarray,
    true_rows: np.ndarray,
    true_cols: np.ndarray,
) -> dict[str, float]:
    """Summarize top-1 spatial error against the teacher's central cell.

    ``within_1``/``within_2`` use Chebyshev distance so within-1 coincides
    with the teacher's 3x3 soft region; Euclidean/Manhattan means and
    medians are reported as continuous errors in cells.
    """

    euclidean, manhattan, chebyshev = _distances(pred_rows, pred_cols, true_rows, true_cols)
    count = int(euclidean.size)
    return {
        "n": float(count),
        "exact_acc": float(np.mean(chebyshev == 0)),
        "within_1_acc": float(np.mean(chebyshev <= 1)),
        "within_2_acc": float(np.mean(chebyshev <= 2)),
        "mean_euclidean": float(np.mean(euclidean)),
        "median_euclidean": float(np.median(euclidean)),
        "mean_manhattan": float(np.mean(manhattan)),
        "median_manhattan": float(np.median(manhattan)),
    }


def soft_region_mass(
    prob_maps: np.ndarray,
    card_slots: np.ndarray,
    soft_targets: np.ndarray,
) -> dict[str, float]:
    """Probability mass the actor assigns to the teacher's good region.

    ``P_target-region = sum over c in soft support of p_theta(c)`` on the
    selected card's map.  ``soft_targets`` is ``[N, H, W]`` with positive
    mass exactly on teacher-approved cells (e.g. the 3x3 neighborhood).
    Reports the mean over samples plus the fraction of samples putting a
    majority of mass inside the region.
    """

    prob_maps = np.asarray(prob_maps, dtype=np.float64)
    soft_targets = np.asarray(soft_targets, dtype=np.float64)
    if prob_maps.ndim != 4:
        raise ValueError("prob_maps must have shape [N, K, H, W]")
    if soft_targets.shape != prob_maps.shape[:1] + prob_maps.shape[2:]:
        raise ValueError("soft_targets must have shape [N, H, W]")
    count = prob_maps.shape[0]
    if count == 0:
        raise ValueError("inputs must not be empty")
    if bool((soft_targets < 0.0).any()) or not np.isfinite(soft_targets).all():
        raise ValueError("soft_targets must be finite and non-negative")
    card_slots = np.asarray(card_slots)
    selected = prob_maps[np.arange(count), card_slots.astype(int)]
    support = soft_targets > 0.0
    if not bool(support.any(axis=(1, 2)).all()):
        raise ValueError("every soft target must approve at least one cell")
    mass = (selected * support).sum(axis=(1, 2))
    return {
        "n": float(count),
        "mean_mass": float(np.mean(mass)),
        "median_mass": float(np.median(mass)),
        "majority_fraction": float(np.mean(mass >= 0.5)),
    }


def per_sample_placement_stats(
    prob_maps: np.ndarray,
    card_slots: np.ndarray,
    soft_targets: np.ndarray,
    true_rows: np.ndarray,
    true_cols: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-sample placement statistics for grouped breakdowns.

    Returns float arrays of shape ``[N]``: top-1 ``pred_row``/``pred_col``,
    ``euclidean``/``manhattan``/``chebyshev`` error to the teacher cell,
    and ``mass`` inside the teacher-approved region.  Callers group these
    by family, card, or any other sample attribute.
    """

    prob_maps = np.asarray(prob_maps, dtype=np.float64)
    card_slots = np.asarray(card_slots)
    count = prob_maps.shape[0]
    pred_rows, pred_cols = top1_cells(prob_maps, card_slots)
    euclidean, manhattan, chebyshev = _distances(
        pred_rows, pred_cols, np.asarray(true_rows), np.asarray(true_cols)
    )
    soft_targets = np.asarray(soft_targets, dtype=np.float64)
    if soft_targets.shape != prob_maps.shape[:1] + prob_maps.shape[2:]:
        raise ValueError("soft_targets must have shape [N, H, W]")
    selected = prob_maps[np.arange(count), card_slots.astype(int)]
    mass = (selected * (soft_targets > 0.0)).sum(axis=(1, 2))
    return {
        "pred_row": pred_rows.astype(float),
        "pred_col": pred_cols.astype(float),
        "euclidean": euclidean,
        "manhattan": manhattan,
        "chebyshev": chebyshev,
        "mass": mass,
    }


__all__ = [
    "per_sample_placement_stats",
    "placement_error_stats",
    "soft_region_mass",
    "top1_cells",
]
