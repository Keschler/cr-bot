"""Paired champion-challenger statistics.

All primary comparisons use paired cells: the challenger and the champion
play the *same* cell, so starting hand, opponent behavior, and environment
stochasticity cancel out.  Reports cite one-sided 95% lower confidence
bounds from a stratified block bootstrap (blocks = side-swap pairs and
other deliberately linked variants), never a bare win percentage.
"""

from __future__ import annotations

import math
from statistics import NormalDist
import numpy as np


def paired_differences(
    challenger: np.ndarray | list[float],
    champion: np.ndarray | list[float],
) -> np.ndarray:
    """Return per-cell ``challenger - champion`` improvements."""

    challenger_arr = np.asarray(challenger, dtype=np.float64)
    champion_arr = np.asarray(champion, dtype=np.float64)
    if challenger_arr.shape != champion_arr.shape:
        raise ValueError("challenger and champion score arrays must match")
    if challenger_arr.size == 0:
        raise ValueError("score arrays must not be empty")
    if not np.isfinite(challenger_arr).all() or not np.isfinite(champion_arr).all():
        raise ValueError("scores must be finite")
    return challenger_arr - champion_arr


def stratified_paired_bootstrap_lcb(
    blocks: list[list[float]] | list[np.ndarray],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Block bootstrap LCB for the mean paired improvement.

    Each block holds the linked ``di`` values of one base draw (e.g. both
    side swaps).  Blocks are resampled with replacement; the block mean
    preserves the benchmark strata and paired structure.  Returns the
    observed mean and the one-sided lower confidence bound.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if type(n_bootstrap) is not int or n_bootstrap < 1000:
        raise ValueError("n_bootstrap must be at least 1000")
    if not blocks:
        raise ValueError("blocks must not be empty")
    block_arrays = [np.asarray(block, dtype=np.float64).ravel() for block in blocks]
    for block in block_arrays:
        if block.size == 0 or not np.isfinite(block).all():
            raise ValueError("every block must contain finite values")
    observed = float(np.mean([float(np.mean(block)) for block in block_arrays]))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(block_arrays), size=(n_bootstrap, len(block_arrays)))
    replicate_means = np.array(
        [float(np.mean([float(np.mean(block_arrays[i])) for i in row])) for row in index]
    )
    quantile = (1.0 - confidence) * 100.0
    lcb = float(np.quantile(replicate_means, quantile / 100.0))
    return {"mean": observed, "lcb": lcb, "confidence": float(confidence)}


def required_sample_size(
    sigma_d: float,
    delta: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Pilot-based promotion sample size (paper eq. 24), rounded up.

    The pilot estimates the *variance* of paired differences only, never
    the candidate's true effect.  Callers round the result further upward
    to satisfy minimum per-stratum counts.
    """

    if not math.isfinite(sigma_d) or sigma_d <= 0.0:
        raise ValueError("sigma_d must be finite and positive")
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_beta = normal.inv_cdf(power)
    return int(math.ceil(((z_alpha + z_beta) * sigma_d / delta) ** 2))


def holm_reject(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down rejections for family-level no-regression statements."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    for value in p_values:
        if not isinstance(value, float) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be floats in [0, 1]")
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    rejected = [False] * len(p_values)
    remaining = len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (remaining - rank):
            rejected[index] = True
        else:
            break
    return rejected


__all__ = [
    "holm_reject",
    "paired_differences",
    "required_sample_size",
    "stratified_paired_bootstrap_lcb",
]
