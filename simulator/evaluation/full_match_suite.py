"""Sealed full-match evaluation: macro-averaged paired match scores."""

from __future__ import annotations

from typing import Final
import numpy as np


WIN: Final = 1.0
DRAW: Final = 0.5
LOSS: Final = 0.0


class MatchOutcome:
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"


def match_score(outcome: str) -> float:
    """Score one completed match: win=1, draw=1/2, loss=0."""

    if outcome == MatchOutcome.WIN:
        return WIN
    if outcome == MatchOutcome.DRAW:
        return DRAW
    if outcome == MatchOutcome.LOSS:
        return LOSS
    raise ValueError(f"unknown match outcome: {outcome!r}")


def macro_match_score(stratum_scores: dict[str, list[float]]) -> float:
    """Macro average over opponent/deck strata (primary match metric).

    Macro-averaging is primary because the evaluation distribution must not
    be dominated by whichever opponent happens to have the most cells.
    """

    if not stratum_scores:
        raise ValueError("stratum_scores must not be empty")
    means: list[float] = []
    for stratum, scores in stratum_scores.items():
        arr = np.asarray(scores, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"stratum {stratum!r} must contain finite scores")
        means.append(float(np.mean(arr)))
    return float(np.mean(means))


def micro_match_score(stratum_scores: dict[str, list[float]]) -> float:
    """Secondary micro average (flat over all cells)."""

    cells: list[float] = []
    for scores in stratum_scores.values():
        cells.extend(float(value) for value in scores)
    arr = np.asarray(cells, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("cells must be finite and non-empty")
    return float(np.mean(arr))


def crown_differential(
    challenger_crowns: list[int],
    champion_crowns: list[int],
) -> float:
    """Secondary mean crown differential (challenger minus champion)."""

    challenger = np.asarray(challenger_crowns, dtype=np.float64)
    champion = np.asarray(champion_crowns, dtype=np.float64)
    if challenger.shape != champion.shape or challenger.size == 0:
        raise ValueError("crown arrays must match and be non-empty")
    return float(np.mean(challenger - champion))


def side_swap_blocks(
    pair_ids: list[str],
    improvements: list[float],
) -> list[list[float]]:
    """Group paired improvements into side-swap blocks for bootstrapping."""

    if len(pair_ids) != len(improvements):
        raise ValueError("pair_ids and improvements must match")
    grouped: dict[str, list[float]] = {}
    for pair_id, improvement in zip(pair_ids, improvements):
        grouped.setdefault(pair_id, []).append(float(improvement))
    return [grouped[key] for key in sorted(grouped)]


__all__ = [
    "DRAW",
    "LOSS",
    "WIN",
    "MatchOutcome",
    "crown_differential",
    "macro_match_score",
    "match_score",
    "micro_match_score",
    "side_swap_blocks",
]
