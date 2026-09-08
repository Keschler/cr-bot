"""Sealed tactical evaluation: branch consequences, not teacher agreement.

For one tactical state, the challenger action and the champion action are
branched from the *identical* state with an identical continuation protocol.
The comparison is the consequence difference ``ti = F(si, aC) - F(si, aB)``
from observable simulator facts.  Action disagreement alone is never a
regression.  The evaluation outcome definition (family weights, primary
weights, catastrophe thresholds) is fixed *before* any challenger trains.
"""

from __future__ import annotations

from typing import Final
import numpy as np


TACTICAL_FAMILIES: Final[tuple[str, ...]] = (
    "offense",
    "ground-defense",
    "air-defense",
    "spell-value",
    "kiting-cycle",
    "low-elixir",
    "counterpush",
    "bridge-defense",
)

CONSEQUENCE_KEYS: Final[tuple[str, ...]] = (
    "own_tower_damage_prevented",
    "enemy_tower_damage",
    "threat_removed_value",
    "surviving_unit_value",
    "elixir_swing",
)


def consequence_vector(
    *,
    own_tower_damage_prevented: float,
    enemy_tower_damage: float,
    threat_removed_value: float,
    surviving_unit_value: float,
    elixir_swing: float,
) -> dict[str, float]:
    """Build one branch's observable consequence vector from simulator facts."""

    values = {
        "own_tower_damage_prevented": own_tower_damage_prevented,
        "enemy_tower_damage": enemy_tower_damage,
        "threat_removed_value": threat_removed_value,
        "surviving_unit_value": surviving_unit_value,
        "elixir_swing": elixir_swing,
    }
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
    return {key: float(value) for key, value in values.items()}


def consequence_primary_score(
    vector: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Collapse a consequence vector with *frozen, predeclared* weights."""

    if set(vector) != set(CONSEQUENCE_KEYS):
        raise ValueError(f"vector must contain exactly {CONSEQUENCE_KEYS}")
    if set(weights) != set(CONSEQUENCE_KEYS):
        raise ValueError(f"weights must contain exactly {CONSEQUENCE_KEYS}")
    return float(sum(vector[key] * float(weights[key]) for key in CONSEQUENCE_KEYS))


def macro_tactical_score(
    family_scores: dict[str, list[float]],
) -> float:
    """Equal-weight macro average so easy families cannot hide a collapse."""

    if not family_scores:
        raise ValueError("family_scores must not be empty")
    means: list[float] = []
    for family, scores in family_scores.items():
        if family not in TACTICAL_FAMILIES:
            raise ValueError(f"unknown tactical family: {family!r}")
        arr = np.asarray(scores, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"family {family!r} must contain finite scores")
        means.append(float(np.mean(arr)))
    return float(np.mean(means))


def catastrophic_regression_rate(
    improvements: np.ndarray | list[float],
    tau: float,
) -> float:
    """Fraction of states with ``ti <= -tau`` (severe, predeclared loss)."""

    if not np.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be a finite positive threshold")
    arr = np.asarray(improvements, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("improvements must be finite and non-empty")
    return float(np.mean(arr <= -float(tau)))


__all__ = [
    "CONSEQUENCE_KEYS",
    "TACTICAL_FAMILIES",
    "catastrophic_regression_rate",
    "consequence_primary_score",
    "consequence_vector",
    "macro_tactical_score",
]
