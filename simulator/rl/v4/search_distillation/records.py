"""Typed records and configuration for search distillation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SEARCH_DISTILLATION_VERSION = "search-distillation-v2"
DEFAULT_CONTEXT = 32
DEFAULT_TEMPERATURE = 0.25
DEFAULT_MIN_LCB = 0.05


@dataclass(frozen=True, slots=True)
class ActionTarget:
    """A rectangular V4 action used by the experiment harness."""

    mode: int
    slot: int = 0
    row: int = 0
    col: int = 0

    def key(self) -> tuple[int, int, int, int]:
        return (int(self.mode), int(self.slot), int(self.row), int(self.col))


@dataclass(frozen=True, slots=True)
class FactorizedTarget:
    """A legal factorised action distribution.

    ``card`` is conditional on PLAY and ``placement`` is conditional on each
    card slot.  WAIT has no meaningful card or placement distribution.
    """

    mode: np.ndarray
    card: np.ndarray
    placement: np.ndarray

    def __post_init__(self) -> None:
        mode = np.asarray(self.mode, dtype=np.float32)
        card = np.asarray(self.card, dtype=np.float32)
        placement = np.asarray(self.placement, dtype=np.float32)
        if mode.shape != (2,):
            raise ValueError("mode target must have shape [2]")
        if card.ndim != 1:
            raise ValueError("card target must be one-dimensional")
        if placement.shape[:1] != card.shape or placement.ndim != 3:
            raise ValueError("placement target must have shape [slots, rows, cols]")
        if not np.isfinite(mode).all() or not np.isfinite(card).all() or not np.isfinite(placement).all():
            raise ValueError("target distributions must be finite")
        if np.any(mode < -1e-6) or np.any(card < -1e-6) or np.any(placement < -1e-6):
            raise ValueError("target distributions must be non-negative")
        if not np.isclose(float(mode.sum()), 1.0, atol=2e-4):
            raise ValueError("mode target must sum to one")
        if mode[1] > 1e-6 and not np.isclose(float(card.sum()), 1.0, atol=2e-4):
            raise ValueError("PLAY card target must sum to one")
        for slot in range(card.shape[0]):
            if card[slot] > 1e-6 and not np.isclose(float(placement[slot].sum()), 1.0, atol=2e-4):
                raise ValueError("each active placement target must sum to one")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "card", card)
        object.__setattr__(self, "placement", placement)


@dataclass(slots=True)
class CachedPrefix:
    """Public observations around one source-match decision."""

    key: str
    match_id: str
    offset: int
    observations: list[dict[str, np.ndarray]]
    legal_play: list[np.ndarray]
    legal_wait: list[bool]
    recorded_action: ActionTarget
    correction_action: ActionTarget | None = None
    branches: list[dict[str, Any]] = field(default_factory=list)
    paired_mean: float = 0.0
    paired_se: float = 0.0
    correction_lcb: float = 0.0
    is_correction: bool = False
    # Optional source metadata used by independent paired outcome evaluation.
    # Keeping paths and small config values instead of whole matches makes the
    # prefix cache remain compact while allowing validation to revisit an
    # action that was absent from the discovery branch set.
    match_path: str | None = None
    opponent_spec: dict[str, Any] | None = None
    outcome_horizon: int = 128
    outcome_replicas: int = 4
    outcome_replica_seed_base: int = 0

    @property
    def current(self) -> dict[str, np.ndarray]:
        return self.observations[-1]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    context: int = DEFAULT_CONTEXT

    def __post_init__(self) -> None:
        if type(self.context) is not int or self.context < 0:
            raise ValueError("context must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ValidatedCorrection:
    """A confirmed search target with its source and paired evidence."""

    entry_id: str
    match_key: tuple[str, str]
    offset: int
    verdict: dict[str, Any]
    action: ActionTarget
    statistics: dict[str, Any]
