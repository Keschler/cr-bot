"""Dataset sizing, adoption thresholds and deterministic seeds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
from simulator.observation_v3 import HAND_TOKEN_FEATURES
from simulator.rl.v4.search_teacher import (
    ACTIVE_CONTINUATION_HORIZON,
    HORIZON_DECISIONS,
    MAX_BRANCHES,
)


SEARCH_GENERATOR_VERSION: str = "search-v4-2"


ADOPT_MARGIN_ACTION: float = 0.15
"""Minimum regret(rule) to adopt a different mode/card from search."""


ADOPT_MARGIN_PLACE: float = 0.05
"""Minimum regret(rule) to adopt a different cell for the same card."""


OPPONENT_STRATEGY: str = "deterministic-cycle"


_SPELL_COL = HAND_TOKEN_FEATURES.index("is_spell")


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class SearchDatasetConfig:
    """Sizing and quota contract for the Stage-2 search dataset."""

    n_states: int = 2000
    seed: int = 0
    timing_pool_states: int = 1200
    sim_candidates: int = 1200
    horizon: int = HORIZON_DECISIONS
    max_branches: int = MAX_BRANCHES
    replay_sequences: int = 260
    min_adopted_share: float = 0.30
    min_spell_share: float = 0.15
    min_transition_share: float = 0.15
    min_sim_share: float = 0.25
    family_min_share: float = 0.08
    timing_family_weights: Any = None
    sim_family_mix: Any = None
    # ``wait`` preserves the original Stage-2 labels.  ``active`` enables
    # paired active continuations and sequential match records; it is opt-in
    # because it costs more simulator work and is not an expert oracle.
    continuation_mode: str = "wait"
    teacher_mode: str | None = None
    continuation_horizon: int | None = None
    opponent_strategy: str = "deterministic-cycle"
    placement_target_mode: str = "tested"
    active_match_decisions: int = 64

    def __post_init__(self) -> None:
        if type(self.n_states) is not int or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("timing_family_weights", "sim_family_mix"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a dict or None")
            if isinstance(value, dict):
                object.__setattr__(self, name, dict(value))
        # ``teacher_mode`` is a compatibility spelling used by early Stage-2
        # experiments.  Keep it as an alias while storing one canonical mode.
        mode = self.teacher_mode if self.teacher_mode is not None else self.continuation_mode
        if not isinstance(mode, str):
            raise TypeError("continuation_mode must be a string")
        mode = mode.strip().casefold().replace("_", "-")
        if mode == "active":
            mode = "active-rule"
        if mode not in {"wait", "active-rule", "active-policy"}:
            raise ValueError(
                "continuation_mode must be wait, active, active-rule, or active-policy"
            )
        if mode == "active-policy":
            raise ValueError("active-policy requires a public controller callback; use search_state directly or active-rule")
        object.__setattr__(self, "continuation_mode", mode)
        if self.continuation_horizon is not None:
            if type(self.continuation_horizon) is not int or self.continuation_horizon < 0:
                raise ValueError("continuation_horizon must be a non-negative integer or None")
        if type(self.active_match_decisions) is not int or self.active_match_decisions <= 0:
            raise ValueError("active_match_decisions must be a positive integer")
        if self.placement_target_mode not in {"tested", "legacy"}:
            raise ValueError("placement_target_mode must be tested or legacy")
        if not isinstance(self.opponent_strategy, str) or not self.opponent_strategy.strip():
            raise ValueError("opponent_strategy must be a non-empty string")

    @property
    def is_active(self) -> bool:
        return self.continuation_mode != "wait"

    @property
    def effective_horizon(self) -> int:
        """Resolve the branch horizon, with a longer active default."""

        if self.continuation_horizon is not None:
            return int(self.continuation_horizon)
        if self.is_active:
            return max(int(self.horizon), int(ACTIVE_CONTINUATION_HORIZON))
        return int(self.horizon)


def _hash_order(seed: int, key: str) -> int:
    return _stable_seed(seed, "search-order", key)

