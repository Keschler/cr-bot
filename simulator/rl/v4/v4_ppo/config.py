"""Configuration and runtime identity for fixed-decision V4 PPO."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class V4OpponentLane:
    """One scripted opponent lane: deck archetype, play strategy, source.

    ``strategy="deterministic-cycle"`` is the legible anchor controller;
    every other lane uses a distinct heuristic strategy so PPO cannot
    memorize one cycle.  ``controller_lane`` only applies to the
    deterministic anchor (heuristic factories manage their own lanes).
    """

    archetype: str
    source: str
    strategy: str = "deterministic-cycle"
    controller_lane: str = "alternate"


SMOKE_OPPONENT_LANES: tuple[V4OpponentLane, ...] = (
    V4OpponentLane("beatdown", "ground-defense", "beatdown"),
    V4OpponentLane("beatdown", "ground-defense", "deterministic-cycle", "left"),
    V4OpponentLane("air-beatdown", "air-defense", "air-beatdown"),
    V4OpponentLane("aggressive-pressure", "isolated-offense", "aggressive-pressure"),
    V4OpponentLane("defensive-cycle", "kiting-cycling-elixir", "defensive-cycle"),
    V4OpponentLane("siege-bait", "spell-situations", "siege-bait"),
    V4OpponentLane("random-legal", "isolated-offense", "random-legal"),
    V4OpponentLane("aggressive-pressure", "bridge-defense", "aggressive-pressure"),
)


@dataclass(frozen=True, slots=True)
class V4PPOConfig:
    """Knobs for one short-scenario V4 PPO smoke run."""

    n_updates: int = 5
    decisions_per_rollout: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    bc_coef: float = 0.05
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    ppo_epochs: int = 2
    seed: int = 0
    device: str = "cpu"
    # Quarantine thresholds are initial values to calibrate, except the
    # hard ones (illegal actions, non-finite losses) which are exact.
    max_approx_kl: float = 0.03
    max_mean_abs_log_ratio: float = 0.05
    max_entropy_collapse_fraction: float = 0.50
    max_placement_grad_ratio: float = 10.0

    def __post_init__(self) -> None:
        if type(self.n_updates) is not int or self.n_updates <= 0:
            raise ValueError("n_updates must be a positive integer")
        if type(self.decisions_per_rollout) is not int or self.decisions_per_rollout <= 0:
            raise ValueError("decisions_per_rollout must be a positive integer")
        if type(self.ppo_epochs) is not int or self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be a positive integer")
        for name in (
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "value_coef",
            "entropy_coef",
            "bc_coef",
            "actor_lr",
            "critic_lr",
            "max_approx_kl",
            "max_mean_abs_log_ratio",
            "max_entropy_collapse_fraction",
            "max_placement_grad_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= float(self.gae_lambda) <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")


V4_RUNTIME_VERSION = "v4-fixed-step-public-history-2"

