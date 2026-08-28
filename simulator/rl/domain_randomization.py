"""Deterministic, bounded domain randomization for training environments.

The current simulator can safely randomize interface-level uncertainty without
silently changing the pinned physics ruleset:

* policy decision cadence by whole physics ticks;
* action latency by whole decision intervals;
* normalized public V2 entity-token noise.

Mechanic parameters such as collision radii, targeting tie-breaks, and attack
timers are deliberately not randomized here.  They require an evidence-backed
ruleset variant and should not be changed by an unlabeled training wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .league import deterministic_seed

# Keep ``import rl`` lightweight and safe when this directory is imported as a
# top-level package from the simulator working directory.  The environment
# and V2 container are imported only when a wrapper is constructed/used.
class DomainRandomizationError(ValueError):
    """Raised when a randomization profile is invalid or misapplied."""


# Pinned indices from the public V2 schema.  Only these four public scalar
# features are perturbed by this wrapper.
_FEATURE_INDEX = {"x": 2, "y": 3, "hp_fraction": 4, "confidence": 28}


def _simulator_env_class() -> Any:
    try:
        from ..env import SimulatorEnv
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.env import SimulatorEnv
    return SimulatorEnv


def _policy_observation_v2_class() -> Any:
    try:
        from ..observation_v2 import PolicyObservationV2
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from simulator.observation_v2 import PolicyObservationV2
    return PolicyObservationV2


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise DomainRandomizationError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainRandomizationError(f"{name} must be a finite non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise DomainRandomizationError(f"{name} must be a finite non-negative number")
    return converted


@dataclass(frozen=True, slots=True)
class DomainRandomizationConfig:
    """Explicit bounds for the currently supported environment perturbations."""

    profile_id: str = "interface-randomization-v1"
    decision_interval_jitter_ticks: int = 0
    action_latency_max_steps: int = 0
    entity_observation_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise DomainRandomizationError("profile_id must be a non-empty string")
        _nonnegative_int(self.decision_interval_jitter_ticks, "decision_interval_jitter_ticks")
        _nonnegative_int(self.action_latency_max_steps, "action_latency_max_steps")
        _finite_nonnegative(self.entity_observation_noise_std, "entity_observation_noise_std")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "decision_interval_jitter_ticks": self.decision_interval_jitter_ticks,
            "action_latency_max_steps": self.action_latency_max_steps,
            "entity_observation_noise_std": self.entity_observation_noise_std,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DomainRandomizationConfig":
        if not isinstance(raw, Mapping):
            raise DomainRandomizationError("domain-randomization config must be an object")
        if raw.get("schema_version", 1) != 1:
            raise DomainRandomizationError(
                f"unsupported domain-randomization schema: {raw.get('schema_version')!r}"
            )
        return cls(
            profile_id=raw.get("profile_id", "interface-randomization-v1"),
            decision_interval_jitter_ticks=raw.get("decision_interval_jitter_ticks", 0),
            action_latency_max_steps=raw.get("action_latency_max_steps", 0),
            entity_observation_noise_std=raw.get("entity_observation_noise_std", 0.0),
        )


@dataclass(frozen=True, slots=True)
class SimulationVariant:
    """One reproducible sampled episode variant."""

    profile_id: str
    episode_index: int
    decision_interval_ticks: int
    action_latency_steps: int
    observation_seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "episode_index": self.episode_index,
            "decision_interval_ticks": self.decision_interval_ticks,
            "action_latency_steps": self.action_latency_steps,
            "observation_seed": self.observation_seed,
        }


class DomainVariantSampler:
    """Pure deterministic sampler for one environment's episode variants."""

    def __init__(
        self,
        config: DomainRandomizationConfig,
        *,
        base_decision_interval_ticks: int,
        seed: int = 0,
    ) -> None:
        if not isinstance(config, DomainRandomizationConfig):
            raise DomainRandomizationError("config must be a DomainRandomizationConfig")
        _nonnegative_int(base_decision_interval_ticks, "base_decision_interval_ticks")
        if base_decision_interval_ticks < 1:
            raise DomainRandomizationError("base_decision_interval_ticks must be positive")
        if type(seed) is not int:
            raise DomainRandomizationError("seed must be an integer")
        self.config = config
        self.base_decision_interval_ticks = base_decision_interval_ticks
        self.seed = seed

    def sample(self, episode_index: int) -> SimulationVariant:
        _nonnegative_int(episode_index, "episode_index")
        sample_seed = deterministic_seed(self.seed, self.config.profile_id, episode_index)
        jitter = self.config.decision_interval_jitter_ticks
        if jitter:
            interval = self.base_decision_interval_ticks + (sample_seed % (2 * jitter + 1)) - jitter
            interval = max(1, interval)
        else:
            interval = self.base_decision_interval_ticks
        latency = (
            deterministic_seed(sample_seed, "latency")
            % (self.config.action_latency_max_steps + 1)
            if self.config.action_latency_max_steps
            else 0
        )
        return SimulationVariant(
            profile_id=self.config.profile_id,
            episode_index=episode_index,
            decision_interval_ticks=int(interval),
            action_latency_steps=int(latency),
            observation_seed=deterministic_seed(sample_seed, "observation"),
        )


class DomainRandomizedEnv:
    """Apply one sampled interface variant around a :class:`SimulatorEnv`."""

    def __init__(
        self,
        environment: Any,
        config: DomainRandomizationConfig,
        *,
        seed: int = 0,
    ) -> None:
        if not isinstance(environment, _simulator_env_class()):
            raise DomainRandomizationError("environment must be a SimulatorEnv")
        self.environment = environment
        self.config = config
        self.sampler = DomainVariantSampler(
            config,
            base_decision_interval_ticks=environment.decision_interval_ticks,
            seed=seed,
        )
        self.variant: SimulationVariant | None = None
        self._episode_index = -1
        self._pending_actions: list[Any] = []
        self._observation_rng: np.random.Generator | None = None

    @property
    def state(self) -> Any:
        return self.environment.state

    @property
    def engine(self) -> Any:
        return self.environment.engine

    def reset(self, **kwargs: Any) -> Any:
        """Start a new episode and sample its variant deterministically."""

        self._episode_index += 1
        self.variant = self.sampler.sample(self._episode_index)
        self.environment.decision_interval_ticks = self.variant.decision_interval_ticks
        self._pending_actions = [None] * self.variant.action_latency_steps
        self._observation_rng = np.random.default_rng(self.variant.observation_seed)
        return self.environment.reset(**kwargs)

    def reset_v2(self, **kwargs: Any) -> Any:
        """Start an episode and return only the wrapped public V2 observations."""

        self._episode_index += 1
        self.variant = self.sampler.sample(self._episode_index)
        self.environment.decision_interval_ticks = self.variant.decision_interval_ticks
        self._pending_actions = [None] * self.variant.action_latency_steps
        self._observation_rng = np.random.default_rng(self.variant.observation_seed)
        reset_v2 = getattr(self.environment, "reset_v2", None)
        if not callable(reset_v2):
            self.environment.reset(**kwargs)
            return self.observe_v2()
        return tuple(
            self._noise_observation(observation)
            for observation in reset_v2(**kwargs)
        )

    def observe(self) -> Any:
        return self.environment.observe()

    def observe_v2(self) -> tuple["PolicyObservationV2", "PolicyObservationV2"]:
        if self.variant is None or self._observation_rng is None:
            raise RuntimeError("reset() must be called before observe_v2()")
        return tuple(
            self._noise_observation(observation)
            for observation in self.environment.observe_v2()
        )  # type: ignore[return-value]

    def step(self, actions: Any) -> "EnvStep":
        if self.variant is None:
            raise RuntimeError("reset() must be called before step()")
        if self._pending_actions:
            applied_actions = self._pending_actions.pop(0)
            self._pending_actions.append(actions)
        else:
            applied_actions = actions
        return self.environment.step(applied_actions)

    def step_v2(self, actions: Any) -> Any:
        """Apply latency and return the wrapped environment's public V2 step."""

        if self.variant is None:
            raise RuntimeError("reset() must be called before step_v2()")
        if self._pending_actions:
            applied_actions = self._pending_actions.pop(0)
            self._pending_actions.append(actions)
        else:
            applied_actions = actions
        return self.environment.step_v2(applied_actions)

    def _noise_observation(self, observation: Any) -> Any:
        observation_type = _policy_observation_v2_class()
        if not isinstance(observation, observation_type):
            raise DomainRandomizationError("environment returned a non-V2 observation")
        standard_deviation = self.config.entity_observation_noise_std
        if standard_deviation == 0.0:
            return observation
        if self._observation_rng is None:  # pragma: no cover - reset invariant
            raise RuntimeError("observation RNG is not initialized")
        tokens = observation.entity_tokens.copy()
        valid_indices = np.flatnonzero(observation.entity_mask)
        for feature_name in ("x", "y", "hp_fraction", "confidence"):
            index = _FEATURE_INDEX[feature_name]
            noise = self._observation_rng.normal(
                0.0,
                standard_deviation,
                size=int(valid_indices.shape[0]),
            )
            tokens[valid_indices, index] = np.clip(
                tokens[valid_indices, index] + noise,
                0.0,
                1.0,
            )
        return observation_type(
            board=observation.board,
            global_vector=observation.global_vector,
            entity_tokens=tokens,
            entity_mask=observation.entity_mask,
            legal_play=observation.legal_play,
            legal_wait=observation.legal_wait,
        )


__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizationError",
    "DomainRandomizedEnv",
    "DomainVariantSampler",
    "SimulationVariant",
]
