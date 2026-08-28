"""Small, reproducible PPO trainer for the simulator policy boundary.

The simulator deliberately does not depend on Gymnasium, PyTorch, or a GPU
runtime.  This module therefore provides a NumPy-only *smoke trainer*: it is a
real masked PPO update and checkpoint format, but its actor is a factorised
linear policy rather than a production CNN.  That makes it useful immediately
for finding simulator failures (illegal actions, non-determinism, terminal
handling, and obviously broken rewards) on a clean installation.  A future
neural actor can consume the same ``Transition``/checkpoint metadata contract.

Only the public observation/action boundary is used.  In particular, the
policy never reads authoritative entities or private opponent state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import exp
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from cr_bot.domain.game_state import Action as PolicyAction

from .actions import SimAction
from .engine import ENGINE_VERSION, BattleEngine, DeterministicCycleController
from .env import RewardConfig, SimulatorEnv, VectorSimulatorEnv
from .observation import (
    ACTION_MASK_SHAPE,
    BOARD_SHAPE,
    GLOBAL_VECTOR_SHAPE,
    PINNED_OBSERVATION_CONTRACT_HASH,
    PolicyObservationV1,
)
from .ruleset import FIXED_RULESET_ID, Ruleset, load_ruleset
from .training_profiles import (
    TrainingProfile,
    TrainingProfileError,
    validate_training_profile,
)


PLAY_ACTION_COUNT = int(np.prod(ACTION_MASK_SHAPE))
WAIT_INDEX = PLAY_ACTION_COUNT
ACTION_COUNT = WAIT_INDEX + 1
TRAINER_SCHEMA_VERSION = 1
DEFAULT_DECISION_INTERVAL_US = 250_000


class TrainingConfigurationError(ValueError):
    """Raised when a training run would silently violate a simulator contract."""


def time_aware_discount(decision_interval_us: int, time_constant_us: int) -> float:
    """Return the per-decision discount for a real-time discount constant."""

    if type(decision_interval_us) is not int or decision_interval_us <= 0:
        raise ValueError("decision_interval_us must be a positive integer")
    if type(time_constant_us) is not int or time_constant_us <= 0:
        raise ValueError("time_constant_us must be a positive integer")
    return float(exp(-float(decision_interval_us) / float(time_constant_us)))


def full_match_decisions(
    ruleset: Ruleset | str = FIXED_RULESET_ID,
    *,
    decision_interval_us: int = DEFAULT_DECISION_INTERVAL_US,
) -> int:
    """Return the decision budget covering regulation plus overtime.

    The result is rounded up so a final partial decision interval is not
    silently omitted. ``decision_interval_us`` follows the same cadence
    contract as :class:`SimulatorEnv` and must be aligned to a physics tick.
    """

    if isinstance(ruleset, str):
        ruleset = load_ruleset(ruleset)
    if type(decision_interval_us) is not int or decision_interval_us <= 0:
        raise ValueError("decision_interval_us must be a positive integer")
    if decision_interval_us % ruleset.tick_us:
        raise ValueError("decision_interval_us must be a multiple of ruleset tick_us")
    duration_us = ruleset.match.regulation_us + ruleset.match.overtime_us
    return max(1, (duration_us + decision_interval_us - 1) // decision_interval_us)


def action_index_to_policy_action(index: int) -> PolicyAction:
    """Decode the flattened local policy action into the existing domain type."""

    if type(index) is not int or not 0 <= index < ACTION_COUNT:
        raise ValueError(f"action index must be an integer in [0, {ACTION_COUNT})")
    if index == WAIT_INDEX:
        return PolicyAction(kind="Wait")
    cells_per_slot = ACTION_MASK_SHAPE[1] * ACTION_MASK_SHAPE[2]
    slot, remainder = divmod(index, cells_per_slot)
    row, column = divmod(remainder, ACTION_MASK_SHAPE[2])
    return PolicyAction(kind="Play", card_idx=slot, cell=(column, row))


def policy_action_to_index(action: PolicyAction) -> int:
    """Encode a policy action, rejecting unsupported ability/evo forms."""

    kind = action.kind.strip().casefold().replace("_", "-")
    if kind in {"wait", "noop", "no-op"}:
        return WAIT_INDEX
    if kind != "play" or action.card_idx is None or action.cell is None:
        raise ValueError("the smoke policy supports only Wait and Play actions")
    slot = action.card_idx
    column, row = action.cell
    if not (0 <= slot < ACTION_MASK_SHAPE[0]):
        raise ValueError("card slot is outside the policy vocabulary")
    if not (0 <= row < ACTION_MASK_SHAPE[1] and 0 <= column < ACTION_MASK_SHAPE[2]):
        raise ValueError("action cell is outside the policy grid")
    return slot * ACTION_MASK_SHAPE[1] * ACTION_MASK_SHAPE[2] + row * ACTION_MASK_SHAPE[2] + column


def _observation_arrays(observation: PolicyObservationV1) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Copy the public tensors needed by a transition."""

    return (
        np.asarray(observation.board, dtype=np.float32).copy(),
        np.asarray(observation.global_vector, dtype=np.float32).copy(),
        np.asarray(observation.legal_play, dtype=np.bool_).copy(),
        bool(observation.legal_wait),
    )


@dataclass(frozen=True, slots=True)
class ActionSample:
    """One policy decision and diagnostics for that decision."""

    action: PolicyAction
    action_index: int
    log_prob: float
    value: float
    entropy: float
    fallback: bool = False


@dataclass(slots=True)
class Transition:
    """PPO rollout record using only policy-visible state."""

    board: np.ndarray
    global_vector: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    action_index: int
    old_log_prob: float
    value: float
    reward: float
    next_value: float
    terminal: bool
    boundary: bool
    entropy: float
    lane: int = 0


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Configuration for a bounded, deterministic smoke-training run."""

    ruleset_id: str = FIXED_RULESET_ID
    num_envs: int = 8
    backend: Literal["reference", "process", "packed-process"] = "reference"
    workers: int | None = None
    decision_interval_us: int = DEFAULT_DECISION_INTERVAL_US
    rollout_steps: int = 128
    total_steps: int = 10_000
    # A match result is not meaningfully discounted by policy-tick count.
    # Callers can opt into a real-time horizon with
    # ``discount_time_constant_us``.
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 2
    gradient_clip: float = 0.5
    seed: int = 0
    opponent: Literal["scripted", "self-play"] = "scripted"
    checkpoint_out: Path = Path("outputs/simulator/training/ppo-smoke.npz")
    checkpoint_every: int = 10_000
    eval_every: int = 10_000
    eval_episodes: int = 8
    eval_max_decisions: int | None = None
    allow_provisional_smoke: bool = False
    discount_time_constant_us: int | None = None
    training_profile: TrainingProfile | None = None
    reward_version: Literal["terminal-outcome-v1", "tower-damage-crowns-v1"] = "terminal-outcome-v1"

    def __post_init__(self) -> None:
        if self.ruleset_id == "":
            raise TrainingConfigurationError("ruleset_id must not be empty")
        if self.training_profile is not None:
            if not isinstance(self.training_profile, TrainingProfile):
                raise TrainingConfigurationError("training_profile must be a TrainingProfile or None")
            if self.training_profile.ruleset_id != self.ruleset_id:
                raise TrainingConfigurationError(
                    "training_profile ruleset_id must match PPOConfig ruleset_id"
                )
        for name in ("num_envs", "rollout_steps", "total_steps", "update_epochs", "eval_episodes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise TrainingConfigurationError(f"{name} must be a positive integer")
        if self.backend not in {"reference", "process", "packed-process"}:
            raise TrainingConfigurationError("unsupported vector backend")
        if self.opponent not in {"scripted", "self-play"}:
            raise TrainingConfigurationError("opponent must be scripted or self-play")
        if self.reward_version not in {"terminal-outcome-v1", "tower-damage-crowns-v1"}:
            raise TrainingConfigurationError("unsupported reward_version")
        if type(self.decision_interval_us) is not int or self.decision_interval_us <= 0:
            raise TrainingConfigurationError("decision_interval_us must be a positive integer")
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise TrainingConfigurationError(f"{name} must be in (0, 1]")
        if self.discount_time_constant_us is not None and (
            type(self.discount_time_constant_us) is not int or self.discount_time_constant_us <= 0
        ):
            raise TrainingConfigurationError("discount_time_constant_us must be a positive integer when provided")
        for name in ("clip_epsilon", "learning_rate", "value_coef", "entropy_coef", "gradient_clip"):
            if float(getattr(self, name)) < 0.0:
                raise TrainingConfigurationError(f"{name} must be non-negative")
        if self.checkpoint_every <= 0 or self.eval_every <= 0:
            raise TrainingConfigurationError("checkpoint/evaluation intervals must be positive")
        if self.eval_max_decisions is not None and (
            type(self.eval_max_decisions) is not int or self.eval_max_decisions <= 0
        ):
            raise TrainingConfigurationError(
                "eval_max_decisions must be a positive integer or None for a full-match evaluation"
            )

    @property
    def effective_gamma(self) -> float:
        """Return the configured per-decision discount without changing PPO."""

        if self.discount_time_constant_us is None:
            return float(self.gamma)
        return time_aware_discount(self.decision_interval_us, self.discount_time_constant_us)


class FactorizedPolicy:
    """A compact masked actor/critic with the exact simulator action grid."""

    def __init__(self, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = np.float32(0.01)
        self.cell_weights = rng.normal(0.0, scale, (ACTION_MASK_SHAPE[0], BOARD_SHAPE[0])).astype(np.float32)
        self.global_weights = rng.normal(0.0, scale, (ACTION_MASK_SHAPE[0], GLOBAL_VECTOR_SHAPE[0])).astype(np.float32)
        self.slot_bias = np.zeros((ACTION_MASK_SHAPE[0],), dtype=np.float32)
        self.position_bias = np.zeros(ACTION_MASK_SHAPE, dtype=np.float32)
        self.wait_cell_weights = rng.normal(0.0, scale, (BOARD_SHAPE[0],)).astype(np.float32)
        self.wait_global_weights = rng.normal(0.0, scale, (GLOBAL_VECTOR_SHAPE[0],)).astype(np.float32)
        self.wait_bias = np.float32(0.0)
        self.value_cell_weights = rng.normal(0.0, scale, (BOARD_SHAPE[0],)).astype(np.float32)
        self.value_global_weights = rng.normal(0.0, scale, (GLOBAL_VECTOR_SHAPE[0],)).astype(np.float32)
        self.value_bias = np.float32(0.0)

    @property
    def parameters(self) -> tuple[np.ndarray, ...]:
        return (
            self.cell_weights,
            self.global_weights,
            self.slot_bias,
            self.position_bias,
            self.wait_cell_weights,
            self.wait_global_weights,
            self.value_cell_weights,
            self.value_global_weights,
        )

    def _validate_observation(self, observation: PolicyObservationV1) -> None:
        if observation.board.shape != BOARD_SHAPE or observation.global_vector.shape != GLOBAL_VECTOR_SHAPE:
            raise ValueError("observation does not match the pinned policy contract")

    def raw_logits_value(self, observation: PolicyObservationV1) -> tuple[np.ndarray, float]:
        self._validate_observation(observation)
        board = np.asarray(observation.board, dtype=np.float32)
        global_vector = np.asarray(observation.global_vector, dtype=np.float32)
        board_part = np.tensordot(self.cell_weights, board, axes=(1, 0))
        logits = board_part + self.slot_bias[:, None, None] + self.position_bias
        logits += np.einsum("sg,g->s", self.global_weights, global_vector)[:, None, None]
        board_mean = board.mean(axis=(1, 2))
        wait_logit = float(np.dot(self.wait_cell_weights, board_mean) + np.dot(self.wait_global_weights, global_vector) + self.wait_bias)
        value = float(np.dot(self.value_cell_weights, board_mean) + np.dot(self.value_global_weights, global_vector) + self.value_bias)
        return np.concatenate((logits.reshape(-1), np.asarray([wait_logit], dtype=np.float32))), value

    @staticmethod
    def _masked_distribution(
        logits: np.ndarray,
        legal_play: np.ndarray,
        legal_wait: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        mask = np.zeros((ACTION_COUNT,), dtype=np.bool_)
        mask[:PLAY_ACTION_COUNT] = np.asarray(legal_play, dtype=np.bool_).reshape(-1)
        mask[WAIT_INDEX] = bool(legal_wait)
        fallback = not bool(mask.any())
        if fallback:
            # The engine currently always permits waiting.  Keeping this
            # defensive path makes malformed observations visible in metrics
            # instead of producing NaNs or an arbitrary illegal play.
            mask[WAIT_INDEX] = True
        valid = np.flatnonzero(mask)
        selected_logits = np.asarray(logits[valid], dtype=np.float64)
        selected_logits -= np.max(selected_logits)
        weights = np.exp(selected_logits)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            probabilities = np.full((len(valid),), 1.0 / len(valid), dtype=np.float64)
        else:
            probabilities = weights / total
        return valid, probabilities, fallback

    def sample(
        self,
        observation: PolicyObservationV1,
        rng: np.random.Generator,
        *,
        deterministic: bool = False,
    ) -> ActionSample:
        logits, value = self.raw_logits_value(observation)
        valid, probabilities, fallback = self._masked_distribution(
            logits, observation.legal_play, observation.legal_wait
        )
        local_index = int(np.argmax(probabilities)) if deterministic else int(rng.choice(len(valid), p=probabilities))
        index = int(valid[local_index])
        probability = max(float(probabilities[local_index]), np.finfo(np.float64).tiny)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))))
        return ActionSample(
            action=action_index_to_policy_action(index),
            action_index=index,
            log_prob=float(np.log(probability)),
            value=value,
            entropy=entropy,
            fallback=fallback,
        )

    def log_prob_value(
        self,
        observation: PolicyObservationV1,
        action_index: int,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        logits, value = self.raw_logits_value(observation)
        valid, probabilities, _ = self._masked_distribution(
            logits, observation.legal_play, observation.legal_wait
        )
        matches = np.flatnonzero(valid == action_index)
        if not len(matches):
            raise ValueError("rollout action is not legal under its stored action mask")
        local = int(matches[0])
        probability = max(float(probabilities[local]), np.finfo(np.float64).tiny)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))))
        return float(np.log(probability)), float(value), entropy, valid, probabilities

    def apply_update(self, gradients: dict[str, np.ndarray], learning_rate: float, gradient_clip: float) -> float:
        norm = float(np.sqrt(sum(float(np.sum(np.square(gradient))) for gradient in gradients.values())))
        scale = 1.0 if gradient_clip <= 0.0 or norm <= gradient_clip else gradient_clip / max(norm, 1e-12)
        for name, gradient in gradients.items():
            delta = np.asarray(gradient, dtype=np.float32) * np.float32(learning_rate * scale)
            if name == "wait_bias":
                self.wait_bias = np.float32(self.wait_bias + float(delta))
            elif name == "value_bias":
                self.value_bias = np.float32(self.value_bias + float(delta))
            else:
                getattr(self, name)[...] += delta
        return norm

    def metadata(self) -> dict[str, object]:
        return {
            "trainer_schema_version": TRAINER_SCHEMA_VERSION,
            "policy_type": "factorized-linear-v1",
            "board_shape": list(BOARD_SHAPE),
            "global_vector_shape": list(GLOBAL_VECTOR_SHAPE),
            "action_count": ACTION_COUNT,
            "wait_index": WAIT_INDEX,
        }

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> Path:
        target = Path(path)
        if target.suffix != ".npz":
            target = target.with_suffix(target.suffix + ".npz" if target.suffix else ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: np.asarray(getattr(self, name)) for name in (
            "cell_weights", "global_weights", "slot_bias", "position_bias",
            "wait_cell_weights", "wait_global_weights", "wait_bias",
            "value_cell_weights", "value_global_weights", "value_bias",
        )}
        combined = dict(self.metadata())
        if metadata:
            combined.update(metadata)
        payload["metadata_json"] = np.asarray(json.dumps(combined, sort_keys=True, allow_nan=False))
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: str | Path, *, expected_metadata: dict[str, object] | None = None) -> tuple["FactorizedPolicy", dict[str, object]]:
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("policy_type") != "factorized-linear-v1":
                raise TrainingConfigurationError("checkpoint is not a factorized policy")
            if expected_metadata:
                for key, expected in expected_metadata.items():
                    if metadata.get(key) != expected:
                        raise TrainingConfigurationError(
                            f"checkpoint contract mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
                        )
            policy = cls(seed=0)
            for name in (
                "cell_weights", "global_weights", "slot_bias", "position_bias",
                "wait_cell_weights", "wait_global_weights", "wait_bias",
                "value_cell_weights", "value_global_weights", "value_bias",
            ):
                if name not in archive:
                    raise TrainingConfigurationError(f"checkpoint is missing policy parameter {name}")
                value = np.asarray(archive[name])
                destination = getattr(policy, name)
                if destination.shape != value.shape:
                    raise TrainingConfigurationError(f"checkpoint shape mismatch for {name}")
                if name == "wait_bias":
                    policy.wait_bias = np.float32(value)
                elif name == "value_bias":
                    policy.value_bias = np.float32(value)
                else:
                    destination[...] = value
            return policy, metadata


def _empty_gradients(policy: FactorizedPolicy) -> dict[str, np.ndarray]:
    return {name: np.zeros_like(getattr(policy, name), dtype=np.float64) for name in (
        "cell_weights", "global_weights", "slot_bias", "position_bias",
        "wait_cell_weights", "wait_global_weights", "wait_bias",
        "value_cell_weights", "value_global_weights", "value_bias",
    )}


def _gae(transitions: Sequence[Transition], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros((len(transitions),), dtype=np.float64)
    returns = np.zeros((len(transitions),), dtype=np.float64)
    running_by_lane: dict[int, float] = {}
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        delta = transition.reward + gamma * transition.next_value - transition.value
        previous = running_by_lane.get(transition.lane, 0.0)
        running = delta + gamma * gae_lambda * (0.0 if transition.boundary else previous)
        running_by_lane[transition.lane] = running
        advantages[index] = running
        returns[index] = running + transition.value
    if len(advantages):
        mean = float(advantages.mean())
        std = float(advantages.std())
        advantages = (advantages - mean) / max(std, 1e-8)
    return advantages, returns


def _ppo_update(
    policy: FactorizedPolicy,
    transitions: Sequence[Transition],
    advantages: np.ndarray,
    returns: np.ndarray,
    config: PPOConfig,
) -> dict[str, float]:
    if not transitions:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0, "gradient_norm": 0.0, "mean_reward": 0.0}
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clipped: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(config.update_epochs):
        gradients = _empty_gradients(policy)
        for index, transition in enumerate(transitions):
            new_log_prob, value, entropy, valid, probabilities = policy.log_prob_value(
                PolicyObservationV1(
                    transition.board, transition.global_vector,
                    np.zeros_like(transition.legal_play), transition.legal_play,
                    transition.legal_wait,
                ),
                transition.action_index,
            )
            advantage = float(advantages[index])
            ratio = float(np.exp(np.clip(new_log_prob - transition.old_log_prob, -20.0, 20.0)))
            unclipped = ratio * advantage
            clipped_ratio = float(np.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon))
            clipped_objective = clipped_ratio * advantage
            objective = min(unclipped, clipped_objective)
            active = (advantage >= 0.0 and ratio <= 1.0 + config.clip_epsilon) or (
                advantage < 0.0 and ratio >= 1.0 - config.clip_epsilon
            )
            coefficient = advantage * ratio if active else 0.0
            logits_gradient = np.zeros((ACTION_COUNT,), dtype=np.float64)
            logits_gradient[valid] = -coefficient * probabilities
            logits_gradient[transition.action_index] += coefficient
            if config.entropy_coef:
                entropy_gradient = -probabilities * (
                    np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)) + entropy
                )
                logits_gradient[valid] += config.entropy_coef * entropy_gradient
            board = transition.board
            global_vector = transition.global_vector
            if transition.action_index == WAIT_INDEX:
                board_mean = board.mean(axis=(1, 2))
                for valid_index, signal in zip(valid, logits_gradient[valid], strict=True):
                    if valid_index == WAIT_INDEX:
                        gradients["wait_cell_weights"] += signal * board_mean
                        gradients["wait_global_weights"] += signal * global_vector
                        gradients["wait_bias"] += signal
                        continue
                    cells_per_slot = ACTION_MASK_SHAPE[1] * ACTION_MASK_SHAPE[2]
                    slot2, remainder2 = divmod(int(valid_index), cells_per_slot)
                    row2, column2 = divmod(remainder2, ACTION_MASK_SHAPE[2])
                    gradients["cell_weights"][slot2] += signal * board[:, row2, column2]
                    gradients["global_weights"][slot2] += signal * global_vector
                    gradients["slot_bias"][slot2] += signal
                    gradients["position_bias"][slot2, row2, column2] += signal
            else:
                cells_per_slot = ACTION_MASK_SHAPE[1] * ACTION_MASK_SHAPE[2]
                slot, remainder = divmod(transition.action_index, cells_per_slot)
                row, column = divmod(remainder, ACTION_MASK_SHAPE[2])
                signal = logits_gradient[transition.action_index]
                gradients["cell_weights"][slot] += signal * board[:, row, column]
                gradients["global_weights"][slot] += signal * global_vector
                gradients["slot_bias"][slot] += signal
                gradients["position_bias"][slot, row, column] += signal
                # The softmax denominator contributes gradients for every
                # valid action, not only the sampled one.
                for valid_index, signal in zip(valid, logits_gradient[valid], strict=True):
                    if valid_index == transition.action_index or signal == 0.0:
                        continue
                    if valid_index == WAIT_INDEX:
                        board_mean = board.mean(axis=(1, 2))
                        gradients["wait_cell_weights"] += signal * board_mean
                        gradients["wait_global_weights"] += signal * global_vector
                        gradients["wait_bias"] += signal
                    else:
                        slot2, remainder2 = divmod(int(valid_index), cells_per_slot)
                        row2, column2 = divmod(remainder2, ACTION_MASK_SHAPE[2])
                        gradients["cell_weights"][slot2] += signal * board[:, row2, column2]
                        gradients["global_weights"][slot2] += signal * global_vector
                        gradients["slot_bias"][slot2] += signal
                        gradients["position_bias"][slot2, row2, column2] += signal
            value_error = value - float(returns[index])
            board_mean = board.mean(axis=(1, 2))
            # Maximize ``-value_coef * 1/2 (V-return)^2``.
            gradients["value_cell_weights"] += -config.value_coef * value_error * board_mean
            gradients["value_global_weights"] += -config.value_coef * value_error * global_vector
            gradients["value_bias"] += -config.value_coef * value_error
            policy_losses.append(-objective)
            value_losses.append(0.5 * value_error * value_error)
            entropies.append(entropy)
            kls.append(transition.old_log_prob - new_log_prob)
            clipped.append(float(not active))
        for gradient in gradients.values():
            gradient /= float(len(transitions))
        gradient_norms.append(policy.apply_update(gradients, config.learning_rate, config.gradient_clip))
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(kls)),
        "clip_fraction": float(np.mean(clipped)),
        "gradient_norm": float(np.mean(gradient_norms)),
        "mean_reward": float(np.mean([transition.reward for transition in transitions])),
    }


def _new_environment(ruleset: Ruleset, config: PPOConfig) -> SimulatorEnv:
    reward = (
        RewardConfig.terminal_outcome()
        if config.reward_version == "terminal-outcome-v1"
        else RewardConfig()
    )
    return SimulatorEnv(
        engine=BattleEngine(ruleset),
        decision_interval_us=config.decision_interval_us,
        reward=reward,
    )


def _new_vector_environment(ruleset: Ruleset, config: PPOConfig) -> VectorSimulatorEnv:
    environments = tuple(_new_environment(ruleset, config) for _ in range(config.num_envs))
    return VectorSimulatorEnv(environments, backend=config.backend, workers=config.workers)


def _episode_seed(base_seed: int, lane: int, episode: int) -> int:
    return int(base_seed + lane * 1_000_003 + episode * 9_999_991)


def _contract_metadata(ruleset: Ruleset, config: PPOConfig) -> dict[str, object]:
    return {
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.content_hash,
        "engine_version": ENGINE_VERSION,
        "observation_contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
        "reward_version": config.reward_version,
        "gamma": config.effective_gamma,
        "gae_lambda": config.gae_lambda,
        "discount_time_constant_us": config.discount_time_constant_us,
        "opponent": config.opponent,
        "seed": config.seed,
        "allow_provisional_smoke": config.allow_provisional_smoke,
        "training_profile_id": (
            config.training_profile.profile_id if config.training_profile is not None else None
        ),
        "training_profile_purpose": (
            config.training_profile.purpose if config.training_profile is not None else None
        ),
    }


def evaluate_policy(
    policy: FactorizedPolicy,
    *,
    ruleset: Ruleset | str = FIXED_RULESET_ID,
    opponent: Literal["scripted", "self-play"] = "scripted",
    episodes: int = 8,
    seed_start: int = 0,
    max_decisions: int | None = None,
    decision_interval_us: int = DEFAULT_DECISION_INTERVAL_US,
    reward_version: Literal["terminal-outcome-v1", "tower-damage-crowns-v1"] = "terminal-outcome-v1",
) -> dict[str, object]:
    """Run deterministic held-out matches and return JSON-safe metrics.

    ``max_decisions=None`` evaluates through the ruleset's full regulation
    plus overtime horizon. A smaller positive value is an explicit
    short-horizon smoke evaluation; episodes that reach that bound without a
    terminal match result are censored and excluded from outcome rates.
    """

    if isinstance(ruleset, str):
        ruleset = load_ruleset(ruleset)
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if max_decisions is not None and (type(max_decisions) is not int or max_decisions <= 0):
        raise ValueError("max_decisions must be a positive integer or None for a full-match evaluation")
    effective_max_decisions = (
        full_match_decisions(ruleset, decision_interval_us=decision_interval_us)
        if max_decisions is None
        else max_decisions
    )
    rng = np.random.default_rng(seed_start + 17)
    returns: list[float] = []
    lengths: list[int] = []
    wins = losses = draws = 0
    completed_episodes = truncated_episodes = 0
    fallback_actions = 0
    for episode in range(episodes):
        env = _new_environment(
            ruleset,
            PPOConfig(
                ruleset_id=ruleset.ruleset_id,
                num_envs=1,
                decision_interval_us=decision_interval_us,
                total_steps=1,
                rollout_steps=1,
                checkpoint_every=1,
                eval_every=1,
                eval_episodes=1,
                allow_provisional_smoke=True,
                reward_version=reward_version,
            ),
        )
        observations = env.reset(seed=seed_start + episode, shuffle_decks=True)
        total_return = 0.0
        episode_completed = False
        episode_truncated = False
        winner: int | None = None
        for decision in range(effective_max_decisions):
            own = policy.sample(observations[0], rng)
            fallback_actions += int(own.fallback)
            if opponent == "self-play":
                other = policy.sample(observations[1], rng)
                other_action: PolicyAction | SimAction = other.action
            else:
                state = env.state
                if state is None:
                    raise RuntimeError("evaluation environment lost its state")
                other_action = DeterministicCycleController().choose_action(env.engine, state, 1)
            result = env.step((own.action, other_action))
            total_return += result.rewards[0]
            if result.terminated or result.truncated:
                episode_completed = bool(result.terminated and not result.truncated)
                episode_truncated = not episode_completed
                winner = result.info.get("winner") if isinstance(result.info.get("winner"), int) else None
                lengths.append(decision + 1)
                break
            observations = result.observations
        if not episode_completed and not episode_truncated:
            # The evaluator's own horizon is a censoring boundary, not a
            # simulator outcome. Do not turn it into a draw.
            episode_truncated = True
            lengths.append(effective_max_decisions)
        returns.append(total_return)
        if episode_completed:
            completed_episodes += 1
            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
        else:
            truncated_episodes += 1
    rated_episodes = completed_episodes
    return {
        "episodes": episodes,
        "completed": completed_episodes,
        "truncated": truncated_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / rated_episodes if rated_episodes else 0.0,
        "loss_rate": losses / rated_episodes if rated_episodes else 0.0,
        "draw_rate": draws / rated_episodes if rated_episodes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_decisions": float(np.mean(lengths)) if lengths else 0.0,
        "fallback_actions": fallback_actions,
        "opponent": opponent,
        "seed_start": seed_start,
        "max_decisions": effective_max_decisions,
    }


class PPOTrainer:
    """Collect vector rollouts, apply masked PPO, and write auditable checkpoints."""

    def __init__(self, config: PPOConfig = PPOConfig()) -> None:
        self.config = config
        self.ruleset = load_ruleset(config.ruleset_id)
        self.training_profile_result: dict[str, object] | None = None
        if config.training_profile is not None:
            try:
                self.training_profile_result = validate_training_profile(
                    config.training_profile,
                    ruleset=self.ruleset,
                )
            except TrainingProfileError as error:
                raise TrainingConfigurationError(str(error)) from error
        ruleset_ready = bool(self.ruleset.metadata.get("training_ready", False))
        profile_ready = bool(
            self.training_profile_result is not None
            and self.training_profile_result.get("training_ready") is True
        )
        explicitly_provisional = bool(
            config.allow_provisional_smoke
            or (
                config.training_profile is not None
                and config.training_profile.purpose == "smoke"
            )
        )
        if not ruleset_ready and not profile_ready and not explicitly_provisional:
            raise TrainingConfigurationError(
                f"ruleset {self.ruleset.ruleset_id!r} is not training-ready; "
                "provide a ready scoped training profile, or pass allow_provisional_smoke=True "
                "only for bounded simulator smoke tests"
            )
        self.policy = FactorizedPolicy(seed=config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.total_steps = 0
        self.updates = 0
        self.episodes = 0
        self.fallback_actions = 0
        self.illegal_action_attempts = 0
        self._episode_counts = [0 for _ in range(config.num_envs)]
        self.started = perf_counter()

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = _contract_metadata(self.ruleset, self.config)
        metadata.update({"config": _jsonable(asdict(self.config)), "total_steps": self.total_steps, "updates": self.updates})
        return metadata

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        return self.policy.save(path or self.config.checkpoint_out, metadata=self.checkpoint_metadata())

    def load_checkpoint(self, path: str | Path) -> None:
        expected = _contract_metadata(self.ruleset, self.config)
        policy, metadata = FactorizedPolicy.load(path, expected_metadata=expected)
        self.policy = policy
        self.total_steps = int(metadata.get("total_steps", 0))
        self.updates = int(metadata.get("updates", 0))

    def _collect_rollout(
        self,
        vector: VectorSimulatorEnv,
        observations: list[tuple[PolicyObservationV1, PolicyObservationV1]],
        *,
        transition_budget: int | None = None,
    ) -> list[Transition]:
        """Collect at most `transition_budget` learner transitions.

        A vector step advances every lane, so the final simulator tick can
        produce more lane results than the requested training budget.  Those
        extra results still get projected/reset locally so the vector remains
        internally consistent, but they are deliberately excluded from the
        rollout and counters.  This keeps `report["total_steps"]` equal to
        the user-requested policy-transition budget instead of silently
        overshooting it by up to `num_envs - 1` transitions.
        """
        if transition_budget is not None:
            if type(transition_budget) is not int or transition_budget < 0:
                raise ValueError("transition_budget must be a non-negative integer")
            if transition_budget == 0:
                return []
        transitions: list[Transition] = []
        controllers = [DeterministicCycleController() for _ in vector.environments]
        for _ in range(self.config.rollout_steps):
            rows: list[tuple[PolicyAction | SimAction, PolicyAction | SimAction]] = []
            samples: list[ActionSample] = []
            for lane, lane_observations in enumerate(observations):
                own = self.policy.sample(lane_observations[0], self.rng)
                samples.append(own)
                if self.config.opponent == "self-play":
                    other: PolicyAction | SimAction = self.policy.sample(lane_observations[1], self.rng).action
                else:
                    state = vector.environments[lane].state
                    if state is None:
                        raise RuntimeError("vector lane has not been reset")
                    other = controllers[lane].choose_action(vector.environments[lane].engine, state, 1)
                rows.append((own.action, other))
            results = vector.step(rows)
            next_observations: list[tuple[PolicyObservationV1, PolicyObservationV1]] = []
            for lane, (before, sample, result) in enumerate(zip(observations, samples, results, strict=True)):
                include = transition_budget is None or len(transitions) < transition_budget
                # A runner tick-limit is a truncation, not a game loss.  Keep
                # its critic bootstrap while still cutting the GAE chain at
                # the episode boundary.
                terminal = bool(result.terminated and not result.truncated)
                boundary = bool(result.terminated or result.truncated)
                next_value = 0.0 if terminal else self.policy.raw_logits_value(result.observations[0])[1]
                if include:
                    self.fallback_actions += int(sample.fallback)
                    if sample.action_index != WAIT_INDEX and not bool(
                        before[0].legal_play.reshape(-1)[sample.action_index]
                    ):
                        self.illegal_action_attempts += 1
                    board, global_vector, legal_play, legal_wait = _observation_arrays(before[0])
                    transitions.append(Transition(
                        board=board,
                        global_vector=global_vector,
                        legal_play=legal_play,
                        legal_wait=legal_wait,
                        action_index=sample.action_index,
                        old_log_prob=sample.log_prob,
                        value=sample.value,
                        reward=float(result.rewards[0]),
                        next_value=float(next_value),
                        terminal=terminal,
                        boundary=boundary,
                        entropy=sample.entropy,
                        lane=lane,
                    ))
                    self.total_steps += 1
                if boundary and include:
                    self.episodes += 1
                    self._episode_counts[lane] += 1
                    next_observations.append(vector.environments[lane].reset(
                        seed=_episode_seed(self.config.seed, lane, self._episode_counts[lane]),
                        shuffle_decks=True,
                    ))
                else:
                    next_observations.append(result.observations)
            observations[:] = next_observations
            if transition_budget is not None and len(transitions) >= transition_budget:
                break
        return transitions

    def train(self) -> dict[str, object]:
        self._episode_counts = [0 for _ in range(self.config.num_envs)]
        vector = _new_vector_environment(self.ruleset, self.config)
        observations = list(vector.reset(tuple(self.config.seed + lane for lane in range(self.config.num_envs))))
        update_metrics: list[dict[str, object]] = []
        next_checkpoint = self.config.checkpoint_every
        next_eval = self.config.eval_every
        try:
            while self.total_steps < self.config.total_steps:
                transitions = self._collect_rollout(
                    vector,
                    observations,
                    transition_budget=self.config.total_steps - self.total_steps,
                )
                advantages, returns = _gae(transitions, self.config.effective_gamma, self.config.gae_lambda)
                metrics = _ppo_update(self.policy, transitions, advantages, returns, self.config)
                self.updates += 1
                metrics.update({"update": self.updates, "total_steps": self.total_steps, "episodes": self.episodes})
                if self.total_steps >= next_checkpoint:
                    self.save_checkpoint()
                    next_checkpoint += self.config.checkpoint_every
                if self.total_steps >= next_eval:
                    metrics["evaluation"] = evaluate_policy(
                        self.policy,
                        ruleset=self.ruleset,
                        opponent=self.config.opponent,
                        episodes=self.config.eval_episodes,
                        seed_start=self.config.seed + 10_000 + self.total_steps,
                        max_decisions=self.config.eval_max_decisions,
                        decision_interval_us=self.config.decision_interval_us,
                        reward_version=self.config.reward_version,
                    )
                    next_eval += self.config.eval_every
                update_metrics.append(metrics)
        finally:
            vector.close()
        checkpoint = self.save_checkpoint()
        elapsed = perf_counter() - self.started
        rewards = [float(row.get("mean_reward", 0.0)) for row in update_metrics]
        return {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "kind": "simulator_ppo_smoke_training",
            "trainer": self.policy.metadata(),
            "ruleset_id": self.ruleset.ruleset_id,
            "ruleset_hash": self.ruleset.content_hash,
            "engine_version": ENGINE_VERSION,
            "observation_contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
            "reward_version": self.config.reward_version,
            "opponent": self.config.opponent,
            "seed": self.config.seed,
            "requested_steps": self.config.total_steps,
            "total_steps": self.total_steps,
            "updates": self.updates,
            "episodes": self.episodes,
            "fallback_actions": self.fallback_actions,
            "illegal_action_attempts": self.illegal_action_attempts,
            "wall_seconds": elapsed,
            "environment_steps_per_second": self.total_steps / elapsed if elapsed else 0.0,
            "mean_update_reward": float(np.mean(rewards)) if rewards else 0.0,
            "checkpoint": str(checkpoint),
            "updates_detail": update_metrics,
            "provisional_smoke": bool(
                self.config.allow_provisional_smoke
                or (
                    self.config.training_profile is not None
                    and self.config.training_profile.purpose == "smoke"
                )
            ),
            "training_profile": self.training_profile_result,
        }


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def train_ppo(config: PPOConfig = PPOConfig()) -> dict[str, object]:
    """Convenience entry point used by scripts and callers."""

    return PPOTrainer(config).train()


__all__ = [
    "ACTION_COUNT",
    "ActionSample",
    "DEFAULT_DECISION_INTERVAL_US",
    "FactorizedPolicy",
    "PPOConfig",
    "PPOTrainer",
    "PLAY_ACTION_COUNT",
    "Transition",
    "TRAINER_SCHEMA_VERSION",
    "TrainingConfigurationError",
    "WAIT_INDEX",
    "action_index_to_policy_action",
    "evaluate_policy",
    "full_match_decisions",
    "policy_action_to_index",
    "time_aware_discount",
    "train_ppo",
]
