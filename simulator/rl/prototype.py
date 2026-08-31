"""Runnable first recurrent-PPO prototype for the deterministic simulator.

This module is the narrow integration layer between the public V2 observation
adapter, the recurrent policy/learner, and :class:`SimulatorEnv`.  It is
deliberately a prototype rather than a claim of strong Clash Royale play:

* the learner controls one fixed Hog-cycle deck;
* the opponent is the existing deterministic cycle controller;
* rewards are the sparse terminal win/draw/loss objective;
* the actor receives only ``SimulatorEnv.observe_v2()``;
* an optional asymmetric critic and opponent-belief targets receive exact
  simulator state only through explicitly training-only callbacks;
* checkpoints contain the observation contract and fail closed if the actor
  contract is not public-only.

Run it from this directory with the requested PyTorch environment, for
example::

    PYTHONPATH=/usr/lib/python3.14/site-packages:.:..:../src \\
      /home/keschler/Documents/Coding/python/cr-bot/outputs/venv/bin/python \\
      -m rl.prototype train --allow-provisional --updates 1 --envs 2 --horizon 8

The ``--allow-provisional`` switch is intentional.  The pinned simulator is
executable but its ruleset is not yet fidelity-ready; a prototype run must
record that fact instead of silently presenting simulator results as live-game
results.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import argparse
import json
import multiprocessing
from math import ceil, isfinite
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence, TextIO

from ._compat import TORCH_AVAILABLE, TorchUnavailableError
from .provenance import code_revision, revision_changed


PROTOTYPE_SCHEMA_VERSION = 1
PROTOTYPE_CHECKPOINT_FORMAT = "recurrent-public-ppo-prototype-v1"
PRIVILEGED_FEATURE_DIM = 23
_EVALUATION_POLICIES = frozenset(
    {"actor", "public-counter", "strategic-counter", "deterministic-counter"}
)


class PrototypeConfigurationError(ValueError):
    """Raised when a prototype would violate a simulator or policy contract."""


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise PrototypeConfigurationError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise PrototypeConfigurationError(f"{name} must be a non-negative integer")


def _probability(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrototypeConfigurationError(f"{name} must be a finite value in [0, 1]")
    if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise PrototypeConfigurationError(f"{name} must be a finite value in [0, 1]")


def _is_cpu_device_request(device: str | None) -> bool:
    """Return whether a caller explicitly requested CPU inference."""

    if not isinstance(device, str):
        return False
    normalized = device.strip().lower()
    return normalized == "cpu" or normalized.startswith("cpu:")


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Small, serializable settings for one recurrent-PPO prototype run."""

    ruleset_id: str = "v1"
    envs: int = 2
    horizon: int = 128
    updates: int = 1
    decision_interval_us: int = 250_000
    seed: int = 0
    target_player: int = 0
    shuffle_decks: bool = True
    opponent: str = "deterministic-cycle"
    device: str = "auto"
    env_backend: str = "reference"
    env_workers: int | None = None
    # Optional one-update pipeline for the trainer-only rollout farm.  The
    # next batch is collected with the previous published weights while PPO
    # updates the current batch; this is explicit because it introduces a
    # bounded one-update behavior-policy lag.
    overlap_rollouts: bool = False
    # Compile the parent policy forward graph with torch.compile when the
    # runtime supports it.  This is an opt-in kernel/scheduling optimization;
    # it does not alter the model topology or actor observation contract.
    compile_policy: bool = False
    # Full training traces are opt-in because they serialize authoritative
    # snapshots and model alternatives at every decision.
    diagnostic_trace_out: str | None = None

    # Optimizer and long-horizon objective.
    learning_rate: float = 3e-4
    update_epochs: int = 2
    sequence_minibatch_size: int = 2
    # Optional temporal chunking for PPO. ``None`` keeps the complete rollout
    # row as one recurrent sequence; production runs should use a chunk that
    # divides ``horizon`` so many independent sequence rows reach the learner.
    sequence_length: int | None = None
    gamma: float = 1.0
    gae_lambda: float = 0.98
    entropy_coef: float = 0.01
    belief_coef: float = 0.05
    behavior_cloning_coef: float = 0.0
    behavior_cloning_factor_coef: float = 0.0
    # Relative card-factor weight inside the supervised factor loss. Keep the
    # default neutral; raise it only when diagnostics identify card-head
    # collapse while mode and placement remain healthy.
    behavior_cloning_card_factor_weight: float = 1.0
    imitation_only: bool = False
    # PPO must collect the actions selected by the actor.  Teacher execution is
    # an explicit, separately reported bootstrap mode and is never the safe
    # default for a new run.
    expert_execution_probability: float = 0.0
    # Suppress low-confidence teacher labels outside public threat states.
    # This is data curation only: it does not alter actor actions, masks, or
    # simulator behavior.
    expert_label_on_threat_only: bool = False
    # Restrict labels to states where the sampled actor decision differs from
    # the teacher decision; this is a sparse reference-preserving recovery
    # mode for a diagnosed policy regression.
    expert_label_on_disagreement: bool = False
    deterministic_rollouts: bool = False
    max_grad_norm: float = 0.5
    placement_max_grad_norm: float | None = None

    # The first model is intentionally small enough for a CPU smoke run while
    # retaining the production topology: public entity Transformer + GRU +
    # autoregressive action heads.
    model_dim: int = 32
    encoder_dim: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 1
    transformer_ff_dim: int = 64
    gru_hidden_dim: int = 32
    gru_layers: int = 1
    # Public hand cards stay as one-hot card-table features projected per slot.
    explicit_hand_features: bool = False
    direct_public_action_features: bool = False
    direct_public_card_features: bool = False
    primary_public_card_features: bool = False
    contextual_public_card_features: bool = False
    current_encoded_action_features: bool = False
    direct_public_mask_features: bool = False
    direct_public_context_features: bool = False
    direct_public_slot_card_features: bool = False
    # Strategic runs may preserve a board-aligned raster map for
    # card-conditioned placement; small smoke tests can disable it.
    spatial_placement_features: bool = False
    spatial_placement_dim: int = 32

    # Interface-only robustness perturbations.  Physics parameters are not
    # randomized here because they need evidence-backed ruleset variants.
    decision_interval_jitter_ticks: int = 0
    action_latency_max_steps: int = 0
    entity_observation_noise_std: float = 0.0

    # Safety gates and optional training-only targets.
    allow_provisional: bool = False
    use_privileged_critic: bool = True
    collect_belief_targets: bool = True
    dense_reward: bool = False
    # A bounded potential difference is optional credit assignment. It does
    # not encode a preferred card or tactical response; terminal outcome
    # remains the primary objective when this is non-zero.
    potential_reward_weight: float = 0.0
    # Optional trust-region safety gate for a complete PPO update. When set,
    # the trainer snapshots the learner before the update and restores it if
    # the observed mean PPO KL exceeds this bound. This is a training-time
    # rollback only; it never selects or masks an environment action.
    max_update_approx_kl: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ruleset_id, str) or not self.ruleset_id.strip():
            raise PrototypeConfigurationError("ruleset_id must be a non-empty string")
        for name in ("envs", "horizon", "updates", "decision_interval_us"):
            _positive_int(name, getattr(self, name))
        _nonnegative_int("seed", self.seed)
        if type(self.target_player) is not int or self.target_player not in (0, 1):
            raise PrototypeConfigurationError("target_player must be 0 or 1")
        for name in (
            "shuffle_decks",
            "allow_provisional",
            "use_privileged_critic",
            "collect_belief_targets",
            "dense_reward",
            "explicit_hand_features",
            "direct_public_action_features",
            "direct_public_card_features",
            "primary_public_card_features",
            "contextual_public_card_features",
            "current_encoded_action_features",
            "direct_public_mask_features",
            "direct_public_context_features",
            "direct_public_slot_card_features",
            "spatial_placement_features",
            "imitation_only",
            "expert_label_on_threat_only",
            "expert_label_on_disagreement",
            "deterministic_rollouts",
            "overlap_rollouts",
            "compile_policy",
        ):
            if type(getattr(self, name)) is not bool:
                raise PrototypeConfigurationError(f"{name} must be boolean")
        if self.diagnostic_trace_out is not None and (
            not isinstance(self.diagnostic_trace_out, (str, Path))
            or not str(self.diagnostic_trace_out).strip()
        ):
            raise PrototypeConfigurationError(
                "diagnostic_trace_out must be a non-empty path or None"
            )
        if self.opponent not in {"scripted", "deterministic-cycle"}:
            raise PrototypeConfigurationError(
                "opponent must be 'scripted' or 'deterministic-cycle'"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise PrototypeConfigurationError("device must be a non-empty string")
        if self.env_backend not in {
            "reference",
            "process",
            "packed-process",
            "persistent-process",
            "rollout-process",
        }:
            raise PrototypeConfigurationError(
                "env_backend must be 'reference', 'process', 'packed-process', "
                "'persistent-process', or 'rollout-process'"
            )
        if self.env_workers is not None:
            _positive_int("env_workers", self.env_workers)
        if not isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0.0:
            raise PrototypeConfigurationError("learning_rate must be positive")
        for name in ("update_epochs", "sequence_minibatch_size"):
            _positive_int(name, getattr(self, name))
        if self.sequence_length is not None:
            _positive_int("sequence_length", self.sequence_length)
            if self.horizon % self.sequence_length:
                raise PrototypeConfigurationError(
                    "sequence_length must divide horizon"
                )
        if not 0.0 < float(self.gamma) <= 1.0:
            raise PrototypeConfigurationError("gamma must be in (0, 1]")
        _probability("gae_lambda", self.gae_lambda)
        for name in (
            "entropy_coef",
            "belief_coef",
            "behavior_cloning_coef",
            "behavior_cloning_factor_coef",
            "behavior_cloning_card_factor_weight",
            "potential_reward_weight",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise PrototypeConfigurationError(f"{name} must be non-negative")
        if self.max_update_approx_kl is not None:
            if (
                isinstance(self.max_update_approx_kl, bool)
                or not isinstance(self.max_update_approx_kl, (int, float))
                or not isfinite(float(self.max_update_approx_kl))
                or float(self.max_update_approx_kl) <= 0.0
            ):
                raise PrototypeConfigurationError(
                    "max_update_approx_kl must be positive and finite or None"
                )
        if self.imitation_only and not (
            self.behavior_cloning_coef > 0.0
            or self.behavior_cloning_factor_coef > 0.0
        ):
            raise PrototypeConfigurationError(
                "imitation_only requires a positive behavior-cloning coefficient"
            )
        _probability(
            "expert_execution_probability",
            self.expert_execution_probability,
        )
        if not isfinite(float(self.max_grad_norm)) or float(self.max_grad_norm) < 0.0:
            raise PrototypeConfigurationError("max_grad_norm must be non-negative")
        if self.placement_max_grad_norm is not None and (
            not isfinite(float(self.placement_max_grad_norm))
            or float(self.placement_max_grad_norm) <= 0.0
        ):
            raise PrototypeConfigurationError(
                "placement_max_grad_norm must be positive or None"
            )
        for name in (
            "model_dim",
            "encoder_dim",
            "transformer_heads",
            "transformer_layers",
            "transformer_ff_dim",
            "gru_hidden_dim",
            "gru_layers",
            "spatial_placement_dim",
        ):
            _positive_int(name, getattr(self, name))
        if self.model_dim % self.transformer_heads:
            raise PrototypeConfigurationError(
                "model_dim must be divisible by transformer_heads"
            )
        if self.primary_public_card_features and not self.direct_public_card_features:
            raise PrototypeConfigurationError(
                "primary_public_card_features require direct_public_card_features"
            )
        if self.current_encoded_action_features and self.encoder_dim > self.gru_hidden_dim:
            raise PrototypeConfigurationError(
                "current encoded action features require encoder_dim <= gru_hidden_dim"
            )
        _nonnegative_int("decision_interval_jitter_ticks", self.decision_interval_jitter_ticks)
        _nonnegative_int("action_latency_max_steps", self.action_latency_max_steps)
        if (
            not isfinite(float(self.entity_observation_noise_std))
            or float(self.entity_observation_noise_std) < 0.0
        ):
            raise PrototypeConfigurationError(
                "entity_observation_noise_std must be non-negative"
            )

    def as_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible runtime configuration."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PrototypeConfig":
        if not isinstance(raw, Mapping):
            raise PrototypeConfigurationError("prototype config must be an object")
        names = {field.name for field in fields(cls)}
        values = {name: raw[name] for name in names if name in raw}
        return cls(**values)


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "The recurrent prototype requires PyTorch. Use the configured "
            "outputs/venv Python or install the optional torch dependency."
        )
    import torch

    return torch


def _simulator_modules() -> tuple[Any, ...]:
    """Import simulator modules in both package and top-level test layouts."""

    try:
        from ..engine import BattleEngine, DeterministicCycleController, ENGINE_VERSION
        from ..env import RewardConfig, SimulatorEnv
        from ..observation_v2 import (
            OBSERVATION_V2_CONTRACT_HASH,
            OBSERVATION_V2_SCHEMA_VERSION,
        )
        from ..ruleset import load_ruleset
        from ..roster import PLAYER_DECK
    except ImportError:  # pragma: no cover - exercised by top-level ``rl`` imports
        from simulator.engine import BattleEngine, DeterministicCycleController, ENGINE_VERSION
        from simulator.env import RewardConfig, SimulatorEnv
        from simulator.observation_v2 import (
            OBSERVATION_V2_CONTRACT_HASH,
            OBSERVATION_V2_SCHEMA_VERSION,
        )
        from simulator.ruleset import load_ruleset
        from simulator.roster import PLAYER_DECK
    return (
        BattleEngine,
        DeterministicCycleController,
        ENGINE_VERSION,
        RewardConfig,
        SimulatorEnv,
        OBSERVATION_V2_CONTRACT_HASH,
        OBSERVATION_V2_SCHEMA_VERSION,
        load_ruleset,
        PLAYER_DECK,
    )


def _model_and_learner(config: PrototypeConfig) -> Any:
    """Construct the public actor and explicitly separated value learner."""

    torch = _require_torch()
    from .learner import LearnerConfig, RecurrentPPOLearner
    from .model import ModelConfig, RecurrentHybridPolicy
    from cr_bot.features.channels import GLOBAL_SCALAR_IDX
    from cr_bot.features.global_features import CARD_COUNT

    # Make an unresumed run reproducible without touching simulator RNG state.
    torch.manual_seed(config.seed)
    model_config = ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=config.model_dim,
        encoder_dim=config.encoder_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_ff_dim=config.transformer_ff_dim,
        gru_hidden_dim=config.gru_hidden_dim,
        gru_layers=config.gru_layers,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
        hand_feature_offset=(len(GLOBAL_SCALAR_IDX) if config.explicit_hand_features else -1),
        hand_card_count=(CARD_COUNT if config.explicit_hand_features else 0),
        direct_public_action_features=config.direct_public_action_features,
        direct_public_card_features=config.direct_public_card_features,
        primary_public_card_features=config.primary_public_card_features,
        contextual_public_card_features=config.contextual_public_card_features,
        current_encoded_action_features=config.current_encoded_action_features,
        direct_public_mask_features=config.direct_public_mask_features,
        direct_public_context_features=config.direct_public_context_features,
        direct_public_slot_card_features=config.direct_public_slot_card_features,
        spatial_placement_features=config.spatial_placement_features,
        spatial_placement_dim=config.spatial_placement_dim,
    )
    learner_config = LearnerConfig(
        learning_rate=config.learning_rate,
        update_epochs=config.update_epochs,
        sequence_minibatch_size=config.sequence_minibatch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        entropy_coef=config.entropy_coef,
        bc_coef=config.behavior_cloning_coef,
        bc_factor_coef=config.behavior_cloning_factor_coef,
        bc_card_factor_weight=config.behavior_cloning_card_factor_weight,
        imitation_only=config.imitation_only,
        # DAgger/teacher forcing is a collector setting rather than a model
        # setting, but keeping it in the serialized runtime config makes the
        # data distribution reproducible from a checkpoint report.
        belief_coef=config.belief_coef,
        max_grad_norm=config.max_grad_norm,
        placement_max_grad_norm=config.placement_max_grad_norm,
        require_privileged_critic=config.use_privileged_critic,
    )
    policy = RecurrentHybridPolicy(model_config)
    if config.use_privileged_critic:
        return RecurrentPPOLearner(
            policy,
            config=learner_config,
            privileged_dim=PRIVILEGED_FEATURE_DIM,
            privileged_critic=True,
            device=config.device,
        )
    return RecurrentPPOLearner(
        policy,
        config=learner_config,
        privileged_critic=False,
        device=config.device,
    )


def _compile_policy_forward(learner: Any, config: PrototypeConfig) -> dict[str, object]:
    """Optionally compile only the parent policy forward graph.

    Rollout workers intentionally do not inherit this setting: compiling a
    separate graph in every simulator process adds more startup cost than it
    saves for the small worker batches.  The parent learner is where PPO
    re-evaluates long recurrent sequences, so a warmed compiled forward is a
    useful long-run optimization while leaving the module topology,
    parameters, masks, and checkpoint format unchanged.
    """

    requested = bool(config.compile_policy)
    result: dict[str, object] = {
        "requested": requested,
        "enabled": False,
        "backend": None,
        "note": None,
    }
    if not requested:
        result["note"] = "disabled"
        return result
    torch = _require_torch()
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        result["note"] = "torch.compile is unavailable"
        return result
    try:
        learner.policy.forward = compile_fn(
            learner.policy.forward,
            mode="reduce-overhead",
            fullgraph=False,
        )
    except Exception as error:  # pragma: no cover - backend/version dependent
        result["note"] = f"compile setup failed: {type(error).__name__}: {error}"
        return result
    result["enabled"] = True
    result["backend"] = "torch.compile"
    result["note"] = "first PPO evaluation includes graph compilation"
    return result


def _mix_seed(seed: int, *parts: int) -> int:
    """Small stable integer mixer for lane/variant seeds."""

    value = seed & ((1 << 64) - 1)
    for part in parts:
        value ^= (int(part) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = (value * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return value


def _reward_config_for(config: PrototypeConfig) -> Any:
    """Resolve the exact environment reward object for a runtime config."""

    RewardConfig = _simulator_modules()[3]
    if config.dense_reward:
        return RewardConfig()
    if config.potential_reward_weight > 0.0:
        return RewardConfig.terminal_with_potential(config.potential_reward_weight)
    return RewardConfig.terminal_outcome()


def _lane_deck_pairs(
    target_player: int,
    target_deck: tuple[str, ...],
    opponent_decks: Sequence[tuple[str, ...]],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Order target/opponent decks by world player for each rollout lane."""

    if target_player not in (0, 1):
        raise PrototypeConfigurationError("target_player must be 0 or 1")
    if target_player == 0:
        return tuple((target_deck, opponent_deck) for opponent_deck in opponent_decks)
    return tuple((opponent_deck, target_deck) for opponent_deck in opponent_decks)


def _make_environment(
    config: PrototypeConfig,
    ruleset: Any,
    lane: int,
    *,
    player_deck: tuple[str, ...] | None = None,
    opponent_deck: tuple[str, ...] | None = None,
    expose_privileged_info: bool = False,
    include_replay_hashes: bool = True,
    basic_scenario_source: str | None = None,
    basic_scenario_decisions: int = 64,
) -> Any:
    (
        BattleEngine,
        _controller,
        _engine_version,
        _RewardConfig,
        SimulatorEnv,
        _contract_hash,
        _schema_version,
        _load_ruleset,
        _deck,
    ) = _simulator_modules()
    environment = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        decision_interval_us=config.decision_interval_us,
        reward=_reward_config_for(config),
        # The public actor path must remain independent of info diagnostics.
        expose_privileged_info=expose_privileged_info,
        include_replay_hashes=include_replay_hashes,
        include_authoritative_state=not expose_privileged_info,
    )
    if any(
        (
            config.decision_interval_jitter_ticks,
            config.action_latency_max_steps,
            config.entity_observation_noise_std,
        )
    ):
        from .domain_randomization import DomainRandomizationConfig, DomainRandomizedEnv

        environment = DomainRandomizedEnv(
            environment,
            DomainRandomizationConfig(
                decision_interval_jitter_ticks=config.decision_interval_jitter_ticks,
                action_latency_max_steps=config.action_latency_max_steps,
                entity_observation_noise_std=config.entity_observation_noise_std,
            ),
            seed=_mix_seed(config.seed, lane, 0xD0A1),
        )
    if basic_scenario_source is not None:
        from .basic_scenarios import BasicMechanicsScenarioEnv, BasicScenarioConfig

        environment = BasicMechanicsScenarioEnv(
            environment,
            BasicScenarioConfig(
                source=basic_scenario_source,
                target_player=config.target_player,
                decision_limit=basic_scenario_decisions,
            ),
        )
    reset_v2 = getattr(environment, "reset_v2", None)
    if not callable(reset_v2):  # pragma: no cover - all prototype envs provide V2 reset
        reset_v2 = environment.reset
    reset_v2(
        seed=_mix_seed(config.seed, lane, 0xE001),
        decks=(
            tuple(_deck) if player_deck is None else tuple(player_deck),
            tuple(_deck) if opponent_deck is None else tuple(opponent_deck),
        ),
        shuffle_decks=config.shuffle_decks,
    )
    return environment


def _opponent_callback() -> Any:
    """Return the non-learning controller used by the first prototype."""

    _engine, controller_type, _version, _reward, _env, _hash, _schema, _load, _deck = (
        _simulator_modules()
    )
    controllers: dict[int, Any] = {}

    def choose(environment: Any, _public_observation: Any, player: int) -> Any:
        # This callback models the opponent's own controller.  It is not used
        # to construct actor tensors and cannot expose its state to the actor.
        state = getattr(environment, "state", None)
        if state is None:
            raise RuntimeError("opponent callback received an uninitialized environment")
        key = id(environment)
        controller = controllers.setdefault(key, controller_type(lane="alternate"))
        return controller.choose_action(environment.engine, state, player)

    return choose


def _evaluation_action_callback(policy_mode: str) -> Callable[[Any, Any, int], Any] | None:
    """Return the explicitly selected evaluation policy.

    The public counter is a deployment-safe reliability policy.  Keeping it
    behind an explicit mode prevents a report from accidentally presenting a
    rule-based action stream as neural actor performance.
    """

    if policy_mode == "actor":
        return None
    if policy_mode == "public-counter":
        from .public_counter import public_counter_action

        return public_counter_action
    if policy_mode == "strategic-counter":
        from .public_counter import strategic_counter_action

        return strategic_counter_action
    if policy_mode == "deterministic-counter":
        from .expert import deterministic_counter_action

        return deterministic_counter_action
    raise PrototypeConfigurationError(
        f"policy_mode must be one of {sorted(_EVALUATION_POLICIES)}, got {policy_mode!r}"
    )


def _card_value(card_id: str) -> float:
    from cr_bot.domain.card_metadata import CARD_METADATA

    metadata = CARD_METADATA.get(card_id)
    raw_id = metadata.get("id") if isinstance(metadata, dict) else None
    if type(raw_id) is not int:
        return 0.0
    return float(raw_id + 1) / 129.0


def _privileged_features(environment: Any, viewer: int) -> Sequence[float]:
    """Build fixed-width critic-only features from authoritative state.

    This function is intentionally passed only to the critic side of the
    collector.  The actor receives the public V2 object separately and never
    receives this vector.
    """

    state = getattr(environment, "state", None)
    if state is None:
        raise RuntimeError("privileged feature callback received an uninitialized environment")
    opponent = 1 - viewer
    maximum_elixir = max(1, int(environment.engine.ruleset.match.max_elixir_milli))
    maximum_tick = max(
        1,
        int(
            (environment.engine.ruleset.match.regulation_us
             + environment.engine.ruleset.match.overtime_us)
            // environment.engine.ruleset.tick_us
        ),
    )
    players = (state.players[viewer], state.players[opponent])
    features = [
        min(1.0, max(0.0, state.tick / maximum_tick)),
        min(1.0, max(0.0, players[0].elixir_milli / maximum_elixir)),
        min(1.0, max(0.0, players[1].elixir_milli / maximum_elixir)),
        min(1.0, max(0.0, players[0].crowns / 3.0)),
        min(1.0, max(0.0, players[1].crowns / 3.0)),
    ]
    for player in players:
        cards = list(player.hand) + list(player.draw_pile[:4])
        cards.extend([None] * (8 - len(cards)))
        features.extend(0.0 if card is None else _card_value(card) for card in cards[:8])
    tower_hp = [0, 0]
    tower_max = [0, 0]
    for entity in state.entities.values():
        if entity.kind == "tower":
            tower_hp[entity.owner] += max(0, int(entity.hp))
            tower_max[entity.owner] += max(1, int(entity.max_hp))
    features.extend(
        (
            tower_hp[viewer] / max(1, tower_max[viewer]),
            tower_hp[opponent] / max(1, tower_max[opponent]),
        )
    )
    if len(features) != PRIVILEGED_FEATURE_DIM:  # pragma: no cover - schema guard
        raise RuntimeError(f"privileged feature width changed: {len(features)}")
    return tuple(float(value) for value in features)


def _validate_ruleset(config: PrototypeConfig) -> Any:
    modules = _simulator_modules()
    ruleset = modules[7](config.ruleset_id)
    if config.decision_interval_us % ruleset.tick_us:
        raise PrototypeConfigurationError(
            "decision_interval_us must be a multiple of the ruleset physics tick"
        )
    if not bool(ruleset.metadata.get("training_ready", False)) and not config.allow_provisional:
        raise PrototypeConfigurationError(
            f"ruleset {ruleset.ruleset_id!r} is not training-ready; pass "
            "allow_provisional=True for a bounded prototype run"
        )
    return ruleset


def _checkpoint_metadata(config: PrototypeConfig, learner: Any, ruleset: Any) -> dict[str, object]:
    (
        _engine,
        _controller,
        engine_version,
        _reward,
        _env,
        contract_hash,
        schema_version,
        _load,
        _deck,
    ) = _simulator_modules()
    return {
        "schema_version": PROTOTYPE_SCHEMA_VERSION,
        "checkpoint_format": PROTOTYPE_CHECKPOINT_FORMAT,
        "code_revision": code_revision(),
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.content_hash,
        "engine_version": engine_version,
        "actor_observation": {
            "source": "SimulatorEnv.observe_v2",
            "schema_version": schema_version,
            "contract_hash": contract_hash,
            "privileged_inputs": False,
        },
        "critic_observation": {
            "privileged_inputs": bool(learner.uses_privileged_critic),
            "training_only": True,
        },
        "reward_config": _reward_config_for(config).as_dict(),
        "config": config.as_dict(),
        "learner_update_count": int(learner.update_count),
        "evaluation_warning": (
            "This prototype uses the pinned deterministic simulator; its ruleset "
            "is executable but not fidelity-ready."
        ),
    }


def save_prototype_checkpoint(
    path: str | Path,
    learner: Any,
    config: PrototypeConfig,
    ruleset: Any | None = None,
) -> Path:
    """Write learner state plus an auditable public-actor contract."""

    torch = _require_torch()
    if ruleset is None:
        ruleset = _validate_ruleset(config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": _checkpoint_metadata(config, learner, ruleset),
            "learner": learner.checkpoint_state(),
        },
        destination,
    )
    return destination


def _temporary_checkpoint_path(destination: Path) -> Path:
    """Create an adjacent candidate path that cannot replace a clean artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".candidate",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _quarantine_checkpoint_path(candidate: Path) -> Path:
    """Rename a rejected candidate to a visible, recoverable quarantine file."""

    quarantine = candidate.with_name(f"{candidate.name}.quarantine")
    candidate.replace(quarantine)
    return quarantine


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    torch = _require_torch()
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise PrototypeConfigurationError("prototype checkpoint must contain an object")
    metadata = payload.get("metadata")
    learner = payload.get("learner")
    if not isinstance(metadata, Mapping) or not isinstance(learner, Mapping):
        raise PrototypeConfigurationError("prototype checkpoint is missing metadata or learner state")
    if metadata.get("checkpoint_format") != PROTOTYPE_CHECKPOINT_FORMAT:
        raise PrototypeConfigurationError("unsupported prototype checkpoint format")
    actor = metadata.get("actor_observation")
    if not isinstance(actor, Mapping) or actor.get("privileged_inputs") is not False:
        raise PrototypeConfigurationError(
            "refusing checkpoint whose actor observation is not explicitly public-only"
        )
    return {"metadata": dict(metadata), "learner": learner}


def _architecture_config(config: PrototypeConfig) -> tuple[object, ...]:
    return (
        config.model_dim,
        config.encoder_dim,
        config.transformer_heads,
        config.transformer_layers,
        config.transformer_ff_dim,
        config.gru_hidden_dim,
        config.gru_layers,
        config.use_privileged_critic,
        config.explicit_hand_features,
        config.direct_public_action_features,
        config.direct_public_card_features,
        config.primary_public_card_features,
        config.contextual_public_card_features,
        config.current_encoded_action_features,
        config.direct_public_mask_features,
        config.direct_public_context_features,
        config.direct_public_slot_card_features,
        config.spatial_placement_features,
        config.spatial_placement_dim,
    )


def _load_prototype_checkpoint(
    path: str | Path,
    *,
    config: PrototypeConfig | None = None,
    device: str | None = None,
    allow_stale_ruleset: bool = False,
    restore_rng: bool = True,
) -> tuple[Any, PrototypeConfig, dict[str, Any]]:
    """Load a prototype artifact and return learner/config/metadata.

    A ruleset hash mismatch remains fail-closed by default.  The explicit
    ``allow_stale_ruleset`` escape hatch exists for observation-only shadow
    inference after a data-only ruleset update; callers that train or evaluate
    the simulator continue to use the default strict behavior.
    """

    if type(allow_stale_ruleset) is not bool:
        raise PrototypeConfigurationError("allow_stale_ruleset must be boolean")
    if type(restore_rng) is not bool:
        raise PrototypeConfigurationError("restore_rng must be boolean")

    payload = _read_checkpoint(path)
    metadata = payload["metadata"]
    stored_config = PrototypeConfig.from_mapping(metadata.get("config", {}))
    effective_config = stored_config if config is None else config
    if config is not None and _architecture_config(config) != _architecture_config(stored_config):
        raise PrototypeConfigurationError(
            "runtime config changes the checkpoint model/critic architecture"
        )
    if device is not None:
        effective_config = PrototypeConfig.from_mapping(
            {**effective_config.as_dict(), "device": device}
        )
    ruleset = _validate_ruleset(
        PrototypeConfig.from_mapping(
            {**effective_config.as_dict(), "allow_provisional": True}
        )
    )
    checkpoint_ruleset_id = metadata.get("ruleset_id")
    checkpoint_ruleset_hash = metadata.get("ruleset_hash")
    if checkpoint_ruleset_id != ruleset.ruleset_id:
        raise PrototypeConfigurationError(
            "checkpoint ruleset does not match the requested runtime ruleset"
        )
    if not (
        isinstance(checkpoint_ruleset_hash, str)
        and len(checkpoint_ruleset_hash) == len("sha256:") + 64
        and checkpoint_ruleset_hash.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in checkpoint_ruleset_hash[7:])
    ):
        raise PrototypeConfigurationError(
            "checkpoint is missing a well-formed ruleset content hash"
        )
    checkpoint_ruleset_match = checkpoint_ruleset_hash == ruleset.content_hash
    if not checkpoint_ruleset_match and not allow_stale_ruleset:
        raise PrototypeConfigurationError(
            "checkpoint ruleset does not match the requested runtime ruleset"
        )
    _engine, _controller, engine_version, _reward, _env, contract_hash, schema_version, _load, _deck = (
        _simulator_modules()
    )
    if metadata.get("engine_version") != engine_version:
        raise PrototypeConfigurationError("checkpoint engine version does not match the runtime")
    actor = metadata.get("actor_observation", {})
    if actor.get("schema_version") != schema_version or actor.get("contract_hash") != contract_hash:
        raise PrototypeConfigurationError("checkpoint observation contract does not match the runtime")
    learner = _model_and_learner(effective_config)
    learner.load_checkpoint_state(payload["learner"], restore_rng=restore_rng)
    # These private, in-memory fields let shadow reports describe exactly why
    # an explicitly stale checkpoint was accepted without changing the
    # serialized checkpoint schema or weakening train/evaluate callers.
    metadata = dict(metadata)
    metadata["_checkpoint_ruleset_id"] = checkpoint_ruleset_id
    metadata["_checkpoint_ruleset_hash"] = checkpoint_ruleset_hash
    metadata["_runtime_ruleset_id"] = ruleset.ruleset_id
    metadata["_runtime_ruleset_hash"] = ruleset.content_hash
    metadata["_checkpoint_ruleset_match"] = checkpoint_ruleset_match
    metadata["_stale_ruleset_allowed"] = (
        bool(allow_stale_ruleset) and not checkpoint_ruleset_match
    )
    return learner, effective_config, metadata


def load_prototype_checkpoint(
    path: str | Path,
    *,
    config: PrototypeConfig | None = None,
    device: str | None = None,
    restore_rng: bool = True,
) -> tuple[Any, PrototypeConfig, dict[str, Any]]:
    """Load a checkpoint with strict current-ruleset validation.

    ``restore_rng=False`` is intended for frozen inference opponents created
    inside another training process.  It prevents loading a serialized
    opponent from changing the learner's global sampling stream.
    """

    return _load_prototype_checkpoint(
        path,
        config=config,
        device=device,
        allow_stale_ruleset=False,
        restore_rng=restore_rng,
    )


def load_shadow_prototype_checkpoint(
    path: str | Path,
    *,
    device: str | None = "auto",
    allow_stale_ruleset: bool = False,
) -> tuple[Any, PrototypeConfig, dict[str, Any]]:
    """Load a checkpoint for actor-only shadow inference.

    The stale-ruleset exception is intentionally exposed only through this
    shadow-specific loader.  It still requires the same ruleset ID, a
    well-formed checkpoint hash, the current engine version, and the current
    public observation contract.
    """

    return _load_prototype_checkpoint(
        path,
        device=device,
        allow_stale_ruleset=allow_stale_ruleset,
        restore_rng=True,
    )


def _make_collector(
    learner: Any,
    config: PrototypeConfig,
    *,
    deterministic: bool = False,
    stop: bool = False,
    freeze_completed_lanes: bool = False,
    expert_action: Callable[[Any, Any, int], Any] | None = None,
    diagnostic_teacher_action: Callable[[Any, Any, int], Any] | None = None,
    expert_execution_probability: float | None = None,
    lane_decks: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None = None,
    lane_offset: int = 0,
    actor_only_observations: bool = False,
    fast_deterministic: bool = False,
    opponent_action: Callable[[Any, Any, int], Any | None] | None = None,
    batch_step: Callable[[Sequence[Sequence[Any | None]]], Sequence[Any]] | None = None,
    diagnostics: bool = False,
) -> Any:
    from .collector import CollectorConfig, RecurrentRolloutCollector

    _deck = _simulator_modules()[-1]
    return RecurrentRolloutCollector(
        learner,
        CollectorConfig(
            horizon=config.horizon,
            target_player=config.target_player,
            seed=config.seed,
            lane_offset=lane_offset,
            shuffle_decks=config.shuffle_decks,
            decks=None if lane_decks is not None else (tuple(_deck), tuple(_deck)),
            lane_decks=lane_decks,
            collect_belief_targets=config.collect_belief_targets,
            actor_only_observations=actor_only_observations,
            deterministic=deterministic,
            fast_deterministic=fast_deterministic,
            expert_execution_probability=(
                config.expert_execution_probability
                if expert_execution_probability is None
                else expert_execution_probability
            ),
            expert_label_on_threat_only=config.expert_label_on_threat_only,
            expert_label_on_disagreement=config.expert_label_on_disagreement,
            stop_on_episode_end=stop,
            freeze_completed_lanes=freeze_completed_lanes,
            diagnostics=diagnostics,
        ),
        opponent_action=_opponent_callback() if opponent_action is None else opponent_action,
        expert_action=expert_action,
        diagnostic_teacher_action=diagnostic_teacher_action,
        privileged_feature_fn=(
            _privileged_features if config.use_privileged_critic else None
        ),
        batch_step=batch_step,
    )


def _make_batch_stepper(
    config: PrototypeConfig,
    environments: Sequence[Any],
) -> tuple[Any | None, Callable[[Sequence[Sequence[Any | None]]], Sequence[Any]] | None]:
    """Create an optional process-backed V2 lane stepper.

    The domain-randomization wrapper owns per-lane timing, latency, and noise
    state that the existing serialized process workers do not carry.  Keep the
    optimized backend explicit and fail closed until that wrapper has a
    canonical transport of its own.
    """

    if config.env_backend in {"reference", "rollout-process"}:
        return None, None
    if any(
        (
            config.decision_interval_jitter_ticks,
            config.action_latency_max_steps,
            config.entity_observation_noise_std,
        )
    ):
        raise PrototypeConfigurationError(
        "process environment backends cannot be combined with domain randomization"
        )
    try:
        from ..env import VectorSimulatorEnv
    except ImportError:  # pragma: no cover - exercised by top-level imports
        from simulator.env import VectorSimulatorEnv

    vector = VectorSimulatorEnv(
        environments,
        backend=config.env_backend,
        workers=config.env_workers,
    )
    return vector, vector.step_v2


def _resume_config(
    config: PrototypeConfig,
    *,
    checkpoint: str | Path | None,
    resume_learning_rate: float | None,
    resume_disable_belief_loss: bool,
    resume_reset_optimizer: bool,
) -> PrototypeConfig:
    """Apply explicit retry settings before constructing the resumed learner.

    The regular runtime configuration remains the source of model and
    observation settings.  Retry controls are deliberately separate so a
    checkpoint can be resumed with a safer optimizer/objective without
    silently changing the serialized model architecture or public actor
    inputs.
    """

    for name, value in (
        ("resume_disable_belief_loss", resume_disable_belief_loss),
        ("resume_reset_optimizer", resume_reset_optimizer),
    ):
        if type(value) is not bool:
            raise PrototypeConfigurationError(f"{name} must be boolean")
    if checkpoint is None and (
        resume_learning_rate is not None
        or resume_disable_belief_loss
        or resume_reset_optimizer
    ):
        raise PrototypeConfigurationError(
            "resume controls require --checkpoint"
        )

    values = config.as_dict()
    if resume_learning_rate is not None:
        if (
            isinstance(resume_learning_rate, bool)
            or not isinstance(resume_learning_rate, (int, float))
            or not isfinite(float(resume_learning_rate))
            or float(resume_learning_rate) <= 0.0
        ):
            raise PrototypeConfigurationError(
                "resume_learning_rate must be a finite positive value"
            )
        values["learning_rate"] = float(resume_learning_rate)
    if resume_disable_belief_loss:
        # Disable both the coefficient and target collection.  The latter
        # avoids spending CPU time building training-only targets, while the
        # former makes the effective learner objective unambiguous.
        values["belief_coef"] = 0.0
        values["collect_belief_targets"] = False
    return PrototypeConfig.from_mapping(values)


def _set_optimizer_learning_rate(learner: Any, learning_rate: float) -> None:
    """Set every Adam parameter group after checkpoint state has been loaded."""

    optimizer = getattr(learner, "optimizer", None)
    parameter_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise PrototypeConfigurationError(
            "resumed learner does not expose optimizer parameter groups"
        )
    for group in parameter_groups:
        group["lr"] = float(learning_rate)


def _apply_resume_controls(
    learner: Any,
    config: PrototypeConfig,
    *,
    resume_learning_rate: float | None,
    resume_reset_optimizer: bool,
) -> None:
    """Apply optimizer retry controls after ``load_checkpoint_state``.

    Loading a learner checkpoint restores its serialized Adam parameter groups
    and moments.  Therefore the learning-rate override must happen *after*
    loading.  Resetting only the optimizer state preserves the model weights
    and update counter while discarding moments from a potentially unstable
    run.
    """

    if resume_learning_rate is None and not resume_reset_optimizer:
        return
    optimizer = getattr(learner, "optimizer", None)
    if optimizer is None:
        raise PrototypeConfigurationError("resumed learner does not expose an optimizer")
    if resume_reset_optimizer:
        state = getattr(optimizer, "state", None)
        clear = getattr(state, "clear", None)
        if not callable(clear):
            raise PrototypeConfigurationError(
                "resumed learner optimizer state cannot be reset"
            )
        clear()
    _set_optimizer_learning_rate(
        learner,
        config.learning_rate if resume_learning_rate is None else resume_learning_rate,
    )


def _apply_update_approx_kl_guard(
    learner: Any,
    metrics: Any,
    *,
    max_update_approx_kl: float | None,
    state_before_update: Any,
    starting_update: int,
    enabled: bool = True,
) -> dict[str, object]:
    """Accept or roll back one PPO update using its measured policy movement.

    This is deliberately a training-time checkpoint decision. It does not
    inspect or alter an environment action, legality mask, or observation.
    Keeping the operation separate also makes the safety envelope directly
    testable without running a simulator rollout. Pure behavior-cloning
    warm-starts opt out: their intentionally large movement from a random
    policy is not PPO policy drift and must not be rolled back by a PPO KL
    threshold.
    """

    if type(enabled) is not bool:
        raise TypeError("enabled must be boolean")

    observed_approx_kl = float(metrics.approx_kl)
    # ``approx_kl`` is the conventional PPO sampled estimate, but its signed
    # mean can be close to zero when different branches move in opposite
    # directions.  Use the learner's conservative absolute movement metric for
    # the rollback decision whenever it is available, while retaining the
    # signed value in the report for diagnosis and compatibility.
    raw_mean_abs_log_ratio = getattr(metrics, "mean_abs_log_ratio", None)
    if raw_mean_abs_log_ratio is None:
        observed_mean_abs_log_ratio = abs(observed_approx_kl)
        guard_metric = "absolute_approx_kl_fallback"
    else:
        observed_mean_abs_log_ratio = float(raw_mean_abs_log_ratio)
        guard_metric = "mean_abs_log_ratio"
    guard: dict[str, object] = {
        "status": "disabled" if enabled else "not_applicable",
        "max_approx_kl": (
            None
            if max_update_approx_kl is None
            else float(max_update_approx_kl)
        ),
        "observed_approx_kl": observed_approx_kl,
        "observed_mean_abs_log_ratio": observed_mean_abs_log_ratio,
        "guard_metric": guard_metric,
        "starting_update": int(starting_update),
        "attempted_update": int(metrics.update_index),
        "accepted_update": int(learner.update_count),
    }
    if not enabled:
        guard["reason"] = "imitation_only_update"
        return guard
    if max_update_approx_kl is None:
        return guard
    if observed_mean_abs_log_ratio > float(max_update_approx_kl):
        if state_before_update is None:  # pragma: no cover - config invariant
            raise RuntimeError("PPO KL guard has no pre-update learner state")
        learner.load_checkpoint_state(state_before_update)
        guard.update(
            {
                "status": "rolled_back",
                "accepted_update": int(learner.update_count),
                "reason": "policy_movement_exceeded_bound",
            }
        )
    else:
        guard["status"] = "accepted"
    return guard


class _TrainingProgress:
    """Render training progress without affecting stdout JSON."""

    _BAR_WIDTH = 24
    _MIN_REFRESH_SECONDS = 0.5
    _NON_TTY_REFRESH_SECONDS = 5.0

    def __init__(
        self,
        total_updates: int,
        transitions_per_update: int,
        *,
        stream: TextIO,
        enabled: bool = True,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        _positive_int("total_updates", total_updates)
        _positive_int("transitions_per_update", transitions_per_update)
        self._total_updates = total_updates
        self._transitions_per_update = transitions_per_update
        self._stream = stream
        self._enabled = enabled
        self._clock = clock
        self._started = clock()
        self._last_line_length = 0
        self._last_render_elapsed = 0.0
        self._rendered = False
        self._closed = False
        isatty = getattr(stream, "isatty", None)
        self._is_tty = bool(isatty()) if callable(isatty) else False
        if self._enabled:
            self._render(0, 0, elapsed=0.0)

    def _render(
        self,
        completed_updates: int,
        completed_transitions: int,
        *,
        elapsed: float,
    ) -> None:
        total_transitions = self._total_updates * self._transitions_per_update
        fraction = min(1.0, max(0.0, completed_transitions / total_transitions))
        filled = int(self._BAR_WIDTH * fraction)
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        line = (
            f"recurrent PPO [{bar}] "
            f"{completed_updates}/{self._total_updates} updates | "
            f"{completed_transitions:,}/{self._total_updates * self._transitions_per_update:,} "
            f"transitions | elapsed {elapsed:.1f}s"
        )
        if self._is_tty:
            padding = max(0, self._last_line_length - len(line))
            self._stream.write(f"\r{line}{' ' * padding}")
            self._last_line_length = max(self._last_line_length, len(line))
        else:
            self._stream.write(line + "\n")
        self._stream.flush()
        self._rendered = True
        self._last_render_elapsed = elapsed

    def update(self, completed_updates: int, completed_transitions: int) -> None:
        """Render one completed PPO update and its accumulated transitions."""

        if not self._enabled or self._closed:
            return
        elapsed = max(0.0, self._clock() - self._started)
        self._render(completed_updates, completed_transitions, elapsed=elapsed)

    def advance(self, completed_transitions: int) -> None:
        """Refresh collection progress without writing once per simulator step."""

        if not self._enabled or self._closed:
            return
        total_transitions = self._total_updates * self._transitions_per_update
        completed_transitions = min(total_transitions, max(0, completed_transitions))
        elapsed = max(0.0, self._clock() - self._started)
        refresh_seconds = (
            self._MIN_REFRESH_SECONDS
            if self._is_tty
            else self._NON_TTY_REFRESH_SECONDS
        )
        if (
            completed_transitions < total_transitions
            and elapsed - self._last_render_elapsed < refresh_seconds
        ):
            return
        completed_updates = min(
            self._total_updates,
            completed_transitions // self._transitions_per_update,
        )
        self._render(completed_updates, completed_transitions, elapsed=elapsed)

    def close(self) -> None:
        """Leave the terminal cursor on a fresh line after the final update."""

        if self._closed:
            return
        self._closed = True
        if self._enabled and self._is_tty and self._rendered:
            self._stream.write("\n")
            self._stream.flush()


def train_prototype(
    config: PrototypeConfig = PrototypeConfig(),
    *,
    checkpoint: str | Path | None = None,
    checkpoint_out: str | Path = "outputs/simulator/training/recurrent-prototype.pt",
    progress_callback: Callable[[int, int], None] | None = None,
    progress_step_callback: Callable[[int], None] | None = None,
    resume_learning_rate: float | None = None,
    resume_disable_belief_loss: bool = False,
    resume_reset_optimizer: bool = False,
    expert_guidance: bool = False,
    expert_action_callback: Callable[[Any, Any, int], Any] | None = None,
    rollout_expert_teacher: str | None = None,
    player_deck: Sequence[str] | None = None,
    opponent_decks: Sequence[Sequence[str]] | None = None,
    opponent_action: Callable[[Any, Any, int], Any | None] | None = None,
    rollout_opponent_specs: Sequence[tuple[str, int]] | None = None,
    opponent_uses_public_observation: bool | None = None,
    basic_scenario_sources: Sequence[str | None] | None = None,
    basic_scenario_decisions: int = 64,
) -> dict[str, object]:
    """Run recurrent PPO updates and save one complete prototype artifact.

    ``resume_learning_rate``, ``resume_disable_belief_loss``, and
    ``resume_reset_optimizer`` are explicit retry controls.  They are applied
    only when ``checkpoint`` is supplied.  In particular, the learning-rate
    override is applied after checkpoint loading because loading Adam state
    restores the checkpoint's serialized parameter-group values.
    """

    # This is the primary end-to-end training benchmark window.  Start before
    # validation, ruleset/checkpoint loading, and model construction so the
    # reported rate cannot silently omit successful-run setup work.
    started = perf_counter()
    torch = _require_torch()
    if not isinstance(config, PrototypeConfig):
        raise TypeError("config must be a PrototypeConfig")
    if type(expert_guidance) is not bool:
        raise TypeError("expert_guidance must be boolean")
    if expert_action_callback is not None and not callable(expert_action_callback):
        raise TypeError("expert_action_callback must be callable when provided")
    if rollout_expert_teacher is not None and rollout_expert_teacher not in {
        "public-counter",
        "strategic-counter",
        "deterministic-counter",
    }:
        raise PrototypeConfigurationError(
            "rollout_expert_teacher must name a supported built-in teacher"
        )
    if opponent_uses_public_observation is not None and type(
        opponent_uses_public_observation
    ) is not bool:
        raise TypeError("opponent_uses_public_observation must be boolean or None")
    if type(basic_scenario_decisions) is not int or basic_scenario_decisions <= 0:
        raise PrototypeConfigurationError(
            "basic_scenario_decisions must be a positive integer"
        )
    resolved_basic_scenario_sources: tuple[str | None, ...] | None = None
    if basic_scenario_sources is not None:
        if len(basic_scenario_sources) != config.envs:
            raise PrototypeConfigurationError(
                "basic_scenario_sources must contain one source or None per environment"
            )
        from .basic_scenarios import BasicScenarioConfig

        resolved_rows: list[str | None] = []
        for source in basic_scenario_sources:
            if source is None:
                resolved_rows.append(None)
                continue
            # Constructing the contract here normalizes and validates every
            # source before a checkpoint/model or environment is mutated.
            row = BasicScenarioConfig(
                source=source,
                target_player=config.target_player,
                decision_limit=basic_scenario_decisions,
            )
            resolved_rows.append(row.source)
        resolved_basic_scenario_sources = tuple(resolved_rows)
        if any(source is not None for source in resolved_rows) and config.env_backend != "reference":
            raise PrototypeConfigurationError(
                "basic-mechanics short scenarios currently require env_backend='reference'; "
                "process transports do not yet carry scenario episode boundaries/rewards"
            )
    if expert_guidance and not (
        config.behavior_cloning_coef > 0.0
        or config.behavior_cloning_factor_coef > 0.0
    ):
        raise PrototypeConfigurationError(
            "expert_guidance requires a positive joint or factor behavior-cloning coefficient"
        )
    if config.imitation_only and not expert_guidance:
        raise PrototypeConfigurationError(
            "imitation_only requires expert_guidance"
        )
    if (
        expert_guidance
        and not config.imitation_only
        and float(config.expert_execution_probability) > 0.0
    ):
        raise PrototypeConfigurationError(
            "PPO cannot use teacher-executed transitions; set "
            "expert_execution_probability=0 for DAgger labels or enable imitation_only"
        )
    effective_config = _resume_config(
        config,
        checkpoint=checkpoint,
        resume_learning_rate=resume_learning_rate,
        resume_disable_belief_loss=resume_disable_belief_loss,
        resume_reset_optimizer=resume_reset_optimizer,
    )
    ruleset = _validate_ruleset(effective_config)
    _deck = _simulator_modules()[-1]
    if player_deck is None:
        resolved_player_deck = tuple(_deck)
    else:
        if isinstance(player_deck, (str, bytes)):
            raise PrototypeConfigurationError(
                "player_deck must contain eight cards, not a string"
            )
        try:
            resolved_player_deck = tuple(player_deck)
        except TypeError as error:
            raise PrototypeConfigurationError(
                "player_deck must be a sequence of card identifiers"
            ) from error
        if len(resolved_player_deck) != 8 or any(
            not isinstance(card, str) or not card.strip()
            for card in resolved_player_deck
        ):
            raise PrototypeConfigurationError(
                "player_deck must contain eight non-empty card identifiers"
            )
        unknown = sorted(set(resolved_player_deck) - set(ruleset.cards))
        if unknown:
            raise PrototypeConfigurationError(
                f"player_deck contains unknown ruleset cards: {unknown}"
            )
        if len(set(resolved_player_deck)) != 8:
            raise PrototypeConfigurationError(
                "player_deck must not contain duplicate cards"
            )
    if opponent_decks is None:
        resolved_opponent_decks = tuple(tuple(_deck) for _ in range(effective_config.envs))
    else:
        if len(opponent_decks) != effective_config.envs:
            raise PrototypeConfigurationError(
                "opponent_decks must contain one eight-card deck per environment"
            )
        resolved_rows: list[tuple[str, ...]] = []
        for index, deck in enumerate(opponent_decks):
            row = tuple(deck)
            if len(row) != 8 or any(not isinstance(card, str) or not card.strip() for card in row):
                raise PrototypeConfigurationError(
                    f"opponent_decks[{index}] must contain eight non-empty card identifiers"
                )
            unknown = sorted(set(row) - set(ruleset.cards))
            if unknown:
                raise PrototypeConfigurationError(
                    f"opponent_decks[{index}] contains unknown ruleset cards: {unknown}"
                )
            if len(set(row)) != 8:
                raise PrototypeConfigurationError(
                    f"opponent_decks[{index}] must not contain duplicate cards"
                )
            resolved_rows.append(row)
        resolved_opponent_decks = tuple(resolved_rows)
    if opponent_action is not None and not callable(opponent_action):
        raise TypeError("opponent_action must be callable when provided")
    if rollout_opponent_specs is not None:
        if len(rollout_opponent_specs) != effective_config.envs:
            raise PrototypeConfigurationError(
                "rollout_opponent_specs must contain one (strategy, seed) pair per environment"
            )
        for index, raw_spec in enumerate(rollout_opponent_specs):
            if (
                not isinstance(raw_spec, Sequence)
                or isinstance(raw_spec, (str, bytes))
                or len(raw_spec) != 2
                or not isinstance(raw_spec[0], str)
                or type(raw_spec[1]) is not int
            ):
                raise PrototypeConfigurationError(
                    f"rollout_opponent_specs[{index}] must be a (strategy, integer seed) pair"
                )
    if effective_config.overlap_rollouts and effective_config.env_backend != "rollout-process":
        raise PrototypeConfigurationError(
            "overlap_rollouts requires the rollout-process backend"
        )
    if effective_config.diagnostic_trace_out is not None and effective_config.env_backend == "rollout-process":
        raise PrototypeConfigurationError(
            "diagnostic_trace_out requires the reference/vector collector; "
            "rollout-process does not transport per-decision model diagnostics"
        )
    # The built-in and generalized simulator-side controllers consume the
    # authoritative state, not the opponent's public observation.  Keeping
    # the old two-view behavior as the default for a caller-supplied callback
    # preserves the public callback contract; generalized training passes the
    # explicit value when it knows whether a frozen public actor is assigned.
    if opponent_uses_public_observation is None:
        resolved_opponent_uses_public_observation = opponent_action is not None
    else:
        resolved_opponent_uses_public_observation = opponent_uses_public_observation
    # Pin the simulator revision for this complete rollout/update sequence.
    # The checkout may remain dirty during development, but a new committed
    # HEAD means the simulator changed underneath the experiment and its
    # checkpoint must not be promoted as if it came from one revision.
    run_start_revision = code_revision()
    if checkpoint is None:
        learner = _model_and_learner(effective_config)
        resumed_from = None
    else:
        learner, loaded_config, _metadata = load_prototype_checkpoint(
            checkpoint,
            config=effective_config,
        )
        # ``config`` controls the new run budget and environment fan-out; the
        # checkpoint architecture/hyperparameters were already checked above.
        if loaded_config.ruleset_id != effective_config.ruleset_id:
            raise PrototypeConfigurationError("resume checkpoint ruleset differs from runtime config")
        _apply_resume_controls(
            learner,
            effective_config,
            resume_learning_rate=resume_learning_rate,
            resume_reset_optimizer=resume_reset_optimizer,
        )
        resumed_from = str(checkpoint)

    policy_compile = _compile_policy_forward(learner, effective_config)

    # ``player_deck`` names the learner's deck, while the simulator's deck
    # tuple is always ordered by world player.  Swap the lane pair when the
    # learner is assigned to player 1 so side-balanced training exercises the
    # same public actor contract from both sides of the arena.
    lane_decks = _lane_deck_pairs(
        effective_config.target_player,
        resolved_player_deck,
        resolved_opponent_decks,
    )
    rollout_farm = None
    if effective_config.env_backend == "rollout-process":
        if opponent_action is not None:
            raise PrototypeConfigurationError(
                "rollout-process uses serialized simulator-side opponent specs; "
                "use a vector backend for a Python opponent callback"
            )
        if expert_guidance and expert_action_callback is not None:
            raise PrototypeConfigurationError(
                "rollout-process supports only built-in expert teachers; omit "
                "expert_action_callback"
            )
        from .rollout_farm import RolloutFarm

        rollout_farm = RolloutFarm(
            effective_config,
            learner,
            lane_decks,
            opponent_specs=rollout_opponent_specs,
            expert_teacher=(
                rollout_expert_teacher
                if rollout_expert_teacher is not None
                else "deterministic-counter"
                if expert_guidance
                else None
            ),
            double_buffer=effective_config.overlap_rollouts,
        )
        environments: Sequence[Any] = ()
        vector_environment = None
        batch_step = None
    else:
        environments = [
            _make_environment(
                effective_config,
                ruleset,
                lane,
                player_deck=lane_decks[lane][0],
                opponent_deck=lane_decks[lane][1],
                basic_scenario_source=(
                    None
                    if resolved_basic_scenario_sources is None
                    else resolved_basic_scenario_sources[lane]
                ),
                basic_scenario_decisions=basic_scenario_decisions,
            )
            for lane in range(effective_config.envs)
        ]
        vector_environment, batch_step = _make_batch_stepper(
            effective_config,
            environments,
        )
    expert_action = None
    if expert_guidance:
        if expert_action_callback is None:
            from .expert import deterministic_counter_action

            expert_action = deterministic_counter_action
        else:
            expert_action = expert_action_callback
    rollout_executor: ThreadPoolExecutor | None = None
    pending_rollout: Future[Any] | None = None
    try:
        update_rows: list[dict[str, object]] = []
        diagnostic_trace_rows: list[dict[str, object]] = []
        diagnostic_update_rows: list[dict[str, object]] = []
        diagnostic_enabled = effective_config.diagnostic_trace_out is not None
        previous_diagnostic_distributions: dict[str, Mapping[str, object]] | None = None
        rollout_farm_timing: list[dict[str, object]] = []
        aggregate = {name: 0 for name in ("completed_matches", "wins", "draws", "losses", "truncated_matches")}
        starting_update = int(learner.update_count)
        rollout_state = None
        rollout_reset_mask = None
        episode_counts: tuple[int, ...] | None = None
        pending_buffer_index = 0
        if rollout_farm is not None and effective_config.overlap_rollouts:
            rollout_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="rl-rollout-submit",
            )

        def collector_config_for(local_index: int) -> PrototypeConfig:
            return PrototypeConfig.from_mapping(
                {
                    **effective_config.as_dict(),
                    "seed": _mix_seed(
                        effective_config.seed,
                        starting_update + local_index,
                        0xC011,
                    ),
                }
            )

        for local_update in range(effective_config.updates):
            # Keep episode reseeding deterministic across update boundaries while
            # still producing a different lane schedule for each update.
            collector_config = collector_config_for(local_update)
            # Rollout-farm workers own the collector.  Avoid constructing an
            # unused parent collector on every PPO update; the reference and
            # vector backends still build the collector exactly as before.
            collector = None
            if rollout_farm is None:
                collector = _make_collector(
                    learner,
                    collector_config,
                    deterministic=effective_config.deterministic_rollouts,
                    expert_action=expert_action,
                    diagnostic_teacher_action=(
                        _evaluation_action_callback("strategic-counter")
                        if diagnostic_enabled
                        else None
                    ),
                    lane_decks=lane_decks,
                    opponent_action=opponent_action,
                    batch_step=batch_step,
                    actor_only_observations=not resolved_opponent_uses_public_observation,
                    diagnostics=diagnostic_enabled,
                )
            current_diagnostic_rows: list[dict[str, object]] = []

            def diagnostic_decision_callback(record: Any) -> None:
                row = _trace_decision(
                    record,
                    target_player=effective_config.target_player,
                    include_positions=True,
                )
                row["update"] = int(learner.update_count + 1)
                row["local_update"] = local_update + 1
                row["lane"] = int(record.lane)
                current_diagnostic_rows.append(row)
            rollout_started = perf_counter()
            rollout_progress = None
            if progress_step_callback is not None:
                transition_offset = local_update * effective_config.envs * effective_config.horizon

                def rollout_progress(
                    completed_transitions: int,
                    *,
                    offset: int = transition_offset,
                ) -> None:
                    progress_step_callback(offset + completed_transitions)

            if rollout_farm is not None:
                from .collector import RolloutResult

                if pending_rollout is not None:
                    farm_batch = pending_rollout.result()
                    pending_rollout = None
                    # The pending batch was collected under the weights that
                    # preceded the update just completed. Publish the new
                    # weights only after that batch is safely detached from
                    # the worker's output buffer.
                    rollout_farm.sync_weights(
                        learner,
                        seed=_mix_seed(
                            effective_config.seed,
                            starting_update + local_update,
                            0xFA57,
                        ),
                    )
                else:
                    if local_update:
                        rollout_farm.sync_weights(
                            learner,
                            seed=_mix_seed(
                                effective_config.seed,
                                starting_update + local_update,
                                0xFA57,
                            ),
                        )
                    farm_batch = rollout_farm.collect(
                        collector_config,
                        buffer_index=pending_buffer_index,
                    )
                rollout_farm_timing.append(
                    {
                        "startup_seconds": farm_batch.startup_seconds,
                        "collect_wall_seconds": farm_batch.collect_wall_seconds,
                        "worker_collect_seconds": list(
                            farm_batch.worker_collect_seconds
                        ),
                    }
                )
                farm_learner_batch = farm_batch.learner_batch
                farm_bootstrap_values = farm_learner_batch.bootstrap_values
                if farm_bootstrap_values is None:
                    if farm_learner_batch.next_values is None:
                        raise RuntimeError(
                            "rollout farm returned no successor values for PPO"
                        )
                    # The farm transports explicit successor values for
                    # nonterminal truncations.  RolloutResult keeps its
                    # historical final-value field, so derive that view at
                    # the boundary without losing the per-transition data
                    # carried by LearnerBatch.
                    farm_bootstrap_values = farm_learner_batch.next_values[:, -1]
                result = RolloutResult(
                    learner_batch=farm_learner_batch,
                    final_observations=(),
                    next_rollout_state=None,
                    bootstrap_values=farm_bootstrap_values,
                    stats=farm_batch.stats,
                    next_reset_mask=None,
                    episode_counts=farm_batch.episode_counts,
                )
                if (
                    rollout_executor is not None
                    and local_update + 1 < effective_config.updates
                ):
                    next_buffer_index = 1 - pending_buffer_index
                    pending_rollout = rollout_executor.submit(
                        rollout_farm.collect,
                        collector_config_for(local_update + 1),
                        buffer_index=next_buffer_index,
                    )
                    pending_buffer_index = next_buffer_index
            else:
                if collector is None:  # pragma: no cover - farm branch returns above
                    raise RuntimeError("rollout collector was not initialized")
                result = collector.collect(
                    environments,
                    rollout_state=rollout_state,
                    reset_mask=rollout_reset_mask,
                    episode_counts=episode_counts,
                    step_callback=rollout_progress,
                    decision_callback=(
                        diagnostic_decision_callback if diagnostic_enabled else None
                    ),
                )
            rollout_wall_seconds = perf_counter() - rollout_started
            rollout_state = result.next_rollout_state
            rollout_reset_mask = result.next_reset_mask
            episode_counts = result.episode_counts
            update_started = perf_counter()
            prepared_batch = (
                learner.prepare_batch(result.learner_batch)
                if diagnostic_enabled
                else result.learner_batch
            )
            diagnostic_update: dict[str, object] = {}
            update_state_before_ppo = (
                learner.checkpoint_state()
                if effective_config.max_update_approx_kl is not None
                else None
            )
            update_start_count = int(learner.update_count)
            if diagnostic_enabled:
                from .diagnostics import explained_variance, ppo_ratio_diagnostics

                trajectory = prepared_batch.trajectory
                advantages = trajectory.advantages.detach()
                returns_tensor = trajectory.returns.detach()
                values_tensor = trajectory.values.detach()
                for row in current_diagnostic_rows:
                    lane = int(row.get("lane", 0))
                    timestep = int(row.get("decision", 0))
                    if lane >= advantages.shape[0] or timestep >= advantages.shape[1]:
                        continue
                    row["return"] = float(returns_tensor[lane, timestep].cpu().item())
                    row["advantage"] = float(advantages[lane, timestep].cpu().item())
                    row["critic_value_prediction"] = float(values_tensor[lane, timestep].cpu().item())
                    policy_row = row.get("policy")
                    if isinstance(policy_row, Mapping):
                        policy_copy = dict(policy_row)
                        policy_copy["return"] = row["return"]
                        policy_copy["advantage"] = row["advantage"]
                        policy_copy["critic_value_prediction"] = row["critic_value_prediction"]
                        row["policy"] = policy_copy
                diagnostic_update.update(
                    {
                        "advantage_distribution": {
                            "mean": float(advantages.mean().cpu().item()),
                            "std": float(advantages.std(unbiased=False).cpu().item()),
                            "min": float(advantages.min().cpu().item()),
                            "max": float(advantages.max().cpu().item()),
                            "positive_fraction": float((advantages > 0).float().mean().cpu().item()),
                        },
                        "return_distribution": {
                            "mean": float(returns_tensor.mean().cpu().item()),
                            "std": float(returns_tensor.std(unbiased=False).cpu().item()),
                            "min": float(returns_tensor.min().cpu().item()),
                            "max": float(returns_tensor.max().cpu().item()),
                        },
                        "explained_variance": explained_variance(values_tensor, returns_tensor),
                        "teacher_disagreement_rate": (
                            sum(row.get("actor_teacher_agreement") is False for row in current_diagnostic_rows)
                            / max(1, sum(row.get("actor_teacher_agreement") is not None for row in current_diagnostic_rows))
                        ),
                        "factor_entropy": {
                            "mode": _mean_trace_metric(current_diagnostic_rows, "mode"),
                            "card": _mean_trace_metric(current_diagnostic_rows, "card"),
                            "placement_selected_card": _mean_trace_metric(current_diagnostic_rows, "placement_selected_card"),
                            "joint": _mean_trace_metric(current_diagnostic_rows, "joint"),
                        },
                    }
                )
                action_distribution = {
                    "actor": _trace_action_distribution(
                        current_diagnostic_rows,
                        "actor_action",
                    ),
                    "teacher": _trace_action_distribution(
                        current_diagnostic_rows,
                        "strategic_teacher_action",
                    ),
                }
                diagnostic_update["action_distribution"] = action_distribution
                diagnostic_update["action_distribution_delta"] = {
                    stream: _trace_action_distribution_delta(
                        None
                        if previous_diagnostic_distributions is None
                        else previous_diagnostic_distributions.get(stream),
                        distribution,
                    )
                    for stream, distribution in action_distribution.items()
                }
                previous_diagnostic_distributions = action_distribution
            metrics = learner.update(
                prepared_batch,
                sequence_length=effective_config.sequence_length,
                diagnostics=diagnostic_enabled,
            )
            if diagnostic_enabled:
                from .diagnostics import ppo_ratio_diagnostics

                trajectory = prepared_batch.trajectory
                with torch.inference_mode():
                    post_update = learner.evaluate_sequence(
                        trajectory.sequence,
                        trajectory.actions,
                        trajectory.action_masks,
                        privileged_features=prepared_batch.privileged_features,
                    )
                ratios, clipped = ppo_ratio_diagnostics(
                    trajectory.old_log_probs,
                    post_update.log_probs,
                    trajectory.advantages,
                    learner.config.clip_epsilon,
                )
                diagnostic_update["post_update_ratio_distribution"] = {
                    "mean": float(ratios.mean().cpu().item()),
                    "std": float(ratios.std(unbiased=False).cpu().item()),
                    "min": float(ratios.min().cpu().item()),
                    "max": float(ratios.max().cpu().item()),
                    "clip_fraction": float(clipped.float().mean().cpu().item()),
                }
                for row in current_diagnostic_rows:
                    lane = int(row.get("lane", 0))
                    timestep = int(row.get("decision", 0))
                    if lane >= ratios.shape[0] or timestep >= ratios.shape[1]:
                        continue
                    row["ppo_probability_ratio"] = float(ratios[lane, timestep].cpu().item())
                    row["ppo_clipping_occurred"] = bool(clipped[lane, timestep].cpu().item())
                    policy_row = row.get("policy")
                    if isinstance(policy_row, Mapping):
                        policy_copy = dict(policy_row)
                        policy_copy["ppo_probability_ratio"] = row["ppo_probability_ratio"]
                        policy_copy["ppo_clipping_occurred"] = row["ppo_clipping_occurred"]
                        row["policy"] = policy_copy
                diagnostic_trace_rows.extend(current_diagnostic_rows)
                diagnostic_update["decisions"] = len(current_diagnostic_rows)
                diagnostic_update["per_head_gradient_norms"] = dict(metrics.per_head_gradient_norms)
            update_guard = _apply_update_approx_kl_guard(
                learner,
                metrics,
                max_update_approx_kl=effective_config.max_update_approx_kl,
                state_before_update=update_state_before_ppo,
                starting_update=update_start_count,
                enabled=not effective_config.imitation_only,
            )
            if diagnostic_enabled:
                diagnostic_update["update_guard"] = update_guard
            update_wall_seconds = perf_counter() - update_started
            if metrics.skipped_steps:
                raise PrototypeConfigurationError(
                    "refusing to save a checkpoint after non-finite optimizer "
                    f"minibatches (skipped_steps={metrics.skipped_steps}, "
                    f"update={metrics.update_index})"
                )
            # The recurrent state was produced by the pre-update parameters.
            # Do not feed that state into the next rollout after the optimizer
            # changes the policy; rebuilding from the next public observation
            # avoids a subtle cross-version hidden-state mismatch.
            rollout_state = None
            rollout_reset_mask = None
            stats = result.stats.as_dict()
            for name in aggregate:
                aggregate[name] += int(stats[name])
            update_rows.append(
                {
                    "update": int(learner.update_count),
                    "attempted_update": int(metrics.update_index),
                    "local_update": local_update + 1,
                    "transitions": effective_config.envs * effective_config.horizon,
                    "metrics": metrics.as_dict(),
                    "rollout": stats,
                    "update_guard": update_guard,
                    "timing": {
                        "rollout_wall_seconds": rollout_wall_seconds,
                        "learner_update_wall_seconds": update_wall_seconds,
                    },
                    **({"diagnostics": diagnostic_update} if diagnostic_enabled else {}),
                }
            )
            if diagnostic_enabled:
                diagnostic_update_rows.append(
                    {
                        "update": int(metrics.update_index),
                        **diagnostic_update,
                        "metrics": metrics.as_dict(),
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    local_update + 1,
                    (local_update + 1) * effective_config.envs * effective_config.horizon,
                )

        destination = Path(checkpoint_out)
        candidate = _temporary_checkpoint_path(destination)
        try:
            save_prototype_checkpoint(
                candidate,
                learner,
                effective_config,
                ruleset,
            )
            run_end_revision = code_revision()
            revision_drift = revision_changed(run_start_revision, run_end_revision)
            report = {
                "kind": "recurrent_public_ppo_prototype",
                "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
                "checkpoint_format": PROTOTYPE_CHECKPOINT_FORMAT,
                "code_revision": run_end_revision,
                "run_code_revision": run_start_revision,
                "revision_guard": {
                    "status": "drifted" if revision_drift else "stable",
                    "start": run_start_revision,
                    "end": run_end_revision,
                },
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "actor_privileged_inputs": False,
                "critic_privileged_inputs": bool(learner.uses_privileged_critic),
                "observation_contract": _checkpoint_metadata(
                    effective_config,
                    learner,
                    ruleset,
                )["actor_observation"],
                "reward_config": _reward_config_for(effective_config).as_dict(),
                "updates": len(update_rows),
                "starting_update": starting_update,
                "final_update": int(learner.update_count),
                "transitions": effective_config.envs * effective_config.horizon * effective_config.updates,
                "sequence_length": effective_config.sequence_length,
                "max_update_approx_kl": effective_config.max_update_approx_kl,
                "outcomes": {
                    **aggregate,
                    "episode_boundaries": (
                        aggregate["completed_matches"] + aggregate["truncated_matches"]
                    ),
                },
                "update_rows": update_rows,
                "training_diagnostics": diagnostic_update_rows,
                "diagnostic_trace_out": (
                    None
                    if effective_config.diagnostic_trace_out is None
                    else str(effective_config.diagnostic_trace_out)
                ),
                "rollout_farm_timing": rollout_farm_timing,
                "policy_compile": policy_compile,
                "overlap_rollouts": bool(effective_config.overlap_rollouts),
                "checkpoint": str(destination),
                "resumed_from": resumed_from,
                "expert_guidance": bool(expert_guidance),
                "imitation_only": bool(effective_config.imitation_only),
                "expert_execution_probability": float(
                    effective_config.expert_execution_probability
                ),
                "actor_controls_actions": not (
                    expert_guidance
                    and effective_config.expert_execution_probability > 0.0
                ),
                "target_player": effective_config.target_player,
                "actor_player": effective_config.target_player,
                "opponent_player": 1 - effective_config.target_player,
                "player_deck": list(resolved_player_deck),
                "opponent_decks": [list(deck) for deck in resolved_opponent_decks],
                "custom_opponent_policy": opponent_action is not None,
                "opponent_uses_public_observation": bool(
                    resolved_opponent_uses_public_observation
                ),
                "rollout_opponent_specs": (
                    None
                    if rollout_opponent_specs is None
                    else [
                        {"strategy": strategy, "seed": int(seed)}
                        for strategy, seed in rollout_opponent_specs
                    ]
                ),
                "basic_scenarios": {
                    "enabled": bool(
                        resolved_basic_scenario_sources is not None
                        and any(
                            source is not None
                            for source in resolved_basic_scenario_sources
                        )
                    ),
                    "decision_limit": basic_scenario_decisions,
                    "lane_sources": (
                        None
                        if resolved_basic_scenario_sources is None
                        else list(resolved_basic_scenario_sources)
                    ),
                    "lane_audits": [
                        environment.scenario_audit()
                        for environment in environments
                        if callable(getattr(environment, "scenario_audit", None))
                    ],
                    "actor_controls_actions": not (
                        expert_guidance
                        and effective_config.expert_execution_probability > 0.0
                    ),
                    "success_definition": "resulting-game-state",
                },
                "resume_controls": {
                    "learning_rate": (
                        None
                        if resume_learning_rate is None
                        else float(resume_learning_rate)
                    ),
                    "belief_loss_disabled": bool(resume_disable_belief_loss),
                    "optimizer_reset": bool(resume_reset_optimizer),
                },
                # Filled from one elapsed-time sample after audit, checkpoint
                # promotion, report construction, and runtime teardown.
                "wall_seconds": 0.0,
                "decisions_per_second": 0.0,
                "throughput_scope": (
                    "successful train_prototype call from validation and checkpoint/model "
                    "setup through rollout, GAE/PPO updates, checkpoint save, exploit audit, "
                    "checkpoint promotion, optional diagnostic-report write, worker/vector "
                    "teardown, and JSON report validation"
                ),
                "throughput_exclusions": (
                    "caller/CLI stdout serialization and --json-out filesystem write performed "
                    "after train_prototype returns"
                ),
                "warning": (
                    "Prototype training uses a provisional deterministic simulator unless "
                    "the selected ruleset reports training_ready=true."
                ),
            }
            from .exploit_audit import audit_simulation_report

            audit = audit_simulation_report(report)
            report["simulation_exploit_audit"] = audit
            # Check again immediately before promotion so a commit made while
            # the audit was running cannot slip through the revision gate.
            promotion_revision = code_revision()
            if revision_changed(run_start_revision, promotion_revision):
                revision_drift = True
                report["code_revision"] = promotion_revision
                report["revision_guard"] = {
                    "status": "drifted",
                    "start": run_start_revision,
                    "end": promotion_revision,
                }
            quarantine_reasons: list[str] = []
            if revision_drift:
                quarantine_reasons.append("code_revision_changed_during_run")
            if audit.get("status") != "clean":
                quarantine_reasons.append("simulation_exploit_audit_not_clean")
            if quarantine_reasons:
                quarantine = _quarantine_checkpoint_path(candidate)
                report["checkpoint_promotion"] = {
                    "status": "quarantined",
                    "destination": str(destination),
                    "quarantined_checkpoint": str(quarantine),
                    "reasons": quarantine_reasons,
                }
                report["quarantined_checkpoint"] = str(quarantine)
            else:
                os.replace(candidate, destination)
                report["checkpoint_promotion"] = {
                    "status": "promoted",
                    "destination": str(destination),
                }
            if diagnostic_enabled:
                diagnostic_path = Path(effective_config.diagnostic_trace_out)  # type: ignore[arg-type]
                _write_json(
                    diagnostic_path,
                    {
                        "kind": "recurrent_public_ppo_prototype_training_trace",
                        "trace_schema_version": 1,
                        "checkpoint": str(destination),
                        "code_revision": report["code_revision"],
                        "ruleset_id": ruleset.ruleset_id,
                        "ruleset_hash": ruleset.content_hash,
                        "actor_controls_actions": report["actor_controls_actions"],
                        "actor_privileged_inputs": False,
                        "critic_privileged_inputs": report["critic_privileged_inputs"],
                        "updates": diagnostic_update_rows,
                        "decisions": diagnostic_trace_rows,
                        "warning": "state snapshots are privileged diagnostic data and are never actor inputs",
                    },
                )
                report["diagnostic_trace_out"] = str(diagnostic_path)
        finally:
            # A candidate is either promoted, renamed to quarantine, or safely
            # removed after an unexpected save/audit error.  In particular,
            # the previous destination remains untouched until the audit is
            # clean.
            candidate.unlink(missing_ok=True)
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True)
        if vector_environment is not None:
            vector_environment.close()
        if rollout_farm is not None:
            rollout_farm.close()
    # Validate the same report shape that the CLI serializes before closing
    # the benchmark window.  Timing fields use placeholders here to avoid the
    # self-referential impossibility of timing the serialization of their own
    # final values.
    json.dumps(report, allow_nan=False)
    wall_seconds = perf_counter() - started
    transitions = int(report["transitions"])
    decisions_per_second = transitions / max(wall_seconds, 1e-9)
    report["wall_seconds"] = wall_seconds
    report["decisions_per_second"] = decisions_per_second
    return report


def _full_match_decisions(ruleset: Any, decision_interval_us: int) -> int:
    duration = int(ruleset.match.regulation_us + ruleset.match.overtime_us)
    return max(1, ceil(duration / decision_interval_us))


def _trace_action(action: Any, *, player: int) -> dict[str, object]:
    """Convert either a policy or simulator action to readable JSON fields."""

    if action is None:
        return {"player": player, "mode": "WAIT"}

    raw_kind = getattr(action, "kind", None)
    is_policy_action = hasattr(action, "card_idx")
    if raw_kind is None and hasattr(action, "card_slot"):
        raw_kind = "Play"
    kind = str(raw_kind or "Wait").strip().casefold().replace("_", "-")
    if kind in {"wait", "noop", "no-op"}:
        return {"player": player, "mode": "WAIT"}
    if kind != "play":
        record: dict[str, object] = {"player": player, "mode": kind.upper()}
        entity_uid = getattr(action, "entity_uid", None)
        if entity_uid is not None:
            record["entity_uid"] = int(entity_uid)
        return record

    raw_slot = getattr(action, "card_idx", None)
    if raw_slot is None:
        raw_slot = getattr(action, "card_slot", None)
    record = {
        "player": player,
        "mode": "PLAY",
        "card_slot": int(raw_slot) if raw_slot is not None else None,
    }
    raw_cell = getattr(action, "cell", None)
    if raw_cell is not None:
        cell = (int(raw_cell[0]), int(raw_cell[1]))
        try:
            try:
                from ..geometry import mirror_cell
            except ImportError:  # pragma: no cover - top-level ``rl`` layout
                from simulator.geometry import mirror_cell

            mirrored = mirror_cell(cell)
        except (ImportError, ValueError):  # pragma: no cover - defensive path
            mirrored = cell
        if is_policy_action:
            policy_cell = cell
            world_cell = mirrored if player == 1 else cell
        else:
            world_cell = cell
            policy_cell = mirrored if player == 1 else cell
        record["policy_cell"] = list(policy_cell)
        record["world_cell"] = list(world_cell)
    return record


def _trace_action_events(info: Mapping[str, object]) -> list[dict[str, object]]:
    """Keep only simulator events that describe either player's action."""

    raw_events = info.get("events", ())
    if not isinstance(raw_events, (tuple, list)):
        return []
    action_kinds = {"card_played", "action_rejected", "card_mirrored"}
    selected: list[dict[str, object]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        kind = raw_event.get("kind")
        data = raw_event.get("data")
        if kind not in action_kinds or not isinstance(data, Mapping):
            continue
        if data.get("player") not in (0, 1):
            continue
        selected.append(
            {
                "tick": raw_event.get("tick"),
                "sequence": raw_event.get("sequence"),
                "kind": kind,
                "data": dict(data),
            }
        )
    return selected


def _state_entities(raw_state: object) -> tuple[object, ...]:
    if isinstance(raw_state, Mapping):
        raw_entities = raw_state.get("entities", ())
        return tuple(raw_entities) if isinstance(raw_entities, (tuple, list)) else ()
    entities = getattr(raw_state, "entities", {})
    return tuple(entities.values()) if isinstance(entities, Mapping) else ()


def _state_field(entity: object, name: str, default: object = None) -> object:
    if isinstance(entity, Mapping):
        return entity.get(name, default)
    return getattr(entity, name, default)


def _state_player(raw_state: object, player: int) -> object | None:
    """Return one player row from either a live or primitive state."""

    if isinstance(raw_state, Mapping):
        players = raw_state.get("players", ())
    else:
        players = getattr(raw_state, "players", ())
    if not isinstance(players, (tuple, list)) or not 0 <= player < len(players):
        return None
    return players[player]


def _tower_hp_snapshot(raw_state: object) -> dict[str, dict[str, dict[str, int]]]:
    """Extract tower HP from the post-step diagnostic state."""

    towers: dict[str, dict[str, dict[str, int]]] = {}
    for raw_entity in _state_entities(raw_state):
        if _state_field(raw_entity, "kind") != "tower":
            continue
        owner = _state_field(raw_entity, "owner")
        role = _state_field(raw_entity, "role")
        hp = _state_field(raw_entity, "hp")
        maximum = _state_field(raw_entity, "max_hp")
        if (
            type(owner) is not int
            or not isinstance(role, str)
            or type(hp) is not int
            or type(maximum) is not int
        ):
            continue
        towers.setdefault(f"player_{owner}", {})[role] = {
            "hp": hp,
            "max_hp": maximum,
        }
    return towers


def _troop_locations(raw_state: object) -> list[dict[str, object]]:
    """Extract alive troops with exact world and grid coordinates."""

    try:
        try:
            from ..geometry import position_to_cell
        except ImportError:  # pragma: no cover - top-level ``rl`` layout
            from simulator.geometry import position_to_cell
    except ImportError:  # pragma: no cover - defensive path
        position_to_cell = None

    locations: list[dict[str, object]] = []
    for raw_entity in _state_entities(raw_state):
        if (
            _state_field(raw_entity, "kind") != "troop"
            or not _state_field(raw_entity, "alive", False)
        ):
            continue
        uid = _state_field(raw_entity, "uid")
        owner = _state_field(raw_entity, "owner")
        card_id = _state_field(raw_entity, "card_id")
        kind = _state_field(raw_entity, "kind")
        x_mtile = _state_field(raw_entity, "x_mtile")
        y_mtile = _state_field(raw_entity, "y_mtile")
        hp = _state_field(raw_entity, "hp")
        maximum = _state_field(raw_entity, "max_hp")
        if (
            type(uid) is not int
            or type(owner) is not int
            or not isinstance(card_id, str)
            or not isinstance(kind, str)
            or type(x_mtile) is not int
            or type(y_mtile) is not int
            or type(hp) is not int
            or type(maximum) is not int
            or hp <= 0
        ):
            continue
        cell = (
            position_to_cell(x_mtile, y_mtile)
            if position_to_cell is not None
            else None
        )
        deploy_remaining_us = _state_field(raw_entity, "deploy_remaining_us")
        locations.append(
            {
                "uid": uid,
                "owner": owner,
                "card_id": card_id,
                "x_mtile": x_mtile,
                "y_mtile": y_mtile,
                "world_cell": list(cell) if cell is not None else None,
                "hp": hp,
                "max_hp": maximum,
                "deploy_remaining_us": (
                    int(deploy_remaining_us)
                    if type(deploy_remaining_us) is int
                    else None
                ),
                "alive": True,
            }
        )
    locations.sort(key=lambda item: int(item["uid"]))
    return locations


def _trace_decision(
    record: Any,
    *,
    target_player: int,
    state_before: object | None = None,
    include_positions: bool = True,
) -> dict[str, object]:
    """Build one JSON-safe evaluation row from a collector decision record.

    ``state_before`` is optional because the collector deliberately exposes
    only the post-step state in its diagnostic callback.  The prototype trace
    supplies a caller-owned snapshot, while lightweight matrix audits can
    omit the expensive position snapshots and still retain action/event
    reconciliation.
    """

    from .diagnostics import state_snapshot

    # The collector's snapshot is authoritative for diagnostic runs.  The
    # explicit argument remains a compatibility fallback for the older trace
    # caller that owns its own before-state copy.
    diagnostic_state_before = getattr(record, "state_before", None)
    if diagnostic_state_before is None:
        diagnostic_state_before = state_before
    policy_diagnostics = getattr(record, "policy_diagnostics", None)
    if not isinstance(policy_diagnostics, Mapping):
        policy_diagnostics = {}
    action = _trace_action(record.target_action, player=target_player)
    opponent = _trace_action(record.opponent_action, player=1 - target_player)
    info = getattr(record.result, "info", {})
    if not isinstance(info, Mapping):
        info = {}
    events = _trace_action_events(info)
    slot = action.get("card_slot")
    selected_card = None
    if action.get("mode") == "PLAY" and type(slot) is int and 0 <= slot < len(record.hand_before):
        selected_card = record.hand_before[slot]
    opponent_hand_before = _state_field(
        _state_player(diagnostic_state_before, 1 - target_player),
        "hand",
        (),
    )
    if not isinstance(opponent_hand_before, (tuple, list)):
        opponent_hand_before = ()
    opponent_slot = opponent.get("card_slot")
    opponent_selected_card = None
    if (
        opponent.get("mode") == "PLAY"
        and type(opponent_slot) is int
        and 0 <= opponent_slot < len(opponent_hand_before)
    ):
        opponent_selected_card = opponent_hand_before[opponent_slot]

    target_events = [
        event
        for event in events
        if isinstance(event.get("data"), Mapping)
        and event["data"].get("player") == target_player
    ]
    opponent_events = [
        event
        for event in events
        if isinstance(event.get("data"), Mapping)
        and event["data"].get("player") == 1 - target_player
    ]
    played_target = next(
        (
            event["data"]
            for event in target_events
            if event.get("kind") == "card_played"
            and isinstance(event.get("data"), Mapping)
            and (slot is None or event["data"].get("card_slot") == slot)
        ),
        None,
    )
    played_opponent = next(
        (
            event["data"]
            for event in opponent_events
            if event.get("kind") == "card_played"
            and isinstance(event.get("data"), Mapping)
        ),
        None,
    )
    rejected_target = next(
        (
            event["data"]
            for event in target_events
            if event.get("kind") == "action_rejected"
            and isinstance(event.get("data"), Mapping)
        ),
        None,
    )
    rejected_opponent = next(
        (
            event["data"]
            for event in opponent_events
            if event.get("kind") == "action_rejected"
            and isinstance(event.get("data"), Mapping)
        ),
        None,
    )
    authoritative_state = info.get("authoritative_state")
    if authoritative_state is None:
        authoritative_state = record.state_after
    diagnostic_enabled = bool(policy_diagnostics) or diagnostic_state_before is not None or include_positions
    snapshot_before = state_snapshot(diagnostic_state_before) if diagnostic_enabled else None
    snapshot_after = state_snapshot(authoritative_state) if diagnostic_enabled else None
    target_after = _state_player(authoritative_state, target_player)
    hand_after = _state_field(target_after, "hand", ())
    if not isinstance(hand_after, (tuple, list)):
        hand_after = ()
    elixir_after = _state_field(target_after, "elixir_milli")
    target_played = isinstance(played_target, Mapping)
    opponent_played = isinstance(played_opponent, Mapping)
    def inferred_play_status(
        requested: Mapping[str, object],
        *,
        played: bool,
        hand_before: Sequence[object],
        hand_after: Sequence[object],
        elixir_before: object,
        elixir_after_value: object,
    ) -> bool | None:
        """Infer application when the environment did not transport events.

        Training environments intentionally keep the event stream out of the
        actor-facing result.  The previous trace treated every such play as
        rejected even when the authoritative state had consumed the card.
        Prefer an explicit event, then use the state transition as the
        diagnostic fallback.  ``None`` means that neither source was
        available; it is not an invalid-action finding.
        """

        if requested.get("mode") == "WAIT":
            return True
        if requested.get("mode") != "PLAY":
            return None
        if played:
            return True
        if list(hand_before) != list(hand_after):
            return True
        if type(elixir_before) is int and type(elixir_after_value) is int:
            # A legal play spends elixir.  Do not call a pure regeneration
            # step a play when the simulator did not expose a post-state.
            if int(elixir_after_value) < int(elixir_before):
                return True
            return False
        return None

    target_accepted = inferred_play_status(
        action,
        played=target_played,
        hand_before=record.hand_before,
        hand_after=hand_after,
        elixir_before=record.elixir_before,
        elixir_after_value=elixir_after,
    )
    opponent_player_after = _state_player(authoritative_state, 1 - target_player)
    opponent_hand_after = _state_field(opponent_player_after, "hand", ())
    if not isinstance(opponent_hand_after, (tuple, list)):
        opponent_hand_after = ()
    opponent_accepted = inferred_play_status(
        opponent,
        played=opponent_played,
        hand_before=(
            _state_field(
                _state_player(diagnostic_state_before, 1 - target_player),
                "hand",
                (),
            )
            or ()
        ),
        hand_after=opponent_hand_after,
        elixir_before=_state_field(
            _state_player(diagnostic_state_before, 1 - target_player),
            "elixir_milli",
        ),
        elixir_after_value=_state_field(opponent_player_after, "elixir_milli"),
    )

    def event_cell(event_data: object) -> list[int] | None:
        if not isinstance(event_data, Mapping):
            return None
        if type(event_data.get("col")) is not int or type(event_data.get("row")) is not int:
            return None
        return [int(event_data["col"]), int(event_data["row"])]

    target_event_cell = event_cell(played_target)
    opponent_event_cell = event_cell(played_opponent)
    target_application_evidence = (
        "event"
        if target_played
        else "state_transition"
        if target_accepted is True and action.get("mode") == "PLAY"
        else "wait"
        if action.get("mode") == "WAIT"
        else "not_applied"
        if target_accepted is False
        else "unknown"
    )
    opponent_application_evidence = (
        "event"
        if opponent_played
        else "state_transition"
        if opponent_accepted is True and opponent.get("mode") == "PLAY"
        else "wait"
        if opponent.get("mode") == "WAIT"
        else "not_applied"
        if opponent_accepted is False
        else "unknown"
    )
    # Vector/collector results intentionally omit the event stream.  The
    # before/after hand and elixir snapshots still prove that a card was
    # consumed, so retain the selected card/cell as the applied action while
    # exposing how that fact was established.  This keeps trajectory reports
    # useful without feeding authoritative diagnostics back into the actor.
    target_applied_card = (
        played_target.get("card_id")
        if isinstance(played_target, Mapping)
        else selected_card
        if target_accepted is True and action.get("mode") == "PLAY"
        else None
    )
    opponent_applied_card = (
        played_opponent.get("card_id")
        if isinstance(played_opponent, Mapping)
        else opponent_selected_card
        if opponent_accepted is True and opponent.get("mode") == "PLAY"
        else None
    )
    target_applied_cell = (
        target_event_cell
        if target_event_cell is not None
        else action.get("world_cell")
        if target_accepted is True and action.get("mode") == "PLAY"
        else None
    )
    opponent_applied_cell = (
        opponent_event_cell
        if opponent_event_cell is not None
        else opponent.get("world_cell")
        if opponent_accepted is True and opponent.get("mode") == "PLAY"
        else None
    )

    row: dict[str, object] = {
        "decision": int(record.decision_index),
        "physics_tick_before": record.physics_tick_before,
        "physics_tick_after": info.get("physics_tick"),
        "elapsed_us_before": record.elapsed_us_before,
        "elapsed_us_after": (
            (
                authoritative_state.get("elapsed_us")
                if isinstance(authoritative_state, Mapping)
                else getattr(authoritative_state, "elapsed_us", None)
            )
        ),
        "mode": action.get("mode"),
        "card_slot": slot,
        "card_id": selected_card,
        "played_card_id": target_applied_card,
        "policy_cell": action.get("policy_cell"),
        "world_cell": action.get("world_cell"),
        "played_world_cell": target_applied_cell,
        "application_evidence": target_application_evidence,
        "hand_before": list(record.hand_before),
        "hand_after": list(hand_after),
        "elixir_milli_before": record.elixir_before,
        "elixir_milli_after": (
            int(elixir_after) if type(elixir_after) is int else None
        ),
        "tower_hp_before": _tower_hp_snapshot(diagnostic_state_before) if include_positions else None,
        "tower_hp_after": _tower_hp_snapshot(authoritative_state),
        "troop_positions_before": (
            _troop_locations(diagnostic_state_before) if include_positions else None
        ),
        "troop_positions_after": (
            _troop_locations(authoritative_state) if include_positions else None
        ),
        "opponent_action": opponent,
        "opponent_card_id": opponent_applied_card,
        "opponent_policy_cell": opponent.get("policy_cell"),
        "opponent_world_cell": opponent.get("world_cell"),
        "opponent_played_world_cell": opponent_applied_cell,
        "opponent_application_evidence": opponent_application_evidence,
        "accepted": target_accepted,
        "action_status": (
            "accepted"
            if target_accepted is True
            else "rejected"
            if target_accepted is False
            else "unknown"
        ),
        "rejection_reason": (
            rejected_target.get("reason")
            if isinstance(rejected_target, Mapping)
            else None
        ),
        "opponent_accepted": opponent_accepted,
        "opponent_rejection_reason": (
            rejected_opponent.get("reason")
            if isinstance(rejected_opponent, Mapping)
            else None
        ),
        "state_hash_after": info.get("state_hash"),
        "event_log_hash_after": info.get("event_log_hash"),
        "reward": float(record.result.rewards[target_player]),
        "terminated": bool(record.result.terminated),
        "truncated": bool(record.result.truncated),
        "winner": info.get("winner"),
        "terminal_reason": info.get("terminal_reason"),
        "action_events": events,
        # These fields are populated only by the explicit diagnostic path.
        # ``state_before``/``state_after`` are privileged debugging snapshots;
        # they are never actor inputs.
        "target_player": target_player,
        "state_before": snapshot_before,
        "state_after": snapshot_after,
        "legal_action_mask": policy_diagnostics.get("legal_action_mask"),
        "actor_action": policy_diagnostics.get("actor_action"),
        "executed_action": policy_diagnostics.get("executed_action"),
        "chosen_action_probability": policy_diagnostics.get("chosen_action_probability"),
        "chosen_action_log_probability": policy_diagnostics.get("chosen_action_log_probability"),
        "top_mode_alternatives": policy_diagnostics.get("top_mode_alternatives", []),
        "top_card_alternatives": policy_diagnostics.get("top_card_alternatives", []),
        "top_placement_alternatives": policy_diagnostics.get("top_placement_alternatives", []),
        "top_alternative_actions": policy_diagnostics.get("top_alternative_actions", []),
        "factor_entropy": policy_diagnostics.get("factor_entropy"),
        "strategic_teacher_action": (
            policy_diagnostics.get("strategic_teacher_action")
            if policy_diagnostics.get("strategic_teacher_action") is not None
            else _trace_action(getattr(record, "teacher_action", None), player=target_player)
            if getattr(record, "teacher_action", None) is not None
            else None
        ),
        "actor_teacher_agreement": policy_diagnostics.get("actor_teacher_agreement"),
        "critic_value_prediction": policy_diagnostics.get("critic_value_prediction"),
        "old_log_probability": policy_diagnostics.get("old_log_probability"),
        "return": policy_diagnostics.get("return"),
        "advantage": policy_diagnostics.get("advantage"),
        "ppo_probability_ratio": policy_diagnostics.get("ppo_probability_ratio"),
        "ppo_clipping_occurred": policy_diagnostics.get("ppo_clipping_occurred"),
        "policy": dict(policy_diagnostics) if policy_diagnostics else None,
    }
    if "replay_hash" in info:
        row["replay_hash_after"] = info.get("replay_hash")
    from .diagnostics import classify_decision, tower_damage

    row["tower_damage_to_opponent"] = tower_damage(
        snapshot_before,
        snapshot_after,
        player=1 - target_player,
    )
    row["tower_damage_to_self"] = tower_damage(
        snapshot_before,
        snapshot_after,
        player=target_player,
    )
    row["suspicious_categories"] = classify_decision(row)
    return row


def _mean_trace_metric(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    """Average one factor entropy across diagnostic decision rows."""

    values: list[float] = []
    for row in rows:
        entropy = row.get("factor_entropy")
        if not isinstance(entropy, Mapping):
            policy = row.get("policy")
            entropy = policy.get("factor_entropy") if isinstance(policy, Mapping) else None
        value = entropy.get(key) if isinstance(entropy, Mapping) else None
        if isinstance(value, (int, float)) and isfinite(float(value)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _trace_action_distribution(
    rows: Sequence[Mapping[str, object]],
    action_key: str,
) -> dict[str, object]:
    """Summarize selected actions for one update's actor or teacher stream."""

    modes: Counter[str] = Counter()
    cards: Counter[str] = Counter()
    placements: Counter[str] = Counter()
    total = 0
    for row in rows:
        action = row.get(action_key)
        if not isinstance(action, Mapping):
            continue
        mode = action.get("mode")
        if not isinstance(mode, str):
            continue
        total += 1
        modes[mode] += 1
        if mode != "PLAY":
            continue
        slot = action.get("card_slot")
        hand = row.get("hand_before")
        card_id = (
            hand[slot]
            if isinstance(hand, (list, tuple))
            and type(slot) is int
            and 0 <= slot < len(hand)
            and isinstance(hand[slot], str)
            else None
        )
        card_key = card_id if card_id is not None else f"slot:{slot}"
        cards[card_key] += 1
        cell = action.get("world_cell")
        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            if all(type(value) is int for value in cell):
                placements[f"{cell[0]},{cell[1]}"] += 1

    def probabilities(counter: Counter[str]) -> dict[str, float]:
        denominator = max(1, total)
        return {
            key: counter[key] / denominator
            for key in sorted(counter)
        }

    return {
        "total": total,
        "mode_counts": dict(sorted(modes.items())),
        "mode_probabilities": probabilities(modes),
        "card_counts": dict(sorted(cards.items())),
        "card_probabilities": probabilities(cards),
        "top_placement_counts": dict(
            sorted(placements.items(), key=lambda item: (-item[1], item[0]))[:20]
        ),
    }


def _trace_action_distribution_delta(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> dict[str, dict[str, float]] | None:
    """Return probability changes against the preceding diagnostic update."""

    if previous is None:
        return None

    def delta(field: str) -> dict[str, float]:
        old = previous.get(field, {})
        new = current.get(field, {})
        old_map = old if isinstance(old, Mapping) else {}
        new_map = new if isinstance(new, Mapping) else {}
        keys = sorted(set(old_map) | set(new_map))
        return {
            str(key): float(new_map.get(key, 0.0)) - float(old_map.get(key, 0.0))
            for key in keys
            if float(new_map.get(key, 0.0)) != float(old_map.get(key, 0.0))
        }

    return {
        "mode_probability_delta": delta("mode_probabilities"),
        "card_probability_delta": delta("card_probabilities"),
    }


def _update_trace_action_summary(
    summary: dict[str, object],
    row: Mapping[str, object],
) -> None:
    """Accumulate readable card/placement counts from one target action row."""

    mode = row.get("mode")
    if mode == "WAIT":
        summary["waits"] = int(summary.get("waits", 0)) + 1
        return
    if mode != "PLAY":
        summary["other_actions"] = int(summary.get("other_actions", 0)) + 1
        return

    summary["plays"] = int(summary.get("plays", 0)) + 1
    requested_card = row.get("card_id")
    requested = summary.setdefault("requested_plays_by_card", {})
    if isinstance(requested, dict) and isinstance(requested_card, str):
        requested[requested_card] = int(requested.get(requested_card, 0)) + 1
    if row.get("accepted") is False:
        summary["rejected_actions"] = int(summary.get("rejected_actions", 0)) + 1
        return

    card = row.get("played_card_id") or requested_card
    played = summary.setdefault("plays_by_card", {})
    if isinstance(played, dict) and isinstance(card, str):
        played[card] = int(played.get(card, 0)) + 1
    cell = row.get("played_world_cell") or row.get("world_cell")
    if isinstance(cell, (list, tuple)) and len(cell) == 2 and all(
        type(value) is int for value in cell
    ):
        placements = summary.setdefault("placements_by_world_cell", {})
        if isinstance(placements, dict):
            key = f"{int(cell[0])},{int(cell[1])}"
            placements[key] = int(placements.get(key, 0)) + 1


def _episode_outcome(
    result: Any | None,
    info: Mapping[str, object],
    *,
    target_player: int,
) -> tuple[str, int | None, str | None]:
    """Normalize a collector lane into an auditable outcome tuple."""

    if result is not None and bool(result.terminated) and not bool(result.truncated):
        winner = info.get("winner")
        if winner == target_player:
            return "win", winner, info.get("terminal_reason")
        if winner == 1 - target_player:
            return "loss", winner, info.get("terminal_reason")
        return "draw", winner, info.get("terminal_reason")
    reason = info.get("terminal_reason")
    if not isinstance(reason, str) or not reason:
        reason = "evaluation_cap"
    return "truncated", info.get("winner"), reason


def _count_terminal_reasons(
    episode_results: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Count terminal/censoring reasons in deterministic episode order."""

    reasons: dict[str, int] = {}
    for episode in episode_results:
        raw_reason = episode.get("terminal_reason")
        reason = raw_reason if isinstance(raw_reason, str) and raw_reason else "<unspecified>"
        reasons[reason] = reasons.get(reason, 0) + 1
    return dict(sorted(reasons.items()))


def _evaluate_episode_worker(
    arguments: tuple[str, int, int | None, int, str],
) -> dict[str, object]:
    """Evaluate one match in an isolated CPU process.

    A match spends most of its time in Python-side simulator physics.  Running
    independent matches in separate processes lets those physics loops use
    different CPU cores.  The worker deliberately uses CPU for the tiny
    policy as well: launching one GPU process per match would serialize on the
    same ROCm device and generally be slower.
    """

    checkpoint, seed, max_decisions, episode_offset, policy_mode = arguments
    try:
        import torch

        # Prevent N worker processes from each claiming all host CPU threads.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:  # pragma: no cover - guarded by checkpoint loading
        pass
    return evaluate_prototype(
        checkpoint,
        episodes=1,
        seed=seed,
        max_decisions=max_decisions,
        device="cpu",
        batch_size=1,
        parallel_episodes=False,
        episode_offset=episode_offset,
        policy_mode=policy_mode,
    )


def _evaluate_parallel_episodes(
    checkpoint: str | Path,
    *,
    episodes: int,
    seed: int,
    max_decisions: int | None,
    device: str | None,
    policy_mode: str,
) -> dict[str, object]:
    """Evaluate independent complete matches concurrently on host CPUs."""

    # ``forkserver`` avoids inheriting a parent ROCm context.  ``spawn`` is a
    # portable fallback for platforms without forkserver support.
    try:
        context = multiprocessing.get_context("forkserver")
    except ValueError:  # pragma: no cover - non-Unix fallback
        context = multiprocessing.get_context("spawn")
    worker_count = min(episodes, 8)
    arguments = [
        (str(checkpoint), seed, max_decisions, episode, policy_mode)
        for episode in range(episodes)
    ]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        reports = list(executor.map(_evaluate_episode_worker, arguments))

    if not reports:  # pragma: no cover - episodes is validated positive
        raise PrototypeConfigurationError("parallel evaluation produced no reports")
    first = reports[0]
    completed = sum(int(report["completed"]) for report in reports)
    truncated = sum(int(report["truncated"]) for report in reports)
    wins = sum(int(report["wins"]) for report in reports)
    draws = sum(int(report["draws"]) for report in reports)
    losses = sum(int(report["losses"]) for report in reports)
    mean_return = sum(float(report["mean_return"]) for report in reports) / episodes
    mean_decisions = (
        sum(float(report["mean_decisions"]) for report in reports) / episodes
    )
    episode_results: list[object] = []
    for report in reports:
        raw_results = report.get("episode_results", ())
        if isinstance(raw_results, list):
            episode_results.extend(raw_results)
    terminal_reasons: dict[str, int] = {}
    for result in episode_results:
        if not isinstance(result, Mapping):
            continue
        reason = result.get("terminal_reason")
        key = reason if isinstance(reason, str) and reason else "<unspecified>"
        terminal_reasons[key] = terminal_reasons.get(key, 0) + 1
    return {
        "kind": first["kind"],
        "checkpoint": str(checkpoint),
        "code_revision": code_revision(),
        "policy_mode": policy_mode,
        "actor_controls_actions": policy_mode == "actor",
        "checkpoint_format": first["checkpoint_format"],
        "ruleset_id": first["ruleset_id"],
        "ruleset_hash": first["ruleset_hash"],
        "actor_privileged_inputs": False,
        "critic_privileged_inputs": bool(first["critic_privileged_inputs"]),
        "episodes": episodes,
        "completed": completed,
        "truncated": truncated,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / completed if completed else 0.0,
        "completion_rate": completed / episodes if episodes else 0.0,
        "truncation_rate": truncated / episodes if episodes else 0.0,
        "all_wins": episodes > 0 and wins == episodes,
        "all_completed_wins": episodes > 0 and completed == episodes and wins == episodes,
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "mean_return": mean_return,
        "mean_decisions": mean_decisions,
        "max_decisions": first["max_decisions"],
        "batch_size": 1,
        "parallel_episodes": True,
        "parallel_workers": worker_count,
        "worker_device": "cpu",
        "requested_device": device,
        "episode_results": episode_results,
        "warning": first.get("warning"),
    }


def _diagnostic_state_copy(state: Any) -> Any:
    """Copy only the authoritative state needed by an evaluation trace.

    ``BattleState`` retains the complete event history.  Deep-copying that
    history once per decision makes a long trace increasingly expensive even
    though the trace serializer needs only the current state fields.  The
    primitive snapshot keeps RNG, entities, towers, and player state while
    deliberately omitting the redundant event log; callers still retain the
    post-step event data transported in the step result.
    """

    if state is None:
        return None
    to_primitive = getattr(state, "to_primitive", None)
    if callable(to_primitive):
        try:
            return to_primitive(include_events=False)
        except TypeError:
            # Compatibility with a state implementation predating the
            # include_events keyword.  This fallback is still bounded to one
            # state object and preserves the old diagnostic behavior.
            pass
    return deepcopy(state)


def _evaluate_public_counter_fast(
    checkpoint: str | Path,
    *,
    episodes: int,
    seed: int,
    max_decisions: int | None,
    device: str | None,
    episode_offset: int,
    policy_mode: str = "public-counter",
) -> dict[str, object]:
    """Evaluate an explicit public policy without unused neural forwards."""

    learner, stored_config, metadata = load_prototype_checkpoint(
        checkpoint,
        device=device,
    )
    _engine, deterministic_controller_type, _version, _reward, _env_class, _hash, _schema, load_ruleset, _deck = (
        _simulator_modules()
    )
    ruleset = load_ruleset(stored_config.ruleset_id)
    decision_cap = max_decisions or _full_match_decisions(
        ruleset,
        stored_config.decision_interval_us,
    )
    config = PrototypeConfig.from_mapping(
        {
            **stored_config.as_dict(),
            "envs": 1,
            "horizon": decision_cap,
            "updates": 1,
            "seed": seed,
            "allow_provisional": True,
        }
    )
    from .public_counter import PublicCounterController, StrategicCounterController
    from .expert import DeterministicCounterController

    controller_type = {
        "public-counter": PublicCounterController,
        "strategic-counter": StrategicCounterController,
        "deterministic-counter": DeterministicCounterController,
    }[policy_mode]

    wins = draws = losses = truncated = 0
    returns: list[float] = []
    lengths: list[int] = []
    episode_results: list[dict[str, object]] = []
    for episode in range(episode_offset, episode_offset + episodes):
        environment = _make_environment(
            config,
            ruleset,
            episode,
            expose_privileged_info=True,
            include_replay_hashes=False,
        )
        observations = environment.observe_v2()
        counter = controller_type()
        opponent = deterministic_controller_type(lane="alternate")
        total_return = 0.0
        finished = False
        last_result: Any | None = None
        decision_count = 0
        action_summary: dict[str, object] = {
            "waits": 0,
            "plays": 0,
            "plays_by_card": {},
            "requested_plays_by_card": {},
            "placements_by_world_cell": {},
            "rejected_actions": 0,
        }
        for decision in range(decision_cap):
            state = environment.state
            if state is None:  # pragma: no cover - environment invariant
                raise PrototypeConfigurationError("evaluation environment lost its state")
            physics_tick_before = state.tick
            elapsed_us_before = state.elapsed_us
            hand_before = tuple(state.players[config.target_player].hand)
            elixir_before = int(state.players[config.target_player].elixir_milli)
            if policy_mode == "deterministic-counter":
                target_action = counter.choose_action(
                    environment.engine,
                    environment.state,
                    config.target_player,
                )
            else:
                target_action = counter.choose_action(
                    observations[config.target_player],
                    player=config.target_player,
                )
            opponent_action = opponent.choose_action(
                environment.engine,
                environment.state,
                1 - config.target_player,
            )
            actions = (
                (target_action, opponent_action)
                if config.target_player == 0
                else (opponent_action, target_action)
            )
            result = environment.step_v2(actions)
            last_result = result
            decision_count += 1
            audit = _trace_decision(
                SimpleNamespace(
                    decision_index=decision,
                    target_action=target_action,
                    opponent_action=opponent_action,
                    result=result,
                    state_after=environment.state,
                    physics_tick_before=physics_tick_before,
                    elapsed_us_before=elapsed_us_before,
                    hand_before=hand_before,
                    elixir_before=elixir_before,
                ),
                target_player=config.target_player,
                include_positions=False,
            )
            _update_trace_action_summary(action_summary, audit)
            total_return += float(result.rewards[config.target_player])
            observations = result.observations
            if result.terminated or result.truncated:
                finished = True
                break
        if not finished:
            truncated += 1
        info = last_result.info if last_result is not None else {}
        if not isinstance(info, Mapping):
            info = {}
        outcome, winner, terminal_reason = _episode_outcome(
            last_result,
            info,
            target_player=config.target_player,
        )
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        elif outcome == "draw":
            draws += 1
        else:
            # The loop already counted a cap-truncated match above only for
            # the no-result path; rely on the normalized outcome for the
            # usual runner-tick case.
            if finished:
                truncated += 1
        lengths.append(decision_count or decision_cap)
        returns.append(total_return)
        episode_results.append(
            {
                "episode": episode,
                "seed": _mix_seed(config.seed, episode, 0xE001),
                "player_deck": list(_deck),
                "opponent_deck": list(_deck),
                "target_player": config.target_player,
                "policy_mode": policy_mode,
                "actor_controls_actions": False,
                "opponent_controller": "deterministic-cycle",
                "outcome": outcome,
                "winner": winner,
                "terminal_reason": terminal_reason,
                "cap_reached": outcome == "truncated" and (decision_count or decision_cap) >= decision_cap,
                "return": float(total_return),
                "decisions": decision_count or decision_cap,
                "tower_hp_end": _tower_hp_snapshot(
                    getattr(environment, "state", None)
                ),
                "troop_positions_end": _troop_locations(
                    getattr(environment, "state", None)
                ),
                "action_summary": action_summary,
            }
        )

    completed = wins + draws + losses
    return {
        "kind": "recurrent_public_ppo_prototype_evaluation",
        "checkpoint": str(checkpoint),
        "policy_mode": policy_mode,
        "actor_controls_actions": False,
        "checkpoint_format": metadata["checkpoint_format"],
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.content_hash,
        "actor_privileged_inputs": False,
        "critic_privileged_inputs": bool(learner.uses_privileged_critic),
        "episodes": episodes,
        "completed": completed,
        "truncated": truncated,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / completed if completed else 0.0,
        "completion_rate": completed / episodes if episodes else 0.0,
        "truncation_rate": truncated / episodes if episodes else 0.0,
        "all_wins": episodes > 0 and wins == episodes,
        "all_completed_wins": episodes > 0 and completed == episodes and wins == episodes,
        "terminal_reasons": _count_terminal_reasons(episode_results),
        "mean_return": sum(returns) / len(returns) if returns else 0.0,
        "mean_decisions": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_decisions": decision_cap,
        "batch_size": 1,
        "parallel_episodes": False,
        "episode_results": episode_results,
        "warning": metadata.get("evaluation_warning"),
    }


def evaluate_prototype(
    checkpoint: str | Path,
    *,
    episodes: int = 1,
    seed: int = 10_000,
    max_decisions: int | None = None,
    device: str | None = "auto",
    trace_out: str | Path | None = None,
    batch_size: int | None = None,
    parallel_episodes: bool = False,
    episode_offset: int = 0,
    policy_mode: str = "actor",
) -> dict[str, object]:
    """Evaluate a checkpoint with deterministic actions and public observations.

    ``policy_mode='actor'`` measures the checkpoint's neural actor and is the
    default. The explicit ``public-counter`` mode uses the public-observation
    counter policy as the action source and provides a deterministic-opponent
    baseline while the actor is still improving. When ``trace_out`` is
    supplied, also write a JSON diagnostic trace. The trace is evaluation-only
    and does not change the actor inputs.
    """

    _positive_int("episodes", episodes)
    if max_decisions is not None:
        _positive_int("max_decisions", max_decisions)
    if batch_size is not None:
        _positive_int("batch_size", batch_size)
    if type(parallel_episodes) is not bool:
        raise PrototypeConfigurationError("parallel_episodes must be a boolean")
    if policy_mode not in _EVALUATION_POLICIES:
        raise PrototypeConfigurationError(
            f"policy_mode must be one of {sorted(_EVALUATION_POLICIES)}, got {policy_mode!r}"
        )
    _nonnegative_int("episode_offset", episode_offset)
    if parallel_episodes and trace_out is not None:
        # A trace needs the per-decision callback and authoritative snapshots,
        # so use the batched in-process path while retaining the requested
        # output instead of silently dropping diagnostics.
        parallel_episodes = False
    if parallel_episodes and episode_offset:
        raise PrototypeConfigurationError(
            "episode_offset cannot be combined with parallel episode evaluation"
        )
    # The public-counter path does not use the neural actor at all.  Keep it
    # on the lightweight in-process simulator path before considering process
    # workers; this also avoids paying worker-startup costs for the explicit
    # deployment/regression baseline.
    if policy_mode in {
        "public-counter",
        "strategic-counter",
        "deterministic-counter",
    } and trace_out is None:
        return _attach_exploit_audit(
            _evaluate_public_counter_fast(
                checkpoint,
                episodes=episodes,
                seed=seed,
                max_decisions=max_decisions,
                device=device,
                episode_offset=episode_offset,
                policy_mode=policy_mode,
            )
        )
    # Worker startup is noticeable for short smoke checks; batching remains
    # faster there.  For a normal full match (1,200 decisions in the pinned
    # ruleset), parallel physics is the better path.
    if parallel_episodes and episodes > 1 and _is_cpu_device_request(device) and (
        max_decisions is None or max_decisions >= 600
    ):
        return _attach_exploit_audit(
            _evaluate_parallel_episodes(
                checkpoint,
                episodes=episodes,
                seed=seed,
                max_decisions=max_decisions,
                device=device,
                policy_mode=policy_mode,
            )
        )
    run_start_revision = code_revision()
    learner, stored_config, metadata = load_prototype_checkpoint(
        checkpoint,
        device=device,
    )
    _engine, _controller, _version, _reward, _env_class, _hash, _schema, load_ruleset, _deck = (
        _simulator_modules()
    )
    ruleset = load_ruleset(stored_config.ruleset_id)
    decision_cap = max_decisions or _full_match_decisions(
        ruleset,
        stored_config.decision_interval_us,
    )
    config = PrototypeConfig.from_mapping(
        {
            **stored_config.as_dict(),
            "envs": 1,
            "horizon": decision_cap,
            "updates": 1,
            "seed": seed,
            "allow_provisional": True,
        }
    )
    learner.policy.eval()
    learner.critic.eval()
    wins = draws = losses = truncated = 0
    returns: list[float] = []
    lengths: list[int] = []
    trace_enabled = trace_out is not None
    trace_episodes: list[dict[str, object]] = []
    episode_results: list[dict[str, object]] = []
    failure_category_counts: dict[str, int] = {}
    _deck = _simulator_modules()[-1]
    # The actor already supports a batch dimension.  Eight lanes amortize the
    # recurrent policy call without changing simulator physics or action timing.
    requested_batch_size = batch_size or min(episodes, 8)
    evaluation_batch_size = min(episodes, requested_batch_size)
    for batch_start in range(0, episodes, evaluation_batch_size):
        episode_ids = list(
            range(
                episode_offset + batch_start,
                episode_offset + min(episodes, batch_start + evaluation_batch_size),
            )
        )
        batch_config = PrototypeConfig.from_mapping(
            {
                **config.as_dict(),
                "envs": len(episode_ids),
                "env_backend": "reference",
                "env_workers": None,
            }
        )
        environments = [
            _make_environment(
                batch_config,
                ruleset,
                episode,
                # Diagnostics stay in ``info`` and are never passed to the
                # actor tensors.  Keeping them enabled for all evaluation
                # modes makes card/placement counts auditable even without a
                # full trace file.
                expose_privileged_info=True,
                include_replay_hashes=trace_enabled,
            )
            for episode in episode_ids
        ]
        episode_rows: list[list[dict[str, object]]] = [
            [] for _ in episode_ids
        ]
        action_summaries: list[dict[str, object]] = [
            {
                "waits": 0,
                "plays": 0,
                "plays_by_card": {},
                "requested_plays_by_card": {},
                "placements_by_world_cell": {},
                "rejected_actions": 0,
                "other_actions": 0,
            }
            for _ in episode_ids
        ]
        decision_counts: list[int] = [0 for _ in episode_ids]
        last_infos: list[Mapping[str, object]] = [{} for _ in episode_ids]
        last_results: list[Any | None] = [None for _ in episode_ids]
        last_states: list[Any | None] = [None for _ in episode_ids]
        before_states: list[Any | None] = [
            _diagnostic_state_copy(getattr(environment, "state", None))
            if trace_enabled
            else None
            for environment in environments
        ]

        def on_decision(record: Any) -> None:
            lane = int(record.lane)
            decision_counts[lane] += 1
            row = _trace_decision(
                record,
                target_player=config.target_player,
                state_before=before_states[lane],
                include_positions=trace_enabled,
            )
            if trace_enabled:
                episode_rows[lane].append(row)
                before_states[lane] = _diagnostic_state_copy(record.state_after)
            last_results[lane] = record.result
            last_states[lane] = record.state_after
            info = getattr(record.result, "info", {})
            if isinstance(info, Mapping):
                last_infos[lane] = info
            _update_trace_action_summary(action_summaries[lane], row)

        vector_environment, batch_step = _make_batch_stepper(
            batch_config,
            environments,
        )
        try:
            # The collector freezes completed lanes so a short match cannot be
            # followed by an accidental second match in the same batch.
            collector = _make_collector(
                learner,
                batch_config,
                deterministic=True,
                stop=True,
                freeze_completed_lanes=True,
                expert_action=_evaluation_action_callback(policy_mode),
                # In actor trace mode this reference is computed separately
                # from expert execution and behavior-cloning storage.
                diagnostic_teacher_action=(
                    _evaluation_action_callback("strategic-counter")
                    if trace_enabled and policy_mode == "actor"
                    else None
                ),
                expert_execution_probability=(
                    1.0
                    if policy_mode in {
                        "public-counter",
                        "strategic-counter",
                        "deterministic-counter",
                    }
                    else None
                ),
                batch_step=batch_step,
                # The evaluation opponent is the simulator-side deterministic
                # controller.  It receives authoritative state through the
                # callback and does not need the unused public view.
                actor_only_observations=True,
                fast_deterministic=(policy_mode == "actor" and not trace_enabled),
                diagnostics=trace_enabled,
            )
            result = collector.collect(
                environments,
                # A lightweight callback also records exact per-episode
                # lengths and action counts when no trace file is requested.
                decision_callback=on_decision,
            )
        finally:
            if vector_environment is not None:
                vector_environment.close()

        batch_returns = (
            result.trajectory.rewards.detach().sum(dim=1).cpu().tolist()
        )
        returns.extend(float(value) for value in batch_returns)
        lengths.extend(
            count if count > 0 else result.trajectory.time_steps
            for count in decision_counts
        )
        for lane, episode in enumerate(episode_ids):
            lane_result = last_results[lane]
            lane_info = last_infos[lane]
            outcome, winner, terminal_reason = _episode_outcome(
                lane_result,
                lane_info,
                target_player=config.target_player,
            )
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            elif outcome == "draw":
                draws += 1
            else:
                truncated += 1
            decisions = decision_counts[lane] or result.trajectory.time_steps
            if trace_enabled:
                from .diagnostics import annotate_trace

                annotated_rows, categories = annotate_trace(
                    episode_rows[lane],
                    target_player=config.target_player,
                )
                # Evaluation has no PPO batch, so expose a clearly labelled
                # Monte-Carlo return-to-go and value residual instead of
                # pretending that evaluation generated a GAE target.
                discount = float(stored_config.gamma)
                return_to_go = 0.0
                for row in reversed(annotated_rows):
                    return_to_go = float(row.get("reward", 0.0)) + discount * return_to_go
                    row["return"] = return_to_go
                    value = row.get("critic_value_prediction")
                    row["advantage"] = (
                        return_to_go - float(value)
                        if isinstance(value, (int, float))
                        else None
                    )
                    row["advantage_kind"] = "monte_carlo_return_to_go_minus_critic"
                    row["ppo_probability_ratio"] = None
                    row["ppo_clipping_occurred"] = None
                    policy_row = row.get("policy")
                    if isinstance(policy_row, Mapping):
                        policy_copy = dict(policy_row)
                        policy_copy["return"] = row["return"]
                        policy_copy["advantage"] = row["advantage"]
                        policy_copy["advantage_kind"] = row["advantage_kind"]
                        policy_copy["ppo_probability_ratio"] = None
                        policy_copy["ppo_clipping_occurred"] = None
                        row["policy"] = policy_copy
                episode_rows[lane] = annotated_rows
                for category, count in categories.items():
                    failure_category_counts[category] = failure_category_counts.get(category, 0) + count
                scored_rows: list[dict[str, object]] = []
                for row in annotated_rows:
                    categories_for_row = row.get("suspicious_categories", ())
                    if not categories_for_row:
                        continue
                    score = (
                        1000.0 * float(row.get("tower_damage_to_self", 0))
                        + 10.0 * float(
                            row.get("tower_damage_to_opponent", 0) == 0
                            and row.get("mode") == "PLAY"
                        )
                        + float(len(categories_for_row))
                    )
                    scored_rows.append(
                        {
                            "decision": row.get("decision"),
                            "score": score,
                            "categories": categories_for_row,
                            "action": row.get("executed_action"),
                            "teacher_action": row.get("strategic_teacher_action"),
                            "tower_damage_to_self": row.get("tower_damage_to_self", 0),
                            "tower_damage_to_opponent": row.get("tower_damage_to_opponent", 0),
                            "advantage": row.get("advantage"),
                        }
                    )
                scored = sorted(
                    scored_rows,
                    key=lambda item: (float(item["score"]), -int(item["decision"])),
                    reverse=True,
                )
            else:
                scored = []
                categories = {}
            state_after = (
                lane_info.get("authoritative_state")
                if lane_info.get("authoritative_state") is not None
                else last_states[lane]
            )
            episode_report: dict[str, object] = {
                "episode": episode,
                "seed": _mix_seed(config.seed, episode, 0xE001),
                "player_deck": list(_deck),
                "opponent_deck": list(_deck),
                # Retain the old singular key as a compatibility alias while
                # making the two sides explicit for audits.
                "deck": list(_deck),
                "target_player": config.target_player,
                "policy_mode": policy_mode,
                "actor_controls_actions": policy_mode == "actor",
                "configured_opponent": stored_config.opponent,
                "opponent_controller": "deterministic-cycle",
                "outcome": outcome,
                "winner": winner,
                "terminal_reason": terminal_reason,
                "cap_reached": outcome == "truncated" and decisions >= decision_cap,
                "return": float(batch_returns[lane]),
                "decisions": decisions,
                "tower_hp_end": _tower_hp_snapshot(state_after),
                "troop_positions_end": _troop_locations(state_after),
                "action_summary": action_summaries[lane],
                "failure_categories": categories,
                "loss_report": {
                    "ranking": "immediate self tower damage, no opponent damage, then diagnostic category count",
                    "top_decisions": scored[:12],
                },
            }
            episode_results.append(episode_report)
            if trace_enabled:
                trace_episodes.append(
                    {
                        **episode_report,
                        "trace": episode_rows[lane],
                    }
                )
    completed = wins + draws + losses
    run_end_revision = code_revision()
    checkpoint_revision = metadata.get("code_revision")
    checkpoint_revision_mismatch = (
        isinstance(checkpoint_revision, Mapping)
        and revision_changed(checkpoint_revision, run_start_revision)
    )
    run_revision_drift = revision_changed(run_start_revision, run_end_revision)
    revision_guard = {
        "status": (
            "drifted"
            if checkpoint_revision_mismatch or run_revision_drift
            else "stable"
        ),
        "run_start": run_start_revision,
        "run_end": run_end_revision,
        "checkpoint": checkpoint_revision,
        "checkpoint_matches_run": (
            None if not isinstance(checkpoint_revision, Mapping)
            else not checkpoint_revision_mismatch
        ),
    }
    report: dict[str, object] = {
        "kind": "recurrent_public_ppo_prototype_evaluation",
        "checkpoint": str(checkpoint),
        "code_revision": run_end_revision,
        "run_code_revision": run_start_revision,
        "revision_guard": revision_guard,
        "policy_mode": policy_mode,
        "actor_controls_actions": policy_mode == "actor",
        "checkpoint_format": metadata["checkpoint_format"],
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_hash": ruleset.content_hash,
        "actor_privileged_inputs": False,
        "critic_privileged_inputs": bool(learner.uses_privileged_critic),
        "episodes": episodes,
        "completed": completed,
        "truncated": truncated,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / completed if completed else 0.0,
        "completion_rate": completed / episodes if episodes else 0.0,
        "truncation_rate": truncated / episodes if episodes else 0.0,
        "all_wins": episodes > 0 and wins == episodes,
        "all_completed_wins": episodes > 0 and completed == episodes and wins == episodes,
        "terminal_reasons": _count_terminal_reasons(episode_results),
        "mean_return": sum(returns) / len(returns) if returns else 0.0,
        "mean_decisions": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_decisions": decision_cap,
        "batch_size": evaluation_batch_size,
        "parallel_episodes": False,
        "episode_results": episode_results,
        "failure_categories": dict(sorted(failure_category_counts.items())),
        "warning": metadata.get("evaluation_warning"),
    }
    if trace_enabled:
        trace_path = Path(trace_out)  # type: ignore[arg-type]
        _write_json(
            trace_path,
            {
                "kind": "recurrent_public_ppo_prototype_evaluation_trace",
                "trace_schema_version": 2,
                "diagnostic_schema_version": 1,
                "checkpoint": str(checkpoint),
                "code_revision": code_revision(),
                "checkpoint_format": metadata["checkpoint_format"],
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "decision_interval_us": stored_config.decision_interval_us,
                "target_player": stored_config.target_player,
                "policy_mode": policy_mode,
                "actor_controls_actions": policy_mode == "actor",
                "position_schema": {
                    "version": "authoritative-world-v2",
                    "coordinate_order": "[column,row] for world_cell; [x,y] for mtiles",
                    "coordinate_units": "mtiles",
                    "mtile_scale": 1_000,
                    "arena": {
                        "width_mtile": int(ruleset.arena.width_mtile),
                        "height_mtile": int(ruleset.arena.height_mtile),
                    },
                    "snapshots": {
                        "before": "pre-action state at the start of each decision",
                        "after": "post-step state before any episode reset",
                    },
                    "visibility": "authoritative_diagnostic",
                },
                "action_schema": {
                    "version": "requested-vs-applied-v2",
                    "card_id": "selected card from hand_before",
                    "played_card_id": (
                        "card_played event card_id, or selected card when the "
                        "before/after state proves application; null when not applied or unknown"
                    ),
                    "world_cell": "requested authoritative [column,row]",
                    "played_world_cell": (
                        "accepted card_played event [column,row], or requested "
                        "cell when the before/after state proves application"
                    ),
                    "application_evidence": (
                        "event, state_transition, wait, not_applied, or unknown"
                    ),
                    "accepted": "WAIT or explicit/state-transition application evidence",
                },
                "decision_diagnostics": {
                    "probabilities": "masked actor distributions; top alternatives are complete actions",
                    "teacher": "strategic-counter label only; actor_controls_actions remains true",
                    "returns": "evaluation rows use Monte-Carlo return-to-go minus critic, not GAE",
                    "ppo": "not applicable in evaluation; training traces contain post-update ratios/clipping",
                    "failure_categories": "evidence-labelled heuristics, not gameplay rules",
                },
                "episodes": trace_episodes,
                "aggregate_failure_categories": dict(sorted(failure_category_counts.items())),
                "warning": metadata.get("evaluation_warning"),
            },
        )
        report["trace_out"] = str(trace_path)
    from .exploit_audit import audit_simulation_report

    report["simulation_exploit_audit"] = audit_simulation_report(
        report,
        trace=None if not trace_enabled else {"episodes": trace_episodes},
    )
    return report


def _attach_exploit_audit(
    report: dict[str, object],
    *,
    trace: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Attach the common simulator-exploitation audit to an early result."""

    from .exploit_audit import audit_simulation_report

    report["simulation_exploit_audit"] = audit_simulation_report(
        report,
        trace=trace,
    )
    return report


def _write_json(path: Path | None, value: Mapping[str, object]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True)
    if path is None:
        print(encoded)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl.prototype",
        description="public-observation recurrent PPO prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train and save a recurrent PPO prototype")
    train.add_argument("--updates", type=int, default=1)
    train.add_argument("--envs", type=int, default=2)
    train.add_argument("--horizon", type=int, default=128)
    train.add_argument("--decision-interval-us", type=int, default=250_000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Adam learning rate for a new run; when resuming, explicitly "
            "re-applies this value after loading the checkpoint"
        ),
    )
    train.add_argument(
        "--resume-learning-rate",
        type=float,
        default=None,
        help="override the Adam learning rate after loading a checkpoint",
    )
    train.add_argument("--update-epochs", type=int, default=3)
    train.add_argument("--sequence-minibatch-size", type=int, default=8)
    train.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help=(
            "temporal recurrent-PPO chunk length; must divide horizon; "
            "omit to train each rollout lane as one sequence"
        ),
    )
    train.add_argument("--gamma", type=float, default=0.9995)
    train.add_argument("--gae-lambda", type=float, default=0.98)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--belief-coef", type=float, default=0.05)
    train.add_argument(
        "--bc-coef",
        type=float,
        default=0.0,
        help="behavior-cloning weight for teacher-guided rollouts",
    )
    train.add_argument(
        "--bc-factor-coef",
        type=float,
        default=0.0,
        help="balanced mode/card/placement imitation loss for teacher-guided rollouts",
    )
    train.add_argument(
        "--bc-card-factor-weight",
        type=float,
        default=1.0,
        help=(
            "relative card-head weight inside the factor imitation loss; "
            "use only for diagnosed card-selection collapse"
        ),
    )
    train.add_argument(
        "--expert-guidance",
        action="store_true",
        help=(
            "collect actions from the deterministic Hog counter-expert while "
            "keeping the actor observation public-only"
        ),
    )
    train.add_argument(
        "--dense-reward",
        action="store_true",
        help=(
            "legacy unbounded tower-damage/crown shaping; prefer the bounded "
            "potential-reward-weight option"
        ),
    )
    train.add_argument(
        "--potential-reward-weight",
        type=float,
        default=0.1,
        help=(
            "temporary normalized tower/crown potential coefficient added to "
            "the terminal win objective (0 disables it)"
        ),
    )
    train.add_argument(
        "--max-update-approx-kl",
        type=float,
        default=None,
        help=(
            "roll back a PPO update when mean approximate KL exceeds this bound; "
            "disabled by default for the low-level prototype trainer"
        ),
    )
    train.add_argument(
        "--placement-max-grad-norm",
        type=float,
        default=None,
        help=(
            "targeted raw-gradient cap for placement/spatial parameters; "
            "leave unset unless placement-head regression evidence justifies it"
        ),
    )
    train.add_argument("--checkpoint", type=Path, help="resume a prototype checkpoint")
    train.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("outputs/simulator/training/recurrent-prototype.pt"),
    )
    train.add_argument(
        "--diagnostic-trace-out",
        type=Path,
        help=(
            "write every training decision, model alternative, GAE target, "
            "PPO ratio, and per-head update statistics as JSON"
        ),
    )
    train.add_argument(
        "--device",
        default="auto",
        help="policy device (default: auto; pass cpu to force host execution)",
    )
    train.add_argument(
        "--env-backend",
        choices=(
            "reference",
            "process",
            "packed-process",
            "persistent-process",
            "rollout-process",
        ),
        default="reference",
        help=(
            "simulator lane backend; persistent-process keeps worker engines alive "
            "and rollout-process runs the public collector inside persistent "
            "workers to remove per-decision state IPC"
        ),
    )
    train.add_argument(
        "--env-workers",
        type=int,
        default=None,
        help="worker processes for a process environment backend (default: one per lane)",
    )
    train.add_argument(
        "--overlap-rollouts",
        action="store_true",
        help=(
            "overlap the next rollout with PPO optimization on rollout-process; "
            "introduces an explicit one-update behavior-policy lag"
        ),
    )
    train.add_argument(
        "--compile-policy",
        action="store_true",
        help=(
            "compile the parent policy forward graph with torch.compile; "
            "startup is higher but long runs may optimize faster"
        ),
    )
    train.add_argument("--no-shuffle", action="store_true")
    train.add_argument("--no-privileged-critic", action="store_true")
    train.add_argument("--no-belief-targets", action="store_true")
    train.add_argument(
        "--resume-no-belief-loss",
        action="store_true",
        help=(
            "when resuming, disable the auxiliary belief loss and its target "
            "collection"
        ),
    )
    train.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help=(
            "when resuming, discard Adam moments and use the current learning "
            "rate"
        ),
    )
    train.add_argument("--decision-interval-jitter-ticks", type=int, default=0)
    train.add_argument("--action-latency-max-steps", type=int, default=0)
    train.add_argument("--entity-observation-noise-std", type=float, default=0.0)
    train.add_argument(
        "--explicit-hand-features",
        action="store_true",
        help=(
            "project each public one-hot card-table hand slot independently; "
            "the Transformer continues to process entities only"
        ),
    )
    train.add_argument(
        "--direct-public-action-features",
        action="store_true",
        help="feed public global elixir/hand features directly to the WAIT/PLAY gate",
    )
    train.add_argument(
        "--direct-public-card-features",
        action="store_true",
        help="feed public global elixir/hand features directly to card-slot selection",
    )
    train.add_argument(
        "--primary-public-card-features",
        action="store_true",
        help="use the direct public card head as the primary card-slot policy",
    )
    train.add_argument(
        "--contextual-public-card-features",
        action="store_true",
        help=(
            "include recurrent public-entity context in the direct card-slot head "
            "for state-dependent defense/pressure choices"
        ),
    )
    train.add_argument(
        "--current-encoded-action-features",
        action="store_true",
        help="add current public encoder features to GRU history for action decoding",
    )
    train.add_argument(
        "--direct-public-mask-features",
        action="store_true",
        help="feed public legality masks directly to the WAIT/PLAY gate",
    )
    train.add_argument(
        "--direct-public-context-features",
        action="store_true",
        help="use a nonlinear public hand/elixir/legality context for the WAIT/PLAY gate",
    )
    train.add_argument(
        "--direct-public-slot-card-features",
        action="store_true",
        help=(
            "score each public hand slot from its one-hot card identity; "
            "requires explicit hand features"
        ),
    )
    train.add_argument(
        "--spatial-placement-features",
        action="store_true",
        help=(
            "retain board-aligned raster features for card-conditioned placement; "
            "use for fresh strategic checkpoints"
        ),
    )
    train.add_argument(
        "--strategic-model",
        action="store_true",
        help=(
            "use the larger public recurrent actor with projected hand features "
            "and spatial placement features; required for a fresh mainline run"
        ),
    )
    train.add_argument(
        "--imitation-only",
        action="store_true",
        help=(
            "during expert guidance, optimize only the supervised teacher-action "
            "loss; use before PPO fine-tuning"
        ),
    )
    train.add_argument(
        "--expert-execution-probability",
        type=float,
        default=0.0,
        help=(
            "probability of executing the teacher action during explicit "
            "expert-guided collection (default: 0; PPO remains actor-controlled)"
        ),
    )
    train.add_argument(
        "--expert-label-on-threat-only",
        action="store_true",
        help=(
            "apply teacher labels only when the public observation shows a "
            "defensive threat; actor actions remain unchanged"
        ),
    )
    train.add_argument(
        "--expert-label-on-disagreement",
        action="store_true",
        help=(
            "apply teacher labels only when the actor's proposed action differs "
            "from the teacher; actor actions remain unchanged"
        ),
    )
    train.add_argument(
        "--deterministic-rollouts",
        action="store_true",
        help=(
            "use argmax actor actions during collection; useful for deterministic "
            "DAgger recovery-state training"
        ),
    )
    train.add_argument("--allow-provisional", action="store_true")
    train.add_argument("--json-out", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a prototype checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--seed", type=int, default=10_000)
    evaluate.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="first reproducible episode index; useful for replaying one failing seed",
    )
    evaluate.add_argument("--max-decisions", type=int)
    evaluate.add_argument(
        "--device",
        default="auto",
        help="inference device (default: auto; pass cpu to force host execution)",
    )
    evaluate.add_argument(
        "--policy",
        dest="policy_mode",
        choices=tuple(sorted(_EVALUATION_POLICIES)),
        default="actor",
        help=(
            "action source: actor evaluates the neural checkpoint; public-counter "
            "is the legacy public-only guard; strategic-counter is a stronger "
            "public-only warm-start audit (default: actor)"
        ),
    )
    evaluate.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="number of matches evaluated together (default: 8)",
    )
    evaluate.add_argument(
        "--trace-out",
        type=Path,
        help="write per-decision actions, simulator events, and final tower HP as JSON",
    )
    evaluate.add_argument(
        "--parallel-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "evaluate independent matches in concurrent CPU workers (default: enabled; "
            "automatically disabled when writing a trace)"
        ),
    )
    evaluate.add_argument("--json-out", type=Path)

    shadow = subparsers.add_parser(
        "shadow",
        help="run public-only actor inference over an MP4, replay cache, or live V4L2 stream",
    )
    shadow.add_argument("--checkpoint", type=Path, required=True)
    source = shadow.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="recorded MP4 input")
    source.add_argument("--replay-cache", type=Path, help="previously extracted cr_bot replay cache")
    source.add_argument(
        "--video-device",
        help="live V4L2 source path or numeric camera index; shadow mode sends no taps",
    )
    shadow.add_argument("--sample-interval", type=float, default=0.1)
    shadow.add_argument("--frame-stride", type=int, default=1)
    shadow.add_argument("--video-start-time", type=float)
    shadow.add_argument("--video-end-time", type=float)
    shadow.add_argument("--max-frames", type=int)
    shadow.add_argument("--max-seconds", type=float)
    shadow.add_argument("--no-normalize", action="store_true")
    shadow.add_argument(
        "--yolo-detections",
        action="store_true",
        help="use YOLO tower-health detections in the existing vision pipeline",
    )
    shadow.add_argument(
        "--alternative-rois",
        action="store_true",
        help="use the alternative bottom-HUD ROI profile for shifted recordings",
    )
    shadow.add_argument(
        "--device",
        default="auto",
        help="inference device (default: auto; pass cpu to force host execution)",
    )
    shadow.add_argument(
        "--allow-stale-ruleset",
        action="store_true",
        help=(
            "allow read-only shadow inference when the checkpoint has the same "
            "ruleset ID but an older content hash"
        ),
    )
    shadow.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "train":
            if args.learning_rate is not None and args.resume_learning_rate is not None:
                raise PrototypeConfigurationError(
                    "pass either --learning-rate or --resume-learning-rate, not both"
                )
            learning_rate = 3e-4 if args.learning_rate is None else args.learning_rate
            resume_learning_rate = args.resume_learning_rate
            if args.checkpoint is not None and args.learning_rate is not None:
                # Preserve the familiar --learning-rate spelling while making
                # its post-load behavior explicit and reliable.
                resume_learning_rate = args.learning_rate
            config = PrototypeConfig(
                envs=args.envs,
                horizon=args.horizon,
                updates=args.updates,
                decision_interval_us=args.decision_interval_us,
                seed=args.seed,
                learning_rate=learning_rate,
                update_epochs=args.update_epochs,
                sequence_minibatch_size=args.sequence_minibatch_size,
                sequence_length=args.sequence_length,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                entropy_coef=args.entropy_coef,
                belief_coef=args.belief_coef,
                behavior_cloning_coef=args.bc_coef,
                behavior_cloning_factor_coef=args.bc_factor_coef,
                behavior_cloning_card_factor_weight=args.bc_card_factor_weight,
                imitation_only=args.imitation_only,
                expert_execution_probability=args.expert_execution_probability,
                expert_label_on_threat_only=args.expert_label_on_threat_only,
                expert_label_on_disagreement=args.expert_label_on_disagreement,
                deterministic_rollouts=args.deterministic_rollouts,
                device=args.device,
                env_backend=args.env_backend,
                env_workers=args.env_workers,
                overlap_rollouts=args.overlap_rollouts,
                compile_policy=args.compile_policy,
                diagnostic_trace_out=(
                    None if args.diagnostic_trace_out is None else str(args.diagnostic_trace_out)
                ),
                shuffle_decks=not args.no_shuffle,
                use_privileged_critic=not args.no_privileged_critic,
                collect_belief_targets=not args.no_belief_targets,
                dense_reward=args.dense_reward,
                potential_reward_weight=args.potential_reward_weight,
                max_update_approx_kl=args.max_update_approx_kl,
                placement_max_grad_norm=args.placement_max_grad_norm,
                decision_interval_jitter_ticks=args.decision_interval_jitter_ticks,
                action_latency_max_steps=args.action_latency_max_steps,
                entity_observation_noise_std=args.entity_observation_noise_std,
                direct_public_action_features=args.direct_public_action_features,
                direct_public_card_features=args.direct_public_card_features,
                primary_public_card_features=args.primary_public_card_features,
                contextual_public_card_features=args.contextual_public_card_features,
                current_encoded_action_features=args.current_encoded_action_features,
                direct_public_mask_features=args.direct_public_mask_features,
                direct_public_context_features=args.direct_public_context_features,
                direct_public_slot_card_features=args.direct_public_slot_card_features,
                model_dim=128 if args.strategic_model else 32,
                encoder_dim=128 if args.strategic_model else 32,
                transformer_heads=4,
                transformer_layers=2 if args.strategic_model else 1,
                transformer_ff_dim=256 if args.strategic_model else 64,
                gru_hidden_dim=256 if args.strategic_model else 32,
                explicit_hand_features=(
                    True if args.strategic_model else args.explicit_hand_features
                ),
                spatial_placement_features=(
                    True if args.strategic_model else args.spatial_placement_features
                ),
                allow_provisional=args.allow_provisional,
            )
            progress = _TrainingProgress(
                config.updates,
                config.envs * config.horizon,
                stream=sys.stderr,
            )
            try:
                train_kwargs: dict[str, Any] = {
                    "checkpoint": args.checkpoint,
                    "checkpoint_out": args.checkpoint_out,
                    "progress_callback": progress.update,
                    "progress_step_callback": progress.advance,
                    "resume_learning_rate": resume_learning_rate,
                    "resume_disable_belief_loss": args.resume_no_belief_loss,
                    "resume_reset_optimizer": args.resume_reset_optimizer,
                }
                if args.expert_guidance:
                    train_kwargs["expert_guidance"] = True
                report = train_prototype(config, **train_kwargs)
            finally:
                progress.close()
        elif args.command == "evaluate":
            report = evaluate_prototype(
                args.checkpoint,
                episodes=args.episodes,
                seed=args.seed,
                episode_offset=args.episode_offset,
                max_decisions=args.max_decisions,
                device=args.device,
                trace_out=args.trace_out,
                batch_size=args.batch_size,
                parallel_episodes=args.parallel_episodes,
                policy_mode=args.policy_mode,
            )
        else:
            if args.alternative_rois:
                from cr_bot.domain.video_constants import activate_alternative_video_rois

                activate_alternative_video_rois()
            from .shadow import run_shadow_media

            report = run_shadow_media(
                args.checkpoint,
                video=args.video,
                replay_cache=args.replay_cache,
                video_device=args.video_device,
                sample_interval_s=args.sample_interval,
                frame_stride=args.frame_stride,
                video_start_time_s=args.video_start_time,
                video_end_time_s=args.video_end_time,
                normalize=not args.no_normalize,
                yolo_detections=args.yolo_detections,
                max_frames=args.max_frames,
                max_seconds=args.max_seconds,
                device=args.device,
                allow_stale_ruleset=args.allow_stale_ruleset,
            )
    except (PrototypeConfigurationError, TorchUnavailableError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    _write_json(args.json_out, report)
    audit = report.get("simulation_exploit_audit")
    return 0 if not isinstance(audit, Mapping) or audit.get("status") == "clean" else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "PRIVILEGED_FEATURE_DIM",
    "PROTOTYPE_CHECKPOINT_FORMAT",
    "PROTOTYPE_SCHEMA_VERSION",
    "PrototypeConfig",
    "PrototypeConfigurationError",
    "evaluate_prototype",
    "load_prototype_checkpoint",
    "load_shadow_prototype_checkpoint",
    "main",
    "save_prototype_checkpoint",
    "train_prototype",
]
