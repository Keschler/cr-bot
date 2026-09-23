"""Inference checkpoint compatibility for the deployed public actor.

Training and evaluation runners are retired; this reader preserves existing
public-actor artifacts and their ruleset/observation validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from ._compat import TORCH_AVAILABLE, TorchUnavailableError

PROTOTYPE_CHECKPOINT_FORMAT = "recurrent-public-ppo-prototype-v1"
PRIVILEGED_FEATURE_DIM = 23


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


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Stored checkpoint configuration, retained for artifact compatibility."""

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
        from ...engine import BattleEngine, DeterministicCycleController, ENGINE_VERSION
        from ...env import RewardConfig, SimulatorEnv
        from ...observation_v2 import (
            OBSERVATION_V2_CONTRACT_HASH,
            OBSERVATION_V2_SCHEMA_VERSION,
        )
        from ...ruleset import load_ruleset
        from ...roster import PLAYER_DECK
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
