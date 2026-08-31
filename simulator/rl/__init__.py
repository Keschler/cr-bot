"""Optional PyTorch production RL model foundation.

The simulator does not require PyTorch.  ``import rl`` is safe without it and
exposes ``TORCH_AVAILABLE``.  When PyTorch is absent, importing a model class
raises :class:`TorchUnavailableError` with installation guidance; when it is
installed, the model and trajectory classes are re-exported here.
"""

from __future__ import annotations

from ._compat import TORCH_AVAILABLE, TorchUnavailableError, require_torch
from .curriculum import (
    BCDecision,
    BCTeacherConfidencePolicy,
    CurriculumConfigurationError,
    CurriculumPhase,
    CurriculumSchedule,
    STRATEGIC_CURRICULUM_SCHEMA_VERSION,
    StrategicCurriculum,
    StrategicCurriculumStage,
    default_strategic_curriculum,
)
from .collector import (
    BatchStepFn,
    CollectorConfig,
    OpponentActionFn,
    PrivilegedFeatureFn,
    RecurrentRolloutCollector,
    RolloutResult,
    RolloutStepCallback,
    RolloutStats,
)
from .domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizationError,
    DomainRandomizedEnv,
    DomainVariantSampler,
    SimulationVariant,
)
from .basic_scenarios import (
    BASIC_MECHANICS_SOURCES,
    BASIC_SCENARIO_REWARD_VERSION,
    BASIC_SCENARIO_SCHEMA_VERSION,
    BasicMechanicsScenarioEnv,
    BasicScenarioConfig,
    BasicScenarioError,
    basic_scenario_source,
    phase_one_rehearsal_source,
)
from .exploit_audit import (
    SIMULATION_EXPLOIT_AUDIT_KIND,
    SIMULATION_EXPLOIT_AUDIT_SCHEMA_VERSION,
    SimulationExploitAuditError,
    audit_json_file,
    audit_replay_hashes,
    audit_simulation_report,
)
from .league import (
    DeckConditionedOpponentScope,
    DeckSpec,
    HistoricalCheckpoint,
    LeagueConfig,
    LeagueConfigurationError,
    LeagueRunState,
    LeagueMatchRecord,
    LeagueOrchestrator,
    LeagueOutcome,
    LeaguePayoffBook,
    LeaguePayoffStats,
    LeagueRatingBook,
    LeagueSampler,
    LeagueSamplingError,
    OpponentSelection,
    PAYOFF_SCHEMA_VERSION,
    PFSPOpponentSampler,
    deterministic_seed,
)
from .evaluation_matrix import (
    EvaluationMatrixConfig,
    EvaluationMatrixError,
    EVALUATION_PROVENANCE_SCHEMA_VERSION,
    MatchResult,
    MatchSpec,
    OpponentDeckSpec as EvaluationOpponentDeckSpec,
    OpponentStrategySpec,
    evaluate_checkpoint_matrix,
    run_evaluation_matrix,
    write_evaluation_matrix_report,
)
from .opponent_pool import (
    ARCHETYPE_NAMES,
    OpponentDeckSpec,
    OpponentPool,
    OpponentPoolError,
    OpponentScenario,
    make_opponent_controller,
)
from .self_play import (
    PublicCheckpointController,
    SELF_PLAY_STRATEGY_PREFIX,
    SelfPlayConfigurationError,
    build_self_play_matrix_config,
    checkpoint_strategy,
    evaluate_against_checkpoints,
)
from .public_counter import (
    PublicCounterController,
    StrategicCounterController,
    public_counter_action,
    strategic_counter_action,
)
from .learner import (
    BeliefTargets,
    LearnerBatch,
    LearnerConfig,
    PolicyEvaluation,
    RecurrentPPOLearner,
    RecurrentRolloutState,
    RecurrentValueHead,
    RolloutStep,
    UpdateMetrics,
    iter_sequence_minibatches,
)

_TORCH_EXPORTS = {
    "ActionBatch",
    "ActionMasks",
    "AutoregressiveLogits",
    "GRURecurrentCore",
    "HybridEncoder",
    "MaskedAutoregressivePolicy",
    "ModelConfig",
    "OpponentBeliefHeads",
    "OpponentBeliefLogits",
    "PrivilegedCritic",
    "RecurrentHybridPolicy",
    "RecurrentPolicyOutput",
    "RecurrentSequence",
    "TrajectoryBatch",
    "PPOObjectiveConfig",
    "PPOObjectiveResult",
    "behavior_cloning_loss",
    "compute_gae",
    "ppo_objective",
}

_FOUNDATION_EXPORTS = {
    "BASIC_MECHANICS_SOURCES",
    "BASIC_SCENARIO_REWARD_VERSION",
    "BASIC_SCENARIO_SCHEMA_VERSION",
    "BasicMechanicsScenarioEnv",
    "BasicScenarioConfig",
    "BasicScenarioError",
    "basic_scenario_source",
    "phase_one_rehearsal_source",
    "BCDecision",
    "BCTeacherConfidencePolicy",
    "CurriculumConfigurationError",
    "CurriculumPhase",
    "CurriculumSchedule",
    "STRATEGIC_CURRICULUM_SCHEMA_VERSION",
    "StrategicCurriculum",
    "StrategicCurriculumStage",
    "default_strategic_curriculum",
    "DeckConditionedOpponentScope",
    "DeckSpec",
    "HistoricalCheckpoint",
    "LeagueConfig",
    "LeagueConfigurationError",
    "LeagueRunState",
    "LeagueMatchRecord",
    "LeagueOrchestrator",
    "LeagueOutcome",
    "LeaguePayoffBook",
    "LeaguePayoffStats",
    "LeagueRatingBook",
    "LeagueSampler",
    "LeagueSamplingError",
    "OpponentSelection",
    "PAYOFF_SCHEMA_VERSION",
    "PFSPOpponentSampler",
    "deterministic_seed",
    "ARCHETYPE_NAMES",
    "OpponentDeckSpec",
    "OpponentPool",
    "OpponentPoolError",
    "OpponentScenario",
    "make_opponent_controller",
    "PublicCheckpointController",
    "SELF_PLAY_STRATEGY_PREFIX",
    "SelfPlayConfigurationError",
    "build_self_play_matrix_config",
    "checkpoint_strategy",
    "evaluate_against_checkpoints",
    "PublicCounterController",
    "StrategicCounterController",
    "public_counter_action",
    "strategic_counter_action",
    "EvaluationMatrixConfig",
    "EvaluationMatrixError",
    "EVALUATION_PROVENANCE_SCHEMA_VERSION",
    "EvaluationOpponentDeckSpec",
    "OpponentStrategySpec",
    "MatchResult",
    "MatchSpec",
    "evaluate_checkpoint_matrix",
    "run_evaluation_matrix",
    "write_evaluation_matrix_report",
    "BeliefTargets",
    "LearnerBatch",
    "LearnerConfig",
    "PolicyEvaluation",
    "RecurrentPPOLearner",
    "RecurrentRolloutState",
    "RecurrentValueHead",
    "RolloutStep",
    "UpdateMetrics",
    "iter_sequence_minibatches",
    "CollectorConfig",
    "OpponentActionFn",
    "PrivilegedFeatureFn",
    "RecurrentRolloutCollector",
    "RolloutResult",
    "RolloutStats",
    "DomainRandomizationConfig",
    "DomainRandomizationError",
    "DomainRandomizedEnv",
    "DomainVariantSampler",
    "SimulationVariant",
    "SIMULATION_EXPLOIT_AUDIT_KIND",
    "SIMULATION_EXPLOIT_AUDIT_SCHEMA_VERSION",
    "SimulationExploitAuditError",
    "audit_json_file",
    "audit_replay_hashes",
    "audit_simulation_report",
}

_PROTOTYPE_EXPORTS = {
    "PRIVILEGED_FEATURE_DIM",
    "PROTOTYPE_CHECKPOINT_FORMAT",
    "PROTOTYPE_SCHEMA_VERSION",
    "PrototypeConfig",
    "PrototypeConfigurationError",
    "evaluate_prototype",
    "load_prototype_checkpoint",
    "save_prototype_checkpoint",
    "train_prototype",
}

_SHADOW_EXPORTS = {
    "SHADOW_RUNNER_VERSION",
    "SHADOW_SCHEMA_VERSION",
    "ShadowConfigurationError",
    "ShadowPolicyRunner",
    "ShadowPrediction",
    "run_shadow_media",
}

_GENERALIZED_EXPORTS = {
    "GENERALIZED_TRAINING_KIND",
    "GENERALIZED_TRAINING_SCHEMA_VERSION",
    "GeneralizedTrainingConfig",
    "GeneralizedTrainingError",
    "build_heldout_matrix_config",
    "evaluate_heldout_matrix",
    "make_scenario_opponent_action",
    "sample_training_scenarios",
    "train_generalized",
}

if TORCH_AVAILABLE:
    from .model import (
        AutoregressiveLogits,
        GRURecurrentCore,
        HybridEncoder,
        MaskedAutoregressivePolicy,
        ModelConfig,
        OpponentBeliefHeads,
        OpponentBeliefLogits,
        PrivilegedCritic,
        RecurrentHybridPolicy,
        RecurrentPolicyOutput,
    )
    from .trajectory import ActionBatch, ActionMasks, RecurrentSequence, TrajectoryBatch
    from .objectives import (
        PPOObjectiveConfig,
        PPOObjectiveResult,
        behavior_cloning_loss,
        compute_gae,
        ppo_objective,
    )

    __all__ = [
        "TORCH_AVAILABLE",
        "TorchUnavailableError",
        "require_torch",
        *_TORCH_EXPORTS,
        *_FOUNDATION_EXPORTS,
        *_PROTOTYPE_EXPORTS,
        *_SHADOW_EXPORTS,
        *_GENERALIZED_EXPORTS,
    ]
else:
    __all__ = [
        "TORCH_AVAILABLE",
        "TorchUnavailableError",
        "require_torch",
        *_TORCH_EXPORTS,
        *_FOUNDATION_EXPORTS,
        *_PROTOTYPE_EXPORTS,
        *_SHADOW_EXPORTS,
        *_GENERALIZED_EXPORTS,
    ]


def __getattr__(name: str):
    if name in _TORCH_EXPORTS and not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "PyTorch is not installed; install the optional torch dependency "
            f"before using rl.{name}."
        )
    if name in _PROTOTYPE_EXPORTS:
        from . import prototype

        return getattr(prototype, name)
    if name in _SHADOW_EXPORTS:
        from . import shadow

        return getattr(shadow, name)
    if name in _GENERALIZED_EXPORTS:
        from . import generalized

        return getattr(generalized, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
