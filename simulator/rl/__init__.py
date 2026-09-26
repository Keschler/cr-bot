"""Lazy public exports for shared RL primitives and V4 training."""
from __future__ import annotations
import importlib
from .foundation._compat import TORCH_AVAILABLE, TorchUnavailableError, require_torch

_TORCH_EXPORTS = {'ActionBatch',
 'ActionMasks',
 'AutoregressiveLogits',
 'GRURecurrentCore',
 'HybridEncoder',
 'MaskedAutoregressivePolicy',
 'ModelConfig',
 'ModelConfigV4',
 'OpponentBeliefHeads',
 'OpponentBeliefLogits',
 'PPOObjectiveConfig',
 'PPOObjectiveResult',
 'PrivilegedCritic',
 'RecurrentHybridPolicy',
 'RecurrentPolicyOutput',
 'RecurrentSequence',
 'RecurrentV4Policy',
 'TrajectoryBatch',
 'V4ActionBatch',
 'V4ActionHead',
 'V4Logits',
 'V4SupervisedWeights',
 'WAIT_DURATIONS',
 'behavior_cloning_loss',
 'compute_gae',
 'count_parameters',
 'ppo_objective',
 'v4_supervised_loss'}

_LAZY_EXPORTS = {'DomainRandomizationConfig': ('foundation.domain_randomization', 'DomainRandomizationConfig'),
 'DomainRandomizationError': ('foundation.domain_randomization', 'DomainRandomizationError'),
 'DomainRandomizedEnv': ('foundation.domain_randomization', 'DomainRandomizedEnv'),
 'DomainVariantSampler': ('foundation.domain_randomization', 'DomainVariantSampler'),
 'SimulationVariant': ('foundation.domain_randomization', 'SimulationVariant'),
 'BASIC_MECHANICS_SOURCES': ('foundation.basic_scenarios', 'BASIC_MECHANICS_SOURCES'),
 'BASIC_SCENARIO_REWARD_VERSION': ('foundation.basic_scenarios', 'BASIC_SCENARIO_REWARD_VERSION'),
 'BASIC_SCENARIO_SCHEMA_VERSION': ('foundation.basic_scenarios', 'BASIC_SCENARIO_SCHEMA_VERSION'),
 'BasicMechanicsScenarioEnv': ('foundation.basic_scenarios', 'BasicMechanicsScenarioEnv'),
 'BasicScenarioConfig': ('foundation.basic_scenarios', 'BasicScenarioConfig'),
 'BasicScenarioError': ('foundation.basic_scenarios', 'BasicScenarioError'),
 'basic_scenario_source': ('foundation.basic_scenarios', 'basic_scenario_source'),
 'phase_one_rehearsal_source': ('foundation.basic_scenarios', 'phase_one_rehearsal_source'),
 'ARCHETYPE_NAMES': ('foundation.opponent_pool', 'ARCHETYPE_NAMES'),
 'OpponentDeckSpec': ('foundation.opponent_pool', 'OpponentDeckSpec'),
 'OpponentPool': ('foundation.opponent_pool', 'OpponentPool'),
 'OpponentPoolError': ('foundation.opponent_pool', 'OpponentPoolError'),
 'OpponentScenario': ('foundation.opponent_pool', 'OpponentScenario'),
 'make_opponent_controller': ('foundation.opponent_pool', 'make_opponent_controller'),
 'BeliefTargets': ('foundation.learner', 'BeliefTargets'),
 'LearnerBatch': ('foundation.learner', 'LearnerBatch'),
 'LearnerConfig': ('foundation.learner', 'LearnerConfig'),
 'PolicyEvaluation': ('foundation.learner', 'PolicyEvaluation'),
 'RecurrentPPOLearner': ('foundation.learner', 'RecurrentPPOLearner'),
 'RecurrentRolloutState': ('foundation.learner', 'RecurrentRolloutState'),
 'RecurrentValueHead': ('foundation.learner', 'RecurrentValueHead'),
 'RolloutStep': ('foundation.learner', 'RolloutStep'),
 'UpdateMetrics': ('foundation.learner', 'UpdateMetrics'),
 'iter_sequence_minibatches': ('foundation.learner', 'iter_sequence_minibatches'),
 'AutoregressiveLogits': ('foundation.model', 'AutoregressiveLogits'),
 'GRURecurrentCore': ('foundation.model', 'GRURecurrentCore'),
 'HybridEncoder': ('foundation.model', 'HybridEncoder'),
 'MaskedAutoregressivePolicy': ('foundation.model', 'MaskedAutoregressivePolicy'),
 'ModelConfig': ('foundation.model', 'ModelConfig'),
 'OpponentBeliefHeads': ('foundation.model', 'OpponentBeliefHeads'),
 'OpponentBeliefLogits': ('foundation.model', 'OpponentBeliefLogits'),
 'PrivilegedCritic': ('foundation.model', 'PrivilegedCritic'),
 'RecurrentHybridPolicy': ('foundation.model', 'RecurrentHybridPolicy'),
 'RecurrentPolicyOutput': ('foundation.model', 'RecurrentPolicyOutput'),
 'ActionBatch': ('foundation.trajectory', 'ActionBatch'),
 'ActionMasks': ('foundation.trajectory', 'ActionMasks'),
 'RecurrentSequence': ('foundation.trajectory', 'RecurrentSequence'),
 'TrajectoryBatch': ('foundation.trajectory', 'TrajectoryBatch'),
 'PPOObjectiveConfig': ('foundation.objectives', 'PPOObjectiveConfig'),
 'PPOObjectiveResult': ('foundation.objectives', 'PPOObjectiveResult'),
 'behavior_cloning_loss': ('foundation.objectives', 'behavior_cloning_loss'),
 'compute_gae': ('foundation.objectives', 'compute_gae'),
 'ppo_objective': ('foundation.objectives', 'ppo_objective'),
 'ModelConfigV4': ('v4.model_v4', 'ModelConfigV4'),
 'RecurrentV4Policy': ('v4.model_v4', 'RecurrentV4Policy'),
 'V4ActionBatch': ('v4.model_v4', 'V4ActionBatch'),
 'V4ActionHead': ('v4.model_v4', 'V4ActionHead'),
 'V4Logits': ('v4.model_v4', 'V4Logits'),
 'WAIT_DURATIONS': ('v4.model_v4', 'WAIT_DURATIONS'),
 'count_parameters': ('v4.model_v4', 'count_parameters'),
 'V4SupervisedWeights': ('v4.losses_v4', 'V4SupervisedWeights'),
 'v4_supervised_loss': ('v4.losses_v4', 'v4_supervised_loss')}

_LAZY_MODULES = {'foundation': 'foundation',
 'v4': 'v4',
 'basic_scenarios': 'foundation.basic_scenarios',
 'distillation': 'v4.distillation',
 'domain_randomization': 'foundation.domain_randomization',
 'frontier': 'v4.frontier',
 'frontier_eval': 'v4.frontier_eval',
 'frontier_ppo': 'v4.frontier_ppo',
 'learner': 'foundation.learner',
 'learning_curve': 'v4.learning_curve',
 'losses_v4': 'v4.losses_v4',
 'model': 'foundation.model',
 'model_v4': 'v4.model_v4',
 'objectives': 'foundation.objectives',
 'opponent_pool': 'foundation.opponent_pool',
 'provenance': 'foundation.provenance',
 'search_dataset': 'v4.search_dataset',
 'search_distillation': 'v4.search_distillation',
 'search_teacher': 'v4.search_teacher',
 'simulator_distillation': 'v4.simulator_distillation',
 'simulator_teacher': 'v4.simulator_teacher',
 'timing_curriculum': 'v4.timing_curriculum',
 'timing_teacher': 'v4.timing_teacher',
 'trajectory': 'foundation.trajectory',
 'v4_ppo': 'v4.v4_ppo'}

__all__ = ["TORCH_AVAILABLE", "TorchUnavailableError", "require_torch", *_LAZY_EXPORTS]

def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        if name in _TORCH_EXPORTS and not TORCH_AVAILABLE:
            raise TorchUnavailableError(
                "PyTorch is not installed; install the optional torch dependency "
                f"before using the rl export {name!r}."
            )
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(f"{__name__}.{module_name}")
        value = getattr(module, attribute_name)
        # Cache resolved exports so repeated package-level access has the same
        # identity and cost as the historical eager re-export.
        globals()[name] = value
        return value
    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{_LAZY_MODULES[name]}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery and help()."""

    return sorted(set(globals()) | set(__all__) | set(_LAZY_MODULES))
