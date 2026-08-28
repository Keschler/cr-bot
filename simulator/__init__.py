"""Deterministic Level-11 Clash Royale simulator.

The public surface intentionally separates the authoritative physics state
from the lossy vision-policy observation adapter.
"""

from .actions import PlayCardAction, UseAbilityAction, WaitAction
from .engine import BASE_HOG_CYCLE_DECK, ENGINE_VERSION, BattleEngine, DeterministicCycleController
from .ruleset import DEFAULT_RULESET_ID, FIXED_RULESET_ID, Ruleset, load_fixed_ruleset, load_ruleset
from .state import BattleState

__all__ = [
    "BASE_HOG_CYCLE_DECK",
    "BattleEngine",
    "BattleState",
    "DEFAULT_RULESET_ID",
    "FIXED_RULESET_ID",
    "DeterministicCycleController",
    "ENGINE_VERSION",
    "PlayCardAction",
    "Ruleset",
    "SimulatorEnv",
    "UseAbilityAction",
    "VectorSimulatorEnv",
    "WaitAction",
    "FactorizedPolicy",
    "PPOConfig",
    "PPOTrainer",
    "TrainingConfigurationError",
    "TrainingProfile",
    "TrainingProfileError",
    "PolicyObservationV2",
    "PublicObservationV2",
    "build_policy_observation_v2",
    "build_public_entity_rows",
    "load_ruleset",
    "load_fixed_ruleset",
    "validate_training_profile",
]


def __getattr__(name: str):
    """Keep NumPy and the vision feature stack lazy for core-only callers."""

    if name in {"SimulatorEnv", "VectorSimulatorEnv"}:
        from .env import SimulatorEnv, VectorSimulatorEnv

        return {"SimulatorEnv": SimulatorEnv, "VectorSimulatorEnv": VectorSimulatorEnv}[name]
    if name in {"FactorizedPolicy", "PPOConfig", "PPOTrainer", "TrainingConfigurationError"}:
        from .trainer import FactorizedPolicy, PPOConfig, PPOTrainer, TrainingConfigurationError

        return {
            "FactorizedPolicy": FactorizedPolicy,
            "PPOConfig": PPOConfig,
            "PPOTrainer": PPOTrainer,
            "TrainingConfigurationError": TrainingConfigurationError,
        }[name]
    if name in {"TrainingProfile", "TrainingProfileError", "validate_training_profile"}:
        from .training_profiles import TrainingProfile, TrainingProfileError, validate_training_profile

        return {
            "TrainingProfile": TrainingProfile,
            "TrainingProfileError": TrainingProfileError,
            "validate_training_profile": validate_training_profile,
        }[name]
    if name in {"PolicyObservationV2", "PublicObservationV2"}:
        from .observation_v2 import PolicyObservationV2, PublicObservationV2

        return {
            "PolicyObservationV2": PolicyObservationV2,
            "PublicObservationV2": PublicObservationV2,
        }[name]
    if name in {"build_policy_observation_v2", "build_public_entity_rows"}:
        from .observation_v2_adapter import (
            build_policy_observation_v2,
            build_public_entity_rows,
        )

        return {
            "build_policy_observation_v2": build_policy_observation_v2,
            "build_public_entity_rows": build_public_entity_rows,
        }[name]
    raise AttributeError(name)
