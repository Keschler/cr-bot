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
    "PolicyObservationV2",
    "PublicObservationV2",
    "build_policy_observation_v2",
    "build_public_entity_rows",
    "load_ruleset",
    "load_fixed_ruleset",
]


def __getattr__(name: str):
    """Keep NumPy and the vision feature stack lazy for core-only callers."""

    if name in {"SimulatorEnv", "VectorSimulatorEnv"}:
        from .env import SimulatorEnv, VectorSimulatorEnv

        return {"SimulatorEnv": SimulatorEnv, "VectorSimulatorEnv": VectorSimulatorEnv}[name]
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
