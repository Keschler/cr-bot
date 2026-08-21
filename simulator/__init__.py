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
    "load_ruleset",
    "load_fixed_ruleset",
]


def __getattr__(name: str):
    """Keep NumPy and the vision feature stack lazy for core-only callers."""

    if name in {"SimulatorEnv", "VectorSimulatorEnv"}:
        from .env import SimulatorEnv, VectorSimulatorEnv

        return {"SimulatorEnv": SimulatorEnv, "VectorSimulatorEnv": VectorSimulatorEnv}[name]
    raise AttributeError(name)
