"""Lazy public exports for the V4 policy and its shared RL primitives."""

from __future__ import annotations

import importlib

from .foundation._compat import TORCH_AVAILABLE, TorchUnavailableError, require_torch

_TORCH_EXPORTS = frozenset(
    {
        "ActionBatch",
        "ActionMasks",
        "ModelConfigV4",
        "PPOObjectiveConfig",
        "PPOObjectiveResult",
        "RecurrentSequence",
        "RecurrentV4Policy",
        "TrajectoryBatch",
        "V4ActionBatch",
        "V4ActionHead",
        "V4Logits",
        "V4SupervisedWeights",
        "WAIT_DURATIONS",
        "behavior_cloning_loss",
        "compute_gae",
        "count_parameters",
        "ppo_objective",
        "v4_supervised_loss",
    }
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ARCHETYPE_NAMES": ("foundation.opponent_pool", "ARCHETYPE_NAMES"),
    "OpponentDeckSpec": ("foundation.opponent_pool", "OpponentDeckSpec"),
    "OpponentPool": ("foundation.opponent_pool", "OpponentPool"),
    "OpponentPoolError": ("foundation.opponent_pool", "OpponentPoolError"),
    "OpponentScenario": ("foundation.opponent_pool", "OpponentScenario"),
    "make_opponent_controller": ("foundation.opponent_pool", "make_opponent_controller"),
    "ActionBatch": ("foundation.trajectory", "ActionBatch"),
    "ActionMasks": ("foundation.trajectory", "ActionMasks"),
    "RecurrentSequence": ("foundation.trajectory", "RecurrentSequence"),
    "TrajectoryBatch": ("foundation.trajectory", "TrajectoryBatch"),
    "PPOObjectiveConfig": ("foundation.objectives", "PPOObjectiveConfig"),
    "PPOObjectiveResult": ("foundation.objectives", "PPOObjectiveResult"),
    "behavior_cloning_loss": ("foundation.objectives", "behavior_cloning_loss"),
    "compute_gae": ("foundation.objectives", "compute_gae"),
    "ppo_objective": ("foundation.objectives", "ppo_objective"),
    "ModelConfigV4": ("v4.model_v4", "ModelConfigV4"),
    "RecurrentV4Policy": ("v4.model_v4", "RecurrentV4Policy"),
    "V4ActionBatch": ("v4.model_v4", "V4ActionBatch"),
    "V4ActionHead": ("v4.model_v4", "V4ActionHead"),
    "V4Logits": ("v4.model_v4", "V4Logits"),
    "WAIT_DURATIONS": ("v4.model_v4", "WAIT_DURATIONS"),
    "count_parameters": ("v4.model_v4", "count_parameters"),
    "V4SupervisedWeights": ("v4.supervised", "V4SupervisedWeights"),
    "v4_supervised_loss": ("v4.supervised", "v4_supervised_loss"),
}

_LAZY_MODULES = {
    "foundation": "foundation",
    "v4": "v4",
    "bootstrap": "v4.bootstrap",
    "frontier": "v4.frontier",
    "frontier_eval": "v4.frontier_eval",
    "frontier_ppo": "v4.frontier_ppo",
    "learning_curve": "v4.learning_curve",
    "model_v4": "v4.model_v4",
    "objectives": "foundation.objectives",
    "opponent_pool": "foundation.opponent_pool",
    "provenance": "foundation.provenance",
    "search_distillation": "v4.search_distillation",
    "simulator_teacher": "v4.simulator_teacher",
    "supervised": "v4.supervised",
    "trajectory": "foundation.trajectory",
    "v4_ppo": "v4.v4_ppo",
}

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
        globals()[name] = value
        return value
    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{_LAZY_MODULES[name]}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_MODULES))
