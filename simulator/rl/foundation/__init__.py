"""Shared RL model, rollout, and tensor foundations.

Modules are intentionally not imported eagerly.  Importing :mod:`rl` remains
safe without PyTorch, and the V4 entry points can load without the historical
training runtime.
"""

__all__ = [
    "_compat",
    "basic_scenarios",
    "opponent_pool",
    "public_counter",
    "inference_checkpoint",
    "seeding",
    "collector",
    "diagnostics",
    "domain_randomization",
    "learner",
    "model",
    "objectives",
    "provenance",
    "trajectory",
]
