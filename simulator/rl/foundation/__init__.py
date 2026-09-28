"""Shared V4 tensor, opponent-pool, and provenance utilities.

Submodules remain lazy so importing :mod:`simulator.rl` does not load PyTorch.
"""

__all__ = [
    "_compat",
    "objectives",
    "opponent_pool",
    "provenance",
    "trajectory",
]
