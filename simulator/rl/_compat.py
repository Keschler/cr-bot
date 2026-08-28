"""Optional PyTorch dependency helpers for the production RL package.

The simulator itself intentionally remains usable without PyTorch.  Importing
``rl`` therefore succeeds on a minimal installation, while importing the
torch-backed model modules produces a targeted error that explains how to
enable them.
"""

from __future__ import annotations

from typing import Any


class TorchUnavailableError(ImportError):
    """Raised when the torch-backed RL modules are requested without torch."""


try:
    import torch as _torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _torch = None
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


def require_torch() -> Any:
    """Return torch or raise an actionable optional-dependency error."""

    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "The simulator RL model requires PyTorch. Install the optional "
            "torch dependency before importing rl.model or rl.trajectory."
        )
    return _torch
