from __future__ import annotations

from typing import Any


class FrameNotFoundError(LookupError):
    """A retained frame with the requested index does not exist (evicted)."""


class CorrectionUnprocessableError(ValueError):
    """Edits are well-formed but cannot be turned into an observation."""


class NoActorError(RuntimeError):
    """No live policy actor is available for re-evaluation."""


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    import math

    return result if math.isfinite(result) else None
