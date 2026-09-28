from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


CHECKPOINT_GLOBS = (
    "outputs/v4/**/*.pt",
)


V4_CHECKPOINT_ENV = "CR_BOT_V4_CHECKPOINT"


def _default_checkpoint() -> str | None:
    """Return the explicitly configured V4 checkpoint, if one exists."""

    configured = os.environ.get(V4_CHECKPOINT_ENV, "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _list_checkpoints(limit: int = 20) -> tuple[list[dict[str, Any]], str | None]:
    """List V4 actor checkpoints and identify the configured default."""
    seen: dict[str, None] = {}
    candidates: list[Path] = []
    for pattern in CHECKPOINT_GLOBS:
        try:
            paths = sorted(REPO_ROOT.glob(pattern))
        except (NotImplementedError, ValueError):
            continue
        for path in paths:
            if not path.is_file() or path.suffix != ".pt":
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen[key] = None
            candidates.append(path)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    default = _default_checkpoint()
    items = [
        {
            "path": str(path),
            "name": path.name
            if path.parent == REPO_ROOT
            else str(path.relative_to(REPO_ROOT)),
            "default": str(path) == default,
        }
        for path in candidates
    ]
    # Ensure the default is listed even when the glob missed it.
    if default and not any(item["path"] == default for item in items):
        path = Path(default)
        items.insert(
            0,
            {
                "path": default,
                "name": path.name,
                "default": True,
                "missing": not path.is_file(),
            },
        )
    return items, default
