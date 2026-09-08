from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


CHECKPOINT_GLOBS = (
    "prototype.pt",
    "*.pt",
    "simulator/outputs/**/*.pt",
    "outputs/**/*.pt",
)


def _default_checkpoint() -> str:
    preferred = REPO_ROOT / "prototype.pt"
    if preferred.is_file():
        return str(preferred)
    try:
        from simulator.physical_lab.prototype_controller import DEFAULT_CHECKPOINT
    except ImportError:
        from physical_lab.prototype_controller import DEFAULT_CHECKPOINT  # type: ignore
    return str(DEFAULT_CHECKPOINT)


def _list_checkpoints(limit: int = 20) -> tuple[list[dict[str, Any]], str | None]:
    """List candidate checkpoint files; default prefers repo-root prototype.pt."""
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
    if not any(item.get("default") for item in items) and items:
        items[0]["default"] = True
        default = items[0]["path"]
    return items, default
