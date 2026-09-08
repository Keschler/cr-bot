from __future__ import annotations

from typing import Any


def unit_label_vocabulary() -> list[str] | None:
    """Return sorted unit class names, or None when labels are unavailable.

    Health-bar artifacts are excluded (they are not placeable/fixable
    units). Never raises: label maps are an optional import.
    """
    try:
        from katacr.constants.label_list import idx2unit
    except ImportError:
        try:
            from cr_bot.vision.yolo_runtime import idx2unit
        except ImportError:
            return None
    try:
        names = {str(name) for name in dict(idx2unit).values()}
    except (TypeError, ValueError, AttributeError):
        return None
    return sorted(
        name for name in names if name and "bar" not in name.casefold()
    )
