from __future__ import annotations

from typing import Any

from ._common import CorrectionUnprocessableError, _finite_number
from .vocab import unit_label_vocabulary


def _match_edit_key(target: Any) -> tuple[str, Any]:
    """Normalize an update/delete target to a match key."""
    if not isinstance(target, dict):
        raise ValueError("edit target must be an object")
    if "track" in target:
        try:
            return ("track", int(target["track"]))
        except (TypeError, ValueError, OverflowError):
            return ("track", str(target["track"]))
    label = target.get("label")
    center = target.get("center")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("edit target needs 'track' or 'label'+'center'")
    if (
        not isinstance(center, (list, tuple))
        or len(center) != 2
        or _finite_number(center[0]) is None
        or _finite_number(center[1]) is None
    ):
        raise ValueError("edit target 'center' must be [x, y] numbers")
    return (
        "label",
        (label.strip(), round(float(center[0]), 1), round(float(center[1]), 1)),
    )


def _find_row_index(rows: list[dict], key: tuple[str, Any]) -> int | None:
    kind, value = key
    if kind == "track":
        for index, row in enumerate(rows):
            track = row.get("track_id")
            try:
                if track is not None and int(track) == int(value):
                    return index
            except (TypeError, ValueError, OverflowError):
                if track is not None and str(track) == str(value):
                    return index
        return None
    label, cx, cy = value
    for index, row in enumerate(rows):
        try:
            if (
                str(row.get("class_name")) == label
                and abs(float(row["center"][0]) - cx) <= 8
                and abs(float(row["center"][1]) - cy) <= 8
            ):
                return index
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return None


def _apply_correction_edits(
    base_rows: list[dict],
    edits: Any,
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> tuple[list[dict], dict]:
    """Validate ``edits`` and apply them to copies of ``base_rows``.

    Returns ``(final_rows, summary)``. Raises ``ValueError`` for malformed
    edits and ``CorrectionUnprocessableError`` when an edit references
    nothing or cannot be satisfied.
    """
    if not isinstance(edits, dict):
        raise ValueError("edits must be an object")
    unknown = set(edits) - {"updates", "deletes", "adds"}
    if unknown:
        raise ValueError(f"unknown edit sections: {sorted(unknown)}")
    for section in ("updates", "deletes", "adds"):
        items = edits.get(section, [])
        if not isinstance(items, list):
            raise ValueError(f"edits.{section} must be a list")
    try:
        from cr_bot.domain.troop_hp_level16 import get_unit_hp_level16
    except ImportError:
        get_unit_hp_level16 = None  # type: ignore[assignment]
    vocab = unit_label_vocabulary()
    vocab_set = set(vocab) if vocab else None

    def check_class(name: Any) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("class_name must be a non-empty string")
        cleaned = name.strip()
        if vocab_set is not None and cleaned not in vocab_set:
            raise ValueError(f"unknown unit label: {cleaned!r}")
        return cleaned

    def check_team(team: Any) -> str:
        if not isinstance(team, str):
            raise ValueError("team must be 'ally' or 'enemy'")
        lowered = team.strip().lower()
        if lowered not in ("ally", "enemy"):
            raise ValueError("team must be 'ally' or 'enemy'")
        return lowered

    rows = [dict(row) for row in base_rows if isinstance(row, dict)]
    normalized: dict[str, list] = {"updates": [], "deletes": [], "adds": []}

    def check_box(box: Any, owner: str) -> list[float]:
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(_finite_number(v) is None for v in box)
        ):
            raise ValueError(f"{owner}.box must be [x1, y1, x2, y2] numbers")
        x1, y1, x2, y2 = (float(v) for v in box)
        if not (x1 < x2 and y1 < y2 and x2 - x1 >= 2 and y2 - y1 >= 2):
            raise ValueError(f"{owner}.box must be a non-empty rectangle")
        if frame_width is not None and not (0 <= x1 <= frame_width and 0 <= x2 <= frame_width):
            raise ValueError(f"{owner}.box x is outside the frame")
        if frame_height is not None and not (0 <= y1 <= frame_height and 0 <= y2 <= frame_height):
            raise ValueError(f"{owner}.box y is outside the frame")
        return [x1, y1, x2, y2]

    for item in edits.get("updates", []):
        if not isinstance(item, dict):
            raise ValueError("each update must be an object")
        if "target" not in item:
            raise ValueError("each update needs a 'target'")
        key = _match_edit_key(item["target"])
        index = _find_row_index(rows, key)
        if index is None:
            raise CorrectionUnprocessableError("update target matches no detection")
        row = rows[index]
        changed = False
        entry: dict[str, Any] = {"target": item["target"]}
        if "class_name" in item:
            new_class = check_class(item["class_name"])
            if new_class != row.get("class_name"):
                row["class_name"] = new_class
                hp = get_unit_hp_level16(new_class) if get_unit_hp_level16 else None
                row["estimated_hp"] = hp
                changed = True
            entry["class_name"] = new_class
        if "team" in item:
            new_team = check_team(item["team"])
            if new_team != row.get("team"):
                row["team"] = new_team
                changed = True
            entry["team"] = new_team
        if "box" in item:
            new_box = check_box(item["box"], "update")
            if list(new_box) != list(row.get("box") or []):
                row["box"] = new_box
                row["center"] = [(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0]
                changed = True
            entry["box"] = new_box
        if not changed:
            raise ValueError("update changes neither class_name, team nor box")
        entry["row"] = dict(row)
        normalized["updates"].append(entry)

    removed: set[int] = set()
    for item in edits.get("deletes", []):
        if not isinstance(item, dict):
            raise ValueError("each delete must be an object")
        if "target" not in item:
            raise ValueError("each delete needs a 'target'")
        key = _match_edit_key(item["target"])
        index = _find_row_index(
            [r for i, r in enumerate(rows) if i not in removed], key
        )
        if index is None:
            raise CorrectionUnprocessableError("delete target matches no detection")
        real_index = [i for i in range(len(rows)) if i not in removed][index]
        removed.add(real_index)
        normalized["deletes"].append({"target": item["target"]})
    rows = [row for i, row in enumerate(rows) if i not in removed]

    for item in edits.get("adds", []):
        if not isinstance(item, dict):
            raise ValueError("each add must be an object")
        if "class_name" not in item or "team" not in item:
            raise ValueError("each add needs 'box', 'class_name' and 'team'")
        x1, y1, x2, y2 = check_box(item.get("box"), "add")
        new_class = check_class(item["class_name"])
        new_team = check_team(item["team"])
        hp = get_unit_hp_level16(new_class) if get_unit_hp_level16 else None
        rows.append(
            {
                "class_name": new_class,
                "team": new_team,
                "confidence": 1.0,
                "track_id": None,
                "box": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "estimated_hp": hp,
            }
        )
        normalized["adds"].append(
            {"box": [x1, y1, x2, y2], "class_name": new_class, "team": new_team}
        )

    summary = {
        "updated": len(normalized["updates"]),
        "deleted": len(normalized["deletes"]),
        "added": len(normalized["adds"]),
    }
    if sum(summary.values()) == 0:
        raise ValueError("edits contain no updates, deletes or adds")
    return rows, {"edits": normalized, "counts": summary}
