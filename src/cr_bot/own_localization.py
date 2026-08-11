from __future__ import annotations

from copy import deepcopy
from typing import Any

from cr_bot.features.action_space import ACTION_GRID


def validate_own_localization_decisions(
    document: dict[str, Any], package: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate a blind own-location worker result against its sealed package."""

    if document.get("stage") != "own_localization_chunk":
        raise ValueError("expected own_localization_chunk stage")
    targets = package.get("targets")
    decisions = document.get("decisions")
    if not isinstance(targets, list) or not isinstance(decisions, list):
        raise ValueError("localization targets and decisions must be lists")
    target_by_id = {
        row.get("event_id"): row for row in targets if isinstance(row, dict)
    }
    if len(target_by_id) != len(targets) or None in target_by_id:
        raise ValueError("package contains duplicate or invalid localization targets")
    decision_by_id = {
        row.get("event_id"): row for row in decisions if isinstance(row, dict)
    }
    if len(decision_by_id) != len(decisions):
        raise ValueError("worker contains duplicate or invalid localization decisions")
    if set(decision_by_id) != set(target_by_id):
        raise ValueError("worker must cover every localization target exactly once")

    required = {
        "event_id",
        "location_frame_index",
        "location_rule",
        "cell",
        "macro_review_artifacts",
        "grid_review_artifacts",
        "confidence",
        "reason",
    }
    validated: list[dict[str, Any]] = []
    for event_id, target in target_by_id.items():
        raw = decision_by_id[event_id]
        if set(raw) != required:
            raise ValueError(
                f"{event_id}: decision must contain exactly {', '.join(sorted(required))}"
            )
        cell = raw["cell"]
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in cell)
            or not 0 <= cell[0] < ACTION_GRID.cols
            or not 0 <= cell[1] < ACTION_GRID.rows
        ):
            raise ValueError(f"{event_id}: invalid [column,row] cell")
        frame = raw["location_frame_index"]
        if frame not in target.get("review_frame_indices", []):
            raise ValueError(f"{event_id}: location frame is not present in the evidence")
        if raw["location_rule"] not in target.get("location_rule_options", []):
            raise ValueError(f"{event_id}: invalid card-specific location rule")
        for key in ("macro_review_artifacts", "grid_review_artifacts"):
            expected = target.get(key)
            if raw[key] != expected or not isinstance(expected, list) or not expected:
                raise ValueError(f"{event_id}: {key} must copy the package exactly")
        if raw["confidence"] not in {"direct", "inferred"}:
            raise ValueError(f"{event_id}: confidence must be direct or inferred")
        if not isinstance(raw["reason"], str) or not raw["reason"].strip():
            raise ValueError(f"{event_id}: a visual reason is required")
        validated.append(deepcopy(raw))
    return validated

