"""Capture-group split assignment and immutable split locks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import EvidenceSplit, PhysicalLabError


SPLIT_LOCK_SCHEMA_VERSION = 1


def assign_capture_group_split(
    capture_group_id: str,
    *,
    salt: str = "physical-lab-v1",
) -> EvidenceSplit:
    """Assign a whole capture group deterministically before inspection."""

    if not isinstance(capture_group_id, str) or not capture_group_id.strip():
        raise PhysicalLabError("capture_group_id is required for split assignment")
    digest = hashlib.sha256(f"{salt}:{capture_group_id}".encode("utf-8")).digest()[0]
    if digest < 51:  # 20%
        return EvidenceSplit.CALIBRATION
    if digest < 102:  # 20%
        return EvidenceSplit.VALIDATION
    return EvidenceSplit.HELDOUT


class SplitLock:
    """Persistent group-to-split lock; changing a label requires a new group."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SPLIT_LOCK_SCHEMA_VERSION, "groups": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PhysicalLabError(f"cannot load split lock {self.path}: {error}") from error
        if not isinstance(raw, dict) or raw.get("schema_version") != SPLIT_LOCK_SCHEMA_VERSION:
            raise PhysicalLabError("split lock has an unsupported schema")
        if not isinstance(raw.get("groups"), dict):
            raise PhysicalLabError("split lock groups must be an object")
        return raw

    def get(self, capture_group_id: str) -> EvidenceSplit | None:
        value = self._load()["groups"].get(capture_group_id)
        if value is None:
            return None
        try:
            return EvidenceSplit(value)
        except ValueError as error:
            raise PhysicalLabError(f"split lock contains invalid split for {capture_group_id!r}") from error

    def lock(self, capture_group_id: str, split: EvidenceSplit | str) -> EvidenceSplit:
        if not isinstance(capture_group_id, str) or not capture_group_id.strip():
            raise PhysicalLabError("capture_group_id is required")
        try:
            split = split if isinstance(split, EvidenceSplit) else EvidenceSplit(split)
        except ValueError as error:
            raise PhysicalLabError(f"invalid evidence split: {split!r}") from error
        manifest = self._load()
        groups = manifest["groups"]
        previous = groups.get(capture_group_id)
        if previous is not None and previous != split.value:
            raise PhysicalLabError(
                f"capture group {capture_group_id!r} is already locked to {previous!r}; "
                "create a new independent capture group"
            )
        groups[capture_group_id] = split.value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return split


__all__ = ["SPLIT_LOCK_SCHEMA_VERSION", "SplitLock", "assign_capture_group_split"]
