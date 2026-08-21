"""Path-safe workspace budgeting for disposable fidelity source videos."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


STORAGE_SCHEMA_VERSION = 1
DEFAULT_MAX_WORKSPACE_BYTES = 200_000_000_000
DEFAULT_LOW_WATER_BYTES = 190_000_000_000
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class StorageBudgetError(ValueError):
    """Raised when a retention manifest or budget request is unsafe."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def workspace_size_bytes(root: Path) -> int:
    """Return apparent file bytes below ``root`` without following symlinks."""

    root = root.resolve()
    if not root.is_dir():
        raise StorageBudgetError(f"workspace root is not a directory: {root}")
    total = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                # A concurrent producer may atomically replace a file while the
                # budget walk is running. The post-operation walk is authoritative.
                continue
    return total


def load_retention_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": STORAGE_SCHEMA_VERSION, "artifacts": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageBudgetError(f"cannot load retention manifest {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != STORAGE_SCHEMA_VERSION:
        raise StorageBudgetError(
            f"retention manifest {path} must use schema_version {STORAGE_SCHEMA_VERSION}"
        )
    if not isinstance(raw.get("artifacts"), list):
        raise StorageBudgetError(f"retention manifest {path} artifacts must be a list")
    return raw


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _registered_video_path(
    record: Mapping[str, Any],
    *,
    workspace_root: Path,
    raw_media_root: Path,
) -> tuple[Path | None, str | None]:
    relative = record.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None, "path must be a non-empty workspace-relative string"
    candidate = (workspace_root / relative).resolve()
    try:
        candidate.relative_to(raw_media_root)
    except ValueError:
        return None, "path is outside the configured raw-media root"
    if candidate.suffix.lower() not in VIDEO_SUFFIXES:
        return None, "path is not a supported raw-video file"
    if candidate.is_symlink():
        return None, "symbolic links are never eviction eligible"
    media_sha256 = record.get("media_sha256")
    if not isinstance(media_sha256, str) or not _SHA256_RE.fullmatch(media_sha256):
        return None, "eviction-eligible record must contain a valid media_sha256"
    if candidate.is_file() and _file_sha256(candidate) != media_sha256:
        return None, "media_sha256 does not match the registered raw video"
    return candidate, None


def enforce_workspace_budget(
    workspace_root: Path,
    *,
    manifest_path: Path,
    raw_media_root: Path,
    max_bytes: int = DEFAULT_MAX_WORKSPACE_BYTES,
    low_water_bytes: int = DEFAULT_LOW_WATER_BYTES,
    reserve_bytes: int = 0,
    evict: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Inspect or enforce the workspace cap using registered disposable videos.

    Only existing, non-symlink video files below ``raw_media_root`` whose
    manifest record has both ``truth_extracted`` and ``eviction_eligible`` set
    to true may be unlinked.
    """

    workspace_root = workspace_root.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path.is_absolute()
        else (workspace_root / manifest_path).resolve()
    )
    raw_media_root = (
        raw_media_root.resolve()
        if raw_media_root.is_absolute()
        else (workspace_root / raw_media_root).resolve()
    )
    try:
        manifest_path.relative_to(workspace_root)
        raw_media_root.relative_to(workspace_root)
    except ValueError as error:
        raise StorageBudgetError("manifest and raw-media roots must be inside workspace") from error
    if max_bytes <= 0:
        raise StorageBudgetError("max_bytes must be positive")
    if not 0 <= low_water_bytes <= max_bytes:
        raise StorageBudgetError("low_water_bytes must be between zero and max_bytes")
    if not 0 <= reserve_bytes <= max_bytes:
        raise StorageBudgetError("reserve_bytes must be between zero and max_bytes")

    manifest = load_retention_manifest(manifest_path)
    initial_bytes = workspace_size_bytes(workspace_root)
    projected_bytes = initial_bytes + reserve_bytes
    eviction_required = projected_bytes > max_bytes
    target_workspace_bytes = min(low_water_bytes, max_bytes - reserve_bytes)
    invalid_records: list[dict[str, str]] = []
    candidates: list[tuple[str, str, Path, dict[str, Any]]] = []

    for index, raw_record in enumerate(manifest["artifacts"]):
        if not isinstance(raw_record, dict):
            invalid_records.append({"record": str(index), "reason": "record must be an object"})
            continue
        if raw_record.get("deleted_at"):
            continue
        if not raw_record.get("truth_extracted") or not raw_record.get("eviction_eligible"):
            continue
        candidate, reason = _registered_video_path(
            raw_record,
            workspace_root=workspace_root,
            raw_media_root=raw_media_root,
        )
        if reason is not None:
            invalid_records.append(
                {"record": str(raw_record.get("path", index)), "reason": reason}
            )
            continue
        assert candidate is not None
        if not candidate.is_file():
            continue
        completed_at = raw_record.get("truth_extracted_at")
        if not isinstance(completed_at, str) or not completed_at:
            invalid_records.append(
                {
                    "record": str(raw_record.get("path", index)),
                    "reason": "eviction-eligible record lacks truth_extracted_at",
                }
            )
            continue
        candidates.append((completed_at, str(raw_record.get("path")), candidate, raw_record))

    candidates.sort(key=lambda row: (row[0], row[1]))
    deleted: list[dict[str, Any]] = []
    estimated_bytes = initial_bytes
    if eviction_required and evict:
        timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        for _, relative, candidate, record in candidates:
            if estimated_bytes <= target_workspace_bytes:
                break
            size = candidate.stat().st_size
            candidate.unlink()
            estimated_bytes -= size
            record["deleted_at"] = timestamp
            record["deleted_reason"] = "workspace_budget"
            record["deleted_size_bytes"] = size
            deleted.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "media_sha256": record.get("media_sha256"),
                }
            )
        if deleted:
            _write_manifest(manifest_path, manifest)

    final_bytes = workspace_size_bytes(workspace_root)
    passed = final_bytes + reserve_bytes <= max_bytes
    return {
        "kind": "simulator_workspace_budget",
        "schema_version": STORAGE_SCHEMA_VERSION,
        "workspace_root": str(workspace_root),
        "manifest_path": str(manifest_path),
        "raw_media_root": str(raw_media_root),
        "max_bytes": max_bytes,
        "low_water_bytes": low_water_bytes,
        "reserve_bytes": reserve_bytes,
        "initial_bytes": initial_bytes,
        "initial_projected_bytes": projected_bytes,
        "final_bytes": final_bytes,
        "final_projected_bytes": final_bytes + reserve_bytes,
        "eviction_required": eviction_required,
        "eviction_enabled": evict,
        "eligible_candidate_count": len(candidates),
        "deleted": deleted,
        "invalid_records": invalid_records,
        "passed": passed,
        "deficit_bytes": max(0, final_bytes + reserve_bytes - max_bytes),
    }


__all__ = [
    "DEFAULT_LOW_WATER_BYTES",
    "DEFAULT_MAX_WORKSPACE_BYTES",
    "STORAGE_SCHEMA_VERSION",
    "StorageBudgetError",
    "enforce_workspace_budget",
    "load_retention_manifest",
    "workspace_size_bytes",
]
