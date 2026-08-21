"""Hashing, output layout, and retention registration for physical runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .lifecycle import LIFECYCLE_PATH
from .schema import PhysicalLabError, canonical_hash, canonical_json


PHYSICAL_OUTPUT_RELATIVE_ROOT = Path("outputs/simulator/fidelity_media/physical_lab")
RETENTION_SCHEMA_VERSION = 1
_VIDEO_KINDS = frozenset({"raw_video"})
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def hash_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise PhysicalLabError(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def ensure_relative_to(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path).resolve()
    root = Path(root).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PhysicalLabError(f"path {candidate} is outside allowed root {root}") from error
    return candidate


def physical_output_root(
    *,
    repository_root: str | Path,
    run_id: str,
) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise PhysicalLabError("run_id must be a single safe path component")
    root = Path(repository_root).resolve() / PHYSICAL_OUTPUT_RELATIVE_ROOT / run_id
    ensure_relative_to(root, Path(repository_root).resolve() / PHYSICAL_OUTPUT_RELATIVE_ROOT)
    return root


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    source: str = "physical_lab"

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source": self.source,
        }


def seal_json(path: str | Path, payload: Mapping[str, Any]) -> ArtifactRef:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(dict(payload)) + "\n"
    destination.write_text(encoded, encoding="utf-8")
    return ArtifactRef(
        artifact_id=destination.stem,
        kind="json",
        path=str(destination),
        sha256=hash_file(destination),
        size_bytes=destination.stat().st_size,
    )


def seal_bytes(path: str | Path, payload: bytes, *, kind: str = "binary") -> ArtifactRef:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return ArtifactRef(
        artifact_id=destination.stem,
        kind=kind,
        path=str(destination),
        sha256=hash_file(destination),
        size_bytes=destination.stat().st_size,
    )


def artifact_manifest(
    *,
    run_id: str,
    experiment_hash: str,
    artifacts: Iterable[ArtifactRef],
    status: str,
) -> dict[str, object]:
    rows = [artifact.to_dict() for artifact in artifacts]
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "physical_lab_artifact_manifest",
        "run_id": run_id,
        "experiment_hash": experiment_hash,
        "status": status,
        "artifacts": rows,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def _load_retention_manifest(destination: Path) -> dict[str, Any]:
    if not destination.exists():
        return {"schema_version": RETENTION_SCHEMA_VERSION, "artifacts": []}
    try:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalLabError(f"cannot load retention manifest {destination}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != RETENTION_SCHEMA_VERSION:
        raise PhysicalLabError("retention manifest has an unsupported schema")
    if not isinstance(manifest.get("artifacts"), list):
        raise PhysicalLabError("retention manifest artifacts must be an array")
    return manifest


def _workspace_relative(path: str | Path, workspace_root: Path) -> tuple[Path, str]:
    """Resolve an artifact and return its safe workspace-relative spelling."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    candidate = ensure_relative_to(candidate, workspace_root)
    return candidate, candidate.relative_to(workspace_root).as_posix()


def _write_retention_manifest(destination: Path, manifest: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def register_retention_records(
    manifest_path: str | Path,
    *,
    run_id: str,
    experiment_hash: str,
    artifacts: Iterable[ArtifactRef],
    raw_media: bool = False,
    workspace_root: str | Path | None = None,
) -> dict[str, object]:
    """Append generic physical-lab records to the existing retention file.

    The existing ``media-budget`` command only evicts records that explicitly
    opt into its raw-video contract.  Physical artifacts default to
    non-evictable, preserving evidence while still making every file and hash
    discoverable from the shared manifest.
    """

    root = None if workspace_root is None else Path(workspace_root).resolve()
    destination = Path(manifest_path)
    if root is not None and not destination.is_absolute():
        destination = (root / destination).resolve()
    manifest = _load_retention_manifest(destination)

    now = datetime.now(timezone.utc).isoformat()
    for artifact in artifacts:
        if root is None:
            if Path(artifact.path).is_absolute():
                raise PhysicalLabError(
                    "absolute artifact paths require workspace_root for retention registration"
                )
            stored_path = Path(artifact.path).as_posix()
        else:
            _, stored_path = _workspace_relative(artifact.path, root)
        manifest["artifacts"].append(
            {
                "artifact_id": f"physical-lab:{run_id}:{artifact.artifact_id}",
                "path": stored_path,
                "artifact_kind": artifact.kind,
                "media_sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "run_id": run_id,
                "experiment_hash": experiment_hash,
                "created_at": now,
                "truth_extracted": False,
                "eviction_eligible": bool(raw_media and artifact.kind == "raw_video"),
                "retention_reason": "physical lab provenance; evidence remains sealed",
            }
        )
    _write_retention_manifest(destination, manifest)
    return manifest


def finalize_retention_records(
    manifest_path: str | Path,
    *,
    run_id: str,
    run_manifest_path: str | Path,
    observation_manifest_path: str | Path,
    workspace_root: str | Path,
    audit_paths: Iterable[str | Path] = (),
    now: datetime | None = None,
) -> dict[str, object]:
    """Promote only a complete, hash-verified physical ingest to evictable raw media.

    Registration and promotion are intentionally separate.  A capture may be
    referenced by a run artifact immediately, but it becomes disposable only
    after the sealed passed lifecycle (including reviewed detector
    provenance), observation manifest, synchronization result, recognized
    replay-cache hash, device provenance, and compact audit artifacts have all
    been verified against the same run and experiment.
    """

    root = Path(workspace_root).resolve()
    destination = Path(manifest_path)
    if not destination.is_absolute():
        destination = (root / destination).resolve()
    manifest = _load_retention_manifest(destination)
    run_path, run_relative = _workspace_relative(run_manifest_path, root)
    observation_path, observation_relative = _workspace_relative(observation_manifest_path, root)
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalLabError(f"cannot load physical ingest provenance: {error}") from error
    if not isinstance(run, dict) or run.get("kind") != "physical_lab_run":
        raise PhysicalLabError("run manifest is not a physical_lab_run")
    if not isinstance(observation, dict) or observation.get("kind") != "physical_lab_observation_manifest":
        raise PhysicalLabError("observation manifest is not a physical_lab_observation_manifest")
    run_hash = run.get("run_hash")
    run_payload = {key: value for key, value in run.items() if key != "run_hash"}
    if not isinstance(run_hash, str) or run_hash != canonical_hash(run_payload):
        raise PhysicalLabError("run manifest hash is missing or invalid")
    observation_hash = observation.get("manifest_hash")
    observation_payload = {
        key: value for key, value in observation.items() if key != "manifest_hash"
    }
    if not isinstance(observation_hash, str) or observation_hash != canonical_hash(observation_payload):
        raise PhysicalLabError("observation manifest hash is missing or invalid")
    if run.get("run_id") != run_id or observation.get("run_id") != run_id:
        raise PhysicalLabError("run and observation IDs do not match retention finalization")
    experiment_hash = run.get("experiment_hash")
    if not isinstance(experiment_hash, str) or observation.get("experiment_hash") != experiment_hash:
        raise PhysicalLabError("run and observation experiment hashes do not match")
    if run.get("status") == "rejected" or observation.get("status") == "rejected":
        raise PhysicalLabError("rejected physical runs cannot become eviction eligible")
    lifecycle = run.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("passed") is not True:
        raise PhysicalLabError("run lifecycle is not a passed report")
    if (
        lifecycle.get("initial_state") != LIFECYCLE_PATH[0].value
        or lifecycle.get("final_state") != LIFECYCLE_PATH[0].value
    ):
        raise PhysicalLabError("run lifecycle did not return to recovery")
    transitions = lifecycle.get("transitions")
    if not isinstance(transitions, list):
        raise PhysicalLabError("run lifecycle transitions are missing")
    observed_targets: list[str] = []
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping) or not isinstance(transition.get("to"), str):
            raise PhysicalLabError(f"run lifecycle transition {index} is malformed")
        observed_targets.append(str(transition["to"]))
    expected_targets = [state.value for state in LIFECYCLE_PATH[1:]]
    if observed_targets != expected_targets:
        raise PhysicalLabError(
            "run lifecycle transitions are incomplete: "
            f"expected={expected_targets}, observed={observed_targets}"
        )
    detector_provenance = lifecycle.get("detector_provenance")
    if not isinstance(detector_provenance, Mapping) or set(detector_provenance) != {"A", "B"}:
        raise PhysicalLabError("run lifecycle lacks both detector provenance records")
    for side in ("A", "B"):
        detector = detector_provenance.get(side)
        if not isinstance(detector, Mapping):
            raise PhysicalLabError(f"run lifecycle detector provenance for {side} is malformed")
        kind = detector.get("kind")
        if not isinstance(kind, str) or not kind.startswith("reviewed_"):
            raise PhysicalLabError(f"run lifecycle detector for {side} is not reviewed")
        detector_hash = detector.get("manifest_sha256") or detector.get("detector_sha256")
        if not isinstance(detector_hash, str) or not _HASH_RE.fullmatch(detector_hash):
            raise PhysicalLabError(f"run lifecycle detector for {side} lacks a sealed hash")
    synchronization = observation.get("synchronization")
    if not isinstance(synchronization, Mapping) or synchronization.get("accepted") is not True:
        raise PhysicalLabError("physical observation synchronization is not accepted")
    capture_group_id = observation.get("capture_group_id")
    evidence_split = observation.get("evidence_split")
    replay_cache_hash = observation.get("replay_cache_hash")
    if not isinstance(capture_group_id, str) or not capture_group_id:
        raise PhysicalLabError("physical observation lacks a capture group")
    if not isinstance(evidence_split, str) or not evidence_split:
        raise PhysicalLabError("physical observation lacks an evidence split")
    if not isinstance(replay_cache_hash, str) or not replay_cache_hash:
        raise PhysicalLabError("physical observation lacks a recognized replay-cache hash")

    device_info = run.get("device_info")
    captures = run.get("captures")
    if not isinstance(device_info, Mapping) or set(device_info) != {"A", "B"}:
        raise PhysicalLabError("run manifest must contain both device provenance records")
    experiment = run.get("experiment")
    if not isinstance(experiment, Mapping) or experiment.get("evidence_split") != evidence_split:
        raise PhysicalLabError("run and observation evidence splits do not match")
    device_serial_hashes: dict[str, str] = {}
    for side in ("A", "B"):
        row = device_info.get(side)
        if (
            not isinstance(row, Mapping)
            or row.get("connected") is not True
            or not isinstance(row.get("serial_hash"), str)
        ):
            raise PhysicalLabError(f"run manifest lacks serial provenance for device {side}")
        device_serial_hashes[side] = str(row["serial_hash"])
    if not isinstance(captures, Mapping) or set(captures) != {"A", "B"}:
        raise PhysicalLabError("run manifest must contain both capture records")

    raw_records = [
        record
        for record in manifest["artifacts"]
        if isinstance(record, dict)
        and record.get("run_id") == run_id
        and record.get("artifact_kind") in _VIDEO_KINDS
        and not record.get("deleted_at")
    ]
    if not raw_records:
        return {
            "run_id": run_id,
            "finalized_count": 0,
            "eviction_eligible": False,
            "reason": "no registered physical raw-video artifacts",
        }

    observed_media_hashes = observation.get("media_hashes")
    if not isinstance(observed_media_hashes, Mapping):
        raise PhysicalLabError("observation media_hashes must be an object")
    capture_rows: dict[str, Mapping[str, Any]] = {}
    for side in ("A", "B"):
        capture = captures[side]
        if not isinstance(capture, Mapping):
            raise PhysicalLabError(f"capture {side} is malformed")
        if capture.get("stream_verified") is not True or capture.get("status") != "complete":
            raise PhysicalLabError(f"capture {side} is not a verified complete stream")
        media_hash = capture.get("media_sha256")
        media_path = capture.get("media_path")
        if (
            not isinstance(media_hash, str)
            or not isinstance(media_path, str)
            or capture.get("frame_count", 0) <= 0
            or not isinstance(capture.get("capture_id"), str)
        ):
            raise PhysicalLabError(f"capture {side} lacks media path/hash provenance")
        capture_rows[side] = capture
        capture_ids = observation.get("capture_ids")
        if not isinstance(capture_ids, list) or capture["capture_id"] not in capture_ids:
            raise PhysicalLabError(f"observation does not reference capture {side}")
        if media_hash not in {str(value) for value in observed_media_hashes.values()}:
            raise PhysicalLabError(f"observation does not reference capture {side} media hash")

    audit_paths_resolved: dict[str, str] = {}
    requested_audits = [run_path, observation_path, *[Path(path) for path in audit_paths]]
    for raw_path in requested_audits:
        path, relative = _workspace_relative(raw_path, root)
        if not path.is_file():
            raise PhysicalLabError(f"physical audit artifact is missing: {relative}")
        actual_hash = hash_file(path)
        existing = audit_paths_resolved.get(relative)
        if existing is not None and existing != actual_hash:
            raise PhysicalLabError(f"physical audit artifact hash changed: {relative}")
        audit_paths_resolved[relative] = actual_hash
    if not any(value == replay_cache_hash for value in audit_paths_resolved.values()):
        raise PhysicalLabError("recognized replay-cache hash is not retained as an audit artifact")

    updated = 0
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    for record in raw_records:
        raw_path, relative = _workspace_relative(str(record.get("path", "")), root)
        if not raw_path.is_file():
            raise PhysicalLabError(f"registered physical media is missing: {relative}")
        registered_hash = record.get("media_sha256")
        actual_hash = hash_file(raw_path)
        if registered_hash != actual_hash:
            raise PhysicalLabError(f"registered physical media hash changed: {relative}")
        matching_side = None
        for side, capture in capture_rows.items():
            capture_path = capture.get("media_path")
            _, capture_relative = _workspace_relative(str(capture_path), root)
            if capture_relative == relative and capture.get("media_sha256") == actual_hash:
                matching_side = side
                break
        if matching_side is None:
            raise PhysicalLabError(f"registered media is not a sealed run capture: {relative}")
        record.update(
            {
                "truth_extracted": True,
                "truth_extracted_at": timestamp,
                "eviction_eligible": True,
                "evidence_group_id": capture_group_id,
                "evidence_split": evidence_split,
                "device_serial_hashes": dict(device_serial_hashes),
                "capture_id": capture_rows[matching_side].get("capture_id"),
                "observation_manifest_path": observation_relative,
                "observation_manifest_sha256": hash_file(observation_path),
                "replay_cache_sha256": replay_cache_hash,
                "audit_artifact_hashes": dict(sorted(audit_paths_resolved.items())),
                "generated_paths": sorted(audit_paths_resolved),
            }
        )
        updated += 1
    _write_retention_manifest(destination, manifest)
    return {
        "run_id": run_id,
        "experiment_hash": experiment_hash,
        "finalized_count": updated,
        "eviction_eligible": True,
        "capture_group_id": capture_group_id,
        "evidence_split": evidence_split,
        "observation_manifest_path": observation_relative,
        "audit_artifact_count": len(audit_paths_resolved),
    }


__all__ = [
    "ArtifactRef",
    "PHYSICAL_OUTPUT_RELATIVE_ROOT",
    "artifact_manifest",
    "ensure_relative_to",
    "finalize_retention_records",
    "hash_file",
    "physical_output_root",
    "register_retention_records",
    "seal_bytes",
    "seal_json",
]
