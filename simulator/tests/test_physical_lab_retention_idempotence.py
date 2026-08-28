from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.physical_lab.artifacts import ArtifactRef, hash_file, register_retention_records
from simulator.physical_lab.schema import PhysicalLabError


def _artifact(path: Path, *, artifact_id: str = "capture-A") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind="raw_video",
        path=str(path),
        sha256=hash_file(path),
        size_bytes=path.stat().st_size,
    )


def test_retention_registration_is_idempotent(tmp_path: Path) -> None:
    media = tmp_path / "capture.mp4"
    media.write_bytes(b"capture")
    manifest_path = tmp_path / "retention.json"

    register_retention_records(
        manifest_path,
        run_id="run-1",
        experiment_hash="sha256:" + "a" * 64,
        artifacts=[_artifact(media)],
        workspace_root=tmp_path,
    )
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    register_retention_records(
        manifest_path,
        run_id="run-1",
        experiment_hash="sha256:" + "a" * 64,
        artifacts=[_artifact(media)],
        workspace_root=tmp_path,
    )
    second = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(first["artifacts"]) == 1
    assert second["artifacts"] == first["artifacts"]


def test_retention_registration_rejects_conflicting_reuse(tmp_path: Path) -> None:
    first_media = tmp_path / "first.mp4"
    second_media = tmp_path / "second.mp4"
    first_media.write_bytes(b"first")
    second_media.write_bytes(b"second")
    manifest_path = tmp_path / "retention.json"

    register_retention_records(
        manifest_path,
        run_id="run-1",
        experiment_hash="sha256:" + "a" * 64,
        artifacts=[_artifact(first_media)],
        workspace_root=tmp_path,
    )

    with pytest.raises(PhysicalLabError, match="conflicting metadata"):
        register_retention_records(
            manifest_path,
            run_id="run-1",
            experiment_hash="sha256:" + "a" * 64,
            artifacts=[_artifact(second_media)],
            workspace_root=tmp_path,
        )

