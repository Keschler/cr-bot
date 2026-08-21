from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from simulator.storage import enforce_workspace_budget, workspace_size_bytes


def _write_manifest(root: Path, artifacts: list[dict[str, object]]) -> Path:
    path = root / "outputs/simulator/fidelity_media/retention.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "artifacts": artifacts}),
        encoding="utf-8",
    )
    return path


def test_budget_evicts_only_oldest_completed_registered_raw_video(tmp_path: Path) -> None:
    raw = tmp_path / "outputs/simulator/fidelity_media/raw"
    raw.mkdir(parents=True)
    old = raw / "old.mp4"
    new = raw / "new.webm"
    old.write_bytes(b"o" * 10_000)
    new.write_bytes(b"n" * 10_000)
    unrelated = tmp_path / "user-video.mp4"
    unrelated.write_bytes(b"u" * 10_000)
    old_hash = "sha256:" + hashlib.sha256(old.read_bytes()).hexdigest()
    new_hash = "sha256:" + hashlib.sha256(new.read_bytes()).hexdigest()
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "path": str(old.relative_to(tmp_path)),
                "truth_extracted": True,
                "truth_extracted_at": "2023-01-01T00:00:00+00:00",
                "eviction_eligible": True,
                "media_sha256": old_hash,
            },
            {
                "path": str(new.relative_to(tmp_path)),
                "truth_extracted": True,
                "truth_extracted_at": "2023-02-01T00:00:00+00:00",
                "eviction_eligible": True,
                "media_sha256": new_hash,
            },
        ],
    )
    initial = workspace_size_bytes(tmp_path)

    report = enforce_workspace_budget(
        tmp_path,
        manifest_path=manifest,
        raw_media_root=raw,
        max_bytes=initial - 5_000,
        low_water_bytes=initial - 5_000,
        evict=True,
        now=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert report["passed"] is True
    assert [row["path"] for row in report["deleted"]] == [str(old.relative_to(tmp_path))]
    assert not old.exists()
    assert new.exists()
    assert unrelated.exists()
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["artifacts"][0]["deleted_reason"] == "workspace_budget"


def test_budget_never_deletes_unregistered_or_outside_raw_root(tmp_path: Path) -> None:
    raw = tmp_path / "outputs/simulator/fidelity_media/raw"
    raw.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x" * 10_000)
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "path": "outside.mp4",
                "truth_extracted": True,
                "truth_extracted_at": "2023-01-01T00:00:00+00:00",
                "eviction_eligible": True,
            }
        ],
    )
    initial = workspace_size_bytes(tmp_path)

    report = enforce_workspace_budget(
        tmp_path,
        manifest_path=manifest,
        raw_media_root=raw,
        max_bytes=initial - 1,
        low_water_bytes=initial - 1,
        evict=True,
    )

    assert report["passed"] is False
    assert report["deleted"] == []
    assert report["deficit_bytes"] > 0
    assert report["invalid_records"][0]["reason"] == "path is outside the configured raw-media root"
    assert outside.exists()


def test_reserved_download_space_is_included_in_budget_gate(tmp_path: Path) -> None:
    raw = tmp_path / "outputs/simulator/fidelity_media/raw"
    raw.mkdir(parents=True)
    manifest = _write_manifest(tmp_path, [])
    current = workspace_size_bytes(tmp_path)

    report = enforce_workspace_budget(
        tmp_path,
        manifest_path=manifest,
        raw_media_root=raw,
        max_bytes=current + 99,
        low_water_bytes=current,
        reserve_bytes=100,
    )

    assert report["passed"] is False
    assert report["eviction_required"] is True
    assert report["deficit_bytes"] == 1


def test_budget_rejects_a_registered_path_when_hash_changed(tmp_path: Path) -> None:
    raw = tmp_path / "outputs/simulator/fidelity_media/raw"
    raw.mkdir(parents=True)
    video = raw / "changed.mp4"
    video.write_bytes(b"current")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "path": str(video.relative_to(tmp_path)),
                "truth_extracted": True,
                "truth_extracted_at": "2023-01-01T00:00:00+00:00",
                "eviction_eligible": True,
                "media_sha256": "sha256:" + hashlib.sha256(b"original").hexdigest(),
            }
        ],
    )
    initial = workspace_size_bytes(tmp_path)

    report = enforce_workspace_budget(
        tmp_path,
        manifest_path=manifest,
        raw_media_root=raw,
        max_bytes=initial - 1,
        low_water_bytes=initial - 1,
        evict=True,
    )

    assert report["passed"] is False
    assert report["deleted"] == []
    assert video.exists()
    assert report["invalid_records"][0]["reason"] == (
        "media_sha256 does not match the registered raw video"
    )
