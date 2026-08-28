from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from simulator.physical_lab import LIFECYCLE_PATH, finalize_retention_records, hog_cannon_probe
from simulator.physical_lab.artifacts import ArtifactRef, hash_file, register_retention_records
from simulator.physical_lab.cache import seal_replay_cache
from simulator.physical_lab.schema import PhysicalLabError, canonical_hash


def _write_recognized_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump({"schema_version": 1}, handle)
        pickle.dump(
            SimpleNamespace(
                frame_idx=0,
                video_time_s=0.0,
                frame_png=b"encoded-frame",
            ),
            handle,
        )


def _write_finalization_fixture(
    root: Path,
    *,
    replay_cache: Path,
) -> tuple[Path, Path, Path, str]:
    run_id = "recognition-run"
    spec = hog_cannon_probe(capture_group_id="recognition-group", evidence_split="validation")
    raw_root = root / "outputs/simulator/fidelity_media/physical_lab" / run_id / "raw"
    raw_root.mkdir(parents=True)
    media: dict[str, tuple[Path, str]] = {}
    for side in ("A", "B"):
        path = raw_root / f"{side}.mp4"
        path.write_bytes(f"capture-{side}".encode("ascii"))
        media[side] = (path, hash_file(path))

    retention_path = root / "outputs/simulator/fidelity_media/retention.json"
    register_retention_records(
        retention_path,
        run_id=run_id,
        experiment_hash=spec.experiment_hash(),
        workspace_root=root,
        artifacts=[
            ArtifactRef(
                artifact_id=f"capture-{side}",
                kind="raw_video",
                path=str(path),
                sha256=media_hash,
                size_bytes=path.stat().st_size,
            )
            for side, (path, media_hash) in media.items()
        ],
    )

    run_path = root / "run.json"
    run_payload = {
        "kind": "physical_lab_run",
        "run_id": run_id,
        "status": "candidate_only",
        "experiment_hash": spec.experiment_hash(),
        "experiment": spec.to_dict(include_hash=True),
        "device_info": {
            side: {"serial_hash": spec.devices[side].serial_hash, "connected": True}
            for side in ("A", "B")
        },
        "captures": {
            side: {
                "capture_id": f"capture-{side}",
                "source_device": side,
                "media_path": str(path),
                "media_sha256": media_hash,
                "stream_verified": True,
                "status": "complete",
                "frame_count": 1,
            }
            for side, (path, media_hash) in media.items()
        },
        "synchronization": {"accepted": True},
        "lifecycle": {
            "initial_state": "recovery",
            "final_state": "recovery",
            "passed": True,
            "transitions": [{"to": state.value} for state in LIFECYCLE_PATH[1:]],
            "detector_provenance": {
                side: {
                    "kind": "reviewed_screen_template_detector",
                    "device_id": side,
                    "manifest_sha256": f"sha256:{('a' if side == 'A' else 'b') * 64}",
                }
                for side in ("A", "B")
            },
        },
    }
    run_payload["run_hash"] = canonical_hash(run_payload)
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")

    observation_path = root / "observation.json"
    observation_payload = {
        "kind": "physical_lab_observation_manifest",
        "run_id": run_id,
        "experiment_hash": spec.experiment_hash(),
        "capture_group_id": "recognition-group",
        "evidence_split": "validation",
        "status": "validation",
        "capture_ids": ["capture-A", "capture-B"],
        "media_hashes": {side: media_hash for side, (_, media_hash) in media.items()},
        "synchronization": {"accepted": True},
        "entities": [],
        "events": [],
        "rejected": [],
        "replay_cache_hash": hash_file(replay_cache),
    }
    observation_payload["manifest_hash"] = canonical_hash(observation_payload)
    observation_path.write_text(json.dumps(observation_payload), encoding="utf-8")
    return retention_path, run_path, observation_path, run_id


def test_opaque_replay_cache_bytes_cannot_promote_raw_media(tmp_path: Path) -> None:
    replay_cache = tmp_path / "opaque-cache.bin"
    replay_cache.write_bytes(b"opaque bytes with a matching hash")
    retention_path, run_path, observation_path, run_id = _write_finalization_fixture(
        tmp_path,
        replay_cache=replay_cache,
    )

    with pytest.raises(PhysicalLabError, match="not recognized"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=tmp_path,
            audit_paths=(replay_cache,),
        )

    saved = json.loads(retention_path.read_text(encoding="utf-8"))
    assert all(not row["eviction_eligible"] for row in saved["artifacts"])


def test_recognized_sealed_replay_cache_can_promote_raw_media(tmp_path: Path) -> None:
    replay_cache = tmp_path / "replay-cache.pkl.gz"
    _write_recognized_cache(replay_cache)
    assert seal_replay_cache(replay_cache).recognized is True
    retention_path, run_path, observation_path, run_id = _write_finalization_fixture(
        tmp_path,
        replay_cache=replay_cache,
    )

    result = finalize_retention_records(
        retention_path,
        run_id=run_id,
        run_manifest_path=run_path,
        observation_manifest_path=observation_path,
        workspace_root=tmp_path,
        audit_paths=(replay_cache,),
    )

    assert result["eviction_eligible"] is True
    saved = json.loads(retention_path.read_text(encoding="utf-8"))
    assert all(row["eviction_eligible"] for row in saved["artifacts"])
    assert {row["capture_side"] for row in saved["artifacts"]} == {"A", "B"}
    assert all(row["run_manifest_path"] == "run.json" for row in saved["artifacts"])


def test_retention_finalization_requires_exact_side_media_bindings(tmp_path: Path) -> None:
    replay_cache = tmp_path / "replay-cache.pkl.gz"
    _write_recognized_cache(replay_cache)
    retention_path, run_path, observation_path, run_id = _write_finalization_fixture(
        tmp_path,
        replay_cache=replay_cache,
    )
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["media_hashes"]["A"], observation["media_hashes"]["B"] = (
        observation["media_hashes"]["B"],
        observation["media_hashes"]["A"],
    )
    observation["manifest_hash"] = canonical_hash(
        {key: value for key, value in observation.items() if key != "manifest_hash"}
    )
    observation_path.write_text(json.dumps(observation), encoding="utf-8")

    with pytest.raises(PhysicalLabError, match="media hash for capture A"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=tmp_path,
            audit_paths=(replay_cache,),
        )

    saved = json.loads(retention_path.read_text(encoding="utf-8"))
    assert all(not row["eviction_eligible"] for row in saved["artifacts"])


def test_retention_finalization_requires_run_observation_sync_match(tmp_path: Path) -> None:
    replay_cache = tmp_path / "replay-cache.pkl.gz"
    _write_recognized_cache(replay_cache)
    retention_path, run_path, observation_path, run_id = _write_finalization_fixture(
        tmp_path,
        replay_cache=replay_cache,
    )
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["synchronization"] = {"accepted": True, "common_marker_count": 1}
    observation["manifest_hash"] = canonical_hash(
        {key: value for key, value in observation.items() if key != "manifest_hash"}
    )
    observation_path.write_text(json.dumps(observation), encoding="utf-8")

    with pytest.raises(PhysicalLabError, match="synchronization records do not match"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=tmp_path,
            audit_paths=(replay_cache,),
        )

    saved = json.loads(retention_path.read_text(encoding="utf-8"))
    assert all(not row["eviction_eligible"] for row in saved["artifacts"])


def test_retention_finalization_requires_run_sync_and_truth_bearing_observation(
    tmp_path: Path,
) -> None:
    replay_cache = tmp_path / "replay-cache.pkl.gz"
    _write_recognized_cache(replay_cache)
    retention_path, run_path, observation_path, run_id = _write_finalization_fixture(
        tmp_path,
        replay_cache=replay_cache,
    )

    valid_run = json.loads(run_path.read_text(encoding="utf-8"))
    run = dict(valid_run)
    run.pop("synchronization")
    run["run_hash"] = canonical_hash(
        {key: value for key, value in run.items() if key != "run_hash"}
    )
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(PhysicalLabError, match="run synchronization"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=tmp_path,
            audit_paths=(replay_cache,),
        )

    # Restore a valid run, then prove that a candidate-only observation cannot
    # make raw media disposable merely because its hashes and cache are valid.
    run_path.write_text(json.dumps(valid_run), encoding="utf-8")
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["status"] = "candidate_only"
    observation["manifest_hash"] = canonical_hash(
        {key: value for key, value in observation.items() if key != "manifest_hash"}
    )
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    with pytest.raises(PhysicalLabError, match="not truth-bearing"):
        finalize_retention_records(
            retention_path,
            run_id=run_id,
            run_manifest_path=run_path,
            observation_manifest_path=observation_path,
            workspace_root=tmp_path,
            audit_paths=(replay_cache,),
        )
