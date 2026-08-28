from __future__ import annotations

from dataclasses import replace
import gzip
import pickle
from pathlib import Path
from types import SimpleNamespace

from simulator.physical_lab import (
    EvidenceStatus,
    ObservationManifest,
    compare_observation_to_replay,
    hog_cannon_probe,
    ingest_extracted_observations,
    run_simulator_replay,
    seal_replay_cache,
)
from simulator.physical_lab.schema import canonical_hash


def _raw_observation() -> dict[str, object]:
    return {
        "entities": [],
        "events": [
            {
                "event_id": "event-1",
                "kind": "hog_deployed",
                "video_time_us": 1_000,
                "match_time_us": 1_000,
                "confidence": 1.0,
                "certainty": "direct",
                "source_frame_indices": [1],
                "evidence_refs": ["capture-A:1"],
                "card_id": "hog-rider",
                "owner": "A",
            }
        ],
    }


def _ingest(*, replay_cache_hash: str | None = None, replay_cache_error: str | None = None):
    spec = hog_cannon_probe(evidence_split="validation")
    return ingest_extracted_observations(
        _raw_observation(),
        run_id="admission-run",
        experiment_hash=spec.experiment_hash(),
        capture_group_id=spec.capture_group_id,
        evidence_split=spec.evidence_split,
        synchronization={"accepted": True},
        replay_cache_hash=replay_cache_hash,
        replay_cache_error=replay_cache_error,
    )


def test_physical_ingest_rejects_missing_replay_cache() -> None:
    manifest = _ingest()

    assert manifest.status is EvidenceStatus.REJECTED
    cache_rejections = [item for item in manifest.rejected if item.record_id == "replay-cache"]
    assert len(cache_rejections) == 1
    assert "recognized replay cache is required" in cache_rejections[0].reason


def test_physical_ingest_preserves_cache_recognition_failure() -> None:
    manifest = _ingest(
        replay_cache_hash="sha256:" + "a" * 64,
        replay_cache_error="replay cache is not recognized by the existing reader",
    )

    assert manifest.status is EvidenceStatus.REJECTED
    assert [item.reason for item in manifest.rejected if item.record_id == "replay-cache"] == [
        "replay cache is not recognized by the existing reader"
    ]


def test_recognized_replay_cache_admits_physical_ingest(tmp_path: Path) -> None:
    cache_path = tmp_path / "replay-cache.pkl.gz"
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump({"schema_version": 1}, handle)
        pickle.dump(
            SimpleNamespace(frame_idx=0, video_time_s=0.0, frame_png=b"encoded-frame"),
            handle,
        )

    seal = seal_replay_cache(cache_path)
    manifest = _ingest(replay_cache_hash=seal.sha256)

    assert seal.recognized is True
    assert manifest.status is EvidenceStatus.VALIDATION
    assert not any(item.record_id == "replay-cache" for item in manifest.rejected)


def test_normalized_evidence_manifest_cannot_bypass_cache_admission() -> None:
    raw: dict[str, object] = {
        "kind": "physical_lab_observation_manifest",
        "schema_version": 1,
        "run_id": "normalized-run",
        "experiment_hash": "sha256:" + "b" * 64,
        "capture_group_id": "normalized-group",
        "evidence_split": "validation",
        "status": "validation",
        "capture_ids": [],
        "media_hashes": {},
        "synchronization": {"accepted": True},
        "entities": [],
        "events": [],
        "rejected": [],
    }
    raw["manifest_hash"] = canonical_hash(raw)

    manifest = ObservationManifest.from_dict(raw)

    assert manifest.status is EvidenceStatus.REJECTED
    assert any(item.record_id == "replay-cache" for item in manifest.rejected)

    attached = replace(manifest, replay_cache_hash="sha256:" + "c" * 64)
    assert attached.status is EvidenceStatus.VALIDATION
    assert attached.rejected == ()


def test_offline_candidate_only_comparison_remains_available_without_cache() -> None:
    spec = hog_cannon_probe()
    observation = ObservationManifest(
        run_id="offline-candidate",
        experiment_hash=spec.experiment_hash(),
        capture_group_id=spec.capture_group_id,
        evidence_split=spec.evidence_split,
        status=EvidenceStatus.CANDIDATE_ONLY,
    )
    replay = run_simulator_replay(spec, action_times={"deploy-cannon": 17_000})

    report = compare_observation_to_replay(observation, replay)

    assert observation.status is EvidenceStatus.CANDIDATE_ONLY
    assert "observation manifest is rejected" not in report.rejection_reasons
    assert not any("replay cache" in reason for reason in report.rejection_reasons)
