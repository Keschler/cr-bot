from __future__ import annotations

import json

import pytest

from cr_bot.annotation_batch import (
    score_locations,
    seal_batch,
    sha256_file,
    validate_batch_policy,
)
from cr_bot.annotation_harness import atomic_write_json


def _policy() -> dict[str, object]:
    return {
        "version": 1,
        "batch_id": "test-batch",
        "datasets": [
            {
                "id": "clip",
                "run_dir": "clip",
                "prediction": "verification.json",
                "semantic_scopes": [
                    {"side": "own", "range": [0, 20]},
                    {"side": "enemy", "range": [0, 10]},
                ],
            }
        ],
    }


def test_batch_policy_rejects_reference_inputs() -> None:
    policy = _policy()
    policy["ground_truth"] = "labels.json"

    with pytest.raises(ValueError, match="forbidden reference key"):
        validate_batch_policy(policy)


def test_batch_seal_requires_complete_hash_matched_semantics(tmp_path) -> None:
    run_dir = tmp_path / "clip"
    run_dir.mkdir()
    prediction = {"run_id": "run", "events": []}
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "run_id": "run",
            "segment": {"start_frame": 0, "end_frame_exclusive": 20},
        },
    )
    atomic_write_json(run_dir / "verification.json", prediction)
    atomic_write_json(
        run_dir / "pipeline_state.json",
        {
            "run_id": "run",
            "status": "semantic_complete",
            "verification_sha256": sha256_file(run_dir / "verification.json"),
        },
    )
    atomic_write_json(tmp_path / "frozen_batch_policy.json", _policy())

    seal = seal_batch(
        batch_dir=tmp_path,
        policy_path=tmp_path / "frozen_batch_policy.json",
        seal_path=tmp_path / "BATCH_SEALED.json",
    )

    assert seal["batch_id"] == "test-batch"
    assert seal["datasets"][0]["prediction_sha256"] == sha256_file(
        run_dir / "verification.json"
    )
    with pytest.raises(ValueError, match="already sealed"):
        seal_batch(
            batch_dir=tmp_path,
            policy_path=tmp_path / "frozen_batch_policy.json",
            seal_path=tmp_path / "BATCH_SEALED.json",
        )


def test_location_score_counts_missing_and_extra_predictions() -> None:
    truth = [
        {"side": "own", "card": "hog-rider", "frame_index": 10, "cell": [5, 20]},
        {"side": "own", "card": "ice-golem", "frame_index": 30, "cell": [8, 21]},
        {"side": "own", "card": "log", "frame_index": 40},
    ]
    package = {
        "targets": [
            {"event_id": "hog", "card": "hog-rider", "event_frame_index": 11},
            {"event_id": "extra", "card": "fireball", "event_frame_index": 50},
        ]
    }
    prediction = {
        "decisions": [
            {"event_id": "hog", "cell": [6, 19]},
            {"event_id": "extra", "cell": [4, 18]},
        ]
    }

    report = score_locations(
        truth_events=truth,
        package=package,
        prediction=prediction,
        frame_tolerance=5,
        cell_tolerance=1,
    )

    assert report["expected"] == 2
    assert report["predicted"] == 2
    assert report["correct"] == 1
    assert report["incorrect"] == 1
    assert report["false_positive_locations"] == 1
