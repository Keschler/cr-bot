from __future__ import annotations

import json

import pytest

from cr_bot.annotation_batch import (
    score_locations,
    seal_batch,
    sha256_file,
    validate_batch_policy,
    verify_location_cascade,
)
from cr_bot.annotation_harness import atomic_write_json
from scripts.codex_annotation.evaluate_sealed_multivideo_annotation import _verify_seal
from scripts.codex_annotation.evaluate_blind_annotation import evaluate as evaluate_semantics


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


def _prepare_location_batch(tmp_path):
    run_dir = tmp_path / "clip"
    cascade_dir = run_dir / "location"
    cascade_dir.mkdir(parents=True)
    prediction = {"run_id": "run", "events": []}
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "run_id": "run",
            "segment": {"start_frame": 0, "end_frame_exclusive": 20},
        },
    )
    atomic_write_json(run_dir / "verification.json", prediction)
    prediction_sha256 = sha256_file(run_dir / "verification.json")
    atomic_write_json(
        run_dir / "pipeline_state.json",
        {
            "run_id": "run",
            "status": "semantic_complete",
            "verification_sha256": prediction_sha256,
        },
    )
    package = {
        "run_id": "run",
        "stage": "own_localization_targets",
        "targets": [
            {
                "event_id": "own-10",
                "card": "hog-rider",
                "event_frame_index": 10,
                "review_frame_indices": [10],
                "location_rule_options": ["spawn_center"],
                "macro_review_artifacts": ["reviews/own-10.jpg"],
                "grid_review_artifacts": ["reviews/own-10.jpg"],
            }
        ],
    }
    location_prediction = {
        "run_id": "run",
        "stage": "own_localization_chunk",
        "decisions": [
            {
                "event_id": "own-10",
                "location_frame_index": 10,
                "location_rule": "spawn_center",
                "cell": [3, 18],
                "macro_review_artifacts": ["reviews/own-10.jpg"],
                "grid_review_artifacts": ["reviews/own-10.jpg"],
                "confidence": "direct",
                "reason": "visible initial center",
            }
        ],
    }
    routing = {
        "events": [
            {
                "event_id": "own-10",
                "attempts": [
                    {"role": "luna_marker", "cell": [3, 18], "confidence": "direct"},
                    {"role": "luna_temporal", "cell": [3, 18], "confidence": "direct"},
                    {"role": "terra_residual", "cell": [3, 18], "confidence": "inferred"},
                ],
                "agreement_cluster_indices": [0, 1, 2],
                "agreement_cluster_roles": [
                    "luna_marker",
                    "luna_temporal",
                    "terra_residual",
                ],
                "selected_role": "luna_marker",
                "selected_cell": [3, 18],
            }
        ]
    }
    atomic_write_json(cascade_dir / "aggregate_package.json", package)
    atomic_write_json(cascade_dir / "sealed_prediction.json", location_prediction)
    atomic_write_json(
        cascade_dir / "frozen_policy.json",
        {
            "source_sha256": prediction_sha256,
            "consensus_policy": "exact-inferred",
        },
    )
    policy_sha256 = sha256_file(cascade_dir / "frozen_policy.json")
    (cascade_dir / "packages/reviews").mkdir(parents=True)
    package_index = {"target_count": 1, "packages": ["location/aggregate_package.json"]}
    atomic_write_json(cascade_dir / "packages/package_index.json", package_index)
    (cascade_dir / "packages/reviews/own-10.txt").write_text(
        "sealed evidence", encoding="utf-8"
    )
    atomic_write_json(
        cascade_dir / "PREPARED.json",
        {
            "policy_sha256": policy_sha256,
            "source_sha256": prediction_sha256,
            "package_index_sha256": sha256_file(
                cascade_dir / "packages/package_index.json"
            ),
            "target_count": 1,
            "package_count": 1,
            "package_sha256s": {
                "location/aggregate_package.json": sha256_file(
                    cascade_dir / "aggregate_package.json"
                )
            },
            "evidence_sha256s": {
                "location/packages/reviews/own-10.txt": sha256_file(
                    cascade_dir / "packages/reviews/own-10.txt"
                )
            },
        },
    )
    prepared_sha256 = sha256_file(cascade_dir / "PREPARED.json")
    atomic_write_json(cascade_dir / "routing.json", routing)
    atomic_write_json(cascade_dir / "cost.json", {"weighted_tokens": 1.0})
    atomic_write_json(
        cascade_dir / "SEALED.json",
        {
            "source_sha256": prediction_sha256,
            "prediction_sha256": sha256_file(cascade_dir / "sealed_prediction.json"),
            "policy_sha256": sha256_file(cascade_dir / "frozen_policy.json"),
            "package_sha256": sha256_file(cascade_dir / "aggregate_package.json"),
            "routing_sha256": sha256_file(cascade_dir / "routing.json"),
            "cost_sha256": sha256_file(cascade_dir / "cost.json"),
            "event_count": 1,
        },
    )
    policy = _policy()
    policy["datasets"][0]["location"] = {
        "cascade_dir": "location",
        "policy_sha256": policy_sha256,
        "prepared_sha256": prepared_sha256,
    }
    atomic_write_json(tmp_path / "frozen_batch_policy.json", policy)
    return run_dir, cascade_dir, prediction_sha256


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


def test_batch_seal_revalidates_complete_location_cascade(tmp_path) -> None:
    _, cascade_dir, prediction_sha256 = _prepare_location_batch(tmp_path)

    location_hashes = verify_location_cascade(
        run_dir=tmp_path / "clip",
        cascade_dir=cascade_dir,
        run_id="run",
        source_prediction_sha256=prediction_sha256,
        expected_policy_sha256=sha256_file(cascade_dir / "frozen_policy.json"),
        expected_prepared_sha256=sha256_file(cascade_dir / "PREPARED.json"),
    )
    seal = seal_batch(
        batch_dir=tmp_path,
        policy_path=tmp_path / "frozen_batch_policy.json",
        seal_path=tmp_path / "BATCH_SEALED.json",
    )

    assert seal["datasets"][0]["location"] == location_hashes
    assert set(location_hashes) == {
        "seal_sha256",
        "prepared_sha256",
        "package_index_sha256",
        "prediction_sha256",
        "policy_sha256",
        "package_sha256",
        "routing_sha256",
        "cost_sha256",
        "source_prediction_sha256",
    }
    verified_policy, verified_seal = _verify_seal(
        tmp_path, tmp_path / "frozen_batch_policy.json"
    )
    assert verified_policy["batch_id"] == "test-batch"
    assert verified_seal == seal


def test_batch_seal_rejects_changed_or_wrong_source_location_artifacts(tmp_path) -> None:
    _, cascade_dir, _ = _prepare_location_batch(tmp_path)
    routing = json.loads((cascade_dir / "routing.json").read_text(encoding="utf-8"))
    routing["events"][0]["selected_cell"] = [4, 18]
    atomic_write_json(cascade_dir / "routing.json", routing)

    with pytest.raises(ValueError, match="location routing hash mismatch"):
        seal_batch(
            batch_dir=tmp_path,
            policy_path=tmp_path / "frozen_batch_policy.json",
            seal_path=tmp_path / "BATCH_SEALED.json",
        )


def test_batch_seal_recomputes_consensus_after_self_consistent_tamper(tmp_path) -> None:
    _, cascade_dir, _ = _prepare_location_batch(tmp_path)
    routing = json.loads((cascade_dir / "routing.json").read_text(encoding="utf-8"))
    routing["events"][0]["selected_role"] = "terra_residual"
    atomic_write_json(cascade_dir / "routing.json", routing)
    location_seal = json.loads(
        (cascade_dir / "SEALED.json").read_text(encoding="utf-8")
    )
    location_seal["routing_sha256"] = sha256_file(cascade_dir / "routing.json")
    atomic_write_json(cascade_dir / "SEALED.json", location_seal)

    with pytest.raises(ValueError, match="selected routing consensus is inconsistent"):
        seal_batch(
            batch_dir=tmp_path,
            policy_path=tmp_path / "frozen_batch_policy.json",
            seal_path=tmp_path / "BATCH_SEALED.json",
        )


def test_batch_seal_rejects_changed_prepared_evidence(tmp_path) -> None:
    _, cascade_dir, _ = _prepare_location_batch(tmp_path)
    (cascade_dir / "packages/reviews/own-10.txt").write_text(
        "changed evidence", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="location prepared artifact changed"):
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


def test_semantic_and_location_scoring_use_maximum_cardinality_matching() -> None:
    semantic = evaluate_semantics(
        [
            {"side": "own", "card": "hog-rider", "frame": 4},
            {"side": "own", "card": "hog-rider", "frame": 10},
        ],
        [
            {"side": "own", "card": "hog-rider", "frame": 0},
            {"side": "own", "card": "hog-rider", "frame": 5},
        ],
        start_frame=0,
        end_frame_exclusive=20,
        tolerance_frames=5,
    )
    assert semantic["true_positives"] == 2
    assert [row["frame_error"] for row in semantic["matches"]] == [4, 5]

    location = score_locations(
        truth_events=[
            {"side": "own", "card": "hog-rider", "frame_index": 4, "cell": [2, 18]},
            {"side": "own", "card": "hog-rider", "frame_index": 10, "cell": [7, 20]},
        ],
        package={
            "targets": [
                {"event_id": "first", "card": "hog-rider", "event_frame_index": 0},
                {"event_id": "second", "card": "hog-rider", "event_frame_index": 5},
            ]
        },
        prediction={
            "decisions": [
                {"event_id": "first", "cell": [2, 18]},
                {"event_id": "second", "cell": [7, 20]},
            ]
        },
        frame_tolerance=5,
        cell_tolerance=1,
    )
    assert location["correct"] == 2
    assert location["false_positive_locations"] == 0
