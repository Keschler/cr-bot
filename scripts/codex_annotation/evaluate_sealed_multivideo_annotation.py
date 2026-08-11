from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_batch import (
    read_object,
    resolve_inside,
    score_locations,
    sha256_file,
    validate_batch_policy,
)
from cr_bot.annotation_harness import atomic_write_json
from scripts.codex_annotation.evaluate_blind_annotation import _load_events, evaluate


def _summary(counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["true_positives"]
    expected = counts["expected"]
    predicted = counts["predicted"]
    precision = tp / predicted if predicted else (1.0 if not expected else 0.0)
    recall = tp / expected if expected else 1.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _verify_seal(batch_dir: Path, policy_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = read_object(policy_path)
    validate_batch_policy(policy)
    seal = read_object(batch_dir / "BATCH_SEALED.json")
    if seal.get("policy_sha256") != sha256_file(policy_path):
        raise ValueError("batch policy hash mismatch")
    if seal.get("batch_id") != policy.get("batch_id"):
        raise ValueError("batch id mismatch")
    sealed = {row["id"]: row for row in seal.get("datasets", [])}
    for dataset in policy["datasets"]:
        row = sealed.get(dataset["id"])
        if row is None:
            raise ValueError(f"dataset {dataset['id']} is absent from batch seal")
        run_dir = resolve_inside(batch_dir, dataset["run_dir"])
        checks = {
            "manifest_sha256": run_dir / "manifest.json",
            "pipeline_state_sha256": run_dir / "pipeline_state.json",
            "prediction_sha256": resolve_inside(run_dir, dataset["prediction"]),
        }
        for key, path in checks.items():
            if row.get(key) != sha256_file(path):
                raise ValueError(f"dataset {dataset['id']} changed after sealing")
        if "location" in dataset:
            cascade_dir = resolve_inside(run_dir, dataset["location"]["cascade_dir"])
            location_checks = {
                "seal_sha256": cascade_dir / "SEALED.json",
                "prediction_sha256": cascade_dir / "sealed_prediction.json",
                "policy_sha256": cascade_dir / "frozen_policy.json",
                "package_sha256": cascade_dir / "aggregate_package.json",
            }
            for key, path in location_checks.items():
                if row.get("location", {}).get(key) != sha256_file(path):
                    raise ValueError(
                        f"dataset {dataset['id']} location changed after sealing"
                    )
    if len(sealed) != len(policy["datasets"]):
        raise ValueError("batch seal contains unexpected datasets")
    return policy, seal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a complete hash-sealed multi-video annotation batch once."
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--tolerance-frames", type=int, default=5)
    parser.add_argument("--cell-tolerance", type=int, default=1)
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve()
    evaluation_path = batch_dir / "batch_evaluation.json"
    lock_path = batch_dir / "BATCH_EVALUATED.json"
    if evaluation_path.exists() or lock_path.exists():
        raise ValueError("sealed batch has already been evaluated")
    policy_path = batch_dir / "frozen_batch_policy.json"
    policy, seal = _verify_seal(batch_dir, policy_path)
    evaluation_manifest = read_object(args.evaluation_manifest.resolve())
    if evaluation_manifest.get("batch_id") != policy["batch_id"]:
        raise ValueError("evaluation manifest batch id mismatch")
    truth_paths = evaluation_manifest.get("ground_truth")
    if not isinstance(truth_paths, dict) or set(truth_paths) != {
        row["id"] for row in policy["datasets"]
    }:
        raise ValueError("evaluation manifest must map every sealed dataset exactly once")

    totals = {key: 0 for key in (
        "expected", "predicted", "true_positives", "false_positives", "false_negatives"
    )}
    dataset_reports = []
    location_totals = {
        "expected": 0,
        "predicted": 0,
        "correct": 0,
        "incorrect": 0,
        "false_positive_locations": 0,
    }
    for dataset in policy["datasets"]:
        run_dir = resolve_inside(batch_dir, dataset["run_dir"])
        truth_path = (ROOT / truth_paths[dataset["id"]]).resolve()
        expected = _load_events(truth_path, predicted=False)
        predicted = _load_events(
            resolve_inside(run_dir, dataset["prediction"]), predicted=True
        )
        scopes = []
        for scope in dataset["semantic_scopes"]:
            side = scope["side"]
            report = evaluate(
                [row for row in expected if row["side"] == side],
                [row for row in predicted if row["side"] == side],
                start_frame=scope["range"][0],
                end_frame_exclusive=scope["range"][1],
                tolerance_frames=args.tolerance_frames,
            )
            scopes.append({"side": side, **report})
            for key in totals:
                totals[key] += int(report[key])
        dataset_report: dict[str, Any] = {"id": dataset["id"], "semantic_scopes": scopes}
        if "location" in dataset:
            cascade_dir = resolve_inside(run_dir, dataset["location"]["cascade_dir"])
            truth_document = read_object(truth_path)
            location = score_locations(
                truth_events=truth_document["events"],
                package=read_object(cascade_dir / "aggregate_package.json"),
                prediction=read_object(cascade_dir / "sealed_prediction.json"),
                frame_tolerance=args.tolerance_frames,
                cell_tolerance=args.cell_tolerance,
            )
            dataset_report["own_locations"] = location
            for key in location_totals:
                location_totals[key] += int(location[key])
        dataset_reports.append(dataset_report)
    semantic = _summary(totals)
    location_totals["accuracy"] = (
        location_totals["correct"] / location_totals["expected"]
        if location_totals["expected"]
        else 1.0
    )
    success = (
        semantic["precision"] == 1.0
        and semantic["recall"] == 1.0
        and location_totals["accuracy"] == 1.0
        and location_totals["false_positive_locations"] == 0
    )
    report = {
        "batch_id": policy["batch_id"],
        "tolerance_frames": args.tolerance_frames,
        "cell_tolerance_per_coordinate": args.cell_tolerance,
        "semantic": semantic,
        "own_locations": location_totals,
        "success_100_percent": success,
        "datasets": dataset_reports,
    }
    atomic_write_json(evaluation_path, report)
    atomic_write_json(
        lock_path,
        {
            "batch_id": policy["batch_id"],
            "batch_seal_sha256": sha256_file(batch_dir / "BATCH_SEALED.json"),
            "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest.resolve()),
            "evaluation_sha256": sha256_file(evaluation_path),
            "ground_truth_sha256s": {
                dataset_id: sha256_file((ROOT / path).resolve())
                for dataset_id, path in truth_paths.items()
            },
            "success_100_percent": success,
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
