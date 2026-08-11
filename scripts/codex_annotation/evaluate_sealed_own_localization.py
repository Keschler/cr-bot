from __future__ import annotations

import argparse
import hashlib
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

from cr_bot.annotation_harness import atomic_write_json
from scripts.codex_annotation.evaluate_own_localization import evaluate_locations


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a hash-sealed localization run once and lock its source."
    )
    parser.add_argument("--cascade-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--frame-tolerance", type=int, default=5)
    parser.add_argument("--cell-tolerance", type=int, default=1)
    args = parser.parse_args()
    cascade_dir = args.cascade_dir.resolve()
    seal_path = cascade_dir / "SEALED.json"
    prediction_path = cascade_dir / "sealed_prediction.json"
    policy_path = cascade_dir / "frozen_policy.json"
    package_path = cascade_dir / "aggregate_package.json"
    evaluation_path = cascade_dir / "evaluation.json"
    evaluated_lock_path = cascade_dir / "EVALUATED.json"
    if evaluated_lock_path.exists():
        raise ValueError("sealed cascade has already been evaluated")

    seal = _read(seal_path)
    if _sha256(prediction_path) != seal.get("prediction_sha256"):
        raise ValueError("sealed prediction hash mismatch")
    if _sha256(policy_path) != seal.get("policy_sha256"):
        raise ValueError("frozen policy hash mismatch")
    if evaluation_path.exists() and evaluation_path.stat().st_mtime <= seal_path.stat().st_mtime:
        raise ValueError("existing evaluation does not postdate the seal")

    truth = _read(args.ground_truth)
    package = _read(package_path)
    prediction = _read(prediction_path)
    report = evaluate_locations(
        truth["events"],
        package,
        prediction,
        frame_tolerance=args.frame_tolerance,
        cell_tolerance=args.cell_tolerance,
    )
    if evaluation_path.exists():
        if _read(evaluation_path) != report:
            raise ValueError("existing post-seal evaluation differs from recomputed report")
    else:
        atomic_write_json(evaluation_path, report)

    policy = _read(policy_path)
    source_sha256 = policy.get("source_sha256")
    if not isinstance(source_sha256, str):
        # v1 was sealed before source hashing was added to the policy. Its
        # package index still records the exact source file used.
        index = _read(cascade_dir / "packages-primary" / "package_index.json")
        source_sha256 = _sha256(Path(index["source_file"]))
    lock = {
        "run_id": prediction["run_id"],
        "source_sha256": source_sha256,
        "prediction_sha256": seal["prediction_sha256"],
        "policy_sha256": seal["policy_sha256"],
        "evaluation_sha256": _sha256(evaluation_path),
        "ground_truth_sha256": _sha256(args.ground_truth.resolve()),
        "correct": report["correct"],
        "expected": report["expected"],
    }
    atomic_write_json(evaluated_lock_path, lock)

    run_dir = cascade_dir
    while run_dir.name != "own_localization_label_independent":
        if run_dir.parent == run_dir:
            raise ValueError("cascade directory is not under own_localization_label_independent")
        run_dir = run_dir.parent
    registry_path = run_dir / "EVALUATED_RUNS.json"
    registry = _read(registry_path) if registry_path.exists() else {"evaluated": []}
    rows = registry.get("evaluated")
    if not isinstance(rows, list):
        raise ValueError("invalid evaluated-run registry")
    if any(row.get("source_sha256") == source_sha256 for row in rows):
        raise ValueError("semantic source is already present in evaluated-run registry")
    rows.append({**lock, "cascade_dir": str(cascade_dir)})
    atomic_write_json(registry_path, registry)
    print(json.dumps({"evaluation": str(evaluation_path), "correct": report["correct"], "expected": report["expected"]}))


if __name__ == "__main__":
    main()
