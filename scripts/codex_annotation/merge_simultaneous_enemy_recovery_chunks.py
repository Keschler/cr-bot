from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import validate_enemy_existence_decisions


RECOVERY_PREFIX = "identity-simultaneous-recovery"
CANONICAL_SEMANTIC_NAMES = {
    "enemy_onsets.json",
    "enemy_identities.json",
    "enemy_cards.json",
    "verification.json",
    "localization.json",
    "completeness.json",
    "decisions.json",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _cluster_rows(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    tolerance_frames: int = 5,
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    clusters: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    for item in sorted(rows, key=lambda value: value[0]["event_frame_index"]):
        frame = item[0]["event_frame_index"]
        if (
            not clusters
            or frame - clusters[-1][0][0]["event_frame_index"]
            > tolerance_frames
        ):
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def merge_recoveries(
    run_dir: Path,
    *,
    worker_outputs: list[Path] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    index = _read(
        run_dir / "work_packages" / f"{RECOVERY_PREFIX}-index.json"
    )
    package_paths = [
        run_dir / value for value in index.get("packages", [])
    ]
    if len(package_paths) != len(set(package_paths)):
        raise ValueError("recovery package index contains duplicates")

    if worker_outputs is None:
        output_paths = [
            run_dir / "worker_outputs" / package_path.name
            for package_path in package_paths
        ]
    else:
        output_paths = [path.resolve() for path in worker_outputs]
        if len(output_paths) != len(package_paths):
            raise ValueError(
                "worker outputs must cover indexed packages exactly"
            )
    outputs_by_range: dict[tuple[int, int], Path] = {}
    for output_path in output_paths:
        document = _read(output_path)
        target = document.get("target_range")
        if (
            not isinstance(target, list)
            or len(target) != 2
            or not all(isinstance(value, int) for value in target)
        ):
            raise ValueError(f"{output_path}: invalid target_range")
        key = (target[0], target[1])
        if key in outputs_by_range:
            raise ValueError(f"duplicate worker output range {key}")
        outputs_by_range[key] = output_path

    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    provenance = []
    for package_path in package_paths:
        package = _read(package_path)
        target = tuple(package["target_range"])
        output_path = outputs_by_range.get(target)
        if output_path is None:
            raise ValueError(f"missing worker output for range {target}")
        document = _read(output_path)
        if document.get("stage") != "enemy_overlap_adjudication_chunk":
            raise ValueError(f"{output_path}: wrong recovery stage")
        if document.get("run_id") != package.get("run_id"):
            raise ValueError(f"{output_path}: run_id mismatch")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        validate_enemy_existence_decisions(document, package)
        candidates = {
            row["onset_id"]: row for row in package["candidates"]
        }
        for row in document["decisions"]:
            if row.get("overlap_event_exists") is True:
                accepted.append((row, candidates[row["onset_id"]]))
        provenance.append(
            {
                "target_range": package["target_range"],
                "worker_output": str(output_path),
                "annotation_session_id": document.get(
                    "annotation_session_id"
                ),
                "model": document.get("model"),
                "reasoning_effort": document.get("reasoning_effort"),
            }
        )

    recoveries = []
    for cluster in _cluster_rows(accepted):
        representative, _ = cluster[0]
        frame = min(row["event_frame_index"] for row, _ in cluster)
        source_onset_ids = [row["onset_id"] for row, _ in cluster]
        artifacts = []
        sampled_frames = set()
        for _, candidate in cluster:
            sampled_frames.update(candidate["sampled_frame_indices"])
            for key in ("full_arena_artifact", "focus_artifact"):
                artifact = candidate[key]
                if artifact not in artifacts:
                    artifacts.append(artifact)
        recoveries.append(
            {
                "onset_id": (
                    f"enemy-simultaneous-unit-{frame:06d}-"
                    f"from-{source_onset_ids[0]}"
                ),
                "source_onset_ids": source_onset_ids,
                "event_frame_index": frame,
                "kind": "unit_or_building",
                "side": "enemy",
                "sampled_frame_indices": sorted(sampled_frames),
                "verification_artifacts": artifacts,
                "evidence": representative["evidence"],
                "reason": representative.get("reason", ""),
            }
        )

    result = {
        "run_id": index.get("run_id"),
        "stage": "enemy_simultaneous_recoveries",
        "worker_provenance": provenance,
        "recoveries": recoveries,
    }
    target = (
        output.resolve()
        if output is not None
        else (
            run_dir
            / "recovery_outputs"
            / "enemy_simultaneous_recoveries.json"
        )
    )
    if target.name in CANONICAL_SEMANTIC_NAMES:
        raise ValueError(
            f"refusing to overwrite canonical semantic file {target.name}"
        )
    atomic_write_json(target, result)
    return {
        "output": str(target),
        "recoveries": len(recoveries),
        "source_candidates": len(accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and merge simultaneous-enemy recovery chunks into a "
            "standalone, non-canonical recovery artifact."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            merge_recoveries(
                args.run_dir,
                worker_outputs=args.worker_output,
                output=args.output,
            )
        )
    )


if __name__ == "__main__":
    main()
