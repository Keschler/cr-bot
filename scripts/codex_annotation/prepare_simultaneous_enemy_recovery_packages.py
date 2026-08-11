from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import (
    validate_enemy_existence_decisions,
    validate_enemy_side_check_decisions,
)


SIDE_OUTPUT_PATTERN = re.compile(
    r"identity-side-(\d{6})-(\d{6})\.json"
)
RECOVERY_PREFIX = "identity-simultaneous-recovery"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _checked_document(
    path: Path,
    package: dict[str, Any],
    *,
    expected_stage: str,
) -> dict[str, Any]:
    document = _read(path)
    if document.get("stage") != expected_stage:
        raise ValueError(f"{path}: expected stage {expected_stage!r}")
    if document.get("run_id") != package.get("run_id"):
        raise ValueError(f"{path}: run_id mismatch")
    if document.get("target_range") != package.get("target_range"):
        raise ValueError(f"{path}: target_range mismatch")
    return document


def prepare_packages(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    package_dir = run_dir / "work_packages"
    output_dir = run_dir / "worker_outputs"
    side_outputs = sorted(
        path
        for path in output_dir.iterdir()
        if SIDE_OUTPUT_PATTERN.fullmatch(path.name)
    )
    if not side_outputs:
        raise ValueError("no canonical enemy side outputs")

    package_paths: list[Path] = []
    total_candidates = 0
    run_id: str | None = None
    for side_output_path in side_outputs:
        match = SIDE_OUTPUT_PATTERN.fullmatch(side_output_path.name)
        assert match is not None
        start, end = (int(value) for value in match.groups())
        suffix = f"{start:06d}-{end:06d}.json"
        side_package = _read(package_dir / f"identity-side-{suffix}")
        side_document = _checked_document(
            side_output_path,
            side_package,
            expected_stage="enemy_side_check_chunk",
        )
        model = side_document.get("model")
        if not isinstance(model, str) or "terra" not in model.lower():
            raise ValueError(
                f"{side_output_path}: canonical side output is not Terra"
            )
        validate_enemy_side_check_decisions(side_document, side_package)

        existence_package = _read(
            package_dir / f"identity-overlap-{suffix}"
        )
        existence_document = _checked_document(
            output_dir / f"identity-overlap-{suffix}",
            existence_package,
            expected_stage="enemy_overlap_adjudication_chunk",
        )
        validate_enemy_existence_decisions(
            existence_document, existence_package
        )
        if run_id is None:
            run_id = str(side_package["run_id"])
        elif side_package.get("run_id") != run_id:
            raise ValueError("side packages do not share a run_id")

        existence_rows = {
            row["onset_id"]: row
            for row in existence_document["decisions"]
            if row.get("overlap_event_exists") is True
        }
        existence_candidates = {
            row["onset_id"]: row
            for row in existence_package["candidates"]
        }
        side_candidates = {
            row["onset_id"]: row for row in side_package["candidates"]
        }
        candidates = []
        for side_row in side_document["decisions"]:
            onset_id = side_row["onset_id"]
            existence_row = existence_rows.get(onset_id)
            if side_row.get("side") != "own" or existence_row is None:
                continue
            existence_candidate = existence_candidates[onset_id]
            side_candidate = side_candidates[onset_id]
            candidates.append(
                {
                    "onset_id": onset_id,
                    "event_frame_index": existence_row[
                        "event_frame_index"
                    ],
                    "sampled_frame_indices": existence_candidate[
                        "sampled_frame_indices"
                    ],
                    "full_arena_artifact": side_candidate[
                        "full_arena_artifact"
                    ],
                    "focus_artifact": existence_candidate["focus_artifact"],
                    "known_own_actor": {
                        "side": "own",
                        "direct": True,
                        "team_indicator": side_row.get("team_indicator"),
                        "origin": side_row.get("origin"),
                        "motion": side_row.get("motion"),
                        "reason": side_row.get("reason", ""),
                    },
                    "existence_reason": existence_row.get("reason", ""),
                }
            )
        if not candidates:
            continue
        target = (
            package_dir
            / f"{RECOVERY_PREFIX}-{start:06d}-{end:06d}.json"
        )
        atomic_write_json(
            target,
            {
                "run_id": side_package["run_id"],
                "fps": side_package["fps"],
                "target_range": [start, end],
                "decision_schema_version": 2,
                "task": "additional_simultaneous_enemy_actor",
                "candidates": candidates,
            },
        )
        package_paths.append(target)
        total_candidates += len(candidates)

    index_path = package_dir / f"{RECOVERY_PREFIX}-index.json"
    atomic_write_json(
        index_path,
        {
            "run_id": run_id,
            "stage": "enemy_simultaneous_recovery_package_index",
            "packages": [
                str(path.relative_to(run_dir)) for path in package_paths
            ],
            "candidate_count": total_candidates,
        },
    )
    return {
        "index": str(index_path),
        "packages": len(package_paths),
        "candidates": total_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare focused recovery packages for existence-confirmed "
            "candidates whose canonical Terra side verdict is own."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_packages(args.run_dir)))


if __name__ == "__main__":
    main()
