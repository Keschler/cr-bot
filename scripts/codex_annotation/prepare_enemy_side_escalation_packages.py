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
from cr_bot.annotation_pipeline import validate_enemy_side_check_decisions


PATTERN = re.compile(r"identity-side-(\d{6})-(\d{6})\.json")
PREFIX = "identity-side-escalation"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create small Sol escalation packages only for unresolved Terra "
            "enemy-side decisions."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    packages_dir = run_dir / "work_packages"
    outputs_dir = run_dir / "worker_outputs"
    package_paths = []
    candidate_count = 0
    run_id: str | None = None
    for base_output_path in sorted(outputs_dir.iterdir()):
        match = PATTERN.fullmatch(base_output_path.name)
        if match is None:
            continue
        base_package_path = packages_dir / base_output_path.name
        base_package = _read(base_package_path)
        base_output = _read(base_output_path)
        validate_enemy_side_check_decisions(base_output, base_package)
        if base_output.get("model") is None or "terra" not in str(
            base_output["model"]
        ).lower():
            raise ValueError(
                f"{base_output_path}: base side output is not Terra"
            )
        if run_id is None:
            run_id = str(base_package["run_id"])
        unresolved_ids = {
            row["onset_id"]
            for row in base_output["decisions"]
            if row["side"] == "unresolved"
        }
        if not unresolved_ids:
            continue
        candidates = [
            row
            for row in base_package["candidates"]
            if row["onset_id"] in unresolved_ids
        ]
        if {row["onset_id"] for row in candidates} != unresolved_ids:
            raise ValueError(
                f"{base_output_path}: unresolved side candidates are missing"
            )
        start, end = (int(value) for value in match.groups())
        target = (
            packages_dir
            / f"{PREFIX}-{start:06d}-{end:06d}.json"
        )
        atomic_write_json(
            target,
            {
                **{
                    key: base_package[key]
                    for key in (
                        "run_id",
                        "fps",
                        "target_range",
                        "decision_schema_version",
                    )
                },
                "task": "resolve_unresolved_enemy_side",
                "candidates": candidates,
            },
        )
        package_paths.append(target)
        candidate_count += len(candidates)
    index = packages_dir / f"{PREFIX}-index.json"
    atomic_write_json(
        index,
        {
            "run_id": run_id,
            "stage": "enemy_side_escalation_package_index",
            "packages": [
                str(path.relative_to(run_dir)) for path in package_paths
            ],
            "candidate_count": candidate_count,
        },
    )
    print(
        json.dumps(
            {
                "index": str(index),
                "packages": len(package_paths),
                "candidates": candidate_count,
            }
        )
    )


if __name__ == "__main__":
    main()
