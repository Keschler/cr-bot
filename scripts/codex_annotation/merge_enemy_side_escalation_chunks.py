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
from cr_bot.annotation_pipeline import validate_enemy_side_check_decisions


PREFIX = "identity-side-escalation"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate targeted side escalations without rewriting workers."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    index = _read(run_dir / "work_packages" / f"{PREFIX}-index.json")
    decisions = []
    provenance = []
    for relative in index.get("packages", []):
        package_path = run_dir / relative
        output_path = run_dir / "worker_outputs" / package_path.name
        package = _read(package_path)
        output = _read(output_path)
        if output.get("stage") != "enemy_side_check_chunk":
            raise ValueError(f"{output_path}: wrong stage")
        if output.get("run_id") != package.get("run_id"):
            raise ValueError(f"{output_path}: run_id mismatch")
        if output.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        validate_enemy_side_check_decisions(output, package)
        decisions.extend(output["decisions"])
        provenance.append(
            {
                key: output.get(key)
                for key in (
                    "target_range",
                    "annotation_session_id",
                    "model",
                    "reasoning_effort",
                )
            }
        )
    ids = [row["onset_id"] for row in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate enemy-side escalation decision")
    result = {
        "run_id": index.get("run_id"),
        "stage": "enemy_side_escalations",
        "worker_provenance": provenance,
        "decisions": decisions,
    }
    target = (
        run_dir
        / "recovery_outputs"
        / "enemy_side_escalations.json"
    )
    atomic_write_json(target, result)
    print(
        json.dumps(
            {
                "output": str(target),
                "decisions": len(decisions),
                "resolved": sum(
                    row["side"] != "unresolved" for row in decisions
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
