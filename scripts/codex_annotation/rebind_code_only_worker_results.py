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

from cr_bot.annotation_pipeline import atomic_write_state, sha256_file


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind valid stable outputs after proving that only worker and "
            "validator code hashes changed."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "pipeline_state.json"
    state = _read(state_path)
    validator_sha256 = sha256_file(ROOT / "src/cr_bot/annotation_pipeline.py")
    worker_sha256 = sha256_file(
        ROOT / "scripts/codex_annotation/run_model_worker.py"
    )
    rebound = []
    for job_id, row in state.get("jobs", {}).items():
        if not isinstance(row, dict) or row.get("status") != "succeeded":
            continue
        stem = job_id.split(":", 1)[1]
        package = run_dir / "work_packages" / f"{stem}.json"
        stable = run_dir / "worker_outputs" / f"{stem}.json"
        if not package.is_file() or not stable.is_file():
            raise ValueError(f"{job_id}: package or stable output is missing")
        if row.get("package_sha256") != sha256_file(package):
            raise ValueError(f"{job_id}: package changed")
        if row.get("output_sha256") != sha256_file(stable):
            raise ValueError(f"{job_id}: stable output changed")
        for reference, expected in row.get("evidence_sha256", {}).items():
            artifact = run_dir / reference
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise ValueError(f"{job_id}: evidence changed: {reference}")
        old_pair = (row.get("validator_sha256"), row.get("worker_sha256"))
        new_pair = (validator_sha256, worker_sha256)
        if old_pair == new_pair:
            continue
        row["validator_sha256"] = validator_sha256
        row["worker_sha256"] = worker_sha256
        row["rebound_code_only"] = True
        rebound.append(job_id)
    if not rebound:
        raise ValueError("no succeeded worker results required code-only rebinding")
    atomic_write_state(state_path, state)
    print(json.dumps({"rebound": rebound}))


if __name__ == "__main__":
    main()
