from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_pipeline import (
    ModelSpec,
    atomic_write_state,
    job_fingerprint,
    sha256_file,
    validate_enemy_identity_decisions,
    validate_enemy_unit_decisions,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adopt a validated controlled worker experiment into pipeline state."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--cost-multiplier", type=float, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stable-output", type=Path, required=True)
    parser.add_argument("--expected-stage", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    package = _read(args.package)
    document = _read(args.attempt)
    result = _read(args.result)
    state_path = run_dir / "pipeline_state.json"
    state = _read(state_path)
    if result.get("status") != "succeeded":
        raise ValueError("only a succeeded worker result can be adopted")
    if result.get("model") != args.model:
        raise ValueError("result model mismatch")
    if result.get("reasoning_effort") != args.reasoning_effort:
        raise ValueError("result reasoning effort mismatch")
    if result.get("cost_multiplier") != args.cost_multiplier:
        raise ValueError("result cost multiplier mismatch")
    if result.get("output_sha256") != sha256_file(args.attempt):
        raise ValueError("result hash does not match attempt")
    if document.get("run_id") != package.get("run_id"):
        raise ValueError("attempt run_id mismatch")
    if document.get("stage") != args.expected_stage:
        raise ValueError("attempt stage mismatch")
    if document.get("target_range") != package.get("target_range"):
        raise ValueError("attempt target_range mismatch")
    if args.expected_stage in {
        "enemy_unit_onsets_chunk",
        "enemy_unit_completeness_chunk",
    }:
        validate_enemy_unit_decisions(document, package)
    elif args.expected_stage == "enemy_identities_chunk":
        validate_enemy_identity_decisions(document, package)
    if args.stable_output.exists():
        raise FileExistsError(args.stable_output)

    args.stable_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.attempt, args.stable_output)
    spec = ModelSpec(
        args.model,
        args.reasoning_effort,
        args.cost_multiplier,
    )
    state["jobs"][args.job_id] = {
        **job_fingerprint(
            package=args.package,
            prompt=args.prompt,
            model_spec=spec,
        ),
        **result,
        "status": "succeeded",
        "adopted_controlled_experiment": True,
        "output_sha256": sha256_file(args.stable_output),
    }
    atomic_write_state(state_path, state)
    print(
        json.dumps(
            {
                "job": args.job_id,
                "stable_output": str(args.stable_output),
                "weighted_tokens": result.get("weighted_tokens"),
            }
        )
    )


if __name__ == "__main__":
    main()
