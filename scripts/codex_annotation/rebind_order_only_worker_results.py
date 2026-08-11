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

from cr_bot.annotation_pipeline import (
    MODEL_PROFILES,
    atomic_write_state,
    job_fingerprint,
    sha256_file,
    validate_enemy_existence_decisions,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _reconstructed_sha256(
    package: dict[str, Any], attached_images: list[str], temporary_path: Path
) -> str:
    order = [Path(value).name for value in attached_images]
    by_artifact = {
        Path(row["focus_artifact"]).name: row for row in package["candidates"]
    }
    if set(order) != set(by_artifact) or len(order) != len(by_artifact):
        raise ValueError("attached images do not exactly cover package candidates")
    reconstructed = {**package, "candidates": [by_artifact[name] for name in order]}
    atomic_write_state(temporary_path, reconstructed)
    digest = sha256_file(temporary_path)
    temporary_path.unlink()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind validated overlap outputs after proving that only a formerly "
            "nondeterministic equal-frame candidate order changed."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "pipeline_state.json"
    state = _read(state_path)
    spec = MODEL_PROFILES[state["profile"]]["enemy_existence"]
    prompt = ROOT / "scripts/codex_annotation/prompts/enemy_overlap_adjudication_chunk.txt"
    rebound = []
    for package_path in sorted(
        (run_dir / "work_packages").glob("identity-overlap-*.json")
    ):
        stable = run_dir / "worker_outputs" / package_path.name
        if not stable.is_file():
            continue
        job_id = f"enemy-existence:{package_path.stem}"
        old = state["jobs"].get(job_id)
        if not isinstance(old, dict) or old.get("status") != "succeeded":
            raise ValueError(f"{job_id}: no succeeded state to rebind")
        package = _read(package_path)
        output = _read(stable)
        validate_enemy_existence_decisions(output, package)
        if old.get("output_sha256") != sha256_file(stable):
            raise ValueError(f"{job_id}: stable output hash changed")
        attached_images = old.get("attached_images")
        if not isinstance(attached_images, list):
            raise ValueError(f"{job_id}: attached image order is unavailable")
        temporary_path = run_dir / "work_packages" / ".order-reconstruction.json"
        if _reconstructed_sha256(package, attached_images, temporary_path) != old.get(
            "package_sha256"
        ):
            raise ValueError(f"{job_id}: old package was not an order-only permutation")
        current = job_fingerprint(
            package=package_path,
            prompt=prompt,
            model_spec=spec,
        )
        changed = {
            key
            for key in current
            if old.get(key) != current.get(key)
        }
        if changed - {"package_sha256"}:
            raise ValueError(f"{job_id}: non-package fingerprint changed: {changed}")
        state["jobs"][job_id] = {
            **old,
            **current,
            "rebound_order_only": True,
        }
        rebound.append(job_id)
    if not rebound:
        raise ValueError("no overlap worker results were rebound")
    atomic_write_state(state_path, state)
    print(json.dumps({"rebound": rebound}))


if __name__ == "__main__":
    main()
