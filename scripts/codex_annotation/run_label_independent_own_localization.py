from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.own_localization import validate_own_localization_decisions
from cr_bot.own_localization_cascade import (
    route_after_primary,
    route_after_terra,
    route_after_tiebreak,
    select_medoid,
)
from scripts.codex_annotation.prepare_own_localization_packages import prepare_packages


PROMPTS = {
    "primary_marker": ROOT / "scripts/codex_annotation/prompts/own_localization_marker_first.txt",
    "primary_badge": ROOT / "scripts/codex_annotation/prompts/own_localization_badge_anchor.txt",
    "luna_tiebreak": ROOT / "scripts/codex_annotation/prompts/own_localization_chunk.txt",
    "terra": ROOT / "scripts/codex_annotation/prompts/own_localization_chunk.txt",
    "sol": ROOT / "scripts/codex_annotation/prompts/own_localization_chunk.txt",
}
MODELS = {
    "primary_marker": ("gpt-5.6-luna", "low"),
    "primary_badge": ("gpt-5.6-luna", "low"),
    "luna_tiebreak": ("gpt-5.6-luna", "low"),
    "terra": ("gpt-5.6-terra", "medium"),
    "sol": ("gpt-5.6-sol", "high"),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_worker(
    *, run_dir: Path, cascade_dir: Path, package_path: Path, tier: str
) -> dict[str, Any]:
    model, effort = MODELS[tier]
    stem = package_path.parent.name
    output = cascade_dir / "worker_outputs" / tier / f"{stem}.json"
    if output.is_file():
        document = _read(output)
        validate_own_localization_decisions(document, _read(package_path))
        return document
    for attempt in (1, 2):
        expected = package_path.parent / f"worker-output-{tier}-a{attempt}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/codex_annotation/run_model_worker.py"),
            "--model", model,
            "--reasoning-effort", effort,
            "--prompt-file", str(PROMPTS[tier]),
            "--run-dir", str(package_path.parent),
            "--session-id", f"label-independent-{tier}-{stem}-a{attempt}",
            "--workdir", str(package_path.parent),
            "--log-dir", str(cascade_dir / "logs" / tier / stem),
            "--label", f"{tier}-{stem}-a{attempt}",
            "--expected-output", str(expected),
            "--expected-stage", "own_localization_chunk",
            "--expected-package", str(package_path),
            "--promote-to", str(output),
            "--result-file", str(cascade_dir / "results" / tier / f"{stem}-a{attempt}.json"),
            "--prompt-var", "PACKAGE_FILE=package.json",
            "--prompt-var", f"OUTPUT_FILE={expected.name}",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode == 0:
            return _read(output)
        if completed.returncode != 65:
            raise RuntimeError(f"{tier} worker failed for {stem}: {completed.returncode}")
    raise RuntimeError(f"{tier} worker stayed structurally invalid for {stem}")


def _rows(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["event_id"]: row for row in document["decisions"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen own-location cascade with no ground-truth input."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cascade_dir = args.output_dir.resolve()
    cascade_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = _sha256(args.source_file.resolve())
    registry_path = run_dir / "own_localization_label_independent" / "EVALUATED_RUNS.json"
    if registry_path.is_file():
        registry = _read(registry_path)
        evaluated = registry.get("evaluated", [])
        if any(row.get("source_sha256") == source_sha256 for row in evaluated):
            raise ValueError(
                "this semantic source has already been evaluated; use a different "
                "unscored run instead of tuning on its failures"
            )

    policy_path = cascade_dir / "frozen_policy.json"
    policy = {
        "version": 1,
        "source_sha256": source_sha256,
        "ground_truth_access": "prohibited_until_sealed_prediction_exists",
        "primary": ["primary_marker", "primary_badge"],
        "primary_acceptance": "two direct legal cells within Chebyshev distance 1",
        "luna_tiebreak": "run for every event failing primary agreement",
        "terra_escalation": "run when no direct legal pair agrees within 1 after tiebreak",
        "sol_escalation": "run when no direct legal pair agrees within 1 after Terra",
        "selection": "stable medoid of all direct legal decisions; inferred only if none direct",
        "models": {tier: {"model": value[0], "effort": value[1]} for tier, value in MODELS.items()},
        "prompt_sha256s": {tier: _sha256(path) for tier, path in PROMPTS.items()},
    }
    if policy_path.exists() and _read(policy_path) != policy:
        raise ValueError("frozen policy changed; use a fresh output directory")
    atomic_write_json(policy_path, policy)

    primary_dir = cascade_dir / "packages-primary"
    individual_dir = cascade_dir / "packages-individual"
    primary_index = prepare_packages(
        run_dir=run_dir,
        source_file=args.source_file.resolve(),
        output_dir=primary_dir,
        chunk_size=args.chunk_size,
    )
    individual_index = prepare_packages(
        run_dir=run_dir,
        source_file=args.source_file.resolve(),
        output_dir=individual_dir,
        chunk_size=1,
    )
    primary_packages = [run_dir / value for value in primary_index["isolated_packages"]]
    individual_packages = [run_dir / value for value in individual_index["isolated_packages"]]
    individual_by_event = {
        _read(path)["targets"][0]["event_id"]: path for path in individual_packages
    }
    targets = {event_id: _read(path)["targets"][0] for event_id, path in individual_by_event.items()}
    attempts: dict[str, list[tuple[str, dict[str, Any]]]] = {event_id: [] for event_id in targets}

    for package_path in primary_packages:
        for tier in ("primary_marker", "primary_badge"):
            for event_id, row in _rows(
                _run_worker(run_dir=run_dir, cascade_dir=cascade_dir, package_path=package_path, tier=tier)
            ).items():
                attempts[event_id].append((tier, row))

    routing = []
    for event_id in sorted(targets, key=lambda value: targets[value]["event_frame_index"]):
        card = targets[event_id]["card"]
        route = route_after_primary(attempts[event_id], card)
        if route == "luna_tiebreak":
            row = _rows(_run_worker(
                run_dir=run_dir,
                cascade_dir=cascade_dir,
                package_path=individual_by_event[event_id],
                tier="luna_tiebreak",
            ))[event_id]
            attempts[event_id].append(("luna_tiebreak", row))
            route = route_after_tiebreak(attempts[event_id], card)
        if route == "terra":
            row = _rows(_run_worker(
                run_dir=run_dir,
                cascade_dir=cascade_dir,
                package_path=individual_by_event[event_id],
                tier="terra",
            ))[event_id]
            attempts[event_id].append(("terra", row))
            route = route_after_terra(attempts[event_id], card)
        if route == "sol":
            row = _rows(_run_worker(
                run_dir=run_dir,
                cascade_dir=cascade_dir,
                package_path=individual_by_event[event_id],
                tier="sol",
            ))[event_id]
            attempts[event_id].append(("sol", row))
        selected = select_medoid(attempts[event_id], card)
        if selected is None:
            raise ValueError(f"no legal localization decision for {event_id}")
        routing.append({
            "event_id": event_id,
            "attempts": [{"tier": tier, "cell": row["cell"], "confidence": row["confidence"]} for tier, row in attempts[event_id]],
            "selected_tier": selected[0],
            "selected_cell": selected[1]["cell"],
        })

    ordered_ids = [row["event_id"] for row in routing]
    selected_rows = [select_medoid(attempts[event_id], targets[event_id]["card"])[1] for event_id in ordered_ids]
    aggregate_package = {
        "run_id": primary_index["run_id"],
        "target_range": [
            min(targets[event_id]["event_frame_index"] for event_id in ordered_ids),
            max(targets[event_id]["event_frame_index"] for event_id in ordered_ids) + 1,
        ],
        "targets": [targets[event_id] for event_id in ordered_ids],
    }
    prediction = {
        "run_id": primary_index["run_id"],
        "stage": "own_localization_chunk",
        "target_range": aggregate_package["target_range"],
        "annotation_session_id": "label-independent-own-localization-v1",
        "model": "fixed-cascade",
        "reasoning_effort": "mixed",
        "decisions": selected_rows,
    }
    validate_own_localization_decisions(prediction, aggregate_package)
    atomic_write_json(cascade_dir / "aggregate_package.json", aggregate_package)
    atomic_write_json(cascade_dir / "routing.json", {"events": routing})
    atomic_write_json(cascade_dir / "sealed_prediction.json", prediction)
    atomic_write_json(cascade_dir / "SEALED.json", {
        "prediction_sha256": _sha256(cascade_dir / "sealed_prediction.json"),
        "policy_sha256": _sha256(policy_path),
        "event_count": len(selected_rows),
    })
    print(json.dumps({"sealed_prediction": str(cascade_dir / "sealed_prediction.json"), "events": len(selected_rows)}))


if __name__ == "__main__":
    main()
