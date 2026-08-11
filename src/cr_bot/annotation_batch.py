from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.eval.action_eval import CARD_ALIASES


FORBIDDEN_POLICY_KEY_PARTS = ("ground_truth", "evaluation", "reference_label")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_inside(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact paths must be non-empty strings")
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes batch directory: {value}") from exc
    return candidate


def _reject_reference_inputs(value: Any, *, path: str = "policy") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in FORBIDDEN_POLICY_KEY_PARTS):
                raise ValueError(f"{path} contains forbidden reference key {key!r}")
            _reject_reference_inputs(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_reference_inputs(nested, path=f"{path}[{index}]")


def validate_batch_policy(policy: dict[str, Any]) -> list[dict[str, Any]]:
    _reject_reference_inputs(policy)
    if policy.get("version") != 1:
        raise ValueError("batch policy version must be 1")
    if not isinstance(policy.get("batch_id"), str) or not policy["batch_id"]:
        raise ValueError("batch policy requires batch_id")
    datasets = policy.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("batch policy requires datasets")
    ids = [row.get("id") for row in datasets if isinstance(row, dict)]
    if len(ids) != len(datasets) or any(not isinstance(value, str) for value in ids):
        raise ValueError("every dataset requires a string id")
    if len(set(ids)) != len(ids):
        raise ValueError("dataset ids must be unique")
    for row in datasets:
        for key in ("run_dir", "prediction", "semantic_scopes"):
            if key not in row:
                raise ValueError(f"dataset {row['id']} is missing {key}")
        scopes = row["semantic_scopes"]
        if not isinstance(scopes, list) or not scopes:
            raise ValueError(f"dataset {row['id']} requires semantic scopes")
        seen_sides: set[str] = set()
        for scope in scopes:
            if not isinstance(scope, dict) or set(scope) != {"side", "range"}:
                raise ValueError(f"dataset {row['id']} has invalid semantic scope")
            side = scope["side"]
            frame_range = scope["range"]
            if side not in {"own", "enemy"} or side in seen_sides:
                raise ValueError(f"dataset {row['id']} has invalid/duplicate side")
            if (
                not isinstance(frame_range, list)
                or len(frame_range) != 2
                or any(not isinstance(value, int) for value in frame_range)
                or frame_range[0] < 0
                or frame_range[1] <= frame_range[0]
            ):
                raise ValueError(f"dataset {row['id']} has invalid frame range")
            seen_sides.add(side)
    return datasets


def seal_batch(*, batch_dir: Path, policy_path: Path, seal_path: Path) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    if seal_path.exists():
        raise ValueError("batch is already sealed")
    policy = read_object(policy_path)
    datasets = validate_batch_policy(policy)
    sealed_datasets = []
    for row in datasets:
        run_dir = resolve_inside(batch_dir, row["run_dir"])
        manifest_path = run_dir / "manifest.json"
        state_path = run_dir / "pipeline_state.json"
        prediction_path = resolve_inside(run_dir, row["prediction"])
        manifest = read_object(manifest_path)
        state = read_object(state_path)
        prediction = read_object(prediction_path)
        if state.get("status") != "semantic_complete":
            raise ValueError(f"dataset {row['id']} is not semantic_complete")
        prediction_sha256 = sha256_file(prediction_path)
        if state.get("verification_sha256") != prediction_sha256:
            raise ValueError(f"dataset {row['id']} verification hash mismatch")
        run_id = manifest.get("run_id")
        if state.get("run_id") != run_id or prediction.get("run_id") != run_id:
            raise ValueError(f"dataset {row['id']} run_id mismatch")
        segment = manifest.get("segment", {})
        segment_start = segment.get("start_frame")
        segment_end = segment.get("end_frame_exclusive")
        for scope in row["semantic_scopes"]:
            if scope["range"][0] < segment_start or scope["range"][1] > segment_end:
                raise ValueError(f"dataset {row['id']} scope escapes prepared segment")
        sealed_row: dict[str, Any] = {
            "id": row["id"],
            "run_id": run_id,
            "manifest_sha256": sha256_file(manifest_path),
            "pipeline_state_sha256": sha256_file(state_path),
            "prediction_sha256": prediction_sha256,
        }
        location = row.get("location")
        if location is not None:
            if not isinstance(location, dict) or set(location) != {"cascade_dir"}:
                raise ValueError(f"dataset {row['id']} has invalid location policy")
            cascade_dir = resolve_inside(run_dir, location["cascade_dir"])
            location_seal_path = cascade_dir / "SEALED.json"
            location_prediction_path = cascade_dir / "sealed_prediction.json"
            location_policy_path = cascade_dir / "frozen_policy.json"
            location_package_path = cascade_dir / "aggregate_package.json"
            location_seal = read_object(location_seal_path)
            location_prediction = read_object(location_prediction_path)
            location_package = read_object(location_package_path)
            if location_seal.get("prediction_sha256") != sha256_file(
                location_prediction_path
            ):
                raise ValueError(f"dataset {row['id']} location prediction hash mismatch")
            if location_seal.get("policy_sha256") != sha256_file(location_policy_path):
                raise ValueError(f"dataset {row['id']} location policy hash mismatch")
            if (
                location_prediction.get("run_id") != run_id
                or location_package.get("run_id") != run_id
            ):
                raise ValueError(f"dataset {row['id']} location run_id mismatch")
            sealed_row["location"] = {
                "seal_sha256": sha256_file(location_seal_path),
                "prediction_sha256": sha256_file(location_prediction_path),
                "policy_sha256": sha256_file(location_policy_path),
                "package_sha256": sha256_file(location_package_path),
            }
        sealed_datasets.append(sealed_row)
    seal = {
        "version": 1,
        "batch_id": policy["batch_id"],
        "policy_sha256": sha256_file(policy_path),
        "datasets": sealed_datasets,
    }
    atomic_write_json(seal_path, seal)
    return seal


def canonical_card(card: str) -> str:
    normalized = card.lower().replace("_", "-")
    if normalized.startswith("evo-"):
        normalized = normalized[4:]
    if normalized == "the-log":
        normalized = "log"
    return CARD_ALIASES.get(normalized, normalized)


def score_locations(
    *,
    truth_events: list[dict[str, Any]],
    package: dict[str, Any],
    prediction: dict[str, Any],
    frame_tolerance: int,
    cell_tolerance: int,
) -> dict[str, Any]:
    targets = {row["event_id"]: row for row in package["targets"]}
    decisions = {row["event_id"]: row for row in prediction["decisions"]}
    if len(targets) != len(package["targets"]) or set(decisions) != set(targets):
        raise ValueError("location prediction must cover every target exactly once")
    expected = [
        row
        for row in truth_events
        if row.get("side") == "own"
        and isinstance(row.get("cell"), list)
        and len(row["cell"]) == 2
    ]
    proposals = [
        {
            "event_id": event_id,
            "card": canonical_card(target["card"]),
            "frame": int(target["event_frame_index"]),
            "cell": decisions[event_id]["cell"],
        }
        for event_id, target in targets.items()
    ]
    unmatched = set(range(len(proposals)))
    rows = []
    for truth in sorted(expected, key=lambda row: int(row["frame_index"])):
        candidates = []
        for index in unmatched:
            proposal = proposals[index]
            if proposal["card"] != canonical_card(truth["card"]):
                continue
            delta = abs(proposal["frame"] - int(truth["frame_index"]))
            if delta <= frame_tolerance:
                candidates.append((delta, index))
        if not candidates:
            rows.append({"ground_truth": truth, "prediction": None, "correct": False})
            continue
        frame_error, index = min(candidates)
        unmatched.remove(index)
        proposal = proposals[index]
        errors = [
            abs(int(proposal["cell"][axis]) - int(truth["cell"][axis]))
            for axis in (0, 1)
        ]
        rows.append(
            {
                "ground_truth": truth,
                "prediction": proposal,
                "frame_error": frame_error,
                "coordinate_errors": errors,
                "correct": all(value <= cell_tolerance for value in errors),
            }
        )
    correct = sum(row["correct"] for row in rows)
    return {
        "frame_tolerance": frame_tolerance,
        "cell_tolerance_per_coordinate": cell_tolerance,
        "expected": len(expected),
        "predicted": len(proposals),
        "correct": correct,
        "incorrect": len(expected) - correct,
        "false_positive_locations": len(unmatched),
        "accuracy": correct / len(expected) if expected else 1.0,
        "rows": rows,
        "false_positive_events": [proposals[index] for index in sorted(unmatched)],
    }
