from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.eval.action_eval import CARD_ALIASES
from cr_bot.own_localization import validate_own_localization_decisions
from cr_bot.own_localization_cascade import select_v2_consensus, select_v4_consensus


FORBIDDEN_POLICY_KEY_PARTS = ("ground_truth", "evaluation", "reference_label")
LOCATION_SEAL_ARTIFACTS = {
    "prediction_sha256": "sealed_prediction.json",
    "policy_sha256": "frozen_policy.json",
    "package_sha256": "aggregate_package.json",
    "routing_sha256": "routing.json",
    "cost_sha256": "cost.json",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCATION_IMPLEMENTATION_PATHS = {
    "runner": REPO_ROOT / "scripts/codex_annotation/run_label_independent_own_localization_v3.py",
    "renderer_v3": REPO_ROOT / "scripts/codex_annotation/prepare_own_localization_packages_v3.py",
    "grid_renderer": REPO_ROOT / "scripts/codex_annotation/prepare_own_localization_packages.py",
    "validator": REPO_ROOT / "src/cr_bot/own_localization.py",
    "cascade": REPO_ROOT / "src/cr_bot/own_localization_cascade.py",
    "worker_harness": REPO_ROOT / "scripts/codex_annotation/run_model_worker.py",
}
LOCATION_PROMPT_PATHS = {
    "luna_marker": REPO_ROOT / "scripts/codex_annotation/prompts/own_localization_v3_marker_batch.txt",
    "luna_temporal": REPO_ROOT / "scripts/codex_annotation/prompts/own_localization_v3_temporal_batch.txt",
    "terra_residual": REPO_ROOT / "scripts/codex_annotation/prompts/own_localization_v3_coordinate_batch.txt",
    "luna_specialized": REPO_ROOT / "scripts/codex_annotation/prompts/own_localization_v3_specialized_batch.txt",
    "terra_verify": REPO_ROOT / "scripts/codex_annotation/prompts/own_localization_v3_verify_batch.txt",
    "sol_specialized": REPO_ROOT / "scripts/codex_annotation/prompts/own_localization_v3_sol_batch.txt",
}


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
        location = row.get("location")
        if location is not None:
            required_location = {"cascade_dir", "policy_sha256", "prepared_sha256"}
            if not isinstance(location, dict) or set(location) != required_location:
                raise ValueError(f"dataset {row['id']} has invalid location policy")
            if not isinstance(location["cascade_dir"], str) or not location["cascade_dir"]:
                raise ValueError(f"dataset {row['id']} has invalid location directory")
            for key in ("policy_sha256", "prepared_sha256"):
                value = location[key]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"dataset {row['id']} has invalid {key}")
    return datasets


def verify_prepared_location(
    *,
    run_dir: Path,
    cascade_dir: Path,
    source_prediction_sha256: str,
    expected_policy_sha256: str,
    expected_prepared_sha256: str,
) -> dict[str, str]:
    """Verify the localization policy, manifest, packages, and evidence freeze."""

    policy_path = cascade_dir / "frozen_policy.json"
    prepared_path = cascade_dir / "PREPARED.json"
    if sha256_file(policy_path) != expected_policy_sha256:
        raise ValueError("location policy differs from frozen batch policy")
    if sha256_file(prepared_path) != expected_prepared_sha256:
        raise ValueError("location prepared manifest differs from frozen batch policy")
    prepared = read_object(prepared_path)
    policy = read_object(policy_path)
    if prepared.get("policy_sha256") != expected_policy_sha256:
        raise ValueError("location prepared policy hash mismatch")
    if prepared.get("source_sha256") != source_prediction_sha256:
        raise ValueError("location prepared source hash mismatch")
    for field, paths in (
        ("implementation_sha256s", LOCATION_IMPLEMENTATION_PATHS),
        ("prompt_sha256s", LOCATION_PROMPT_PATHS),
    ):
        frozen_hashes = policy.get(field)
        if frozen_hashes is None:
            continue
        if not isinstance(frozen_hashes, dict) or set(frozen_hashes) != set(paths):
            raise ValueError(f"location policy has invalid {field}")
        for name, path in paths.items():
            if frozen_hashes[name] != sha256_file(path):
                raise ValueError(f"location frozen {field} changed: {name}")
    package_index_path = cascade_dir / "packages/package_index.json"
    if prepared.get("package_index_sha256") != sha256_file(package_index_path):
        raise ValueError("location prepared package-index hash mismatch")
    for collection in ("package_sha256s", "evidence_sha256s"):
        expected_hashes = prepared.get(collection)
        if not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError(f"location prepared {collection} must not be empty")
        for relative_path, expected_sha256 in expected_hashes.items():
            path = resolve_inside(run_dir, relative_path)
            if expected_sha256 != sha256_file(path):
                raise ValueError(f"location prepared artifact changed: {relative_path}")
    return {
        "policy_sha256": sha256_file(policy_path),
        "prepared_sha256": sha256_file(prepared_path),
        "package_index_sha256": sha256_file(package_index_path),
    }


def verify_location_cascade(
    *,
    run_dir: Path,
    cascade_dir: Path,
    run_id: str,
    source_prediction_sha256: str,
    expected_policy_sha256: str,
    expected_prepared_sha256: str,
) -> dict[str, str]:
    """Verify a complete blind localization seal and return batch-seal hashes."""

    prepared_hashes = verify_prepared_location(
        run_dir=run_dir,
        cascade_dir=cascade_dir,
        source_prediction_sha256=source_prediction_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_prepared_sha256=expected_prepared_sha256,
    )
    prepared = read_object(cascade_dir / "PREPARED.json")

    seal_path = cascade_dir / "SEALED.json"
    seal = read_object(seal_path)
    documents = {
        key: cascade_dir / filename
        for key, filename in LOCATION_SEAL_ARTIFACTS.items()
    }
    for key, path in documents.items():
        if seal.get(key) != sha256_file(path):
            label = key.removesuffix("_sha256").replace("_", " ")
            raise ValueError(f"location {label} hash mismatch")
    if seal.get("source_sha256") != source_prediction_sha256:
        raise ValueError("location source hash does not match semantic prediction")

    policy = read_object(documents["policy_sha256"])
    package = read_object(documents["package_sha256"])
    prediction = read_object(documents["prediction_sha256"])
    routing = read_object(documents["routing_sha256"])
    read_object(documents["cost_sha256"])
    if policy.get("source_sha256") != source_prediction_sha256:
        raise ValueError("location policy source hash mismatch")
    if prediction.get("run_id") != run_id or package.get("run_id") != run_id:
        raise ValueError("location run_id mismatch")

    package_index = read_object(cascade_dir / "packages/package_index.json")
    frozen_targets: list[dict[str, Any]] = []
    for relative_path in package_index.get("packages", []):
        frozen_package = read_object(resolve_inside(run_dir, relative_path))
        targets = frozen_package.get("targets")
        if not isinstance(targets, list):
            raise ValueError("location frozen package targets must be a list")
        frozen_targets.extend(targets)
    if (
        len(frozen_targets) != prepared.get("target_count")
        or package.get("targets") != frozen_targets
    ):
        raise ValueError("location aggregate targets differ from prepared packages")

    decisions = validate_own_localization_decisions(prediction, package)
    if (
        seal.get("event_count") != len(decisions)
        or prepared.get("target_count") != len(decisions)
    ):
        raise ValueError("location seal event count mismatch")
    routing_rows = routing.get("events")
    if not isinstance(routing_rows, list):
        raise ValueError("location routing events must be a list")
    route_by_id = {
        row.get("event_id"): row for row in routing_rows if isinstance(row, dict)
    }
    decision_by_id = {row["event_id"]: row for row in decisions}
    target_by_id = {row["event_id"]: row for row in frozen_targets}
    if len(route_by_id) != len(routing_rows) or set(route_by_id) != set(decision_by_id):
        raise ValueError("location routing must cover every decision exactly once")
    for event_id, decision in decision_by_id.items():
        route = route_by_id[event_id]
        if route.get("selected_cell") != decision["cell"]:
            raise ValueError(f"{event_id}: routed and sealed cells differ")
        selected_role = route.get("selected_role")
        attempts = route.get("attempts")
        if not isinstance(selected_role, str) or not isinstance(attempts, list):
            raise ValueError(f"{event_id}: invalid selected routing role")
        if any(
            not isinstance(attempt, dict) or not isinstance(attempt.get("role"), str)
            for attempt in attempts
        ):
            raise ValueError(f"{event_id}: invalid routing attempt")
        attempt_pairs = [(attempt["role"], attempt) for attempt in attempts]
        consensus_policy = policy.get("consensus_policy")
        if consensus_policy == "exact-inferred":
            selection = select_v4_consensus(attempt_pairs, target_by_id[event_id]["card"])
        elif consensus_policy in {None, "strict-direct"}:
            selection = select_v2_consensus(attempt_pairs, target_by_id[event_id]["card"])
        else:
            raise ValueError("unsupported sealed localization consensus policy")
        if selection is None:
            raise ValueError(f"{event_id}: routing has no qualified consensus")
        (expected_role, expected_decision), expected_cluster = selection
        if (
            selected_role != expected_role
            or decision["cell"] != expected_decision.get("cell")
            or route.get("agreement_cluster_indices") != expected_cluster
            or route.get("agreement_cluster_roles")
            != [attempts[index]["role"] for index in expected_cluster]
        ):
            raise ValueError(f"{event_id}: selected routing consensus is inconsistent")

    return {
        "seal_sha256": sha256_file(seal_path),
        **prepared_hashes,
        **{key: sha256_file(path) for key, path in documents.items()},
        "source_prediction_sha256": source_prediction_sha256,
    }


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
            cascade_dir = resolve_inside(run_dir, location["cascade_dir"])
            try:
                sealed_row["location"] = verify_location_cascade(
                    run_dir=run_dir,
                    cascade_dir=cascade_dir,
                    run_id=run_id,
                    source_prediction_sha256=prediction_sha256,
                    expected_policy_sha256=location["policy_sha256"],
                    expected_prepared_sha256=location["prepared_sha256"],
                )
            except ValueError as exc:
                raise ValueError(f"dataset {row['id']} {exc}") from exc
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


def optimal_frame_match_indices(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    *,
    tolerance: int,
    group_fields: tuple[str, ...],
    frame_field: str = "frame",
) -> list[tuple[int, int, int]]:
    """Maximize frame-tolerant matches, then minimize their total error.

    Matching is solved independently for each exact group (normally side and
    canonical card). Sorted one-dimensional absolute-distance matching always
    has an optimal non-crossing solution, which permits a deterministic dynamic
    program instead of order-dependent greedy assignment.
    """

    if tolerance < 0:
        raise ValueError("frame tolerance must be non-negative")
    expected_groups: dict[tuple[Any, ...], list[int]] = {}
    predicted_groups: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(expected):
        expected_groups.setdefault(tuple(row[field] for field in group_fields), []).append(index)
    for index, row in enumerate(predicted):
        predicted_groups.setdefault(tuple(row[field] for field in group_fields), []).append(index)

    all_pairs: list[tuple[int, int, int]] = []
    for group in sorted(set(expected_groups) | set(predicted_groups), key=repr):
        expected_indices = sorted(
            expected_groups.get(group, []),
            key=lambda index: (int(expected[index][frame_field]), index),
        )
        predicted_indices = sorted(
            predicted_groups.get(group, []),
            key=lambda index: (int(predicted[index][frame_field]), index),
        )

        @lru_cache(maxsize=None)
        def solve(
            expected_offset: int, predicted_offset: int
        ) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
            if expected_offset == len(expected_indices) or predicted_offset == len(
                predicted_indices
            ):
                return 0, 0, ()
            choices = [
                solve(expected_offset + 1, predicted_offset),
                solve(expected_offset, predicted_offset + 1),
            ]
            expected_index = expected_indices[expected_offset]
            predicted_index = predicted_indices[predicted_offset]
            delta = abs(
                int(expected[expected_index][frame_field])
                - int(predicted[predicted_index][frame_field])
            )
            if delta <= tolerance:
                count, error, pairs = solve(expected_offset + 1, predicted_offset + 1)
                choices.append(
                    (
                        count + 1,
                        error + delta,
                        ((expected_index, predicted_index, delta), *pairs),
                    )
                )
            return min(choices, key=lambda value: (-value[0], value[1], value[2]))

        all_pairs.extend(solve(0, 0)[2])
    return sorted(
        all_pairs,
        key=lambda pair: (int(expected[pair[0]][frame_field]), pair[0], pair[1]),
    )


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
    expected_match_rows = [
        {"card": canonical_card(row["card"]), "frame": int(row["frame_index"])}
        for row in expected
    ]
    matches = optimal_frame_match_indices(
        expected_match_rows,
        proposals,
        tolerance=frame_tolerance,
        group_fields=("card",),
    )
    matched_by_expected = {expected_index: (proposal_index, delta) for expected_index, proposal_index, delta in matches}
    unmatched = set(range(len(proposals))) - {
        proposal_index for _, proposal_index, _ in matches
    }
    rows = []
    for truth_index in sorted(
        range(len(expected)), key=lambda index: int(expected[index]["frame_index"])
    ):
        truth = expected[truth_index]
        matched = matched_by_expected.get(truth_index)
        if matched is None:
            rows.append({"ground_truth": truth, "prediction": None, "correct": False})
            continue
        index, frame_error = matched
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
