from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _documents(
    *,
    run_dir: Path,
    prefix: str,
    expected_stage: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pattern = re.compile(rf"{re.escape(prefix)}-\d{{6}}-\d{{6}}\.json")
    package_names = sorted(
        path.name
        for path in (run_dir / "work_packages").iterdir()
        if pattern.fullmatch(path.name)
    )
    if not package_names:
        raise ValueError(f"no {prefix} work packages")
    documents = []
    for name in package_names:
        path = run_dir / "worker_outputs" / name
        if not path.is_file():
            raise ValueError(f"missing worker output {path}")
        document = _read(path)
        if document.get("stage") != expected_stage:
            raise ValueError(f"{path}: expected stage {expected_stage!r}")
        package = _read(run_dir / "work_packages" / name)
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{path}: target_range does not match package")
        documents.append((package, document))
    return documents


def _row_candidate_ids(
    document: dict[str, Any], key: str, *, package_name: str
) -> set[str]:
    rows = document.get(key, [])
    if not isinstance(rows, list):
        raise ValueError(f"{package_name}: {key} must be a list")
    values = [
        row.get("candidate_id") if isinstance(row, dict) else None
        for row in rows
    ]
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"{package_name}: {key} has an invalid candidate_id")
    if len(values) != len(set(values)):
        raise ValueError(f"{package_name}: {key} has duplicate candidate_ids")
    return set(values)


def _validate_pending_resolutions(
    documents: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    retained_event_candidates: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Require every pending boundary drag to terminate in a later package."""
    decisions = []
    package_candidates = []
    for package, document in documents:
        label = str(package.get("target_range"))
        candidate_ids = {
            row.get("candidate_id")
            for row in package.get("candidates", [])
            if isinstance(row, dict)
        }
        event_ids = _row_candidate_ids(document, "events", package_name=label)
        rejected_ids = _row_candidate_ids(
            document, "rejected_candidates", package_name=label
        )
        pending_ids = _row_candidate_ids(
            document, "pending_at_end", package_name=label
        )
        if any(
            left & right
            for index, left in enumerate(
                (event_ids, rejected_ids, pending_ids)
            )
            for right in (event_ids, rejected_ids, pending_ids)[index + 1 :]
        ):
            raise ValueError(
                f"{label}: candidate appears in multiple decision categories"
            )
        decisions.append((event_ids, rejected_ids, pending_ids))
        package_candidates.append(candidate_ids)

    resolutions = []
    forced_rejections: set[str] = set()
    for source_index, (package, _) in enumerate(documents):
        source_pending = decisions[source_index][2]
        for candidate_id in sorted(source_pending):
            saw_following_context = False
            resolution = None
            for later_index in range(source_index + 1, len(documents)):
                if candidate_id not in package_candidates[later_index]:
                    continue
                saw_following_context = True
                event_ids, rejected_ids, _ = decisions[later_index]
                if candidate_id in event_ids:
                    resolution = (later_index, "event")
                    break
                if candidate_id in rejected_ids:
                    resolution = (later_index, "rejected")
                    break
            if resolution is None:
                detail = (
                    "remains pending in every following overlapping chunk"
                    if saw_following_context
                    else "has no following overlapping chunk"
                )
                raise ValueError(
                    f"pending candidate {candidate_id} from target_range "
                    f"{package.get('target_range')} {detail}; expand the context "
                    "halo or resolve it before merging"
                )
            later_index, decision = resolution
            later_package = documents[later_index][0]
            if (
                decision == "event"
                and candidate_id not in retained_event_candidates
            ):
                raise ValueError(
                    f"pending candidate {candidate_id} resolves as an event in "
                    f"target_range {later_package.get('target_range')}, but its "
                    "corrected frame is owned by a different chunk"
                )
            if decision == "rejected":
                forced_rejections.add(candidate_id)
            resolutions.append(
                {
                    "candidate_id": candidate_id,
                    "pending_target_range": package.get("target_range"),
                    "resolved_target_range": later_package.get("target_range"),
                    "decision": decision,
                }
            )
    return resolutions, forced_rejections


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge bounded semantic worker chunks.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--side",
        choices=["own", "enemy", "both"],
        default="both",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    run_id = manifest["run_id"]
    own_documents = (
        _documents(
            run_dir=run_dir,
            prefix="own",
            expected_stage="own_semantics_chunk",
        )
        if args.side in {"own", "both"}
        else []
    )
    enemy_documents = (
        _documents(
            run_dir=run_dir,
            prefix="enemy",
            expected_stage="enemy_onsets_chunk",
        )
        if args.side in {"enemy", "both"}
        else []
    )
    for _, document in [*own_documents, *enemy_documents]:
        if document.get("run_id") != run_id:
            raise ValueError("chunk run_id does not match manifest")
    own_events = [
        row
        for package, document in own_documents
        for row in document.get("events", [])
        if (
            package["target_range"][0]
            <= row["event_frame_index"]
            < package["target_range"][1]
        )
    ]
    accepted_candidates = {row["candidate_id"] for row in own_events}
    pending_resolutions, forced_rejections = _validate_pending_resolutions(
        own_documents,
        retained_event_candidates=accepted_candidates,
    )
    rejected_by_id = {}
    for package, document in own_documents:
        owned = {
            row["candidate_id"]
            for row in package["candidates"]
            if row["package_role"] == "owned_peak"
        }
        for row in document.get("rejected_candidates", []):
            candidate_id = row["candidate_id"]
            if (
                (candidate_id in owned or candidate_id in forced_rejections)
                and candidate_id not in accepted_candidates
            ):
                rejected_by_id[candidate_id] = row
    rejected = list(rejected_by_id.values())
    onsets = [
        row
        for _, document in enemy_documents
        for row in document.get("onsets", [])
    ]
    if len({(row["candidate_id"], row["card"], row["event_frame_index"]) for row in own_events}) != len(own_events):
        raise ValueError("duplicate own event across chunks")
    if len({row["onset_id"] for row in onsets}) != len(onsets):
        raise ValueError("duplicate enemy onset across chunks")
    own_provenance = [
        {
            key: document.get(key)
            for key in (
                "target_range",
                "annotation_session_id",
                "model",
                "reasoning_effort",
            )
        }
        for _, document in own_documents
    ]
    enemy_provenance = [
        {
            key: document.get(key)
            for key in (
                "target_range",
                "annotation_session_id",
                "model",
                "reasoning_effort",
            )
        }
        for _, document in enemy_documents
    ]
    own_output = {
        "run_id": run_id,
        "stage": "own_semantics",
        "annotation_session_id": "merged-chunk-workers",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": own_provenance,
        "events": own_events,
        "rejected_candidates": rejected,
        "pending_resolutions": pending_resolutions,
    }
    enemy_output = {
        "run_id": run_id,
        "stage": "enemy_onsets",
        "annotation_session_id": "merged-chunk-workers",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": enemy_provenance,
        "onsets": sorted(onsets, key=lambda row: row["event_frame_index"]),
    }
    if own_documents:
        atomic_write_json(run_dir / "own_semantics.json", own_output)
    if enemy_documents:
        atomic_write_json(run_dir / "enemy_onsets.json", enemy_output)
    print(json.dumps({"own_events": len(own_events), "enemy_onsets": len(onsets)}))


if __name__ == "__main__":
    main()
