from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import (
    validate_enemy_spell_confirmation_decisions,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _deduplicate_confirmed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda value: value["event_frame_index"]):
        if (
            not groups
            or row["event_frame_index"] - groups[-1][-1]["event_frame_index"] > 5
        ):
            groups.append([row])
        else:
            groups[-1].append(row)
    selected = []
    for group in groups:
        median = round(
            statistics.median(row["event_frame_index"] for row in group)
        )
        selected.append(
            min(
                group,
                key=lambda row: (
                    abs(row["event_frame_index"] - median),
                    abs(
                        row["event_frame_index"]
                        - row["proposal_frame_index"]
                    ),
                    row["review_id"],
                ),
            )
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter broad spell proposals through exact independent confirmation."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    onset_path = run_dir / "enemy_onsets.json"
    onsets = _read(onset_path)
    if onsets.get("run_id") != manifest.get("run_id"):
        raise ValueError("enemy onsets do not match manifest")

    pattern = re.compile(r"enemy-spell-confirmation-\d{6}-\d{6}\.json")
    packages = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if pattern.fullmatch(path.name)
    )
    if not packages:
        raise ValueError("no enemy spell confirmation packages")

    decisions: dict[str, dict[str, Any]] = {}
    review_by_id: dict[str, dict[str, Any]] = {}
    provenance = []
    for package_path in packages:
        package = _read(package_path)
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing spell confirmation output {output_path}")
        output = _read(output_path)
        if output.get("run_id") != manifest.get("run_id"):
            raise ValueError(f"{output_path}: run_id mismatch")
        if output.get("stage") != "enemy_spell_confirmation_chunk":
            raise ValueError(f"{output_path}: wrong stage")
        if output.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        validate_enemy_spell_confirmation_decisions(output, package)
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
        for review in package["reviews"]:
            review_id = review["review_id"]
            if review_id in review_by_id:
                raise ValueError(f"duplicate spell confirmation review {review_id}")
            review_by_id[review_id] = review
        for decision in output["decisions"]:
            review_id = decision["review_id"]
            if review_id in decisions:
                raise ValueError(f"duplicate spell confirmation decision {review_id}")
            decisions[review_id] = decision
    if set(decisions) != set(review_by_id):
        raise ValueError("spell confirmation outputs do not cover all reviews")

    confirmed = []
    for review_id, decision in decisions.items():
        if decision["decision"] != "confirmed":
            continue
        review = review_by_id[review_id]
        confirmed.append(
            {
                **decision,
                "review_id": review_id,
                "source_onset_id": review.get("source_onset_id"),
                "source_candidate_id": review.get("source_candidate_id"),
                "proposal_frame_index": review["proposal_frame_index"],
            }
        )
    retained = _deduplicate_confirmed(confirmed)
    raw_by_id = {
        row["onset_id"]: row
        for row in onsets.get("onsets", [])
        if isinstance(row, dict)
    }
    filtered = [
        row for row in onsets.get("onsets", []) if row.get("kind") != "spell"
    ]
    for index, row in enumerate(retained):
        source = raw_by_id.get(row.get("source_onset_id"))
        onset_id = (
            source["onset_id"]
            if source is not None
            else f"enemy-spell-{row['event_frame_index']:06d}-gate{index:03d}"
        )
        filtered.append(
            {
                "onset_id": onset_id,
                "candidate_id": (
                    source.get("candidate_id")
                    if source is not None
                    else row.get("source_candidate_id")
                ),
                "event_frame_index": row["event_frame_index"],
                "kind": "spell",
                "focus_cell": None,
                "track_id": f"confirmed-spell-{row['event_frame_index']:06d}",
                "sampled_frame_indices": review_by_id[row["review_id"]][
                    "sampled_frame_indices"
                ],
                "absence_confirmed": True,
                "persistence_confirmed": True,
                "evidence": {
                    "elixir_drop": None,
                    "hand_transition": None,
                    "deployment_onset": None,
                    "first_visible_object": None,
                    "side_direction": True,
                    "impact_sequence": True,
                },
                "verification_artifacts": row["confirmation_artifacts"],
                "identity_artifacts": [],
            }
        )
    onsets["onsets"] = sorted(
        filtered, key=lambda row: row["event_frame_index"]
    )
    atomic_write_json(onset_path, onsets)
    audit = {
        "run_id": manifest["run_id"],
        "stage": "enemy_spell_confirmation",
        "annotation_session_id": "merged-enemy-spell-confirmation-workers",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": provenance,
        "decisions": [
            {
                **decision,
                "retained": any(
                    retained_row["review_id"] == review_id
                    for retained_row in retained
                ),
            }
            for review_id, decision in sorted(decisions.items())
        ],
    }
    atomic_write_json(run_dir / "enemy_spell_confirmation.json", audit)
    print(
        json.dumps(
            {
                "confirmed": len(confirmed),
                "retained": len(retained),
                "rejected_or_unresolved": len(decisions) - len(confirmed),
            }
        )
    )


if __name__ == "__main__":
    main()
