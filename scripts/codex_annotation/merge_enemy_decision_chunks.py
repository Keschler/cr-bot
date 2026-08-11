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
from cr_bot.annotation_pipeline import validate_enemy_identity_decisions


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge side/identity chunk outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    package_pattern = re.compile(r"identity-\d{6}-\d{6}\.json")
    packages = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if package_pattern.fullmatch(path.name)
    )
    if not packages:
        raise ValueError("no enemy side work packages")
    documents = []
    for package_path in packages:
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing worker output {output_path}")
        package = _read(package_path)
        document = _read(output_path)
        if document.get("stage") != "enemy_identities_chunk":
            raise ValueError(f"{output_path}: wrong stage")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        validate_enemy_identity_decisions(document, package)
        documents.append(document)
    decisions = [
        row for document in documents for row in document.get("decisions", [])
    ]
    if any(document.get("run_id") != manifest["run_id"] for document in documents):
        raise ValueError("identity chunk run_id mismatch")
    if len({row["onset_id"] for row in decisions}) != len(decisions):
        raise ValueError("duplicate onset decisions")
    onsets = _read(run_dir / "enemy_onsets.json")["onsets"]
    if {row["onset_id"] for row in decisions} != {
        row["onset_id"] for row in onsets
    }:
        raise ValueError("identity chunks must cover every onset")
    onsets_by_id = {row["onset_id"]: row for row in onsets}
    overlap_pattern = re.compile(
        r"identity-overlap-\d{6}-\d{6}\.json"
    )
    overlap_packages = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if overlap_pattern.fullmatch(path.name)
    )
    overlap_by_id: dict[str, dict[str, Any]] = {}
    for package_path in overlap_packages:
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing overlap worker output {output_path}")
        package = _read(package_path)
        document = _read(output_path)
        if document.get("stage") != "enemy_overlap_adjudication_chunk":
            raise ValueError(f"{output_path}: wrong overlap stage")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: overlap target_range mismatch")
        expected_ids = {
            row["onset_id"] for row in package.get("candidates", [])
        }
        rows = document.get("decisions")
        if not isinstance(rows, list):
            raise ValueError(f"{output_path}: overlap decisions must be a list")
        actual_ids = {
            row.get("onset_id") for row in rows if isinstance(row, dict)
        }
        if len(rows) != len(actual_ids) or actual_ids != expected_ids:
            raise ValueError(
                f"{output_path}: overlap decisions must cover candidates exactly"
            )
        for row in rows:
            if not isinstance(row.get("overlap_event_exists"), bool):
                raise ValueError(
                    f"{row.get('onset_id')}: invalid overlap verdict"
                )
            if not isinstance(row.get("evidence"), dict):
                raise ValueError(
                    f"{row.get('onset_id')}: overlap evidence is required"
                )
            if row["overlap_event_exists"]:
                evidence = row["evidence"]
                if (
                    row.get("side") != "enemy"
                    or any(
                        evidence.get(key) is not True
                        for key in (
                            "secondary_absent_before",
                            "secondary_appears_at_marker",
                            "secondary_persists_after",
                            "direct_enemy_side",
                        )
                    )
                ):
                    raise ValueError(
                        f"{row['onset_id']}: accepted overlap lacks direct evidence"
                    )
            overlap_by_id[row["onset_id"]] = row
    decisions_by_id = {row["onset_id"]: row for row in decisions}
    if set(overlap_by_id) != {
        onset_id
        for onset_id, row in decisions_by_id.items()
        if (
            row.get("event_exists") is False
            and onsets_by_id[onset_id]["kind"] == "unit_or_building"
        )
    }:
        raise ValueError(
            "overlap adjudication must cover every rejected unit candidate"
        )
    for onset_id, overlap in overlap_by_id.items():
        if not overlap["overlap_event_exists"]:
            continue
        row = decisions_by_id[onset_id]
        row["event_exists"] = True
        row["side"] = "enemy"
        row["existence_evidence"] = {
            "absent_before": overlap["evidence"]["secondary_absent_before"],
            "independent_after": overlap["evidence"][
                "secondary_appears_at_marker"
            ],
            "persistent_after": overlap["evidence"][
                "secondary_persists_after"
            ],
        }
        row["side_evidence"] = {
            "origin": "direct focused adjudication",
            "team_indicator": "red",
            "direction": None,
            "direct": True,
        }
        row["reason"] = (
            "independent overlap adjudication: "
            f"{overlap.get('reason', '')}"
        )
    unresolved = 0
    for row in decisions:
        if not isinstance(row.get("event_exists"), bool):
            raise ValueError(
                f"{row.get('onset_id')}: event_exists must be boolean"
            )
        existence = row.get("existence_evidence")
        if not isinstance(existence, dict):
            raise ValueError(
                f"{row.get('onset_id')}: existence evidence is required"
            )
        side = row.get("side")
        if side not in {"own", "enemy", "unresolved"}:
            raise ValueError(f"{row.get('onset_id')}: invalid side {side!r}")
        onset = onsets_by_id[row["onset_id"]]
        if row.get("card") is not None:
            raise ValueError(f"{row['onset_id']}: side stage must not identify a card")
        side_evidence = row.get("side_evidence")
        if not isinstance(side_evidence, dict):
            raise ValueError(f"{row['onset_id']}: side evidence is required")
        if (
            row["event_exists"]
            and side in {"own", "enemy"}
            and side_evidence.get("direct") is not True
        ):
            raise ValueError(f"{row['onset_id']}: retained side is not direct")
        if (
            row.get("identity_frame_index") is not None
            or row.get("identity_artifacts") != []
        ):
            raise ValueError(
                f"{row['onset_id']}: side stage created identity evidence"
            )
        unresolved += side == "unresolved"
    output = {
        "run_id": manifest["run_id"],
        "stage": "enemy_identities",
        "annotation_session_id": "merged-identity-chunks",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": [
            {
                key: document.get(key)
                for key in (
                    "target_range",
                    "annotation_session_id",
                    "model",
                    "reasoning_effort",
                )
            }
            for document in documents
        ],
        "decisions": sorted(
            decisions,
            key=lambda row: onsets_by_id[row["onset_id"]][
                "event_frame_index"
            ],
        ),
    }
    atomic_write_json(run_dir / "enemy_identities.json", output)
    print(
        json.dumps(
            {
                "decisions": len(decisions),
                "enemy": sum(row["side"] == "enemy" for row in decisions),
                "own": sum(row["side"] == "own" for row in decisions),
                "events": sum(row["event_exists"] for row in decisions),
                "rejected": sum(
                    not row["event_exists"] for row in decisions
                ),
                "unresolved": unresolved,
            }
        )
    )


if __name__ == "__main__":
    main()
