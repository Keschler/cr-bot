from __future__ import annotations

import argparse
import hashlib
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
from cr_bot.annotation_pipeline import validate_own_semantic_decisions


PACKAGE_PATTERN = re.compile(r"own-adjudicate-\d{6}-\d{6}\.json")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_proposal_event(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the latest complete source row without semantic filtering."""
    rows = proposal.get("candidate_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{proposal.get('proposal_id')}: candidate_rows is empty")
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("event_frame_index"), int)
            or isinstance(row.get("event_frame_index"), bool)
        ):
            raise ValueError(
                f"{proposal.get('proposal_id')}: invalid candidate event row"
            )
    # Rows are primary followed by recall-sweep rows. On an exact frame tie,
    # retain the later source row while remaining fully deterministic.
    _, selected = max(
        enumerate(rows),
        key=lambda item: (item[1]["event_frame_index"], item[0]),
    )
    event = json.loads(json.dumps(selected))
    if event.get("card") != proposal.get("card"):
        raise ValueError(f"{proposal.get('proposal_id')}: selected card changed")
    if event.get("candidate_id") not in proposal.get("candidate_ids", []):
        raise ValueError(
            f"{proposal.get('proposal_id')}: selected unrelated candidate"
        )
    proposed_frames = proposal.get("proposed_frames")
    if (
        not isinstance(proposed_frames, list)
        or event["event_frame_index"] not in proposed_frames
        or event["event_frame_index"] != max(proposed_frames)
    ):
        raise ValueError(
            f"{proposal.get('proposal_id')}: latest proposed frame is inconsistent"
        )
    return event


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize every anonymous own proposal as a high-recall union; "
            "the independent release-review model is the precision gate."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    package_paths = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if PACKAGE_PATTERN.fullmatch(path.name)
    )
    if not package_paths:
        raise ValueError("no own adjudication proposal packages")

    proposal_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    evidence_by_candidate: dict[str, dict[str, Any]] = {}
    package_provenance = []
    proposal_selections = []
    for package_path in package_paths:
        package = _read(package_path)
        if package.get("run_id") != manifest.get("run_id"):
            raise ValueError(f"{package_path}: run_id mismatch")
        proposals = package.get("proposals")
        if not isinstance(proposals, list):
            raise ValueError(f"{package_path}: proposals must be a list")
        package_provenance.append(
            {
                "package": f"work_packages/{package_path.name}",
                "target_range": package.get("target_range"),
                "proposal_count": len(proposals),
                "sha256": _sha256(package_path),
            }
        )
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise ValueError(f"{package_path}: proposal must be an object")
            proposal_id = proposal.get("proposal_id")
            if not isinstance(proposal_id, str) or proposal_id in proposal_ids:
                raise ValueError(f"duplicate or invalid proposal_id {proposal_id!r}")
            proposal_ids.add(proposal_id)
            event = select_proposal_event(proposal)
            events.append(event)
            proposal_selections.append(
                {
                    "proposal_id": proposal_id,
                    "candidate_id": event["candidate_id"],
                    "event_frame_index": event["event_frame_index"],
                    "card": event["card"],
                }
            )
            evidence_rows = proposal.get("candidate_evidence")
            if not isinstance(evidence_rows, list):
                raise ValueError(f"{proposal_id}: candidate_evidence must be a list")
            for evidence in evidence_rows:
                if not isinstance(evidence, dict):
                    raise ValueError(f"{proposal_id}: invalid candidate evidence")
                candidate_id = evidence.get("candidate_id")
                if not isinstance(candidate_id, str):
                    raise ValueError(f"{proposal_id}: invalid evidence candidate")
                previous = evidence_by_candidate.setdefault(candidate_id, evidence)
                if previous != evidence:
                    raise ValueError(
                        f"{proposal_id}: conflicting evidence for {candidate_id}"
                    )

    validate_own_semantic_decisions(
        {"events": events},
        {"candidates": list(evidence_by_candidate.values())},
        require_candidate_coverage=False,
    )
    previous = _read(run_dir / "own_semantics.json")
    worker_provenance = list(previous.get("worker_provenance", []))
    for path in sorted((run_dir / "worker_outputs").glob("own-complete-*.json")):
        document = _read(path)
        worker_provenance.append(
            {
                key: document.get(key)
                for key in (
                    "target_range",
                    "annotation_session_id",
                    "model",
                    "reasoning_effort",
                )
            }
        )
    output = {
        "run_id": manifest["run_id"],
        "stage": "own_semantics",
        "annotation_session_id": "deterministic-own-proposal-union-v1",
        "model": "deterministic",
        "reasoning_effort": "none",
        "worker_provenance": worker_provenance,
        "materialization_provenance": {
            "algorithm": "latest-complete-row-per-proposal-v1",
            "packages": package_provenance,
            "proposal_selections": proposal_selections,
        },
        "events": sorted(
            events,
            key=lambda row: (
                row["event_frame_index"], row["card"], row["candidate_id"]
            ),
        ),
        "rejected_candidates": [],
    }
    atomic_write_json(run_dir / "own_semantics.json", output)
    print(json.dumps({"proposals": len(proposal_ids), "events": len(events)}))


if __name__ == "__main__":
    main()
