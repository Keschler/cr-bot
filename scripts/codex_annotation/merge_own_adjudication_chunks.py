from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge final own adjudication chunks.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = []
    proposal_ids = set()
    rejected = []
    package_paths = sorted(
        (run_dir / "work_packages").glob("own-adjudicate-*.json")
    )
    if not package_paths:
        raise ValueError("no own adjudication packages")
    proposals_by_id = {}
    for package_path in package_paths:
        package = json.loads(package_path.read_text(encoding="utf-8"))
        for proposal in package["proposals"]:
            proposal_id = proposal["proposal_id"]
            if proposal_id in proposals_by_id:
                raise ValueError(
                    f"duplicate proposal_id across packages: {proposal_id}"
                )
            proposals_by_id[proposal_id] = proposal
    expected_ids = set(proposals_by_id)
    provenance = []
    for package_path in package_paths:
        path = run_dir / "worker_outputs" / package_path.name
        if not path.is_file():
            raise ValueError(f"missing worker output {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["run_id"] != manifest["run_id"]:
            raise ValueError("own adjudication run_id mismatch")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        if document.get("stage") != "own_adjudication_chunk":
            raise ValueError(f"{path}: wrong stage")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{path}: target_range does not match package")
        provenance.append(
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
        for row in document["events"]:
            proposal_id = row["proposal_id"]
            if proposal_id in proposal_ids:
                raise ValueError("duplicate proposal decision")
            proposal_ids.add(proposal_id)
            events.append(
                {key: value for key, value in row.items() if key != "proposal_id"}
            )
        for row in document["rejected_proposals"]:
            proposal_id = row["proposal_id"]
            if proposal_id in proposal_ids:
                raise ValueError("duplicate proposal decision")
            proposal_ids.add(proposal_id)
            proposal = proposals_by_id.get(proposal_id)
            if proposal is None:
                raise ValueError(f"unknown rejected proposal {proposal_id}")
            candidate_ids = proposal.get("candidate_ids")
            if (
                not isinstance(candidate_ids, list)
                or not candidate_ids
                or any(not isinstance(value, str) for value in candidate_ids)
            ):
                raise ValueError(
                    f"proposal {proposal_id} has no candidate_ids"
                )
            for candidate_id in candidate_ids:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "proposal_id": proposal_id,
                        "reason": row.get("reason", "rejected by adjudication"),
                    }
                )
    if proposal_ids != expected_ids:
        missing = sorted(expected_ids - proposal_ids)
        extra = sorted(proposal_ids - expected_ids)
        raise ValueError(
            f"own adjudication coverage mismatch; missing={missing}, extra={extra}"
        )
    accepted_candidate_ids = {row["candidate_id"] for row in events}
    rejected_by_candidate = {}
    for row in rejected:
        candidate_id = row["candidate_id"]
        if candidate_id not in accepted_candidate_ids:
            rejected_by_candidate.setdefault(candidate_id, row)
    rejected = sorted(
        rejected_by_candidate.values(), key=lambda row: row["candidate_id"]
    )
    output = {
        "run_id": manifest["run_id"],
        "stage": "own_semantics",
        "annotation_session_id": "merged-own-adjudication",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": provenance,
        "events": sorted(events, key=lambda row: row["event_frame_index"]),
        "rejected_candidates": rejected,
    }
    atomic_write_json(run_dir / "own_semantics.json", output)
    print(json.dumps({"events": len(events), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
