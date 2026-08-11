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

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import validate_own_release_review_decisions
from cr_bot.annotation_stages import WORKFLOW_VERSION


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge genuinely independent own-release worker decisions, remove "
            "canceled/unresolved proposals, and build release_review.json."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    verification_path = run_dir / "verification.json"
    verification = _read(verification_path)
    if args.session_id == verification.get("annotation_session_id"):
        raise ValueError("release review aggregate session must be fresh")
    proposed = {
        event["event_id"]: event
        for event in verification.get("events", [])
        if event.get("side") == "own"
    }
    if len(proposed) != sum(
        event.get("side") == "own" for event in verification.get("events", [])
    ):
        raise ValueError("verification contains duplicate own event IDs")

    own = _read(run_dir / "own_semantics.json")
    adjudication_sessions = {
        row.get("annotation_session_id")
        for row in own.get("worker_provenance", [])
        if isinstance(row, dict) and row.get("annotation_session_id")
    }
    packages = sorted(
        (run_dir / "work_packages").glob("own-release-??????-??????.json")
    )
    if not packages and proposed:
        raise ValueError("no independent own-release packages")

    decisions: dict[str, dict[str, Any]] = {}
    candidate_by_event: dict[str, str] = {}
    provenance = []
    release_sessions: set[str] = set()
    for package_path in packages:
        package = _read(package_path)
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing independent release output {output_path}")
        output = _read(output_path)
        if output.get("run_id") != verification.get("run_id"):
            raise ValueError(f"{output_path}: run_id mismatch")
        if output.get("stage") != "own_release_review_chunk":
            raise ValueError(f"{output_path}: wrong release-review stage")
        if output.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        validate_own_release_review_decisions(output, package)
        worker_session = output.get("annotation_session_id")
        if not isinstance(worker_session, str) or not worker_session:
            raise ValueError(f"{output_path}: missing worker session provenance")
        if worker_session in adjudication_sessions:
            raise ValueError(
                f"{output_path}: release worker reused an adjudication session"
            )
        if worker_session == verification.get("annotation_session_id"):
            raise ValueError(
                f"{output_path}: release worker reused the verification session"
            )
        if (
            worker_session != "deterministic-empty-package"
            and worker_session in release_sessions
        ):
            raise ValueError(
                f"{output_path}: release worker session was reused across packages"
            )
        release_sessions.add(worker_session)
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
            event_id = review["event_id"]
            if event_id in candidate_by_event:
                raise ValueError(f"duplicate release package event {event_id}")
            candidate_by_event[event_id] = review["candidate_id"]
        for row in output["decisions"]:
            event_id = row["event_id"]
            if event_id in decisions:
                raise ValueError(f"duplicate release decision {event_id}")
            decisions[event_id] = row

    if set(decisions) != set(proposed):
        raise ValueError(
            "independent release decisions must cover every proposed own event; "
            f"missing={sorted(set(proposed) - set(decisions))}, "
            f"extra={sorted(set(decisions) - set(proposed))}"
        )
    released_ids = {
        event_id
        for event_id, row in decisions.items()
        if row["decision"] == "released"
    }
    accepted_candidates = {
        proposed[event_id]["candidate_id"] for event_id in released_ids
    }
    filtered_events = []
    rejected = list(verification.get("rejected_candidates", []))
    reviews = []
    for event in verification["events"]:
        event_id = event["event_id"]
        if event.get("side") != "own":
            filtered_events.append(event)
            continue
        decision = decisions[event_id]
        if decision["decision"] != "released":
            candidate_id = candidate_by_event[event_id]
            if candidate_id not in accepted_candidates:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": (
                            f"independent_release_{decision['decision']}: "
                            f"{decision['reason']}"
                        ),
                    }
                )
            continue
        updated = dict(event)
        updated["confirmation_frame_index"] = decision[
            "confirmation_frame_index"
        ]
        updated["confirmation_artifacts"] = decision[
            "confirmation_artifacts"
        ]
        updated["own_confirmation"] = decision["checks"]
        filtered_events.append(updated)
        reviews.append(
            {
                "event_id": event_id,
                "decision": "released",
                "confirmation_frame_index": decision[
                    "confirmation_frame_index"
                ],
                "confirmation_artifacts": decision[
                    "confirmation_artifacts"
                ],
                "checks": decision["checks"],
            }
        )

    verification["events"] = sorted(
        filtered_events,
        key=lambda row: (row["event_frame_index"], row["side"], row["card"]),
    )
    verification["rejected_candidates"] = rejected
    atomic_write_json(verification_path, verification)

    models = sorted({row["model"] for row in provenance})
    efforts = sorted({row["reasoning_effort"] for row in provenance})
    document = {
        "run_id": verification["run_id"],
        "stage": "release_review",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": args.session_id,
        "model": models[0] if len(models) == 1 else "+".join(models),
        "reasoning_effort": efforts[0] if len(efforts) == 1 else "mixed",
        "instructions": (
            "Merged from fresh-session model workers that independently inspected "
            "only canonical post-release sheets. Canceled or unresolved proposals "
            "were removed from verification before checkpointing."
        ),
        "worker_provenance": provenance,
        "reviews": sorted(reviews, key=lambda row: row["event_id"]),
    }
    atomic_write_json(run_dir / "release_review.json", document)
    print(
        json.dumps(
            {
                "release_reviews": len(reviews),
                "removed": len(proposed) - len(reviews),
                "workers": len(provenance),
            }
        )
    )


if __name__ == "__main__":
    main()
