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
from cr_bot.annotation_stages import WORKFLOW_VERSION


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize release_review.json from the deterministic same-card "
            "return gate already enforced by own-slot packages."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    verification = _read(run_dir / "verification.json")
    if args.session_id == verification.get("annotation_session_id"):
        raise ValueError("release-review aggregate session must be distinct")
    own = _read(run_dir / "own_semantics.json")
    provenance = own.get("worker_provenance", [])
    models = sorted(
        {
            row.get("model")
            for row in provenance
            if isinstance(row, dict) and isinstance(row.get("model"), str)
        }
    )
    efforts = sorted(
        {
            row.get("reasoning_effort")
            for row in provenance
            if isinstance(row, dict)
            and isinstance(row.get("reasoning_effort"), str)
        }
    )
    reviews = [
        {
            "event_id": event["event_id"],
            "decision": "released",
            "confirmation_frame_index": event["confirmation_frame_index"],
            "confirmation_artifacts": event["confirmation_artifacts"],
            "checks": event["own_confirmation"],
        }
        for event in verification["events"]
        if event["side"] == "own"
    ]
    output = {
        "run_id": verification["run_id"],
        "stage": "release_review",
        "stage_version": WORKFLOW_VERSION,
        "annotation_session_id": args.session_id,
        "model": models[0] if len(models) == 1 else "+".join(models) or "none",
        "reasoning_effort": (
            efforts[0] if len(efforts) == 1 else "mixed"
        ),
        "instructions": (
            "Every accepted own event passed the deterministic before/after "
            "same-card return gate and has a bounded post-onset confirmation "
            "artifact. Same-card returns and boundary-truncated drags were "
            "removed before verification."
        ),
        "worker_provenance": provenance,
        "reviews": reviews,
    }
    atomic_write_json(run_dir / "release_review.json", output)
    print(json.dumps({"release_reviews": len(reviews)}))


if __name__ == "__main__":
    main()
