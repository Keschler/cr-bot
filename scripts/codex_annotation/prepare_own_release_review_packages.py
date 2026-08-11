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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare independent own-release packages containing only the "
            "canonical post-release sheets."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=400)
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")

    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    verification = _read(run_dir / "verification.json")
    if verification.get("run_id") != manifest.get("run_id"):
        raise ValueError("verification run_id does not match manifest")
    events = verification.get("events")
    if not isinstance(events, list):
        raise ValueError("verification events must be a list")
    reviews = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("verification event must be an object")
        if event.get("side") != "own":
            continue
        artifacts = event.get("confirmation_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise ValueError(
                f"{event.get('candidate_id')}: exactly one release artifact required"
            )
        artifact_path = run_dir / artifacts[0]
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing release artifact {artifact_path}")
        reviews.append(
            {
                "event_id": event["event_id"],
                "candidate_id": event["candidate_id"],
                "event_frame_index": int(event["event_frame_index"]),
                "confirmation_frame_index": int(
                    event["confirmation_frame_index"]
                ),
                "confirmation_artifacts": artifacts,
            }
        )

    segment = manifest["segment"]
    output_dir = run_dir / "work_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for start in range(
        int(segment["start_frame"]),
        int(segment["end_frame_exclusive"]),
        args.chunk_frames,
    ):
        end = min(int(segment["end_frame_exclusive"]), start + args.chunk_frames)
        selected = [
            row for row in reviews if start <= row["event_frame_index"] < end
        ]
        path = output_dir / f"own-release-{start:06d}-{end:06d}.json"
        atomic_write_json(
            path,
            {
                "run_id": manifest["run_id"],
                "stage": "own_release_review_package",
                "target_range": [start, end],
                "reviews": selected,
            },
        )
        summaries.append({"range": [start, end], "reviews": len(selected)})
    print(json.dumps({"packages": summaries, "total": len(reviews)}))


if __name__ == "__main__":
    main()
