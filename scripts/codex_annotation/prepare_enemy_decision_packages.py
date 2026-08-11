from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split discovered onsets into bounded side/identity packages."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=400)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    onsets = _read(run_dir / "enemy_onsets.json")
    manifest = _read(run_dir / "manifest.json")
    own_path = run_dir / "own_semantics.json"
    own = _read(own_path) if own_path.is_file() else {"events": [], "rejected_candidates": []}
    own_release_frames = sorted(
        {
            int(row["event_frame_index"])
            for row in own.get("events", [])
            if isinstance(row, dict)
        }
    )
    rejected_drags = []
    candidates = {
        row["candidate_id"]: row
        for row in manifest["candidate_discovery"]["own_candidates"]
    }
    for row in own.get("rejected_candidates", []):
        if not isinstance(row, dict):
            continue
        candidate = candidates.get(row.get("candidate_id"))
        if candidate is None:
            continue
        rejected_drags.append(
            {
                "candidate_id": candidate["candidate_id"],
                "approximate_frame_index": candidate["approximate_frame_index"],
                "inspection_range": [
                    candidate["inspection_start_frame"],
                    candidate["inspection_end_frame_exclusive"],
                ],
                "reason": row.get("reason", ""),
            }
        )
    output_dir = run_dir / "work_packages"
    summaries = []
    start = manifest["segment"]["start_frame"]
    stop = manifest["segment"]["end_frame_exclusive"]
    for chunk_start in range(start, stop, args.chunk_frames):
        chunk_end = min(stop, chunk_start + args.chunk_frames)
        rows = [
            {
                **row,
                # The primary pass gets global arena context only. Focused
                # crops are reserved for an independent overlap adjudicator.
                "verification_artifacts": row[
                    "verification_artifacts"
                ][:1],
            }
            for row in onsets["onsets"]
            if chunk_start <= row["event_frame_index"] < chunk_end
        ]
        path = output_dir / f"identity-{chunk_start:06d}-{chunk_end:06d}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": onsets["run_id"],
                    "fps": manifest["fps"],
                    "segment": manifest["segment"],
                    "target_range": [chunk_start, chunk_end],
                    "onsets": rows,
                    "own_release_frames": [
                        frame
                        for frame in own_release_frames
                        if chunk_start - 20 <= frame < chunk_end + 20
                    ],
                    "rejected_own_drags": [
                        row
                        for row in rejected_drags
                        if chunk_start - 20
                        <= row["approximate_frame_index"]
                        < chunk_end + 20
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append({"range": [chunk_start, chunk_end], "onsets": len(rows)})
    print(json.dumps({"packages": summaries}))


if __name__ == "__main__":
    main()
