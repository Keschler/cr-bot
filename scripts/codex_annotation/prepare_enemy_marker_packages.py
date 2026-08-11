from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split highlighted enemy unit-marker bursts into work packages."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=400)
    parser.add_argument("--context-halo-frames", type=int, default=20)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        (run_dir / "enemy_marker_candidates.json").read_text(encoding="utf-8")
    )
    output_dir = run_dir / "work_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    segment = manifest["segment"]
    summaries = []
    for start in range(
        segment["start_frame"],
        segment["end_frame_exclusive"],
        args.chunk_frames,
    ):
        end = min(segment["end_frame_exclusive"], start + args.chunk_frames)
        context_start = max(
            segment["start_frame"], start - args.context_halo_frames
        )
        context_end = min(
            segment["end_frame_exclusive"], end + args.context_halo_frames
        )
        bursts = [
            {
                **row,
                "package_role": (
                    "owned_burst"
                    if start <= row["start_frame"] < end
                    else "context_burst"
                ),
            }
            for row in source["bursts"]
            if row["review_range"][1] > context_start
            and row["review_range"][0] < context_end
        ]
        path = output_dir / f"enemy-units-{start:06d}-{end:06d}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "fps": manifest["fps"],
                    "segment": segment,
                    "target_range": [start, end],
                    "owned_event_range": [start, end],
                    "context_range": [context_start, context_end],
                    "bursts": bursts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append({"range": [start, end], "bursts": len(bursts)})
    print(json.dumps({"packages": summaries}))


if __name__ == "__main__":
    main()
