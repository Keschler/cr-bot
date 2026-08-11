from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import REVIEW_MAX_FRAMES, render_review_sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a complete interval as compact, indexed review sheets. "
            "Use this instead of one unreadable full-interval montage."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--side", choices=["own", "enemy"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--chunk-frames", type=int)
    parser.add_argument("--tile-width", type=int, default=360)
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segment = manifest["segment"]
    start = (
        segment["start_frame"]
        if args.start_frame is None
        else args.start_frame
    )
    end = (
        segment["end_frame_exclusive"]
        if args.end_frame is None
        else args.end_frame
    )
    purpose = "own_context" if args.side == "own" else "arena"
    limit = REVIEW_MAX_FRAMES[purpose]
    chunk_frames = limit if args.chunk_frames is None else args.chunk_frames
    if chunk_frames <= 0 or chunk_frames > limit:
        parser.error(f"--chunk-frames must be in 1..{limit} for {args.side}")
    if start < segment["start_frame"] or end > segment["end_frame_exclusive"]:
        parser.error("requested range is outside the prepared segment")
    if start >= end:
        parser.error("requested range must not be empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranges: list[list[int]] = []
    artifacts: list[str] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + chunk_frames)
        output = args.output_dir / (
            f"completeness-{args.side}-{cursor:06d}-{chunk_end:06d}.jpg"
        )
        render_review_sheet(
            run_dir=args.run_dir,
            output_path=output,
            start_frame=cursor,
            end_frame=chunk_end,
            purpose=purpose,
            columns=min(5, chunk_end - cursor),
            tile_width=args.tile_width,
        )
        ranges.append([cursor, chunk_end])
        try:
            artifact = str(output.resolve().relative_to(args.run_dir.resolve()))
        except ValueError:
            artifact = str(output.resolve())
        artifacts.append(artifact)
        cursor = chunk_end

    print(
        json.dumps(
            {
                "side": args.side,
                "reviewed_ranges": ranges,
                "review_artifacts": artifacts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
