from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import prepare_annotation_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a blind, local video-annotation run for visual review by Codex."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0, dest="start_time_s")
    parser.add_argument("--end", type=float, dest="end_time_s")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--own-change-threshold", type=float, default=10.0)
    parser.add_argument(
        "--enemy-window-seconds",
        type=float,
        default=2.0,
        help="Length of each chronological enemy discovery window.",
    )
    args = parser.parse_args()

    manifest = prepare_annotation_run(
        video_path=args.video,
        output_dir=args.output_dir,
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        jpeg_quality=args.jpeg_quality,
        own_change_threshold=args.own_change_threshold,
        enemy_window_seconds=args.enemy_window_seconds,
    )
    print(args.output_dir / "manifest.json")
    print(
        f"prepared_frames={len(manifest['frames'])} "
        f"own_candidates={len(manifest['candidate_discovery']['own_candidates'])} "
        f"enemy_windows={len(manifest['candidate_discovery']['enemy_scan_windows'])}"
    )
    print(f"stage 1: fill {args.output_dir / 'verification.json'}")
    print(
        "stage 2: a fresh session fills release_review.json, then checkpoint "
        "release_review and verification before rendering macro or grid reviews"
    )


if __name__ == "__main__":
    main()
