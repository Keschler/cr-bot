from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import render_review_sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render every compact staged candidate sheet in one command."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tile-width",
        type=int,
        default=360,
        help="Default width retained for compatibility.",
    )
    parser.add_argument("--own-tile-width", type=int)
    parser.add_argument("--enemy-tile-width", type=int)
    parser.add_argument(
        "--skip-own",
        action="store_true",
        help="Render only arena scan sheets; dense own discovery is separate.",
    )
    args = parser.parse_args()
    own_tile_width = args.own_tile_width or args.tile_width
    enemy_tile_width = args.enemy_tile_width or args.tile_width

    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    discovery = manifest["candidate_discovery"]
    own = discovery["own_candidates"]
    enemy = discovery["enemy_scan_windows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in ([] if args.skip_own else own):
        candidate_id = candidate["candidate_id"]
        suffix = candidate_id.replace(":", "-")
        render_review_sheet(
            run_dir=args.run_dir,
            output_path=args.output_dir / f"verify-{suffix}.jpg",
            candidate_id=candidate_id,
            purpose="own_context",
            columns=3,
            tile_width=own_tile_width,
        )
        approximate = candidate["approximate_frame_index"]
        segment_end = manifest["segment"]["end_frame_exclusive"]
        confirmation_start = min(segment_end, approximate + max(1, round(0.5 * manifest["fps"])))
        confirmation_end = min(segment_end, confirmation_start + 6)
        if confirmation_start < confirmation_end:
            render_review_sheet(
                run_dir=args.run_dir,
                output_path=args.output_dir / f"release-{suffix}.jpg",
                start_frame=confirmation_start,
                end_frame=confirmation_end,
                event_id=f"release-{suffix}",
                purpose="own_confirmation",
                columns=3,
                tile_width=max(own_tile_width, 480),
            )

    for candidate in enemy:
        candidate_id = candidate["candidate_id"]
        suffix = candidate_id.replace(":", "-")
        frame_count = (
            candidate["inspection_end_frame_exclusive"]
            - candidate["inspection_start_frame"]
        )
        render_review_sheet(
            run_dir=args.run_dir,
            output_path=args.output_dir / f"verify-{suffix}.jpg",
            candidate_id=candidate_id,
            purpose="arena",
            columns=min(5, frame_count),
            tile_width=enemy_tile_width,
        )

    print(
        json.dumps(
            {
                "own_candidate_sheets": 0 if args.skip_own else len(own),
                "own_release_sheets": 0 if args.skip_own else len(own),
                "enemy_candidate_sheets": len(enemy),
                "output_dir": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
