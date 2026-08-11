from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import render_review_sheet


def _grid_center(value: str) -> tuple[int, int]:
    try:
        column, row = value.split(",", maxsplit=1)
        return int(column), int(row)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected COLUMN,ROW") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render frame-numbered contact sheets for blind Codex annotation."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-id",
        help=(
            "Deterministic event ID; required for own confirmation, identity, "
            "macro, and grid reviews."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidate-id")
    selection.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--purpose",
        choices=[
            "full",
            "arena",
            "own_context",
            "own_confirmation",
            "identity",
            "macro",
            "grid",
        ],
        default="arena",
    )
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--grid-center", type=_grid_center)
    parser.add_argument("--grid-radius", type=int, default=3)
    parser.add_argument(
        "--focus-cell",
        type=_grid_center,
        help="Approximate COLUMN,ROW used only to crop an identity review; no grid is drawn.",
    )
    parser.add_argument("--focus-radius", type=int, default=4)
    args = parser.parse_args()
    if args.start_frame is not None and args.end_frame is None:
        parser.error("--end-frame is required with --start-frame")

    output = render_review_sheet(
        run_dir=args.run_dir,
        output_path=args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        candidate_id=args.candidate_id,
        event_id=args.event_id,
        purpose=args.purpose,
        columns=args.columns,
        tile_width=args.tile_width,
        grid_center=args.grid_center,
        grid_radius=args.grid_radius,
        focus_cell=args.focus_cell,
        focus_radius=args.focus_radius,
    )
    print(output)


if __name__ == "__main__":
    main()
