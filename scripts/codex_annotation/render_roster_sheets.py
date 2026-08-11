from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import add_frame_identity
from cr_bot.domain.rois import ROIS


def _contact_sheet(tiles: list[np.ndarray], *, columns: int) -> np.ndarray:
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * height : row * height + tile.shape[0],
            column * width : column * width + tile.shape[1],
        ] = tile
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render sparse, grid-free sheets for blind deck-roster discovery. "
            "These are context aids, not event evidence."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=["own", "enemy"], required=True)
    parser.add_argument("--sample-every-frames", type=int, default=10)
    parser.add_argument("--frames-per-sheet", type=int, default=10)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=360)
    args = parser.parse_args()

    if min(
        args.sample_every_frames,
        args.frames_per_sheet,
        args.columns,
        args.tile_width,
    ) <= 0:
        parser.error("sampling and layout arguments must be positive")
    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    records = {
        int(row["source_frame_index"]): row for row in manifest["frames"]
    }
    segment = manifest["segment"]
    selected = list(
        range(
            segment["start_frame"],
            segment["end_frame_exclusive"],
            args.sample_every_frames,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    arena_x, arena_y, arena_width, arena_height = ROIS["battlefield"]
    for sheet_index, offset in enumerate(
        range(0, len(selected), args.frames_per_sheet)
    ):
        frame_indexes = selected[offset : offset + args.frames_per_sheet]
        tiles: list[np.ndarray] = []
        for frame_index in frame_indexes:
            labeled = cv2.imread(str(args.run_dir / records[frame_index]["path"]))
            if labeled is None:
                raise FileNotFoundError(records[frame_index]["path"])
            frame = labeled[manifest["label_margin_px"] :]
            if args.side == "enemy":
                view = frame[
                    arena_y : arena_y + arena_height,
                    arena_x : arena_x + arena_width,
                ]
            else:
                view = frame[arena_y:, arena_x : arena_x + arena_width]
            view = add_frame_identity(
                view, source_frame_index=frame_index, fps=float(manifest["fps"])
            )
            scale = args.tile_width / view.shape[1]
            tiles.append(
                cv2.resize(
                    view,
                    (args.tile_width, max(1, round(view.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            )
        output = args.output_dir / (
            f"roster-{args.side}-{sheet_index:03d}-"
            f"{frame_indexes[0]:06d}-{frame_indexes[-1]:06d}.jpg"
        )
        if not cv2.imwrite(str(output), _contact_sheet(tiles, columns=args.columns)):
            raise OSError(output)
        outputs.append(str(output.resolve()))
    print(json.dumps({"side": args.side, "artifacts": outputs}, indent=2))


if __name__ == "__main__":
    main()
