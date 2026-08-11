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

from cr_bot.annotation_harness import (
    REVIEW_MAX_PIXELS,
    add_frame_identity,
    atomic_write_json,
)
from cr_bot.annotation_stages import record_review
from cr_bot.vision.deployment_markers import (
    ENEMY_ARENA_ROI,
    marker_review_frame_indices,
)


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


def _bounded_crop(
    image: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    width = min(width, image.shape[1])
    height = min(height, image.shape[0])
    left = max(0, min(image.shape[1] - width, center_x - width // 2))
    top = max(0, min(image.shape[0] - height, center_y - height // 2))
    return image[top : top + height, left : left + width]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render highlighted, card-free enemy marker candidate sheets."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tile-width", type=int, default=420)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    candidate_path = (
        args.candidates.resolve()
        if args.candidates is not None
        else run_dir / "enemy_marker_candidates.json"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "reviews"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    document = json.loads(candidate_path.read_text(encoding="utf-8"))
    frames = {
        int(row["source_frame_index"]): row for row in manifest["frames"]
    }
    tracks = {
        int(row["track_id"]): {
            int(frame): bbox
            for frame, bbox in zip(
                row["observation_frames"],
                row["observation_bboxes"],
                strict=True,
            )
        }
        for row in document["candidates"]
    }
    segment = manifest["segment"]
    roi_x, roi_y, roi_width, roi_height = ENEMY_ARENA_ROI
    rendered = []
    for burst in document["bursts"]:
        anchor = int(burst["start_frame"])
        frame_indices = marker_review_frame_indices(
            anchor_frame=anchor,
            marker_end_frame_exclusive=int(burst["end_frame_exclusive"]),
            segment_start_frame=int(segment["start_frame"]),
            segment_end_frame_exclusive=int(segment["end_frame_exclusive"]),
        )
        start = frame_indices[0]
        end = frame_indices[-1] + 1
        full_tiles = []
        focus_tiles = []
        first_bboxes = burst["first_bboxes"]
        focus_x = round(
            sum(int(left) + int(width) / 2 for left, _, width, _ in first_bboxes)
            / len(first_bboxes)
        )
        focus_y = round(
            sum(int(top) + int(height) / 2 for _, top, _, height in first_bboxes)
            / len(first_bboxes)
        )
        for frame_index in frame_indices:
            labeled = cv2.imread(str(run_dir / frames[frame_index]["path"]))
            if labeled is None:
                raise FileNotFoundError(frames[frame_index]["path"])
            content = labeled[manifest["label_margin_px"] :].copy()
            for track_id in burst["track_ids"]:
                bbox = tracks[int(track_id)].get(frame_index)
                if bbox is None:
                    continue
                left, top, width, height = (int(value) for value in bbox)
                cv2.rectangle(
                    content,
                    (left - 5, top - 5),
                    (left + width + 5, top + height + 5),
                    (0, 255, 255),
                    4,
                )
                cv2.putText(
                    content,
                    f"T{track_id}",
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    content,
                    f"T{track_id}",
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            full_view = content[
                roi_y : roi_y + roi_height,
                roi_x : roi_x + roi_width,
            ]
            full_view = add_frame_identity(
                full_view,
                source_frame_index=frame_index,
                fps=float(manifest["fps"]),
            )
            scale = args.tile_width / full_view.shape[1]
            full_tiles.append(
                cv2.resize(
                    full_view,
                    (
                        args.tile_width,
                        max(1, round(full_view.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            )
            focus_view = _bounded_crop(
                content,
                center_x=focus_x,
                center_y=focus_y,
                width=560,
                height=720,
            )
            focus_view = add_frame_identity(
                focus_view,
                source_frame_index=frame_index,
                fps=float(manifest["fps"]),
            )
            # Overlapping spawn bodies (notably Mega Knight) became
            # indistinguishable at the 360 px full-arena economy setting.
            # Preserve the empirically passing focus resolution independently.
            focus_tile_width = max(420, args.tile_width)
            focus_scale = focus_tile_width / focus_view.shape[1]
            focus_tiles.append(
                cv2.resize(
                    focus_view,
                    (
                        focus_tile_width,
                        max(1, round(focus_view.shape[0] * focus_scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            )
        sheet = _contact_sheet(full_tiles, columns=min(3, len(full_tiles)))
        if sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS:
            raise ValueError(
                f"{burst['burst_id']} sheet is too large; lower --tile-width"
            )
        suffix = burst["burst_id"].replace(":", "-")
        output = output_dir / f"{suffix}.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(output)
        focus_sheet = _contact_sheet(
            focus_tiles, columns=min(3, len(focus_tiles))
        )
        if focus_sheet.shape[0] * focus_sheet.shape[1] > REVIEW_MAX_PIXELS:
            raise ValueError(
                f"{burst['burst_id']} focus sheet is too large; "
                "lower --tile-width"
            )
        focus_output = output_dir / f"{suffix}-focus.jpg"
        if not cv2.imwrite(str(focus_output), focus_sheet):
            raise OSError(focus_output)
        record_review(
            run_dir=run_dir,
            output_path=output,
            purpose="arena",
            start_frame=start,
            end_frame=end,
            candidate_id=None,
            event_id=burst["burst_id"],
        )
        record_review(
            run_dir=run_dir,
            output_path=focus_output,
            purpose="identity",
            start_frame=start,
            end_frame=end,
            candidate_id=None,
            event_id=burst["burst_id"],
        )
        try:
            artifact = str(output.resolve().relative_to(run_dir))
        except ValueError:
            artifact = str(output.resolve())
        burst["review_artifact"] = artifact
        try:
            focus_artifact = str(focus_output.resolve().relative_to(run_dir))
        except ValueError:
            focus_artifact = str(focus_output.resolve())
        burst["focus_review_artifact"] = focus_artifact
        burst["review_range"] = [start, end]
        burst["sampled_frame_indices"] = frame_indices
        rendered.extend([str(output), str(focus_output)])
    atomic_write_json(candidate_path, document)
    print(json.dumps({"rendered": len(rendered), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
