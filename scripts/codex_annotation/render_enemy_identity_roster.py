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

from cr_bot.annotation_harness import REVIEW_MAX_PIXELS, atomic_write_json


def _labeled_panel(
    image: np.ndarray,
    *,
    label: str,
    panel_width: int,
) -> np.ndarray:
    rows = 2
    columns = 3
    tile_height = image.shape[0] // rows
    tile_width = image.shape[1] // columns
    # Preserve an early, middle, and late view while avoiding a six-frame
    # attachment for every target.
    selected_indices = (0, 2, 4)
    tiles = []
    for index in selected_indices:
        row, column = divmod(index, columns)
        tiles.append(
            image[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ]
        )
    content = np.concatenate(tiles, axis=1)
    scale = panel_width / content.shape[1]
    content = cv2.resize(
        content,
        (panel_width, max(1, round(content.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    header = np.zeros((72, panel_width, 3), dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (18, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, content], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render compact target-labelled enemy unit identity roster sheets."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="Target document; defaults to RUN_DIR/enemy_identity_targets.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Roster JSON output; defaults to RUN_DIR/enemy_identity_roster.json.",
    )
    parser.add_argument(
        "--output-prefix",
        default="enemy-identity-roster",
        help="Rendered roster sheet filename prefix.",
    )
    parser.add_argument("--panel-width", type=int, default=1200)
    parser.add_argument("--targets-per-sheet", type=int, default=3)
    args = parser.parse_args()
    if args.panel_width <= 0 or args.targets_per_sheet <= 0:
        parser.error("panel layout values must be positive")

    run_dir = args.run_dir.resolve()
    targets_path = (
        args.targets_file.resolve()
        if args.targets_file is not None
        else run_dir / "enemy_identity_targets.json"
    )
    source = json.loads(
        targets_path.read_text(encoding="utf-8")
    )
    targets = [
        row
        for row in source["targets"]
        if row["kind"] == "unit_or_building"
    ]
    review_dir = run_dir / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    sheet_targets: list[list[str]] = []
    for sheet_index, offset in enumerate(
        range(0, len(targets), args.targets_per_sheet)
    ):
        subset = targets[offset : offset + args.targets_per_sheet]
        panels = []
        for target in subset:
            artifacts_for_target = target.get("identity_artifacts", [])
            if len(artifacts_for_target) < 2:
                raise ValueError(
                    f"{target['onset_id']}: delayed identity sheets are required"
                )
            image = cv2.imread(str(run_dir / artifacts_for_target[0]))
            if image is None:
                raise FileNotFoundError(artifacts_for_target[0])
            panels.append(
                _labeled_panel(
                    image,
                    label=(
                        f"TARGET {target['onset_id']}  "
                        f"ONSET {int(target['event_frame_index']):06d}"
                    ),
                    panel_width=args.panel_width,
                )
            )
        width = max(panel.shape[1] for panel in panels)
        height = sum(panel.shape[0] for panel in panels)
        sheet = np.zeros((height, width, 3), dtype=np.uint8)
        top = 0
        for panel in panels:
            sheet[top : top + panel.shape[0], : panel.shape[1]] = panel
            top += panel.shape[0]
        if sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS:
            raise ValueError(
                "identity roster sheet is too large; reduce layout values"
            )
        output = review_dir / f"{args.output_prefix}-{sheet_index:03d}.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(output)
        artifacts.append(str(output.relative_to(run_dir)))
        sheet_targets.append([row["onset_id"] for row in subset])

    document = {
        "run_id": source["run_id"],
        "stage": "enemy_identity_roster",
        "artifacts": artifacts,
        "sheets": [
            {"artifact": artifact, "onset_ids": onset_ids}
            for artifact, onset_ids in zip(
                artifacts,
                sheet_targets,
                strict=True,
            )
        ],
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "enemy_identity_roster.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, document)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "targets": len(targets),
                "sheets": math.ceil(len(targets) / args.targets_per_sheet),
            }
        )
    )


if __name__ == "__main__":
    main()
