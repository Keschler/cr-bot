from __future__ import annotations

import argparse
import html
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
KATACR_ROOT = ROOT / "vendor/external/KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from cr_bot.app.pipeline import normalize_frame
from cr_bot.eval.visualize_cells import arena_px_for, put_label_box
from cr_bot.features.action_space import ACTION_GRID


DEFAULT_VIDEO = (
    ROOT
    / "dataset_generation/data/video_clips/downloaded_videos/"
    / "HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/eval/action_grid_shift_checks"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render action-grid overlays shifted vertically.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frame-index", type=int, default=1387)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shifts",
        default="0,8,16,24,32,40,48,56,64,72,80",
        help="Comma-separated y shifts in pixels.",
    )
    parser.add_argument("--troop-center", default="569.9,1683.1")
    parser.add_argument("--clock-center", default="573.1,1798.9")
    args = parser.parse_args()

    shifts = [int(value.strip()) for value in args.shifts.split(",") if value.strip()]
    troop_center = parse_point(args.troop_center)
    clock_center = parse_point(args.clock_center)
    out_dir = args.output_dir / f"frame_{args.frame_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = read_frame(args.video, args.frame_index)
    base = normalize_frame(frame)
    arena_px = arena_px_for(base)

    points = [
        ("ice-golem troop", troop_center, (0, 255, 0)),
        ("ally deploy clock", clock_center, (0, 0, 255)),
    ]
    rows = []
    for shift in shifts:
        overlay = base.copy()
        grid_px = draw_shifted_grid(overlay, arena_px, shift)
        labels = []
        for label, point, color in points:
            cell = shifted_cell(point[0], point[1], arena_px, shift)
            labels.append(f"{label}: cell={cell}")
            draw_point(overlay, label, point, color, cell)
        put_label_box(
            overlay,
            [
                f"frame={args.frame_index} grid_shift_y=+{shift}px",
                f"grid_px=({grid_px[0]},{grid_px[1]})-({grid_px[2]},{grid_px[3]})",
                *labels,
            ],
        )
        file_name = f"grid_shift_y_{shift:02d}.png"
        cv2.imwrite(str(out_dir / file_name), overlay)
        rows.append(
            {
                "shift": shift,
                "file": file_name,
                "grid_px": grid_px,
                "labels": labels,
            }
        )

    write_index(out_dir, args.frame_index, rows)
    print(out_dir / "index.html")
    for row in rows:
        grid_px = row["grid_px"]
        labels = "; ".join(row["labels"])
        print(
            f"+{row['shift']:02d}px {row['file']} "
            f"grid=({grid_px[0]},{grid_px[1]})-({grid_px[2]},{grid_px[3]}) "
            f"{labels}"
        )


def parse_point(text: str) -> tuple[float, float]:
    x_text, y_text = text.split(",", maxsplit=1)
    return float(x_text), float(y_text)


def read_frame(video: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"could not read frame {frame_index} from {video}")
        return frame
    finally:
        cap.release()


def draw_shifted_grid(img, arena_px, shift_y: int) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah + shift_y))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah + shift_y))

    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (35, 35, 35), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (35, 35, 35), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 3)
    return x0, y0, x1, y1


def shifted_cell(
    x: float,
    y: float,
    arena_px: tuple[int, int, int, int],
    shift_y: int,
) -> tuple[int, int] | None:
    ax, ay, aw, ah = arena_px
    x0 = ax + ACTION_GRID.x0 * aw
    y0 = ay + ACTION_GRID.y0 * ah + shift_y
    x1 = ax + ACTION_GRID.x1 * aw
    y1 = ay + ACTION_GRID.y1 * ah + shift_y
    if not (x0 <= x < x1 and y0 <= y < y1):
        return None
    col = int((x - x0) / (x1 - x0) * ACTION_GRID.cols)
    row = int((y - y0) / (y1 - y0) * ACTION_GRID.rows)
    return col, row


def draw_point(
    img,
    label: str,
    point: tuple[float, float],
    color: tuple[int, int, int],
    cell: tuple[int, int] | None,
) -> None:
    x, y = point
    center = (int(round(x)), int(round(y)))
    cv2.drawMarker(img, center, color, cv2.MARKER_CROSS, 42, 4)
    cv2.circle(img, center, 18, color, 3)
    text = f"{label} cell={cell}"
    cv2.putText(
        img,
        text,
        (int(x) + 18, int(y) - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (int(x) + 18, int(y) - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def write_index(out_dir: Path, frame_index: int, rows: list[dict]) -> None:
    html_rows = []
    for row in rows:
        grid_px = row["grid_px"]
        labels = html.escape("; ".join(row["labels"]))
        file_name = html.escape(row["file"])
        html_rows.append(
            "<tr>"
            f"<td>+{row['shift']}px</td>"
            f"<td>({grid_px[0]},{grid_px[1]})-({grid_px[2]},{grid_px[3]})</td>"
            f"<td>{labels}</td>"
            f"<td><a href=\"{file_name}\"><img src=\"{file_name}\" loading=\"lazy\"></a></td>"
            "</tr>"
        )
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }}
    img {{ width: 260px; }}
  </style>
</head>
<body>
  <h1>Action Grid Shift Checks - Frame {frame_index}</h1>
  <table>
    <tr><th>Shift</th><th>Grid px</th><th>Cells</th><th>Overlay</th></tr>
    {"".join(html_rows)}
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(content)


if __name__ == "__main__":
    main()
