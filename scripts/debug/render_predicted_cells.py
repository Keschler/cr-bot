from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
KATACR_ROOT = ROOT / "vendor/external/KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from cr_bot.eval.action_eval import ActionEvent, parse_predictions_txt
from cr_bot.app.pipeline import normalize_frame
from cr_bot.domain.rois import ROIS
from cr_bot.features.action_space import ACTION_GRID


DEFAULT_PREDICTIONS = ROOT / "outputs/video/capture/3400Ladder.txt"
DEFAULT_VIDEO = ROOT / "dataset_generation/data/video_clips/downloaded_videos/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/eval/cell_visualizations/3400Ladder_enemy_plays"


class FrameSource:
    def __init__(self, *, video: Path | None, frames_dir: Path | None, frame_pattern: str):
        if video is None and frames_dir is None:
            raise ValueError("provide either --video or --frames-dir")
        if video is not None and frames_dir is not None:
            raise ValueError("provide only one of --video or --frames-dir")
        self.video = video
        self.frames_dir = frames_dir
        self.frame_pattern = frame_pattern
        self.cap = cv2.VideoCapture(str(video)) if video is not None else None
        if self.cap is not None and not self.cap.isOpened():
            raise ValueError(f"could not open video: {video}")

    def fps(self) -> float | None:
        if self.cap is None:
            return None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None

    def read(self, frame_index: int):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f"could not read frame {frame_index} from {self.video}")
            return normalize_frame(frame), None
        assert self.frames_dir is not None
        path = self.frames_dir / self.frame_pattern.format(frame_index=frame_index)
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"could not read frame image: {path}")
        return normalize_frame(frame), path

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def arena_px_for(frame) -> tuple[int, int, int, int]:
    return ROIS["battlefield"]


def draw_grid(img, arena_px, *, label_cells: bool) -> None:
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))

    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (35, 35, 35), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (35, 35, 35), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)

    if not label_cells:
        return

    for row in range(ACTION_GRID.rows):
        for col in range(ACTION_GRID.cols):
            cx, cy = ACTION_GRID.cell_to_pixel_center(col, row, arena_px)
            cv2.putText(img, f"{col},{row}", (int(cx - 13), int(cy + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{col},{row}", (int(cx - 13), int(cy + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (245, 245, 245), 1, cv2.LINE_AA)


def draw_cell(img, arena_px, cell: tuple[int, int] | None, label: str, color: tuple[int, int, int]) -> None:
    if cell is None:
        return
    col, row = cell
    cx, cy = ACTION_GRID.cell_to_pixel_center(col, row, arena_px)
    aw = arena_px[2]
    ah = arena_px[3]
    cell_w = ACTION_GRID.width * aw / ACTION_GRID.cols
    cell_h = ACTION_GRID.height * ah / ACTION_GRID.rows
    x0 = int(round(cx - cell_w / 2))
    y0 = int(round(cy - cell_h / 2))
    x1 = int(round(cx + cell_w / 2))
    y1 = int(round(cy + cell_h / 2))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 4)
    cv2.drawMarker(img, (int(round(cx)), int(round(cy))), color, cv2.MARKER_CROSS, 24, 2)
    cv2.putText(img, label, (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def put_label_box(img, lines: list[str]) -> None:
    x, y = 18, 36
    line_h = 27
    max_chars = max(len(line) for line in lines)
    width = min(img.shape[1] - 2 * x, max(520, max_chars * 12))
    height = line_h * len(lines) + 16
    cv2.rectangle(img, (x - 8, y - 26), (x - 8 + width, y - 26 + height), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(img, line, (x, y + idx * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def fmt(value) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def cell_text(cell: tuple[int, int] | None) -> str:
    return "" if cell is None else f"{cell[0]},{cell[1]}"


def render_prediction(frame, event: ActionEvent, *, event_idx: int, label_cells: bool):
    overlay = frame.copy()
    arena_px = arena_px_for(overlay)
    draw_grid(overlay, arena_px, label_cells=label_cells)
    draw_cell(overlay, arena_px, event.cell, "pred", (0, 220, 255))
    put_label_box(
        overlay,
        [
            f"event={event_idx} side={event.side} card={event.card}",
            f"video={fmt(event.video_time_s)} time_left={fmt(event.time_left_s)} cell={event.cell}",
            f"track_id={event.track_id}",
        ],
    )
    return overlay


def write_index(output_dir: Path, rows: list[dict]) -> None:
    html_rows = []
    for row in rows:
        image = row["image"]
        html_rows.append(
            "<tr>"
            f"<td>{row['idx']}</td>"
            f"<td>{row['card']}</td>"
            f"<td>{row['frame']}</td>"
            f"<td>{row['video_time']}</td>"
            f"<td>{row['time_left']}</td>"
            f"<td>{row['cell']}</td>"
            f"<td>{row['track_id']}</td>"
            f"<td><a href=\"{image}\"><img src=\"{image}\" loading=\"lazy\"></a></td>"
            "</tr>"
        )
    content = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: sans-serif; margin: 20px; }
    table { border-collapse: collapse; }
    td, th { border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }
    img { width: 260px; }
  </style>
</head>
<body>
  <h1>Predicted Cell Visualizations</h1>
  <table>
    <tr><th>#</th><th>card</th><th>frame</th><th>video</th><th>time_left</th><th>cell</th><th>track_id</th><th>overlay</th></tr>
""" + "\n".join(html_rows) + """
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(content)


def event_frame_index(event: ActionEvent, fps: float) -> int:
    if event.frame_index is not None:
        return event.frame_index
    if event.video_time_s is None:
        raise ValueError(f"event has neither frame_index nor video_time_s: {event}")
    return int(round(event.video_time_s * fps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render predicted action-grid cells from a capture txt.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--frame-pattern", default="{frame_index:06d}.jpg")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--side", choices=["enemy", "own"], default="enemy")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--no-cell-labels", action="store_true")
    args = parser.parse_args()

    events = [event for event in parse_predictions_txt(args.predictions) if event.side == args.side]
    frame_source = FrameSource(video=args.video, frames_dir=args.frames_dir, frame_pattern=args.frame_pattern)
    fps = args.fps or frame_source.fps() or 10.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    try:
        for idx, event in enumerate(events, start=1):
            frame_idx = event_frame_index(event, fps)
            frame, _ = frame_source.read(frame_idx)
            overlay = render_prediction(
                frame,
                event,
                event_idx=idx,
                label_cells=not args.no_cell_labels,
            )
            file_name = f"{idx:03d}_{event.side}_{event.card}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(args.output_dir / file_name), overlay)
            rows.append(
                {
                    "idx": idx,
                    "card": event.card,
                    "frame": frame_idx,
                    "video_time": fmt(event.video_time_s),
                    "time_left": fmt(event.time_left_s),
                    "cell": cell_text(event.cell),
                    "track_id": "" if event.track_id is None else str(event.track_id),
                    "image": file_name,
                }
            )
            summary.append(
                {
                    "side": event.side,
                    "card": event.card,
                    "video_time_s": event.video_time_s,
                    "time_left_s": event.time_left_s,
                    "cell": list(event.cell) if event.cell is not None else None,
                    "track_id": event.track_id,
                }
            )
    finally:
        frame_source.close()

    write_index(args.output_dir, rows)
    (args.output_dir / "cell_suggestions.tsv").write_text(
        "idx\tcard\tframe_index\tvideo_time_s\ttime_left_s\tcell\ttrack_id\timage\n"
        + "\n".join(
            "\t".join(
                [
                    str(row["idx"]),
                    row["card"],
                    str(row["frame"]),
                    row["video_time"],
                    row["time_left"],
                    row["cell"],
                    row["track_id"],
                    row["image"],
                ]
            )
            for row in rows
        )
        + "\n"
    )
    (args.output_dir / f"{args.side}_plays.json").write_text(json.dumps({"events": summary}, indent=2) + "\n")
    print(f"wrote {len(rows)} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
