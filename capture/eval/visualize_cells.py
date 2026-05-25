from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KATACR_ROOT = ROOT / "vendor/external/KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from eval.action_eval import ActionEvent, evaluate, load_ground_truth, parse_predictions_txt
from features.action_space import ACTION_GRID
from katacr.build_dataset.utils.split_part import process_part, ratio2name
from main import PROCESSING_RESOLUTION, normalize_frame
from rois import ROIS


DEFAULT_OUTPUT_DIR = ROOT / "eval/cell_visualizations"


def load_ground_truth_with_fps(path: Path) -> tuple[list[ActionEvent], float | None]:
    data = json.loads(path.read_text())
    fps = data.get("fps") if isinstance(data, dict) else None
    return load_ground_truth(path), float(fps) if fps is not None else None


def find_matches(
    ground_truth: list[ActionEvent],
    predictions_path: Path | None,
    *,
    side: str,
    video_time_tolerance_s: float,
) -> dict[int, ActionEvent]:
    if predictions_path is None:
        return {}

    predictions = parse_predictions_txt(predictions_path)
    result = evaluate(
        ground_truth,
        predictions,
        side=side,
        time_left_tolerance_s=999.0,
        video_time_tolerance_s=video_time_tolerance_s,
        cell_tolerance=999,
        strict_evolution=False,
    )
    matches: dict[int, ActionEvent] = {}
    for match in result.matches:
        for idx, event in enumerate(ground_truth):
            if event is match.expected:
                matches[idx] = match.predicted
                break
    return matches


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
        if video is not None and not self.cap.isOpened():
            raise ValueError(f"could not open video: {video}")

    def fps(self) -> float | None:
        if self.cap is None:
            return None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None

    def read(self, frame_index: int) -> tuple[object, Path | None]:
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f"could not read frame {frame_index} from {self.video}")
            return normalize_frame(frame), None

        assert self.frames_dir is not None
        path = self._frame_path(frame_index)
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"could not read frame image: {path}")
        return normalize_frame(frame), path

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def _frame_path(self, frame_index: int) -> Path:
        assert self.frames_dir is not None
        direct = self.frames_dir / self.frame_pattern.format(frame_index=frame_index)
        if direct.exists():
            return direct

        candidates = sorted(self.frames_dir.glob(f"*{frame_index:04d}*"))
        candidates.extend(sorted(self.frames_dir.glob(f"*{frame_index:05d}*")))
        candidates.extend(sorted(self.frames_dir.glob(f"*{frame_index:06d}*")))
        for candidate in candidates:
            if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                return candidate
        raise FileNotFoundError(
            f"no frame image found for frame {frame_index}; tried {direct} and numeric glob fallbacks"
        )


def arena_px_for(frame) -> tuple[int, int, int, int]:
    try:
        ratio_name = ratio2name(frame)
        if ratio_name is not None:
            _, box_params = process_part(frame, 2, verbose=True)
            fx, fy, fw, fh = box_params
            frame_h, frame_w = frame.shape[:2]
            return (
                int(frame_w * fx),
                int(frame_h * fy),
                int(frame_w * fw),
                int(frame_h * fh),
            )
    except Exception:
        pass
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
            cv2.putText(
                img,
                f"{col},{row}",
                (int(cx - 13), int(cy + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.27,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"{col},{row}",
                (int(cx - 13), int(cy + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.27,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )


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
    cv2.putText(
        img,
        label,
        (x0, max(24, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def render_event(frame, event: ActionEvent, predicted: ActionEvent | None, *, event_idx: int, label_cells: bool):
    overlay = frame.copy()
    arena_px = arena_px_for(overlay)
    draw_grid(overlay, arena_px, label_cells=label_cells)
    draw_cell(overlay, arena_px, predicted.cell if predicted else None, "pred", (0, 220, 255))
    draw_cell(overlay, arena_px, event.cell, "gt", (0, 255, 0))

    lines = [
        f"event={event_idx} side={event.side} card={event.card}",
        f"gt frame={event.frame_index} video={fmt(event.video_time_s)} cell={event.cell}",
    ]
    if predicted is not None:
        lines.append(
            f"pred video={fmt(predicted.video_time_s)} time_left={fmt(predicted.time_left_s)} cell={predicted.cell}"
        )
    put_label_box(overlay, lines)
    return overlay


def put_label_box(img, lines: list[str]) -> None:
    x, y = 18, 36
    line_h = 27
    max_chars = max(len(line) for line in lines)
    width = min(img.shape[1] - 2 * x, max(520, max_chars * 12))
    height = line_h * len(lines) + 16
    cv2.rectangle(img, (x - 8, y - 26), (x - 8 + width, y - 26 + height), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def write_index(output_dir: Path, rows: list[dict]) -> None:
    html_rows = []
    for row in rows:
        image = html.escape(row["image"])
        html_rows.append(
            "<tr>"
            f"<td>{row['idx']}</td>"
            f"<td>{html.escape(row['card'])}</td>"
            f"<td>{row['frame']}</td>"
            f"<td>{row['video_time']}</td>"
            f"<td>{html.escape(row['gt_cell'])}</td>"
            f"<td>{html.escape(row['pred_cell'])}</td>"
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
  <h1>Cell Visualizations</h1>
  <table>
    <tr><th>#</th><th>card</th><th>frame</th><th>video</th><th>gt cell</th><th>pred cell</th><th>overlay</th></tr>
""" + "\n".join(html_rows) + """
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(content)


def write_suggestions(output_dir: Path, rows: list[dict]) -> None:
    lines = ["idx\tcard\tframe_index\tvideo_time_s\tground_truth_cell\tpredicted_cell\timage"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["idx"]),
                    row["card"],
                    str(row["frame"]),
                    row["video_time"],
                    row["gt_cell"],
                    row["pred_cell"],
                    row["image"],
                ]
            )
        )
    (output_dir / "cell_suggestions.tsv").write_text("\n".join(lines) + "\n")


def event_frame_index(event: ActionEvent, fps: float) -> int:
    if event.frame_index is not None:
        return event.frame_index
    if event.video_time_s is None:
        raise ValueError(f"event has neither frame_index nor video_time_s: {event}")
    return int(round(event.video_time_s * fps))


def fmt(value) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def cell_text(cell: tuple[int, int] | None) -> str:
    return "" if cell is None else f"{cell[0]},{cell[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render action-grid overlays for eval labels.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", type=Path, help="Optional txt output to highlight matched predicted cells.")
    parser.add_argument("--video", type=Path, help="Source video to read frames from.")
    parser.add_argument("--frames-dir", type=Path, help="Directory of extracted frames.")
    parser.add_argument("--frame-pattern", default="{frame_index:06d}.jpg")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--side", choices=["own", "enemy"], default="own")
    parser.add_argument("--fps", type=float, help="Override fps for frame_index/video_time conversion.")
    parser.add_argument("--video-time-tolerance", type=float, default=2.0)
    parser.add_argument("--no-cell-labels", action="store_true", help="Hide per-cell col,row text.")
    args = parser.parse_args()

    ground_truth, ground_truth_fps = load_ground_truth_with_fps(args.ground_truth)
    ground_truth = [event for event in ground_truth if event.side == args.side]
    frame_source = FrameSource(video=args.video, frames_dir=args.frames_dir, frame_pattern=args.frame_pattern)
    fps = args.fps or ground_truth_fps or frame_source.fps() or 10.0
    matched_predictions = find_matches(
        ground_truth,
        args.predictions,
        side=args.side,
        video_time_tolerance_s=args.video_time_tolerance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for idx, event in enumerate(ground_truth, start=1):
            frame_idx = event_frame_index(event, fps)
            frame, _ = frame_source.read(frame_idx)
            predicted = matched_predictions.get(idx - 1)
            overlay = render_event(
                frame,
                event,
                predicted,
                event_idx=idx,
                label_cells=not args.no_cell_labels,
            )
            file_name = f"{idx:03d}_{event.side}_{event.card}_frame_{frame_idx:06d}.jpg"
            out_path = args.output_dir / file_name
            cv2.imwrite(str(out_path), overlay)
            rows.append(
                {
                    "idx": idx,
                    "card": event.card,
                    "frame": frame_idx,
                    "video_time": fmt(event.video_time_s),
                    "gt_cell": cell_text(event.cell),
                    "pred_cell": cell_text(predicted.cell if predicted else None),
                    "image": file_name,
                }
            )
    finally:
        frame_source.close()

    write_index(args.output_dir, rows)
    write_suggestions(args.output_dir, rows)
    print(f"wrote {len(rows)} overlays to {args.output_dir}")
    print(f"open {args.output_dir / 'index.html'}")
    print(f"cell suggestions: {args.output_dir / 'cell_suggestions.tsv'}")


if __name__ == "__main__":
    main()
