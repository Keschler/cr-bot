from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "capture"
INVOCATION_CWD = Path.cwd()
sys.path.insert(0, str(CAPTURE_ROOT))
os.chdir(CAPTURE_ROOT)

from image_utils import draw_rois  # noqa: E402
from main import (  # noqa: E402
    normalize_frame,
    process_frame,
    render_match_debug,
    render_timer_debug,
    render_tower_hp_debug,
)
from rois import ROIS  # noqa: E402
from vision.yolo_runtime import build_detector, summarize_detections  # noqa: E402


def default_output_dir(frame_path: Path) -> Path:
    return Path("/tmp") / f"cr-bot-debug-{frame_path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save capture debug images for one frame.")
    parser.add_argument("frame", type=Path, help="Frame image path to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write debug images. Defaults to /tmp/cr-bot-debug-<frame-stem>.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Process the frame at its current size instead of resizing to 1080x2400.",
    )
    parser.add_argument(
        "--no-yolo-tower-hp",
        action="store_true",
        help="Use fixed tower HP ROIs instead of YOLO tower-bar detections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_path = args.frame.expanduser()
    if not frame_path.is_absolute():
        frame_path = INVOCATION_CWD / frame_path
    frame_path = frame_path.resolve()
    out_dir = args.output_dir or default_output_dir(frame_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to read frame: {frame_path}")
    if not args.no_normalize:
        frame = normalize_frame(frame)

    detector = build_detector()
    result = process_frame(
        frame,
        detector,
        show_rois=False,
        yolo_tower_hp_detections=not args.no_yolo_tower_hp,
    )

    cv2.imwrite(str(out_dir / "00_original.jpg"), frame)
    cv2.imwrite(str(out_dir / "01_yolo_rendered.jpg"), result["rendered"])
    cv2.imwrite(str(out_dir / "02_rois.jpg"), draw_rois(frame.copy(), ROIS))
    cv2.imwrite(str(out_dir / "03_timer_debug.jpg"), render_timer_debug(result["timer_debug_steps"]))
    cv2.imwrite(str(out_dir / "04_tower_hp_debug.jpg"), render_tower_hp_debug(result["tower_hp_debug_steps"]))
    cv2.imwrite(str(out_dir / "05_match_debug.jpg"), render_match_debug(frame, result["matches"]))

    for name, image in result["timer_debug_steps"].items():
        cv2.imwrite(str(out_dir / f"timer_step_{name}.jpg"), image)

    summary = {
        "frame": str(frame_path),
        "output_dir": str(out_dir),
        "time": result["time"],
        "time_left_s": result["time_left_s"],
        "total_remaining_s": result["total_remaining_s"],
        "overtime": result["overtime"],
        "elixir": result["elixir"]["estimated_value"] + result["elixir"]["displayed_digit"][0],
        "elixir_digit": result["elixir"]["displayed_digit"],
        "state": result["state"],
        "towers": result["towers_hp"],
        "yolo": summarize_detections(result["yolo_boxes"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(out_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
