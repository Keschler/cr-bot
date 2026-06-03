from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.app.pipeline import normalize_frame  # noqa: E402
from cr_bot.paths import KATACR_ROOT  # noqa: E402
from cr_bot.vision.tower_hp import extract_tower_hp_crops, extract_tower_hp_crops_from_yolo  # noqa: E402
from cr_bot.vision.yolo_runtime import build_detector, remap_boxes_to_frame  # noqa: E402

if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.build_dataset.utils.split_part import process_part  # noqa: E402


DEFAULT_VIDEO = ROOT / "dataset_generation/data/video_clips/downloaded_videos/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/tower_hp_ocr_crops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tower HP OCR crops and a review CSV for CRNN training."
    )
    parser.add_argument("--video", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-every", type=int, default=10, help="Export every Nth decoded frame.")
    parser.add_argument("--duration", type=float, default=None, help="Optional max video seconds per input.")
    parser.add_argument(
        "--mode",
        choices=("fixed", "yolo", "both"),
        default="both",
        help="fixed exports live ROIs; yolo exports expert tower-bar crops.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Optional source id. Defaults to the video stem.",
    )
    return parser.parse_args()


def frame_yolo_boxes(frame, detector):
    arena, box_params = process_part(frame, 2, verbose=True)
    fx, fy, fw, fh = box_params
    frame_h, frame_w = frame.shape[:2]
    arena_px = (
        int(frame_w * fx),
        int(frame_h * fy),
        int(frame_w * fw),
        int(frame_h * fh),
    )
    result = detector.infer(arena)
    yolo_boxes = getattr(result, "untracked_data", result.get_data())
    return remap_boxes_to_frame(yolo_boxes, arena.shape, arena_px)


def write_crop(image, output_root: Path, source_id: str, mode: str, frame_idx: int, tower_name: str) -> Path:
    crop_dir = output_root / "images" / source_id / mode
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / f"frame_{frame_idx:06d}_{tower_name}.png"
    cv2.imwrite(str(path), image)
    return path


def export_video(video_path: Path, args: argparse.Namespace, writer: csv.DictWriter, detector) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    source_id = args.source_id or video_path.stem
    count = 0
    try:
        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if args.duration is not None and video_time_s > args.duration:
                break
            if frame_idx % args.sample_every != 0:
                continue

            frame = normalize_frame(frame)
            crop_sets = []
            if args.mode in ("fixed", "both"):
                crop_sets.extend(extract_tower_hp_crops(frame))
            if args.mode in ("yolo", "both"):
                yolo_boxes = frame_yolo_boxes(frame, detector)
                crops, _result, _paused = extract_tower_hp_crops_from_yolo(frame, yolo_boxes)
                crop_sets.extend(crops)

            for tower_crop in crop_sets:
                crop_path = write_crop(
                    tower_crop.image,
                    args.output_root,
                    source_id,
                    tower_crop.mode,
                    frame_idx,
                    tower_crop.tower_name,
                )
                writer.writerow(
                    {
                        "image_path": crop_path.relative_to(ROOT),
                        "source_id": source_id,
                        "frame_index": frame_idx,
                        "video_time_s": f"{video_time_s:.3f}",
                        "tower_name": tower_crop.tower_name,
                        "crop_mode": tower_crop.mode,
                        "readable": "",
                        "label": "",
                        "notes": "",
                    }
                )
                count += 1
    finally:
        cap.release()

    return count


def main() -> None:
    args = parse_args()
    videos = args.video or [DEFAULT_VIDEO]
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "labels.csv"

    detector = build_detector() if args.mode in ("yolo", "both") else None
    fieldnames = [
        "image_path",
        "source_id",
        "frame_index",
        "video_time_s",
        "tower_name",
        "crop_mode",
        "readable",
        "label",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        total = 0
        for video_path in videos:
            total += export_video(video_path, args, writer, detector)

    print(f"exported {total} crops")
    print(f"review CSV: {csv_path}")


if __name__ == "__main__":
    main()
