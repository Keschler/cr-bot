from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.vision.cards import extract_hand_state
from cr_bot.vision.image_utils import HAND_CARD_ART_ROI, crop
from cr_bot.domain.rois import ROIS


DATASET_ROOT = ROOT / "data/card_classifier"
DEFAULT_FRAMES_ROOT = DATASET_ROOT / "frames"
DEFAULT_METADATA_CSV = DATASET_ROOT / "metadata/labels_normalized.csv"

HAND_SLOT_KEYS = {
    "card_1": "hand_card_slot_1",
    "card_2": "hand_card_slot_2",
    "card_3": "hand_card_slot_3",
    "card_4": "hand_card_slot_4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append normalized card-classifier labels for one or more frame directories.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help="Root containing per-video frame directories.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="Normalized metadata CSV to append to.",
    )
    parser.add_argument(
        "--video",
        action="append",
        dest="videos",
        default=[],
        help="Video/frame directory name to import. Repeat for multiple videos.",
    )
    parser.add_argument(
        "--review-score",
        type=int,
        default=40,
        help="Mark scores below this as high-priority review.",
    )
    parser.add_argument(
        "--review-score-medium",
        type=int,
        default=70,
        help="Mark scores below this as medium-priority review.",
    )
    return parser.parse_args()


def review_priority(score: int, has_label: bool, low_threshold: int, medium_threshold: int) -> str:
    if not has_label or score < low_threshold:
        return "high"
    if score < medium_threshold:
        return "medium"
    return "low"


def export_crop(frame, frame_path: Path, video_id: str, slot_name: str) -> tuple[Path, tuple[int, int, int, int], str]:
    if slot_name == "next_card":
        slot_roi_name = "next_card_slot"
        slot_roi = ROIS[slot_roi_name]
        slot_crop = crop(frame, slot_roi)
        crop_type = "next"
    else:
        slot_roi_name = HAND_SLOT_KEYS[slot_name]
        slot_roi = ROIS[slot_roi_name]
        hand_crop = crop(frame, slot_roi)
        slot_crop = crop(hand_crop, HAND_CARD_ART_ROI)
        crop_type = "hand"

    crop_dir = DATASET_ROOT / "crops" / crop_type / video_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{frame_path.stem}_{slot_name}.jpg"
    cv2.imwrite(str(crop_path), slot_crop)
    return crop_path, slot_roi, crop_type


def main() -> None:
    args = parse_args()
    frames_root = args.frames_root.resolve()
    metadata_csv = args.metadata_csv.resolve()

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")
    if not args.videos:
        raise ValueError("Pass at least one --video name to import.")

    with metadata_csv.open(newline="", encoding="utf-8-sig") as f:
        existing_rows = list(csv.DictReader(f))

    existing_frame_paths = {row["frame_path"] for row in existing_rows}
    fieldnames = list(existing_rows[0].keys())
    appended_rows: list[dict[str, str | int | bool]] = []

    for video_id in args.videos:
        frame_dir = frames_root / video_id
        if not frame_dir.exists():
            raise FileNotFoundError(f"Missing frame directory: {frame_dir}")

        for frame_path in sorted(frame_dir.glob("*.jpg")):
            frame_path_str = str(frame_path.resolve())
            if frame_path_str in existing_frame_paths:
                continue

            frame = cv2.imread(frame_path_str)
            if frame is None:
                continue

            state = extract_hand_state(frame)
            for slot_name in ["card_1", "card_2", "card_3", "card_4", "next_card"]:
                card_name, score = state[slot_name]
                if not card_name:
                    continue
                crop_path, slot_roi, crop_type = export_crop(frame, frame_path, video_id, slot_name)
                priority = review_priority(
                    score=score,
                    has_label=True,
                    low_threshold=args.review_score,
                    medium_threshold=args.review_score_medium,
                )
                appended_rows.append(
                    {
                        "video_id": video_id,
                        "frame_path": frame_path_str,
                        "slot_name": slot_name,
                        "crop_type": crop_type,
                        "crop_path": str(crop_path.resolve()),
                        "card_name": card_name,
                        "score": score,
                        "needs_review": priority != "low",
                        "review_priority": priority,
                        "slot_roi_x": slot_roi[0],
                        "slot_roi_y": slot_roi[1],
                        "slot_roi_w": slot_roi[2],
                        "slot_roi_h": slot_roi[3],
                    }
                )

    all_rows = existing_rows + appended_rows
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Appended {len(appended_rows)} rows to {metadata_csv}")


if __name__ == "__main__":
    main()
