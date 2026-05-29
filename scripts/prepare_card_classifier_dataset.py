from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.vision.image_utils import HAND_CARD_ART_ROI, crop
from cr_bot.domain.rois import ROIS


DEFAULT_DATASET_ROOT = ROOT / "data/card_classifier"
DEFAULT_LABELS_CSV = DEFAULT_DATASET_ROOT / "labels.csv"

HAND_SLOT_KEYS = {
    "card_1": "hand_card_slot_1",
    "card_2": "hand_card_slot_2",
    "card_3": "hand_card_slot_3",
    "card_4": "hand_card_slot_4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize frame-level hand labels into crop-level card classifier data.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory for the card classifier dataset.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Frame-level auto-label CSV produced from extracted frames.",
    )
    parser.add_argument(
        "--review-score",
        type=int,
        default=40,
        help="Mark crops with scores below this threshold as needs_review.",
    )
    parser.add_argument(
        "--review-score-medium",
        type=int,
        default=70,
        help="Mark crops with scores below this threshold as review_priority=medium.",
    )
    return parser.parse_args()


def parse_label(raw: str) -> tuple[str | None, int]:
    value = ast.literal_eval(raw)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"Unexpected label value: {raw}")
    card_name, score = value
    if card_name is not None and not isinstance(card_name, str):
        raise ValueError(f"Unexpected card name in {raw}")
    return card_name, int(score)


def review_priority(score: int, has_label: bool, low_threshold: int, medium_threshold: int) -> str:
    if not has_label or score < low_threshold:
        return "high"
    if score < medium_threshold:
        return "medium"
    return "low"


def export_crop(
    frame,
    frame_path: Path,
    dataset_root: Path,
    video_id: str,
    slot_name: str,
) -> tuple[Path, tuple[int, int, int, int], str]:
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

    crop_dir = dataset_root / "crops" / crop_type / video_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{frame_path.stem}_{slot_name}.jpg"
    cv2.imwrite(str(crop_path), slot_crop)
    return crop_path, slot_roi, crop_type


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    labels_csv = args.labels_csv.resolve()

    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")

    metadata_dir = dataset_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows: list[dict[str, str | int | bool]] = []

    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        frame_rows = list(reader)

    if not frame_rows:
        raise ValueError(f"No rows found in {labels_csv}")

    for frame_row in frame_rows:
        video_id = frame_row["video_id"]
        frame_path = Path(frame_row["frame_path"])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Failed to read frame: {frame_path}")

        for slot_name in ["card_1", "card_2", "card_3", "card_4", "next_card"]:
            card_name, score = parse_label(frame_row[slot_name])
            crop_path, slot_roi, crop_type = export_crop(
                frame=frame,
                frame_path=frame_path,
                dataset_root=dataset_root,
                video_id=video_id,
                slot_name=slot_name,
            )
            priority = review_priority(
                score=score,
                has_label=card_name is not None,
                low_threshold=args.review_score,
                medium_threshold=args.review_score_medium,
            )
            normalized_rows.append(
                {
                    "video_id": video_id,
                    "frame_path": str(frame_path),
                    "slot_name": slot_name,
                    "crop_type": crop_type,
                    "crop_path": str(crop_path),
                    "card_name": card_name or "",
                    "score": score,
                    "needs_review": priority != "low",
                    "review_priority": priority,
                    "slot_roi_x": slot_roi[0],
                    "slot_roi_y": slot_roi[1],
                    "slot_roi_w": slot_roi[2],
                    "slot_roi_h": slot_roi[3],
                }
            )

    normalized_csv = metadata_dir / "labels_normalized.csv"
    with normalized_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_path",
                "slot_name",
                "crop_type",
                "crop_path",
                "card_name",
                "score",
                "needs_review",
                "review_priority",
                "slot_roi_x",
                "slot_roi_y",
                "slot_roi_w",
                "slot_roi_h",
            ],
        )
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"Wrote {len(normalized_rows)} crop rows to {normalized_csv}")
    print(f"Exported crops under {dataset_root / 'crops'}")


if __name__ == "__main__":
    main()
