from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CSV = ROOT / "data/card_classifier/metadata/split.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data/card_classifier/imagefolder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build torchvision ImageFolder directories from card-classifier split metadata.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help="Crop-level metadata CSV with split assignments.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination root for ImageFolder-style directories.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="Whether to copy images or create symlinks.",
    )
    return parser.parse_args()


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    split_csv = args.split_csv.resolve()
    output_root = args.output_root.resolve()

    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")

    rows_written = 0
    classes_by_crop_type: dict[str, set[str]] = {"hand": set(), "next": set()}
    rows = []
    with split_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_name = row["card_name"].strip()
            if not card_name:
                continue
            rows.append(row)
            classes_by_crop_type[row["crop_type"]].add(card_name)

    # Create every class directory in every split so ImageFolder keeps a stable class_to_idx.
    for crop_type, classes in classes_by_crop_type.items():
        for split in ["train", "val", "test"]:
            for card_name in classes:
                (output_root / crop_type / split / card_name).mkdir(parents=True, exist_ok=True)

    for row in rows:
        card_name = row["card_name"].strip()
        if not card_name:
            continue
        src = Path(row["crop_path"]).resolve()
        crop_type = row["crop_type"]
        split = row["split"]
        dst = output_root / crop_type / split / card_name / src.name
        materialize(src, dst, args.mode)
        rows_written += 1

    print(f"Wrote {rows_written} entries into {output_root}")
    print("Layout:")
    print(output_root / "hand")
    print(output_root / "next")


if __name__ == "__main__":
    main()
