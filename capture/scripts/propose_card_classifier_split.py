from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED_CSV = ROOT / "data/card_classifier/metadata/labels_normalized.csv"
DEFAULT_OUTPUT_CSV = ROOT / "data/card_classifier/metadata/split.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a chunk-based train/val/test split for card-classifier crops.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_NORMALIZED_CSV,
        help="Normalized crop-level CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to write the split.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of frame indices per temporal chunk within a video.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target validation ratio by chunk count within each video.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Target test ratio by chunk count within each video.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Ignore rows below this score when estimating class coverage.",
    )
    parser.add_argument(
        "--ignore-blank",
        action="store_true",
        help="Ignore blank card labels for coverage checks.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def frame_index_from_path(frame_path: str) -> int:
    stem = Path(frame_path).stem
    return int(stem.split("_")[-1])


def normalized_label(row: dict[str, str]) -> str | None:
    card_name = row["card_name"].strip()
    if not card_name:
        return None
    return card_name


def build_chunk_rows(
    rows: list[dict[str, str]],
    *,
    chunk_size: int,
) -> tuple[dict[tuple[str, int], list[dict[str, str]]], dict[str, list[int]]]:
    chunk_rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    video_chunks: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        frame_index = frame_index_from_path(row["frame_path"])
        chunk_id = (frame_index - 1) // chunk_size
        key = (row["video_id"], chunk_id)
        chunk_rows[key].append(row)
        video_chunks[row["video_id"]].add(chunk_id)

    ordered_video_chunks = {
        video_id: sorted(chunk_ids)
        for video_id, chunk_ids in video_chunks.items()
    }
    return chunk_rows, ordered_video_chunks


def choose_chunk_split(chunk_ids: list[int], val_ratio: float, test_ratio: float) -> dict[int, str]:
    total = len(chunk_ids)
    test_n = max(1, round(total * test_ratio)) if total >= 3 else 0
    val_n = max(1, round(total * val_ratio)) if total >= 4 else 0

    if test_n + val_n >= total:
        if total >= 3:
            test_n = 1
            val_n = 1 if total >= 4 else 0
        else:
            test_n = 0
            val_n = 0

    split_by_chunk = {chunk_id: "train" for chunk_id in chunk_ids}
    if test_n:
        for chunk_id in chunk_ids[-test_n:]:
            split_by_chunk[chunk_id] = "test"
    if val_n:
        start = max(0, total - test_n - val_n)
        for chunk_id in chunk_ids[start:start + val_n]:
            split_by_chunk[chunk_id] = "val"
    return split_by_chunk


def assign_initial_splits(
    ordered_video_chunks: dict[str, list[int]],
    *,
    val_ratio: float,
    test_ratio: float,
) -> dict[tuple[str, int], str]:
    assignments: dict[tuple[str, int], str] = {}
    for video_id, chunk_ids in ordered_video_chunks.items():
        per_video = choose_chunk_split(chunk_ids, val_ratio, test_ratio)
        for chunk_id, split in per_video.items():
            assignments[(video_id, chunk_id)] = split
    return assignments


def coverage_key(row: dict[str, str], label: str | None) -> str:
    return f"{row['crop_type']}::{label or '__blank__'}"


def classes_by_split(
    chunk_rows: dict[tuple[str, int], list[dict[str, str]]],
    assignments: dict[tuple[str, int], str],
    *,
    min_score: int,
    ignore_blank: bool,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for key, rows in chunk_rows.items():
        split = assignments[key]
        for row in rows:
            score = int(row["score"])
            if score < min_score:
                continue
            label = normalized_label(row)
            if label is None and ignore_blank:
                continue
            result[split].add(coverage_key(row, label))
    return result


def ensure_train_coverage(
    chunk_rows: dict[tuple[str, int], list[dict[str, str]]],
    assignments: dict[tuple[str, int], str],
    *,
    min_score: int,
    ignore_blank: bool,
) -> tuple[dict[tuple[str, int], str], set[str], set[str]]:
    updated = dict(assignments)
    while True:
        split_classes = classes_by_split(
            chunk_rows,
            updated,
            min_score=min_score,
            ignore_blank=ignore_blank,
        )
        train_classes = split_classes.get("train", set())
        val_missing = split_classes.get("val", set()) - train_classes
        test_missing = split_classes.get("test", set()) - train_classes
        missing = val_missing | test_missing
        if not missing:
            return updated, val_missing, test_missing

        moved_any = False
        for key, split in list(updated.items()):
            if split == "train":
                continue
            chunk_labels = set()
            for row in chunk_rows[key]:
                score = int(row["score"])
                if score < min_score:
                    continue
                label = normalized_label(row)
                if label is None and ignore_blank:
                    continue
                chunk_labels.add(coverage_key(row, label))
            if chunk_labels & missing:
                updated[key] = "train"
                moved_any = True
        if not moved_any:
            return updated, val_missing, test_missing


def chunk_coverage_labels(
    rows: list[dict[str, str]],
    *,
    min_score: int,
    ignore_blank: bool,
) -> set[str]:
    labels = set()
    for row in rows:
        score = int(row["score"])
        if score < min_score:
            continue
        label = normalized_label(row)
        if label is None and ignore_blank:
            continue
        labels.add(coverage_key(row, label))
    return labels


def ensure_val_coverage(
    chunk_rows: dict[tuple[str, int], list[dict[str, str]]],
    assignments: dict[tuple[str, int], str],
    *,
    min_score: int,
    ignore_blank: bool,
) -> tuple[dict[tuple[str, int], str], set[str]]:
    updated = dict(assignments)

    while True:
        split_classes = classes_by_split(
            chunk_rows,
            updated,
            min_score=min_score,
            ignore_blank=ignore_blank,
        )
        train_classes = split_classes.get("train", set())
        val_classes = split_classes.get("val", set())
        missing = train_classes - val_classes
        if not missing:
            return updated, set()

        train_chunk_labels: dict[tuple[str, int], set[str]] = {}
        train_label_chunk_counts: Counter[str] = Counter()
        for key, split in updated.items():
            if split != "train":
                continue
            labels = chunk_coverage_labels(
                chunk_rows[key],
                min_score=min_score,
                ignore_blank=ignore_blank,
            )
            train_chunk_labels[key] = labels
            for label in labels:
                train_label_chunk_counts[label] += 1

        candidate_key = None
        candidate_score = (-1, -1)
        for key, labels in train_chunk_labels.items():
            covered_missing = labels & missing
            if not covered_missing:
                continue
            if any(train_label_chunk_counts[label] <= 1 for label in labels):
                continue
            score = (len(covered_missing), -len(labels))
            if score > candidate_score:
                candidate_score = score
                candidate_key = key

        if candidate_key is None:
            return updated, missing

        updated[candidate_key] = "val"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.labels_csv.resolve())
    if not rows:
        raise ValueError(f"No rows found in {args.labels_csv}")

    chunk_rows, ordered_video_chunks = build_chunk_rows(rows, chunk_size=args.chunk_size)
    assignments = assign_initial_splits(
        ordered_video_chunks,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    assignments, val_missing, test_missing = ensure_train_coverage(
        chunk_rows,
        assignments,
        min_score=args.min_score,
        ignore_blank=args.ignore_blank,
    )
    assignments, train_missing_from_val = ensure_val_coverage(
        chunk_rows,
        assignments,
        min_score=args.min_score,
        ignore_blank=args.ignore_blank,
    )

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    split_counts = Counter(assignments.values())
    row_counts = Counter()
    for key, split in assignments.items():
        row_counts[split] += len(chunk_rows[key])

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "frame_path", "slot_name", "crop_path", "crop_type", "card_name", "score", "split", "chunk_id"],
        )
        writer.writeheader()
        for key in sorted(chunk_rows, key=lambda item: (item[0], item[1])):
            video_id, chunk_id = key
            split = assignments[key]
            for row in chunk_rows[key]:
                writer.writerow(
                    {
                        "video_id": row["video_id"],
                        "frame_path": row["frame_path"],
                        "slot_name": row["slot_name"],
                        "crop_path": row["crop_path"],
                        "crop_type": row["crop_type"],
                        "card_name": row["card_name"],
                        "score": row["score"],
                        "split": split,
                        "chunk_id": chunk_id,
                    }
                )

    split_classes = classes_by_split(
        chunk_rows,
        assignments,
        min_score=args.min_score,
        ignore_blank=args.ignore_blank,
    )
    print(f"Wrote chunk-based split to {output_csv}")
    print(f"chunk_size={args.chunk_size}")
    print(f"chunk counts: {dict(split_counts)}")
    print(f"row counts: {dict(row_counts)}")
    print(f"train classes: {len(split_classes.get('train', set()))}")
    print(f"val classes: {len(split_classes.get('val', set()))}")
    print(f"test classes: {len(split_classes.get('test', set()))}")
    print(f"val missing from train: {sorted(val_missing)}")
    print(f"test missing from train: {sorted(test_missing)}")
    print(f"train missing from val: {sorted(train_missing_from_val)}")


if __name__ == "__main__":
    main()
