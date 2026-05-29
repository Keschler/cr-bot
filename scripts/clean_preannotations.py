#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    x1: float
    y1: float
    x2: float
    y2: float

    def contains_center(self, cx: float, cy: float, pad: float = 0.0) -> bool:
        return (
            self.x1 - pad <= cx <= self.x2 + pad
            and self.y1 - pad <= cy <= self.y2 + pad
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove repeated false positives from Labelme-style pre-annotation JSONs."
    )
    parser.add_argument("folder", type=Path, help="Folder containing exported .json files")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Only remove shapes with this exact label. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--keep-base-label",
        action="append",
        default=[],
        help="Keep only these base labels (for example 'giant' keeps 'giant0' and 'giant1').",
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional screen region to scrub. If omitted, matching labels are removed everywhere.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Extra padding around the region in pixels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matches without rewriting files.",
    )
    return parser.parse_args()


def shape_center(shape: dict) -> tuple[float, float] | None:
    points = shape.get("points") or []
    if len(points) != 2:
        return None
    (x1, y1), (x2, y2) = points
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def base_label(label: str) -> str:
    return re.sub(r"[0-9]+$", "", label)


def main() -> None:
    args = parse_args()
    region = Region(*args.region) if args.region else None
    labels = set(args.label)
    keep_base_labels = set(args.keep_base_label)
    json_files = sorted(args.folder.glob("*.json"))

    removed_shapes = 0
    touched_files = 0

    for path in json_files:
        data = json.loads(path.read_text())
        shapes = data.get("shapes", [])
        kept = []
        file_removed = 0

        for shape in shapes:
            center = shape_center(shape)
            exact_label = shape.get("label", "")
            base = base_label(exact_label)
            keep_mode = bool(keep_base_labels)
            label_ok = not labels or exact_label in labels
            region_ok = region is None or (
                center is not None and region.contains_center(*center, pad=args.pad)
            )
            keep_ok = (not keep_mode) or (base in keep_base_labels)
            should_remove = (keep_mode and not keep_ok and region_ok) or (
                not keep_mode and label_ok and region_ok
            )
            if should_remove:
                file_removed += 1
                removed_shapes += 1
                continue
            kept.append(shape)

        if file_removed:
            touched_files += 1
            if not args.dry_run:
                data["shapes"] = kept
                path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n")

    mode = "Would remove" if args.dry_run else "Removed"
    print(f"{mode} {removed_shapes} shapes from {touched_files} files in {args.folder}")


if __name__ == "__main__":
    main()
