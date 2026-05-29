from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET = ROOT / "vendor/external/Clash-Royale-Detection-Dataset"
DEFAULT_LOCAL_CLIP = ROOT / "data/part2/clip/1"
DEFAULT_SEED_ROOT = ROOT / "data/seed_dataset"
DEFAULT_CLIP_NAME = "capture_clip"
DEFAULT_ROUND = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a merged Clash Royale seed dataset for KataCR.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help="Root of the upstream Clash-Royale-Detection-Dataset clone.",
    )
    parser.add_argument(
        "--local-clip",
        type=Path,
        default=DEFAULT_LOCAL_CLIP,
        help="Local folder containing extracted JPG frames to annotate.",
    )
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=DEFAULT_SEED_ROOT,
        help="Destination root for the merged dataset.",
    )
    parser.add_argument(
        "--clip-name",
        default=DEFAULT_CLIP_NAME,
        help="Video name to use under images/part2 for the local clip.",
    )
    parser.add_argument(
        "--round-name",
        default=DEFAULT_ROUND,
        help="Round name to use under images/part2/<clip-name>/.",
    )
    return parser.parse_args()


def ensure_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source.resolve())


def build_seed_dataset(
    source_dataset: Path,
    local_clip: Path,
    seed_root: Path,
    clip_name: str,
    round_name: str,
) -> tuple[Path, int]:
    source_images = source_dataset / "images"
    source_part2 = source_images / "part2"
    if not source_dataset.exists():
        raise FileNotFoundError(f"Seed dataset root not found: {source_dataset}")
    if not source_part2.exists():
        raise FileNotFoundError(f"Seed dataset not found: {source_part2}")
    if not local_clip.exists():
        raise FileNotFoundError(f"Local clip frames not found: {local_clip}")

    seed_images = seed_root / "images"
    seed_part2 = seed_root / "images/part2"
    seed_images.mkdir(parents=True, exist_ok=True)
    seed_part2.mkdir(parents=True, exist_ok=True)
    (seed_root / "version_info").mkdir(parents=True, exist_ok=True)

    for source_entry in sorted(source_images.iterdir()):
        if source_entry.name == "part2":
            continue
        ensure_symlink(source_entry, seed_images / source_entry.name)

    for source_entry in sorted(source_part2.iterdir()):
        if source_entry.name == clip_name:
            continue
        ensure_symlink(source_entry, seed_part2 / source_entry.name)

    clip_target = seed_part2 / clip_name / round_name
    if clip_target.exists():
        existing_labels = [
            path for path in clip_target.glob("*.json")
            if path.name != "import_map.json"
        ] + list(clip_target.glob("*.txt"))
        if existing_labels:
            raise FileExistsError(
                f"{clip_target} already contains labels. Reuse it instead of overwriting.",
            )
    clip_target.mkdir(parents=True, exist_ok=True)

    frames = sorted(local_clip.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {local_clip}")

    import_map: list[dict[str, str]] = []
    for index, frame_path in enumerate(frames):
        target_stem = f"{index:05d}"
        target_path = clip_target / f"{target_stem}.jpg"
        if not target_path.exists():
            shutil.copy2(frame_path, target_path)
        import_map.append(
            {
                "source": str(frame_path.relative_to(ROOT)),
                "target": str(target_path.relative_to(seed_root)),
            }
        )

    map_path = clip_target / "import_map.json"
    map_path.write_text(json.dumps(import_map, indent=2) + "\n", encoding="utf-8")
    return clip_target, len(frames)


def main() -> None:
    args = parse_args()
    clip_target, frame_count = build_seed_dataset(
        source_dataset=args.source_dataset,
        local_clip=args.local_clip,
        seed_root=args.seed_root,
        clip_name=args.clip_name,
        round_name=args.round_name,
    )
    print(f"Prepared seed dataset at {args.seed_root}")
    print(f"Imported {frame_count} local frames into {clip_target}")
    print("Open this folder in Labelme to annotate the new clip:")
    print(clip_target)


if __name__ == "__main__":
    main()
