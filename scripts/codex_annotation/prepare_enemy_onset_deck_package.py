from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json


def build_onset_deck_package(
    source: dict[str, object], target_range: list[int]
) -> dict[str, object]:
    targets = source.get("targets")
    if not isinstance(targets, list):
        raise ValueError("identity target document must contain targets")
    attached_images: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("identity target rows must be objects")
        verification = target.get("verification_artifacts")
        identity = target.get("identity_artifacts")
        if not isinstance(verification, list) or not isinstance(identity, list):
            raise ValueError("target evidence lists are required")
        # The first verification artifact is the full-arena temporal sequence.
        # Attach it for side/onset authority, plus every delayed identity view.
        # Focus sheets are intentionally omitted to keep a full-run package
        # below common multimodal attachment limits.
        selected = verification if target.get("kind") == "spell" else verification[:1]
        for artifact in [*selected, *identity]:
            if not isinstance(artifact, str):
                raise ValueError("target evidence paths must be strings")
            if artifact not in seen:
                seen.add(artifact)
                attached_images.append(artifact)
    return {
        "run_id": source.get("run_id"),
        "stage": "enemy_identity_targets",
        "target_range": target_range,
        "targets": targets,
        "attached_images": attached_images,
        "deck_constraint": {"maximum_base_card_slots": 8},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a full-run onset-first enemy deck package."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-frame",
        type=int,
        action="append",
        default=[],
        help="Optionally retain only these target event frames.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source = json.loads(args.targets_file.read_text(encoding="utf-8"))
    if args.event_frame:
        selected_frames = set(args.event_frame)
        source = dict(source)
        source["targets"] = [
            row
            for row in source["targets"]
            if int(row["event_frame_index"]) in selected_frames
        ]
        found_frames = {
            int(row["event_frame_index"]) for row in source["targets"]
        }
        if found_frames != selected_frames:
            missing = sorted(selected_frames - found_frames)
            raise ValueError(f"no identity target for event frames: {missing}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    segment = manifest["segment"]
    package = build_onset_deck_package(
        source,
        [int(segment["start_frame"]), int(segment["end_frame_exclusive"])],
    )
    atomic_write_json(args.output, package)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "targets": len(package["targets"]),
                "attached_images": len(package["attached_images"]),
            }
        )
    )


if __name__ == "__main__":
    main()
