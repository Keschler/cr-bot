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


def main() -> None:
    parser = argparse.ArgumentParser(description="Split card-free identity targets.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="Input target document; defaults to RUN_DIR/enemy_identity_targets.json.",
    )
    parser.add_argument("--chunk-frames", type=int, default=400)
    parser.add_argument(
        "--output-prefix",
        default="cards",
        help="Package filename prefix; use a distinct value for benchmarks.",
    )
    parser.add_argument(
        "--one-target-per-package",
        action="store_true",
        help=(
            "Create an isolated multimodal worker package for each target. "
            "This prevents visual cross-target binding errors."
        ),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_path = (
        args.targets_file.resolve()
        if args.targets_file is not None
        else run_dir / "enemy_identity_targets.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir = run_dir / "work_packages"
    summaries = []
    start = manifest["segment"]["start_frame"]
    stop = manifest["segment"]["end_frame_exclusive"]
    if args.one_target_per_package:
        seen_ranges: set[tuple[int, int]] = set()
        for target in source["targets"]:
            event_frame = int(target["event_frame_index"])
            target_range = (event_frame, min(stop, event_frame + 1))
            if target_range in seen_ranges:
                raise ValueError(
                    "one-target packages require unique event frame indices"
                )
            seen_ranges.add(target_range)
            path = output_dir / (
                f"{args.output_prefix}-"
                f"{target_range[0]:06d}-{target_range[1]:06d}.json"
            )
            atomic_write_json(
                path,
                {
                    "run_id": source["run_id"],
                    "stage": "enemy_identity_targets",
                    "target_range": list(target_range),
                    "targets": [target],
                },
            )
            summaries.append(
                {"range": list(target_range), "targets": 1}
            )
        print(json.dumps({"packages": summaries}))
        return

    for chunk_start in range(start, stop, args.chunk_frames):
        chunk_end = min(stop, chunk_start + args.chunk_frames)
        targets = [
            row
            for row in source["targets"]
            if chunk_start <= row["event_frame_index"] < chunk_end
        ]
        path = output_dir / (
            f"{args.output_prefix}-{chunk_start:06d}-{chunk_end:06d}.json"
        )
        atomic_write_json(
            path,
            {
                "run_id": source["run_id"],
                "stage": "enemy_identity_targets",
                "target_range": [chunk_start, chunk_end],
                "targets": targets,
            },
        )
        summaries.append({"range": [chunk_start, chunk_end], "targets": len(targets)})
    print(json.dumps({"packages": summaries}))


if __name__ == "__main__":
    main()
