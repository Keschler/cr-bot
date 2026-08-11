from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split exact enemy-spell reviews into independent gate packages."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=200)
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")

    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    source = _read(run_dir / "enemy_spell_confirmation_candidates.json")
    if source.get("run_id") != manifest.get("run_id"):
        raise ValueError("spell confirmation candidates do not match manifest")
    reviews = source.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("spell confirmation reviews must be a list")

    segment = manifest["segment"]
    start = int(segment["start_frame"])
    stop = int(segment["end_frame_exclusive"])
    output_dir = run_dir / "work_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for chunk_start in range(start, stop, args.chunk_frames):
        chunk_end = min(stop, chunk_start + args.chunk_frames)
        selected = [
            row
            for row in reviews
            if chunk_start <= int(row["proposal_frame_index"]) < chunk_end
        ]
        package = {
            "run_id": manifest["run_id"],
            "stage": "enemy_spell_confirmation_package",
            "target_range": [chunk_start, chunk_end],
            "segment": segment,
            "reviews": selected,
        }
        path = (
            output_dir
            / f"enemy-spell-confirmation-{chunk_start:06d}-{chunk_end:06d}.json"
        )
        atomic_write_json(path, package)
        summaries.append({"range": [chunk_start, chunk_end], "reviews": len(selected)})
    print(json.dumps({"packages": summaries, "total": len(reviews)}))


if __name__ == "__main__":
    main()
