from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2

from cr_bot.temporal_spells.config import TemporalSpellConfig
from cr_bot.temporal_spells.dataset import (
    clip_end_times,
    extract_causal_clip,
    read_manifest,
    split_rows_by_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract causal temporal-spell clips.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    args = parser.parse_args()
    config = TemporalSpellConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for event_index, row in enumerate(read_manifest(args.manifest)):
        event_time = float(row.get("event_time_s", row["event_frame"] / row.get("fps", 30.0)))
        for offset_index, clip_end in enumerate(clip_end_times(event_time, args.offsets)):
            clip_id = f"{event_index:06d}_{offset_index:02d}"
            clip_dir = args.output / "clips" / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for frame_index, frame in enumerate(extract_causal_clip(row["video"], clip_end, config)):
                path = clip_dir / f"{frame_index:02d}.png"
                cv2.imwrite(str(path), frame)
                paths.append(str(path))
            prepared.append({**row, "clip_end_s": clip_end, "frame_paths": paths})
    splits = split_rows_by_session(prepared)
    for split, rows in splits.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        counts = Counter((row["card"], row["ownership"]) for row in rows)
        summary = ", ".join(
            f"{card}/{ownership}={count}"
            for (card, ownership), count in sorted(counts.items())
        )
        print(f"{split}: clips={len(rows)} {summary}")


if __name__ == "__main__":
    main()
