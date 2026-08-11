from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_pipeline import validate_enemy_existence_decisions


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare side-only checks for existence-confirmed unit events."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--existence-output", type=Path, action="append")
    parser.add_argument("--chunk-frames", type=int, default=400)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    outputs = args.existence_output or sorted(
        (run_dir / "worker_outputs").glob(
            "identity-overlap-??????-??????.json"
        )
    )
    if not outputs:
        raise ValueError("no enemy existence outputs")
    onsets = {
        row["onset_id"]: row
        for row in _read(run_dir / "enemy_onsets.json")["onsets"]
    }
    package_dir = run_dir / "work_packages"
    accepted = []
    for path in outputs:
        document = _read(path.resolve())
        if document.get("stage") != "enemy_overlap_adjudication_chunk":
            raise ValueError(f"{path}: wrong existence stage")
        start, end = document["target_range"]
        existence_package = _read(
            package_dir
            / f"identity-overlap-{start:06d}-{end:06d}.json"
        )
        if document.get("run_id") != existence_package.get("run_id"):
            raise ValueError(f"{path}: run_id mismatch")
        if document["target_range"] != existence_package["target_range"]:
            raise ValueError(f"{path}: target_range mismatch")
        validate_enemy_existence_decisions(document, existence_package)
        for decision in document["decisions"]:
            if decision.get("overlap_event_exists") is not True:
                continue
            onset = onsets[decision["onset_id"]]
            corrected_frame = decision.get("event_frame_index")
            if not isinstance(corrected_frame, int):
                raise ValueError(
                    f"{decision['onset_id']}: accepted event lacks corrected frame"
                )
            accepted.append(
                {
                    "onset_id": onset["onset_id"],
                    "approximate_frame_index": onset["event_frame_index"],
                    "event_frame_index": corrected_frame,
                    "full_arena_artifact": onset[
                        "verification_artifacts"
                    ][0],
                    "existence_evidence": decision["evidence"],
                    "existence_reason": decision.get("reason", ""),
                    "source_onset_ids": [onset["onset_id"]],
                }
            )
    manifest = _read(run_dir / "manifest.json")
    chunk_ranges = [
        [
            start,
            min(
                manifest["segment"]["end_frame_exclusive"],
                start + args.chunk_frames,
            ),
        ]
        for start in range(
            manifest["segment"]["start_frame"],
            manifest["segment"]["end_frame_exclusive"],
            args.chunk_frames,
        )
    ]
    summaries = []
    for start, end in chunk_ranges:
        candidates = [
            row
            for row in accepted
            if start <= row["event_frame_index"] < end
        ]
        source = _read(
            package_dir / f"identity-{start:06d}-{end:06d}.json"
        )
        output = (
            package_dir / f"identity-side-{start:06d}-{end:06d}.json"
        )
        output.write_text(
            json.dumps(
                {
                    "run_id": source["run_id"],
                    "fps": source["fps"],
                    "target_range": [start, end],
                    "decision_schema_version": 2,
                    "candidates": candidates,
                    "own_release_frames": source["own_release_frames"],
                    "rejected_own_drags": source["rejected_own_drags"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(
            {"range": [start, end], "candidates": len(candidates)}
        )
    print(json.dumps({"packages": summaries}))


if __name__ == "__main__":
    main()
