from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare focused adjudication packages for unit candidates rejected "
            "by the full-arena existence/side pass."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--primary-output",
        type=Path,
        action="append",
        help="Primary enemy_identities_chunk output; repeat for each chunk.",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Review every unit candidate instead of only primary rejections.",
    )
    parser.add_argument("--chunk-frames", type=int, default=400)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    package_dir = run_dir / "work_packages"
    summaries = []
    all_onsets = {
        row["onset_id"]: row
        for row in _read(run_dir / "enemy_onsets.json")["onsets"]
    }
    if args.all_candidates and not args.primary_output:
        manifest = _read(run_dir / "manifest.json")
        primary_outputs: list[Path | dict] = [
            {
                "run_id": manifest["run_id"],
                "stage": "enemy_identities_chunk",
                "target_range": [
                    start,
                    min(
                        manifest["segment"]["end_frame_exclusive"],
                        start + args.chunk_frames,
                    ),
                ],
                "decisions": [],
            }
            for start in range(
                manifest["segment"]["start_frame"],
                manifest["segment"]["end_frame_exclusive"],
                args.chunk_frames,
            )
        ]
    else:
        primary_outputs = args.primary_output or sorted(
            (run_dir / "worker_outputs").glob(
                "identity-??????-??????.json"
            )
        )
    if not primary_outputs:
        raise ValueError("no primary enemy identity outputs")
    for output_path in primary_outputs:
        primary = (
            output_path
            if isinstance(output_path, dict)
            else _read(output_path.resolve())
        )
        if primary.get("stage") != "enemy_identities_chunk":
            raise ValueError(f"{output_path}: wrong primary stage")
        start, end = primary["target_range"]
        source_path = package_dir / f"identity-{start:06d}-{end:06d}.json"
        source = _read(source_path)
        onsets = {row["onset_id"]: row for row in source["onsets"]}
        selected_ids = (
            {
                onset_id
                for onset_id, onset in all_onsets.items()
                if (
                    onset.get("kind") == "unit_or_building"
                    and start <= onset["event_frame_index"] < end
                )
            }
            if args.all_candidates
            else {
                decision["onset_id"]
                for decision in primary["decisions"]
                if decision.get("event_exists") is False
            }
        )
        candidates = []
        for onset_id in sorted(
            selected_ids,
            key=lambda value: (
                all_onsets[value]["event_frame_index"],
                value,
            ),
        ):
            onset = all_onsets.get(onset_id)
            if onset is None or onset.get("kind") != "unit_or_building":
                continue
            artifacts = onset.get("verification_artifacts", [])
            if len(artifacts) < 2:
                raise ValueError(
                    f"{onset['onset_id']}: focused marker artifact is missing"
                )
            candidates.append(
                {
                    "onset_id": onset["onset_id"],
                    "event_frame_index": onset["event_frame_index"],
                    "sampled_frame_indices": onset[
                        "sampled_frame_indices"
                    ],
                    "focus_artifact": artifacts[1],
                }
            )
        target = (
            package_dir / f"identity-overlap-{start:06d}-{end:06d}.json"
        )
        target.write_text(
            json.dumps(
                {
                    "run_id": source["run_id"],
                    "fps": source["fps"],
                    "target_range": [start, end],
                    "decision_schema_version": 2,
                    "candidates": candidates,
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
