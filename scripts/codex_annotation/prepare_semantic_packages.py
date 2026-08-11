from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _artifact(candidate_id: str) -> str:
    return f"reviews/verify-{candidate_id.replace(':', '-')}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create bounded, card-free work packages for semantic workers."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=200)
    parser.add_argument(
        "--skip-own",
        action="store_true",
        help="Prepare only enemy scan/spell packages.",
    )
    parser.add_argument(
        "--context-halo-frames",
        type=int,
        default=20,
        help="Read-only context supplied on both sides of each owned event range.",
    )
    args = parser.parse_args()
    if args.chunk_frames < 50:
        parser.error("--chunk-frames must be at least 50")
    if args.context_halo_frames < 0:
        parser.error("--context-halo-frames must be non-negative")
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    discovery_by_id = {}
    if not args.skip_own:
        discovery_index = _read(run_dir / "own_discovery_index.json")
        if (
            discovery_index.get("run_id") != manifest["run_id"]
            or discovery_index.get("format_version") != 4
        ):
            raise ValueError("own discovery evidence is stale or incompatible")
        discovery_by_id = {
            row["candidate_id"]: row
            for row in discovery_index["candidates"]
        }
    segment = manifest["segment"]
    discovery = manifest["candidate_discovery"]
    output_dir = run_dir / "work_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for start in range(
        segment["start_frame"],
        segment["end_frame_exclusive"],
        args.chunk_frames,
    ):
        end = min(segment["end_frame_exclusive"], start + args.chunk_frames)
        context_start = max(
            segment["start_frame"], start - args.context_halo_frames
        )
        context_end = min(
            segment["end_frame_exclusive"], end + args.context_halo_frames
        )
        own = []
        for candidate in (
            [] if args.skip_own else discovery["own_candidates"]
        ):
            approximate = candidate["approximate_frame_index"]
            if not context_start <= approximate < context_end:
                continue
            discovery_row = discovery_by_id[candidate["candidate_id"]]
            own.append(
                {
                    **candidate,
                    "package_role": (
                        "owned_peak"
                        if start <= approximate < end
                        else "context_peak"
                    ),
                    "discovery_artifact": discovery_row["artifact"],
                    "discovery_frame_indices": discovery_row[
                        "sampled_frame_indices"
                    ],
                }
            )
        primary_enemy = [
            {
                **candidate,
                "verification_artifact": _artifact(candidate["candidate_id"]),
            }
            for candidate in discovery["enemy_scan_windows"]
            if candidate["candidate_id"].startswith("enemy-scan:")
            and candidate["inspection_end_frame_exclusive"] > context_start
            and candidate["inspection_start_frame"] < context_end
        ]
        boundary_enemy = [
            {
                **candidate,
                "verification_artifact": _artifact(candidate["candidate_id"]),
            }
            for candidate in discovery["enemy_scan_windows"]
            if candidate["candidate_id"].startswith("enemy-boundary:")
            and (
                abs(candidate["inspection_start_frame"] - start) <= 12
                or abs(candidate["inspection_end_frame_exclusive"] - end) <= 12
            )
        ]
        stem = f"{start:06d}-{end:06d}"
        common = {
            "run_id": manifest["run_id"],
            "fps": manifest["fps"],
            "segment": segment,
            "target_range": [start, end],
            "owned_event_range": [start, end],
            "context_range": [context_start, context_end],
        }
        own_path = output_dir / f"own-{stem}.json"
        enemy_path = output_dir / f"enemy-{stem}.json"
        enemy_spell_path = output_dir / f"enemy-spells-{stem}.json"
        if not args.skip_own:
            own_path.write_text(
                json.dumps({**common, "candidates": own}, indent=2) + "\n",
                encoding="utf-8",
            )
        enemy_path.write_text(
            json.dumps(
                {
                    **common,
                    "primary_windows": primary_enemy,
                    "boundary_windows": boundary_enemy,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        enemy_spell_path.write_text(
            json.dumps(
                {
                    **common,
                    "primary_windows": primary_enemy,
                    "boundary_windows": boundary_enemy,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(
            {
                "range": [start, end],
                "own": len(own),
                "enemy_primary": len(primary_enemy),
                "enemy_boundary": len(boundary_enemy),
            }
        )
    print(json.dumps({"output_dir": str(output_dir), "packages": summaries}))


if __name__ == "__main__":
    main()
