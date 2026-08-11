from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.vision.deployment_markers import (
    deployment_candidate_tracks,
    detect_enemy_team_markers,
    group_candidate_tracks,
    marker_burst_candidate_frame,
    track_enemy_team_markers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate card-free enemy unit candidates from red deployment UI "
            "markers. This does not use a detector, tracker cache, or labels."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    observations = []
    for record in manifest["frames"]:
        labeled = cv2.imread(str(run_dir / record["path"]))
        if labeled is None:
            raise FileNotFoundError(run_dir / record["path"])
        frame = labeled[manifest["label_margin_px"] :]
        frame_index = int(record["source_frame_index"])
        observations.append(
            (
                frame_index,
                detect_enemy_team_markers(frame, frame_index=frame_index),
            )
        )
    tracks = track_enemy_team_markers(observations)
    candidates = deployment_candidate_tracks(
        tracks,
        segment_start_frame=manifest["segment"]["start_frame"],
    )
    bursts = group_candidate_tracks(candidates)
    scan_windows = [
        row
        for row in manifest["candidate_discovery"]["enemy_scan_windows"]
        if row["candidate_id"].startswith("enemy-scan:")
    ]

    def supporting_scan(frame_index: int) -> str:
        matches = [
            row
            for row in scan_windows
            if row["inspection_start_frame"]
            <= frame_index
            < row["inspection_end_frame_exclusive"]
        ]
        if not matches:
            raise ValueError(f"no enemy scan window covers frame {frame_index}")
        return min(
            matches,
            key=lambda row: (
                row["candidate_id"].split(":")[-1] != "p1",
                row["inspection_start_frame"],
            ),
        )["candidate_id"]
    output = {
        "run_id": manifest["run_id"],
        "stage": "enemy_marker_candidates",
        "method": "red-ui-marker-tracks-v2-gap6-distance70",
        "candidates": [
            {
                "candidate_id": f"enemy-marker:{track.track_id:06d}",
                "track_id": track.track_id,
                "event_frame_index": track.first.frame_index,
                "first_bbox": list(track.first.bbox),
                "first_center": [
                    round(track.first.center[0], 2),
                    round(track.first.center[1], 2),
                ],
                "observation_frames": [
                    row.frame_index for row in track.observations
                ],
                "observation_bboxes": [
                    list(row.bbox) for row in track.observations
                ],
            }
            for track in candidates
        ],
        "bursts": [
            {
                "burst_id": f"enemy-marker-burst:{burst.burst_id:06d}",
                "start_frame": burst.start_frame,
                "end_frame_exclusive": burst.end_frame_exclusive,
                "candidate_frame_index": marker_burst_candidate_frame(
                    burst,
                    segment_start_frame=manifest["segment"]["start_frame"],
                ),
                "supporting_candidate_id": supporting_scan(burst.start_frame),
                "track_ids": [track.track_id for track in burst.tracks],
                "track_start_frames": [
                    track.first.frame_index for track in burst.tracks
                ],
                "first_bboxes": [
                    list(track.first.bbox) for track in burst.tracks
                ],
            }
            for burst in bursts
        ],
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "enemy_marker_candidates.json"
    )
    atomic_write_json(output_path, output)
    print(
        json.dumps(
            {
                "tracks": len(tracks),
                "candidates": len(candidates),
                "bursts": len(bursts),
                "output": str(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
