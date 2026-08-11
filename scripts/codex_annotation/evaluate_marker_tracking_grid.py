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

from cr_bot.vision.deployment_markers import (
    deployment_candidate_tracks,
    detect_enemy_team_markers,
    group_candidate_tracks,
    marker_burst_candidate_frame,
    track_enemy_team_markers,
)


def _score(expected: list[int], predicted: list[int], tolerance: int) -> dict:
    used: set[int] = set()
    matches = []
    for frame in expected:
        choices = sorted(
            (abs(candidate - frame), index, candidate)
            for index, candidate in enumerate(predicted)
            if index not in used and abs(candidate - frame) <= tolerance
        )
        if not choices:
            continue
        delta, index, candidate = choices[0]
        used.add(index)
        matches.append((frame, candidate, delta))
    tp = len(matches)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(expected) if expected else 1.0
    return {
        "candidates": len(predicted),
        "tp": tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "missed": [frame for frame in expected if frame not in {m[0] for m in matches}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation-only grid search for marker track continuity."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--tolerance-frames", type=int, default=5)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    events = truth.get("events", truth)
    expected = [
        int(row["frame_index"])
        for row in events
        if (
            row["side"] == "enemy"
            and row["card"] not in {"log", "fireball"}
            and manifest["segment"]["start_frame"]
            <= row["frame_index"]
            < manifest["segment"]["end_frame_exclusive"]
        )
    ]
    observations = []
    for record in manifest["frames"]:
        image = cv2.imread(str(run_dir / record["path"]))
        if image is None:
            raise FileNotFoundError(record["path"])
        frame = image[manifest["label_margin_px"] :]
        frame_index = int(record["source_frame_index"])
        observations.append(
            (
                frame_index,
                detect_enemy_team_markers(frame, frame_index=frame_index),
            )
        )
    results = []
    for gap in (2, 4, 6, 8, 12, 16, 20):
        for distance in (70.0, 95.0, 130.0, 180.0):
            tracks = track_enemy_team_markers(
                observations,
                max_gap_frames=gap,
                max_distance_px=distance,
            )
            candidates = deployment_candidate_tracks(
                tracks,
                segment_start_frame=manifest["segment"]["start_frame"],
            )
            bursts = group_candidate_tracks(candidates)
            predicted = [
                marker_burst_candidate_frame(
                    burst,
                    segment_start_frame=manifest["segment"]["start_frame"],
                )
                for burst in bursts
            ]
            results.append(
                {
                    "max_gap_frames": gap,
                    "max_distance_px": distance,
                    **_score(
                        expected,
                        predicted,
                        args.tolerance_frames,
                    ),
                }
            )
    print(
        json.dumps(
            sorted(
                results,
                key=lambda row: (
                    -row["recall"],
                    -row["precision"],
                    row["candidates"],
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
