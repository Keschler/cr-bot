"""Render identity evidence for bodies near a deployment marker.

Marker detection is an event-time cue, not necessarily a bounding box for the
deployed body.  In particular, a tiny Ice Spirit can appear beside a marker
that is underneath an older Hog Rider.  This renderer keeps the blind target
row but attaches a compact sheet of nearby tracked candidates so a worker can
follow the body that actually appears at the onset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import (  # noqa: E402
    REVIEW_MAX_PIXELS,
    add_frame_identity,
    atomic_write_json,
)
from cr_bot.annotation_stages import record_review  # noqa: E402
from render_enemy_identity_targets import (  # noqa: E402
    _contact_sheet,
    _crop_with_origin,
    _nearest_bbox,
    _track_observations,
)


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, width, height = bbox
    return left + width / 2.0, top + height / 2.0


def _nearby_tracks(
    *,
    tracks: dict[int, dict[int, tuple[int, int, int, int]]],
    anchor: tuple[float, float],
    onset: int,
    temporal_radius: int,
    spatial_radius: float,
    limit: int,
) -> list[tuple[int, dict[int, tuple[int, int, int, int]], float, int]]:
    candidates: list[tuple[int, dict[int, tuple[int, int, int, int]], float, int]] = []
    for track_id, observations in tracks.items():
        nearby = [
            (abs(frame - onset), frame, bbox)
            for frame, bbox in observations.items()
            if abs(frame - onset) <= temporal_radius
        ]
        if not nearby:
            continue
        onset_distance, _, bbox = min(nearby)
        cx, cy = _center(bbox)
        distance = math.hypot(cx - anchor[0], cy - anchor[1])
        if distance > spatial_radius:
            continue
        # Prefer tracks that begin close to the deployment and persist; keep
        # the distance as a deterministic tie-breaker.  We deliberately keep
        # both the marker track and a nearby body track when present.
        span = max(observations) - min(observations) + 1
        candidates.append((track_id, observations, distance, onset_distance - min(span, 20) / 100.0))
    candidates.sort(key=lambda row: (row[3], row[2], row[0]))
    return candidates[: max(1, limit)]


def _tile(
    *,
    content: np.ndarray,
    frame_index: int,
    fps: float,
    track_id: int,
    observations: dict[int, tuple[int, int, int, int]],
    width: int,
    height: int,
    tile_width: int,
) -> np.ndarray:
    bbox = _nearest_bbox(observations, frame_index, max_distance=10)
    if bbox is None:
        bbox = observations[min(observations)]
    left, top, box_width, box_height = bbox
    view, origin_x, origin_y = _crop_with_origin(
        content,
        center_x=round(left + box_width / 2),
        center_y=round(top + box_height / 2),
        width=width,
        height=height,
    )
    local_left = left - origin_x
    local_top = top - origin_y
    cv2.rectangle(
        view,
        (local_left - 8, local_top - 8),
        (local_left + box_width + 8, local_top + box_height + 8),
        (255, 255, 0),
        4,
    )
    label = f"CANDIDATE TRACK T{track_id}"
    cv2.putText(
        view,
        label,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        label,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    view = add_frame_identity(
        view,
        source_frame_index=frame_index,
        fps=fps,
    )
    scale = tile_width / view.shape[1]
    return cv2.resize(
        view,
        (tile_width, max(1, round(view.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render nearby tracked-body identity candidate sheets."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=420)
    parser.add_argument("--local-width", type=int, default=360)
    parser.add_argument("--local-height", type=int, default=520)
    parser.add_argument("--temporal-radius", type=int, default=10)
    parser.add_argument("--spatial-radius", type=float, default=260.0)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--sample-offsets",
        default="5,7,9,11,13,15,17,19,21",
        help="Comma-separated frame offsets from onset for each candidate panel.",
    )
    args = parser.parse_args()
    if args.tile_width <= 0 or args.local_width <= 0 or args.local_height <= 0:
        parser.error("tile and local dimensions must be positive")
    if args.temporal_radius < 0 or args.spatial_radius <= 0 or args.max_candidates <= 0:
        parser.error("candidate bounds must be positive")
    try:
        sample_offsets = [
            int(value.strip())
            for value in args.sample_offsets.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("--sample-offsets must contain comma-separated integers")
    if not sample_offsets:
        parser.error("--sample-offsets must not be empty")

    run_dir = args.run_dir.resolve()
    source = json.loads(args.targets_file.resolve().read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    marker_document = json.loads(
        (run_dir / "enemy_marker_candidates.json").read_text(encoding="utf-8")
    )
    bursts, tracks = _track_observations(marker_document)
    frames = {int(row["source_frame_index"]): row for row in manifest["frames"]}
    segment_start = int(manifest["segment"]["start_frame"])
    segment_end = int(manifest["segment"]["end_frame_exclusive"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in source["targets"]:
        if target.get("kind") != "unit_or_building":
            continue
        onset = int(target["event_frame_index"])
        burst = bursts.get(str(target.get("track_id")))
        # Some target files retain the semantic target id (for example a
        # simultaneous deployment) rather than the marker-burst key.  The
        # verification artifact is the auditable source of that association;
        # recover the canonical burst key before failing.
        if burst is None:
            for artifact in target.get("verification_artifacts", []):
                match = re.search(r"enemy-marker-burst-(\d{6})", str(artifact))
                if match:
                    burst = bursts.get(f"enemy-marker-burst:{match.group(1)}")
                    if burst is not None:
                        break
        if burst is None:
            raise ValueError(f"{target['onset_id']}: missing marker burst")
        marker_boxes = []
        for track_id in burst["track_ids"]:
            bbox = _nearest_bbox(tracks.get(int(track_id), {}), onset, max_distance=10)
            if bbox is not None:
                marker_boxes.append(bbox)
        if not marker_boxes:
            raise ValueError(f"{target['onset_id']}: marker has no onset bbox")
        marker_x = sum(_center(b)[0] for b in marker_boxes) / len(marker_boxes)
        marker_y = sum(_center(b)[1] for b in marker_boxes) / len(marker_boxes)
        candidates = _nearby_tracks(
            tracks=tracks,
            anchor=(marker_x, marker_y),
            onset=onset,
            temporal_radius=args.temporal_radius,
            spatial_radius=args.spatial_radius,
            limit=args.max_candidates,
        )
        if not candidates:
            raise ValueError(f"{target['onset_id']}: no nearby candidate tracks")

        sampled = sorted(
            {
                max(segment_start, min(segment_end - 1, onset + offset))
                for offset in sample_offsets
            }
        )
        tiles: list[np.ndarray] = []
        for track_id, observations, _, _ in candidates:
            for frame_index in sampled:
                image = cv2.imread(str(run_dir / frames[frame_index]["path"]))
                if image is None:
                    raise FileNotFoundError(frames[frame_index]["path"])
                content = image[manifest["label_margin_px"] :].copy()
                tiles.append(
                    _tile(
                        content=content,
                        frame_index=frame_index,
                        fps=float(manifest["fps"]),
                        track_id=track_id,
                        observations=observations,
                        width=args.local_width,
                        height=args.local_height,
                        tile_width=args.tile_width,
                    )
                )
        sheet = _contact_sheet(tiles, columns=3)
        if sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS:
            raise ValueError(f"{target['onset_id']}: neighbor sheet is too large")
        name = target["onset_id"].replace(":", "-")
        output = output_dir / f"identity-v3-neighbors-{name}.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(output)
        record_review(
            run_dir=run_dir,
            output_path=output,
            purpose="identity",
            start_frame=sampled[0],
            end_frame=sampled[-1] + 1,
            candidate_id=None,
            event_id=target["onset_id"],
        )
        target["identity_frame_index"] = sampled[0]
        target["identity_artifacts"] = [str(output.relative_to(run_dir))]
        target["identity_neighbor_track_ids"] = [row[0] for row in candidates]
        target["identity_render_options"] = {
            "mode": "neighbor_candidates",
            "temporal_radius": args.temporal_radius,
            "spatial_radius": args.spatial_radius,
            "max_candidates": args.max_candidates,
        }

    args.output_targets.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_targets.resolve(), source)
    print(
        json.dumps(
            {
                "output_targets": str(args.output_targets.resolve()),
                "targets": len(source["targets"]),
            }
        )
    )


if __name__ == "__main__":
    main()
