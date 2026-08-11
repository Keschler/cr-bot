from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from cr_bot.annotation_harness import (
    REVIEW_MAX_PIXELS,
    add_frame_identity,
    atomic_write_json,
)
from cr_bot.annotation_stages import record_review


def _contact_sheet(tiles: list[np.ndarray], *, columns: int) -> np.ndarray:
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * height : row * height + tile.shape[0],
            column * width : column * width + tile.shape[1],
        ] = tile
    return sheet


def _sample_frames(
    *,
    onset: int,
    offsets: tuple[int, ...],
    segment_start: int,
    segment_end: int,
) -> list[int]:
    return sorted(
        {
            max(segment_start, min(segment_end - 1, onset + offset))
            for offset in offsets
        }
    )


def _bounded_crop(
    image: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    width = min(width, image.shape[1])
    height = min(height, image.shape[0])
    left = max(0, min(image.shape[1] - width, center_x - width // 2))
    top = max(0, min(image.shape[0] - height, center_y - height // 2))
    return image[top : top + height, left : left + width]


def _track_observations(
    marker_document: dict[str, object],
) -> tuple[
    dict[str, dict[str, object]],
    dict[int, dict[int, tuple[int, int, int, int]]],
]:
    bursts = {
        str(row["burst_id"]): row
        for row in marker_document.get("bursts", [])
        if isinstance(row, dict)
    }
    tracks: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for row in marker_document.get("candidates", []):
        if not isinstance(row, dict):
            continue
        tracks[int(row["track_id"])] = {
            int(frame): tuple(int(value) for value in bbox)
            for frame, bbox in zip(
                row["observation_frames"],
                row["observation_bboxes"],
                strict=True,
            )
        }
    return bursts, tracks


def _nearest_bbox(
    observations: dict[int, tuple[int, int, int, int]],
    frame_index: int,
    *,
    max_distance: int = 3,
) -> tuple[int, int, int, int] | None:
    candidates = [
        (abs(observed_frame - frame_index), observed_frame, bbox)
        for observed_frame, bbox in observations.items()
        if abs(observed_frame - frame_index) <= max_distance
    ]
    return min(candidates)[2] if candidates else None


@dataclass(frozen=True)
class _TrackSelection:
    """Deterministic summary used by the v2 identity crop renderer.

    Marker tracking occasionally produces several tracks in the same burst:
    an old unit can be visible beside the newly spawned unit, and the burst
    can contain multiple small marker components. Keeping the scoring inputs
    in a serializable value object makes the selection auditable in the copied
    target document without changing the v1 renderer.
    """

    track_id: int
    score: float
    onset_distance: int
    start_frame: int
    end_frame: int
    observation_count: int
    span_frames: int
    displacement_px: float
    eligible: bool


def _track_selection_stats(
    *,
    track_id: int,
    observations: dict[int, tuple[int, int, int, int]],
    onset_frame: int,
    burst_start_frame: int,
    onset_tolerance: int,
) -> _TrackSelection:
    """Score one marker track for the selected-track v2 renderer.

    Eligibility is deliberately tied to both the target onset and the marker
    burst start. The former handles a verified onset that is a frame or two
    before the first marker; the latter prevents a persistent old track from
    being selected merely because it has a large span. The small asymmetric
    window also accommodates simultaneous-event bursts produced by the marker
    merger, where a target can begin several frames after the burst's first
    component.
    """

    if not observations:
        raise ValueError(f"track {track_id} has no observations")
    frames = sorted(observations)
    start_frame = frames[0]
    end_frame = frames[-1]
    observation_count = len(frames)
    span_frames = end_frame - start_frame + 1
    first = observations[start_frame]
    last = observations[end_frame]
    first_center = (
        first[0] + first[2] / 2.0,
        first[1] + first[3] / 2.0,
    )
    last_center = (
        last[0] + last[2] / 2.0,
        last[1] + last[3] / 2.0,
    )
    displacement_px = math.hypot(
        last_center[0] - first_center[0],
        last_center[1] - first_center[1],
    )
    onset_distance = abs(start_frame - onset_frame)
    window_start = min(onset_frame, burst_start_frame) - 1
    window_end = max(onset_frame, burst_start_frame) + onset_tolerance
    eligible = window_start <= start_frame <= window_end

    # The onset term has the largest effect, while persistence and motion
    # break ties between same-onset tracks. Saturating the latter terms keeps
    # a long-lived old actor from overwhelming a newly moving target.
    onset_score = 2.0 / (1.0 + onset_distance)
    persistence_score = min(1.0, observation_count / 8.0) + min(
        1.0, span_frames / 12.0
    )
    movement_score = min(1.0, displacement_px / 60.0)
    score = onset_score + 1.25 * persistence_score + 1.25 * movement_score
    if not eligible:
        score -= 100.0
    return _TrackSelection(
        track_id=int(track_id),
        score=float(score),
        onset_distance=int(onset_distance),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        observation_count=int(observation_count),
        span_frames=int(span_frames),
        displacement_px=float(displacement_px),
        eligible=bool(eligible),
    )


def _select_dominant_track(
    *,
    burst: dict[str, object],
    tracks: dict[int, dict[int, tuple[int, int, int, int]]],
    onset_frame: int,
    onset_tolerance: int = 2,
) -> tuple[_TrackSelection, tuple[int, int, int, int]]:
    """Select one auditable marker track and its onset anchor bbox.

    The old renderer intentionally highlighted every closest-start box. This
    v2 path chooses one track, so a larger neighbouring Hog/Mega Knight cannot
    visually dominate a tiny target such as an Ice Spirit. If malformed data
    has no eligible track, the least-bad track is still returned and marked
    as ineligible in the metadata rather than silently dropping a target.
    """

    if onset_tolerance < 0:
        raise ValueError("onset_tolerance must be non-negative")
    paired_tracks = list(
        zip(
            burst["track_ids"],
            burst["track_start_frames"],
            burst["first_bboxes"],
            strict=True,
        )
    )
    selections = []
    for track_id, _, _ in paired_tracks:
        observations = tracks.get(int(track_id), {})
        if observations:
            selections.append(
                _track_selection_stats(
                    track_id=int(track_id),
                    observations=observations,
                    onset_frame=onset_frame,
                    burst_start_frame=int(burst["start_frame"]),
                    onset_tolerance=onset_tolerance,
                )
            )
    if not selections:
        raise ValueError(
            f"{burst.get('burst_id', '<burst>')}: no track observations"
        )
    selected = min(
        selections,
        key=lambda row: (
            -row.score,
            row.onset_distance,
            -row.observation_count,
            -row.span_frames,
            -row.displacement_px,
            row.track_id,
        ),
    )
    anchor = _nearest_bbox(
        tracks[selected.track_id],
        onset_frame,
        max_distance=max(3, onset_tolerance + 1),
    )
    if anchor is None:
        # The selected marker can disappear immediately (for example, an Ice
        # Spirit). Keep the first observed bbox as a stable local-crop anchor
        # instead of drifting to a nearby actor.
        first_frame = min(tracks[selected.track_id])
        anchor = tracks[selected.track_id][first_frame]
    return selected, anchor


def _crop_with_origin(
    image: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int]:
    """Return a bounded crop and its origin in the source image."""

    width = min(width, image.shape[1])
    height = min(height, image.shape[0])
    left = max(0, min(image.shape[1] - width, center_x - width // 2))
    top = max(0, min(image.shape[0] - height, center_y - height // 2))
    return image[top : top + height, left : left + width], left, top


def _draw_selected_track(
    view: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    origin_x: int,
    origin_y: int,
    track_id: int,
    mask_context: bool,
) -> np.ndarray:
    """Annotate one target bbox and optionally de-emphasize its context."""

    left, top, width, height = bbox
    local_left = left - origin_x
    local_top = top - origin_y
    # Marker boxes are small; keep enough surrounding unit body visible while
    # making an older, adjacent actor less salient in the optional masked view.
    if mask_context:
        mask = np.zeros(view.shape[:2], dtype=np.uint8)
        center_x = local_left + width // 2
        center_y = local_top + height // 2
        radius_x = max(52, width * 3)
        radius_y = max(76, height * 3)
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius_x, radius_y),
            0,
            0,
            360,
            255,
            -1,
        )
        softened = cv2.GaussianBlur(view, (0, 0), sigmaX=13)
        dimmed = cv2.addWeighted(softened, 0.35, np.zeros_like(view), 0.65, 0)
        view = np.where(mask[..., None] > 0, view, dimmed).astype(np.uint8)
    cv2.rectangle(
        view,
        (local_left - 8, local_top - 8),
        (local_left + width + 8, local_top + height + 8),
        (255, 255, 0),
        4,
    )
    label_y = max(24, local_top - 12)
    cv2.putText(
        view,
        f"TARGET TRACK T{track_id}",
        (max(6, local_left), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        f"TARGET TRACK T{track_id}",
        (max(6, local_left), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return view


def _frame_difference_view(
    content: np.ndarray,
    baseline: np.ndarray,
    *,
    threshold: int = 24,
) -> np.ndarray:
    """Overlay temporal changes against the pre-onset frame.

    This is an auxiliary localization aid, not a replacement for the normal
    RGB identity crop. It makes a newly appearing small body stand out when a
    large older unit remains in the same local crop.
    """

    if content.shape != baseline.shape:
        raise ValueError("frame-difference images must have identical shapes")
    delta = cv2.absdiff(content, baseline)
    gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.4)
    changed = gray >= threshold
    heat = np.zeros_like(content)
    heat[changed] = (0, 0, 255)
    return cv2.addWeighted(content, 0.78, heat, 0.42, 0)


def _render_dominant_tile(
    *,
    content: np.ndarray,
    frame_index: int,
    fps: float,
    selected: _TrackSelection,
    anchor_bbox: tuple[int, int, int, int],
    observations: dict[int, tuple[int, int, int, int]],
    width: int,
    height: int,
    track_follow_frames: int,
    mask_context: bool,
    tile_width: int,
    center_offset_x: int,
    center_offset_y: int,
) -> np.ndarray:
    bbox = _nearest_bbox(
        observations,
        frame_index,
        max_distance=max(0, track_follow_frames),
    )
    if bbox is None:
        bbox = anchor_bbox
    left, top, bbox_width, bbox_height = bbox
    center_x = round(left + bbox_width / 2) + center_offset_x
    center_y = round(top + bbox_height / 2) + center_offset_y
    view, origin_x, origin_y = _crop_with_origin(
        content,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
    )
    view = _draw_selected_track(
        view,
        bbox=bbox,
        origin_x=origin_x,
        origin_y=origin_y,
        track_id=selected.track_id,
        mask_context=mask_context,
    )
    view = add_frame_identity(
        view,
        source_frame_index=frame_index,
        fps=fps,
    )
    scale = tile_width / view.shape[1]
    return cv2.resize(
        view,
        (
            tile_width,
            max(1, round(view.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _target_center(
    *,
    burst: dict[str, object],
    tracks: dict[int, dict[int, tuple[int, int, int, int]]],
    frame_index: int,
    onset_frame: int,
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    paired_tracks = list(
        zip(
            burst["track_ids"],
            burst["track_start_frames"],
            burst["first_bboxes"],
            strict=True,
        )
    )
    closest_distance = min(
        abs(int(start_frame) - onset_frame)
        for _, start_frame, _ in paired_tracks
    )
    selected = [
        (track_id, first_bbox)
        for track_id, start_frame, first_bbox in paired_tracks
        if abs(int(start_frame) - onset_frame) == closest_distance
    ]
    boxes = [
        bbox
        for track_id, _ in selected
        if (
            bbox := _nearest_bbox(
                tracks.get(int(track_id), {}),
                frame_index,
            )
        )
        is not None
    ]
    if not boxes:
        boxes = [
            tuple(int(value) for value in bbox)
            for _, bbox in selected
        ]
    center_x = round(
        sum(left + width / 2 for left, _, width, _ in boxes) / len(boxes)
    )
    center_y = round(
        sum(top + height / 2 for _, top, _, height in boxes) / len(boxes)
    )
    return center_x, center_y, boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render deterministic delayed, grid-free enemy identity sheets."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--targets-file",
        type=Path,
        help=(
            "Input target document; defaults to "
            "RUN_DIR/enemy_identity_targets.json."
        ),
    )
    parser.add_argument(
        "--output-targets",
        type=Path,
        help=(
            "Output target document. Required by --track-selection dominant "
            "so the v1 target document is never overwritten."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for rendered sheets; defaults to RUN_DIR/reviews.",
    )
    parser.add_argument(
        "--track-selection",
        choices=("legacy", "dominant"),
        default="legacy",
        help="Keep v1 all-nearest boxes or render selected-track v2 crops.",
    )
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument(
        "--local-width",
        type=int,
        default=360,
        help="Selected-track v2 crop width in content pixels.",
    )
    parser.add_argument(
        "--local-height",
        type=int,
        default=520,
        help="Selected-track v2 crop height in content pixels.",
    )
    parser.add_argument(
        "--onset-tolerance",
        type=int,
        default=2,
        help="Allowed marker-start distance used by dominant-track eligibility.",
    )
    parser.add_argument(
        "--track-follow-frames",
        type=int,
        default=12,
        help="Maximum frame gap for following a selected track in v2 sheets.",
    )
    parser.add_argument(
        "--center-offset-x",
        type=int,
        default=0,
        help="Selected-track crop center offset in content pixels.",
    )
    parser.add_argument(
        "--center-offset-y",
        type=int,
        default=0,
        help="Selected-track crop center offset in content pixels.",
    )
    parser.add_argument(
        "--mask-context",
        action="store_true",
        help="Dim context outside an expanded selected-track ellipse in v2.",
    )
    parser.add_argument(
        "--include-frame-difference",
        action="store_true",
        help=(
            "Write an additional v2 temporal-difference sheet against the "
            "pre-onset frame."
        ),
    )
    args = parser.parse_args()
    if args.tile_width <= 0:
        parser.error("--tile-width must be positive")
    if args.local_width <= 0 or args.local_height <= 0:
        parser.error("--local-width and --local-height must be positive")
    if args.onset_tolerance < 0:
        parser.error("--onset-tolerance must be non-negative")
    if args.track_follow_frames < 0:
        parser.error("--track-follow-frames must be non-negative")
    if args.track_selection == "dominant" and args.output_targets is None:
        parser.error(
            "--output-targets is required with --track-selection dominant"
        )
    run_dir = args.run_dir.resolve()
    source_path = (
        args.targets_file.resolve()
        if args.targets_file is not None
        else run_dir / "enemy_identity_targets.json"
    )
    output_targets = (
        args.output_targets.resolve()
        if args.output_targets is not None
        else source_path
    )
    if (
        args.track_selection == "dominant"
        and output_targets == source_path
    ):
        parser.error(
            "--output-targets must differ from --targets-file in dominant mode"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    marker_document = json.loads(
        (run_dir / "enemy_marker_candidates.json").read_text(encoding="utf-8")
    )
    bursts, tracks = _track_observations(marker_document)
    frames = {
        int(row["source_frame_index"]): row for row in manifest["frames"]
    }
    segment_start = int(manifest["segment"]["start_frame"])
    segment_end = int(manifest["segment"]["end_frame_exclusive"])
    review_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "reviews"
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    selected_tracks: dict[str, dict[str, object]] = {}
    for target in source["targets"]:
        if target["kind"] != "unit_or_building":
            continue
        onset = int(target["event_frame_index"])
        burst_id = target.get("track_id")
        burst = bursts.get(str(burst_id))
        if burst is None:
            for artifact in target.get("verification_artifacts", []):
                match = re.search(r"enemy-marker-burst-(\d{6})", artifact)
                if match is not None:
                    burst_id = f"enemy-marker-burst:{match.group(1)}"
                    burst = bursts.get(burst_id)
                    if burst is not None:
                        break
        if burst is None:
            raise ValueError(
                f"{target['onset_id']}: missing marker burst {burst_id!r}"
            )
        if args.track_selection == "dominant":
            selected, anchor_bbox = _select_dominant_track(
                burst=burst,
                tracks=tracks,
                onset_frame=onset,
                onset_tolerance=args.onset_tolerance,
            )
            artifacts = []
            identity_frame = None
            groups = (
                ("a", (5, 7, 9, 11, 13, 15)),
                ("b", (8, 11, 14, 17, 20, 23)),
            )
            observations = tracks[selected.track_id]
            difference_tiles: list[np.ndarray] = []
            baseline_index = max(segment_start, onset - 2)
            baseline_image = cv2.imread(
                str(run_dir / frames[baseline_index]["path"])
            )
            if baseline_image is None:
                raise FileNotFoundError(frames[baseline_index]["path"])
            baseline_content = baseline_image[
                manifest["label_margin_px"] :
            ].copy()
            for suffix, offsets in groups:
                sampled = _sample_frames(
                    onset=onset,
                    offsets=offsets,
                    segment_start=segment_start,
                    segment_end=segment_end,
                )
                tiles = []
                for frame_index in sampled:
                    image = cv2.imread(
                        str(run_dir / frames[frame_index]["path"])
                    )
                    if image is None:
                        raise FileNotFoundError(frames[frame_index]["path"])
                    content = image[manifest["label_margin_px"] :].copy()
                    tiles.append(
                        _render_dominant_tile(
                            content=content,
                            frame_index=frame_index,
                            fps=float(manifest["fps"]),
                            selected=selected,
                            anchor_bbox=anchor_bbox,
                            observations=observations,
                            width=args.local_width,
                            height=args.local_height,
                            track_follow_frames=args.track_follow_frames,
                            mask_context=args.mask_context,
                            tile_width=args.tile_width,
                            center_offset_x=args.center_offset_x,
                            center_offset_y=args.center_offset_y,
                        )
                    )
                    if args.include_frame_difference:
                        difference_content = _frame_difference_view(
                            content,
                            baseline_content,
                        )
                        difference_tiles.append(
                            _render_dominant_tile(
                                content=difference_content,
                                frame_index=frame_index,
                                fps=float(manifest["fps"]),
                                selected=selected,
                                anchor_bbox=anchor_bbox,
                                observations=observations,
                                width=args.local_width,
                                height=args.local_height,
                                track_follow_frames=args.track_follow_frames,
                                mask_context=args.mask_context,
                                tile_width=args.tile_width,
                                center_offset_x=args.center_offset_x,
                                center_offset_y=args.center_offset_y,
                            )
                        )
                sheet = _contact_sheet(tiles, columns=min(3, len(tiles)))
                if sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS:
                    raise ValueError(
                        f"{target['onset_id']} v2 identity sheet is too large"
                    )
                name = target["onset_id"].replace(":", "-")
                output = review_dir / f"identity-v2-{name}-{suffix}.jpg"
                if not cv2.imwrite(str(output), sheet):
                    raise OSError(output)
                record_review(
                    run_dir=run_dir,
                    output_path=output,
                    purpose="identity",
                    start_frame=sampled[0],
                    end_frame=sampled[-1] + 1,
                    candidate_id=None,
                    event_id=(
                        f"identity-enemy-"
                        f"{int(target['event_frame_index']):06d}"
                    ),
                )
                try:
                    artifact = str(output.relative_to(run_dir))
                except ValueError:
                    artifact = str(output.resolve())
                artifacts.append(artifact)
                if identity_frame is None:
                    identity_frame = sampled[0]
                rendered += 1
            if args.include_frame_difference:
                difference_sheet = _contact_sheet(
                    difference_tiles,
                    columns=3,
                )
                if (
                    difference_sheet.shape[0] * difference_sheet.shape[1]
                    > REVIEW_MAX_PIXELS
                ):
                    raise ValueError(
                        f"{target['onset_id']} v2 difference sheet is too large"
                    )
                name = target["onset_id"].replace(":", "-")
                difference_output = (
                    review_dir / f"identity-v2-{name}-difference.jpg"
                )
                if not cv2.imwrite(str(difference_output), difference_sheet):
                    raise OSError(difference_output)
                record_review(
                    run_dir=run_dir,
                    output_path=difference_output,
                    purpose="identity",
                    start_frame=onset + 5,
                    end_frame=min(segment_end, onset + 24),
                    candidate_id=None,
                    event_id=(
                        f"identity-enemy-"
                        f"{int(target['event_frame_index']):06d}"
                    ),
                )
                try:
                    difference_artifact = str(
                        difference_output.relative_to(run_dir)
                    )
                except ValueError:
                    difference_artifact = str(difference_output.resolve())
                artifacts.append(difference_artifact)
            target["identity_frame_index"] = identity_frame
            target["identity_artifacts"] = artifacts
            target["identity_track_id"] = selected.track_id
            target["identity_track_stats"] = asdict(selected)
            target["identity_render_options"] = {
                "mode": "dominant",
                "local_width": args.local_width,
                "local_height": args.local_height,
                "onset_tolerance": args.onset_tolerance,
                "track_follow_frames": args.track_follow_frames,
                "center_offset_x": args.center_offset_x,
                "center_offset_y": args.center_offset_y,
                "mask_context": bool(args.mask_context),
                "include_frame_difference": bool(
                    args.include_frame_difference
                ),
            }
            selected_tracks[target["onset_id"]] = asdict(selected)
            continue
        groups = (
            ("a", (5, 7, 9, 11, 13, 15)),
            ("b", (8, 11, 14, 17, 20, 23)),
        )
        artifacts = []
        identity_frame = None
        for suffix, offsets in groups:
            sampled = _sample_frames(
                onset=onset,
                offsets=offsets,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            tiles = []
            for frame_index in sampled:
                image = cv2.imread(str(run_dir / frames[frame_index]["path"]))
                if image is None:
                    raise FileNotFoundError(frames[frame_index]["path"])
                content = image[manifest["label_margin_px"] :].copy()
                center_x, center_y, boxes = _target_center(
                    burst=burst,
                    tracks=tracks,
                    frame_index=frame_index,
                    onset_frame=onset,
                )
                for left, top, width, height in boxes:
                    cv2.rectangle(
                        content,
                        (left - 8, top - 8),
                        (left + width + 8, top + height + 8),
                        (255, 255, 0),
                        4,
                    )
                view = _bounded_crop(
                    content,
                    center_x=center_x,
                    center_y=center_y,
                    width=620,
                    height=760,
                )
                view = add_frame_identity(
                    view,
                    source_frame_index=frame_index,
                    fps=float(manifest["fps"]),
                )
                scale = args.tile_width / view.shape[1]
                tiles.append(
                    cv2.resize(
                        view,
                        (
                            args.tile_width,
                            max(1, round(view.shape[0] * scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            sheet = _contact_sheet(tiles, columns=min(3, len(tiles)))
            if sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS:
                raise ValueError(
                    f"{target['onset_id']} identity sheet is too large"
                )
            name = target["onset_id"].replace(":", "-")
            output = review_dir / f"identity-{name}-{suffix}.jpg"
            if not cv2.imwrite(str(output), sheet):
                raise OSError(output)
            record_review(
                run_dir=run_dir,
                output_path=output,
                purpose="identity",
                start_frame=sampled[0],
                end_frame=sampled[-1] + 1,
                candidate_id=None,
                event_id=(
                    f"identity-enemy-"
                    f"{int(target['event_frame_index']):06d}"
                ),
            )
            artifacts.append(str(output.relative_to(run_dir)))
            if identity_frame is None:
                identity_frame = sampled[0]
            rendered += 1
        target["identity_frame_index"] = identity_frame
        target["identity_artifacts"] = artifacts
    output_targets.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_targets, source)
    print(
        json.dumps(
            {
                "targets": len(source["targets"]),
                "rendered": rendered,
                "track_selection": args.track_selection,
                "output_targets": str(output_targets),
                "selected_tracks": selected_tracks,
            }
        )
    )


if __name__ == "__main__":
    main()
