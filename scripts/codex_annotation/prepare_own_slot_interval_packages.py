from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import (
    REVIEW_MAX_PIXELS,
    _make_contact_sheet,
    add_frame_identity,
    atomic_write_json,
)
from cr_bot.annotation_stages import record_review
from cr_bot.domain.rois import ROIS
from cr_bot.vision.elixir import extract_elixir


HAND_CARD_ART_ROI = (18, 38, 184, 212)
EMPTY_SATURATION_THRESHOLD = 220.0
EMPTY_EDGE_MEAN_THRESHOLD = 12.0
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200
HAND_SLOT_COUNT = 4
CARD_RETURN_LOOKBACK_FRAMES = 12
CARD_RETURN_LOOKAHEAD_FRAMES = 14
SAME_CARD_RETURN_THRESHOLD = 0.90


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def empty_card_art_statistics(card_art: np.ndarray) -> tuple[float, float]:
    if card_art.size == 0:
        raise ValueError("card-art crop is empty")
    saturation = cv2.cvtColor(card_art, cv2.COLOR_BGR2HSV)[:, :, 1]
    gray = cv2.cvtColor(card_art, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
    )
    return float(saturation.mean()), float(edges.mean())


def is_empty_card_art(card_art: np.ndarray) -> bool:
    saturation_mean, edge_mean = empty_card_art_statistics(card_art)
    return (
        saturation_mean > EMPTY_SATURATION_THRESHOLD
        and edge_mean < EMPTY_EDGE_MEAN_THRESHOLD
    )


def card_art_histogram(card_art: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(card_art, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [30, 16],
        [0, 180, 0, 256],
    )
    cv2.normalize(histogram, histogram)
    return histogram


def same_card_return_score(
    before_card_arts: list[np.ndarray],
    after_card_arts: list[np.ndarray],
) -> float | None:
    """Measure whether the same hand card returns after an empty interval.

    Multiple frames make the comparison robust to selection glow, elixir
    greying, and the first animated frame after a card returns to the hand.
    """

    if not before_card_arts or not after_card_arts:
        return None
    before = [card_art_histogram(value) for value in before_card_arts]
    after = [card_art_histogram(value) for value in after_card_arts]
    return max(
        float(cv2.compareHist(left, right, cv2.HISTCMP_CORREL))
        for left in before
        for right in after
    )


def merge_empty_frame_intervals(
    frame_indices: list[int],
    *,
    max_separating_frames: int = 3,
    min_empty_frames: int = 2,
) -> list[tuple[int, int]]:
    """Return inclusive intervals after bridging short non-empty gaps."""

    if max_separating_frames < 0 or min_empty_frames < 1:
        raise ValueError("interval thresholds are invalid")
    frames = sorted(set(frame_indices))
    if not frames:
        return []

    merged: list[tuple[int, int, int]] = []
    start = frames[0]
    previous = frames[0]
    empty_count = 1
    for frame in frames[1:]:
        separating_frames = frame - previous - 1
        if separating_frames <= max_separating_frames:
            previous = frame
            empty_count += 1
            continue
        merged.append((start, previous, empty_count))
        start = frame
        previous = frame
        empty_count = 1
    merged.append((start, previous, empty_count))
    return [
        (interval_start, interval_end)
        for interval_start, interval_end, count in merged
        if count >= min_empty_frames
    ]


def _detect_empty_frames(
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[int, list[int]], dict[int, int]]:
    empty_by_slot = {slot: [] for slot in range(1, HAND_SLOT_COUNT + 1)}
    displayed_elixir_by_frame: dict[int, int] = {}
    label_margin = int(manifest["label_margin_px"])
    art_x, art_y, art_width, art_height = HAND_CARD_ART_ROI
    for record in manifest["frames"]:
        frame_index = int(record["source_frame_index"])
        labeled = cv2.imread(str(run_dir / record["path"]))
        if labeled is None:
            raise FileNotFoundError(run_dir / record["path"])
        content = labeled[label_margin:]
        displayed_elixir_by_frame[frame_index] = int(
            extract_elixir(content)["displayed_digit"]
        )
        for slot in empty_by_slot:
            slot_image = _crop(content, ROIS[f"hand_card_slot_{slot}"])
            card_art = slot_image[
                art_y : art_y + art_height,
                art_x : art_x + art_width,
            ]
            if is_empty_card_art(card_art):
                empty_by_slot[slot].append(frame_index)
    return empty_by_slot, displayed_elixir_by_frame


def _elixir_drop_transitions(
    *,
    displayed_elixir_by_frame: dict[int, int],
    interval_start: int,
    interval_end: int,
    segment_end: int,
) -> list[dict[str, int]]:
    transitions = []
    for frame_index in range(
        interval_start,
        min(segment_end, interval_end + 7),
    ):
        before = displayed_elixir_by_frame.get(frame_index - 1)
        after = displayed_elixir_by_frame.get(frame_index)
        if before is None or after is None or after >= before:
            continue
        transitions.append(
            {
                "frame_index": frame_index,
                "before": before,
                "after": after,
                "drop": before - after,
            }
        )
    return transitions


def _card_art(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    slot: int,
) -> np.ndarray:
    labeled = cv2.imread(str(run_dir / record["path"]))
    if labeled is None:
        raise FileNotFoundError(run_dir / record["path"])
    content = labeled[int(manifest["label_margin_px"]) :]
    slot_image = _crop(content, ROIS[f"hand_card_slot_{slot}"])
    art_x, art_y, art_width, art_height = HAND_CARD_ART_ROI
    return slot_image[
        art_y : art_y + art_height,
        art_x : art_x + art_width,
    ]


def _return_evidence(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    frames_by_index: dict[int, dict[str, Any]],
    slot: int,
    interval_start: int,
    interval_end: int,
) -> dict[str, Any]:
    segment = manifest["segment"]
    segment_start = int(segment["start_frame"])
    segment_end = int(segment["end_frame_exclusive"])

    before: list[np.ndarray] = []
    for frame_index in range(
        max(segment_start, interval_start - CARD_RETURN_LOOKBACK_FRAMES),
        interval_start,
    ):
        art = _card_art(
            run_dir=run_dir,
            manifest=manifest,
            record=frames_by_index[frame_index],
            slot=slot,
        )
        if not is_empty_card_art(art):
            before.append(art)

    after: list[np.ndarray] = []
    first_return_frame = None
    for frame_index in range(
        interval_end + 1,
        min(
            segment_end,
            interval_end + CARD_RETURN_LOOKAHEAD_FRAMES + 1,
        ),
    ):
        art = _card_art(
            run_dir=run_dir,
            manifest=manifest,
            record=frames_by_index[frame_index],
            slot=slot,
        )
        if is_empty_card_art(art):
            continue
        if first_return_frame is None:
            first_return_frame = frame_index
        after.append(art)

    score = same_card_return_score(before, after)
    has_confirmation_horizon = interval_end + 5 < segment_end
    same_card_returned = (
        score is not None and score >= SAME_CARD_RETURN_THRESHOLD
    )
    outcome_constraint = (
        "canceled"
        if same_card_returned or not has_confirmation_horizon
        else "released"
    )
    return {
        "outcome_constraint": outcome_constraint,
        "same_card_return_score": score,
        "same_card_return_threshold": SAME_CARD_RETURN_THRESHOLD,
        "first_nonempty_return_frame": first_return_frame,
        "has_post_release_confirmation_horizon": has_confirmation_horizon,
    }


def _nearest_unique_candidate_id(
    interval_start: int,
    candidates: list[dict[str, Any]],
) -> str | None:
    distances = [
        (
            abs(int(candidate["approximate_frame_index"]) - interval_start),
            candidate["candidate_id"],
        )
        for candidate in candidates
        if isinstance(candidate.get("candidate_id"), str)
        and isinstance(candidate.get("approximate_frame_index"), int)
    ]
    if not distances:
        return None
    minimum = min(distance for distance, _ in distances)
    nearest = [
        candidate_id
        for distance, candidate_id in distances
        if distance == minimum
    ]
    return nearest[0] if len(nearest) == 1 else None


def _scaled_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(
        image,
        (width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
    top = (height - image.shape[0]) // 2
    return cv2.copyMakeBorder(
        image,
        top,
        height - image.shape[0] - top,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _interval_frame_tile(
    content: np.ndarray,
    *,
    frame_index: int,
    fps: float,
    slot: int,
) -> np.ndarray:
    hud_x, hud_y, hud_width, hud_height = (30, 1900, 1030, 500)
    hud = content[
        hud_y : hud_y + hud_height,
        hud_x : hud_x + hud_width,
    ].copy()
    slot_x, slot_y, slot_width, slot_height = ROIS[
        f"hand_card_slot_{slot}"
    ]
    cv2.rectangle(
        hud,
        (slot_x - hud_x, slot_y - hud_y),
        (
            slot_x - hud_x + slot_width,
            slot_y - hud_y + slot_height,
        ),
        (0, 255, 255),
        5,
    )
    cv2.putText(
        hud,
        f"SLOT {slot}",
        (max(5, slot_x - hud_x), 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    hud = _scaled_width(hud, 320)

    arena = _crop(content, ROIS["battlefield"])
    arena = _scaled_width(arena, 180)
    height = max(hud.shape[0], arena.shape[0])
    synchronized = np.hstack(
        [_pad_height(hud, height), _pad_height(arena, height)]
    )
    return add_frame_identity(
        synchronized,
        source_frame_index=frame_index,
        fps=fps,
    )


def _fit_review_limit(sheet: np.ndarray) -> np.ndarray:
    pixels = sheet.shape[0] * sheet.shape[1]
    if pixels <= REVIEW_MAX_PIXELS:
        return sheet
    scale = math.sqrt(REVIEW_MAX_PIXELS / pixels) * 0.995
    fitted = cv2.resize(
        sheet,
        (
            max(1, math.floor(sheet.shape[1] * scale)),
            max(1, math.floor(sheet.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    if fitted.shape[0] * fitted.shape[1] > REVIEW_MAX_PIXELS:
        raise ValueError("failed to fit own-slot review within pixel limit")
    return fitted


def _render_interval(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    frames_by_index: dict[int, dict[str, Any]],
    slot: int,
    interval_start: int,
    interval_end: int,
) -> tuple[str, list[int]]:
    segment = manifest["segment"]
    review_start = max(
        int(segment["start_frame"]),
        interval_start - 2,
    )
    review_end = min(
        int(segment["end_frame_exclusive"]) - 1,
        interval_end + 6,
    )
    sampled = list(range(review_start, review_end + 1))
    tiles = []
    for frame_index in sampled:
        record = frames_by_index.get(frame_index)
        if record is None:
            raise ValueError(f"frame {frame_index} was not prepared")
        labeled = cv2.imread(str(run_dir / record["path"]))
        if labeled is None:
            raise FileNotFoundError(run_dir / record["path"])
        content = labeled[int(manifest["label_margin_px"]) :]
        tiles.append(
            _interval_frame_tile(
                content,
                frame_index=frame_index,
                fps=float(manifest["fps"]),
                slot=slot,
            )
        )

    tile_ratio = tiles[0].shape[1] / tiles[0].shape[0]
    columns = max(1, math.ceil(math.sqrt(len(tiles) / tile_ratio)))
    sheet = _fit_review_limit(_make_contact_sheet(tiles, columns=columns))
    output = run_dir / "reviews" / (
        f"own-slot-{slot}-{interval_start:06d}-{interval_end:06d}.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"failed to write {output}")
    record_review(
        run_dir=run_dir,
        output_path=output,
        purpose="own_context",
        start_frame=review_start,
        end_frame=review_end + 1,
        candidate_id=None,
        event_id=None,
    )
    return str(output.relative_to(run_dir)), sampled


def _render_card_identity(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    frames_by_index: dict[int, dict[str, Any]],
    slot: int,
    interval_start: int,
    interval_end: int,
) -> str | None:
    segment_start = int(manifest["segment"]["start_frame"])
    candidate_frames = []
    for frame_index in range(
        max(segment_start, interval_start - CARD_RETURN_LOOKBACK_FRAMES),
        interval_start,
    ):
        art = _card_art(
            run_dir=run_dir,
            manifest=manifest,
            record=frames_by_index[frame_index],
            slot=slot,
        )
        if not is_empty_card_art(art):
            candidate_frames.append(frame_index)
    sampled = candidate_frames[-3:]
    if not sampled:
        # A clip can begin while a hand slot is already empty.  That interval
        # is left-truncated: there is no visible departing card and therefore
        # no blind evidence from which to identify an event.  Do not invent an
        # identity or abort preparation of the remaining, fully observed
        # intervals.
        return None

    tiles = []
    for frame_index in sampled:
        record = frames_by_index[frame_index]
        labeled = cv2.imread(str(run_dir / record["path"]))
        if labeled is None:
            raise FileNotFoundError(run_dir / record["path"])
        content = labeled[int(manifest["label_margin_px"]) :]
        slot_view = _crop(content, ROIS[f"hand_card_slot_{slot}"])
        tile = add_frame_identity(
            slot_view,
            source_frame_index=frame_index,
            fps=float(manifest["fps"]),
        )
        tiles.append(_scaled_width(tile, 480))
    sheet = _fit_review_limit(
        _make_contact_sheet(tiles, columns=len(tiles))
    )
    output = run_dir / "reviews" / (
        f"own-slot-card-{slot}-{interval_start:06d}-{interval_end:06d}.jpg"
    )
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"failed to write {output}")
    record_review(
        run_dir=run_dir,
        output_path=output,
        purpose="identity",
        start_frame=sampled[0],
        end_frame=sampled[-1] + 1,
        candidate_id=None,
        event_id=None,
    )
    return str(output.relative_to(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect deterministic empty hand-slot intervals and render bounded "
            "release/cancellation work packages."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=200)
    args = parser.parse_args()
    if args.chunk_frames < 20:
        parser.error("--chunk-frames must be at least 20")

    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    segment = manifest["segment"]
    frames_by_index = {
        int(row["source_frame_index"]): row for row in manifest["frames"]
    }
    empty_by_slot, displayed_elixir_by_frame = _detect_empty_frames(
        run_dir, manifest
    )
    candidates = list(
        manifest.get("candidate_discovery", {}).get("own_candidates", [])
    )

    intervals: list[dict[str, Any]] = []
    skipped_without_pre_interval_identity: list[dict[str, int]] = []
    for slot, empty_frames in empty_by_slot.items():
        for start, end in merge_empty_frame_intervals(empty_frames):
            card_identity_artifact = _render_card_identity(
                run_dir=run_dir,
                manifest=manifest,
                frames_by_index=frames_by_index,
                slot=slot,
                interval_start=start,
                interval_end=end,
            )
            if card_identity_artifact is None:
                skipped_without_pre_interval_identity.append(
                    {"slot": slot, "start": start, "end": end}
                )
                continue
            artifact, sampled = _render_interval(
                run_dir=run_dir,
                manifest=manifest,
                frames_by_index=frames_by_index,
                slot=slot,
                interval_start=start,
                interval_end=end,
            )
            intervals.append(
                {
                    "interval_id": (
                        f"own-slot:{slot}:{start:06d}-{end:06d}"
                    ),
                    "slot": slot,
                    "empty_range": [start, end],
                    "sampled_frame_indices": sampled,
                    "artifact": artifact,
                    "card_identity_artifact": card_identity_artifact,
                    "candidate_id": _nearest_unique_candidate_id(
                        start,
                        candidates,
                    ),
                    "return_evidence": _return_evidence(
                        run_dir=run_dir,
                        manifest=manifest,
                        frames_by_index=frames_by_index,
                        slot=slot,
                        interval_start=start,
                        interval_end=end,
                    ),
                    "elixir_drop_transitions": _elixir_drop_transitions(
                        displayed_elixir_by_frame=displayed_elixir_by_frame,
                        interval_start=start,
                        interval_end=end,
                        segment_end=int(segment["end_frame_exclusive"]),
                    ),
                }
            )
    intervals.sort(
        key=lambda row: (
            row["empty_range"][0],
            row["slot"],
            row["empty_range"][1],
        )
    )

    package_dir = run_dir / "work_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    package_summaries = []
    segment_start = int(segment["start_frame"])
    segment_end = int(segment["end_frame_exclusive"])
    for start in range(segment_start, segment_end, args.chunk_frames):
        end = min(segment_end, start + args.chunk_frames)
        owned = [
            row
            for row in intervals
            if start <= row["empty_range"][0] < end
        ]
        path = package_dir / f"own-slot-{start:06d}-{end:06d}.json"
        atomic_write_json(
            path,
            {
                "run_id": manifest["run_id"],
                "stage": "own_slot_intervals_package",
                "decision_schema_version": 1,
                "fps": manifest["fps"],
                "segment": segment,
                "target_range": [start, end],
                "detector": {
                    "card_art_roi": list(HAND_CARD_ART_ROI),
                    "empty_if": {
                        "mean_hsv_saturation_gt": EMPTY_SATURATION_THRESHOLD,
                        "canny_edge_mean_lt": EMPTY_EDGE_MEAN_THRESHOLD,
                        "canny_thresholds": [
                            CANNY_LOW_THRESHOLD,
                            CANNY_HIGH_THRESHOLD,
                        ],
                    },
                    "merge_separating_frames_lte": 3,
                    "minimum_empty_frames": 2,
                    "empty_range_convention": "inclusive",
                },
                "intervals": owned,
            },
        )
        package_summaries.append(
            {
                "target_range": [start, end],
                "interval_count": len(owned),
                "path": str(path),
            }
        )
    print(
        json.dumps(
            {
                "interval_count": len(intervals),
                "skipped_without_pre_interval_identity": (
                    skipped_without_pre_interval_identity
                ),
                "packages": package_summaries,
            }
        )
    )


if __name__ == "__main__":
    main()
