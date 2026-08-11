from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import (
    _make_contact_sheet,
    _review_view,
    add_frame_identity,
    atomic_write_json,
)
from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.rois import ROIS
from cr_bot.features.action_space import ACTION_GRID


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _event_id(event: dict[str, Any]) -> str:
    return f"event-own-{int(event['event_frame_index']):06d}-{event['card']}"


def _own_events(source: dict[str, Any]) -> list[dict[str, Any]]:
    source_events = source.get("events")
    if not isinstance(source_events, list) or not source_events:
        raise ValueError("semantic source must contain events")
    if any(not isinstance(row, dict) for row in source_events):
        raise ValueError("semantic source events must be objects")
    events = [row for row in source_events if row.get("side", "own") == "own"]
    if not events:
        raise ValueError("semantic source must contain own events")
    return events


def _rule_options(card: str) -> list[str]:
    base = card[4:] if card.startswith("evo-") else card
    kind = CARD_METADATA[base]["kind"]
    if base == "log":
        return ["initial_rolling_object_center"]
    if kind == "spell":
        return ["target_center", "impact_center"]
    if kind == "building":
        return ["deployment_center", "spawn_center"]
    return ["spawn_center", "deployment_center"]


def _frame_indices(frame: int, start: int, end: int) -> tuple[list[int], list[int]]:
    # The semantic onset can legitimately lead or trail the first visible
    # placement marker by several frames.  Keep a dense clean timeline and put
    # both sides of the onset into the exact-coordinate sheets.  The old
    # evidence had no pre-onset grid frames, which made it needlessly hard to
    # distinguish a newly spawned body from an actor already in the arena.
    macro = list(range(max(start, frame - 9), min(end, frame + 19)))
    offsets = (
        -9, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18,
    )
    grid = sorted({frame + offset for offset in offsets if start <= frame + offset < end})
    return macro, grid


def _grid_frame_groups(frame: int, grid_frames: list[int]) -> tuple[list[int], list[int]]:
    """Split exact evidence into readable onset and follow-through sheets."""

    onset = [value for value in grid_frames if value <= frame + 3]
    follow = [value for value in grid_frames if value > frame + 3]
    return onset, follow


def _render_sheet(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    frame_indices: list[int],
    purpose: str,
    output: Path,
) -> None:
    by_index = {row["source_frame_index"]: row for row in manifest["frames"]}
    tiles = []
    tile_width = 420 if purpose == "macro" else 700
    columns = 5 if purpose == "macro" else 3
    for frame_index in frame_indices:
        record = by_index.get(frame_index)
        if record is None:
            raise ValueError(f"prepared frame {frame_index} is missing")
        labeled = cv2.imread(str(run_dir / record["path"]))
        if labeled is None:
            raise FileNotFoundError(run_dir / record["path"])
        frame = labeled[int(manifest["label_margin_px"]) :]
        if purpose == "grid_full":
            view = _axis_grid_view(frame, first_row=0)
        elif purpose == "grid_own":
            view = _axis_grid_view(frame, first_row=14)
        else:
            view = _review_view(
                frame,
                purpose="arena",
                grid_center=None,
                grid_radius=3,
                focus_cell=None,
                focus_radius=4,
            )
        view = add_frame_identity(
            view, source_frame_index=frame_index, fps=float(manifest["fps"])
        )
        scale = tile_width / view.shape[1]
        tiles.append(
            cv2.resize(
                view,
                (tile_width, max(1, round(view.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), _make_contact_sheet(tiles, columns=columns)):
        raise OSError(f"failed to write {output}")


def _axis_grid_view(frame: np.ndarray, *, first_row: int = 0) -> np.ndarray:
    """Draw an unobstructed grid with large external coordinate rulers."""

    if not 0 <= first_row < ACTION_GRID.rows:
        raise ValueError("first grid row is outside the action grid")

    x, y, width, height = ROIS["battlefield"]
    arena = frame[y : y + height, x : x + width].copy()
    gx0 = round(ACTION_GRID.x0 * width)
    gy0 = round(ACTION_GRID.y0 * height)
    gx1 = round(ACTION_GRID.x1 * width)
    gy1 = round(ACTION_GRID.y1 * height)
    for column in range(ACTION_GRID.cols + 1):
        line_x = round(gx0 + column / ACTION_GRID.cols * (gx1 - gx0))
        cv2.line(arena, (line_x, gy0), (line_x, gy1), (25, 25, 25), 2)
    for row in range(ACTION_GRID.rows + 1):
        line_y = round(gy0 + row / ACTION_GRID.rows * (gy1 - gy0))
        cv2.line(arena, (gx0, line_y), (gx1, line_y), (25, 25, 25), 2)

    left_ruler = 58
    top_ruler = 48
    crop_top = 0
    if first_row:
        crop_top = round(
            gy0 + first_row / ACTION_GRID.rows * (gy1 - gy0)
        )
    visible_arena = arena[crop_top:]
    canvas = np.zeros(
        (visible_arena.shape[0] + top_ruler, width + left_ruler, 3),
        dtype=np.uint8,
    )
    canvas[top_ruler:, left_ruler:] = visible_arena
    for column in range(ACTION_GRID.cols):
        center_x = round(
            left_ruler
            + gx0
            + (column + 0.5) / ACTION_GRID.cols * (gx1 - gx0)
        )
        cv2.putText(
            canvas,
            str(column),
            (center_x - (12 if column >= 10 else 6), 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for row in range(first_row, ACTION_GRID.rows):
        center_y = round(
            top_ruler
            + gy0
            + (row + 0.5) / ACTION_GRID.rows * (gy1 - gy0)
            - crop_top
        )
        cv2.putText(
            canvas,
            str(row),
            (5, center_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def prepare_packages(
    *,
    run_dir: Path,
    source_file: Path,
    output_dir: Path,
    chunk_size: int,
    event_ids: set[str] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    source = _read(source_file)
    manifest = _read(run_dir / "manifest.json")
    if source.get("run_id") != manifest.get("run_id"):
        raise ValueError("semantic source run_id does not match manifest")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    events = _own_events(source)
    if event_ids:
        events = [row for row in events if _event_id(row) in event_ids]
        found = {_event_id(row) for row in events}
        if found != event_ids:
            raise ValueError(f"unknown localization event IDs: {sorted(event_ids - found)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "reviews"
    package_dir = output_dir / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    segment = manifest["segment"]
    targets = []
    seen_ids: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("own semantic events must be objects")
        event_id = _event_id(event)
        if event_id in seen_ids:
            raise ValueError(f"duplicate localization event ID {event_id}")
        seen_ids.add(event_id)
        frame = int(event["event_frame_index"])
        macro_frames, grid_frames = _frame_indices(
            frame,
            int(segment["start_frame"]),
            int(segment["end_frame_exclusive"]),
        )
        stem = event_id.removeprefix("event-own-")
        macro_path = review_dir / f"{stem}-macro.jpg"
        grid_onset_path = review_dir / f"{stem}-grid-onset.jpg"
        grid_follow_path = review_dir / f"{stem}-grid-follow.jpg"
        _render_sheet(
            run_dir=run_dir,
            manifest=manifest,
            frame_indices=macro_frames,
            purpose="macro",
            output=macro_path,
        )
        grid_onset_frames, grid_follow_frames = _grid_frame_groups(frame, grid_frames)
        base_card = (
            event["card"][4:]
            if event["card"].startswith("evo-")
            else event["card"]
        )
        kind = CARD_METADATA[base_card]["kind"]
        grid_purpose = "grid_full" if kind == "spell" and base_card != "log" else "grid_own"
        grid_paths = []
        for frames, path in (
            (grid_onset_frames, grid_onset_path),
            (grid_follow_frames, grid_follow_path),
        ):
            if not frames:
                continue
            _render_sheet(
                run_dir=run_dir,
                manifest=manifest,
                frame_indices=frames,
                purpose=grid_purpose,
                output=path,
            )
            grid_paths.append(str(path.relative_to(run_dir)))
        targets.append(
            {
                "event_id": event_id,
                "candidate_id": event["candidate_id"],
                "card": event["card"],
                "elixir_cost": CARD_METADATA[
                    event["card"][4:] if event["card"].startswith("evo-") else event["card"]
                ]["elixir_cost"],
                "event_frame_index": frame,
                "review_frame_indices": sorted(set(macro_frames) | set(grid_frames)),
                "location_rule_options": _rule_options(event["card"]),
                "macro_review_artifacts": [str(macro_path.relative_to(run_dir))],
                "grid_review_artifacts": grid_paths,
                "grid_scope": "full_arena" if grid_purpose == "grid_full" else "rows_14_31",
            }
        )

    packages = []
    isolated_packages = []
    for offset in range(0, len(targets), chunk_size):
        chunk = targets[offset : offset + chunk_size]
        package = {
            "run_id": manifest["run_id"],
            "stage": "own_localization_targets",
            "evidence_version": 2,
            "target_range": [
                min(row["event_frame_index"] for row in chunk),
                max(row["event_frame_index"] for row in chunk) + 1,
            ],
            "grid": {
                "columns": ACTION_GRID.cols,
                "rows": ACTION_GRID.rows,
                "cell_order": "[column,row]",
                "origin": "top-left",
            },
            "targets": chunk,
            "attached_images": [
                artifact
                for row in chunk
                for key in ("macro_review_artifacts", "grid_review_artifacts")
                for artifact in row[key]
            ],
        }
        path = package_dir / f"own-localization-{offset:03d}-{offset + len(chunk):03d}.json"
        atomic_write_json(path, package)
        packages.append(str(path.relative_to(run_dir)))
        isolated_dir = output_dir / "isolated" / path.stem
        isolated_reviews = isolated_dir / "reviews"
        isolated_reviews.mkdir(parents=True, exist_ok=True)
        isolated = deepcopy(package)
        for target in isolated["targets"]:
            for key in ("macro_review_artifacts", "grid_review_artifacts"):
                copied = []
                for artifact in target[key]:
                    source_artifact = run_dir / artifact
                    destination = isolated_reviews / source_artifact.name
                    shutil.copyfile(source_artifact, destination)
                    copied.append(f"reviews/{destination.name}")
                target[key] = copied
        isolated["attached_images"] = [
            artifact
            for row in isolated["targets"]
            for key in ("macro_review_artifacts", "grid_review_artifacts")
            for artifact in row[key]
        ]
        isolated_path = isolated_dir / "package.json"
        atomic_write_json(isolated_path, isolated)
        isolated_packages.append(str(isolated_path.relative_to(run_dir)))
    index = {
        "run_id": manifest["run_id"],
        "stage": "own_localization_package_index",
        "evidence_version": 2,
        "source_file": str(source_file.resolve()),
        "target_count": len(targets),
        "chunk_size": chunk_size,
        "packages": packages,
        "isolated_packages": isolated_packages,
    }
    atomic_write_json(output_dir / "package_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare blind own-event location packages.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--event-id", action="append", default=[])
    args = parser.parse_args()
    index = prepare_packages(
        run_dir=args.run_dir,
        source_file=args.source_file.resolve(),
        output_dir=args.output_dir.resolve(),
        chunk_size=args.chunk_size,
        event_ids=set(args.event_id) or None,
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
