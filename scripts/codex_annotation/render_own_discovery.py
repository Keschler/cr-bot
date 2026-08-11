from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


FORMAT_VERSION = 4


def _candidate_groups(
    candidates: list[dict],
    *,
    segment_start: int,
    segment_end: int,
    before_frames: int,
    after_frames: int,
    max_group_span_frames: int,
) -> list[tuple[int, int, list[dict]]]:
    """Coalesce overlapping candidate timelines without creating huge sheets."""

    groups: list[tuple[int, int, list[dict]]] = []
    for candidate in sorted(
        candidates, key=lambda row: row["approximate_frame_index"]
    ):
        approximate = int(candidate["approximate_frame_index"])
        start = max(segment_start, approximate - before_frames)
        end = min(segment_end - 1, approximate + after_frames)
        if groups:
            group_start, group_end, rows = groups[-1]
            proposed_end = max(group_end, end)
            if (
                start <= group_end + 1
                and proposed_end - group_start + 1 <= max_group_span_frames
            ):
                groups[-1] = (
                    group_start,
                    proposed_end,
                    [*rows, candidate],
                )
                continue
        groups.append((start, end, [candidate]))
    return groups


def _scaled_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(
        image,
        (width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _arena_tile(content: np.ndarray, *, frame_index: int, fps: float) -> np.ndarray:
    arena = content[
        ROIS["battlefield"][1] : (
            ROIS["battlefield"][1] + ROIS["battlefield"][3]
        ),
        ROIS["battlefield"][0] : (
            ROIS["battlefield"][0] + ROIS["battlefield"][2]
        ),
    ]
    arena = _scaled_width(arena, 240)
    return add_frame_identity(
        arena,
        source_frame_index=frame_index,
        fps=fps,
    )


def _hud_tile(content: np.ndarray, *, frame_index: int, fps: float) -> np.ndarray:
    # Preserve the complete hand, next-card slot, and elixir digit/bar at a
    # larger scale than the arena. This is where compensated spends and
    # canceled drags are distinguishable.
    hud = content[1900:2400, 30:1040]
    hud = _scaled_width(hud, 360)
    return add_frame_identity(
        hud,
        source_frame_index=frame_index,
        fps=fps,
    )


def _pad_width(image: np.ndarray, width: int) -> np.ndarray:
    left = (width - image.shape[1]) // 2
    return cv2.copyMakeBorder(
        image,
        0,
        0,
        left,
        width - image.shape[1] - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render sparse ±window own-HUD discovery timelines."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--before-frames", type=int, default=16)
    parser.add_argument("--after-frames", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=2)
    parser.add_argument("--max-group-span-frames", type=int, default=60)
    args = parser.parse_args()
    if (
        args.before_frames < 1
        or args.after_frames < 1
        or args.sample_step < 1
        or args.max_group_span_frames < 1
    ):
        parser.error("discovery horizons and sample step must be positive")

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = {
        row["source_frame_index"]: row for row in manifest["frames"]
    }
    segment = manifest["segment"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    index_rows = []
    groups = _candidate_groups(
        manifest["candidate_discovery"]["own_candidates"],
        segment_start=int(segment["start_frame"]),
        segment_end=int(segment["end_frame_exclusive"]),
        before_frames=args.before_frames,
        after_frames=args.after_frames,
        max_group_span_frames=args.max_group_span_frames,
    )
    for start, end, candidates in groups:
        approximate_frames = {
            int(row["approximate_frame_index"]) for row in candidates
        }
        indices = sorted(
            {
                *range(start, end + 1, args.sample_step),
                *approximate_frames,
                end,
            }
        )
        hud_tiles = []
        arena_tiles = []
        arena_indices = sorted({*indices[::3], *approximate_frames, end})
        for frame_index in indices:
            record = frames[frame_index]
            labeled = cv2.imread(str(run_dir / record["path"]))
            if labeled is None:
                raise FileNotFoundError(run_dir / record["path"])
            content = labeled[manifest["label_margin_px"] :]
            hud_tiles.append(
                _hud_tile(
                    content,
                    frame_index=frame_index,
                    fps=float(manifest["fps"]),
                )
            )
            if frame_index in arena_indices:
                arena_tiles.append(
                    _arena_tile(
                        content,
                        frame_index=frame_index,
                        fps=float(manifest["fps"]),
                    )
                )
        first_approximate = min(approximate_frames)
        last_approximate = max(approximate_frames)
        output = args.output_dir / (
            f"discover-own-group-{first_approximate:06d}-"
            f"{last_approximate:06d}.jpg"
        )
        hud_sheet = _make_contact_sheet(hud_tiles, columns=4)
        arena_sheet = _make_contact_sheet(arena_tiles, columns=4)
        width = max(hud_sheet.shape[1], arena_sheet.shape[1])
        sheet = np.vstack(
            [_pad_width(hud_sheet, width), _pad_width(arena_sheet, width)]
        )
        if sheet.shape[0] * sheet.shape[1] > REVIEW_MAX_PIXELS:
            raise ValueError(
                f"own discovery group {first_approximate}-{last_approximate} "
                "is too large"
            )
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"failed to write {output}")
        record_review(
            run_dir=run_dir,
            output_path=output,
            purpose="own_discovery",
            start_frame=min(indices),
            end_frame=max(indices) + 1,
            candidate_id=None,
            event_id=None,
        )
        for candidate in candidates:
            index_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "approximate_frame_index": candidate[
                        "approximate_frame_index"
                    ],
                    "sampled_frame_indices": indices,
                    "artifact": f"reviews/{output.name}",
                }
            )
        rendered += 1
    atomic_write_json(
        run_dir / "own_discovery_index.json",
        {
            "run_id": manifest["run_id"],
            "format_version": FORMAT_VERSION,
            "before_frames": args.before_frames,
            "after_frames": args.after_frames,
            "sample_step": args.sample_step,
            "max_group_span_frames": args.max_group_span_frames,
            "artifact_count": rendered,
            "candidates": index_rows,
        },
    )
    print(json.dumps({"rendered": rendered, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
