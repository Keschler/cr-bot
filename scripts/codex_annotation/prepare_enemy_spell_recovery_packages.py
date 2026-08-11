from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json, render_review_sheet
from cr_bot.domain.card_metadata import CARD_METADATA


OWN_CONFOUND_TOLERANCE = 8
UNIT_CONFOUND_TOLERANCE = 5
RECOVERY_DELAY = 8
PROPOSAL_RECOVERY_DELAY = 15
RECOVERY_FRAMES = 12


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _nearest_frame(
    frame: int,
    rows: list[dict[str, Any]],
    *,
    tolerance: int,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if abs(frame - int(row["event_frame_index"])) <= tolerance
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            abs(frame - int(row["event_frame_index"])),
            int(row["event_frame_index"]),
        ),
    )


def _render_lane_recovery(
    run_dir: Path,
    *,
    review_id: str,
    start: int,
    end: int,
    lane: str,
) -> tuple[list[int], list[str]]:
    lane_x = {"left": 3, "right": 14}[lane]
    artifacts = []
    sampled = []
    for window_index, window_start in enumerate(range(start, end, 6)):
        window_end = min(end, window_start + 6)
        sampled.extend(range(window_start, window_end))
        # The first pixels of a downward enemy spell are beside the upper
        # red tower, not at the bridge. A second, slightly lower crop proves
        # continuation while keeping each review small enough for reliable
        # immediate-predecessor comparison.
        center = (lane_x, 6 + 2 * window_index)
        path = (
            run_dir
            / "reviews"
            / (
                f"{review_id.replace(':', '-')}-{window_start:06d}-"
                f"{window_end:06d}-{lane}.jpg"
            )
        )
        render_review_sheet(
            run_dir=run_dir,
            output_path=path,
            start_frame=window_start,
            end_frame=window_end,
            candidate_id=None,
            event_id=review_id,
            purpose="identity",
            columns=3,
            tile_width=720,
            focus_cell=center,
            focus_radius=5,
        )
        artifacts.append(str(path.relative_to(run_dir)))
    return sorted(set(sampled)), artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify broad spell proposals by independent own/unit confounds "
            "and render focused searches for a later additional enemy spell."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=200)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    own = _read(run_dir / "own_semantics.json")
    onsets = _read(run_dir / "enemy_onsets.json")
    identities = _read(run_dir / "enemy_identities.json")
    if any(
        document.get("run_id") != manifest["run_id"]
        for document in (own, onsets, identities)
    ):
        raise ValueError("spell reconciliation inputs do not share a run_id")
    segment = manifest["segment"]
    segment_start = int(segment["start_frame"])
    segment_end = int(segment["end_frame_exclusive"])
    own_spells = [
        row
        for row in own.get("events", [])
        if isinstance(row, dict)
        and CARD_METADATA.get(row.get("card"), {}).get("kind") == "spell"
    ]
    onset_by_id = {
        row["onset_id"]: row
        for row in onsets.get("onsets", [])
        if isinstance(row, dict)
    }
    accepted_units = [
        {
            "onset_id": row["onset_id"],
            "event_frame_index": row["event_frame_index"],
        }
        for row in identities.get("decisions", [])
        if isinstance(row, dict)
        and row.get("event_exists") is True
        and onset_by_id.get(row.get("onset_id"), {}).get("kind")
        == "unit_or_building"
    ]
    classifications = []
    recovery_reviews = []
    for onset in onsets.get("onsets", []):
        if not isinstance(onset, dict) or onset.get("kind") != "spell":
            continue
        frame = int(onset["event_frame_index"])
        own_match = _nearest_frame(
            frame,
            own_spells,
            tolerance=OWN_CONFOUND_TOLERANCE,
        )
        unit_match = _nearest_frame(
            frame,
            accepted_units,
            tolerance=UNIT_CONFOUND_TOLERANCE,
        )
        if unit_match is not None:
            # A marker-confirmed actor explains the broad visual effect. Do not
            # ask a spell worker to reinterpret its spawn/ability sequence.
            classification = "unit_confound"
        elif own_match is not None:
            classification = "own_spell_confound"
            start = max(
                segment_start,
                int(own_match["event_frame_index"]) + RECOVERY_DELAY,
                frame + PROPOSAL_RECOVERY_DELAY,
            )
            end = min(segment_end, start + RECOVERY_FRAMES)
            for lane in ("left", "right"):
                review_id = (
                    f"spell-recovery:{onset['onset_id']}:{lane}"
                )
                sampled, artifacts = _render_lane_recovery(
                    run_dir,
                    review_id=review_id,
                    start=start,
                    end=end,
                    lane=lane,
                )
                recovery_reviews.append(
                    {
                        "review_id": review_id,
                        "source_onset_id": onset["onset_id"],
                        "source_candidate_id": onset.get("candidate_id"),
                        "proposal_frame_index": frame,
                        "segment_end_sentinel": False,
                        "lane": lane,
                        "inspection_range": [start, end],
                        "sampled_frame_indices": sampled,
                        "confirmation_artifacts": artifacts,
                        "known_own_spell": {
                            "card": own_match["card"],
                            "event_frame_index": own_match[
                                "event_frame_index"
                            ],
                        },
                    }
                )
        else:
            classification = "unconfounded_broad_spell"
        classifications.append(
            {
                "onset_id": onset["onset_id"],
                "event_frame_index": frame,
                "classification": classification,
                "own_spell_event_frame_index": (
                    own_match["event_frame_index"]
                    if own_match is not None
                    else None
                ),
                "unit_event_frame_index": (
                    unit_match["event_frame_index"]
                    if unit_match is not None
                    else None
                ),
            }
        )
    source = {
        "run_id": manifest["run_id"],
        "stage": "enemy_spell_reconciliation_candidates",
        "classifications": classifications,
        "reviews": recovery_reviews,
    }
    atomic_write_json(
        run_dir / "enemy_spell_reconciliation_candidates.json",
        source,
    )
    package_dir = run_dir / "work_packages"
    package_summaries = []
    for start in range(segment_start, segment_end, args.chunk_frames):
        end = min(segment_end, start + args.chunk_frames)
        for lane in ("left", "right"):
            reviews = [
                row
                for row in recovery_reviews
                if row["lane"] == lane
                and start <= int(row["proposal_frame_index"]) < end
            ]
            path = (
                package_dir
                / (
                    f"enemy-spell-recovery-{lane}-{start:06d}-"
                    f"{end:06d}.json"
                )
            )
            atomic_write_json(
                path,
                {
                    "run_id": manifest["run_id"],
                    "stage": "enemy_spell_confirmation_package",
                    "target_range": [start, end],
                    "segment": segment,
                    "task": "additional_spell_after_known_own_spell",
                    "reviews": reviews,
                },
            )
            package_summaries.append(
                {
                    "target_range": [start, end],
                    "lane": lane,
                    "reviews": len(reviews),
                }
            )
    sentinel_frame = segment_end - 1
    sentinel_start = max(segment_start, sentinel_frame - 12)
    sentinel_path = run_dir / "reviews" / "spell-boundary-sentinel.jpg"
    render_review_sheet(
        run_dir=run_dir,
        output_path=sentinel_path,
        start_frame=sentinel_start,
        end_frame=segment_end,
        candidate_id=None,
        event_id=None,
        purpose="arena",
        columns=4,
        tile_width=360,
    )
    boundary_start = (
        segment_start
        + ((sentinel_frame - segment_start) // args.chunk_frames)
        * args.chunk_frames
    )
    boundary_package = (
        package_dir
        / (
            f"enemy-spell-boundary-{boundary_start:06d}-"
            f"{segment_end:06d}.json"
        )
    )
    atomic_write_json(
        boundary_package,
        {
            "run_id": manifest["run_id"],
            "stage": "enemy_spell_confirmation_package",
            "target_range": [boundary_start, segment_end],
            "segment": segment,
            "task": "segment_end_spell_sentinel",
            "reviews": [
                {
                    "review_id": (
                        f"spell-review:segment-end-{segment_end:06d}"
                    ),
                    "source_onset_id": None,
                    "source_candidate_id": None,
                    "proposal_frame_index": sentinel_frame,
                    "segment_end_sentinel": True,
                    "inspection_range": [sentinel_start, segment_end],
                    "sampled_frame_indices": list(
                        range(sentinel_start, segment_end)
                    ),
                    "confirmation_artifacts": [
                        str(sentinel_path.relative_to(run_dir))
                    ],
                }
            ],
        },
    )
    print(
        json.dumps(
            {
                "classifications": classifications,
                "recovery_reviews": len(recovery_reviews),
                "packages": package_summaries,
                "boundary_package": str(boundary_package),
            }
        )
    )


if __name__ == "__main__":
    main()
