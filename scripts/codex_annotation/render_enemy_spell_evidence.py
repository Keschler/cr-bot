from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json, render_review_sheet


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render exact, side-aware before/after evidence for broad enemy-spell "
            "proposals plus an end-of-segment recall sentinel."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    document = _read(run_dir / "enemy_onsets.json")
    if document.get("run_id") != manifest.get("run_id"):
        raise ValueError("enemy onsets do not match manifest")
    segment_start = int(manifest["segment"]["start_frame"])
    segment_end = int(manifest["segment"]["end_frame_exclusive"])

    proposals = [
        {
            "review_id": f"spell-review:{onset['onset_id']}",
            "source_onset_id": onset["onset_id"],
            "source_candidate_id": onset.get("candidate_id"),
            "proposal_frame_index": int(onset["event_frame_index"]),
            "segment_end_sentinel": False,
        }
        for onset in document.get("onsets", [])
        if isinstance(onset, dict) and onset.get("kind") == "spell"
    ]
    # Broad windows cannot look beyond their owned segment.  Always inspect the
    # final 1.3 seconds so a projectile first visible on the last frame remains
    # recallable even though no forward-resolution frames exist.
    sentinel_frame = segment_end - 1
    proposals.append(
        {
            "review_id": f"spell-review:segment-end-{segment_end:06d}",
            "source_onset_id": None,
            "source_candidate_id": None,
            "proposal_frame_index": sentinel_frame,
            "segment_end_sentinel": True,
        }
    )

    reviews = []
    for proposal in proposals:
        frame = proposal["proposal_frame_index"]
        name = _safe_name(proposal["review_id"])
        before_start = max(segment_start, frame - 12)
        before_end = min(segment_end, frame + 2)
        after_start = max(segment_start, frame - 1)
        after_end = min(segment_end, frame + 31)
        before = run_dir / "reviews" / f"{name}-before.jpg"
        after_early = run_dir / "reviews" / f"{name}-after-early.jpg"
        after_late = run_dir / "reviews" / f"{name}-after-late.jpg"
        render_review_sheet(
            run_dir=run_dir,
            output_path=before,
            start_frame=before_start,
            end_frame=before_end,
            candidate_id=None,
            event_id=None,
            purpose="arena",
            columns=4,
            tile_width=360,
        )
        # Arena sheets deliberately cap each image at 20 frames so the tiles
        # remain readable. Keep the full three-second horizon, split across
        # two chronological after-sheets.
        after_split = min(after_end, after_start + 16)
        render_review_sheet(
            run_dir=run_dir,
            output_path=after_early,
            start_frame=after_start,
            end_frame=after_split,
            candidate_id=None,
            event_id=None,
            purpose="arena",
            columns=4,
            tile_width=360,
        )
        confirmation_artifacts = [
            str(before.relative_to(run_dir)),
            str(after_early.relative_to(run_dir)),
        ]
        if after_split < after_end:
            render_review_sheet(
                run_dir=run_dir,
                output_path=after_late,
                start_frame=after_split,
                end_frame=after_end,
                candidate_id=None,
                event_id=None,
                purpose="arena",
                columns=4,
                tile_width=360,
            )
            confirmation_artifacts.append(
                str(after_late.relative_to(run_dir))
            )
        sampled = sorted(
            set(range(before_start, before_end))
            | set(range(after_start, after_end))
        )
        reviews.append(
            {
                **proposal,
                "inspection_range": [sampled[0], sampled[-1] + 1],
                "sampled_frame_indices": sampled,
                "confirmation_artifacts": confirmation_artifacts,
            }
        )

    output = {
        "run_id": manifest["run_id"],
        "stage": "enemy_spell_confirmation_candidates",
        "reviews": reviews,
    }
    path = run_dir / "enemy_spell_confirmation_candidates.json"
    atomic_write_json(path, output)
    print(json.dumps({"output": str(path), "reviews": len(reviews)}))


if __name__ == "__main__":
    main()
