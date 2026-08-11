from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json, render_review_sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render canonical exact and post-release own evidence after the "
            "deterministic proposal union, without model renderer tool calls."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    document = json.loads(
        (run_dir / "own_semantics.json").read_text(encoding="utf-8")
    )
    start = int(manifest["segment"]["start_frame"])
    end = int(manifest["segment"]["end_frame_exclusive"])
    rendered_exact: dict[int, str] = {}
    rendered = 0
    for event in document["events"]:
        frame = int(event["event_frame_index"])
        candidate_id = event["candidate_id"]
        suffix = candidate_id.replace(":", "-")
        if frame + 5 >= end:
            raise ValueError(
                f"{candidate_id}: event is too close to the segment end for "
                "post-release confirmation"
            )
        exact_artifact = rendered_exact.get(frame)
        if exact_artifact is None:
            exact_output = run_dir / "reviews" / f"evidence-own-{frame:06d}.jpg"
            render_review_sheet(
                run_dir=run_dir,
                output_path=exact_output,
                start_frame=max(start, frame - 1),
                end_frame=min(end, frame + 3),
                candidate_id=None,
                event_id=None,
                purpose="own_context",
                columns=4,
                tile_width=480,
            )
            exact_artifact = f"reviews/{exact_output.name}"
            rendered_exact[frame] = exact_artifact
            rendered += 1
        confirmation_start = frame + 5
        confirmation_end = min(end, frame + 13)
        confirmation_output = run_dir / "reviews" / (
            f"release-{suffix}-event-{frame:06d}.jpg"
        )
        render_review_sheet(
            run_dir=run_dir,
            output_path=confirmation_output,
            start_frame=confirmation_start,
            end_frame=confirmation_end,
            candidate_id=None,
            event_id=f"release-{suffix}",
            purpose="own_confirmation",
            columns=4,
            tile_width=420,
        )
        event["verification_artifacts"] = [exact_artifact]
        event["confirmation_frame_index"] = confirmation_start
        event["confirmation_artifacts"] = [
            f"reviews/{confirmation_output.name}"
        ]
        rendered += 1
    atomic_write_json(run_dir / "own_semantics.json", document)
    print(json.dumps({"events": len(document["events"]), "rendered": rendered}))


if __name__ == "__main__":
    main()
