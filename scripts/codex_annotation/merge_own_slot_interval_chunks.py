from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts" / "codex_annotation"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import validate_own_slot_interval_decisions
from cr_bot.domain.card_metadata import CARD_METADATA


PACKAGE_PATTERN = re.compile(r"own-slot-\d{6}-\d{6}\.json")
EVIDENCE = {
    "elixir_drop": True,
    "hand_transition": True,
    "deployment_onset": True,
    "first_visible_object": True,
    "side_direction": None,
    "impact_sequence": None,
}
OWN_CONFIRMATION = {
    "release_confirmed": True,
    "elixir_spend_persisted": True,
    "hand_cycle_completed": True,
    "post_release_effect": True,
}


def select_elixir_onset(
    interval: dict[str, Any],
    *,
    card: str,
    fallback_event_frame: int,
) -> int:
    transitions = interval.get("elixir_drop_transitions")
    if not isinstance(transitions, list) or not transitions:
        return min(
            int(interval["empty_range"][1]),
            max(int(interval["empty_range"][0]) + 1, fallback_event_frame),
        )
    cost = int(CARD_METADATA[card]["elixir_cost"])
    candidates: list[tuple[tuple[int, int, int], int]] = []
    for left in range(len(transitions)):
        total = 0
        first_frame = int(transitions[left]["frame_index"])
        for right in range(left, min(len(transitions), left + 3)):
            frame = int(transitions[right]["frame_index"])
            if frame - first_frame > 4:
                break
            total += int(transitions[right]["drop"])
            candidates.append(
                (
                    (
                        abs(total - cost),
                        frame - first_frame,
                        first_frame,
                    ),
                    first_frame,
                )
            )
    score, frame = min(candidates)
    # Large mismatches usually belong to an overlapping later play; retain the
    # visually selected fallback instead of inventing a meter-based onset.
    return fallback_event_frame if score[0] > 1 else frame


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _package_paths(run_dir: Path) -> list[Path]:
    package_dir = run_dir / "work_packages"
    return sorted(
        path
        for path in package_dir.iterdir()
        if PACKAGE_PATTERN.fullmatch(path.name)
    )


def deduplicate_exact_own_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Collapse only semantically identical events from overlapping intervals.

    Own-slot intervals may overlap at chunk or tracking boundaries.  When two
    independently valid intervals resolve to the same card and meter-aligned
    frame, retaining both would duplicate one play.  The
    evidence representative is selected deterministically without consulting
    labels, evaluation results, or any other worker output.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            str(event["card"]),
            int(event["event_frame_index"]),
        )
        grouped.setdefault(key, []).append(event)
    deduplicated = [
        min(
            rows,
            key=lambda row: (
                int(row["confirmation_frame_index"]),
                str(row["candidate_id"]),
                tuple(row["verification_artifacts"]),
                tuple(row["confirmation_artifacts"]),
            ),
        )
        for rows in grouped.values()
    ]
    return deduplicated, len(events) - len(deduplicated)


def select_confirmation_frame(
    interval: dict[str, Any],
    *,
    event_frame: int,
    segment_end: int,
) -> int:
    """Select the earliest valid post-release frame, extending when needed.

    Elixir transitions are intentionally searched for several frames beyond
    the empty interval. If alignment lands there, the original interval sheet
    may stop before ``event+5`` even though the source segment has ample
    confirmation horizon. The merger can render that deterministic source
    frame directly; no additional semantic judgment is needed.
    """

    sampled = set(int(value) for value in interval["sampled_frame_indices"])
    candidates = sorted(
        frame
        for frame in sampled
        if event_frame + 5 <= frame <= event_frame + 15
    )
    if candidates:
        return candidates[0]
    fallback = event_frame + 5
    if fallback < segment_end:
        return fallback
    raise ValueError(
        f"{interval['interval_id']}: no post-release confirmation horizon "
        f"after meter-aligned frame {event_frame}"
    )


def _event(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    interval: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    card = str(decision["card"])
    event_frame = select_elixir_onset(
        interval,
        card=card,
        fallback_event_frame=int(decision["event_frame_index"]),
    )
    # Slot intervals are an independent completeness lane. A card may remain
    # selected for an arbitrary time before release, so attaching its verified
    # elixir onset to the nearest frame-local HUD-change peak creates a false
    # temporal association. Preserve the interval itself as the candidate.
    interval_token = str(interval["interval_id"]).replace(
        "own-slot:", "slot-", 1
    ).replace(":", "-")
    candidate_id = f"completeness:own:{interval_token}"
    artifact = decision["artifact"]
    confirmation_frame = select_confirmation_frame(
        interval,
        event_frame=event_frame,
        segment_end=int(manifest["segment"]["end_frame_exclusive"]),
    )
    if not (run_dir / "review_index.json").is_file():
        confirmation_artifact = artifact
    else:
        release_event_id = f"release-{candidate_id.replace(':', '-')}"
        confirmation_path = run_dir / "reviews" / (
            f"own-slot-confirm-{candidate_id.replace(':', '-')}-"
            f"{event_frame:06d}.jpg"
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "render_annotation_review.py"),
                "--run-dir",
                str(run_dir),
                "--event-id",
                release_event_id,
                "--start-frame",
                str(event_frame),
                "--end-frame",
                str(confirmation_frame + 1),
                "--purpose",
                "own_confirmation",
                "--columns",
                "4",
                "--tile-width",
                "480",
                "--output",
                str(confirmation_path),
            ],
            cwd=ROOT,
            check=True,
        )
        confirmation_artifact = str(confirmation_path.relative_to(run_dir))
    return {
        "candidate_id": candidate_id,
        "card": card,
        "event_frame_index": event_frame,
        "evidence": dict(EVIDENCE),
        "verification_artifacts": [artifact],
        "confirmation_frame_index": confirmation_frame,
        "confirmation_artifacts": [confirmation_artifact],
        "own_confirmation": dict(OWN_CONFIRMATION),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and merge deterministic own-slot interval judgments into "
            "own_semantics.json."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-output-suffix",
        default="",
        help=(
            "Optional suffix appended to each package filename when reading "
            "worker outputs. This supports explicitly validated cached blind "
            "results without promoting them to current pipeline state."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    run_id = manifest["run_id"]
    package_paths = _package_paths(run_dir)
    if not package_paths:
        raise ValueError("no own-slot work packages")

    events: list[dict[str, Any]] = []
    declined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    accepted_source_candidates: set[str] = set()
    provenance: list[dict[str, Any]] = []
    seen_intervals: set[str] = set()
    for package_path in package_paths:
        package = _read(package_path)
        output_path = (
            run_dir
            / "worker_outputs"
            / f"{package_path.name}{args.worker_output_suffix}"
        )
        if not output_path.is_file():
            raise ValueError(f"missing own-slot worker output {output_path}")
        document = _read(output_path)
        validate_own_slot_interval_decisions(document, package)
        if document.get("run_id") != run_id:
            raise ValueError(f"{output_path}: run_id does not match manifest")

        intervals = {
            row["interval_id"]: row for row in package["intervals"]
        }
        for decision in document["decisions"]:
            interval_id = decision["interval_id"]
            if interval_id in seen_intervals:
                raise ValueError(f"duplicate own-slot interval {interval_id}")
            seen_intervals.add(interval_id)
            interval = intervals[interval_id]
            if decision["decision"] == "released":
                source_candidate_id = interval.get("candidate_id")
                if isinstance(source_candidate_id, str):
                    accepted_source_candidates.add(source_candidate_id)
                events.append(
                    _event(
                        run_dir=run_dir,
                        manifest=manifest,
                        interval=interval,
                        decision=decision,
                    )
                )
            else:
                declined.append((interval, decision))
        provenance.append(
            {
                key: document.get(key)
                for key in (
                    "target_range",
                    "annotation_session_id",
                    "model",
                    "reasoning_effort",
                )
            }
        )

    events, duplicate_events = deduplicate_exact_own_events(events)
    accepted_candidates = accepted_source_candidates | {
        row["candidate_id"] for row in events
    }
    rejected_by_candidate: dict[str, dict[str, str]] = {}
    for interval, decision in declined:
        candidate_id = interval.get("candidate_id")
        if (
            isinstance(candidate_id, str)
            and candidate_id
            and candidate_id not in accepted_candidates
        ):
            rejected_by_candidate[candidate_id] = {
                "candidate_id": candidate_id,
                "reason": (
                    f"own_slot_{decision['decision']}: {decision['reason']}"
                ),
            }

    events.sort(
        key=lambda row: (
            row["event_frame_index"],
            row["card"],
            row["candidate_id"],
        )
    )
    output = {
        "run_id": run_id,
        "stage": "own_semantics",
        "annotation_session_id": "merged-own-slot-workers",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": provenance,
        "events": events,
        "rejected_candidates": sorted(
            rejected_by_candidate.values(),
            key=lambda row: row["candidate_id"],
        ),
        "pending_resolutions": [],
    }
    atomic_write_json(run_dir / "own_semantics.json", output)
    print(
        json.dumps(
            {
                "output": str(run_dir / "own_semantics.json"),
                "intervals": len(seen_intervals),
                "events": len(events),
                "declined": len(declined),
                "deduplicated_exact_events": duplicate_events,
            }
        )
    )


if __name__ == "__main__":
    main()
