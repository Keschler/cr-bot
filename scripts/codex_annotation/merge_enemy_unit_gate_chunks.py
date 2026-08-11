from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import (
    validate_enemy_existence_decisions,
    validate_enemy_side_check_decisions,
)

RECOVERY_TOLERANCE_FRAMES = 5


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_chunks(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
    pattern: re.Pattern[str],
    expected_stage: str,
    validator,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    packages = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if pattern.fullmatch(path.name)
    )
    if not packages:
        raise ValueError(f"no packages for stage {expected_stage}")
    segment = manifest["segment"]
    expected_ranges = []
    chunk_start = int(segment["start_frame"])
    ranges = []
    rows = []
    for package_path in packages:
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing worker output {output_path}")
        package = _read(package_path)
        document = _read(output_path)
        if package.get("run_id") != manifest["run_id"]:
            raise ValueError(f"{package_path}: run_id mismatch")
        if document.get("stage") != expected_stage:
            raise ValueError(f"{output_path}: wrong stage")
        if document.get("run_id") != manifest["run_id"]:
            raise ValueError(f"{output_path}: run_id mismatch")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        start, end = package["target_range"]
        ranges.append((int(start), int(end)))
        validator(document, package)
        rows.append((package, document))
    for start, end in sorted(ranges):
        if start != chunk_start or end <= start:
            raise ValueError(
                f"{expected_stage} packages are stale, overlapping, or gapped"
            )
        expected_ranges.append((start, end))
        chunk_start = end
    if chunk_start != int(segment["end_frame_exclusive"]):
        raise ValueError(
            f"{expected_stage} packages do not cover the full segment"
        )
    return rows


def _unique_rows(
    chunks: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    key: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _, document in chunks:
        for row in document[key]:
            onset_id = row["onset_id"]
            if onset_id in result:
                raise ValueError(f"duplicate cross-package onset {onset_id}")
            result[onset_id] = row
    return result


def _novel_simultaneous_recoveries(
    recoveries: list[dict[str, Any]],
    *,
    accepted_enemy_frames: list[int],
    tolerance_frames: int = RECOVERY_TOLERANCE_FRAMES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject a recovery when it only rediscovers an accepted enemy event."""
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    known_frames = list(accepted_enemy_frames)
    for recovery in sorted(
        recoveries, key=lambda row: int(row["event_frame_index"])
    ):
        frame = int(recovery["event_frame_index"])
        if any(abs(frame - known) <= tolerance_frames for known in known_frames):
            duplicates.append(recovery)
            continue
        kept.append(recovery)
        known_frames.append(frame)
    return kept, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge sequence-aware Sol existence and independent Terra side checks."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    onset_document = _read(run_dir / "enemy_onsets.json")
    # A paused pipeline may rerun deterministic merge stages. Reconstruct
    # simultaneous recoveries from their standalone source on every pass
    # instead of treating a previously materialized recovery as a new base
    # onset and then colliding with the same stable recovery ID.
    onsets = [
        row
        for row in onset_document["onsets"]
        if not str(row.get("onset_id", "")).startswith(
            "enemy-simultaneous-unit-"
        )
    ]
    onset_document["onsets"] = onsets
    onsets_by_id = {row["onset_id"]: row for row in onsets}
    if len(onsets_by_id) != len(onsets):
        raise ValueError("enemy onsets contain duplicate IDs")

    existence_chunks = _load_chunks(
        run_dir,
        manifest=manifest,
        pattern=re.compile(r"identity-overlap-\d{6}-\d{6}\.json"),
        expected_stage="enemy_overlap_adjudication_chunk",
        validator=validate_enemy_existence_decisions,
    )
    existence_rows = _unique_rows(existence_chunks, key="decisions")
    accepted_existence = {
        onset_id: row
        for onset_id, row in existence_rows.items()
        if row["overlap_event_exists"]
    }
    side_chunks = _load_chunks(
        run_dir,
        manifest=manifest,
        pattern=re.compile(r"identity-side-\d{6}-\d{6}\.json"),
        expected_stage="enemy_side_check_chunk",
        validator=validate_enemy_side_check_decisions,
    )
    side_rows = _unique_rows(side_chunks, key="decisions")
    side_escalation_path = (
        run_dir / "recovery_outputs" / "enemy_side_escalations.json"
    )
    side_escalation_provenance: list[dict[str, Any]] = []
    if side_escalation_path.is_file():
        escalation = _read(side_escalation_path)
        if escalation.get("run_id") != manifest["run_id"]:
            raise ValueError("enemy side escalation run_id mismatch")
        if escalation.get("stage") != "enemy_side_escalations":
            raise ValueError("enemy side escalation has wrong stage")
        escalated_ids: set[str] = set()
        for row in escalation.get("decisions", []):
            if not isinstance(row, dict):
                raise ValueError("enemy side escalation decision is invalid")
            onset_id = row.get("onset_id")
            if (
                not isinstance(onset_id, str)
                or onset_id in escalated_ids
                or onset_id not in side_rows
            ):
                raise ValueError("invalid enemy side escalation onset")
            if side_rows[onset_id]["side"] != "unresolved":
                raise ValueError(
                    f"{onset_id}: escalation may only replace unresolved side"
                )
            side_rows[onset_id] = row
            escalated_ids.add(onset_id)
        raw_provenance = escalation.get("worker_provenance", [])
        if isinstance(raw_provenance, list):
            side_escalation_provenance = [
                row for row in raw_provenance if isinstance(row, dict)
            ]
    side_candidates: dict[str, dict[str, Any]] = {}
    for package, _ in side_chunks:
        for row in package["candidates"]:
            onset_id = row["onset_id"]
            if onset_id in side_candidates:
                raise ValueError(
                    f"duplicate cross-package side candidate {onset_id}"
                )
            side_candidates[onset_id] = row
    if set(side_rows) != set(side_candidates):
        raise ValueError("side outputs do not cover deduplicated candidates")
    if set(side_candidates) != set(accepted_existence):
        raise ValueError(
            "side packages do not cover accepted existence rows exactly"
        )
    if any(
        row.get("source_onset_ids") != [onset_id]
        for onset_id, row in side_candidates.items()
    ):
        raise ValueError("pre-side candidates must not be deduplicated")

    grouped: list[list[dict[str, Any]]] = []
    for onset_id, candidate in sorted(
        side_candidates.items(),
        key=lambda item: (
            side_rows[item[0]]["side"],
            item[1]["event_frame_index"],
        ),
    ):
        side = side_rows[onset_id]["side"]
        row = {**candidate, "side": side}
        if (
            not grouped
            or grouped[-1][0]["side"] != side
            or row["event_frame_index"]
            - grouped[-1][0]["event_frame_index"]
            > 5
        ):
            grouped.append([row])
        else:
            grouped[-1].append(row)
    deduplicated_candidates: dict[str, dict[str, Any]] = {}
    for group in grouped:
        median_frame = round(
            statistics.median(row["event_frame_index"] for row in group)
        )
        canonical = min(
            group,
            key=lambda row: (
                abs(
                    row["approximate_frame_index"]
                    - row["event_frame_index"]
                ),
                abs(row["event_frame_index"] - median_frame),
                row["onset_id"],
            ),
        )
        deduplicated_candidates[canonical["onset_id"]] = {
            **canonical,
            "source_onset_ids": sorted(
                row["onset_id"] for row in group
            ),
        }

    decisions = []
    decisions_by_id = {}
    for onset in onsets:
        if onset["kind"] == "spell":
            row = {
                "onset_id": onset["onset_id"],
                "event_exists": True,
                "event_frame_index": onset["event_frame_index"],
                "existence_evidence": {
                    "absent_before": onset["absence_confirmed"],
                    "independent_after": True,
                    "persistent_after": onset["persistence_confirmed"],
                },
                "side": "enemy",
                "card": None,
                "side_evidence": {
                    "origin": "enemy spell scan",
                    "team_indicator": None,
                    "direction": None,
                    "direct": True,
                },
                "identity_frame_index": None,
                "identity_artifacts": [],
                "reason": "independently discovered enemy spell sequence",
            }
        else:
            row = {
                "onset_id": onset["onset_id"],
                "event_exists": False,
                "event_frame_index": None,
                "existence_evidence": {
                    "absent_before": False,
                    "independent_after": False,
                    "persistent_after": False,
                },
                "side": "unresolved",
                "card": None,
                "side_evidence": {
                    "origin": None,
                    "team_indicator": None,
                    "direction": None,
                    "direct": False,
                },
                "identity_frame_index": None,
                "identity_artifacts": [],
                "reason": "unit marker candidate rejected or deduplicated",
            }
        decisions.append(row)
        decisions_by_id[row["onset_id"]] = row

    for onset_id, candidate in deduplicated_candidates.items():
        side = side_rows[onset_id]
        corrected_frame = int(candidate["event_frame_index"])
        evidence = candidate["existence_evidence"]
        resolved = evidence.get(
            "secondary_persists_or_resolves_after",
            evidence.get("secondary_persists_after"),
        )
        row = decisions_by_id[onset_id]
        row.update(
            {
                "event_exists": True,
                "event_frame_index": corrected_frame,
                "existence_evidence": {
                    "absent_before": evidence["secondary_absent_before"],
                    "independent_after": evidence[
                        "secondary_appears_at_marker"
                    ],
                    "persistent_after": resolved,
                },
                "side": side["side"],
                "side_evidence": {
                    "origin": side.get("origin"),
                    "team_indicator": side.get("team_indicator"),
                    "direction": None,
                    "direct": side["direct"],
                },
                "reason": (
                    f"existence: {candidate.get('existence_reason', '')}; "
                    f"side: {side.get('reason', '')}"
                ),
            }
        )
        onsets_by_id[onset_id]["event_frame_index"] = corrected_frame

    recovery_path = (
        run_dir
        / "recovery_outputs"
        / "enemy_simultaneous_recoveries.json"
    )
    recoveries: list[dict[str, Any]] = []
    recovery_provenance: list[dict[str, Any]] = []
    if recovery_path.is_file():
        recovery_document = _read(recovery_path)
        if recovery_document.get("run_id") != manifest["run_id"]:
            raise ValueError("simultaneous recovery run_id mismatch")
        if recovery_document.get("stage") != "enemy_simultaneous_recoveries":
            raise ValueError("simultaneous recovery has wrong stage")
        raw_recoveries = recovery_document.get("recoveries")
        if not isinstance(raw_recoveries, list) or any(
            not isinstance(row, dict) for row in raw_recoveries
        ):
            raise ValueError("simultaneous recoveries must be a list")
        recoveries = raw_recoveries
        raw_provenance = recovery_document.get("worker_provenance", [])
        if isinstance(raw_provenance, list):
            recovery_provenance = [
                row for row in raw_provenance if isinstance(row, dict)
            ]
    accepted_enemy_frames = [
        int(candidate["event_frame_index"])
        for onset_id, candidate in deduplicated_candidates.items()
        if side_rows[onset_id]["side"] == "enemy"
    ]
    novel_recoveries, duplicate_recoveries = _novel_simultaneous_recoveries(
        recoveries,
        accepted_enemy_frames=accepted_enemy_frames,
    )
    for recovery in novel_recoveries:
        onset_id = recovery.get("onset_id")
        if not isinstance(onset_id, str) or onset_id in onsets_by_id:
            raise ValueError("invalid or duplicate simultaneous recovery ID")
        frame = int(recovery["event_frame_index"])
        artifacts = recovery.get("verification_artifacts")
        sampled = recovery.get("sampled_frame_indices")
        evidence = recovery.get("evidence")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not isinstance(sampled, list)
            or frame not in sampled
            or not isinstance(evidence, dict)
            or evidence.get("direct_new_actor") is not True
            or evidence.get("secondary_absent_before") is not True
            or evidence.get("secondary_appears_at_marker") is not True
            or evidence.get(
                "secondary_persists_or_resolves_after"
            )
            is not True
        ):
            raise ValueError(f"{onset_id}: incomplete recovery evidence")
        onset = {
            "onset_id": onset_id,
            "candidate_id": None,
            "event_frame_index": frame,
            "kind": "unit_or_building",
            "focus_cell": None,
            "track_id": onset_id,
            "sampled_frame_indices": sampled,
            "absence_confirmed": True,
            "persistence_confirmed": True,
            "evidence": {
                "elixir_drop": None,
                "hand_transition": None,
                "deployment_onset": True,
                "first_visible_object": True,
                "side_direction": True,
                "impact_sequence": None,
            },
            "verification_artifacts": artifacts,
            "identity_artifacts": [],
        }
        onset_document["onsets"].append(onset)
        onsets_by_id[onset_id] = onset
        decisions.append(
            {
                "onset_id": onset_id,
                "event_exists": True,
                "event_frame_index": frame,
                "existence_evidence": {
                    "absent_before": True,
                    "independent_after": True,
                    "persistent_after": True,
                },
                "side": "enemy",
                "card": None,
                "side_evidence": {
                    "origin": "direct simultaneous-actor recovery",
                    "team_indicator": "red",
                    "direction": None,
                    "direct": True,
                },
                "identity_frame_index": None,
                "identity_artifacts": [],
                "reason": recovery.get("reason", ""),
            }
        )

    if any(
        row["event_exists"] and row["side"] == "unresolved"
        for row in decisions
    ):
        unresolved = [
            row["onset_id"]
            for row in decisions
            if row["event_exists"] and row["side"] == "unresolved"
        ]
        raise ValueError(
            "accepted enemy events have unresolved side: "
            + ", ".join(unresolved)
        )
    onset_document["onsets"] = sorted(
        onset_document["onsets"],
        key=lambda row: row["event_frame_index"],
    )
    atomic_write_json(run_dir / "enemy_onsets.json", onset_document)
    output = {
        "run_id": manifest["run_id"],
        "stage": "enemy_identities",
        "annotation_session_id": "merged-sequence-existence-and-side-workers",
        "model": "+".join(
            sorted(
                {
                    str(document.get("model"))
                    for _, document in [*existence_chunks, *side_chunks]
                }
            )
        ),
        "reasoning_effort": "+".join(
            sorted(
                {
                    str(document.get("reasoning_effort"))
                    for _, document in [*existence_chunks, *side_chunks]
                }
            )
        ),
        "worker_provenance": [
            {
                "stage": document["stage"],
                "target_range": document["target_range"],
                "annotation_session_id": document.get(
                    "annotation_session_id"
                ),
                "model": document.get("model"),
                "reasoning_effort": document.get("reasoning_effort"),
            }
            for _, document in [*existence_chunks, *side_chunks]
        ]
        + [
            {"stage": "enemy_side_escalation", **row}
            for row in side_escalation_provenance
        ]
        + [
            {"stage": "enemy_simultaneous_recovery", **row}
            for row in recovery_provenance
        ],
        "decisions": sorted(
            decisions,
            key=lambda row: onsets_by_id[row["onset_id"]][
                "event_frame_index"
            ],
        ),
    }
    atomic_write_json(run_dir / "enemy_identities.json", output)
    print(
        json.dumps(
            {
                "decisions": len(decisions),
                "events": sum(row["event_exists"] for row in decisions),
                "enemy": sum(
                    row["event_exists"] and row["side"] == "enemy"
                    for row in decisions
                ),
                "own": sum(
                    row["event_exists"] and row["side"] == "own"
                    for row in decisions
                ),
                "deduplicated": len(accepted_existence)
                - len(deduplicated_candidates),
                "simultaneous_recovered": len(novel_recoveries),
                "simultaneous_duplicates": len(duplicate_recoveries),
            }
        )
    )


if __name__ == "__main__":
    main()
