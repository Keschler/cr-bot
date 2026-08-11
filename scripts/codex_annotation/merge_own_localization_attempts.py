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

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.own_localization import validate_own_localization_decisions


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge explicitly selected, validated blind own locations."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-package", type=Path, required=True)
    parser.add_argument("--aggregate-prediction", type=Path, required=True)
    args = parser.parse_args()

    candidate = _read(args.candidate)
    manifest = _read(args.selection)
    entries = manifest.get("selections")
    if not isinstance(entries, list) or not entries:
        raise ValueError("selection manifest must contain a non-empty selections list")

    decisions: dict[str, dict[str, Any]] = {}
    targets: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, str]] = []
    validated_sources: dict[tuple[Path, Path], tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"event_id", "source", "package"}:
            raise ValueError("each selection requires exactly event_id, source, and package")
        event_id = entry["event_id"]
        source = Path(entry["source"])
        package_path = Path(entry["package"])
        key = (source, package_path)
        if key not in validated_sources:
            document = _read(source)
            package = _read(package_path)
            validate_own_localization_decisions(document, package)
            validated_sources[key] = (document, package)
        document, package = validated_sources[key]
        matches = [row for row in document["decisions"] if row["event_id"] == event_id]
        target_matches = [row for row in package["targets"] if row["event_id"] == event_id]
        if len(matches) != 1 or len(target_matches) != 1:
            raise ValueError(f"{event_id}: selected source/package must contain one matching row")
        if event_id in decisions:
            raise ValueError(f"duplicate selection for {event_id}")
        decisions[event_id] = dict(matches[0])
        targets[event_id] = dict(target_matches[0])
        provenance.append(
            {"event_id": event_id, "source": str(source), "package": str(package_path)}
        )

    own_events = [row for row in candidate.get("events", []) if row.get("side") == "own"]
    if len(own_events) != len(decisions):
        raise ValueError(
            f"candidate has {len(own_events)} own events but manifest has {len(decisions)}"
        )
    by_semantics: dict[tuple[int, str], str] = {}
    for event_id, target in targets.items():
        key = (int(target["event_frame_index"]), str(target["card"]))
        if key in by_semantics:
            raise ValueError(f"duplicate localization target semantics: {key}")
        by_semantics[key] = event_id

    merged_events = []
    used: set[str] = set()
    for raw in candidate["events"]:
        row = dict(raw)
        if row.get("side") == "own":
            key = (int(row["event_frame_index"]), str(row["card"]))
            event_id = by_semantics.get(key)
            if event_id is None:
                raise ValueError(f"no selected localization for own event {key}")
            decision = decisions[event_id]
            row.update(
                {
                    "cell": decision["cell"],
                    "location_frame_index": decision["location_frame_index"],
                    "location_rule": decision["location_rule"],
                    "location_confidence": decision["confidence"],
                    "location_reason": decision["reason"],
                    "location_event_id": event_id,
                }
            )
            used.add(event_id)
        merged_events.append(row)
    if used != set(decisions):
        raise ValueError("one or more selected localization decisions were not merged")

    merged = dict(candidate)
    merged["events"] = merged_events
    merged["own_location_selection_provenance"] = provenance
    atomic_write_json(args.output, merged)

    ordered_ids = [row["location_event_id"] for row in merged_events if row.get("side") == "own"]
    aggregate_package = {
        "run_id": candidate["run_id"],
        "target_range": candidate["target_range"],
        "targets": [targets[event_id] for event_id in ordered_ids],
    }
    aggregate_prediction = {
        "run_id": candidate["run_id"],
        "stage": "own_localization_chunk",
        "target_range": candidate["target_range"],
        "annotation_session_id": "selected-blind-own-localization-cascade",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "decisions": [decisions[event_id] for event_id in ordered_ids],
    }
    validate_own_localization_decisions(aggregate_prediction, aggregate_package)
    atomic_write_json(args.aggregate_package, aggregate_package)
    atomic_write_json(args.aggregate_prediction, aggregate_prediction)
    print(json.dumps({"output": str(args.output), "own_locations": len(ordered_ids)}))


if __name__ == "__main__":
    main()
