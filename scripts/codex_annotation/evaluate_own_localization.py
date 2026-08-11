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
from cr_bot.eval.action_eval import CARD_ALIASES
from cr_bot.own_localization import validate_own_localization_decisions


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _card(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    if normalized.startswith("evo-"):
        normalized = normalized[4:]
    if normalized == "the-log":
        normalized = "log"
    return CARD_ALIASES.get(normalized, normalized)


def evaluate_locations(
    truth_events: list[dict[str, Any]],
    package: dict[str, Any],
    prediction: dict[str, Any],
    *,
    frame_tolerance: int,
    cell_tolerance: int,
) -> dict[str, Any]:
    decisions = validate_own_localization_decisions(prediction, package)
    targets = {row["event_id"]: row for row in package["targets"]}
    truth = [row for row in truth_events if row.get("side") == "own"]
    unused = set(range(len(truth)))
    rows = []
    for decision in decisions:
        target = targets[decision["event_id"]]
        candidates = []
        for index in unused:
            expected = truth[index]
            if _card(expected["card"]) != _card(target["card"]):
                continue
            delta = abs(
                int(expected["frame_index"]) - int(target["event_frame_index"])
            )
            if delta <= frame_tolerance:
                candidates.append((delta, index))
        if not candidates:
            rows.append(
                {
                    "event_id": decision["event_id"],
                    "semantic_match": False,
                    "location_correct": False,
                    "predicted_cell": decision["cell"],
                }
            )
            continue
        frame_error, index = min(candidates)
        unused.remove(index)
        expected = truth[index]
        predicted_cell = decision["cell"]
        expected_cell = expected.get("cell")
        coordinate_errors = (
            None
            if not isinstance(expected_cell, list) or len(expected_cell) != 2
            else [
                abs(int(predicted_cell[0]) - int(expected_cell[0])),
                abs(int(predicted_cell[1]) - int(expected_cell[1])),
            ]
        )
        correct = coordinate_errors is not None and all(
            value <= cell_tolerance for value in coordinate_errors
        )
        rows.append(
            {
                "event_id": decision["event_id"],
                "semantic_match": True,
                "frame_error": frame_error,
                "predicted_cell": predicted_cell,
                "expected_cell": expected_cell,
                "coordinate_errors": coordinate_errors,
                "location_correct": correct,
            }
        )
    correct_count = sum(row["location_correct"] is True for row in rows)
    return {
        "frame_tolerance": frame_tolerance,
        "cell_tolerance_per_coordinate": cell_tolerance,
        "expected": len(rows),
        "predicted": len(decisions),
        "correct": correct_count,
        "incorrect": len(rows) - correct_count,
        "accuracy": correct_count / len(rows) if rows else 1.0,
        "failed_event_ids": [
            row["event_id"] for row in rows if row["location_correct"] is not True
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outer-session scoring for a validated blind own-location chunk."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-tolerance", type=int, default=5)
    parser.add_argument("--cell-tolerance", type=int, default=1)
    args = parser.parse_args()
    truth = _read(args.ground_truth)
    package = _read(args.package)
    prediction = _read(args.prediction)
    report = evaluate_locations(
        truth["events"],
        package,
        prediction,
        frame_tolerance=args.frame_tolerance,
        cell_tolerance=args.cell_tolerance,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

