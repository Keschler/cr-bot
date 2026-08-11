from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.eval.action_eval import CARD_ALIASES
from cr_bot.annotation_harness import atomic_write_json


def _canonical_card(card: str) -> str:
    normalized = card.replace("_", "-")
    if normalized.startswith("evo-"):
        normalized = normalized[4:]
    if normalized == "the-log":
        normalized = "log"
    return CARD_ALIASES.get(normalized, normalized)


def _load_events(path: Path, *, predicted: bool) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document["events"] if isinstance(document, dict) else document
    frame_key = "event_frame_index" if predicted else "frame_index"
    return [
        {
            "side": row["side"],
            "card": _canonical_card(row["card"]),
            "frame": int(row[frame_key]),
        }
        for row in rows
    ]


def evaluate(
    expected: list[dict[str, object]],
    predicted: list[dict[str, object]],
    *,
    start_frame: int,
    end_frame_exclusive: int,
    tolerance_frames: int,
) -> dict[str, object]:
    expected = [
        row for row in expected if start_frame <= int(row["frame"]) < end_frame_exclusive
    ]
    predicted = [
        row for row in predicted if start_frame <= int(row["frame"]) < end_frame_exclusive
    ]
    unmatched = set(range(len(predicted)))
    matches: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    for truth in sorted(expected, key=lambda row: int(row["frame"])):
        candidates: list[tuple[int, int]] = []
        for index in unmatched:
            proposal = predicted[index]
            if proposal["side"] != truth["side"] or proposal["card"] != truth["card"]:
                continue
            delta = abs(int(proposal["frame"]) - int(truth["frame"]))
            if delta <= tolerance_frames:
                candidates.append((delta, index))
        if not candidates:
            misses.append(truth)
            continue
        delta, index = min(candidates)
        unmatched.remove(index)
        matches.append(
            {
                "ground_truth": truth,
                "predicted": predicted[index],
                "frame_error": delta,
            }
        )
    false_positives = [predicted[index] for index in sorted(unmatched)]
    true_positive_count = len(matches)
    precision = (
        true_positive_count / len(predicted) if predicted else (1.0 if not expected else 0.0)
    )
    recall = true_positive_count / len(expected) if expected else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "range": [start_frame, end_frame_exclusive],
        "tolerance_frames": tolerance_frames,
        "expected": len(expected),
        "predicted": len(predicted),
        "true_positives": true_positive_count,
        "false_positives": len(false_positives),
        "false_negatives": len(misses),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "misses": misses,
        "false_positive_events": false_positives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a blinded verification.json without scoring location."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--tolerance-frames", type=int, default=5)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--side", choices=["own", "enemy"])
    args = parser.parse_args()

    expected = _load_events(args.ground_truth, predicted=False)
    predicted = _load_events(args.prediction, predicted=True)
    if args.side is not None:
        expected = [row for row in expected if row["side"] == args.side]
        predicted = [row for row in predicted if row["side"] == args.side]
    report = evaluate(
        expected,
        predicted,
        start_frame=args.start_frame,
        end_frame_exclusive=args.end_frame,
        tolerance_frames=args.tolerance_frames,
    )
    if args.summary_only:
        report = {
            key: report[key]
            for key in (
                "range",
                "tolerance_frames",
                "expected",
                "predicted",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
            )
        }
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
