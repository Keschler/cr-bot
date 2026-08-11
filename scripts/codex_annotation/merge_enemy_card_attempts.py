from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.annotation_pipeline import validate_enemy_card_decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge ordered blind enemy-card attempts; later rows win."
    )
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--decision-glob", action="append", default=[])
    parser.add_argument("--decision-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--decision-row",
        action="append",
        default=[],
        metavar="PATH::ONSET_ID",
        help="Apply one selected row from a multi-target blind artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--own-events-file",
        type=Path,
        help="Optionally include validated own semantics for a combined benchmark.",
    )
    args = parser.parse_args()
    targets = json.loads(args.targets_file.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for pattern in args.decision_glob:
        paths.extend(Path(value) for value in sorted(glob.glob(pattern)))
    paths.extend(args.decision_file)
    selected: dict[str, tuple[dict[str, object], Path]] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document.get("cards", []):
            selected[row["onset_id"]] = (dict(row), path)
    for item in args.decision_row:
        if "::" not in item:
            parser.error("--decision-row must use PATH::ONSET_ID")
        value, onset_id = item.rsplit("::", maxsplit=1)
        path = Path(value)
        document = json.loads(path.read_text(encoding="utf-8"))
        matches = [row for row in document.get("cards", []) if row.get("onset_id") == onset_id]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected exactly one row for {onset_id}")
        selected[onset_id] = (dict(matches[0]), path)
    target_rows = targets["targets"]
    missing = [row["onset_id"] for row in target_rows if row["onset_id"] not in selected]
    if missing:
        raise ValueError(f"missing card decisions for: {', '.join(missing)}")
    cards = []
    provenance = []
    validation_targets = []
    for target in target_rows:
        onset_id = target["onset_id"]
        row, source = selected[onset_id]
        cards.append(row)
        validation_target = dict(target)
        # A later evidence experiment may legitimately replace the delayed
        # sheet. Preserve the selected worker's exact cited provenance.
        validation_target["identity_frame_index"] = row.get("identity_frame_index")
        validation_target["identity_artifacts"] = row.get("identity_artifacts", [])
        validation_targets.append(validation_target)
        provenance.append({"onset_id": onset_id, "source": str(source)})
    own_events = []
    if args.own_events_file is not None:
        own_document = json.loads(args.own_events_file.read_text(encoding="utf-8"))
        own_events = [
            {
                "side": "own",
                "card": row["card"],
                "event_frame_index": int(row["event_frame_index"]),
            }
            for row in own_document.get("events", [])
        ]
    document = {
        "run_id": targets["run_id"],
        "stage": "enemy_cards_chunk",
        "target_range": [0, 1931],
        "annotation_session_id": "ordered-blind-attempt-cascade",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "cards": cards,
        "events": own_events + [
            {
                "side": "enemy",
                "card": row["card"],
                "event_frame_index": int(target["event_frame_index"]),
            }
            for target, row in zip(target_rows, cards)
        ],
        "selection_provenance": provenance,
    }
    validation_package = dict(targets)
    validation_package["targets"] = validation_targets
    validate_enemy_card_decisions(document, validation_package)
    atomic_write_json(args.output, document)
    print(json.dumps({"output": str(args.output), "cards": len(cards)}))


if __name__ == "__main__":
    main()
