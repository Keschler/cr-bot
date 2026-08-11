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

from cr_bot.annotation_harness import atomic_write_json
from cr_bot.domain.card_metadata import CARD_METADATA


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge enemy card-only chunks.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--package-prefix",
        default="cards",
        help="Work-package/output filename prefix.",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="Target document; defaults to RUN_DIR/enemy_identity_targets.json.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    target_document = _read(
        args.targets_file.resolve()
        if args.targets_file is not None
        else run_dir / "enemy_identity_targets.json"
    )
    targets = target_document["targets"]
    package_pattern = re.compile(
        rf"{re.escape(args.package_prefix)}-\d{{6}}-\d{{6}}\.json"
    )
    packages = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if package_pattern.fullmatch(path.name)
    )
    if not packages:
        raise ValueError("no enemy card work packages")
    documents = []
    card_rows = []
    for package_path in packages:
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing worker output {output_path}")
        package = _read(package_path)
        document = _read(output_path)
        if document.get("run_id") != manifest["run_id"]:
            raise ValueError(f"{output_path}: run_id mismatch")
        if document.get("stage") != "enemy_cards_chunk":
            raise ValueError(f"{output_path}: wrong stage")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        documents.append(document)
        card_rows.extend(document.get("cards", []))
    onset_ids = [row.get("onset_id") for row in card_rows]
    if len(onset_ids) != len(set(onset_ids)):
        raise ValueError("duplicate enemy card decisions")
    rows_by_id = {row["onset_id"]: row for row in card_rows}
    target_ids = {row["onset_id"] for row in targets}
    if set(rows_by_id) != target_ids:
        missing = sorted(target_ids - set(rows_by_id))
        extra = sorted(set(rows_by_id) - target_ids)
        raise ValueError(
            f"enemy card chunks must cover every target; "
            f"missing={missing}, extra={extra}"
        )
    rows = []
    unresolved = []
    for target in targets:
        onset_id = target["onset_id"]
        row = rows_by_id.get(onset_id)
        if row is None or row.get("card") is None:
            unresolved.append(onset_id)
            continue
        card = row["card"]
        if card == "the-log":
            raise ValueError(f"{onset_id}: use canonical card slug 'log'")
        metadata_card = card[4:] if card.startswith("evo-") else card
        metadata = CARD_METADATA.get(metadata_card)
        if metadata is None:
            raise ValueError(f"{onset_id}: unknown canonical card {card!r}")
        expected_kind = target["kind"]
        actual_kind = metadata["kind"]
        if expected_kind == "spell":
            if actual_kind != "spell":
                raise ValueError(f"{onset_id}: spell onset has non-spell card")
            if row.get("identity_frame_index") is not None:
                raise ValueError(f"{onset_id}: spell must not have identity frame")
            if row.get("identity_artifacts") != []:
                raise ValueError(f"{onset_id}: spell must not have identity sheets")
        else:
            if actual_kind not in {"troop", "building"}:
                raise ValueError(f"{onset_id}: unit onset has spell card")
            if row.get("visibility") != "clear":
                unresolved.append(onset_id)
                continue
            if not isinstance(row.get("identity_frame_index"), int):
                raise ValueError(f"{onset_id}: missing identity frame")
            artifact_count = len(row.get("identity_artifacts", []))
            neighbor_mode = (
                target.get("identity_render_options", {}).get("mode")
                == "neighbor_candidates"
            )
            if artifact_count < 2 and not (neighbor_mode and artifact_count >= 1):
                raise ValueError(f"{onset_id}: missing two identity sheets")
            if (
                row.get("identity_frame_index")
                != target.get("identity_frame_index")
            ):
                raise ValueError(f"{onset_id}: identity frame changed in card stage")
            if row.get("identity_artifacts") != target.get("identity_artifacts"):
                raise ValueError(
                    f"{onset_id}: identity artifacts changed in card stage"
                )
        if row.get("confidence") != "direct":
            unresolved.append(onset_id)
            continue
        rows.append(row)
    if unresolved:
        raise ValueError(
            "enemy card identity remains unresolved for: "
            + ", ".join(unresolved)
        )
    output = {
        "run_id": manifest["run_id"],
        "stage": "enemy_cards",
        "annotation_session_id": "merged-card-chunks",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": [
            {
                key: document.get(key)
                for key in (
                    "target_range",
                    "annotation_session_id",
                    "model",
                    "reasoning_effort",
                )
            }
            for document in documents
        ],
        "cards": rows,
    }
    atomic_write_json(run_dir / "enemy_cards.json", output)
    print(json.dumps({"cards": len(rows), "unresolved": 0}))


if __name__ == "__main__":
    main()
