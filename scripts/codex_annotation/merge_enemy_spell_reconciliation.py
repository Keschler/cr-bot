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
    validate_enemy_spell_confirmation_decisions,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_worker_family(
    run_dir: Path,
    *,
    prefix: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pattern = re.compile(
        rf"{re.escape(prefix)}-\d{{6}}-\d{{6}}\.json"
    )
    decisions = {}
    provenance = []
    for package_path in sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if pattern.fullmatch(path.name)
    ):
        package = _read(package_path)
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            if package.get("reviews") == []:
                continue
            raise ValueError(f"missing spell worker output {output_path}")
        output = _read(output_path)
        if output.get("run_id") != package.get("run_id"):
            raise ValueError(f"{output_path}: run_id mismatch")
        if output.get("stage") != "enemy_spell_confirmation_chunk":
            raise ValueError(f"{output_path}: wrong stage")
        if output.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target range mismatch")
        validate_enemy_spell_confirmation_decisions(output, package)
        provenance.append(
            {
                "stage": prefix,
                **{
                    key: output.get(key)
                    for key in (
                        "target_range",
                        "annotation_session_id",
                        "model",
                        "reasoning_effort",
                    )
                },
            }
        )
        reviews = {
            row["review_id"]: row for row in package["reviews"]
        }
        for decision in output["decisions"]:
            review_id = decision["review_id"]
            if review_id in decisions:
                raise ValueError(f"duplicate spell decision {review_id}")
            decisions[review_id] = {
                **decision,
                "review": reviews[review_id],
            }
    return decisions, provenance


def _deduplicate_spells(
    rows: list[dict[str, Any]],
    *,
    tolerance: int = 5,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda value: value["event_frame_index"]):
        if (
            not groups
            or row["event_frame_index"]
            - groups[-1][-1]["event_frame_index"]
            > tolerance
        ):
            groups.append([row])
        else:
            groups[-1].append(row)
    selected = []
    for group in groups:
        median = round(
            statistics.median(row["event_frame_index"] for row in group)
        )
        selected.append(
            min(
                group,
                key=lambda row: (
                    abs(row["event_frame_index"] - median),
                    row["onset_id"],
                ),
            )
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile broad enemy spells with independent own/unit confounds, "
            "focused post-own recovery, and the clip-end sentinel."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    source = _read(
        run_dir / "enemy_spell_reconciliation_candidates.json"
    )
    onsets = _read(run_dir / "enemy_onsets.json")
    identities = _read(run_dir / "enemy_identities.json")
    if any(
        document.get("run_id") != manifest["run_id"]
        for document in (source, onsets, identities)
    ):
        raise ValueError("spell reconciliation inputs do not share a run_id")
    classifications = {
        row["onset_id"]: row
        for row in source.get("classifications", [])
        if isinstance(row, dict)
    }
    onset_by_id = {
        row["onset_id"]: row
        for row in onsets.get("onsets", [])
        if isinstance(row, dict)
    }
    raw_spell_ids = {
        onset_id
        for onset_id in classifications
        if onset_by_id.get(onset_id, {}).get("kind") == "spell"
    }
    if raw_spell_ids != set(classifications):
        raise ValueError("spell classifications do not cover raw spells exactly")
    recovery_left, recovery_left_provenance = _load_worker_family(
        run_dir,
        prefix="enemy-spell-recovery-left",
    )
    recovery_right, recovery_right_provenance = _load_worker_family(
        run_dir,
        prefix="enemy-spell-recovery-right",
    )
    if set(recovery_left) & set(recovery_right):
        raise ValueError("left and right spell recovery reviews overlap")
    recovery = {**recovery_left, **recovery_right}
    recovery_provenance = [
        *recovery_left_provenance,
        *recovery_right_provenance,
    ]
    boundary, boundary_provenance = _load_worker_family(
        run_dir,
        prefix="enemy-spell-boundary",
    )
    final_spells = []
    audit_rows = []
    for onset_id, classification in classifications.items():
        onset = onset_by_id[onset_id]
        kind = classification["classification"]
        retained = kind == "unconfounded_broad_spell"
        decision = None
        recovery_rows: list[dict[str, Any]] = []
        if kind == "own_spell_confound":
            recovery_rows = [
                row
                for row in recovery.values()
                if row["review"].get("source_onset_id") == onset_id
            ]
            if not recovery_rows:
                raise ValueError(f"{onset_id}: missing focused recovery decision")
            confirmed = [
                row
                for row in recovery_rows
                if row["decision"] == "confirmed"
            ]
            if confirmed:
                decision = min(
                    confirmed,
                    key=lambda row: (
                        int(row["event_frame_index"]),
                        row["review_id"],
                    ),
                )
            retained = decision is not None
        elif kind not in {"unit_confound", "unconfounded_broad_spell"}:
            raise ValueError(f"{onset_id}: unknown spell classification {kind}")
        if retained:
            row = dict(onset)
            if decision is not None:
                row["event_frame_index"] = decision["event_frame_index"]
                row["sampled_frame_indices"] = decision["review"][
                    "sampled_frame_indices"
                ]
                row["verification_artifacts"] = decision[
                    "confirmation_artifacts"
                ]
                row["track_id"] = (
                    f"recovered-spell-{row['event_frame_index']:06d}"
                )
            row["absence_confirmed"] = True
            row["persistence_confirmed"] = True
            row["evidence"]["side_direction"] = True
            row["evidence"]["impact_sequence"] = True
            final_spells.append(row)
        audit_rows.append(
            {
                **classification,
                "recovery_decision": (
                    "confirmed"
                    if decision is not None
                    else (
                        sorted(
                            {
                                row["decision"]
                                for row in recovery_rows
                            }
                        )
                        if recovery_rows
                        else None
                    )
                ),
                "recovered_frame_index": (
                    decision["event_frame_index"]
                    if decision is not None
                    else None
                ),
                "retained": retained,
            }
        )
    for decision in boundary.values():
        if decision["decision"] != "confirmed":
            continue
        frame = decision["event_frame_index"]
        final_spells.append(
            {
                "onset_id": f"enemy-spell-boundary-{frame:06d}",
                "candidate_id": None,
                "event_frame_index": frame,
                "kind": "spell",
                "focus_cell": None,
                "track_id": f"boundary-spell-{frame:06d}",
                "sampled_frame_indices": decision["review"][
                    "sampled_frame_indices"
                ],
                "absence_confirmed": True,
                "persistence_confirmed": True,
                "evidence": {
                    "elixir_drop": None,
                    "hand_transition": None,
                    "deployment_onset": None,
                    "first_visible_object": True,
                    "side_direction": True,
                    "impact_sequence": True,
                },
                "verification_artifacts": decision[
                    "confirmation_artifacts"
                ],
                "identity_artifacts": [],
            }
        )
    final_spells = _deduplicate_spells(final_spells)
    non_spells = [
        row
        for row in onsets["onsets"]
        if row.get("kind") != "spell"
    ]
    onsets["onsets"] = sorted(
        [*non_spells, *final_spells],
        key=lambda row: row["event_frame_index"],
    )
    atomic_write_json(run_dir / "enemy_onsets.json", onsets)

    final_by_id = {row["onset_id"]: row for row in final_spells}
    identity_rows = [
        row
        for row in identities["decisions"]
        if row.get("onset_id") not in raw_spell_ids
        or row.get("onset_id") in final_by_id
    ]
    identity_by_id = {row["onset_id"]: row for row in identity_rows}
    for onset_id, onset in final_by_id.items():
        row = identity_by_id.get(onset_id)
        if row is None:
            row = {
                "onset_id": onset_id,
                "card": None,
                "identity_frame_index": None,
                "identity_artifacts": [],
            }
            identity_rows.append(row)
            identity_by_id[onset_id] = row
        row.update(
            {
                "event_exists": True,
                "event_frame_index": onset["event_frame_index"],
                "existence_evidence": {
                    "absent_before": True,
                    "independent_after": True,
                    "persistent_after": True,
                },
                "side": "enemy",
                "side_evidence": {
                    "origin": "spell reconciliation",
                    "team_indicator": "red",
                    "direction": "downward",
                    "direct": True,
                },
                "reason": "independent spell discovery after confound reconciliation",
            }
        )
    identities["decisions"] = sorted(
        identity_rows,
        key=lambda row: onset_by_id.get(
            row["onset_id"], final_by_id.get(row["onset_id"])
        )["event_frame_index"],
    )
    atomic_write_json(run_dir / "enemy_identities.json", identities)
    audit = {
        "run_id": manifest["run_id"],
        "stage": "enemy_spell_reconciliation",
        "worker_provenance": [
            *recovery_provenance,
            *boundary_provenance,
        ],
        "classifications": audit_rows,
        "boundary_decisions": [
            {
                **{key: value for key, value in row.items() if key != "review"},
                "retained": row["decision"] == "confirmed",
            }
            for row in boundary.values()
        ],
        "final_spell_onset_ids": sorted(final_by_id),
    }
    atomic_write_json(
        run_dir / "enemy_spell_reconciliation.json",
        audit,
    )
    print(
        json.dumps(
            {
                "raw_spells": len(raw_spell_ids),
                "final_spells": len(final_spells),
                "unit_confounds": sum(
                    row["classification"] == "unit_confound"
                    for row in classifications.values()
                ),
                "own_confounds": sum(
                    row["classification"] == "own_spell_confound"
                    for row in classifications.values()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
