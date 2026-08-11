from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json


def _materialize_unit_onsets(
    *,
    package: dict,
) -> list[dict]:
    """Materialize every persistent marker burst as a recall-safe candidate."""
    target_start, target_end = package["target_range"]
    onsets = []
    for burst in package["bursts"]:
        if burst["package_role"] != "owned_burst":
            continue
        marker_start = int(burst["start_frame"])
        if not target_start <= marker_start < target_end:
            raise ValueError(
                f"{burst['burst_id']}: owned marker is outside target range"
            )
        frame = int(burst["candidate_frame_index"])
        if not target_start <= frame < target_end:
            frame = max(target_start, marker_start - 1)
        burst_number = int(burst["burst_id"].split(":")[-1])
        onset_id = f"enemy-unit-{frame:06d}-b{burst_number:06d}"
        onsets.append(
            {
                "onset_id": onset_id,
                "candidate_id": burst["supporting_candidate_id"],
                "event_frame_index": frame,
                "kind": "unit_or_building",
                "focus_cell": None,
                "track_id": burst["burst_id"],
                "sampled_frame_indices": burst["sampled_frame_indices"],
                "absence_confirmed": None,
                "persistence_confirmed": True,
                "evidence": {
                    "elixir_drop": None,
                    "hand_transition": None,
                    "deployment_onset": True,
                    "first_visible_object": True,
                    "side_direction": None,
                    "impact_sequence": None,
                },
                "verification_artifacts": [
                    burst["review_artifact"],
                    burst["focus_review_artifact"],
                ],
                "identity_artifacts": [],
            }
        )
    return onsets


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_family(
    run_dir: Path,
    *,
    prefix: str,
    expected_stage: str,
) -> list[dict]:
    pattern = re.compile(rf"{re.escape(prefix)}-\d{{6}}-\d{{6}}\.json")
    packages = sorted(
        path
        for path in (run_dir / "work_packages").iterdir()
        if pattern.fullmatch(path.name)
    )
    if not packages:
        raise ValueError(f"no {prefix} work packages")
    documents = []
    for package_path in packages:
        output_path = run_dir / "worker_outputs" / package_path.name
        if not output_path.is_file():
            raise ValueError(f"missing worker output {output_path}")
        package = _read(package_path)
        document = _read(output_path)
        if document.get("stage") != expected_stage:
            raise ValueError(f"{output_path}: wrong stage")
        if document.get("target_range") != package.get("target_range"):
            raise ValueError(f"{output_path}: target_range mismatch")
        start, end = package["target_range"]
        for onset in document.get("onsets", []):
            frame = onset.get("event_frame_index")
            if not isinstance(frame, int) or not start <= frame < end:
                raise ValueError(
                    f"{output_path}: onset is outside the owned range"
                )
        documents.append(document)
    return documents


def _load_unit_candidates(run_dir: Path) -> list[dict]:
    packages = sorted(
        (run_dir / "work_packages").glob(
            "enemy-units-??????-??????.json"
        )
    )
    if not packages:
        raise ValueError("no enemy-unit marker packages")
    return [
        {
            "run_id": package["run_id"],
            "stage": "enemy_unit_marker_candidates_chunk",
            "target_range": package["target_range"],
            "annotation_session_id": "deterministic-marker-candidates",
            "model": "none",
            "reasoning_effort": "none",
            "onsets": _materialize_unit_onsets(package=package),
        }
        for package in map(_read, packages)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge marker-unit and independent spell onset chunks."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--units-only",
        action="store_true",
        help="Materialize deterministic unit candidates before spell workers.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    units = _load_unit_candidates(run_dir)
    spells = (
        []
        if args.units_only
        else _load_family(
            run_dir,
            prefix="enemy-spells",
            expected_stage="enemy_spell_onsets_chunk",
        )
    )
    documents = [*units, *spells]
    if any(row.get("run_id") != manifest["run_id"] for row in documents):
        raise ValueError("enemy onset chunk run_id mismatch")
    onsets = [
        onset for document in documents for onset in document.get("onsets", [])
    ]
    ids = [row["onset_id"] for row in onsets]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate enemy onset IDs")
    for row in onsets:
        if row.get("persistence_confirmed") is not True:
            raise ValueError(f"{row['onset_id']}: persistence is not confirmed")
        if not isinstance(row.get("evidence"), dict):
            raise ValueError(f"{row['onset_id']}: observed evidence is required")
    output = {
        "run_id": manifest["run_id"],
        "stage": "enemy_onsets",
        "annotation_session_id": "merged-enemy-onset-workers",
        "model": "mixed-workers",
        "reasoning_effort": "mixed",
        "worker_provenance": [
            {
                key: document.get(key)
                for key in (
                    "stage",
                    "target_range",
                    "annotation_session_id",
                    "model",
                    "reasoning_effort",
                )
            }
            for document in documents
        ],
        "onsets": sorted(onsets, key=lambda row: row["event_frame_index"]),
    }
    atomic_write_json(run_dir / "enemy_onsets.json", output)
    print(json.dumps({"enemy_onsets": len(onsets)}))


if __name__ == "__main__":
    main()
