from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one compact full-roster enemy card assignment package."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="Target document; defaults to RUN_DIR/enemy_identity_targets.json.",
    )
    parser.add_argument(
        "--roster-file",
        type=Path,
        help="Roster document; defaults to RUN_DIR/enemy_identity_roster.json.",
    )
    parser.add_argument("--output-prefix", default="cards")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    targets_path = (
        args.targets_file.resolve()
        if args.targets_file is not None
        else run_dir / "enemy_identity_targets.json"
    )
    roster_path = (
        args.roster_file.resolve()
        if args.roster_file is not None
        else run_dir / "enemy_identity_roster.json"
    )
    source = json.loads(
        targets_path.read_text(encoding="utf-8")
    )
    roster = json.loads(
        roster_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if roster.get("run_id") != source.get("run_id"):
        raise ValueError("identity roster run_id mismatch")
    spell_artifacts = [
        artifact
        for target in source["targets"]
        if target["kind"] == "spell"
        for artifact in target["verification_artifacts"]
    ]
    segment = manifest["segment"]
    target_range = [
        int(segment["start_frame"]),
        int(segment["end_frame_exclusive"]),
    ]
    output = {
        "run_id": source["run_id"],
        "stage": "enemy_identity_targets",
        "target_range": target_range,
        "targets": source["targets"],
        # run_model_worker deliberately attaches this compact allowlist instead
        # of recursively attaching every original six-frame identity sheet.
        "attached_images": roster["artifacts"] + spell_artifacts,
        "roster_sheets": roster["sheets"],
    }
    path = (
        args.output.resolve()
        if args.output is not None
        else run_dir
        / "work_packages"
        / (
            f"{args.output_prefix}-{target_range[0]:06d}-"
            f"{target_range[1]:06d}.json"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, output)
    print(
        json.dumps(
            {
                "output": str(path),
                "targets": len(source["targets"]),
                "attached_images": len(output["attached_images"]),
            }
        )
    )


if __name__ == "__main__":
    main()
