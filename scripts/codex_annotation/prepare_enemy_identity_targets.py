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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create card-free enemy identity targets after side filtering."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    onsets = _read(run_dir / "enemy_onsets.json")
    side_decisions = _read(run_dir / "enemy_identities.json")
    by_id = {
        row["onset_id"]: row
        for row in onsets.get("onsets", [])
        if isinstance(row, dict)
    }
    targets = []
    for decision in side_decisions.get("decisions", []):
        if (
            not isinstance(decision, dict)
            or decision.get("event_exists") is not True
            or decision.get("side") != "enemy"
        ):
            continue
        onset = by_id.get(decision.get("onset_id"))
        if onset is None:
            raise ValueError(f"unknown onset {decision.get('onset_id')!r}")
        targets.append(
            {
                "onset_id": onset["onset_id"],
                "event_frame_index": onset["event_frame_index"],
                "kind": onset["kind"],
                "track_id": onset.get("track_id"),
                "verification_artifacts": (
                    onset["verification_artifacts"]
                    if onset["kind"] == "spell"
                    # Unit identity is much less reliable when the delayed
                    # full-arena views contain older actors. Keep the compact
                    # full onset sequence and its high-resolution marker-focus
                    # companion so the identity worker can anchor the new body
                    # before following it into the delayed views.
                    else onset["verification_artifacts"]
                ),
                "identity_frame_index": None,
                "identity_artifacts": [],
            }
        )
    output = {
        "run_id": onsets["run_id"],
        "stage": "enemy_identity_targets",
        "targets": targets,
    }
    path = run_dir / "enemy_identity_targets.json"
    atomic_write_json(path, output)
    print(json.dumps({"output": str(path), "targets": len(targets)}))


if __name__ == "__main__":
    main()
