from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_stages import checkpoint_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and checkpoint one stage of a blind Codex annotation. "
            "Changing an earlier stage invalidates all later checkpoints."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["release_review", "verification", "localization", "completeness"],
    )
    args = parser.parse_args()

    checkpoint = checkpoint_stage(args.run_dir, args.stage)
    print(
        f"checkpointed {args.stage}: "
        f"sha256={checkpoint['sha256']} "
        f"at={checkpoint['checkpointed_at']}"
    )


if __name__ == "__main__":
    main()
