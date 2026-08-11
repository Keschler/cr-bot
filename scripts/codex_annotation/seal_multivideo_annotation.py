from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_batch import seal_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash-seal a complete multi-video blind annotation batch."
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve()
    policy_path = (args.policy or batch_dir / "frozen_batch_policy.json").resolve()
    seal = seal_batch(
        batch_dir=batch_dir,
        policy_path=policy_path,
        seal_path=batch_dir / "BATCH_SEALED.json",
    )
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    main()
