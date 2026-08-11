from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.annotation_harness import finalize_annotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and lock a blind Codex video annotation."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    final, audit, lock = finalize_annotation(
        run_dir=args.run_dir,
        decisions_path=args.decisions or args.run_dir / "decisions.json",
        output_path=args.output,
        audit_output_path=args.audit_output,
    )
    print(f"locked final: {final}")
    print(f"locked audit: {audit}")
    print(f"lock record: {lock}")


if __name__ == "__main__":
    main()
