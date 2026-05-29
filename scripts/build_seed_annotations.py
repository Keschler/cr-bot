from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KATACR_ROOT = ROOT / "vendor/external/KataCR"
SEED_ROOT = ROOT / "data/seed_dataset"

os.environ.setdefault("KATACR_DATASET_PATH", str(SEED_ROOT))
sys.path.insert(0, str(KATACR_ROOT))

from katacr.build_dataset.label_builder import LabelBuilder  # noqa: E402


def main() -> None:
    if not SEED_ROOT.exists():
        raise FileNotFoundError(
            f"Seed dataset not found at {SEED_ROOT}. Run scripts/prepare_seed_dataset.py first.",
        )

    version_info = SEED_ROOT / "version_info"
    version_info.mkdir(parents=True, exist_ok=True)

    builder = LabelBuilder(path_dataset=SEED_ROOT)
    builder.path_const_dataset = version_info / "dataset.py"
    builder.build()


if __name__ == "__main__":
    main()
