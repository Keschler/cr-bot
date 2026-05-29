from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KATACR_ROOT = ROOT / "vendor/external/KataCR"
SEED_ROOT = ROOT / "data/seed_dataset"

os.environ.setdefault("KATACR_DATASET_PATH", str(SEED_ROOT))
sys.path.insert(0, str(KATACR_ROOT))

from katacr.yolov8.model_setup import MultiModelSetup  # noqa: E402


def main() -> None:
    setup = MultiModelSetup(auto=True, verbose=False)
    setup.setup_config_files()


if __name__ == "__main__":
    main()
