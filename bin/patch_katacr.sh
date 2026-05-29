#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
KATACR_CONSTANT="$REPO_ROOT/vendor/external/KataCR/katacr/build_dataset/constant.py"

if [[ ! -f "$KATACR_CONSTANT" ]]; then
  echo "Missing KataCR constant file: $KATACR_CONSTANT" >&2
  echo "Did you clone with --recurse-submodules?" >&2
  exit 1
fi

python - "$KATACR_CONSTANT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

if "import os\n" not in text:
    text = text.replace("from pathlib import Path\n", "import os\nfrom pathlib import Path\n")

old = 'path_dataset = Path("/home/yy/Coding/datasets/Clash-Royale-Dataset")'
new = '''path_dataset = Path(os.environ.get(
  "KATACR_DATASET_PATH",
  "/home/yy/Coding/datasets/Clash-Royale-Dataset",
))'''

if old in text:
    text = text.replace(old, new)
elif "KATACR_DATASET_PATH" not in text:
    raise SystemExit(
        "Could not find the expected upstream path_dataset line. "
        "The submodule may have changed; patch it manually."
    )
else:
    print("KataCR dataset path patch already applied.")

path.write_text(text, encoding="utf-8")
PY

echo "KataCR is patched to honor KATACR_DATASET_PATH."
