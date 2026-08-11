import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# Keep tests that use a cv2 fallback mock from replacing a real installation.
try:
    import cv2  # noqa: F401
except ImportError:
    pass
