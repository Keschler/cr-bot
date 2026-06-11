import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Keep tests that use a cv2 fallback mock from replacing a real installation.
try:
    import cv2  # noqa: F401
except ImportError:
    pass
