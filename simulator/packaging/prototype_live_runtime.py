"""Runtime setup for the frozen V4 live executable."""

from pathlib import Path
import os
import sys


_BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", Path.cwd()))

# The build spec puts exactly the selected V4 state dict under this directory.
# Make that explicit build input available to the controller without baking a
# historical experiment path into source code.
if not os.environ.get("CR_BOT_V4_CHECKPOINT", "").strip():
    _bundled_checkpoints = sorted((_BUNDLE_ROOT / "outputs" / "v4").glob("*.pt"))
    if len(_bundled_checkpoints) == 1:
        os.environ["CR_BOT_V4_CHECKPOINT"] = str(_bundled_checkpoints[0])

# KataCR imports optional training components unless this flag is present.
# The live executable only needs the inference path.
os.environ.setdefault("KATACR_INFERENCE_ONLY", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "prototype-live-mpl")
)
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "prototype-live-yolo")
)

# ``cr_bot.paths`` normally discovers the repository by walking through the
# checkout's ``src/cr_bot`` layout.  A frozen application has ``cr_bot`` at
# the bundle root, so seed the vendored KataCR import path before the app
# starts.  prototype_controller performs the same path rebase defensively
# after importing the module.
_katacr_root = _BUNDLE_ROOT / "vendor" / "external" / "KataCR"
if _katacr_root.is_dir() and str(_katacr_root) not in sys.path:
    sys.path.insert(0, str(_katacr_root))
