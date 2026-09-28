import importlib.util
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


# A test module whose third-party dependency is not installed in the current
# environment must not abort collection for the whole suite.  ``pytest`` treats
# an import error during collection as a fatal error, so an optional model
# stack (torchvision for the tower-HP OCR models, scipy for the audio and
# mining features) silently prevented every simulator test from running.  Each
# entry names the module and the distribution it needs; the module is ignored
# only while that distribution is genuinely absent, and it still runs in any
# environment that has it (including CI, which installs requirements.txt).
_OPTIONAL_MODULE_REQUIREMENTS = {
    "test_audio_dataset.py": "scipy",
    "test_mining_pipeline.py": "scipy",
    "test_own_actions_raw_hand.py": "torchvision",
    "test_own_slot_intervals.py": "torchvision",
    "test_tower_hp_crnn_integration.py": "torchvision",
    "test_tower_hp_ocr.py": "torchvision",
}

# Untracked in-progress modules whose subject code has left the working tree.
# ``test_own_actions_log.py`` imports ``capture/game_state.py`` and
# ``capture/trackers/own_actions.py``; that capture layout was reorganised and
# those files no longer exist (only ``capture/trackers/__pycache__`` remains),
# so the module cannot be repaired by an import edit alone.
_UNTRACKED_SUBJECT_MISSING = ("test_own_actions_log.py",)

collect_ignore = sorted(
    {module_name for module_name, distribution in _OPTIONAL_MODULE_REQUIREMENTS.items()
     if importlib.util.find_spec(distribution) is None}
    | set(_UNTRACKED_SUBJECT_MISSING)
)
