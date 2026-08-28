from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np

from cr_bot.paths import CACHE_DIR, KATACR_DATASET_ROOT, KATACR_ROOT, REPO_ROOT


MPLCONFIGDIR = CACHE_DIR / "matplotlib"
ULTRALYTICS_CONFIG_DIR = CACHE_DIR / "ultralytics"

_BOOTSTRAPPED = False


def bootstrap_katacr_runtime() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    if str(KATACR_ROOT) not in sys.path:
        sys.path.insert(0, str(KATACR_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    katacr_dataset_env = os.environ.get("KATACR_DATASET_PATH")
    if not katacr_dataset_env or not Path(katacr_dataset_env).exists():
        os.environ["KATACR_DATASET_PATH"] = str(KATACR_DATASET_ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid

    _patch_ultralytics_results_compat()
    _patch_ultralytics_track_compat()
    _patch_ultralytics_plotting_compat()
    _BOOTSTRAPPED = True


def load_split_part_tools():
    bootstrap_katacr_runtime()
    from katacr.build_dataset.utils.split_part import process_part, ratio2name

    return process_part, ratio2name


def load_label_maps():
    bootstrap_katacr_runtime()
    from katacr.constants.label_list import idx2unit, unit2idx

    return idx2unit, unit2idx


def _patch_ultralytics_track_compat() -> None:
    import ultralytics.trackers.track as track

    if hasattr(track, "yaml_load"):
        return
    if hasattr(track, "YAML"):
        track.yaml_load = track.YAML.load
        return
    from ultralytics.utils import YAML

    track.YAML = YAML
    track.yaml_load = YAML.load


def _patch_ultralytics_results_compat() -> None:
    """Keep the vendored KataCR result class importable across Ultralytics releases.

    KataCR imports ``LetterBox`` from ``ultralytics.engine.results``.  Newer
    Ultralytics releases expose the same class from ``ultralytics.data.augment``
    instead, while older releases still provide the original location.  Add a
    compatibility alias only when the old location is missing so both layouts
    remain supported.
    """
    import ultralytics.engine.results as results

    if hasattr(results, "LetterBox"):
        return

    from ultralytics.data.augment import LetterBox

    results.LetterBox = LetterBox


def _patch_ultralytics_plotting_compat() -> None:
    import ultralytics.utils.plotting as plotting

    if not hasattr(plotting, "contextlib"):
        plotting.contextlib = contextlib
