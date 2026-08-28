"""Compatibility checks for the vendored KataCR vision runtime."""

from __future__ import annotations

import importlib

import pytest


def test_katacr_letterbox_import_compatibility() -> None:
    """KataCR's legacy results import works with current Ultralytics layouts."""

    pytest.importorskip("ultralytics")

    results = importlib.import_module("ultralytics.engine.results")
    augment = importlib.import_module("ultralytics.data.augment")
    from cr_bot.vision.katacr_runtime import _patch_ultralytics_results_compat

    original = getattr(results, "LetterBox", None)
    _patch_ultralytics_results_compat()

    assert hasattr(results, "LetterBox")
    if original is None:
        assert results.LetterBox is augment.LetterBox
    else:
        assert results.LetterBox is original


def test_katacr_tracker_constructor_supports_current_and_legacy_signatures() -> None:
    """The vendored tracker adapter handles both Ultralytics constructor APIs."""

    pytest.importorskip("ultralytics")
    from cr_bot.vision.katacr_runtime import bootstrap_katacr_runtime

    bootstrap_katacr_runtime()
    from katacr.yolov8.custom_trackers import _build_tracker

    args = object()

    class ModernTracker:
        def __init__(self, *, args):
            self.args = args
            self.max_frames_lost = 5

    modern = _build_tracker(ModernTracker, args)
    assert modern.args is args
    assert modern.max_time_lost == 5

    class LegacyTracker:
        def __init__(self, *, args, frame_rate):
            self.args = args
            self.frame_rate = frame_rate

    legacy = _build_tracker(LegacyTracker, args, frame_rate=30)
    assert legacy.args is args
    assert legacy.frame_rate == 30
