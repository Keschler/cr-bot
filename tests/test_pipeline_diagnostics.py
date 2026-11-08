"""Vision timing checks with synthetic frames and no detector/OCR models."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cr_bot.domain.frame_analysis import FrameAnalysisResult, TimerReading


@pytest.fixture
def pipeline_runtime(monkeypatch):
    calls = []
    clock = [0.0]
    frame = np.zeros((16, 8, 3), dtype=np.uint8)
    native = np.ones((12, 6, 3), dtype=np.uint8)
    arena = frame[4:12, 2:6]
    rendered = frame.copy()

    def operation(name, result):
        def run(*args, **kwargs):
            clock[0] += 0.001
            calls.append((name, args, kwargs))
            return result
        return run

    draw_boxes = operation("draw_boxes", rendered)
    split_tools = (
        operation("process_part", (arena, (0.25, 0.25, 0.5, 0.5))),
        operation("ratio2name", "synthetic"),
    )
    dependencies = {
        "cards": {"extract_hand_state": operation("hand_state", {"card_1": "hog-rider"})},
        "elixir": {"extract_elixir": operation("elixir", {"estimated_value": 4.0})},
        "health": {"estimate_health": operation("health", None)},
        "image_utils": {
            "detect_elixir_change": operation("elixir_change", {"covered": False}),
            "draw_rois": operation("draw_rois", frame),
        },
        "katacr_runtime": {"load_split_part_tools": lambda: split_tools},
        "timer": {
            "extract_timer": operation("extract_timer", TimerReading("2:30", False)),
            "parse_time_left_s": operation("parse_time_left_s", 150.0),
            "total_remaining_seconds": operation("total_remaining_seconds", 150.0),
        },
        "tower_hp": {"extract_tower_hp": operation("tower_hp", {"own_king": 4000})},
        "units": {
            "match_from_dict": operation("match_from_dict", "typed_match"),
            "match_troops_to_bars": operation("match_troops_to_bars", ["raw_match"]),
        },
        "yolo_runtime": {
            "convert_yolo": operation("convert_yolo", (["troop"], ["bar"])),
            "extract_clock_boxes": operation("clock_boxes", []),
            "extract_emote_boxes": operation("emote_boxes", []),
            "load_yolo_runtime": operation("load_yolo_runtime", (None, draw_boxes, None)),
            "remap_boxes_to_frame": operation("remap", "mapped_boxes"),
        },
    }
    for suffix, attributes in dependencies.items():
        name = f"cr_bot.vision.{suffix}"
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    # Execute an isolated pipeline module so the fixture neither imports the
    # actual model runtimes nor replaces other tests' real pipeline bindings.
    path = Path(__file__).resolve().parents[1] / "src" / "cr_bot" / "app" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("_pipeline_diagnostics", path)
    assert spec is not None and spec.loader is not None
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    result = SimpleNamespace(
        get_data=operation("get_data", "tracked_boxes"), untracked_data="untracked_boxes"
    )
    detector = SimpleNamespace(infer=operation("infer", result))
    return SimpleNamespace(
        pipeline=pipeline, calls=calls, frame=frame, native=native, detector=detector,
    )


@pytest.mark.parametrize("native_rois", [False, True])
@pytest.mark.parametrize("yolo_tower_hp", [False, True])
def test_opt_in_timings_preserve_pipeline_calls_and_results(pipeline_runtime, native_rois, yolo_tower_hp):
    runtime = pipeline_runtime
    rois = {"match_timer": [1, 2, 3, 4]} if native_rois else None
    kwargs = {
        "rois": rois,
        "native_frame": runtime.native if native_rois else None,
        "yolo_tower_hp_detections": yolo_tower_hp,
    }
    baseline = runtime.pipeline.process_frame(runtime.frame, runtime.detector, **kwargs)
    baseline_calls = runtime.calls[:]
    runtime.calls.clear()
    timed = runtime.pipeline.process_frame(runtime.frame, runtime.detector, collect_timing=True, **kwargs)
    assert baseline.vision_timing_ms == {}
    assert timed.vision_timing_ms == pytest.approx({
        "crop": 3.0, "detector": 2.0, "yolo_postprocess": 4.0,
        "draw_boxes": 1.0, "elixir": 2.0, "tower_hp": 1.0,
        "timer": 3.0, "hand_state": 1.0, "unit_postprocess": 4.0,
    })
    # Calls contain the same frame/ROI objects, arguments, and operation order.
    assert runtime.calls == baseline_calls
    for key in baseline:
        original, actual = baseline[key], timed[key]
        if isinstance(original, np.ndarray):
            np.testing.assert_array_equal(actual, original)
        else:
            assert actual == original
    assert timed.arena_px == (2, 4, 4, 8)
    by_name = {name: (args, options) for name, args, options in runtime.calls}
    hud_frame = runtime.native if native_rois else runtime.frame
    assert by_name["elixir"][0][0] is hud_frame
    assert by_name["hand_state"][0][0] is hud_frame
    assert by_name["extract_timer"][0][0] is hud_frame
    assert by_name["tower_hp"][0][0] is (runtime.frame if yolo_tower_hp else hud_frame)
    if native_rois:
        assert by_name["elixir"][1]["rois"] is rois


def test_default_pipeline_does_not_read_timing_clock(pipeline_runtime, monkeypatch):
    runtime = pipeline_runtime

    def unavailable_clock():
        raise AssertionError("timing must remain disabled by default")

    monkeypatch.setattr(runtime.pipeline.time, "perf_counter", unavailable_clock)
    result = runtime.pipeline.process_frame(runtime.frame, runtime.detector)
    assert result.vision_timing_ms == {}


def test_diagnostics_keep_legacy_constructor_mapping_and_replace_contract(pipeline_runtime):
    runtime = pipeline_runtime
    timed = runtime.pipeline.process_frame(runtime.frame, runtime.detector, collect_timing=True)
    legacy_kwargs = {
        field.name: getattr(timed, field.name)
        for field in fields(FrameAnalysisResult)
        if field.name != "vision_timing_ms"
    }
    first = FrameAnalysisResult(**legacy_kwargs)
    second = FrameAnalysisResult(**legacy_kwargs)
    first.vision_timing_ms["detector"] = 12.0
    assert second.vision_timing_ms == {}
    assert len(first) == len(timed) == 16
    assert list(first) == list(timed)
    assert "vision_timing_ms" not in list(timed)
    for changed in (
        replace(timed, rendered=None),
        timed.with_hand_state({"card_1": "cannon"}),
        timed.with_towers_hp({"own_king": 3900}),
        timed.with_clock(time_left_s=149.0, total_remaining_s=149.0),
    ):
        assert changed.vision_timing_ms == timed.vision_timing_ms
