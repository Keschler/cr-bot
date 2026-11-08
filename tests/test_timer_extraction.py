import cv2
import numpy as np
import pytest

from cr_bot.vision import timer


def _assert_same(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_same(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("text,color", [("2:30", (255, 255, 255)), ("0:45", (0, 0, 255)), ("", (0, 0, 0))])
@pytest.mark.parametrize("adapted", [False, True])
def test_combined_timer_preserves_text_overtime_and_debug_with_one_search(monkeypatch, text, color, adapted):
    frame = np.zeros((2400, 1080, 3), dtype=np.uint8)
    cv2.putText(frame, text, (885, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    rois = {"match_timer": [880, 100, 195, 85], "timer_box": [860, 80, 215, 130]} if adapted else None
    original_debug = {}
    expected_text = timer.extract_time(frame, debug_steps=original_debug, rois=rois)
    expected_overtime = timer.is_overtime(frame, rois=rois)
    original_locate = timer._locate_timer
    calls = []

    def locate(image, *args, **kwargs):
        calls.append((image, kwargs.get("rois")))
        return original_locate(image, *args, **kwargs)

    monkeypatch.setattr(timer, "_locate_timer", locate)
    combined_debug = {}
    combined = timer.extract_timer(frame, debug_steps=combined_debug, rois=rois)
    assert combined.text == expected_text
    assert combined.overtime == expected_overtime
    assert len(calls) == 1
    assert calls[0][0] is frame and calls[0][1] is rois
    _assert_same(original_debug, combined_debug)


def test_combined_timer_does_not_cache_a_previous_frame(monkeypatch):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    calls = []

    def locate(frame, **kwargs):
        calls.append(frame)
        return {"time": "2:30" if frame is frames[0] else "0:10", "timer_frame": frame, "timer_box": frame}

    monkeypatch.setattr(timer, "_locate_timer", locate)
    monkeypatch.setattr(timer, "_read_timer_from_roi", lambda frame, **kwargs: "2:30" if frame is frames[0] else "0:10")
    monkeypatch.setattr(timer, "_measure_red_ratio", lambda frame, expert: 0.0 if frame is frames[0] else 1.0)
    monkeypatch.setattr(timer, "_measure_timer_text_red_ratio", lambda frame: 0.0)
    assert timer.extract_timer(frames[0]).text == "2:30"
    second = timer.extract_timer(frames[1])
    assert second.text == "0:10" and second.overtime is True
    assert len(calls) == 2
