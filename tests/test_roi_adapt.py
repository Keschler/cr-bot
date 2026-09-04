"""Per-video ROI adaptation tests (synthetic frames, no models)."""

import copy
import inspect
import math

import cv2
import numpy as np
import pytest

from cr_bot.domain.rois import ROIS
from cr_bot.vision import roi_adapt
from cr_bot.vision.roi_adapt import (
    adapt_rois_for_probe,
    normalize_crop_to,
    resolve_crop,
    scaled_rois,
    validate_and_merge,
)


def test_scaled_math_exact_1080x1920():
    got = scaled_rois(1080, 1920)
    assert set(got.keys()) == set(ROIS.keys())
    # Inverse-stretch: x*1.0, y*0.8, w*1.0, h*0.8, clamped.
    assert got["hand_card_slot_1"] == [230, 1616, 220, 240]
    assert got["next_card_slot"] == [40, 1808, 120, 100]
    assert got["match_timer"] == [920, 128, 130, 48]
    # timer_box width clamped (890+260 > 1080 -> 190).
    assert got["timer_box"] == [890, 80, 190, 104]
    assert got["opponent_king_health_text"] == [500, 136, 120, 28]
    assert got["elixir_fill_slot_1"] == [270, 1856, 100, 40]
    # Bottom bar is clamped to the native frame (1848+120 > 1920).
    assert got["elixir_bar"] == [240, 1848, 805, 72]
    assert got["battlefield"] == [30, 251, 1010, 1184]


def test_scaled_bedrock_identity():
    got = scaled_rois(1080, 2400)
    for key, (x, y, w, h) in ROIS.items():
        exp_w = min(w, 1080 - x)
        exp_h = min(h, 2400 - y)
        assert got[key] == [x, y, exp_w, exp_h]
    # Spot-check clamped bedrock edges.
    assert got["elixir_bar"] == [240, 2310, 805, 90]
    assert got["timer_box"] == [890, 100, 190, 130]


def test_scaled_clamping_tiny_frame():
    got = scaled_rois(200, 200)
    nw, nh = 200, 200
    for rect in got.values():
        x, y, w, h = rect
        assert 0 <= x < nw
        assert 0 <= y < nh
        assert w >= 1 and h >= 1
        assert x + w <= nw
        assert y + h <= nh


def test_scaled_bad_size_raises():
    for bad in [(0, 1920), (1080, 0), (-1, 1920), (1080, -5), ("1080", 1920),
                (1080, None), (None, None), (True, 1920), (1080.0, 1920)]:
        with pytest.raises(ValueError):
            scaled_rois(*bad)


def test_validate_merge_defaults_and_overlay():
    base = scaled_rois(1080, 1920)
    assert validate_and_merge(None, 1080, 1920) == base
    merged = validate_and_merge({"hand_card_slot_1": [10, 20, 30, 40]}, 1080, 1920)
    assert merged["hand_card_slot_1"] == [10, 20, 30, 40]
    # Unknown keys ignored.
    merged2 = validate_and_merge({"nope": [1, 2, 3, 4]}, 1080, 1920)
    assert "nope" not in merged2
    assert merged2 == base
    # Non-finite / malformed values ignored, not raised.
    merged3 = validate_and_merge(
        {
            "hand_card_slot_1": [float("nan"), 0, 10, 10],
            "hand_card_slot_2": [0, 0, float("inf"), 10],
            "hand_card_slot_3": "bad",
            "hand_card_slot_4": [1, 2, 3],
        },
        1080,
        1920,
    )
    assert merged3["hand_card_slot_1"] == base["hand_card_slot_1"]
    assert merged3["hand_card_slot_2"] == base["hand_card_slot_2"]
    # Out-of-bounds client rect is clamped in-bounds.
    merged4 = validate_and_merge({"match_timer": [1000, 1900, 500, 500]}, 1080, 1920)
    x, y, w, h = merged4["match_timer"]
    assert x + w <= 1080 and y + h <= 1920


def test_validate_merge_bad_input_raises():
    with pytest.raises(ValueError):
        validate_and_merge(None, 0, 1920)
    with pytest.raises(ValueError):
        validate_and_merge(None, 1080, -1)
    with pytest.raises(ValueError):
        validate_and_merge(None, "x", 1920)
    with pytest.raises(ValueError):
        validate_and_merge([1, 2, 3], 1080, 1920)


def test_no_global_rois_mutation():
    before = copy.deepcopy(dict(ROIS))
    scaled_rois(1080, 1920)
    validate_and_merge({"hand_card_slot_1": [1, 2, 3, 4]}, 1080, 1920)
    blank = np.zeros((1920, 1080, 3), dtype=np.uint8)
    try:
        adapt_rois_for_probe(blank)
    except Exception:
        pass
    assert dict(ROIS) == before


def _synthetic_probe_frame():
    nw, nh = 1080, 1920
    img = np.zeros((nh, nw, 3), dtype=np.uint8)
    # Fake elixir bar (yellow) in bottom 18% strip.
    bar = [200, 1780, 700, 80]
    cv2.rectangle(
        img,
        (bar[0], bar[1]),
        (bar[0] + bar[2], bar[1] + bar[3]),
        (0, 255, 255),
        thickness=-1,
    )
    # Fake hand row: 4 white cards in lower third.
    row = [150, 1500, 780, 220]
    for i in range(4):
        cx = row[0] + i * 195
        cv2.rectangle(
            img, (cx, row[1]), (cx + 190, row[1] + row[3]), (255, 255, 255), thickness=-1
        )
        # Dark inner border so Canny sees strong edges.
        cv2.rectangle(
            img, (cx, row[1]), (cx + 190, row[1] + row[3]), (0, 0, 0), thickness=3
        )
    # Fake timer: dark pill with bright fake text in top-right search region
    # (solid white rectangles are rejected as mostly-bright blobs).
    timer = [880, 80, 130, 60]
    cv2.rectangle(
        img,
        (timer[0], timer[1]),
        (timer[0] + timer[2], timer[1] + timer[3]),
        (10, 10, 10),
        thickness=-1,
    )
    cv2.rectangle(img, (895, 86, 100, 12), (255, 255, 255), thickness=-1)
    cv2.rectangle(img, (900, 104, 75, 20), (255, 255, 255), thickness=-1)
    return img, bar, row, timer


def _rect_dist(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def test_landmark_detectors_on_synthetic():
    img, bar, row, timer = _synthetic_probe_frame()
    entries, overlay_jpeg, warnings, (nw, nh) = adapt_rois_for_probe(img)
    assert (nw, nh) == (1080, 1920)
    by_name = {e["name"]: e for e in entries}
    assert set(by_name.keys()) == set(ROIS.keys())
    # Elixir bar snapped within tolerance.
    assert by_name["elixir_bar"]["source"] == "landmark"
    assert _rect_dist(by_name["elixir_bar"]["rect"], bar) <= 30
    # Fill slots subdivided across the snapped bar.
    for i in range(1, 11):
        e = by_name[f"elixir_fill_slot_{i}"]
        assert e["source"] == "landmark"
        bx, _, bw, _ = by_name["elixir_bar"]["rect"]
        sx, _, sw, _ = e["rect"]
        assert bx - 5 <= sx <= bx + bw + 5
        assert 1 <= sw <= bw
    # Digit anchored to bar left.
    digit = by_name["elixir_digit"]
    assert digit["source"] == "landmark"
    assert abs(digit["rect"][0] - (by_name["elixir_bar"]["rect"][0] + 45)) <= 40
    # Hand slots snapped.
    for i in range(1, 5):
        e = by_name[f"hand_card_slot_{i}"]
        assert e["source"] == "landmark"
        assert abs(e["rect"][1] - row[1]) <= 40
        assert abs(e["rect"][3] - row[3]) <= 60
    # Row spans roughly the drawn row.
    x0 = min(by_name[f"hand_card_slot_{i}"]["rect"][0] for i in range(1, 5))
    x1 = max(
        by_name[f"hand_card_slot_{i}"]["rect"][0]
        + by_name[f"hand_card_slot_{i}"]["rect"][2]
        for i in range(1, 5)
    )
    assert abs(x0 - row[0]) <= 40
    assert abs(x1 - (row[0] + row[2])) <= 60
    # Next slot left of row.
    nxt = by_name["next_card_slot"]
    assert nxt["source"] == "landmark"
    assert nxt["rect"][0] + nxt["rect"][2] <= x0 + 5
    # Timer snapped, bedrock aspect kept.
    tm = by_name["match_timer"]
    assert tm["source"] == "landmark"
    assert _rect_dist(tm["rect"], timer) <= 35
    assert tm["rect"][2] / max(1, tm["rect"][3]) == pytest.approx(
        130 / 60, rel=0.15
    )
    tb = by_name["timer_box"]
    assert tb["source"] == "landmark"
    # Tower-HP stays scaled.
    for key in ("opponent_king_health_text", "player_king_health_text"):
        assert by_name[key]["source"] in ("scaled", "native")
        assert by_name[key]["confidence"] == pytest.approx(0.6)
    # Confidence ordering: landmarks above scaled fallbacks.
    landmark_confs = [
        e["confidence"] for e in entries if e["source"] == "landmark"
    ]
    scaled_confs = [
        e["confidence"] for e in entries if e["source"] == "scaled"
    ]
    assert landmark_confs
    assert scaled_confs
    assert min(landmark_confs) > max(scaled_confs)
    # Overlay JPEG width <= 720.
    assert overlay_jpeg is not None
    decoded = cv2.imdecode(np.frombuffer(overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[1] <= 720


def _hsv_color(h, s, v):
    color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(color[0]), int(color[1]), int(color[2]))


def _synthetic_hud_frame():
    """832x1920 frame mimicking the HOG spectator clip HUD layout."""
    nw, nh = 832, 1920
    img = np.zeros((nh, nw, 3), dtype=np.uint8)
    # Arena-ish background (brown, no blue).
    img[:] = _hsv_color(12, 120, 130)
    # Blue hand dock with bottom edge at the frame bottom.
    dock = [140, 1620, 692, 300]
    cv2.rectangle(
        img,
        (dock[0], dock[1]),
        (dock[0] + dock[2], dock[1] + dock[3]),
        _hsv_color(106, 220, 210),
        thickness=-1,
    )
    # 4 card portraits on the dock.
    for i in range(4):
        cx = dock[0] + 20 + i * 168
        cv2.rectangle(
            img, (cx, 1650), (cx + 140, 1830), _hsv_color(20, 200, 220), thickness=-1
        )
    # Pink elixir track + white drop digit.
    track = [185, 1855, 600, 40]
    cv2.rectangle(
        img,
        (track[0], track[1]),
        (track[0] + track[2], track[1] + track[3]),
        _hsv_color(149, 200, 185),
        thickness=-1,
    )
    # Next icon left of the dock.
    nxt = [20, 1790, 80, 110]
    cv2.rectangle(
        img,
        (nxt[0], nxt[1]),
        (nxt[0] + nxt[2], nxt[1] + nxt[3]),
        _hsv_color(15, 180, 200),
        thickness=-1,
    )
    cv2.rectangle(
        img,
        (nxt[0], nxt[1]),
        (nxt[0] + nxt[2], nxt[1] + nxt[3]),
        (255, 255, 255),
        thickness=3,
    )
    # Dark timer pill with bright fake text, top-right.
    pill = [690, 55, 140, 60]
    cv2.rectangle(
        img,
        (pill[0], pill[1]),
        (pill[0] + pill[2], pill[1] + pill[3]),
        (10, 10, 10),
        thickness=-1,
    )
    cv2.rectangle(img, (710, 62, 100, 12), (255, 255, 255), thickness=-1)
    cv2.rectangle(img, (716, 90, 80, 22), (255, 255, 255), thickness=-1)
    return img, dock, track, nxt, pill


def test_landmark_detectors_on_hud_frame():
    img, dock, track, nxt, pill = _synthetic_hud_frame()
    entries, overlay_jpeg, warnings, (nw, nh) = adapt_rois_for_probe(img)
    assert (nw, nh) == (832, 1920)
    assert not warnings
    by_name = {e["name"]: e for e in entries}
    # Hand slots split the dock, on the cards.
    for i in range(1, 5):
        e = by_name[f"hand_card_slot_{i}"]
        assert e["source"] == "landmark"
        assert abs(e["rect"][1] - 1650) <= 40
        assert abs(e["rect"][0] + e["rect"][2] / 2 - (dock[0] + (i - 0.5) * dock[2] / 4)) <= 40
    x0 = by_name["hand_card_slot_1"]["rect"][0]
    # Elixir bar snapped to the pink track.
    assert by_name["elixir_bar"]["source"] == "landmark"
    assert _rect_dist(by_name["elixir_bar"]["rect"], track) <= 45
    # Next icon left of the hand row, near the drawn icon.
    got_nxt = by_name["next_card_slot"]
    assert got_nxt["source"] == "landmark"
    assert got_nxt["rect"][0] + got_nxt["rect"][2] <= x0 + 5
    assert _rect_dist(got_nxt["rect"], nxt) <= 60
    # Timer pill snapped with bedrock aspect.
    tm = by_name["match_timer"]
    assert tm["source"] == "landmark"
    assert _rect_dist(tm["rect"], pill) <= 40
    assert tm["rect"][2] / max(1, tm["rect"][3]) == pytest.approx(130 / 60, rel=0.15)
    assert overlay_jpeg is not None


def test_tall_blue_banner_falls_back():
    """A full-width blue region that is too tall is not the hand dock."""
    nw, nh = 832, 1920
    img = np.zeros((nh, nw, 3), dtype=np.uint8)
    img[:] = _hsv_color(12, 120, 130)
    # Bottom-half blue banner (like a VS/intro screen), far too tall.
    cv2.rectangle(
        img, (0, 960), (nw, nh), _hsv_color(106, 220, 210), thickness=-1
    )
    entries, _, warnings, _ = adapt_rois_for_probe(img)
    by_name = {e["name"]: e for e in entries}
    assert by_name["hand_card_slot_1"]["source"] == "scaled"
    assert by_name["next_card_slot"]["source"] == "scaled"
    assert by_name["elixir_bar"]["source"] == "scaled"
    assert by_name["match_timer"]["source"] == "scaled"
    assert len(warnings) >= 3


def test_fail_soft_garbage_frame():
    garbage = np.full((600, 800, 3), 127, dtype=np.uint8)
    entries, overlay_jpeg, warnings, (nw, nh) = adapt_rois_for_probe(garbage)
    assert (nw, nh) == (800, 600)
    assert all(e["source"] == "scaled" for e in entries)
    assert all(e["confidence"] == pytest.approx(0.6) for e in entries)
    assert len(warnings) >= 3
    assert overlay_jpeg is not None


def test_probe_unreadable_raises():
    with pytest.raises(ValueError):
        adapt_rois_for_probe(None)
    with pytest.raises(ValueError):
        adapt_rois_for_probe(np.zeros((0, 0, 3), dtype=np.uint8))


def test_normalize_crop_aspect_and_size():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :100] = (255, 0, 0)
    img[:, 100:] = (0, 255, 0)
    out = normalize_crop_to(img, 130, 60)
    assert out.shape[0] == 60 and out.shape[1] == 130
    # Already-correct size stays exact.
    img2 = np.zeros((60, 130, 3), dtype=np.uint8)
    out2 = normalize_crop_to(img2, 130, 60)
    assert out2.shape == (60, 130, 3)


def test_resolve_crop_default_and_adapted_shapes():
    canvas = np.zeros((2400, 1080, 3), dtype=np.uint8)
    default = resolve_crop(canvas, "hand_card_slot_1")
    assert default.shape == (300, 220, 3)
    native = np.zeros((1920, 1080, 3), dtype=np.uint8)
    rois = scaled_rois(1080, 1920)
    adapted = resolve_crop(native, "hand_card_slot_1", rois=rois)
    assert adapted.shape == (300, 220, 3)
    adapted_timer = resolve_crop(native, "match_timer", rois=rois)
    assert adapted_timer.shape == (60, 130, 3)


def test_vision_signatures_thread_rois_keyword_only():
    from cr_bot.vision import cards, elixir, image_utils, timer, tower_hp
    from cr_bot.app import pipeline

    for fn in (
        cards.extract_hand_state,
        elixir.estimate_total_slots,
        elixir.extract_elixir,
        image_utils.detect_if_king_tower_activated,
        image_utils.detect_if_support_tower_alive,
        image_utils.detect_elixir_change,
        tower_hp.extract_tower_hp_crops,
        tower_hp.extract_tower_hp,
        timer.extract_time,
        timer.is_overtime,
    ):
        params = inspect.signature(fn).parameters
        assert "rois" in params
        assert params["rois"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["rois"].default is None
    assert "rois" in inspect.signature(timer._locate_timer).parameters
    assert "rois" in inspect.signature(elixir.read_elixir_value).parameters
    pipe_params = inspect.signature(pipeline.process_frame).parameters
    assert pipe_params["rois"].kind is inspect.Parameter.KEYWORD_ONLY
    assert pipe_params["native_frame"].kind is inspect.Parameter.KEYWORD_ONLY
