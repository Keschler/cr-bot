"""Per-video ROI adaptation for non-1080x2400 frames.

Bedrock canvas is 1080x2400 (see ``cr_bot.domain.rois.ROIS``). The arena/YOLO
path is aspect-robust and stays on the normalized canvas. Only fixed-ROI
crops (hand, elixir, tower-HP text/bars, timer) are adapted via this module.

Notes on untouched paths (contract):
- ``domain/rois.py`` values are never mutated here.
- ``match_state.py`` ``game_start``/``in_game`` and timer/game-start logic
  stay on the canvas path; callers only route fixed-ROI extractions through
  ``resolve_crop`` with adapted ``rois`` when a native frame is available.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from cr_bot.domain.rois import ROIS

BEDROCK_W = 1080
BEDROCK_H = 2400

_FALLBACK_CONFIDENCE = 0.6


def _validate_native_size(native_w, native_h) -> tuple[int, int]:
    for value in (native_w, native_h):
        if isinstance(value, bool):
            raise ValueError(f"bad native size: {(native_w, native_h)!r}")
        try:
            import numbers

            if not isinstance(value, numbers.Integral):
                raise ValueError(f"bad native size: {(native_w, native_h)!r}")
        except ValueError:
            raise
        except Exception:
            raise ValueError(f"bad native size: {(native_w, native_h)!r}")
        if int(value) <= 0:
            raise ValueError(f"bad native size: {(native_w, native_h)!r}")
    return int(native_w), int(native_h)


def _clamp_rect(x, y, w, h, nw: int, nh: int) -> list[int]:
    x = int(round(float(x)))
    y = int(round(float(y)))
    w = int(round(float(w)))
    h = int(round(float(h)))
    x = max(0, min(x, nw - 1))
    y = max(0, min(y, nh - 1))
    w = max(1, min(w, nw - x))
    h = max(1, min(h, nh - y))
    return [x, y, w, h]


def scaled_rois(native_w, native_h) -> dict:
    """Inverse-stretch remap of bedrock ROIS into native pixels."""
    nw, nh = _validate_native_size(native_w, native_h)
    out: dict[str, list[int]] = {}
    for name, (x, y, w, h) in ROIS.items():
        nx = int(round(float(x) * nw / BEDROCK_W))
        ny = int(round(float(y) * nh / BEDROCK_H))
        nw_ = int(round(float(w) * nw / BEDROCK_W))
        nh_ = int(round(float(h) * nh / BEDROCK_H))
        out[name] = _clamp_rect(nx, ny, nw_, nh_, nw, nh)
    return out


def _is_finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        # Accept numpy numeric scalars as well.
        try:
            import numbers

            if not isinstance(value, numbers.Real):
                return False
        except Exception:
            return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_and_merge(client_rois, native_w, native_h) -> dict:
    """Merge client overrides onto scaled defaults.

    Unknown keys are ignored. Values that are not 4 finite numbers are
    ignored. Bad native size raises ``ValueError``.
    """
    nw, nh = _validate_native_size(native_w, native_h)
    merged = scaled_rois(nw, nh)
    if client_rois is None:
        return merged
    if not isinstance(client_rois, dict):
        raise ValueError("roi_set must be a dict or None")
    for key, value in client_rois.items():
        if key not in ROIS:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        if not all(_is_finite_number(v) for v in value):
            continue
        merged[key] = _clamp_rect(value[0], value[1], value[2], value[3], nw, nh)
    return merged


def normalize_crop_to(img, target_w, target_h):
    """Scale-to-fit preserving aspect, pad with BORDER_REPLICATE."""
    try:
        tw = int(target_w)
        th = int(target_h)
    except (TypeError, ValueError):
        raise ValueError("bad target size")
    if isinstance(target_w, bool) or isinstance(target_h, bool):
        raise ValueError("bad target size")
    if tw <= 0 or th <= 0:
        raise ValueError("bad target size")
    if img is None or not hasattr(img, "shape"):
        raise ValueError("unreadable crop image")
    try:
        h, w = img.shape[:2]
    except Exception:
        raise ValueError("unreadable crop image")
    if w <= 0 or h <= 0:
        raise ValueError("unreadable crop image")
    if img.size == 0:
        raise ValueError("unreadable crop image")
    scale = min(tw / float(w), th / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        resized = img
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_w = tw - new_w
    pad_h = th - new_h
    if pad_w <= 0 and pad_h <= 0:
        if resized.shape[1] == tw and resized.shape[0] == th:
            return resized.copy() if resized is img else resized
        # Exact-size fallback (should not happen given rounding above).
        return cv2.resize(resized, (tw, th), interpolation=cv2.INTER_LINEAR)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_REPLICATE
    )


def resolve_crop(frame, key, *, rois=None):
    """Crop for ``key`` on canvas (``rois`` None) or native (adapted)."""
    if rois is None:
        from cr_bot.vision.image_utils import crop as _crop

        return _crop(frame, ROIS[key])
    bedrock = ROIS[key]
    target_w, target_h = int(bedrock[2]), int(bedrock[3])
    rect = rois.get(key) if isinstance(rois, dict) else None
    if rect is None or not isinstance(rect, (list, tuple)) or len(rect) != 4:
        # Fallback: scaled position derived from the frame size itself.
        try:
            fh, fw = frame.shape[:2]
            rect = scaled_rois(int(fw), int(fh))[key]
        except Exception:
            from cr_bot.vision.image_utils import crop as _crop

            return _crop(frame, bedrock)
    try:
        x, y, w, h = (int(round(float(v))) for v in rect)
    except (TypeError, ValueError):
        from cr_bot.vision.image_utils import crop as _crop

        return _crop(frame, bedrock)
    try:
        fh, fw = frame.shape[:2]
    except Exception:
        from cr_bot.vision.image_utils import crop as _crop

        return _crop(frame, bedrock)
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    cropped = frame[y : y + h, x : x + w]
    return normalize_crop_to(cropped, target_w, target_h)


# ---------------------------------------------------------------------------
# Probe-time landmark refinement
# ---------------------------------------------------------------------------

_ELIXIR_KEYS = ["elixir_bar", "elixir_digit"] + [
    f"elixir_fill_slot_{i}" for i in range(1, 11)
]
_HAND_KEYS = [f"hand_card_slot_{i}" for i in range(1, 5)] + ["next_card_slot"]
_TIMER_KEYS = ["match_timer", "timer_box"]


def _detect_hand_dock(native_bgr, nw: int, nh: int) -> tuple[int, int, int, int]:
    """Snap the blue bottom HUD dock holding hand cards + elixir bar.

    The dock is a large blue panel spanning most of the frame width with
    its bottom edge at the frame bottom (same HUD on all aspect ratios).
    Returns the dock rect ``(x, y, w, h)`` in native pixels or raises
    ``RuntimeError`` so callers fall back to scaled ROIs.
    """
    y0 = int(nh * 0.70)
    strip = native_bgr[y0:nh]
    if strip.size == 0 or strip.shape[0] < 32 or strip.shape[1] < 32:
        raise RuntimeError("dock strip too small")
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.array([95, 110, 50], dtype=np.uint8),
        np.array([122, 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((15, 15), dtype=np.uint8)
    merged = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("no blue dock mass")
    best = None
    best_area = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect = _clamp_rect(x, y0 + y, w, h, nw, nh)
        rx, ry, rw, rh = rect
        if rw < nw * 0.55:
            continue
        if rx > nw * 0.25:
            continue
        if ry < int(nh * 0.70) or ry > int(nh * 0.93):
            continue
        if rh < int(nh * 0.08) or rh > int(nh * 0.25):
            continue
        if ry + rh < int(nh * 0.97):
            continue
        area = float(rw * rh)
        if area > best_area:
            best_area = area
            best = rect
    if best is None:
        raise RuntimeError("no dock-shaped blue mass")
    bx, by, bw, bh = best
    # Confirm the rect is actually blue-filled (not a stray edge blob).
    dock_crop = native_bgr[by : by + bh, bx : bx + bw]
    if dock_crop.size == 0:
        raise RuntimeError("dock crop empty")
    dock_hsv = cv2.cvtColor(dock_crop, cv2.COLOR_BGR2HSV)
    blue_fill = cv2.inRange(
        dock_hsv,
        np.array([95, 110, 50], dtype=np.uint8),
        np.array([122, 255, 255], dtype=np.uint8),
    )
    fill_ratio = float(np.count_nonzero(blue_fill)) / float(bw * bh)
    if fill_ratio < 0.25:
        raise RuntimeError("dock blue fill too low")
    return best


def _refine_elixir_from_dock(
    nw: int, nh: int, dock: tuple[int, int, int, int], entries: dict
) -> None:
    """Derive elixir ROIs from the snapped dock (fixed HUD proportions)."""
    dx, dy, dw, dh = dock
    ex = dx + int(round(0.05 * dw))
    ey = dy + int(round(0.78 * dh))
    ew = max(1, int(round(0.90 * dw)))
    eh = max(1, int(round(0.13 * dh)))
    bar = _clamp_rect(ex, ey, ew, eh, nw, nh)
    bx, by, bw, bh = bar
    if bw < nw * 0.25:
        raise RuntimeError("dock-derived elixir bar too narrow")
    entries["elixir_bar"] = {
        "name": "elixir_bar",
        "rect": bar,
        "source": "landmark",
        "confidence": 0.85,
    }
    for i in range(10):
        # Bedrock-relative insets within the track (slot pitch 78/805).
        slot = _clamp_rect(
            bx + int(round(bw * (0.037 + 0.097 * i))),
            by + int(round(bh * 0.067)),
            max(1, int(round(bw * 0.124))),
            max(1, int(round(bh * 0.33))),
            nw,
            nh,
        )
        # Keep slots inside the bar horizontally.
        slot[0] = max(bx, min(slot[0], bx + bw - 1))
        slot[2] = max(1, min(slot[2], bx + bw - slot[0]))
        key = f"elixir_fill_slot_{i + 1}"
        entries[key] = {
            "name": key,
            "rect": slot,
            "source": "landmark",
            "confidence": 0.8,
        }
    digit_rect = _clamp_rect(
        bx,
        by - int(round(bh * 0.5)),
        max(1, int(round(bw * 0.10))),
        max(1, int(round(bh * 1.5))),
        nw,
        nh,
    )
    entries["elixir_digit"] = {
        "name": "elixir_digit",
        "rect": digit_rect,
        "source": "landmark",
        "confidence": 0.8,
    }
def _refine_elixir(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap the elixir bar: direct HSV mass first, dock-derived fallback."""
    try:
        _refine_elixir_hsv(native_bgr, nw, nh, entries)
        return
    except RuntimeError:
        pass
    dock = _detect_hand_dock(native_bgr, nw, nh)
    _refine_elixir_from_dock(nw, nh, dock, entries)


def _refine_elixir_hsv(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap elixir bar via yellow/pink HSV mass in bottom 18% strip."""
    y0 = int(nh * 0.82)
    strip = native_bgr[y0:nh]
    if strip.size == 0 or strip.shape[0] < 8:
        raise RuntimeError("elixir strip too small")
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv, np.array([20, 80, 80], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)
    )
    pink = cv2.inRange(
        hsv, np.array([125, 40, 80], dtype=np.uint8), np.array([175, 255, 255], dtype=np.uint8)
    )
    mask = cv2.bitwise_or(yellow, pink)
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no elixir HSV mass")
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    strip_area = float(strip.shape[0] * strip.shape[1])
    if area < max(2000.0, 0.005 * strip_area):
        raise RuntimeError("elixir mass too small")
    x, y, w, h = cv2.boundingRect(best)
    bar = _clamp_rect(x, y0 + y, w, h, nw, nh)
    bx, by, bw, bh = bar
    if bw < nw * 0.25:
        raise RuntimeError("elixir bar too narrow")
    if not (10 <= bh <= max(24, int(nh * 0.09))):
        raise RuntimeError("elixir bar height implausible")
    if by < int(nh * 0.72):
        raise RuntimeError("elixir bar too high")
    entries["elixir_bar"] = {
        "name": "elixir_bar",
        "rect": bar,
        "source": "landmark",
        "confidence": 0.9,
    }
    slot_h = max(1, int(round(bh * 50.0 / 150.0)))
    slot_y = by + max(0, int(round(bh * 10.0 / 150.0)))
    # Even subdivision across the snapped bar width.
    for i in range(10):
        sx = bx + int(i * bw / 10)
        ex = bx + int((i + 1) * bw / 10)
        rect = _clamp_rect(sx, slot_y, max(1, ex - sx), slot_h, nw, nh)
        key = f"elixir_fill_slot_{i + 1}"
        entries[key] = {
            "name": key,
            "rect": rect,
            "source": "landmark",
            "confidence": 0.85,
        }
    digit_w = max(4, int(round(bw * 53.0 / 805.0)))
    digit_h = max(4, int(round(bh * 53.0 / 150.0)))
    digit_x = bx + int(round(bw * 55.0 / 805.0))
    digit_y = by
    digit_rect = _clamp_rect(digit_x, digit_y, digit_w, digit_h, nw, nh)
    entries["elixir_digit"] = {
        "name": "elixir_digit",
        "rect": digit_rect,
        "source": "landmark",
        "confidence": 0.85,
    }


def _row_band_from_edges(edges: np.ndarray) -> tuple[int, int] | None:
    row_density = edges.mean(axis=1).astype(np.float64) / 255.0
    kernel = np.ones(7, dtype=np.float64) / 7.0
    smooth = np.convolve(row_density, kernel, mode="same")
    mean = float(smooth.mean())
    std = float(smooth.std())
    thresh = max(0.02, mean + 0.5 * std)
    above = smooth > thresh
    if not np.any(above):
        return None
    # Largest contiguous run.
    best_start = best_len = 0
    cur_start = None
    for idx, flag in enumerate(above):
        if flag and cur_start is None:
            cur_start = idx
        if not flag and cur_start is not None:
            length = idx - cur_start
            if length > best_len:
                best_len, best_start = length, cur_start
            cur_start = None
    if cur_start is not None:
        length = len(above) - cur_start
        if length > best_len:
            best_len, best_start = length, cur_start
    if best_len <= 0:
        return None
    return best_start, best_start + best_len


def _col_extent_from_edges(band_edges: np.ndarray) -> tuple[int, int] | None:
    if band_edges.size == 0:
        return None
    col_density = band_edges.mean(axis=0).astype(np.float64) / 255.0
    kernel = np.ones(7, dtype=np.float64) / 7.0
    smooth = np.convolve(col_density, kernel, mode="same")
    mean = float(smooth.mean())
    std = float(smooth.std())
    thresh = max(0.02, mean + 0.5 * std)
    idx = np.where(smooth > thresh)[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)


def _refine_hand(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap hand row: blue dock split first, edge band as fallback."""
    try:
        dock = _detect_hand_dock(native_bgr, nw, nh)
    except RuntimeError:
        _refine_hand_edges(native_bgr, nw, nh, entries)
        return
    dx, dy, dw, dh = dock
    slot_y = dy + int(round(0.10 * dh))
    slot_h = max(1, int(round(0.60 * dh)))
    for i in range(4):
        sx = dx + int(i * dw / 4)
        ex = dx + int((i + 1) * dw / 4)
        rect = _clamp_rect(sx, slot_y, max(1, ex - sx), slot_h, nw, nh)
        key = f"hand_card_slot_{i + 1}"
        entries[key] = {
            "name": key,
            "rect": rect,
            "source": "landmark",
            "confidence": 0.9,
        }
    _derive_next_from_row(nw, nh, entries, dx, slot_y, dw, slot_h, confidence=0.85)


def _derive_next_from_row(
    nw: int,
    nh: int,
    entries: dict,
    rx: int,
    ry: int,
    rw: int,
    rh: int,
    confidence: float = 0.8,
) -> None:
    """Place the Next icon left of the snapped hand row (fixed HUD layout)."""
    avg_slot_w = max(1, rw // 4)
    next_w = max(8, int(round(avg_slot_w * 120.0 / 220.0)))
    next_h = max(8, int(round(rh * 125.0 / 300.0)))
    next_x = max(0, rx - next_w - int(nw * 0.02))
    next_y = ry + int(round(rh * (2260 - 2020) / 300.0))
    next_rect = _clamp_rect(next_x, next_y, next_w, next_h, nw, nh)
    entries["next_card_slot"] = {
        "name": "next_card_slot",
        "rect": next_rect,
        "source": "landmark",
        "confidence": float(confidence),
    }


def _refine_hand_edges(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap hand row via horizontal edge-density band in lower third."""
    y0 = int(nh * 0.66)
    strip = native_bgr[y0:nh]
    if strip.size == 0 or strip.shape[0] < 32 or strip.shape[1] < 32:
        raise RuntimeError("hand strip too small")
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 80, 160)
    # Horizontal dilation merges adjacent card edges into one dense band
    # while keeping the hand row separate from the lower elixir bar.
    kernel = np.ones((7, 25), dtype=np.uint8)
    dense = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no hand edge band")
    elixir_rect = None
    existing = entries.get("elixir_bar")
    if isinstance(existing, dict) and existing.get("source") == "landmark":
        try:
            elixir_rect = [int(v) for v in existing["rect"]]
        except (TypeError, ValueError):
            elixir_rect = None
    candidates = []
    bottom_min = int(nh * 0.85)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        full = _clamp_rect(x, y0 + y, w, h, nw, nh)
        fx, fy, fw, fh = full
        if fw < int(nw * 0.30):
            continue
        if not (40 <= fh <= max(80, int(nh * 0.22))):
            continue
        if float(cv2.contourArea(contour)) < 5000:
            continue
        # The hand dock sits at the frame bottom; arena bands merged by
        # dilation (king tower, rocks) end higher up and must not win.
        if fy + fh < bottom_min:
            continue
        # Hand sits above the very-bottom elixir strip; this keeps the
        # elixir bar blob from being mistaken for the hand row.
        if fy >= int(nh * 0.88):
            continue
        if elixir_rect is not None:
            ex, ey, ew, eh = elixir_rect
            ix0, iy0 = max(fx, ex), max(fy, ey)
            ix1, iy1 = min(fx + fw, ex + ew), min(fy + fh, ey + eh)
            if ix1 > ix0 and iy1 > iy0:
                continue
        candidates.append((float(cv2.contourArea(contour)), full))
    if not candidates:
        raise RuntimeError("no hand row candidate")
    candidates.sort(key=lambda item: item[0], reverse=True)
    rx, ry, rw, rh = candidates[0][1]
    # Derive 4 equal slots covering the snapped row.
    for i in range(4):
        sx = rx + int(i * rw / 4)
        ex = rx + int((i + 1) * rw / 4)
        rect = _clamp_rect(sx, ry, max(1, ex - sx), rh, nw, nh)
        key = f"hand_card_slot_{i + 1}"
        entries[key] = {
            "name": key,
            "rect": rect,
            "source": "landmark",
            "confidence": 0.85,
        }
    _derive_next_from_row(nw, nh, entries, rx, ry, rw, rh)


def _refine_timer(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap the timer: OCR-validated locator first, shape pill fallback.

    ``timer._locate_timer`` only accepts candidates that actually read as a
    match clock (``M:SS``), which rejects skull/tower highlights. Some
    spectator layouts (small colon) defeat OCR validation, so when the
    locator finds no readable clock we fall back to the dark-pill shape
    detector before giving up to scaled ROIs.
    """
    try:
        _refine_timer_located(native_bgr, nw, nh, entries)
        return
    except RuntimeError:
        pass
    _refine_timer_pill(native_bgr, nw, nh, entries)


def _refine_timer_located(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap the timer via the shared OCR-validated locator.

    Runs once per probe frame, so full template matching is acceptable.
    Raises ``RuntimeError`` when no readable clock is found.
    """
    try:
        from cr_bot.vision.timer import _locate_timer
    except ImportError as error:
        raise RuntimeError("timer locator is not importable") from error
    try:
        located = _locate_timer(native_bgr)
    except Exception as error:
        raise RuntimeError(f"timer locator failed: {error}") from error
    bbox = located.get("frame_bbox") if isinstance(located, dict) else None
    if bbox is None or len(bbox) != 4:
        raise RuntimeError("timer locator found no readable clock")
    try:
        box = _clamp_rect(
            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), nw, nh
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("timer locator returned a bad box") from error
    _store_timer_entries(nw, nh, entries, tuple(box))


def _store_timer_entries(nw: int, nh: int, entries: dict, box: tuple[int, int, int, int]) -> None:
    """Store match_timer/timer_box from a snapped pill box (bedrock aspect)."""
    bx, by, bw, bh = box
    # Keep bedrock aspect (130x60) for the snapped timer.
    timer_w = max(20, int(bw))
    timer_h = max(8, int(round(timer_w * 60.0 / 130.0)))
    timer_x = bx
    timer_y = by + (bh - timer_h) // 2
    timer_rect = _clamp_rect(timer_x, timer_y, timer_w, timer_h, nw, nh)
    tx, ty, tw, th = timer_rect
    if not (30 <= tw <= max(60, int(nw * 0.35))):
        raise RuntimeError("timer width implausible")
    if not (12 <= th <= max(30, int(nh * 0.08))):
        raise RuntimeError("timer height implausible")
    entries["match_timer"] = {
        "name": "match_timer",
        "rect": timer_rect,
        "source": "landmark",
        "confidence": 0.9,
    }
    box_w = max(8, int(round(tw * 260.0 / 130.0)))
    box_h = max(8, int(round(th * 130.0 / 60.0)))
    box_x = tx + int(round(tw * (890 - 920) / 130.0))
    box_y = ty + int(round(th * (100 - 160) / 60.0))
    box_rect = _clamp_rect(box_x, box_y, box_w, box_h, nw, nh)
    entries["timer_box"] = {
        "name": "timer_box",
        "rect": box_rect,
        "source": "landmark",
        "confidence": 0.85,
    }


def _refine_timer_pill(native_bgr, nw: int, nh: int, entries: dict) -> None:
    """Snap the 'Time left' pill via its bright text on a dark box.

    Shape-only fallback for spectator layouts whose small colon defeats
    OCR validation in ``_locate_timer``. Text clusters are grouped and the
    enclosing box must be dark-filled; bright blobs without a dark box
    (skull glints) are rejected.
    """
    sx = int(nw * 0.68)
    sy = 0
    sw = max(1, nw - sx)
    sh = max(1, int(nh * 0.11))
    search = native_bgr[sy : sy + sh, sx : sx + sw]
    if search.size == 0 or search.shape[0] < 16 or search.shape[1] < 16:
        raise RuntimeError("timer search too small")
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 15), dtype=np.uint8)
    merged = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groups: list[list[int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if float(cv2.contourArea(contour)) < 300 or w < 30 or h < 8:
            continue
        cx, cy = sx + x + w / 2.0, sy + y + h / 2.0
        if cx < nw * 0.70 or cy > nh * 0.10:
            continue
        groups.append([x, y, w, h])
    if not groups:
        raise RuntimeError("no timer text")
    # Merge nearby text lines ("Time left:" + digits) into one block.
    groups.sort(key=lambda g: g[0] + g[1])
    merged_groups: list[list[int]] = []
    for g in groups:
        placed = False
        for m in merged_groups:
            mx0, my0 = m[0] - 30, m[1] - 30
            mx1, my1 = m[0] + m[2] + 30, m[1] + m[3] + 30
            if g[0] < mx1 and g[0] + g[2] > mx0 and g[1] < my1 and g[1] + g[3] > my0:
                m[0], m[1] = min(m[0], g[0]), min(m[1], g[1])
                m[2] = max(m[0] + m[2], g[0] + g[2]) - m[0]
                m[3] = max(m[1] + m[3], g[1] + g[3]) - m[1]
                placed = True
                break
        if not placed:
            merged_groups.append(list(g))
    merged_groups.sort(
        key=lambda m: float(m[2] * m[3]), reverse=True
    )
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    # Translucent brown pills (V~100) count as box background; only bright
    # surroundings (skull glints, tower art) are excluded. A second bright-
    # fraction gate below rejects mostly-bright blobs.
    dark = cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([179, 255, 130], dtype=np.uint8),
    )
    margin = 10
    for gx, gy, gw, gh in merged_groups:
        ex0, ey0 = gx - margin, gy - margin
        ex1, ey1 = gx + gw + margin, gy + gh + margin
        pill_w, pill_h = ex1 - ex0, ey1 - ey0
        if pill_w < int(nw * 0.07) or pill_w > int(nw * 0.30):
            continue
        if pill_h < int(nh * 0.015) or pill_h > int(nh * 0.06):
            continue
        aspect = float(pill_w) / max(1.0, float(pill_h))
        if not (1.4 <= aspect <= 4.5):
            continue
        box = _clamp_rect(sx + ex0, sy + ey0, pill_w, pill_h, nw, nh)
        qx, qy, qw, qh = box[0] - sx, box[1] - sy, box[2], box[3]
        cell = dark[qy : qy + qh, qx : qx + qw]
        if cell.size == 0:
            continue
        dark_ratio = float(np.count_nonzero(cell)) / float(max(1, qw * qh))
        if dark_ratio < 0.35:
            continue
        cell_bright = bright[qy : qy + qh, qx : qx + qw]
        bright_ratio = float(np.count_nonzero(cell_bright)) / float(max(1, qw * qh))
        if bright_ratio > 0.45:
            continue
        _store_timer_entries(nw, nh, entries, box)
        return
    raise RuntimeError("no timer pill")


def adapt_rois_for_probe(native_bgr):
    """Adapt ROIs for one native probe frame.

    Returns ``(roi_entries, overlay_jpeg_bytes, warnings, (nw, nh))``.
    """
    if native_bgr is None or not hasattr(native_bgr, "shape"):
        raise ValueError("unreadable probe frame")
    try:
        nh, nw = native_bgr.shape[:2]
    except Exception:
        raise ValueError("unreadable probe frame")
    if nw <= 0 or nh <= 0 or native_bgr.size == 0:
        raise ValueError("unreadable probe frame")
    if len(native_bgr.shape) < 2:
        raise ValueError("unreadable probe frame")
    nw, nh = int(nw), int(nh)

    base = scaled_rois(nw, nh)
    is_native = (nw, nh) == (BEDROCK_W, BEDROCK_H)
    fallback_source = "native" if is_native else "scaled"
    fallback_conf = 1.0 if is_native else _FALLBACK_CONFIDENCE
    entries: dict[str, dict] = {
        name: {
            "name": name,
            "rect": list(rect),
            "source": fallback_source,
            "confidence": float(fallback_conf),
        }
        for name, rect in base.items()
    }
    warnings: list[str] = []

    try:
        _refine_elixir(native_bgr, nw, nh, entries)
    except Exception as error:
        warnings.append(f"elixir landmark failed, using scaled fallback: {error}")
    try:
        _refine_hand(native_bgr, nw, nh, entries)
    except Exception as error:
        warnings.append(f"hand landmark failed, using scaled fallback: {error}")
    try:
        _refine_timer(native_bgr, nw, nh, entries)
    except Exception as error:
        warnings.append(f"timer landmark failed, using scaled fallback: {error}")

    roi_entries = [entries[name] for name in ROIS.keys()]

    overlay = native_bgr.copy()
    for item in roi_entries:
        x, y, w, h = (int(v) for v in item["rect"])
        color = (0, 255, 0) if item["source"] == "landmark" else (255, 0, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    max_width = 720
    preview = overlay
    if nw > max_width:
        scale = max_width / float(nw)
        new_h = max(1, int(round(nh * scale)))
        preview = cv2.resize(overlay, (max_width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg_bytes = bytes(buf.tobytes()) if ok and buf is not None else None

    return roi_entries, jpeg_bytes, warnings, (nw, nh)
