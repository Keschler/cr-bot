import cv2
import numpy as np
import re

from constants import (
    ENEMY_RED_HSV_HIGH_1,
    ENEMY_RED_HSV_HIGH_2,
    ENEMY_RED_HSV_LOW_1,
    ENEMY_RED_HSV_LOW_2,
    OVERTIME_SECONDS,
    OVERTIME_RED_RATIO_THRESHOLD,
)
from extractors.tower_hp import EXPERT_TEMPLATES
from rois import ROIS
from image_utils import (
    build_digit_debug_views,
    classify_digit,
    crop,
    preprocess_digit,
    read_number_from_roi,
    segment_digits,
    _measure_red_ratio,
)
from paths import TEMPLATES_DIR

TEMPLATE_DIR = TEMPLATES_DIR / "numbers"
EXPERT_TEMPLATE_DIR = TEMPLATES_DIR / "expert_numbers"



def read_template(name: str):
    path = TEMPLATE_DIR / name
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Failed to read timer template: {path}")
    return template


def load_templates():
    raw_templates = {
        0: read_template("0.png"),
        1: read_template("1.png"),
        2: read_template("2.png"),
        3: read_template("3.png"),
        4: read_template("4.png"),
        5: read_template("5.png"),
        6: read_template("6.png"),
        7: read_template("7.png"),
        8: read_template("8.png"),
        9: read_template("9.png"),
        ":": read_template("semi_colon.png"),
    }

    return {
        digit: preprocess_digit(template)
        for digit, template in raw_templates.items()
    }


TEMPLATES = load_templates()

# Search broadly in the top-right HUD first, then localize the timer within it.
TOP_RIGHT_TIMER_SEARCH_ROI = (0.62, 0.00, 0.38, 0.24)


def _normalized_crop(frame, roi):
    frame_h, frame_w = frame.shape[:2]
    x = int(frame_w * roi[0])
    y = int(frame_h * roi[1])
    w = int(frame_w * roi[2])
    h = int(frame_h * roi[3])
    return frame[y:y + h, x:x + w], (x, y, w, h)


def _normalize_timer_text(text: str | None) -> str | None:
    if text is None:
        return None

    cleaned = re.sub(r"[^0-9:]", "", str(text))
    if not cleaned:
        return None

    cleaned = re.sub(r":+", ":", cleaned).strip(":")
    if re.fullmatch(r"\d{1,2}:\d{2}", cleaned):
        minutes, seconds = cleaned.split(":", 1)
        if 0 <= int(minutes) <= 3 and 0 <= int(seconds) <= 59:
            return cleaned
        return None

    if ":" in cleaned:
        digits = re.sub(r"[^0-9]", "", cleaned)
        if len(digits) >= 3:
            # OCR sometimes duplicates colons or drops spacing; recover by treating
            # the final two digits as seconds and the preceding digits as minutes.
            digits = digits[-4:]
            seconds = digits[-2:]
            minutes = digits[:-2]
            normalized = f"{int(minutes)}:{seconds}"
            if re.fullmatch(r"\d{1,2}:\d{2}", normalized):
                norm_minutes, norm_seconds = normalized.split(":", 1)
                if not (0 <= int(norm_minutes) <= 3 and 0 <= int(norm_seconds) <= 59):
                    return None
                return normalized

    return None


def _timer_binary(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bright_mask = cv2.inRange(gray, 170, 255)
    lower_red_mask = cv2.inRange(
        hsv,
        np.array(ENEMY_RED_HSV_LOW_1, dtype=np.uint8),
        np.array(ENEMY_RED_HSV_HIGH_1, dtype=np.uint8),
    )
    upper_red_mask = cv2.inRange(
        hsv,
        np.array(ENEMY_RED_HSV_LOW_2, dtype=np.uint8),
        np.array(ENEMY_RED_HSV_HIGH_2, dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask)

    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    strong_red = ((r > g + 20) & (r > b + 20) & (r > 80)).astype(np.uint8) * 255
    red_mask = cv2.bitwise_and(red_mask, strong_red)

    binary = cv2.bitwise_or(bright_mask, red_mask)
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _measure_timer_digit_red_presence(img: np.ndarray) -> float:
    if img.size == 0:
        return 0.0

    h = img.shape[0]
    digit_img = img[int(h * 0.30):, :]
    if digit_img.size == 0:
        digit_img = img

    hsv = cv2.cvtColor(digit_img, cv2.COLOR_BGR2HSV)
    lower_red_mask = cv2.inRange(
        hsv,
        np.array(ENEMY_RED_HSV_LOW_1, dtype=np.uint8),
        np.array(ENEMY_RED_HSV_HIGH_1, dtype=np.uint8),
    )
    upper_red_mask = cv2.inRange(
        hsv,
        np.array(ENEMY_RED_HSV_LOW_2, dtype=np.uint8),
        np.array(ENEMY_RED_HSV_HIGH_2, dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask) > 0

    b = digit_img[:, :, 0].astype(np.int16)
    g = digit_img[:, :, 1].astype(np.int16)
    r = digit_img[:, :, 2].astype(np.int16)
    strong_red = (r > g + 20) & (r > b + 20) & (r > 80)
    return float((red_mask & strong_red).mean())


def _read_timer_from_roi_red(img: np.ndarray, debug_steps=None) -> str | None:
    h = img.shape[0]
    digit_img = img[int(h * 0.30):, :]
    if digit_img.size == 0:
        digit_img = img

    binary = _timer_binary(digit_img)

    raw_boxes = segment_digits(binary)
    if raw_boxes:
        top_boxes = [box for box in raw_boxes if box[1] < binary.shape[0] * 0.45]
        if top_boxes:
            x0 = max(0, min(x for x, _, _, _ in top_boxes) - 4)
            y0 = max(0, min(y for _, y, _, _ in top_boxes) - 4)
            x1 = min(binary.shape[1], max(x + w for x, _, w, _ in top_boxes) + 4)
            y1 = min(binary.shape[0], max(y + h for _, y, _, h in top_boxes) + 4)
            digit_img = digit_img[y0:y1, x0:x1]
            binary = binary[y0:y1, x0:x1]

    if debug_steps is not None:
        debug_steps["raw"] = digit_img.copy()
    if debug_steps is not None:
        debug_steps["binary"] = binary.copy()

    boxes = segment_digits(binary)
    if not boxes:
        if debug_steps is not None:
            box_view, digits_view = build_digit_debug_views(binary, [], [])
            debug_steps["boxes"] = box_view
            debug_steps["digits"] = digits_view
        return None

    max_width = max(w for _, _, w, _ in boxes)
    max_height = max(h for _, _, _, h in boxes)

    chars = []
    digit_views = []
    for x, y, w, h in boxes:
        if w < max_width * 0.5 and h < max_height * 0.5:
            chars.append(":")
            digit_views.append((":", binary[y:y + h, x:x + w]))
            continue

        digit_img = binary[y:y + h, x:x + w]
        digit, _ = classify_digit(digit_img, TEMPLATES, mode="timer")
        chars.append(str(digit))
        digit_views.append((str(digit), digit_img))

    if debug_steps is not None:
        box_view, digits_view = build_digit_debug_views(binary, boxes, digit_views)
        debug_steps["boxes"] = box_view
        debug_steps["digits"] = digits_view

    return "".join(chars)


def _read_timer_from_roi(img: np.ndarray, debug_steps=None, yolo_templates=None) -> str | None:
    if _measure_timer_digit_red_presence(img) >= 0.02:
        return _read_timer_from_roi_red(img, debug_steps=debug_steps)
    if yolo_templates:
        templates = TEMPLATES
    else:
        templates = EXPERT_TEMPLATES

    return read_number_from_roi(
        img,
        templates,
        semicolon=True,
        debug_steps=debug_steps,
        digit_mode="timer",
    )


def _candidate_score(bbox, search_shape, timer_text: str) -> tuple[float, float, float]:
    x, y, w, h = bbox
    search_h, search_w = search_shape[:2]
    right_bias = (x + w) / max(search_w, 1)
    top_bias = 1.0 - (y / max(search_h, 1))
    colon_bonus = 0.25 if ":" in timer_text else 0.0
    area_penalty = -float(w * h)
    return (colon_bonus + right_bias + top_bias, area_penalty, -y)


def _measure_timer_text_red_ratio(timer_frame: np.ndarray) -> float:
    if timer_frame.size == 0:
        return 0.0

    h = timer_frame.shape[0]
    digit_region = timer_frame[int(h * 0.25):, :]
    if digit_region.size == 0:
        digit_region = timer_frame

    text_mask = preprocess_digit(digit_region) > 0
    if not np.any(text_mask):
        return 0.0

    hsv = cv2.cvtColor(digit_region, cv2.COLOR_BGR2HSV)
    lower_red_mask = cv2.inRange(
        hsv,
        np.array(ENEMY_RED_HSV_LOW_1, dtype=np.uint8),
        np.array(ENEMY_RED_HSV_HIGH_1, dtype=np.uint8),
    )
    upper_red_mask = cv2.inRange(
        hsv,
        np.array(ENEMY_RED_HSV_LOW_2, dtype=np.uint8),
        np.array(ENEMY_RED_HSV_HIGH_2, dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask) > 0

    b = digit_region[:, :, 0].astype(np.int16)
    g = digit_region[:, :, 1].astype(np.int16)
    r = digit_region[:, :, 2].astype(np.int16)
    strong_red = (r > g + 20) & (r > b + 20)

    red_text_pixels = red_mask & strong_red & text_mask
    return float(red_text_pixels.sum() / max(int(text_mask.sum()), 1))


def _locate_timer(frame, debug_steps=None):
    search_img, (search_x, search_y, _, _) = _normalized_crop(frame, TOP_RIGHT_TIMER_SEARCH_ROI)
    if debug_steps is not None:
        debug_steps["search_raw"] = search_img.copy()

    candidates = []
    annotated_views = []
    mode_specs = [
        ("standard", preprocess_digit, lambda img: read_number_from_roi(img, TEMPLATES, semicolon=True, digit_mode="timer")),
        ("red", _timer_binary, _read_timer_from_roi_red),
    ]

    for mode_name, binary_fn, reader_fn in mode_specs:
        binary = binary_fn(search_img)

        # Merge adjacent timer glyphs into a single candidate region before contouring.
        kernel = np.ones((9, 15), dtype=np.uint8)
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 1500 or w < 45 or h < 30:
                continue
            if y > search_img.shape[0] * 0.45:
                continue
            if w > search_img.shape[1] * 0.60 or h > search_img.shape[0] * 0.35:
                continue

            pad_x = max(6, int(w * 0.08))
            pad_y = max(6, int(h * 0.12))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(search_img.shape[1], x + w + pad_x)
            y1 = min(search_img.shape[0], y + h + pad_y)
            candidate_img = search_img[y0:y1, x0:x1]
            candidate_text = _normalize_timer_text(reader_fn(candidate_img))
            if candidate_text is None:
                continue

            bbox = (x0, y0, x1 - x0, y1 - y0)
            candidates.append(
                {
                    "timer_text": candidate_text,
                    "bbox": bbox,
                    "score": _candidate_score(bbox, search_img.shape, candidate_text),
                    "mode": mode_name,
                }
            )
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)

        annotated_views.append((mode_name, annotated))

    if debug_steps is not None:
        debug_steps["search_binary"] = annotated_views[0][1][:, :, 0] if annotated_views else np.zeros(search_img.shape[:2], dtype=np.uint8)
        debug_steps["search_boxes"] = annotated_views[0][1] if annotated_views else cv2.cvtColor(np.zeros(search_img.shape[:2], dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    if candidates:
        best = max(candidates, key=lambda c: c["score"])
        x, y, w, h = best["bbox"]
        timer_frame = search_img[y:y + h, x:x + w]
        # Include the banner/background around the digits so overtime color checks
        # still work after localizing away from the old fixed ROI.
        timer_box_x0 = max(0, x - int(w * 0.18))
        timer_box_x1 = min(search_img.shape[1], x + w + int(w * 0.18))
        timer_box_y0 = max(0, y - int(h * 0.8))
        timer_box_y1 = min(search_img.shape[0], y + h + int(h * 0.1))
        timer_box = search_img[timer_box_y0:timer_box_y1, timer_box_x0:timer_box_x1]
        return {
            "time": best["timer_text"],
            "timer_frame": timer_frame,
            "timer_box": timer_box,
            "frame_bbox": (search_x + x, search_y + y, w, h),
        }

    timer_frame = crop(frame, ROIS["match_timer"])
    timer_box = crop(frame, ROIS["timer_box"])
    fallback_text = _normalize_timer_text(_read_timer_from_roi(timer_frame))
    if fallback_text is None:
        fallback_text = re.sub(r":+", ":", str(_read_timer_from_roi(timer_frame)))
    return {
        "time": fallback_text,
        "timer_frame": timer_frame,
        "timer_box": timer_box,
        "frame_bbox": None,
    }


def extract_time(frame, debug_steps=None, yolo_templates=None):
    located = _locate_timer(frame, debug_steps=debug_steps)
    timer_debug = debug_steps if debug_steps is not None else None
    time = _read_timer_from_roi(located["timer_frame"], debug_steps=timer_debug, yolo_templates=None)
    normalized = _normalize_timer_text(located["time"])
    reread_normalized = _normalize_timer_text(time)
    if reread_normalized is not None:
        normalized = reread_normalized
    if normalized is None:
        normalized = re.sub(r':+', ':', str(time)).strip(":")
    return normalized

def is_overtime(frame):
    located = _locate_timer(frame)
    timer_box = located["timer_box"]
    red_ratio = _measure_red_ratio(timer_box, False)
    red_text_ratio = _measure_timer_text_red_ratio(located["timer_frame"])
    if red_ratio >= OVERTIME_RED_RATIO_THRESHOLD or red_text_ratio >= 0.35:
        return True
    else:
        return False

def parse_time_left_s(timer_text) -> float:
      normalized = _normalize_timer_text(timer_text)
      if normalized is None: # When normal extraction fails
          matches = re.findall(r"\d{1,2}:\d{2}", str(timer_text)) # extract the last valid M:SS substring
          if not matches:
              return 0.0
          normalized = _normalize_timer_text(matches[-1])
          if normalized is None:
              return 0.0

      minutes, seconds = normalized.split(":", 1)
      return int(minutes) * 60 + int(seconds)


def total_remaining_seconds(time_left, overtime):
    if overtime:
        return time_left
    return time_left + OVERTIME_SECONDS
