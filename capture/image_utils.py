from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from constants import (
    ALLY_BROADER_BLUE_HSV_HIGH,
    ALLY_BROADER_BLUE_HSV_LOW,
    ALLY_CYAN_HSV_HIGH,
    ALLY_CYAN_HSV_LOW,
    ENEMY_RED_HSV_HIGH_1,
    ENEMY_RED_HSV_HIGH_2,
    ENEMY_RED_HSV_LOW_1,
    ENEMY_RED_HSV_LOW_2,
    HEALTH_BAR_BROADER_BLUE_RATIO_THRESHOLD,
    TOWER_BAR_VISIBLE_RATIO_THRESHOLD,
)
from paths import MODELS_DIR
from rois import ROIS

HAND_CARD_ART_ROI = (18, 38, 184, 212)
CARD_TEMPLATE_SIZE = (150, 180)
CARD_FEATURE_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
CARD_ORB = cv2.ORB_create(nfeatures=500)
HAND_CLASSIFIER_PATH = MODELS_DIR / "hand_classifier_best.pt"
NEXT_CLASSIFIER_PATH = MODELS_DIR / "next_classifier_best.pt"
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225],
  ),
])
_CLASSIFIER_CACHE = {}


def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def draw_rois(frame, rois):
    color = (0, 255, 0)
    count = 0
    for name, (x, y, w, h) in rois.items():
        color = (255, 0, 0) if count % 2 == 0 else (0, 255, 0)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def preprocess_digit(img):
    if img is None:
        raise ValueError("preprocess_digit received None")

    if len(img.shape) == 2:
        img_gray = img
    elif len(img.shape) == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(img_gray, 195, 255, cv2.THRESH_BINARY)
    return thresholded


def estimate_slot_fraction(slot_img):
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(col_mean, kernel, mode="same")

    edge_width = max(3, len(smooth) // 5)
    left_level = np.mean(smooth[:edge_width])
    right_level = np.mean(smooth[-edge_width:])

    if left_level - right_level < 5:
        return 0.0

    threshold = (left_level + right_level) / 2
    filled_cols = np.where(smooth >= threshold)[0]

    if len(filled_cols) == 0:
        return 0.0

    rightmost = filled_cols[-1]
    fraction = (rightmost + 1) / len(smooth)
    return float(np.clip(fraction, 0.0, 1.0))


def purple_amount(slot_img):
      hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)

      lower = np.array([125, 40, 80])
      upper = np.array([170, 255, 255])
      mask = cv2.inRange(hsv, lower, upper)

      col_ratio = mask.mean(axis=0) / 255.0
      purple_cols = np.where(col_ratio > 0.15)[0]

      if len(purple_cols) == 0:
          return 0.0

      rightmost = purple_cols[-1]
      return (rightmost + 1) / mask.shape[1]

def segment_digits(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
    
        if w > 4 and h > 8:
            boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    return boxes

def extract_digit_images(binary_img, boxes):
    digits = []
    for x,y,w,h in boxes:
        digit = binary_img[y:y+h, x:x+w]
        digits.append(digit)
    return digits

def _binary_region_mean(binary_img, y0, y1, x0, x1):
    h, w = binary_img.shape[:2]
    ys0 = max(0, min(h, int(round(h * y0))))
    ys1 = max(ys0 + 1, min(h, int(round(h * y1))))
    xs0 = max(0, min(w, int(round(w * x0))))
    xs1 = max(xs0 + 1, min(w, int(round(w * x1))))
    return float((binary_img[ys0:ys1, xs0:xs1] > 0).mean())


def _tower_digit_override(best_digit, scores, norm_digit, source_digit=None):
    top_mid = _binary_region_mean(norm_digit, 0.00, 0.25, 0.35, 0.65)
    center_mid = _binary_region_mean(norm_digit, 0.35, 0.65, 0.40, 0.60)
    left_mid = _binary_region_mean(norm_digit, 0.35, 0.65, 0.15, 0.35)
    right_mid = _binary_region_mean(norm_digit, 0.35, 0.65, 0.65, 0.85)
    bottom_mid = _binary_region_mean(norm_digit, 0.75, 1.00, 0.35, 0.65)
    center_column = _binary_region_mean(norm_digit, 0.10, 0.90, 0.42, 0.58)
    right_column = _binary_region_mean(norm_digit, 0.10, 0.90, 0.72, 0.92)
    source_aspect = None
    if source_digit is not None:
        source_h, source_w = source_digit.shape[:2]
        source_aspect = source_w / max(1, source_h)

    # Tower-font 4 often gets matched as 0 when the open bottom and tall side rails
    # are preserved but the template correlation still prefers the rounded glyph.
    if (
        best_digit == 0
        and scores.get(4, -1.0) > scores.get(0, -1.0) - 0.30
        and top_mid > 0.85
        and left_mid > 0.90
        and right_mid > 0.90
        and center_mid < 0.30
        and bottom_mid < 0.50
    ):
        return 4

    if (
        best_digit == 0
        and scores.get(4, -1.0) > scores.get(0, -1.0) - 0.15
        and top_mid > 0.90
        and left_mid > 0.95
        and right_mid > 0.95
        and center_mid > 0.40
        and 0.40 < bottom_mid < 0.65
    ):
        return 4

    if (
        best_digit == 0
        and scores.get(4, -1.0) > 0.20
        and scores.get(4, -1.0) > scores.get(0, -1.0) - 0.30
        and top_mid > 0.75
        and center_mid < 0.05
        and left_mid > 0.75
        and right_mid > 0.85
        and bottom_mid < 0.35
    ):
        return 4

    if (
        best_digit == 0
        and scores.get(4, -1.0) > 0.25
        and scores.get(4, -1.0) > scores.get(0, -1.0) - 0.22
        and top_mid > 0.95
        and left_mid > 0.95
        and right_mid > 0.95
        and 0.30 < center_mid < 0.75
        and 0.45 < bottom_mid < 0.60
    ):
        return 4

    if (
        best_digit == 0
        and scores.get(4, -1.0) > scores.get(0, -1.0) - 0.25
        and top_mid > 0.90
        and center_mid < 0.35
        and 0.60 < left_mid < 0.90
        and right_mid > 0.95
        and bottom_mid < 0.40
        and _binary_region_mean(norm_digit, 0.75, 1.00, 0.00, 0.35) < 0.25
    ):
        return 4

    if (
        best_digit == 3
        and scores.get(1, -1.0) > scores.get(3, -1.0) - 0.08
        and top_mid > 0.90
        and center_mid > 0.95
        and 0.40 < left_mid < 0.70
        and right_mid > 0.95
        and bottom_mid > 0.95
    ):
        return 1

    if (
        best_digit == 3
        and source_aspect is not None
        and source_aspect < 0.40
        and center_column > 0.90
        and left_mid > 0.90
        and right_mid > 0.90
        and bottom_mid > 0.90
    ):
        return 1

    # A narrow tower-font 1 can correlate with 3 because of its top and bottom
    # caps. Do not apply this when the right edge is filled like an actual 3.
    if (
        best_digit == 3
        and scores.get(1, -1.0) > scores.get(3, -1.0) - 0.20
        and center_column > 0.55
        and right_column < 0.55
        and right_mid < 0.75
        and center_mid > 0.55
        and bottom_mid > 0.45
    ):
        return 1

    # Tower-font 6 often gets matched as 5 when the left wall and inner bowl are
    # visible but the lower-right curve is a bit weak after thresholding.
    if (
        best_digit == 5
        and scores.get(6, -1.0) > scores.get(5, -1.0) - 0.20
        and left_mid > 0.90
        and right_mid < 0.70
        and center_mid > 0.45
        and bottom_mid < 0.45
    ):
        return 6

    if (
        best_digit == 6
        and scores.get(5, -1.0) > scores.get(6, -1.0) - 0.06
        and left_mid < 0.90
        and right_mid < 0.75
        and center_mid > 0.55
        and bottom_mid > 0.70
    ):
        return 5

    if (
        best_digit == 6
        and scores.get(5, -1.0) > scores.get(6, -1.0) - 0.09
        and top_mid > 0.75
        and center_mid > 0.70
        and left_mid < 0.75
        and right_mid < 0.75
        and bottom_mid > 0.75
        and _binary_region_mean(norm_digit, 0.00, 0.25, 0.00, 0.35) > 0.90
    ):
        return 5

    if (
        best_digit == 6
        and scores.get(0, -1.0) > scores.get(6, -1.0) - 0.05
        and center_mid < 0.10
        and center_column < 0.30
        and left_mid > 0.75
        and right_mid > 0.60
        and bottom_mid > 0.75
    ):
        return 0

    return best_digit


def classify_digit(digit_img, templates, out_size=(32,48), mode="default"):
    source_digit = digit_img
    digit_img = cv2.resize(digit_img, out_size)

    best_digit = None
    best_score = -1.0
    scores = {}

    for digit, tmpl in templates.items():
        tmpl_resized = cv2.resize(tmpl, out_size)
        result = cv2.matchTemplate(digit_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        score = result[0,0]
        scores[digit] = float(score)

        if score > best_score:
            best_score = score
            best_digit = digit
    if best_score <= 0:
        return 0, best_score

    if mode == "tower" or mode == "timer":
        best_digit = _tower_digit_override(best_digit, scores, digit_img, source_digit=source_digit)

    return best_digit, best_score 


def _normalize_card_name(name, evolved=False):
    if name is None:
        return None
    normalized = name

    if normalized.endswith("-ev1"):
        normalized = normalized[:-4]
    if normalized.endswith("-hero"):
        normalized = normalized[:-5]
    if normalized == "the-log":
        normalized = "log"

    if evolved:
        normalized = f"evo-{normalized}"

    return normalized


def _is_evolved_slot(slot_img):
    header = slot_img[: min(40, slot_img.shape[0]), :]
    hsv = cv2.cvtColor(header, cv2.COLOR_BGR2HSV)
    purple_mask = (
        (hsv[:, :, 0] >= 120)
        & (hsv[:, :, 0] <= 170)
        & (hsv[:, :, 1] >= 60)
    )
    has_purple_header = float(purple_mask.mean()) > 0.2

    if slot_img.shape[1] > 150:
        pip_centers = ((80, 16), (132, 16))
    else:
        pip_centers = ((47, 6), (74, 6))

    filled_pips = 0
    for center_x, center_y in pip_centers:
        patch = hsv[
            max(0, center_y - 4):center_y + 5,
            max(0, center_x - 4):center_x + 5,
        ]
        if patch.size == 0:
            continue

        mean_saturation = float(patch[:, :, 1].mean())
        mean_value = float(patch[:, :, 2].mean())
        if mean_saturation >= 160 and mean_value >= 180:
            filled_pips += 1

    return has_purple_header, filled_pips >= 2



def _feature_match_card_template(slot_img, templates, crop_roi=None):
    best_name = None
    best_score = -1

    x, y, w, h = crop_roi or (0, 0, slot_img.shape[1], slot_img.shape[0])
    match_img = slot_img[y:y + h, x:x + w]
    resized_img = cv2.resize(match_img, CARD_TEMPLATE_SIZE)
    slot_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    _, slot_desc = CARD_ORB.detectAndCompute(slot_gray, None)
    if slot_desc is None:
        return None, best_score

    for name, tmpl_data in templates.items():
        tmpl_desc = tmpl_data["descriptor"]
        if tmpl_desc is None:
            continue

        matches = CARD_FEATURE_MATCHER.match(slot_desc, tmpl_desc)
        good_matches = sum(1 for match in matches if match.distance < 50)

        if good_matches > best_score:
            best_score = good_matches
            best_name = name

    return best_name, best_score


def _load_card_classifier(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    cache_key = str(checkpoint_path.resolve())
    cached = _CLASSIFIER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    payload = {
        "model": model,
        "classes": classes,
    }
    _CLASSIFIER_CACHE[cache_key] = payload
    return payload


def _classify_card_with_model(slot_img, checkpoint_path, crop_roi=None):
    if not Path(checkpoint_path).exists():
        return None, None

    x, y, w, h = crop_roi or (0, 0, slot_img.shape[1], slot_img.shape[0])
    match_img = slot_img[y:y + h, x:x + w]
    rgb_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    image_tensor = CLASSIFIER_TRANSFORM(pil_img).unsqueeze(0)

    payload = _load_card_classifier(checkpoint_path)
    model = payload["model"]
    classes = payload["classes"]

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return classes[pred_idx.item()], float(confidence.item() * 100.0)


def classify_card_for_slot(slot_img, base_templates, evo_templates, slot_name):
    has_purple_header, has_filled_evo_pips = _is_evolved_slot(slot_img)
    evolved = has_purple_header and has_filled_evo_pips

    def normalize_result(best_name, best_score, evolved_flag=False):
        if best_name is None:
            return None, best_score
        return _normalize_card_name(best_name, evolved=evolved_flag), best_score

    if slot_name == "next_card":
        best_name, best_score = _classify_card_with_model(slot_img, NEXT_CLASSIFIER_PATH)
        if best_name is None:
            if evolved:
                best_name, best_score = _feature_match_card_template(slot_img, evo_templates)
                return normalize_result(best_name, best_score, evolved_flag=True)
            best_name, best_score = _feature_match_card_template(slot_img, base_templates)
            return normalize_result(best_name, best_score)
        return normalize_result(best_name, best_score, evolved_flag=evolved)

    best_name, best_score = _classify_card_with_model(
        slot_img,
        HAND_CLASSIFIER_PATH,
        crop_roi=HAND_CARD_ART_ROI,
    )
    if best_name is None:
        best_name, best_score = _classify_card_with_model(
            slot_img,
            HAND_CLASSIFIER_PATH,
            crop_roi=HAND_CARD_ART_ROI,
        )
    return normalize_result(best_name, best_score, evolved_flag=evolved)


def build_digit_debug_views(binary_img, boxes, digit_views):
    box_view = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(box_view, (x, y), (x + w, y + h), (0, 0, 255), 1)

    tile_h = 56
    tile_w = 40
    if not digit_views:
        digits_view = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(digits_view, "none", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return box_view, digits_view

    tiles = []
    for label, digit_img in digit_views:
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        if len(digit_img.shape) == 2:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(digit_img, (24, 36), interpolation=cv2.INTER_NEAREST)
        tile[4:40, 8:32] = resized
        cv2.putText(tile, label, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    return box_view, np.hstack(tiles)


def read_number_from_roi(img, templates, semicolon=False, debug_steps=None, digit_mode="default"):
    if debug_steps is not None:
        debug_steps["raw"] = img.copy()
    binary = preprocess_digit(img)
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
        if semicolon and w < max_width * 0.5 and h < max_height * 0.5:
            chars.append(":")
            digit_views.append((":", binary[y:y + h, x:x + w]))
            continue

        dimg = binary[y:y + h, x:x + w]
        digit, score = classify_digit(dimg, templates, mode=digit_mode)
        ratio = w/h
        if score <= 0.1 or ratio > 1:
            continue
        chars.append(str(digit)) # Classify each digit separately
        digit_views.append((str(digit), dimg))

    if debug_steps is not None:
        box_view, digits_view = build_digit_debug_views(binary, boxes, digit_views)
        debug_steps["boxes"] = box_view
        debug_steps["digits"] = digits_view

    value = "".join(chars)
    if semicolon:
        return value
    if value:
        return int(value)
    else:
        return None

def _extract_king_bar_fill(bar_img):
    height, width = bar_img.shape[:2]
    x0 = max(0, int(width * 0.12))
    x1 = min(width, int(width * 0.94))
    y0 = max(0, int(height * 0.2))
    y1 = min(height, int(height * 0.75))
    return bar_img[y0:y1, x0:x1]


def _measure_cyan_ratio(bar_img):
    fill = _extract_king_bar_fill(bar_img)
    hsv = cv2.cvtColor(fill, cv2.COLOR_BGR2HSV)

    cyan_mask = cv2.inRange(
        hsv,
        np.array(ALLY_CYAN_HSV_LOW, dtype=np.uint8),
        np.array(ALLY_CYAN_HSV_HIGH, dtype=np.uint8),
    )
    return float(cyan_mask.mean() / 255.0)


def _measure_red_ratio(bar_img, king: bool):
    if king:
        fill = _extract_king_bar_fill(bar_img)
        hsv = cv2.cvtColor(fill, cv2.COLOR_BGR2HSV)
    else:
        hsv = cv2.cvtColor(bar_img, cv2.COLOR_BGR2HSV)

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
    return float(red_mask.mean() / 255.0)


def _measure_white_ratio(bar_img, king: bool):
    if king:
        fill = _extract_king_bar_fill(bar_img)
    else:
        fill = bar_img

    hsv = cv2.cvtColor(fill, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array((0, 0, 185), dtype=np.uint8),
        np.array((179, 55, 255), dtype=np.uint8),
    )
    kernel = np.ones((2, 2), dtype=np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    return float(white_mask.mean() / 255.0)


def detect_if_king_tower_activated(img):
    own_king_hp_bar = crop(img, ROIS["player_king_health_bar"])
    enemy_king_hp_bar = crop(img, ROIS["opponent_king_health_bar"])

    own_king = (
        _measure_cyan_ratio(own_king_hp_bar) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
        or _measure_white_ratio(own_king_hp_bar, True) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
    )
    enemy_king = (
        _measure_red_ratio(enemy_king_hp_bar, True) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
        or _measure_white_ratio(enemy_king_hp_bar, True) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
    )

    return {
        "own_king_activated": own_king,
        "enemy_king_activated": enemy_king,
    }

def detect_if_support_tower_alive(img):
    own_support_left_bar = crop(img, ROIS["player_left_support_health_bar"])
    own_support_right_bar = crop(img, ROIS["player_right_support_health_bar"])

    enemy_support_left_bar = crop(img, ROIS["opponent_left_support_health_bar"])
    enemy_support_right_bar = crop(img, ROIS["opponent_right_support_health_bar"])

    support_left = (
        _measure_cyan_ratio(own_support_left_bar) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
        or _measure_white_ratio(own_support_left_bar, False) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
    )
    support_right = (
        _measure_cyan_ratio(own_support_right_bar) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
        or _measure_white_ratio(own_support_right_bar, False) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
    )

    enemy_support_right = (
        _measure_red_ratio(enemy_support_right_bar, True) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
        or _measure_white_ratio(enemy_support_right_bar, False) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
    )
    enemy_support_left = (
        _measure_red_ratio(enemy_support_left_bar, True) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
        or _measure_white_ratio(enemy_support_left_bar, False) >= TOWER_BAR_VISIBLE_RATIO_THRESHOLD
    )

    return {
            "support_left_activated": support_left,
            "support_right_activated": support_right,
            "enemy_support_left_activated": enemy_support_left,
            "enemy_support_right_activated": enemy_support_right
            }
