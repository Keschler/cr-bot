import os
from pathlib import Path

import cv2
import numpy as np

from rois import ROIS

HAND_CARD_ART_ROI = (18, 38, 184, 212)
CARD_TEMPLATE_SIZE = (150, 180)
CARD_FEATURE_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
CARD_ORB = cv2.ORB_create(nfeatures=500)


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

    _, thresholded = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
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

def classify_digit(digit_img, templates, out_size=(32,48)):
    digit_img = cv2.resize(digit_img, out_size)

    best_digit = None
    best_score = -1.0

    for digit, tmpl in templates.items():
        tmpl_resized = cv2.resize(tmpl, out_size)
        result = cv2.matchTemplate(digit_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        score = result[0,0]

        if score > best_score:
            best_score = score
            best_digit = digit
    if best_score <= 0:
        return 0, best_score

    return best_digit, best_score 

def classify_card(slot_img, templates):
    return classify_card_for_slot(slot_img, templates, None)


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


def classify_card_for_slot(slot_img, templates, slot_name):
    has_purple_header, has_filled_evo_pips = _is_evolved_slot(slot_img)
    evolved = has_purple_header and has_filled_evo_pips

    evo_templates = {
        name: tmpl for name, tmpl in templates.items()
        if name.endswith("-ev1")
    }
    base_templates = {
        name: tmpl for name, tmpl in templates.items()
        if not name.endswith("-ev1")
    }

    def normalize_result(best_name, best_score, evolved_flag=False):
        if best_name is None:
            return None, best_score
        return _normalize_card_name(best_name, evolved=evolved_flag), best_score

    if evolved and slot_name == "next_card":  # Active Evo Card
        best_name, best_score = _feature_match_card_template(slot_img, evo_templates)
        return normalize_result(best_name, best_score, evolved_flag=True)

    if evolved:
        best_name, best_score = _feature_match_card_template(
            slot_img,
            base_templates,
            crop_roi=HAND_CARD_ART_ROI,
        )
        return normalize_result(best_name, best_score, evolved_flag=True)

    if has_purple_header and slot_name == "next_card":  # Not active Evo card
        best_name, best_score = _feature_match_card_template(slot_img, evo_templates)
        return normalize_result(best_name, best_score)

    if slot_name != "next_card":  # Four Hand Cards
        best_name, best_score = _feature_match_card_template(
            slot_img,
            base_templates,
            crop_roi=HAND_CARD_ART_ROI,
        )
        return normalize_result(best_name, best_score)

    best_name, best_score = _feature_match_card_template(slot_img, base_templates)  # Normal Next Card
    return normalize_result(best_name, best_score)


def show_digit_segmentation_debug(binary_img):
      contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      contour_view = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
      box_view = contour_view.copy()

      boxes = []
      for c in contours:
          cv2.drawContours(contour_view, [c], -1, (0, 255, 0), 1)

          x, y, w, h = cv2.boundingRect(c)
          if w > 4 and h > 8:
              boxes.append((x, y, w, h))
              cv2.rectangle(box_view, (x, y), (x + w, y + h), (0, 0, 255), 1)

      boxes.sort(key=lambda b: b[0])

      cv2.imshow("binary", binary_img)
      cv2.imshow("contours", contour_view)
      cv2.imshow("boxes", box_view)
      cv2.waitKey(0)

      return boxes


def read_number_from_roi(img, templates, semicolon=False):
    binary = preprocess_digit(img)
    boxes = segment_digits(binary)
    #boxes = show_digit_segmentation_debug(binary) ## Debug
    if not boxes:
        return None

    max_width = max(w for _, _, w, _ in boxes)
    max_height = max(h for _, _, _, h in boxes)

    chars = []
    for x, y, w, h in boxes:
        if semicolon and w < max_width * 0.5 and h < max_height * 0.5:
            chars.append(":")
            continue

        dimg = binary[y:y + h, x:x + w]
        digit, score = classify_digit(dimg, templates)
        print(f"{digit} got {score}")
        chars.append(str(digit)) # Classify each digit separately

    value = "".join(chars)
    if semicolon:
        return value

    return int(value)

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
        np.array([80, 80, 80], dtype=np.uint8),
        np.array([110, 255, 255], dtype=np.uint8),
    )
    return float(cyan_mask.mean() / 255.0)


def _measure_red_ratio(bar_img):
    fill = _extract_king_bar_fill(bar_img)
    hsv = cv2.cvtColor(fill, cv2.COLOR_BGR2HSV)

    lower_red_mask = cv2.inRange(
        hsv,
        np.array([0, 80, 80], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )
    upper_red_mask = cv2.inRange(
        hsv,
        np.array([170, 80, 80], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask)
    return float(red_mask.mean() / 255.0)


def detect_if_king_tower_activated(img):
    own_king_hp_bar = crop(img, ROIS["player_king_health_bar"])
    enemy_king_hp_bar = crop(img, ROIS["opponent_king_health_bar"])

    own_king = _measure_cyan_ratio(own_king_hp_bar) >= 0.3
    enemy_king = _measure_red_ratio(enemy_king_hp_bar) >= 0.3

    return {
        "own_king_activated": own_king,
        "enemy_king_activated": enemy_king,
    }

def detect_if_support_tower_alive(img):
    own_support_left_bar = crop(img, ROIS["player_left_support_health_bar"])
    own_support_right_bar = crop(img, ROIS["player_right_support_health_bar"])

    enemy_support_left_bar = crop(img, ROIS["opponent_left_support_health_bar"])
    enemy_support_right_bar = crop(img, ROIS["opponent_right_support_health_bar"])

    support_left = _measure_cyan_ratio(own_support_left_bar) >= 0.3
    support_right = _measure_cyan_ratio(own_support_right_bar) >= 0.3

    enemy_support_right = _measure_red_ratio(enemy_support_left_bar) >= 0.3
    enemy_support_left = _measure_red_ratio(enemy_support_right_bar) >= 0.3

    return {
            "support_left_activated": support_left,
            "support_right_activated": support_right,
            "enemy_support_left_activated": enemy_support_left,
            "enemy_support_right_activated": enemy_support_right
            }


def extract_health_bar(img, bar):
    health_bar = img[
      int(bar["y1"]):int(bar["y2"]),
      int(bar["x1"]):int(bar["x2"])
  ]
    if health_bar.size == 0:
        print("health bar check: invalid crop")
        return

    team = bar.get("team")
    height, width = health_bar.shape[:2]
    aspect_ratio = width / max(height, 1)

    fill = _extract_king_bar_fill(health_bar)
    hsv = cv2.cvtColor(fill, cv2.COLOR_BGR2HSV)

    if team == "enemy":
        lower_mask = cv2.inRange(
            hsv,
            np.array([0, 80, 80], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
        )
        upper_mask = cv2.inRange(
            hsv,
            np.array([170, 80, 80], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        color_mask = cv2.bitwise_or(lower_mask, upper_mask)
        color_ratio = _measure_red_ratio(health_bar)
    else:
        color_mask = cv2.inRange(
            hsv,
            np.array([80, 80, 80], dtype=np.uint8),
            np.array([110, 255, 255], dtype=np.uint8),
        )
        color_ratio = _measure_cyan_ratio(health_bar)
        if color_ratio < 0.12:
            broader_blue_mask = cv2.inRange(
                hsv,
                np.array([90, 35, 35], dtype=np.uint8),
                np.array([135, 255, 255], dtype=np.uint8),
            )
            broader_blue_ratio = float(broader_blue_mask.mean() / 255.0)
            if broader_blue_ratio > color_ratio:
                color_mask = broader_blue_mask
                color_ratio = broader_blue_ratio

    kernel = np.ones((3, 3), dtype=np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0.0
    horizontal_continuity = 0.0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(largest))
        x, y, w, h = cv2.boundingRect(largest)
        active_columns = np.count_nonzero(color_mask.max(axis=0) > 0)
        horizontal_continuity = active_columns / max(fill.shape[1], 1)

    gray = cv2.cvtColor(health_bar, cv2.COLOR_BGR2GRAY)
    edge_mask = cv2.Canny(gray, 60, 160)
    gray_std = float(gray.std())
    edge_density = float((edge_mask > 0).mean())
    horizontal_edge_peak = float((edge_mask > 0).mean(axis=1).max())
    low_texture = gray_std < 12.0 and edge_density < 0.02
    is_top_ui_bar = int(bar["y1"]) < 150
    strong_color_bar = (
        color_ratio >= 0.22
        and horizontal_continuity >= 0.30
        and largest_area >= 60.0
    )
    grayscale_bar = (
        aspect_ratio >= 3.2
        and gray_std >= 45.0
        and edge_density >= 0.08
        and horizontal_edge_peak >= 0.6
    )

    is_probable_health_bar = (
        not is_top_ui_bar
        and not low_texture
        and aspect_ratio >= 1.5
        and (strong_color_bar or grayscale_bar)
    )
    
    return is_probable_health_bar

'''
    print(
        "health bar check:",
        {
            "team": team,
            "is_probable_health_bar": is_probable_health_bar,
            "aspect_ratio": round(aspect_ratio, 3),
            "color_ratio": round(color_ratio, 3),
            "horizontal_continuity": round(horizontal_continuity, 3),
            "largest_area": round(largest_area, 3),
            "gray_std": round(gray_std, 3),
            "edge_density": round(edge_density, 3),
            "horizontal_edge_peak": round(horizontal_edge_peak, 3),
            "is_top_ui_bar": is_top_ui_bar,
        },
    )
'''

