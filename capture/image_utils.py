import cv2
import numpy as np

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

    _, thresholded = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY)
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

    if evolved and slot_name == "next_card": # Active Evo Card
        best_name, best_score = _feature_match_card_template(slot_img, evo_templates)
        return _normalize_card_name(best_name, evolved=True), best_score

    if evolved:
        best_name, best_score = _feature_match_card_template(
            slot_img,
            base_templates,
            crop_roi=HAND_CARD_ART_ROI,
        )
        return _normalize_card_name(best_name, evolved=True), best_score

    if has_purple_header and slot_name == "next_card": # Not active Evo card
        best_name, best_score = _feature_match_card_template(slot_img, evo_templates)
        return _normalize_card_name(best_name), best_score

    if slot_name != "next_card": # Four Hand Cards
        best_name, best_score = _feature_match_card_template(
            slot_img,
            base_templates,
            crop_roi=HAND_CARD_ART_ROI,
        )
        return _normalize_card_name(best_name), best_score

    best_name, best_score = _feature_match_card_template(slot_img, base_templates) # Normal Next Card
    return _normalize_card_name(best_name), best_score



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
