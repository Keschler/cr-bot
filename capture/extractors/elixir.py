import cv2

from image_utils import crop, estimate_slot_fraction, preprocess_digit
from rois import ELIXIR_SLOT_ROIS, ROIS


def load_templates():
    return {
        0: cv2.imread("templates/elixir_0.png", cv2.IMREAD_GRAYSCALE),
        1: cv2.imread("templates/elixir_1.png", cv2.IMREAD_GRAYSCALE),
        2: cv2.imread("templates/elixir_2.png", cv2.IMREAD_GRAYSCALE),
        3: cv2.imread("templates/elixir_3.png", cv2.IMREAD_GRAYSCALE),
        4: cv2.imread("templates/elixir_4.png", cv2.IMREAD_GRAYSCALE),
        5: cv2.imread("templates/elixir_5.png", cv2.IMREAD_GRAYSCALE),
        6: cv2.imread("templates/elixir_6.png", cv2.IMREAD_GRAYSCALE),
        7: cv2.imread("templates/elixir_7.png", cv2.IMREAD_GRAYSCALE),
        8: cv2.imread("templates/elixir_8.png", cv2.IMREAD_GRAYSCALE),
        9: cv2.imread("templates/elixir_9.png", cv2.IMREAD_GRAYSCALE),
        10: cv2.imread("templates/elixir_10.png", cv2.IMREAD_GRAYSCALE),
    }

def build_digit_templates():
    raw_templates = load_templates()
    return {
            digit: preprocess_digit(crop(template, ROIS["elixir_digit"]))
            for digit, template in raw_templates.items()
            }

DIGIT_TEMPLATES = build_digit_templates()


def detect_elixir_digit(digit_img, templates=DIGIT_TEMPLATES):
    black_white_elixir = preprocess_digit(digit_img)

    best_digit = None
    best_score = -1.0

    for digit, template_digit in templates.items():
        result = cv2.matchTemplate(black_white_elixir, template_digit, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0]

        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit, best_score


def read_elixir_value(displayed_digit, frame, slot_rois=ELIXIR_SLOT_ROIS):
    number = displayed_digit[0]

    if number >= 10:
        return 0.0

    next_slot = crop(frame, slot_rois[number])
    return estimate_slot_fraction(next_slot)


def extract_elixir(frame, templates=DIGIT_TEMPLATES):
    elixir_digit = crop(frame, ROIS["elixir_digit"])
    displayed_digit = detect_elixir_digit(elixir_digit, templates)
    elixir_estimate = read_elixir_value(displayed_digit, frame)
    return {
        "digit_image": elixir_digit,
        "displayed_digit": displayed_digit,
        "estimated_value": elixir_estimate,
    }
