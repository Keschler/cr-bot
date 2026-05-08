import cv2
from pathlib import Path

from image_utils import crop, estimate_slot_fraction, preprocess_digit
from paths import TEMPLATES_DIR
from rois import ELIXIR_SLOT_ROIS, ROIS

TEMPLATE_DIR = TEMPLATES_DIR


def read_template(path: Path):
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Failed to read elixir template: {path}")
    return template


def load_templates():
    return {
        digit: read_template(TEMPLATE_DIR / f"elixir_{digit}.png")
        for digit in range(11)
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
