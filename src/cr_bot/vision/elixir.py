import cv2
from pathlib import Path

from cr_bot.vision.image_utils import crop, estimate_slot_fraction, preprocess_digit, pink_amount
from cr_bot.paths import TEMPLATES_DIR
from cr_bot.domain.rois import ELIXIR_SLOT_ROIS, ROIS

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


def read_elixir_value(displayed_digit, frame, slot_rois=ELIXIR_SLOT_ROIS, *, rois=None):
    number = displayed_digit

    if number >= 10:
        return 0.0

    if rois is None:
        next_slot = crop(frame, slot_rois[number])
    else:
        from cr_bot.vision.roi_adapt import resolve_crop as _resolve_crop

        next_slot = _resolve_crop(frame, f"elixir_fill_slot_{number + 1}", rois=rois)
    return estimate_slot_fraction(next_slot)

def estimate_total_slots(frame, *, rois=None):
    if rois is None:
        pink_fractions = [pink_amount(crop(frame, roi)) for roi in ELIXIR_SLOT_ROIS]
    else:
        from cr_bot.vision.roi_adapt import resolve_crop as _resolve_crop

        pink_fractions = [
            pink_amount(_resolve_crop(frame, f"elixir_fill_slot_{i}", rois=rois))
            for i in range(1, 11)
        ]
    for idx, fraction in enumerate(pink_fractions):
        if fraction >= 0.7:
            pink_fractions[idx] = 1
        else:
            pink_fractions[idx] = 0
    fraction_sum = sum(pink_fractions)
    return fraction_sum


def extract_elixir(frame, templates=DIGIT_TEMPLATES, *, rois=None):
    if rois is None:
        full_elixir_slots = estimate_total_slots(frame)
        elixir_estimate = read_elixir_value(full_elixir_slots, frame)
    else:
        full_elixir_slots = estimate_total_slots(frame, rois=rois)
        elixir_estimate = read_elixir_value(full_elixir_slots, frame, rois=rois)

    displayed_digit = full_elixir_slots
    return {
        "displayed_digit": displayed_digit,
        "estimated_value": elixir_estimate,
    }
