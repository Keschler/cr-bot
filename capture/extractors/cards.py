from pathlib import Path

import cv2

from image_utils import crop, classify_card_for_slot
from rois import ROIS


CARDS_TEMPLATE_DIR = Path("templates/cr-api-assets/cards-150")


def _is_supported_template(template_path):
    stem = template_path.stem
    return not ("-hero-ev" in stem)


def load_templates():
    templates = {}
    orb = cv2.ORB_create(nfeatures=500)

    for template_path in sorted(CARDS_TEMPLATE_DIR.glob("*.png")):
        if not _is_supported_template(template_path):
            continue

        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template is None:
            continue

        template_bgr = template[:, :, :3] if len(template.shape) == 3 and template.shape[2] == 4 else template
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        _, template_descriptor = orb.detectAndCompute(template_gray, None)

        templates[template_path.stem] = {
            "bgr": template_bgr,
            "alpha": template[:, :, 3] if len(template.shape) == 3 and template.shape[2] == 4 else None,
            "descriptor": template_descriptor,
        }

    return templates


TEMPLATES = load_templates()

def extract_hand_state(frame):
    cards = {
        "card_1": {"image": crop(frame, ROIS["hand_card_slot_1"]), "detected_card": None},
        "card_2": {"image": crop(frame, ROIS["hand_card_slot_2"]), "detected_card": None},
        "card_3": {"image": crop(frame, ROIS["hand_card_slot_3"]), "detected_card": None},
        "card_4": {"image": crop(frame, ROIS["hand_card_slot_4"]), "detected_card": None},
        "next_card": {"image": crop(frame, ROIS["next_card_slot"]), "detected_card": None},
    }

    for card, card_data in cards.items():
        card_data["detected_card"] = classify_card_for_slot(
            card_data["image"],
            TEMPLATES,
            card,
        )
    
    return {
            card: card_data["detected_card"]
            for card, card_data in cards.items()
            }
