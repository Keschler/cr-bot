import cv2

from cr_bot.vision.image_utils import crop, classify_card_for_slot, classify_hand_card_slots
from cr_bot.paths import TEMPLATES_DIR
from cr_bot.domain.rois import ROIS


CARDS_TEMPLATE_DIR = TEMPLATES_DIR / "cr-api-assets/cards-150"


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

BASE_TEMPLATES = {
      name: tmpl
      for name, tmpl in TEMPLATES.items()
      if not name.endswith("-ev1")
  }

EVO_TEMPLATES = {
  name: tmpl
  for name, tmpl in TEMPLATES.items()
  if name.endswith("-ev1")
}


def extract_hand_state(frame, *, rois=None):
    if rois is None:
        cards = {
            "card_1": {"image": crop(frame, ROIS["hand_card_slot_1"]), "detected_card": None},
            "card_2": {"image": crop(frame, ROIS["hand_card_slot_2"]), "detected_card": None},
            "card_3": {"image": crop(frame, ROIS["hand_card_slot_3"]), "detected_card": None},
            "card_4": {"image": crop(frame, ROIS["hand_card_slot_4"]), "detected_card": None},
            "next_card": {"image": crop(frame, ROIS["next_card_slot"]), "detected_card": None},
        }
    else:
        from cr_bot.vision.roi_adapt import resolve_crop as _resolve_crop

        cards = {
            "card_1": {"image": _resolve_crop(frame, "hand_card_slot_1", rois=rois), "detected_card": None},
            "card_2": {"image": _resolve_crop(frame, "hand_card_slot_2", rois=rois), "detected_card": None},
            "card_3": {"image": _resolve_crop(frame, "hand_card_slot_3", rois=rois), "detected_card": None},
            "card_4": {"image": _resolve_crop(frame, "hand_card_slot_4", rois=rois), "detected_card": None},
            "next_card": {"image": _resolve_crop(frame, "next_card_slot", rois=rois), "detected_card": None},
        }

    hand_slots = ["card_1", "card_2", "card_3", "card_4"]
    hand_images = [cards[slot]["image"] for slot in hand_slots]
    for slot, detected_card in zip(hand_slots, classify_hand_card_slots(hand_images)):
        cards[slot]["detected_card"] = detected_card

    cards["next_card"]["detected_card"] = classify_card_for_slot(
        cards["next_card"]["image"],
        BASE_TEMPLATES,
        EVO_TEMPLATES,
        "next_card",
    )
    
    return {
            card: card_data["detected_card"]
            for card, card_data in cards.items()
            }
