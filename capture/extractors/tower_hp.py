import cv2
from image_utils import crop, read_number_from_roi, preprocess_digit, detect_if_king_tower_activated, detect_if_support_tower_alive

from rois import ROIS


def load_templates():
    raw_templates = {
        0: cv2.imread("templates/numbers/0.png", cv2.IMREAD_GRAYSCALE),
        1: cv2.imread("templates/numbers/1.png", cv2.IMREAD_GRAYSCALE),
        2: cv2.imread("templates/numbers/2.png", cv2.IMREAD_GRAYSCALE),
        3: cv2.imread("templates/numbers/3.png", cv2.IMREAD_GRAYSCALE),
        4: cv2.imread("templates/numbers/4.png", cv2.IMREAD_GRAYSCALE),
        5: cv2.imread("templates/numbers/5.png", cv2.IMREAD_GRAYSCALE),
        6: cv2.imread("templates/numbers/6.png", cv2.IMREAD_GRAYSCALE),
        7: cv2.imread("templates/numbers/7.png", cv2.IMREAD_GRAYSCALE),
        8: cv2.imread("templates/numbers/8.png", cv2.IMREAD_GRAYSCALE),
        9: cv2.imread("templates/numbers/9.png", cv2.IMREAD_GRAYSCALE),
    }

    return {
        digit: preprocess_digit(template)
        for digit, template in raw_templates.items()
    }


TEMPLATES = load_templates()

def extract_tower_hp(frame):
    towers_hp = {
        "enemy_king": {
            "image": crop(frame, ROIS["opponent_king_health_text"]),
            "value": None,
        },
        "own_king": {
            "image": crop(frame, ROIS["player_king_health_text"]),
            "value": None,
        },
        "enemy_support_left": {
            "image": crop(frame, ROIS["opponent_left_support_health_text"]),
            "value": None,
        },
        "enemy_support_right": {
            "image": crop(frame, ROIS["opponent_right_support_health_text"]),
            "value": None,
        },
        "own_support_left": {
            "image": crop(frame, ROIS["player_left_support_health_text"]),
            "value": None,
        },
        "own_support_right": {
            "image": crop(frame, ROIS["player_right_support_health_text"]),
            "value": None,
        },
    }

    for tower_name, tower_data in towers_hp.items():
        tower_data["value"] = read_number_from_roi(tower_data["image"], TEMPLATES)


    king_tower_activated = detect_if_king_tower_activated(frame)

    if not king_tower_activated["own_king_activated"]:
        towers_hp["own_king"]["value"] = 7032
    if not king_tower_activated["enemy_king_activated"]:
        towers_hp["enemy_king"]["value"] = 7032

    support_tower_alive = detect_if_support_tower_alive(frame)

    if not support_tower_alive["support_left_activated"]:
        towers_hp["own_support_left"]["value"] = 0
    if not support_tower_alive["support_right_activated"]:
        towers_hp["own_support_right"]["value"] = 0
    if not support_tower_alive["enemy_support_left_activated"]:
        towers_hp["enemy_support_left"]["value"] = 0
    if not support_tower_alive["enemy_support_right_activated"]:
        towers_hp["enemy_support_right"]["value"] = 0

    return {
        tower_name: tower_data["value"]
        for tower_name, tower_data in towers_hp.items()
    }
