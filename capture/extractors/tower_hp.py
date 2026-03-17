import cv2
from image_utils import crop, read_number_from_roi, preprocess_digit

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
        "enemy_king_hp_bar": {
            "image": crop(frame, ROIS["enemy_king_hp_bar"]),
            "value": None,
        },
        "own_king_hp_bar": {
            "image": crop(frame, ROIS["own_king_hp_bar"]),
            "value": None,
        },
        "enemy_support_hp_bar_left": {
            "image": crop(frame, ROIS["enemy_support_hp_bar_left"]),
            "value": None,
        },
        "enemy_support_hp_bar_right": {
            "image": crop(frame, ROIS["enemy_support_hp_bar_right"]),
            "value": None,
        },
        "own_support_hp_bar_left": {
            "image": crop(frame, ROIS["own_support_hp_bar_left"]),
            "value": None,
        },
        "own_support_hp_bar_right": {
            "image": crop(frame, ROIS["own_support_hp_bar_right"]),
            "value": None,
        },
    }

    for tower_name, tower_data in towers_hp.items():
        tower_data["value"] = read_number_from_roi(tower_data["image"], TEMPLATES)


    return {
            tower_name: tower_data["value"]
            for tower_name, tower_data in towers_hp.items()
            } 
