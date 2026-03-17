import cv2

from rois import ROIS
from image_utils import crop, read_number_from_roi, preprocess_digit

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
        ":": cv2.imread("templates/numbers/semi_colon.png", cv2.IMREAD_GRAYSCALE),
    }

    return {
        digit: preprocess_digit(template)
        for digit, template in raw_templates.items()
    }


TEMPLATES = load_templates()
def extract_time(frame):
    timer_frame = crop(frame, ROIS["timer"])
    time = read_number_from_roi(timer_frame, TEMPLATES, semicolon=True)
    return time
