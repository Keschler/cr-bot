import cv2
import re

from rois import ROIS
from image_utils import crop, read_number_from_roi, preprocess_digit, _measure_red_ratio

NORMAL_SECONDS = 180
OVERTIME_SECONDS = 120
TOTAL_MATCH_SECONDS = NORMAL_SECONDS + OVERTIME_SECONDS

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
    timer_frame = crop(frame, ROIS["match_timer"])
    time = read_number_from_roi(timer_frame, TEMPLATES, semicolon=True)
    fixed_time = re.sub(r':+', ':', str(time))
    return fixed_time 

def is_overtime(frame):
    timer_box = crop(frame, ROIS["timer_box"])
    red_ratio = _measure_red_ratio(timer_box, False)
    if red_ratio >= 0.5:
        return True
    else:
        return False

def parse_time_left_s(timer_text) -> float:
      text = str(timer_text)
      if ":" not in text:
          return 0.0

      minutes, seconds = text.split(":", 1)
      return int(minutes) * 60 + int(seconds)


def total_remaining_seconds(time_left, overtime):
    if overtime:
        return time_left
    return time_left + OVERTIME_SECONDS
