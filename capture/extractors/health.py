from image_utils import extract_health_bar
import cv2
import numpy as np

def get_valid_troop_bars(frame, matched):
    valid_troop_bars = []
    for match in matched:
        bar = match["bar"]
        if bar is None:
            continue
        print(match["troop"]["class_name"])
        if extract_health_bar(frame, bar):
           valid_troop_bars.append([match["troop"], bar]) 
    return valid_troop_bars

def filter_real_bars(frame, bars):
    return [bar for bar in bars if extract_health_bar(frame, bar)]


def estimate_health(frame, bars):
    for bar in bars:
        health_bar = frame[
          int(bar["y1"]):int(bar["y2"]),
          int(bar["x1"]):int(bar["x2"])
        ]
        hsv = cv2.cvtColor(health_bar, cv2.COLOR_BGR2HSV)
        
        cyan_lower = np.array([80, 80, 80])
        cyan_upper = np.array([100, 255, 255])

        red_lower_1 = np.array([0, 80, 80])
        red_upper_1 = np.array([10, 255, 255])

        red_lower_2 = np.array([170, 80, 80])
        red_upper_2 = np.array([179, 255, 255])

        cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

        red_mask_1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv, red_lower_2, red_upper_2)

        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        mask = cyan_mask if bar["team"] == "ally" else red_mask

        filled_per_col = np.sum(mask > 0, axis=0)
        threshold = max(1, mask.shape[0] // 4)
        cols = np.where(filled_per_col > threshold)[0]
        if len(cols) == 0:
            bar["estimated_hp"] = 0.0
            continue
        filled_width = cols[-1] - cols[0] + 1

        bar["estimated_hp"] = filled_width / mask.shape[1]
