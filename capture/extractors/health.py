import cv2
import numpy as np
from constants import (
    ALLY_CYAN_HSV_HIGH,
    ALLY_CYAN_HSV_LOW,
    ENEMY_RED_HSV_HIGH_1,
    ENEMY_RED_HSV_HIGH_2,
    ENEMY_RED_HSV_LOW_1,
    ENEMY_RED_HSV_LOW_2,
)
from image_utils import _extract_king_bar_fill


def _leading_fill_fraction(mask: np.ndarray) -> float:
    filled_per_col = np.sum(mask > 0, axis=0)
    threshold = max(1, mask.shape[0] // 4)
    active = filled_per_col > threshold
    if not np.any(active) or not active[0]:
        return 0.0

    inactive = np.where(~active)[0]
    filled_width = int(inactive[0]) if len(inactive) > 0 else int(mask.shape[1])
    return filled_width / max(mask.shape[1], 1)


def estimate_health(frame, bars):
    for bar in bars:
        health_bar = frame[
          int(bar["y1"]):int(bar["y2"]),
          int(bar["x1"]):int(bar["x2"])
        ]
        fill = _extract_king_bar_fill(health_bar)
        h = fill.shape[0]
        fill_band = fill[h // 3:(2 * h) // 3, :]
        hsv = cv2.cvtColor(fill_band, cv2.COLOR_BGR2HSV)
        
        cyan_lower = np.array(ALLY_CYAN_HSV_LOW)
        cyan_upper = np.array(ALLY_CYAN_HSV_HIGH)

        red_lower_1 = np.array(ENEMY_RED_HSV_LOW_1)
        red_upper_1 = np.array(ENEMY_RED_HSV_HIGH_1)

        red_lower_2 = np.array(ENEMY_RED_HSV_LOW_2)
        red_upper_2 = np.array(ENEMY_RED_HSV_HIGH_2)

        cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

        red_mask_1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv, red_lower_2, red_upper_2)

        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        if bar["team"] == "ally":
            mask = cyan_mask
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            b = fill_band[:, :, 0].astype(np.int16)
            g = fill_band[:, :, 1].astype(np.int16)
            r = fill_band[:, :, 2].astype(np.int16)
            strong_red = ((r > g + 35) & (r > b + 35)).astype(np.uint8) * 255
            mask = cv2.bitwise_and(red_mask, strong_red)
            kernel = np.ones((2, 2), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        frac = _leading_fill_fraction(mask)

        if frac == 0.0: # If the troop was hit in the frame, making the hp bar white
            white_lower = np.array((0,0,185))
            white_upper = np.array((179,55,255))
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            kernel = np.ones((2,2), dtype=np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            frac = _leading_fill_fraction(white_mask)
        bar["estimated_hp"] = frac
