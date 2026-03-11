import subprocess
from sys import thread_info
import numpy as np
import cv2


def capture_frame():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )
    img = np.frombuffer(result.stdout, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)

'''
frame = capture_frame()
cv2.imshow("screen", frame)
cv2.waitKey(1)
'''


ROIS = {
    "elixir_bar": (240, 2310, 805, 150),
    "elixir_number": (295, 2309, 53, 53),
    "elixir_slot_1": (270, 2320, 100, 50),
    "elixir_slot_2": (348, 2320, 100, 50),
    "elixir_slot_3": (426, 2320, 100, 50),
    "elixir_slot_4": (504, 2320, 100, 50),
    "elixir_slot_5": (582, 2320, 100, 50),
    "elixir_slot_6": (660, 2320, 100, 50),
    "elixir_slot_7": (738, 2320, 100, 50),
    "elixir_slot_8": (816, 2320, 100, 50),
    "elixir_slot_9": (894, 2320, 100, 50),
    "elixir_slot_10": (972, 2320, 68, 50),
    "card_1": (230, 2020, 220, 300),
    "card_2": (430, 2020, 220, 300),
    "card_3": (630, 2020, 220, 300),
    "card_4": (840, 2020, 220, 300),
    "next_card": (40, 2260, 120, 125),

    "enemy_king_tower": (425, 300, 235, 215),
    "enemy_support_tower_left": (105, 440, 245, 220),
    "enemy_support_tower_right": (730, 440, 245, 220),
    "enemy_king_hp_bar": (400, 170, 280, 70),
    "enemy_support_hp_bar_left": (120, 380, 195, 60),
    "enemy_support_hp_bar_right": (750, 380, 205, 60),

    "own_king_tower": (425, 1570, 235, 235),
    "own_support_tower_left": (100, 1345, 250, 225),
    "own_support_tower_right": (730, 1345, 250, 225),
    "own_king_hp_bar": (400, 1740, 280, 70),
    "own_support_hp_bar_left": (120, 1420, 195, 60),
    "own_support_hp_bar_right": (750, 1420, 205, 60),
}

def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

def draw_rois(frame):
    color = (0,255,0)
    count = 0
    for name, (x,y,w,h) in ROIS.items():
        if count % 2 == 0:
            color = (255, 0, 0)
        else:
            color = (0,255,0)
        count += 1
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def estimate_slot_fraction(slot_img):
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    
    col_mean = gray.mean(axis=0)
    
    k = 7
    kernel = np.ones(k) / k
    smooth = np.convolve(col_mean, kernel, mode="same")

    left_level = np.mean(smooth[:max(3, len(smooth)//5)])
    right_level = np.mean(smooth[-max(3, len(smooth)//5):])

    if left_level - right_level < 5:
        return 0.0
    
    threshold = (left_level + right_level) / 2

    filled_cols = np.where(smooth >= threshold)[0]

    if len(filled_cols) == 0:
        return 0
    rightmost = filled_cols[:-1]
    fraction = (rightmost + 1) / len(smooth)
    return float(np.clip(fraction, 0.0 , 1.0))



def read_elixir_value(displayed_digit, slot_rois, frame):
    n = displayed_digit

    if n>=10:
        return 10
    next_slot = crop(frame, slot_rois[n])
    frac = estimate_slot_fraction(next_slot)

def preprocess_digit(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    
    return th


frame = cv2.imread("screen.png")
print("frame", frame.shape)
frame = draw_rois(frame)



elixir_digit = preprocess_digit(cv2.imread("templates/9.png"))

#displayed_digit = 
#elixir_estimate = read_elixir_value()



cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("debug", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("debug", elixir_digit)
cv2.waitKey(0)
