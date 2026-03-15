import cv2
import numpy as np


def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def draw_rois(frame, rois):
    color = (0, 255, 0)
    count = 0
    for name, (x, y, w, h) in rois.items():
        color = (255, 0, 0) if count % 2 == 0 else (0, 255, 0)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def preprocess_digit(img):
    if img is None:
        raise ValueError("preprocess_digit received None")

    if len(img.shape) == 2:
        img_gray = img
    elif len(img.shape) == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY)
    return thresholded


def estimate_slot_fraction(slot_img):
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(col_mean, kernel, mode="same")

    edge_width = max(3, len(smooth) // 5)
    left_level = np.mean(smooth[:edge_width])
    right_level = np.mean(smooth[-edge_width:])

    if left_level - right_level < 5:
        return 0.0

    threshold = (left_level + right_level) / 2
    filled_cols = np.where(smooth >= threshold)[0]

    if len(filled_cols) == 0:
        return 0.0

    rightmost = filled_cols[-1]
    fraction = (rightmost + 1) / len(smooth)
    return float(np.clip(fraction, 0.0, 1.0))



def segment_digits(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
    
        if w > 4 and h > 8:
            boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    return boxes

def extract_digit_images(binary_img, boxes):
    digits = []
    for x,y,w,h in boxes:
        digit = binary_img[y:y+h, x:x+w]
        digits.append(digit)
    return digits

def classify_digit(digit_img, templates, out_size=(32,48)):
    digit_img = cv2.resize(digit_img, out_size)

    best_digit = None
    best_score = -1.0

    for digit, tmpl in templates.items():
        tmpl_resized = cv2.resize(tmpl, out_size)
        result = cv2.matchTemplate(digit_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        score = result[0,0]

        if score > best_score:
            best_score = score
            best_digit = digit
    if best_score <= 0:
        return 0, best_score

    return best_digit, best_score 

def show_digit_segmentation_debug(binary_img):
      contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      contour_view = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
      box_view = contour_view.copy()

      boxes = []
      for c in contours:
          cv2.drawContours(contour_view, [c], -1, (0, 255, 0), 1)

          x, y, w, h = cv2.boundingRect(c)
          if w > 4 and h > 8:
              boxes.append((x, y, w, h))
              cv2.rectangle(box_view, (x, y), (x + w, y + h), (0, 0, 255), 1)

      boxes.sort(key=lambda b: b[0])

      cv2.imshow("binary", binary_img)
      cv2.imshow("contours", contour_view)
      cv2.imshow("boxes", box_view)
      cv2.waitKey(0)

      return boxes


def read_number_from_roi(img, templates):
    binary = preprocess_digit(img)
    boxes = show_digit_segmentation_debug(binary)
    digit_imgs = extract_digit_images(binary, boxes)


    chars = []
    for dimg in digit_imgs:
        digit, score = classify_digit(dimg, templates)
        print(f"{digit} got {score}")
        chars.append(str(digit)) # Classify each digit seperately
    
    if not chars:
        return None

    return int("".join(chars))

