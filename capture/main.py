import cv2

from extractors.elixir import extract_elixir
from extractors.tower_hp import extract_tower_hp
from extractors.timer import extract_time
from extractors.cards import extract_hand_state
from image_utils import draw_rois
from rois import ROIS


def main():
    frame = cv2.imread("image.png")
    print("frame", frame.shape)
    #frame = draw_rois(frame, ROIS)

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("debug", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("debug", frame)
    #cv2.waitKey(0)

    elixir = extract_elixir(frame)
    print(elixir["displayed_digit"], elixir["displayed_digit"][0])
    print(f"Estimated elixir {elixir['estimated_value']}")

    towers_hp = extract_tower_hp(frame)
    print(f"Towers hp{towers_hp}")

    time = extract_time(frame)
    print(f"Current Time{time}")

    state = extract_hand_state(frame)
    print(f"Current hand{state}")




if __name__ == "__main__":
    main()
