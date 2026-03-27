from pathlib import Path
from typing import Any
import os

import cv2

from extractors.cards import extract_hand_state
from extractors.elixir import extract_elixir
from extractors.timer import extract_time
from extractors.tower_hp import extract_tower_hp
from extractors.health import filter_real_bars
from image_utils import draw_rois
from rois import ROIS


ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR_WEIGHTS = [
    ROOT / "detector1_v0.7.13.pt",
    ROOT / "detector2_v0.7.13.pt",
]


def load_yolo_runtime():
    try:
        from scripts.run_seed_inference import CombinedDetector, draw_boxes, idx2unit
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "YOLO runtime dependencies are missing. Activate the training/inference env "
            "(for example `.venv-train`) before running main.py."
        ) from exc
    return CombinedDetector, draw_boxes, idx2unit


def build_detector() -> Any:
    CombinedDetector, _, _ = load_yolo_runtime()
    missing = [str(path) for path in DEFAULT_DETECTOR_WEIGHTS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing detector weights: " + ", ".join(missing)
        )
    return CombinedDetector(DEFAULT_DETECTOR_WEIGHTS, conf=0.25, iou=0.6)


def summarize_detections(boxes) -> str:
    _, _, idx2unit = load_yolo_runtime()
    if boxes.numel() == 0:
        return "none"

    counts: dict[str, int] = {}
    for row in boxes.cpu().numpy():
        label = idx2unit.get(int(row[5]), str(int(row[5])))
        team = "enemy" if int(row[6]) == 1 else "ally"
        key = f"{label}:{team}"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(
        f"{label} x{count}" for label, count in sorted(counts.items())
    )


def process_frame(frame, detector, show_rois: bool = False):
    _, draw_boxes, _ = load_yolo_runtime()
    frame_to_analyze = draw_rois(frame, ROIS) if show_rois else frame
    yolo_boxes = detector.infer(frame)
    rendered = draw_boxes(frame_to_analyze, yolo_boxes)
    elixir = extract_elixir(frame)
    towers_hp = extract_tower_hp(frame)
    current_time = extract_time(frame)
    state = extract_hand_state(frame)
    return {
        "rendered": rendered,
        "elixir": elixir,
        "towers_hp": towers_hp,
        "time": current_time,
        "state": state,
        "yolo_boxes": yolo_boxes,
    }

def convert_yolo(boxes):
    troops = []
    bars = []
    not_troops = ["bar-level", "tower-bar", "queen-tower", "emote", "evolution-symbol", "elixir"]
    _, _, idx2unit = load_yolo_runtime()
    for box in boxes:
        x1, y1, x2, y2, conf, cls, team = box.tolist()
        class_name = idx2unit[int(cls)]
        team_name = "enemy" if int(team) == 1 else "ally"
        if class_name == "bar":
            bars.append({
                "class_name": class_name,
                "team": team_name,
                "confusion": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2, 
                "center_y": (y1 + y2) / 2
                })
        elif class_name not in not_troops:
            troops.append({
                "class_name": class_name,
                "team": team_name,
                "confusion": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2, 
                "center_y": (y1 + y2) / 2
                })

    return troops, bars

def find_corresponding_bar(troop, bars):
    corresponding_bar = None
    
    best_score = 100000 # the lower the better

    for bar in bars:
        x_distance = abs(bar["center_x"] - troop["center_x"])
        vertical_gap = troop["center_y"] - bar["center_y"]

        if vertical_gap < 0 or bar["team"] != troop["team"]:
            continue
        score = x_distance + 2 * vertical_gap
        if score < best_score:
            best_score = score
            corresponding_bar = bar
    return corresponding_bar

def match_troops_to_bars(troops, bars):
    matches = []
    available_bars = bars.copy()
    for troop in troops:
        corresponding_bar = find_corresponding_bar(troop, available_bars)
        matches.append({"troop": troop, "bar": corresponding_bar})
        if corresponding_bar is not None:
            available_bars.remove(corresponding_bar)

    return matches



def main(debug: bool):
    detector = build_detector()

    if debug:
        frame = cv2.imread("data/video_clips/output.png")
        if frame is None:
            raise FileNotFoundError("Failed to read data/video_clips/test.png")

        result = process_frame(frame, detector, show_rois=False)
        has_display = bool(os.environ.get("DISPLAY"))
        if has_display:
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("debug", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("debug", result["rendered"])

        elixir = result["elixir"]

        print(elixir["displayed_digit"], elixir["displayed_digit"][0])
        print(f"Estimated elixir {elixir['estimated_value'] + elixir['displayed_digit'][0]}")
        print(f"Towers hp {result['towers_hp']}")
        print(f"Current time {result['time']}")
        print(f"Current hand {result['state']}")
        print(f"YOLO detections {summarize_detections(result['yolo_boxes'])}")


        troops, bars = convert_yolo(result['yolo_boxes'])
        real_bars = filter_real_bars(frame, bars)
        matches = match_troops_to_bars(troops, real_bars)
        print(f"matches {matches}")
        if has_display:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture("/dev/video37", cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open", cap)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break

        result = process_frame(frame, detector, show_rois=False)
        cv2.imshow("feed", result["rendered"])

        elixir = result["elixir"]
        detection_summary = summarize_detections(result["yolo_boxes"])
        print(
            f"elixir {elixir['estimated_value'] + elixir['displayed_digit'][0]} | "
            f"towers {result['towers_hp']} | "
            f"time {result['time']} | "
            f"state {result['state']} | "
            f"yolo {detection_summary}"
        )
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(True)
