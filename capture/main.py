from pathlib import Path
from typing import Any
from dataclasses import dataclass
import os
import json
import time

import cv2

from extractors.cards import extract_hand_state
from extractors.elixir import extract_elixir
from extractors.timer import extract_time
from extractors.tower_hp import extract_tower_hp
from extractors.health import filter_real_bars, estimate_health
from troop_hp_level16 import get_troop_hp_level16
from image_utils import draw_rois
from rois import ROIS


ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR_WEIGHTS = [
    ROOT / "detector1_v0.7.13.pt",
    ROOT / "detector2_v0.7.13.pt",
]


def get_default_video_device() -> str:
    env_device = os.environ.get("VIDEO_DEVICE")
    if env_device:
        return env_device

    for path in sorted(Path("/sys/class/video4linux").glob("video*/name")):
        try:
            if "dummy video device" in path.read_text().lower():
                return f"/dev/{path.parent.name}"
        except OSError:
            continue

    return "/dev/video37"

@dataclass(slots=True)
class Detection:
    class_name: str
    team: str
    confusion: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    estimated_hp: float | None=None

@dataclass(slots=True)
class Match:
    troop: Detection
    bar: Detection | None


@dataclass(slots=True)
class GameState:
    time_left_s: float
    elixir_self: float
    #elixir_enemy: float 
    tower_hp_self: list[float]
    tower_hp_enemy: list[float]
    hand_cards: list[str]
    next_card: str



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
    timings: dict[str, float] = {}

    start = time.perf_counter()
    frame_to_analyze = draw_rois(frame, ROIS) if show_rois else frame
    timings["draw_rois"] = time.perf_counter() - start

    start = time.perf_counter()
    yolo_boxes = detector.infer(frame)
    timings["infer"] = time.perf_counter() - start

    start = time.perf_counter()
    rendered = draw_boxes(frame_to_analyze, yolo_boxes)
    timings["draw_boxes"] = time.perf_counter() - start

    start = time.perf_counter()
    elixir = extract_elixir(frame)
    timings["extract_elixir"] = time.perf_counter() - start

    start = time.perf_counter()
    towers_hp = extract_tower_hp(frame)
    timings["extract_tower_hp"] = time.perf_counter() - start

    start = time.perf_counter()
    current_time = extract_time(frame)
    timings["extract_time"] = time.perf_counter() - start

    start = time.perf_counter()
    state = extract_hand_state(frame)
    timings["extract_hand_state"] = time.perf_counter() - start

    start = time.perf_counter()
    troops, bars = convert_yolo(yolo_boxes)
    timings["convert_yolo"] = time.perf_counter() - start

    start = time.perf_counter()
    real_bars = filter_real_bars(frame, bars)
    timings["filter_real_bars"] = time.perf_counter() - start

    start = time.perf_counter()
    estimate_health(frame, real_bars)
    timings["estimate_health"] = time.perf_counter() - start

    start = time.perf_counter()
    matches = match_troops_to_bars(troops, real_bars)
    timings["match_troops_to_bars"] = time.perf_counter() - start

    start = time.perf_counter()
    typed_matches = [match_from_dict(m) for m in matches]
    timings["match_from_dict"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    return {
        "rendered": rendered,
        "elixir": elixir,
        "towers_hp": towers_hp,
        "time": current_time,
        "state": state,
        "yolo_boxes": yolo_boxes,
        "matches": typed_matches,
        "timings": timings,
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

def detection_from_dict(d: dict) -> Detection:
      return Detection(
          class_name=d["class_name"],
          team=d["team"],
          confusion=d["confusion"],
          x1=d["x1"],
          y1=d["y1"],
          x2=d["x2"],
          y2=d["y2"],
          center_x=d["center_x"],
          center_y=d["center_y"],
          estimated_hp=d.get("estimated_hp")
      )

def match_from_dict(d: dict) -> Match:

    troop = detection_from_dict(d["troop"])
    bar = detection_from_dict(d["bar"]) if d["bar"] is not None else None

    if bar is not None:
        troop_hp = get_troop_hp_level16(troop.class_name)
        if troop_hp:
            troop.estimated_hp = bar.estimated_hp * get_troop_hp_level16(troop.class_name)
    else:
        troop.estimated_hp = get_troop_hp_level16(troop.class_name)

    return Match(
    troop=troop,
    bar=bar,
    )


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


        if has_display:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    video_device = get_default_video_device()

    if not Path(video_device).exists():
        raise FileNotFoundError(
            f"Missing video device {video_device}. "
            "Create a v4l2loopback device and point VIDEO_DEVICE at it."
        )

    cap = cv2.VideoCapture(video_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video device {video_device}. "
            "Check v4l2loopback, permissions, and that scrcpy is writing to the same device."
        )

    cv2.namedWindow("feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break

        result = process_frame(frame, detector, show_rois=False)
        cv2.imshow("feed", result["rendered"])

        elixir = result["elixir"]
        detection_summary = summarize_detections(result["yolo_boxes"])
        print(f"time:   {result['time']}")
        print(f"elixir: {elixir['estimated_value'] + elixir['displayed_digit'][0]:.2f}")
        print(f"yolo:   {detection_summary}")

        print("towers:")
        for name, hp in result["towers_hp"].items():
            print(f"{name}: {hp}")

        print("state:")
        for slot, value in result["state"].items():
            print(f"  {slot}: {value}")

        print("matches:")
        for m in result["matches"]:
            print(
              f"  troop={m.troop.class_name:<18} "
              f"team={m.troop.team:<5} "
              f"conf={m.troop.confusion:.3f} "
              f"hp={m.troop.estimated_hp}"
            )
        print()

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(False)
