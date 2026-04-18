import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from numpy import true_divide


from extractors.cards import extract_hand_state
from extractors.elixir import extract_elixir
from extractors.timer import extract_time, is_overtime, parse_time_left_s, total_remaining_seconds
from extractors.tower_hp import extract_tower_hp
from extractors.health import filter_real_bars, estimate_health
from extractors.units import match_troops_to_bars, match_from_dict
from image_utils import draw_rois
from rois import ROIS
from game_state import GameState, HudState, PrincessTowerState
from vision.yolo_runtime import load_yolo_runtime, build_detector, parse_box_row, summarize_detections, remap_boxes_to_frame, convert_yolo
from trackers.enemy_cards import EnemyCardTracker 
from katacr.build_dataset.utils.split_part import process_part, ratio2name




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



def process_frame(frame, detector, show_rois: bool = False):
    _, draw_boxes, _ = load_yolo_runtime()
    frame_to_analyze = draw_rois(frame, ROIS) if show_rois else frame
    ratio_name = ratio2name(frame)
    if ratio_name is None:
        height, width = frame.shape[:2]
        raise ValueError(
            f"Unsupported frame aspect ratio {height / width:.4f} for KataCR part2 crop "
            f"(frame shape: {frame.shape})"
        )
    arena, box_params = process_part(frame, 2, verbose=True)
    fx, fy, fw, fh = box_params
    frame_h, frame_w = frame.shape[:2]

    crop_x = int(frame_w * fx)
    crop_y = int(frame_h * fy)
    crop_w = int(frame_w * fw)
    crop_h = int(frame_h * fh)

    result = detector.infer(arena)
    yolo_boxes = result.get_data()
    yolo_boxes = remap_boxes_to_frame(
      yolo_boxes,
      arena.shape,
      (crop_x, crop_y, crop_w, crop_h),
    )

    rendered = draw_boxes(frame_to_analyze, yolo_boxes)
    elixir = extract_elixir(frame)
    towers_hp = extract_tower_hp(frame)

    current_time_text = extract_time(frame)
    overtime = is_overtime(frame)
    time_left_s = parse_time_left_s(current_time_text)
    total_remaining_s = total_remaining_seconds(time_left_s, overtime)

    state = extract_hand_state(frame)

    troops, bars = convert_yolo(yolo_boxes)
    real_bars = filter_real_bars(frame, bars)
    estimate_health(frame, real_bars)
    matches = match_troops_to_bars(troops, real_bars)
    typed_matches = [match_from_dict(m) for m in matches]
    return {
        "rendered": rendered,
        "elixir": elixir,
        "towers_hp": towers_hp,
        "time": current_time_text,
        "time_left_s": time_left_s,
        "total_remaining_s": total_remaining_s,
        "overtime": overtime,
        "state": state,
        "yolo_boxes": yolo_boxes,
        "matches": typed_matches
    }

def build_game_state(result, *, seen_enemy_cards=None, elixir_enemy_est=None):
    towers_hp = result["towers_hp"]
    hand = result["state"]

    def tower_alive(hp):
        return hp is not None and hp > 0

    def king_active(hp):
        return hp is not None and hp < 7032

    princess_towers = PrincessTowerState(
        own_left_alive=tower_alive(towers_hp["own_support_left"]),
          own_right_alive=tower_alive(towers_hp["own_support_right"]),
          enemy_left_alive=tower_alive(towers_hp["enemy_support_left"]),
          enemy_right_alive=tower_alive(towers_hp["enemy_support_right"]),
    )

    hud = HudState(
        time_left_s=result["time_left_s"],
          overtime=result["overtime"],
          elixir_self=result["elixir"]["estimated_value"] + result["elixir"]["displayed_digit"][1],
          hand_cards=[
              hand["card_1"],
              hand["card_2"],
              hand["card_3"],
              hand["card_4"],
          ],
          next_card=hand["next_card"],
          tower_hp_self=[
              towers_hp["own_support_left"],
              towers_hp["own_king"],
              towers_hp["own_support_right"],
          ],
          tower_hp_enemy=[
              towers_hp["enemy_support_left"],
              towers_hp["enemy_king"],
              towers_hp["enemy_support_right"],
          ],
          princess_towers=princess_towers,
    )
    
    own_units = [m for m in result["matches"] if m.troop.team == "ally"]
    enemy_units = [m for m in result["matches"] if m.troop.team == "enemy"]

    return GameState(
        hud=hud,
        total_remaining_s=result["total_remaining_s"],
        own_units=own_units,
        enemy_units=enemy_units,
        seen_enemy_cards=seen_enemy_cards or [],
        elixir_enemy_est=0.0 if elixir_enemy_est is None else elixir_enemy_est,
        own_king_active=king_active(towers_hp["own_king"]),
        enemy_king_active=king_active(towers_hp["enemy_king"]),
    )


def main(debug: bool):
    detector = build_detector()
    enemy_card_tracker = EnemyCardTracker()

    if debug:
        frame = cv2.imread("pictures/screen.png")
        if frame is None:
            raise FileNotFoundError(f"Failed to read {debug_frame}")

        result = process_frame(frame, detector, show_rois=False)
        game_state = build_game_state(result)
        has_display = os.environ.get("SHOW_DEBUG_WINDOW") == "1"
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
        print(f"Overtime {result['overtime']}")


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
    input("Start")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break

        result = process_frame(frame, detector, show_rois=False)
        enemy_card_tracker.update(result["total_remaining_s"], result["matches"])
        print()
        print(f"enemy elixir est: {enemy_card_tracker.elixir_enemy_est:.2f}")
        print(f"seen enemy cards: {sorted(enemy_card_tracker.confirmed_seen_cards)}")
        print(f"enemy plays: {enemy_card_tracker.detected_card_plays[-5:]}")
        print()
        
        game_state = build_game_state(
            result, 
            seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards), 
            elixir_enemy_est=enemy_card_tracker.elixir_enemy_est
        )

        cv2.imshow("feed", result["rendered"])

        elixir = result["elixir"]
        detection_summary = summarize_detections(result["yolo_boxes"])
        print(f"time:   {result['time']}")
        print(f"elixir: {elixir['estimated_value'] + elixir['displayed_digit'][1]:.2f}")
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
              f"conf={m.troop.confidence:.3f} "
              f"hp={m.troop.estimated_hp}"
            )
        print()

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(False)
