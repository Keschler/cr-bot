from __future__ import annotations

import cv2
import numpy as np
import sys

from cr_bot.domain.rois import ROIS
from cr_bot.paths import KATACR_ROOT
from cr_bot.vision.cards import extract_hand_state
from cr_bot.vision.elixir import extract_elixir
from cr_bot.vision.health import estimate_health
from cr_bot.vision.image_utils import detect_elixir_change, draw_rois
from cr_bot.vision.timer import (
    extract_time,
    is_overtime,
    parse_time_left_s,
    total_remaining_seconds,
)
from cr_bot.vision.tower_hp import extract_tower_hp
from cr_bot.vision.units import match_from_dict, match_troops_to_bars
from cr_bot.vision.yolo_runtime import (
    convert_yolo,
    extract_clock_boxes,
    extract_emote_boxes,
    load_yolo_runtime,
    remap_boxes_to_frame,
)

if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.build_dataset.utils.split_part import process_part, ratio2name


PROCESSING_RESOLUTION = (1080, 2400)  # width, height


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if (width, height) == PROCESSING_RESOLUTION:
        return frame
    return cv2.resize(frame, PROCESSING_RESOLUTION, interpolation=cv2.INTER_AREA)


def process_frame(
    frame,
    detector,
    show_rois: bool = False,
    yolo_tower_hp_detections: bool = False,
):
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
    arena_px = (crop_x, crop_y, crop_w, crop_h)

    result = detector.infer(arena)
    yolo_boxes = result.get_data()
    tower_hp_yolo_boxes = getattr(result, "untracked_data", yolo_boxes)
    yolo_boxes = remap_boxes_to_frame(
        yolo_boxes,
        arena.shape,
        arena_px,
    )
    clock_boxes = extract_clock_boxes(yolo_boxes)
    emote_boxes = extract_emote_boxes(yolo_boxes)
    tower_hp_yolo_boxes = remap_boxes_to_frame(
        tower_hp_yolo_boxes,
        arena.shape,
        arena_px,
    )

    rendered = draw_boxes(frame_to_analyze, yolo_boxes)
    elixir = extract_elixir(frame)
    elixir_change = detect_elixir_change(frame)

    tower_hp_debug_steps = {}
    timer_debug_steps = {}
    if yolo_tower_hp_detections:
        towers_hp = extract_tower_hp(
            frame,
            tower_hp_yolo_boxes,
            debug_steps_by_tower=tower_hp_debug_steps,
            support_tower_yolo_boxes=tower_hp_yolo_boxes,
        )
        current_time_text = extract_time(frame, debug_steps=timer_debug_steps)
    else:
        towers_hp = extract_tower_hp(
            frame,
            debug_steps_by_tower=tower_hp_debug_steps,
            support_tower_yolo_boxes=tower_hp_yolo_boxes,
        )
        current_time_text = extract_time(
            frame,
            debug_steps=timer_debug_steps,
            yolo_templates=yolo_tower_hp_detections,
        )

    overtime = is_overtime(frame)
    time_left_s = parse_time_left_s(current_time_text)
    total_remaining_s = total_remaining_seconds(time_left_s, overtime)

    state = extract_hand_state(frame)

    troops, bars = convert_yolo(yolo_boxes)
    estimate_health(frame, bars)
    matches = match_troops_to_bars(troops, bars)
    typed_matches = [match_from_dict(m) for m in matches]
    return {
        "rendered": rendered,
        "elixir": elixir,
        "elixir_change": elixir_change,
        "towers_hp": towers_hp,
        "time": current_time_text,
        "time_left_s": time_left_s,
        "total_remaining_s": total_remaining_s,
        "overtime": overtime,
        "state": state,
        "yolo_boxes": yolo_boxes,
        "clock_boxes": clock_boxes,
        "emote_boxes": emote_boxes,
        "matches": typed_matches,
        "arena_px": arena_px,
        "tower_hp_debug_steps": tower_hp_debug_steps,
        "timer_debug_steps": timer_debug_steps,
    }
