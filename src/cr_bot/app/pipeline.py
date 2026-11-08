from __future__ import annotations

import time

import cv2
import numpy as np

from cr_bot.domain.frame_analysis import FrameAnalysisResult
from cr_bot.domain.rois import ROIS
from cr_bot.vision.cards import extract_hand_state
from cr_bot.vision.elixir import extract_elixir
from cr_bot.vision.health import estimate_health
from cr_bot.vision.image_utils import detect_elixir_change, draw_rois
from cr_bot.vision.katacr_runtime import load_split_part_tools
from cr_bot.vision.timer import (
    extract_timer,
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

process_part, ratio2name = load_split_part_tools()


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
    *,
    rois=None,
    native_frame=None,
    collect_timing: bool = False,
) -> FrameAnalysisResult:
    vision_timing_ms: dict[str, float] = {}
    stage_started = time.perf_counter() if collect_timing else 0.0

    def record_timing(name: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        vision_timing_ms[name] = (now - stage_started) * 1000.0
        stage_started = now

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
    if collect_timing:
        record_timing("crop")

    result = detector.infer(arena)
    yolo_boxes = result.get_data()
    tower_hp_yolo_boxes = getattr(result, "untracked_data", yolo_boxes)
    if collect_timing:
        record_timing("detector")
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
    if collect_timing:
        record_timing("yolo_postprocess")

    rendered = draw_boxes(frame_to_analyze, yolo_boxes)
    if collect_timing:
        record_timing("draw_boxes")
    tower_hp_debug_steps = {}
    timer_debug_steps = {}
    if rois is None or native_frame is None:
        elixir = extract_elixir(frame)
        elixir_change = detect_elixir_change(frame)
        if collect_timing:
            record_timing("elixir")

        if yolo_tower_hp_detections:
            towers_hp = extract_tower_hp(
                frame,
                tower_hp_yolo_boxes,
                debug_steps_by_tower=tower_hp_debug_steps,
                support_tower_yolo_boxes=tower_hp_yolo_boxes,
            )
            if collect_timing:
                record_timing("tower_hp")
            timer_reading = extract_timer(frame, debug_steps=timer_debug_steps)
        else:
            towers_hp = extract_tower_hp(
                frame,
                debug_steps_by_tower=tower_hp_debug_steps,
                support_tower_yolo_boxes=tower_hp_yolo_boxes,
            )
            if collect_timing:
                record_timing("tower_hp")
            timer_reading = extract_timer(
                frame,
                debug_steps=timer_debug_steps,
                yolo_templates=yolo_tower_hp_detections,
            )

        current_time_text, overtime = timer_reading.text, timer_reading.overtime
        time_left_s = parse_time_left_s(current_time_text)
        total_remaining_s = total_remaining_seconds(time_left_s, overtime)
        if collect_timing:
            record_timing("timer")

        hand_state = extract_hand_state(frame)
    else:
        elixir = extract_elixir(native_frame, rois=rois)
        elixir_change = detect_elixir_change(native_frame, rois=rois)
        if collect_timing:
            record_timing("elixir")

        if yolo_tower_hp_detections:
            towers_hp = extract_tower_hp(
                frame,
                tower_hp_yolo_boxes,
                debug_steps_by_tower=tower_hp_debug_steps,
                support_tower_yolo_boxes=tower_hp_yolo_boxes,
            )
            if collect_timing:
                record_timing("tower_hp")
            timer_reading = extract_timer(
                native_frame, debug_steps=timer_debug_steps, rois=rois
            )
        else:
            towers_hp = extract_tower_hp(
                native_frame,
                debug_steps_by_tower=tower_hp_debug_steps,
                support_tower_yolo_boxes=tower_hp_yolo_boxes,
                rois=rois,
            )
            if collect_timing:
                record_timing("tower_hp")
            timer_reading = extract_timer(
                native_frame,
                debug_steps=timer_debug_steps,
                yolo_templates=yolo_tower_hp_detections,
                rois=rois,
            )

        current_time_text, overtime = timer_reading.text, timer_reading.overtime
        time_left_s = parse_time_left_s(current_time_text)
        total_remaining_s = total_remaining_seconds(time_left_s, overtime)
        if collect_timing:
            record_timing("timer")

        hand_state = extract_hand_state(native_frame, rois=rois)
    if collect_timing:
        record_timing("hand_state")

    troops, bars = convert_yolo(yolo_boxes)
    estimate_health(frame, bars)
    matches = match_troops_to_bars(troops, bars)
    typed_matches = [match_from_dict(m) for m in matches]
    if collect_timing:
        record_timing("unit_postprocess")
    return FrameAnalysisResult(
        rendered=rendered,
        elixir=elixir,
        elixir_change=elixir_change,
        towers_hp=towers_hp,
        time=current_time_text,
        time_left_s=time_left_s,
        total_remaining_s=total_remaining_s,
        overtime=overtime,
        hand_state=hand_state,
        yolo_boxes=yolo_boxes,
        clock_boxes=clock_boxes,
        emote_boxes=emote_boxes,
        matches=typed_matches,
        arena_px=arena_px,
        tower_hp_debug_steps=tower_hp_debug_steps,
        timer_debug_steps=timer_debug_steps,
        vision_timing_ms=vision_timing_ms,
    )
