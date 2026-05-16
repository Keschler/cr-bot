import os
from pathlib import Path

import cv2
import numpy as np
import time


from extractors.cards import extract_hand_state
from extractors.elixir import extract_elixir
from extractors.timer import extract_time, is_overtime, parse_time_left_s, total_remaining_seconds
from extractors.tower_hp import extract_tower_hp
from extractors.health import estimate_health
from extractors.units import match_troops_to_bars, match_from_dict
from extractors.match_state import game_start, game_end_from_result
from image_utils import draw_rois
from rois import ROIS
from scripts.run_seed_inference import filter_excluded_classes
from katacr.build_dataset.utils.split_part import process_part, ratio2name
from state_builder import build_game_state
from vision.yolo_runtime import load_yolo_runtime, build_detector, summarize_detections, remap_boxes_to_frame, convert_yolo, extract_clock_boxes
from trackers.enemy_cards import EnemyCardTracker 
from trackers.match_clock import MatchClockFilter
from trackers.own_actions import OwnActionTracker
from trackers.tower_hp_filter import TowerHPFilter


PROCESSING_RESOLUTION = (1080, 2400)  # width, height


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if (width, height) == PROCESSING_RESOLUTION:
        return frame
    return cv2.resize(frame, PROCESSING_RESOLUTION, interpolation=cv2.INTER_AREA)



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
    tower_hp_yolo_boxes = remap_boxes_to_frame(
      tower_hp_yolo_boxes,
      arena.shape,
      arena_px,
    )

    rendered = draw_boxes(frame_to_analyze, yolo_boxes)
    elixir = extract_elixir(frame)

    tower_hp_debug_steps = {}
    timer_debug_steps = {}
    if yolo_tower_hp_detections:
        towers_hp = extract_tower_hp(frame, tower_hp_yolo_boxes, debug_steps_by_tower=tower_hp_debug_steps)
        current_time_text = extract_time(frame, debug_steps=timer_debug_steps)
    else:
        towers_hp = extract_tower_hp(frame, debug_steps_by_tower=tower_hp_debug_steps)
        current_time_text = extract_time(frame, debug_steps=timer_debug_steps, yolo_templates=yolo_tower_hp_detections)


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
        "towers_hp": towers_hp,
        "time": current_time_text,
        "time_left_s": time_left_s,
        "total_remaining_s": total_remaining_s,
        "overtime": overtime,
        "state": state,
        "yolo_boxes": yolo_boxes,
        "clock_boxes": clock_boxes,
        "matches": typed_matches,
        "arena_px": arena_px,
        "tower_hp_debug_steps": tower_hp_debug_steps,
        "timer_debug_steps": timer_debug_steps,
    }


def render_debug_panel(img: np.ndarray | None, label: str, tile_w: int, tile_h: int) -> np.ndarray:
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if img is None or img.size == 0:
        cv2.putText(tile, "missing", (8, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return tile

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = min((tile_w - 10) / img.shape[1], (tile_h - 26) / img.shape[0])
    resized = cv2.resize(
        img,
        (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    y0 = 22 + (tile_h - 22 - resized.shape[0]) // 2
    x0 = (tile_w - resized.shape[1]) // 2
    tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return tile


def render_tower_hp_debug(steps_by_tower: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    order = [
        "enemy_king",
        "enemy_support_left",
        "enemy_support_right",
        "own_king",
        "own_support_left",
        "own_support_right",
    ]
    step_order = ["raw", "binary", "boxes", "digits"]
    cell_w = 180
    cell_h = 90
    rows = []

    for tower_name in order:
        row_tiles = []
        steps = steps_by_tower.get(tower_name) or {}
        for step_name in step_order:
            row_tiles.append(render_debug_panel(steps.get(step_name), f"{tower_name}:{step_name}", cell_w, cell_h))
        rows.append(np.hstack(row_tiles))

    return np.vstack(rows)



def render_timer_debug(steps: dict[str, np.ndarray]) -> np.ndarray:
    step_order = ["raw", "binary", "boxes", "digits"]
    cell_w = 220
    cell_h = 110
    tiles = [render_debug_panel(steps.get(step_name), f"timer:{step_name}", cell_w, cell_h) for step_name in step_order]
    return np.hstack(tiles)


def crop_detection(frame: np.ndarray, detection, pad: int = 6) -> np.ndarray | None:
    if detection is None:
        return None

    h, w = frame.shape[:2]
    x1 = max(0, int(detection.x1) - pad)
    y1 = max(0, int(detection.y1) - pad)
    x2 = min(w, int(detection.x2) + pad)
    y2 = min(h, int(detection.y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def render_match_debug(frame: np.ndarray, matches) -> np.ndarray:
    cell_w = 180
    cell_h = 110
    rows = []

    for idx, match in enumerate(matches):
        troop_label = f"{idx}:{match.troop.class_name}:{match.troop.team}"
        bar_label = f"bar:{match.bar.team}" if match.bar is not None else "bar:missing"
        troop_crop = crop_detection(frame, match.troop, pad=12)
        bar_crop = crop_detection(frame, match.bar, pad=6)
        row = np.hstack([
            render_debug_panel(troop_crop, troop_label, cell_w, cell_h),
            render_debug_panel(bar_crop, bar_label, cell_w, cell_h),
        ])
        rows.append(row)

    if not rows:
        return np.hstack([
            render_debug_panel(None, "troop", cell_w, cell_h),
            render_debug_panel(None, "bar", cell_w, cell_h),
        ])

    return np.vstack(rows)

def print_frame_result(result, enemy_card_tracker, own_action_tracker=None):
    if enemy_card_tracker.elixir_enemy_est is None:
        print("enemy elixir is undefined")
    else:
        print(f"enemy elixir est: {enemy_card_tracker.elixir_enemy_est:.2f}")
    print(f"seen enemy cards: {sorted(enemy_card_tracker.confirmed_seen_cards)}")
    print("enemy plays:")
    for play in enemy_card_tracker.detected_card_plays:
        print(
            f"  card={play['card']:<20} "
            f"cost={play['cost']} "
            f"time_left={play['time_left_s']} "
            f"track_id={play['track_id']}"
        )
    if own_action_tracker is not None:
        print("own plays:")
        for action in own_action_tracker.actions:
            print(
                f"  card={action['card']:<20} "
                f"slot={action['slot_idx']} "
                f"cell={action['cell']} "
                f"time_left={action['time_left_s']}"
            )
    print()

    elixir = result["elixir"]
    detection_summary = summarize_detections(result["yolo_boxes"])
    print(f"time:   {result['time_left_s']}")
    print(f"elixir: {elixir['estimated_value'] + elixir['displayed_digit'][0]}")
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


def has_visible_match_timer(result) -> bool:
    return ":" in str(result.get("time") or "")


def main(
    debug: bool,
    video: str | None = None,
    frame_stride: int = 1,
    normalize: bool = True,
    debug_frame_path: str | None = None,
    yolo_detections: bool = False,
):
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")

    detector = build_detector()
    print("detector sucessfully built!")
    enemy_card_tracker = EnemyCardTracker()
    own_action_tracker = OwnActionTracker()
    match_clock_filter = MatchClockFilter()
    tower_hp_filter = TowerHPFilter()

    if debug:
        if not debug_frame_path:
            raise RuntimeError("Pass a debug frame path when running main(debug=True).")

        frame = cv2.imread(debug_frame_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read debug frame: {debug_frame_path}")
        if normalize:
            frame = normalize_frame(frame)

        result = process_frame(frame, detector, show_rois=False, yolo_tower_hp_detections=yolo_detections)
        result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])

        enemy_card_tracker.start_match(
          result["time_left_s"],
          result["total_remaining_s"],
          now_s=time.monotonic(),
      )

        game_state = build_game_state(result)
        own_action_tracker.update(game_state, result["arena_px"])
        enemy_card_tracker.reconcile_own_actions(
            own_action_tracker.actions,
            arena_px=result["arena_px"],
        )
        enemy_card_tracker.update(
            result["total_remaining_s"],
            result["matches"],
            clock_boxes=result["clock_boxes"],
            now_s=time.monotonic(),
            own_actions=own_action_tracker.actions,
            arena_px=result["arena_px"],
        )
        has_display = os.environ.get("SHOW_DEBUG_WINDOW") == "1"
        if has_display:
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("debug", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("debug", result["rendered"])
            cv2.namedWindow("tower_hp_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("tower_hp_debug", render_tower_hp_debug(result["tower_hp_debug_steps"]))
            cv2.namedWindow("timer_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("timer_debug", render_timer_debug(result["timer_debug_steps"]))
            cv2.namedWindow("match_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("match_debug", render_match_debug(frame, result["matches"]))

        elixir = result["elixir"]
        print(f"Estimated elixir {elixir['estimated_value'] + elixir['displayed_digit'][0]}")
        print(f"Overtime {result['overtime']}")

        detection_summary = summarize_detections(result["yolo_boxes"])
        print(f"time:   {result['time']} time_left {result['total_remaining_s']}")
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

        print("enemy plays:")
        for play in enemy_card_tracker.detected_card_plays:
          print(
              f"  card={play['card']:<20} "
              f"cost={play['cost']} "
              f"time_left={play['time_left_s']} "
              f"track_id={play['track_id']}"
          )
        print("own plays:")
        for action in own_action_tracker.actions:
          print(
              f"  card={action['card']:<20} "
              f"slot={action['slot_idx']} "
              f"cell={action['cell']} "
              f"time_left={action['time_left_s']}"
          )
        print()


        if has_display:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if video:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        game_started = False
        not_in_game_streak = 0
        has_display = os.environ.get("SHOW_DEBUG_WINDOW") == "1"
        if has_display:
            cv2.namedWindow("feed", cv2.WINDOW_NORMAL)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if (frame_idx - 1) % frame_stride != 0:
                continue

            if normalize:
                frame = normalize_frame(frame)

            result = process_frame(
                frame,
                detector,
                show_rois=False,
                yolo_tower_hp_detections=yolo_detections,
            )
            video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if not game_started and (game_start(frame) or has_visible_match_timer(result)):
                game_started = True
                result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])
                match_clock_filter.initialise(result["time_left_s"], video_time_s)
                enemy_card_tracker.start_match(
                    result["time_left_s"],
                    result["total_remaining_s"],
                    now_s=video_time_s,
                )
                game_state = build_game_state(
                    result,
                    seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards),
                    elixir_enemy_est=enemy_card_tracker.elixir_enemy_est,
                    game_started=game_started,
                )
                own_action_tracker.update(game_state, result["arena_px"])
                enemy_card_tracker.reconcile_own_actions(
                    own_action_tracker.actions,
                    arena_px=result["arena_px"],
                )
            elif game_started:
                if match_clock_filter.initialised:
                    filtered_time_left_s = match_clock_filter.update(result["time_left_s"], video_time_s)
                    result["time_left_s"] = filtered_time_left_s
                    result["total_remaining_s"] = total_remaining_seconds(filtered_time_left_s, result["overtime"])
                else:
                    match_clock_filter.initialise(result["time_left_s"], video_time_s)

                result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])
                game_state = build_game_state(
                    result,
                    seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards),
                    elixir_enemy_est=enemy_card_tracker.elixir_enemy_est,
                    game_started=game_started,
                )
                own_action_tracker.update(game_state, result["arena_px"])
                enemy_card_tracker.reconcile_own_actions(
                    own_action_tracker.actions,
                    arena_px=result["arena_px"],
                )
                enemy_card_tracker.update(
                    result["total_remaining_s"],
                    result["matches"],
                    now_s=video_time_s,
                    clock_boxes=result["clock_boxes"],
                    own_actions=own_action_tracker.actions,
                    arena_px=result["arena_px"],
                )

                if game_end_from_result(result):
                    not_in_game_streak += 1
                    if not_in_game_streak >= 20:
                        game_started = False
                        not_in_game_streak = 0
                        enemy_card_tracker = EnemyCardTracker()
                        own_action_tracker = OwnActionTracker()
                        match_clock_filter = MatchClockFilter()
                        tower_hp_filter = TowerHPFilter()
                        continue
                else:
                    not_in_game_streak = 0
            else:
                print(f"frame {frame_idx}: not in game")
                continue

            print(f"frame {frame_idx} video_time={video_time_s:.2f}s")
            print_frame_result(result, enemy_card_tracker, own_action_tracker)

            if has_display:
                cv2.imshow("feed", result["rendered"])
                if cv2.waitKey(1) == 27:
                    break

        cap.release()
        if has_display:
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
    ok, warmup_frame = cap.read()
    if not ok:
      raise RuntimeError("Could not read warmup frame")
    if normalize:
        warmup_frame = normalize_frame(warmup_frame)

    process_frame(warmup_frame, detector, show_rois=False)


    cv2.namedWindow("feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    game_started = False
    not_in_game_streak = 0
    

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break

        if normalize:
            frame = normalize_frame(frame)
        
        if not game_started and game_start(frame): # First frame of game_start
            game_started = True

            result = process_frame(frame, detector, show_rois=False)
            result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])

            now = time.monotonic()
            match_clock_filter.initialise(result["time_left_s"], now)


            enemy_card_tracker.start_match(
                result["time_left_s"],
                result["total_remaining_s"],
                now_s=time.monotonic(),
            )
            game_state = build_game_state(
                result,
                seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards),
                elixir_enemy_est=enemy_card_tracker.elixir_enemy_est,
                game_started=game_started
            )
            own_action_tracker.update(game_state, result["arena_px"])
        elif game_started:
            result = process_frame(frame, detector, show_rois=False)
            result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])

            now = time.monotonic()
            if match_clock_filter.initialised:
                filtered_time_left_s = match_clock_filter.update(result["time_left_s"], now)
                result["time_left_s"] = filtered_time_left_s
                result["total_remaining_s"] = total_remaining_seconds(filtered_time_left_s, result["overtime"])
            else:
                match_clock_filter.initialise(result["time_left_s"], now)

            game_state = build_game_state(
                result,
                seen_enemy_cards=list(enemy_card_tracker.confirmed_seen_cards),
                elixir_enemy_est=enemy_card_tracker.elixir_enemy_est,
                game_started=game_started
            )
            own_action_tracker.update(game_state, result["arena_px"])
            enemy_card_tracker.reconcile_own_actions(
                own_action_tracker.actions,
                arena_px=result["arena_px"],
            )
            enemy_card_tracker.update(
                result["total_remaining_s"],
                result["matches"],
                now_s=now,
                clock_boxes=result["clock_boxes"],
                own_actions=own_action_tracker.actions,
                arena_px=result["arena_px"],
            )
            if game_end_from_result(result):
                not_in_game_streak += 1
                if not_in_game_streak >= 20:
                    game_started = False
                    not_in_game_streak = 0
                    enemy_card_tracker = EnemyCardTracker()
                    own_action_tracker = OwnActionTracker()
                    match_clock_filter = MatchClockFilter()
                    tower_hp_filter = TowerHPFilter()
                    continue
            else:
                not_in_game_streak = 0
        else:
            print("not in game")
            continue
        

        print()
        print_frame_result(result, enemy_card_tracker, own_action_tracker)

        cv2.imshow("feed", result["rendered"])

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
