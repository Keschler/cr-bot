import os
from pathlib import Path

import cv2
import time


from cr_bot.app.pipeline import normalize_frame, process_frame
from cr_bot.app.state_builder import build_game_state
from cr_bot.debug.output import (
    print_debug_frame_result,
    print_frame_result,
    render_match_debug,
    render_timer_debug,
    render_tower_hp_debug,
)
from cr_bot.paths import APP_ROOT
from cr_bot.vision.timer import total_remaining_seconds
from cr_bot.vision.match_state import game_start, game_end_from_result
from cr_bot.vision.yolo_runtime import build_detector
from cr_bot.trackers.enemy_cards import EnemyCardTracker 
from cr_bot.trackers.match_clock import MatchClockFilter
from cr_bot.trackers.own_actions import OwnActionTracker
from cr_bot.trackers.tower_hp_filter import TowerHPFilter
from cr_bot.trackers.hand_state_filter import HandStateFilter


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


def has_visible_match_timer(result) -> bool:
    time_left_s = result.get("time_left_s")
    return time_left_s is not None and float(time_left_s) > 0.0


def main(
    debug: bool,
    video: str | None = None,
    frame_stride: int = 1,
    video_duration_s: float | None = None,
    normalize: bool = True,
    debug_frame_path: str | None = None,
    yolo_detections: bool = False,
):
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if video_duration_s is not None and video_duration_s <= 0:
        raise ValueError("video_duration_s must be greater than 0")

    detector = build_detector()
    print("detector sucessfully built!")
    enemy_card_tracker = EnemyCardTracker()
    own_action_tracker = OwnActionTracker()
    match_clock_filter = MatchClockFilter()
    tower_hp_filter = TowerHPFilter()
    hand_state_filter = HandStateFilter()

    if debug:
        if not debug_frame_path:
            raise RuntimeError("Pass a debug frame path when running main(debug=True).")

        frame = cv2.imread(debug_frame_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read debug frame: {debug_frame_path}")
        if normalize:
            frame = normalize_frame(frame)

        result = process_frame(frame, detector, show_rois=False, yolo_tower_hp_detections=yolo_detections)
        result["state"] = hand_state_filter.update(result["state"])
        debug_output_dir = APP_ROOT / "outputs" / "video" / "capture"
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        tower_debug_path = debug_output_dir / "tower_hp_debug.png"
        cv2.imwrite(str(tower_debug_path), render_tower_hp_debug(result["tower_hp_debug_steps"]))
        print(f"tower hp debug image: {tower_debug_path}")

        enemy_card_tracker.start_match(
          result["time_left_s"],
          result["total_remaining_s"],
          now_s=time.monotonic(),
      )

        game_state = build_game_state(result)
        own_action_tracker.update(
            game_state,
            result["arena_px"],
            frame=frame,
            clock_boxes=result["clock_boxes"],
            own_actions_blocked=len(result["emote_boxes"]) >= 2,
            elixir_change=result["elixir_change"],
        )
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

        print_debug_frame_result(result, enemy_card_tracker, own_action_tracker)

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
            video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if video_duration_s is not None and video_time_s > video_duration_s:
                break

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
            result["state"] = hand_state_filter.update(result["state"])

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
                own_action_tracker.update(
                    game_state,
                    result["arena_px"],
                    frame=frame,
                    clock_boxes=result["clock_boxes"],
                    own_actions_blocked=len(result["emote_boxes"]) >= 2,
                    elixir_change=result["elixir_change"],
                    video_time_s=video_time_s,
                )
                enemy_card_tracker.reconcile_own_actions(
                    own_action_tracker.actions,
                    arena_px=result["arena_px"],
                )
            elif game_started:
                if match_clock_filter.initialised:
                    filtered_time_left_s = match_clock_filter.update(result["time_left_s"], video_time_s, result["overtime"])
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
                own_action_tracker.update(
                    game_state,
                    result["arena_px"],
                    frame=frame,
                    clock_boxes=result["clock_boxes"],
                    own_actions_blocked=len(result["emote_boxes"]) >= 2,
                    elixir_change=result["elixir_change"],
                    video_time_s=video_time_s,
                )
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
                        hand_state_filter = HandStateFilter()
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
            result["state"] = hand_state_filter.update(result["state"])
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
            own_action_tracker.update(
                game_state,
                result["arena_px"],
                frame=frame,
                clock_boxes=result["clock_boxes"],
                own_actions_blocked=len(result["emote_boxes"]) >= 2,
                elixir_change=result["elixir_change"],
            )
        elif game_started:
            result = process_frame(frame, detector, show_rois=False)
            result["state"] = hand_state_filter.update(result["state"])
            result["towers_hp"] = tower_hp_filter.update(result["towers_hp"])

            now = time.monotonic()
            if match_clock_filter.initialised:
                filtered_time_left_s = match_clock_filter.update(result["time_left_s"], now, result["overtime"])
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
            own_action_tracker.update(
                game_state,
                result["arena_px"],
                frame=frame,
                clock_boxes=result["clock_boxes"],
                own_actions_blocked=len(result["emote_boxes"]) >= 2,
                elixir_change=result["elixir_change"],
            )
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
                    hand_state_filter = HandStateFilter()
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
