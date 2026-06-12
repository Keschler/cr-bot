import os
from dataclasses import replace
from pathlib import Path

import cv2
import time


from cr_bot.app.match_session import MatchSession
from cr_bot.app.pipeline import normalize_frame, process_frame
from cr_bot.debug.output import (
    print_debug_frame_result,
    print_frame_result,
    render_match_debug,
    render_timer_debug,
    render_tower_hp_debug,
)
from cr_bot.paths import APP_ROOT
from cr_bot.replay import ReplayCacheReader, ReplayCacheWriter
from cr_bot.vision.yolo_runtime import build_detector


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

def main(
    debug: bool,
    video: str | None = None,
    frame_stride: int = 1,
    video_sample_interval_s: float | None = None,
    video_duration_s: float | None = None,
    video_start_time_s: float | None = None,
    video_end_time_s: float | None = None,
    normalize: bool = True,
    debug_frame_path: str | None = None,
    yolo_detections: bool = False,
    write_replay_cache: str | None = None,
    replay_cache: str | None = None,
):
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if video_sample_interval_s is not None and video_sample_interval_s <= 0:
        raise ValueError("video_sample_interval_s must be greater than 0")
    if video_sample_interval_s is not None and frame_stride != 1:
        raise ValueError("video_sample_interval_s cannot be combined with frame_stride")
    if video_duration_s is not None and video_duration_s <= 0:
        raise ValueError("video_duration_s must be greater than 0")
    if video_start_time_s is not None and video_start_time_s < 0:
        raise ValueError("video_start_time_s must be non-negative")
    if video_end_time_s is not None and video_end_time_s <= 0:
        raise ValueError("video_end_time_s must be greater than 0")
    if (
        video_start_time_s is not None
        and video_end_time_s is not None
        and video_end_time_s <= video_start_time_s
    ):
        raise ValueError("video_end_time_s must be greater than video_start_time_s")

    session = MatchSession()

    if replay_cache:
        for record in ReplayCacheReader(replay_cache):
            frame = record.decode_frame()
            analysis = record.analysis
            if analysis.yolo_boxes is None:
                analysis = replace(analysis, yolo_boxes=[])
            step = session.process(
                analysis,
                frame=frame,
                now_s=record.video_time_s,
            )
            if not step.should_emit:
                print(f"frame {record.frame_idx}: not in game")
                continue
            print(f"frame {record.frame_idx} video_time={record.video_time_s:.2f}s")
            print_frame_result(
                step.analysis,
                session.enemy_card_tracker,
                session.own_action_tracker,
            )
        return

    detector = build_detector()
    print("detector sucessfully built!")

    if debug:
        if not debug_frame_path:
            raise RuntimeError("Pass a debug frame path when running main(debug=True).")

        frame = cv2.imread(debug_frame_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read debug frame: {debug_frame_path}")
        if normalize:
            frame = normalize_frame(frame)

        analysis = process_frame(frame, detector, show_rois=False, yolo_tower_hp_detections=yolo_detections)
        debug_output_dir = APP_ROOT / "outputs" / "video" / "capture"
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        tower_debug_path = debug_output_dir / "tower_hp_debug.png"
        cv2.imwrite(str(tower_debug_path), render_tower_hp_debug(analysis.tower_hp_debug_steps))
        print(f"tower hp debug image: {tower_debug_path}")

        step = session.process(analysis, frame=frame, now_s=time.monotonic())
        analysis = step.analysis
        has_display = os.environ.get("SHOW_DEBUG_WINDOW") == "1"
        if has_display:
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("debug", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("debug", analysis.rendered)
            cv2.namedWindow("tower_hp_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("tower_hp_debug", render_tower_hp_debug(analysis.tower_hp_debug_steps))
            cv2.namedWindow("timer_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("timer_debug", render_timer_debug(analysis.timer_debug_steps))
            cv2.namedWindow("match_debug", cv2.WINDOW_NORMAL)
            cv2.imshow("match_debug", render_match_debug(frame, analysis.matches))

        print_debug_frame_result(
            analysis,
            session.enemy_card_tracker,
            session.own_action_tracker,
        )

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
        if video_start_time_s is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, video_start_time_s * 1000.0)

        has_display = os.environ.get("SHOW_DEBUG_WINDOW") == "1"
        if has_display:
            cv2.namedWindow("feed", cv2.WINDOW_NORMAL)

        cache_writer = (
            ReplayCacheWriter(write_replay_cache)
            if write_replay_cache is not None
            else None
        )
        if cache_writer is not None:
            cache_writer.__enter__()
        try:
            frame_idx = 0
            next_sample_time_s = video_start_time_s or 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if video_start_time_s is not None and video_time_s < video_start_time_s:
                    continue
                if video_duration_s is not None and video_time_s > video_duration_s:
                    break
                if video_end_time_s is not None and video_time_s > video_end_time_s:
                    break

                if (
                    video_sample_interval_s is not None
                    and video_time_s + 1e-9 < next_sample_time_s
                ):
                    continue
                if (frame_idx - 1) % frame_stride != 0:
                    continue
                if video_sample_interval_s is not None:
                    intervals_elapsed = max(
                        1,
                        int(
                            (video_time_s - next_sample_time_s)
                            // video_sample_interval_s
                        )
                        + 1,
                    )
                    next_sample_time_s += intervals_elapsed * video_sample_interval_s

                if normalize:
                    frame = normalize_frame(frame)

                analysis = process_frame(
                    frame,
                    detector,
                    show_rois=False,
                    yolo_tower_hp_detections=yolo_detections,
                )
                if cache_writer is not None:
                    cache_writer.write(
                        frame_idx=frame_idx,
                        video_time_s=video_time_s,
                        analysis=analysis,
                        frame=frame,
                    )
                step = session.process(analysis, frame=frame, now_s=video_time_s)
                if not step.should_emit:
                    print(f"frame {frame_idx}: not in game")
                    continue

                analysis = step.analysis
                print(f"frame {frame_idx} video_time={video_time_s:.2f}s")
                print_frame_result(
                    analysis,
                    session.enemy_card_tracker,
                    session.own_action_tracker,
                )

                if has_display:
                    cv2.imshow("feed", analysis.rendered)
                    if cv2.waitKey(1) == 27:
                        break
        except BaseException as exc:
            if cache_writer is not None:
                cache_writer.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            if cache_writer is not None:
                cache_writer.__exit__(None, None, None)

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

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break

        if normalize:
            frame = normalize_frame(frame)
        
        analysis = process_frame(frame, detector, show_rois=False)
        step = session.process(analysis, frame=frame, now_s=time.monotonic())
        if not step.should_emit:
            print("not in game")
            continue
        analysis = step.analysis

        print()
        print_frame_result(
            analysis,
            session.enemy_card_tracker,
            session.own_action_tracker,
        )

        cv2.imshow("feed", analysis.rendered)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
