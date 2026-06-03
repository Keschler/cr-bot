from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
import sys
from time import perf_counter

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.app import pipeline  # noqa: E402
from cr_bot.vision.yolo_runtime import build_detector  # noqa: E402


DEFAULT_VIDEOS = [
    ROOT / "dataset_generation/data/video_clips/downloaded_videos/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4",
    ROOT / "dataset_generation/data/video_clips/10_fps_2.6HogCycle.mp4",
]


@dataclass
class Timing:
    seconds: float = 0.0
    calls: int = 0


class Timings:
    def __init__(self) -> None:
        self.values: dict[str, Timing] = defaultdict(Timing)

    def add(self, name: str, seconds: float) -> None:
        timing = self.values[name]
        timing.seconds += seconds
        timing.calls += 1

    def clear(self) -> None:
        self.values.clear()


def timed_call(timings: Timings, name: str, fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        started = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timings.add(name, perf_counter() - started)

    return wrapped


def instrument_pipeline(timings: Timings) -> None:
    stage_functions = {
        "aspect_ratio": "ratio2name",
        "arena_crop": "process_part",
        "remap_boxes": "remap_boxes_to_frame",
        "extract_clock_boxes": "extract_clock_boxes",
        "extract_emote_boxes": "extract_emote_boxes",
        "extract_elixir": "extract_elixir",
        "detect_elixir_change": "detect_elixir_change",
        "extract_tower_hp": "extract_tower_hp",
        "extract_timer": "extract_time",
        "detect_overtime": "is_overtime",
        "extract_hand_state": "extract_hand_state",
        "convert_yolo": "convert_yolo",
        "estimate_health": "estimate_health",
        "match_troops_to_bars": "match_troops_to_bars",
        "build_typed_match": "match_from_dict",
    }
    for timing_name, attribute_name in stage_functions.items():
        setattr(
            pipeline,
            attribute_name,
            timed_call(timings, timing_name, getattr(pipeline, attribute_name)),
        )

    load_yolo_runtime = pipeline.load_yolo_runtime

    @wraps(load_yolo_runtime)
    def timed_load_yolo_runtime():
        started = perf_counter()
        try:
            combined_detector, draw_boxes, idx2unit = load_yolo_runtime()
        finally:
            timings.add("runtime_lookup", perf_counter() - started)
        return combined_detector, timed_call(timings, "render_yolo_boxes", draw_boxes), idx2unit

    pipeline.load_yolo_runtime = timed_load_yolo_runtime


def read_frame(cap: cv2.VideoCapture, timings: Timings):
    started = perf_counter()
    ok, frame = cap.read()
    timings.add("decode_frame", perf_counter() - started)
    return ok, frame


def normalize_frame(frame, timings: Timings):
    started = perf_counter()
    try:
        return pipeline.normalize_frame(frame)
    finally:
        timings.add("normalize_frame", perf_counter() - started)


def analyze_frame(frame, detector, timings: Timings) -> None:
    started = perf_counter()
    try:
        pipeline.process_frame(frame, detector, show_rois=False)
    finally:
        timings.add("analyze_frame", perf_counter() - started)


def wrap_detector(detector, timings: Timings) -> None:
    detector.infer = timed_call(timings, "yolo_inference", detector.infer)


def warm_up(video_path: Path, detector, timings: Timings) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open warmup video: {video_path}")
    try:
        started = perf_counter()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read warmup frame: {video_path}")
        frame = pipeline.normalize_frame(frame)
        pipeline.process_frame(frame, detector, show_rois=False)
        return perf_counter() - started
    finally:
        cap.release()
        timings.clear()


def profile_video(video_path: Path, *, detector, duration_s: float, timings: Timings) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    measured_wall_started = perf_counter()
    analyzed_frames = 0
    last_video_time_s = 0.0
    try:
        while True:
            ok, frame = read_frame(cap, timings)
            if not ok:
                break
            video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if video_time_s > duration_s:
                break

            frame = normalize_frame(frame, timings)
            analyze_frame(frame, detector, timings)
            analyzed_frames += 1
            last_video_time_s = video_time_s
    finally:
        cap.release()

    wall_seconds = perf_counter() - measured_wall_started
    if analyzed_frames == 0:
        raise RuntimeError(f"no frames analyzed from {video_path}")
    return {
        "path": video_path,
        "source_fps": source_fps,
        "source_frames": source_frames,
        "analyzed_frames": analyzed_frames,
        "analyzed_video_seconds": last_video_time_s,
        "wall_seconds": wall_seconds,
        "timings": dict(timings.values),
    }


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def print_report(result: dict) -> None:
    frames = result["analyzed_frames"]
    wall_seconds = result["wall_seconds"]
    video_seconds = result["analyzed_video_seconds"]
    timings = result["timings"]
    analyze_seconds = timings["analyze_frame"].seconds

    print()
    print(f"Video: {result['path']}")
    print(
        f"  source: fps={result['source_fps']:.3f} "
        f"frames={result['source_frames']} profiled_video={video_seconds:.3f}s"
    )
    print(
        f"  result: analyzed_frames={frames} wall={format_seconds(wall_seconds)} "
        f"wall_fps={frames / wall_seconds:.3f} "
        f"analysis_fps={frames / analyze_seconds:.3f} "
        f"realtime_factor={video_seconds / wall_seconds:.3f}x"
    )
    print("  stages:")
    accounted_seconds = 0.0
    for name, timing in sorted(timings.items(), key=lambda item: item[1].seconds, reverse=True):
        if name == "analyze_frame":
            continue
        if name not in {"decode_frame", "normalize_frame"}:
            accounted_seconds += timing.seconds
        percent = timing.seconds / analyze_seconds * 100.0
        milliseconds_per_frame = timing.seconds / frames * 1000.0
        print(
            f"    {name:<24} total={timing.seconds:8.3f}s "
            f"per_frame={milliseconds_per_frame:8.2f}ms "
            f"analysis={percent:6.2f}% calls={timing.calls}"
        )
    unaccounted_seconds = max(0.0, analyze_seconds - accounted_seconds)
    print(
        f"    {'pipeline_unaccounted':<24} total={unaccounted_seconds:8.3f}s "
        f"per_frame={unaccounted_seconds / frames * 1000.0:8.2f}ms "
        f"analysis={unaccounted_seconds / analyze_seconds * 100.0:6.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the real video-analysis pipeline and print per-stage timings."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        default=DEFAULT_VIDEOS,
        help="Videos to profile. Defaults to the 3400-ladder and 10 FPS hog-cycle clips.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Analyze this many seconds from the start of each video. Default: 10.",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than 0")

    videos = [path.resolve() for path in args.videos]
    for video in videos:
        if not video.exists():
            parser.error(f"missing video: {video}")

    timings = Timings()
    instrument_pipeline(timings)

    started = perf_counter()
    detector = build_detector()
    detector_build_seconds = perf_counter() - started
    wrap_detector(detector, timings)
    warmup_seconds = warm_up(videos[0], detector, timings)

    print(f"Detector build: {format_seconds(detector_build_seconds)}")
    print(f"Warmup frame:   {format_seconds(warmup_seconds)}")
    for video in videos:
        timings.clear()
        result = profile_video(video, detector=detector, duration_s=args.duration, timings=timings)
        print_report(result)


if __name__ == "__main__":
    main()
