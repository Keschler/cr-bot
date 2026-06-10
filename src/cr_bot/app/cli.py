import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run once against --debug-frame instead of live video capture.",
    )
    parser.add_argument(
        "--debug-frame",
        help="Frame image path to process in debug mode.",
    )
    parser.add_argument(
        "--video",
        help="Video file path to process instead of live video capture.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Analyze every Nth video frame when --video is used.",
    )
    parser.add_argument(
        "--video-duration",
        type=float,
        help="Stop analyzing --video after this many seconds from the start.",
    )
    parser.add_argument(
        "--video-start-time",
        type=float,
        help="Start analyzing --video at this absolute video timestamp in seconds.",
    )
    parser.add_argument(
        "--video-end-time",
        type=float,
        help="Stop analyzing --video at this absolute video timestamp in seconds.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Process frames at their captured size instead of normalizing to 1080x2400.",
    )
    parser.add_argument(
        "--yolo-detections",
        action="store_true",
        help="Use YOLO tower health bar detections instead of fixed ROIS",
    )
    parser.add_argument(
        "--alternative-rois",
        action="store_true",
        help="Use the alternative bottom-HUD ROI profile for shifted video layouts.",
    )
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.video_duration is not None and args.video_duration <= 0:
        parser.error("--video-duration must be greater than 0")
    if args.video_start_time is not None and args.video_start_time < 0:
        parser.error("--video-start-time must be non-negative")
    if args.video_end_time is not None and args.video_end_time <= 0:
        parser.error("--video-end-time must be greater than 0")
    if (
        args.video_start_time is not None
        and args.video_end_time is not None
        and args.video_end_time <= args.video_start_time
    ):
        parser.error("--video-end-time must be greater than --video-start-time")
    debug = args.debug or args.debug_frame is not None
    if args.alternative_rois:
        from cr_bot.domain.video_constants import activate_alternative_video_rois

        activate_alternative_video_rois()

    from cr_bot.app.main import main as run_capture

    run_capture(
        debug,
        video=args.video,
        frame_stride=args.frame_stride,
        video_duration_s=args.video_duration,
        video_start_time_s=args.video_start_time,
        video_end_time_s=args.video_end_time,
        normalize=not args.no_normalize,
        debug_frame_path=args.debug_frame,
        yolo_detections=args.yolo_detections,
    )


if __name__ == "__main__":
    main()
