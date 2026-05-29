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
        "--no-normalize",
        action="store_true",
        help="Process frames at their captured size instead of normalizing to 1080x2400.",
    )
    parser.add_argument(
        "--yolo-detections",
        action="store_true",
        help="Use YOLO tower health bar detections instead of fixed ROIS",
    )
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.video_duration is not None and args.video_duration <= 0:
        parser.error("--video-duration must be greater than 0")

    debug = args.debug or args.debug_frame is not None
    from cr_bot.app.main import main as run_capture

    run_capture(
        debug,
        video=args.video,
        frame_stride=args.frame_stride,
        video_duration_s=args.video_duration,
        normalize=not args.no_normalize,
        debug_frame_path=args.debug_frame,
        yolo_detections=args.yolo_detections,
    )


if __name__ == "__main__":
    main()
