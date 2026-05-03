import argparse

from main import main


if __name__ == "__main__":
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
        "--no-normalize",
        action="store_true",
        help="Process frames at their captured size instead of normalizing to 1080x2400.",
    )
    parser.add_argument("--yolo-detections",
                        action="store_true",
                        help="Use YOLO tower health bar detections instead of fixed ROIS")
    args = parser.parse_args()

    debug = args.debug or args.debug_frame is not None
    main(debug, normalize=not args.no_normalize, debug_frame_path=args.debug_frame, yolo_detections=args.yolo_detections)
