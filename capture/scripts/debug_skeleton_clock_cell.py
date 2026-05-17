from __future__ import annotations

from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.action_space import ACTION_GRID
from main import normalize_frame, process_frame
from state_builder import build_game_state
from vision.yolo_runtime import build_detector


VIDEO_PATH = ROOT / "assets/pictures/10_fps_gameplay.mp4"
OUTPUT_DIR = ROOT / "debug_output/skeleton_clock_cell"
FRAME_NUMBERS = range(88, 93)


def draw_grid(img, arena_px):
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))

    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (45, 45, 45), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (45, 45, 45), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = (int(round(box[key])) for key in ("x1", "y1", "x2", "y2"))
    cx = int(round(box["center_x"]))
    cy = int(round(box["center_y"]))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA)


def skeleton_clock_score(skeleton, clock):
    horizontal_gap = abs(clock["center_x"] - skeleton.center_x)
    vertical_gap = clock["center_y"] - skeleton.center_y
    if horizontal_gap > 100 or not (-40 <= vertical_gap <= 220):
        return None
    return horizontal_gap + abs(vertical_gap - 80) * 0.5


def render(frame, result, frame_idx, video_time_s):
    overlay = frame.copy()
    draw_grid(overlay, result["arena_px"])
    game_state = build_game_state(result)
    skeletons = [
        match.troop
        for match in game_state.own_units
        if match.troop.class_name == "skeleton"
    ]

    cv2.rectangle(overlay, (10, 8), (980, 52), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"frame={frame_idx} video_time={video_time_s:.2f}s skeleton clock cell debug",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for idx, clock in enumerate(result["clock_boxes"]):
        cell = ACTION_GRID.pixel_to_cell(clock["center_x"], clock["center_y"], result["arena_px"])
        color = (255, 255, 0) if clock["team"] == "ally" else (0, 165, 255)
        label = f"clock{idx}:{clock['team']} conf={clock['confidence']:.2f} cell={cell}"
        draw_box(overlay, clock, color, label)

    for idx, skeleton in enumerate(skeletons):
        box = {
            "x1": skeleton.x1,
            "y1": skeleton.y1,
            "x2": skeleton.x2,
            "y2": skeleton.y2,
            "center_x": skeleton.center_x,
            "center_y": skeleton.center_y,
        }
        cell = ACTION_GRID.pixel_to_cell(skeleton.center_x, skeleton.center_y, result["arena_px"])
        draw_box(
            overlay,
            box,
            (0, 255, 0),
            f"skel{idx}:id={skeleton.track_id} conf={skeleton.confidence:.2f} cell={cell}",
        )

        best = None
        for clock_idx, clock in enumerate(result["clock_boxes"]):
            if clock["team"] != "ally" or clock["confidence"] < 0.5:
                continue
            score = skeleton_clock_score(skeleton, clock)
            if score is None:
                continue
            if best is None or score < best[0]:
                best = (score, clock_idx, clock)

        if best is not None:
            score, clock_idx, clock = best
            start = (int(round(skeleton.center_x)), int(round(skeleton.center_y)))
            end = (int(round(clock["center_x"])), int(round(clock["center_y"])))
            clock_cell = ACTION_GRID.pixel_to_cell(clock["center_x"], clock["center_y"], result["arena_px"])
            cv2.line(overlay, start, end, (0, 0, 255), 2)
            mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(
                overlay,
                f"best clock{clock_idx} score={score:.1f} clock_cell={clock_cell}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return overlay


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = build_detector()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO_PATH}")

    summary = []
    for frame_idx in FRAME_NUMBERS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = normalize_frame(frame)
        result = process_frame(frame, detector, show_rois=False)
        video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        overlay = render(frame, result, frame_idx, video_time_s)
        out = OUTPUT_DIR / f"frame_{frame_idx:04d}_video_time_{video_time_s:05.2f}.jpg"
        cv2.imwrite(str(out), overlay)

        for clock_idx, clock in enumerate(result["clock_boxes"]):
            cell = ACTION_GRID.pixel_to_cell(clock["center_x"], clock["center_y"], result["arena_px"])
            summary.append(
                f"frame={frame_idx} clock{clock_idx} team={clock['team']} "
                f"conf={clock['confidence']:.3f} center=({clock['center_x']:.1f},{clock['center_y']:.1f}) cell={cell}"
            )

    cap.release()
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote skeleton clock debug images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
