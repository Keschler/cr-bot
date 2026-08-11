from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.app.pipeline import normalize_frame
from cr_bot.eval.visualize_cells import arena_px_for
from cr_bot.features.action_space import ACTION_GRID


WINDOW_NAME = "cell-selector"


class CellSelector:
    def __init__(self, image, arena_px: tuple[int, int, int, int]) -> None:
        self.image = image
        self.arena_px = arena_px
        self.hover_xy: tuple[int, int] | None = None
        self.hover_cell: tuple[int, int] | None = None
        self.selected_xy: tuple[int, int] | None = None
        self.selected_cell: tuple[int, int] | None = None

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        self.hover_xy = (x, y)
        self.hover_cell = ACTION_GRID.pixel_to_cell(x, y, self.arena_px)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_xy = (x, y)
            self.selected_cell = self.hover_cell
            print(f"selected pixel=({x}, {y}) cell={self.selected_cell}")

    def render(self):
        canvas = self.image.copy()
        draw_grid(canvas, self.arena_px)
        self._draw_cursor(canvas)
        self._draw_selection(canvas)
        self._draw_status(canvas)
        return canvas

    def _draw_cursor(self, canvas) -> None:
        if self.hover_xy is None:
            return
        x, y = self.hover_xy
        color = (0, 220, 255) if self.hover_cell is not None else (0, 0, 255)
        cv2.drawMarker(
            canvas,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=1,
        )

    def _draw_selection(self, canvas) -> None:
        if self.selected_xy is None:
            return
        x, y = self.selected_xy
        cv2.circle(canvas, (x, y), 8, (0, 255, 0), 2)
        if self.selected_cell is None:
            return
        text = f"selected cell={self.selected_cell}"
        cv2.putText(
            canvas,
            text,
            (x + 10, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_status(self, canvas) -> None:
        lines = [
            "hover to inspect, left click to select, c to clear, q or esc to quit",
        ]
        if self.hover_xy is None:
            lines.append("hover: -")
        else:
            lines.append(
                f"hover pixel={self.hover_xy} cell={self.hover_cell}"
            )
        if self.selected_xy is None:
            lines.append("selected: -")
        else:
            lines.append(
                f"selected pixel={self.selected_xy} cell={self.selected_cell}"
            )

        for idx, text in enumerate(lines):
            y = 24 + idx * 24
            cv2.putText(
                canvas,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a frame and interactively inspect action-grid cells."
    )
    parser.add_argument("--image", type=Path, help="Still image to inspect.")
    parser.add_argument("--video", type=Path, help="Video to inspect.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index for --video.")
    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        raise SystemExit("pass exactly one of --image or --video")

    frame = read_image(args.image) if args.image else read_frame(args.video, args.frame_index)
    image = normalize_frame(frame)
    arena_px = arena_px_for(image)

    selector = CellSelector(image=image, arena_px=arena_px)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, selector.on_mouse)

    print(f"arena_px={arena_px}")
    ax, ay, aw, ah = arena_px
    grid_x = ax + ACTION_GRID.x0 * aw
    grid_y = ay + ACTION_GRID.y0 * ah
    grid_width = ACTION_GRID.width * aw
    grid_height = ACTION_GRID.height * ah
    print(f"grid_area_px=({grid_x:.1f}, {grid_y:.1f}, {grid_width:.1f}, {grid_height:.1f})")
    print(
        "block_size_px="
        f"({grid_width / ACTION_GRID.cols:.8f}, {grid_height / ACTION_GRID.rows:.8f})"
    )
    while True:
        cv2.imshow(WINDOW_NAME, selector.render())
        key = cv2.waitKey(16) & 0xFF
        if key in {27, ord("q")}:
            break
        if key == ord("c"):
            selector.selected_xy = None
            selector.selected_cell = None

    cv2.destroyAllWindows()


def read_frame(video: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"could not read frame {frame_index} from {video}")
        return frame
    finally:
        cap.release()


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def draw_grid(img, arena_px: tuple[int, int, int, int]) -> None:
    ax, ay, aw, ah = arena_px
    x0 = int(round(ax + ACTION_GRID.x0 * aw))
    y0 = int(round(ay + ACTION_GRID.y0 * ah))
    x1 = int(round(ax + ACTION_GRID.x1 * aw))
    y1 = int(round(ay + ACTION_GRID.y1 * ah))

    for col in range(ACTION_GRID.cols + 1):
        x = int(round(x0 + col / ACTION_GRID.cols * (x1 - x0)))
        cv2.line(img, (x, y0), (x, y1), (40, 40, 40), 1)
    for row in range(ACTION_GRID.rows + 1):
        y = int(round(y0 + row / ACTION_GRID.rows * (y1 - y0)))
        cv2.line(img, (x0, y), (x1, y), (40, 40, 40), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)


if __name__ == "__main__":
    main()
