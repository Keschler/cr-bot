from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_ERRORS_CSV = ROOT / "outputs/tower_hp_ocr_eval_errors.csv"
DEFAULT_OUTPUT = ROOT / "outputs/tower_hp_ocr_eval_errors.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tower HP OCR error crops as a contact sheet.")
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--crop-width", type=int, default=260)
    parser.add_argument("--crop-height", type=int, default=90)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def load_errors(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    if image is None or image.size == 0:
        return tile
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return tile


def draw_text_lines(tile: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for line in lines:
        cv2.putText(tile, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        y += 18


def render_error(error: dict[str, str], crop_width: int, crop_height: int) -> np.ndarray:
    text_height = 92
    tile = np.zeros((crop_height + text_height, crop_width, 3), dtype=np.uint8)
    image = cv2.imread(str(resolve_path(error["image_path"])), cv2.IMREAD_COLOR)
    tile[:crop_height] = fit_image(image, crop_width, crop_height)
    expected = error["expected_label"] if error["expected_readable"] == "true" else "unreadable"
    predicted = error["predicted_label"] or "unreadable"
    model_ocr = error.get("model_ocr") or error.get("old_ocr") or "none"
    lines = [
        f"expected: {expected}",
        f"pred:     {predicted}   model: {model_ocr}",
        f"{error['tower_name']} frame={error['frame_index']} t={error['video_time_s']}",
        Path(error["image_path"]).name,
    ]
    draw_text_lines(tile, lines, 8, crop_height + 18)
    color = (0, 0, 255)
    cv2.rectangle(tile, (0, 0), (crop_width - 1, crop_height + text_height - 1), color, 1)
    return tile


def render_sheet(errors: list[dict[str, str]], columns: int, crop_width: int, crop_height: int) -> np.ndarray:
    if not errors:
        sheet = np.zeros((120, 600, 3), dtype=np.uint8)
        cv2.putText(sheet, "no errors", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return sheet
    tiles = [render_error(error, crop_width, crop_height) for error in errors]
    rows = []
    tile_h, tile_w = tiles[0].shape[:2]
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start:start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def main() -> None:
    args = parse_args()
    errors = load_errors(args.errors_csv)
    sheet = render_sheet(errors, args.columns, args.crop_width, args.crop_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"rendered {len(errors)} errors: {args.output}")


if __name__ == "__main__":
    main()
