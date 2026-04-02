from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
KATACR_ROOT = ROOT / "vendor/external/KataCR"
SEED_ROOT = ROOT / "data/seed_dataset"
ULTRALYTICS_CONFIG_DIR = ROOT / ".cache/ultralytics"

os.environ.setdefault("KATACR_DATASET_PATH", str(SEED_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(KATACR_ROOT))

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

from katacr.yolov8.train import YOLO_CR  # noqa: E402
from katacr.constants.label_list import idx2unit, unit2idx  # noqa: E402


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VID_SUFFIXES = {".avi", ".gif", ".m4v", ".mkv", ".mp4", ".mpeg", ".mpg", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merged detector inference on an image or video.")
    parser.add_argument("--weights", nargs="+", required=True, help="One or more detector weight files.")
    parser.add_argument("--source", required=True, help="Path to an image or video source.")
    parser.add_argument("--video-interval", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument(
        "--export-labelme",
        type=Path,
        default=None,
        help="Export extracted frames and Labelme-style pre-annotation JSON files to this directory.",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=[],
        help="Class names to suppress from the final rendered detections.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def draw_boxes(frame: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
    rendered = frame.copy()
    rows = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes
    for row in rows:
        values = row.tolist() if hasattr(row, "tolist") else list(row)
        if len(values) == 7:
            x1, y1, x2, y2, conf, cls, bel = values
        elif len(values) == 8:
            x1, y1, x2, y2, _track_id, conf, cls, bel = values
        else:
            raise ValueError(f"Unexpected YOLO box format with {len(values)} values: {values}")
        label = idx2unit.get(int(cls), str(int(cls)))
        team = int(bel)
        color = (255, 64, 64) if team == 1 else (64, 160, 255)
        cv2.rectangle(rendered, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label}:{conf:.2f}:{team}"
        cv2.putText(
            rendered,
            text,
            (int(x1), max(18, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return rendered


class CombinedDetector:
    def __init__(self, weights: list[Path], conf: float, iou: float) -> None:
        self.models = [YOLO_CR(str(path)) for path in weights]
        self.conf = conf
        self.iou = iou

    def infer(self, frame: np.ndarray) -> torch.Tensor:
        results = [model.predict(frame, verbose=False, conf=self.conf, device=0, imgsz=896)[0] for model in self.models]
        preds = []
        for result in results:
            boxes = result.orig_boxes.clone()
            for index in range(len(boxes)):
                boxes[index, 5] = unit2idx[result.names[int(boxes[index, 5])]]
                preds.append(boxes[index])
        if not preds:
            return torch.zeros((0, 7))
        merged = torch.cat(preds, 0).reshape(-1, 7)
        keep = torchvision.ops.nms(merged[:, :4], merged[:, 4], iou_threshold=self.iou)
        return merged[keep]


def filter_excluded_classes(boxes: torch.Tensor, excluded_names: set[str]) -> torch.Tensor:
    if not excluded_names or boxes.numel() == 0:
        return boxes
    keep_rows = [
        index for index, row in enumerate(boxes)
        if idx2unit.get(int(row[5])) not in excluded_names
    ]
    if not keep_rows:
        return torch.zeros((0, boxes.shape[1]), dtype=boxes.dtype, device=boxes.device)
    return boxes[keep_rows]


def build_labelme_annotation(
    image_name: str,
    frame: np.ndarray,
    boxes: torch.Tensor,
) -> dict:
    height, width = frame.shape[:2]
    shapes = []
    for row in boxes.cpu().numpy():
        x1, y1, x2, y2, _conf, cls, bel = row[:7]
        label = f"{idx2unit.get(int(cls), str(int(cls)))}{int(bel)}"
        shapes.append(
            {
                "label": label,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
        )
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"Failed to encode frame {image_name} for Labelme export.")
    return {
        "version": "5.11.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": base64.b64encode(encoded.tobytes()).decode("utf-8"),
        "imageHeight": height,
        "imageWidth": width,
    }


def iter_source(path: Path, video_interval: int):
    suffix = path.suffix.lower()
    if suffix in IMG_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        yield "image", path, image, None
        return
    if suffix not in VID_SUFFIXES:
        raise ValueError(f"Unsupported source: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    frame_index = 0
    while True:
        for _ in range(max(1, video_interval)):
            ok = cap.grab()
            if not ok:
                cap.release()
                return
        ok, frame = cap.retrieve()
        if not ok:
            cap.release()
            return
        frame_index += 1
        yield "video", path, frame, cap


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    weights = [Path(path).resolve() for path in args.weights]
    detector = CombinedDetector(weights, conf=args.conf, iou=args.iou)
    excluded_names = set(args.exclude_classes)
    export_labelme_dir = args.export_labelme.resolve() if args.export_labelme is not None else None

    save_dir = ROOT / "runs" / "inference" / time.strftime("%Y%m%d_%H%M%S")
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)
    if export_labelme_dir is not None:
        export_labelme_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    writer_path = None
    export_index = 0
    for mode, path, frame, cap in iter_source(source, args.video_interval):
        merged = detector.infer(frame)
        merged = filter_excluded_classes(merged, excluded_names)
        rendered = draw_boxes(frame, merged)
        if export_labelme_dir is not None:
            image_name = f"{export_index:05d}.jpg"
            image_path = export_labelme_dir / image_name
            json_path = export_labelme_dir / f"{export_index:05d}.json"
            cv2.imwrite(str(image_path), frame)
            annotation = build_labelme_annotation(image_name, frame, merged)
            json_path.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
            export_index += 1
        if args.show:
            cv2.imshow("seed_inference", rendered)
            cv2.waitKey(1)
        if args.no_save:
            continue
        if mode == "image":
            output_path = save_dir / path.name
            cv2.imwrite(str(output_path), rendered)
        else:
            output_path = save_dir / f"{path.stem}.mp4"
            if writer_path != output_path:
                if writer is not None:
                    writer.release()
                fps = cap.get(cv2.CAP_PROP_FPS) / max(1, args.video_interval)
                height, width = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                writer_path = output_path
            writer.write(rendered)
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
