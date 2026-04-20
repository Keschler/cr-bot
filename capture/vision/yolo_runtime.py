import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KATACR_ROOT = ROOT / "vendor/external/KataCR"
SEED_ROOT = ROOT / "data/seed_dataset"
MODELS_DIR = ROOT / "models"

if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

os.environ.setdefault("KATACR_DATASET_PATH", str(SEED_ROOT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from katacr.yolov8.combo_detect import ComboDetector


DEFAULT_DETECTOR_WEIGHTS = [
    MODELS_DIR / "detector1_v0.7.13.pt",
    MODELS_DIR / "detector2_v0.7.13.pt",
]






def load_yolo_runtime():
    try:
        from scripts.run_seed_inference import CombinedDetector, draw_boxes, idx2unit
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "YOLO runtime dependencies are missing. Activate the training/inference env "
            "(for example `.venv-train`) before running main.py."
        ) from exc
    return CombinedDetector, draw_boxes, idx2unit


def build_detector() -> Any:
    return ComboDetector(
            DEFAULT_DETECTOR_WEIGHTS,
            show_conf=True,
            conf=0.7,
            iou_thre=0.6,
            tracker='bytetrack'
            )


def parse_box_row(row):
    values = row.tolist() if hasattr(row, "tolist") else list(row)
    if len(values) == 7:
        x1, y1, x2, y2, conf, cls, team = values
        track_id = None
    elif len(values) == 8:
        x1, y1, x2, y2, track_id, conf, cls, team = values
    else:
        raise ValueError(f"Unexpected YOLO box format with {len(values)} values: {values}")
    return x1, y1, x2, y2, track_id, conf, cls, team


def summarize_detections(boxes) -> str:
    _, _, idx2unit = load_yolo_runtime()
    if len(boxes) == 0:
        return "none"

    counts: dict[str, int] = {}
    rows = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes
    for row in rows:
        _x1, _y1, _x2, _y2, _track_id, _conf, cls, team = parse_box_row(row)
        label = idx2unit.get(int(cls), str(int(cls)))
        team = "enemy" if int(team) == 1 else "ally"
        key = f"{label}:{team}"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(
        f"{label} x{count}" for label, count in sorted(counts.items())
    )

def remap_boxes_to_frame(boxes, arena_shape, crop_xywh):
      arena_h, arena_w = arena_shape[:2]
      crop_x, crop_y, crop_w, crop_h = crop_xywh
      scale_x = crop_w / arena_w
      scale_y = crop_h / arena_h

      remapped = boxes.clone() if hasattr(boxes, "clone") else boxes.copy()
      remapped[:, [0, 2]] = remapped[:, [0, 2]] * scale_x + crop_x
      remapped[:, [1, 3]] = remapped[:, [1, 3]] * scale_y + crop_y
      return remapped

def convert_yolo(boxes):
    troops = []
    bars = []
    not_troops = ["bar-level", "tower-bar", "queen-tower", "emote", "evolution-symbol", "elixir", "king-tower-bar", "clock"]
    _, _, idx2unit = load_yolo_runtime()
    for box in boxes:
        x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(box)
        track_id = int(track_id) if track_id is not None else None
        class_name = idx2unit[int(cls)]
        team_name = "enemy" if int(team) == 1 else "ally"
        if class_name == "bar":
            bars.append({
                "track_id": track_id,
                "class_name": class_name,
                "team": team_name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2, 
                "center_y": (y1 + y2) / 2
                })
        elif class_name not in not_troops:
            troops.append({
                "track_id": track_id,
                "class_name": class_name,
                "team": team_name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2, 
                "center_y": (y1 + y2) / 2
                })

    return troops, bars
