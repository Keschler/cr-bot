import os
import sys
import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

from cr_bot.domain.constants import YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
from cr_bot.paths import CACHE_DIR, KATACR_DATASET_ROOT, KATACR_ROOT, MODELS_DIR, REPO_ROOT
from cr_bot.vision.model_loader import yolo_device

MPLCONFIGDIR = CACHE_DIR / "matplotlib"
ULTRALYTICS_CONFIG_DIR = CACHE_DIR / "ultralytics"

if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

katacr_dataset_env = os.environ.get("KATACR_DATASET_PATH")
if not katacr_dataset_env or not Path(katacr_dataset_env).exists():
    os.environ["KATACR_DATASET_PATH"] = str(KATACR_DATASET_ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


def _patch_ultralytics_track_compat() -> None:
    import ultralytics.trackers.track as track

    if hasattr(track, "yaml_load"):
        return
    if hasattr(track, "YAML"):
        track.yaml_load = track.YAML.load
        return
    from ultralytics.utils import YAML

    track.YAML = YAML
    track.yaml_load = YAML.load


_patch_ultralytics_track_compat()


def _patch_ultralytics_plotting_compat() -> None:
    import ultralytics.utils.plotting as plotting

    if not hasattr(plotting, "contextlib"):
        plotting.contextlib = contextlib


_patch_ultralytics_plotting_compat()

from katacr.constants.label_list import idx2unit, unit2idx
from katacr.yolov8.custom_result import CRResults
from katacr.yolov8.custom_trackers import cr_on_predict_postprocess_end, cr_on_predict_start
from katacr.yolov8.train import YOLO_CR


DEFAULT_DETECTOR_WEIGHTS = [
    MODELS_DIR / "detector1_v0.7.13.pt",
    MODELS_DIR / "detector2_v0.7.13.pt",
]


class AppDetector:
    """Inference-only detector for the app runtime."""

    def __init__(
        self,
        path_detectors,
        show_conf=True,
        conf=0.7,
        iou_thre=0.6,
        tracker="bytetrack",
    ):
        self.models = [YOLO_CR(str(path)) for path in path_detectors]
        self.show_conf = show_conf
        self.conf = conf
        self.iou_thre = iou_thre
        self.device = yolo_device()
        self.tracker = None
        if tracker == "bytetrack":
            self.conf = 0.1
            self.tracker_cfg_path = str(KATACR_ROOT / "katacr/yolov8/bytetrack.yaml")
            cr_on_predict_start(self, persist=True)

    def infer(self, frame, pil=False):
        if pil:
            frame = frame[..., ::-1]

        results = [
            model.predict(
                frame,
                verbose=False,
                conf=self.conf,
                device=self.device,
                imgsz=896,
            )[0]
            for model in self.models
        ]

        preds = []
        for result in results:
            boxes = result.orig_boxes.clone()
            for index in range(len(boxes)):
                boxes[index, 5] = unit2idx[result.names[int(boxes[index, 5])]]
                preds.append(boxes[index])

        if not preds:
            preds = torch.zeros((0, 7))
        else:
            preds = torch.cat(preds, 0).reshape(-1, 7)
            keep = torchvision.ops.nms(preds[:, :4], preds[:, 4], iou_threshold=self.iou_thre)
            preds = preds[keep]

        self.result = CRResults(frame, path="", names=idx2unit, boxes=preds)
        if self.tracker is not None:
            cr_on_predict_postprocess_end(self, persist=True)

        data = self.result.get_data()
        self.result.boxes.data = data[
            ~(((data[:, 0] > 390) & (data[:, 3] < 120)) | ((data[:, 2] < 280) & (data[:, 3] < 80)))
        ]
        return self.result






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
    return AppDetector(
            DEFAULT_DETECTOR_WEIGHTS,
            show_conf=True,
            conf=YOLO_CONF_THRESHOLD,
            iou_thre=YOLO_IOU_THRESHOLD,
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
    not_troops = ["bar-level", "tower-bar", "queen-tower", "emote", "evolution-symbol", "elixir", "king-tower-bar", "clock", "king-tower", "dagger-duchess-tower-bar"]
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

def extract_clock_boxes(boxes):
      clocks = []
      _, _, idx2unit = load_yolo_runtime()

      for box in boxes:
          x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(box)
          if idx2unit[int(cls)] != "clock":
              continue

          clocks.append({
              "team": "enemy" if int(team) == 1 else "ally",
              "track_id": None if track_id is None else int(track_id),
              "confidence": conf,
              "x1": x1,
              "y1": y1,
              "x2": x2,
              "y2": y2,
              "center_x": (x1 + x2) / 2,
              "center_y": (y1 + y2) / 2,
          })

      return clocks


def extract_emote_boxes(boxes):
      emotes = []
      _, _, idx2unit = load_yolo_runtime()

      for box in boxes:
          x1, y1, x2, y2, track_id, conf, cls, team = parse_box_row(box)
          if idx2unit[int(cls)] != "emote":
              continue

          emotes.append({
              "team": "enemy" if int(team) == 1 else "ally",
              "confidence": conf,
              "x1": x1,
              "y1": y1,
              "x2": x2,
              "y2": y2,
              "center_x": (x1 + x2) / 2,
              "center_y": (y1 + y2) / 2,
          })

      return emotes
