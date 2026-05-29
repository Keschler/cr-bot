from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KATACR_ROOT = ROOT / "vendor/external/KataCR"
SEED_ROOT = ROOT / "data/seed_dataset"
DEFAULT_CONFIG = ROOT / "configs/katacr_train_baseline.yaml"
ULTRALYTICS_CONFIG_DIR = ROOT / "outputs/cache/ultralytics"

os.environ.setdefault("KATACR_DATASET_PATH", str(SEED_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(KATACR_ROOT))

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

from ultralytics.cfg import get_cfg  # noqa: E402
from katacr.yolov8.train import YOLO_CR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KataCR baseline detector.")
    parser.add_argument("--detector", type=int, default=1, choices=[1, 2])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolov8n.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a last.pt checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume is not None:
        model = YOLO_CR(str(args.resume), task="detect")
        model.train(resume=str(args.resume), device=args.device)
        return

    model = YOLO_CR(args.model, task="detect")
    cfg = dict(get_cfg(str(args.config)))
    name = f"detector{args.detector}"
    cfg["name"] = f"{name}_{cfg['name']}"
    cfg["data"] = KATACR_ROOT / f"katacr/yolov8/{name}/data.yaml"
    cfg["project"] = str(ROOT / "outputs/runs")
    cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    model.train(**cfg)


if __name__ == "__main__":
    main()
