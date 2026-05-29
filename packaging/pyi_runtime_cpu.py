"""Runtime settings for the CPU-only PyInstaller bundle."""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("YOLO_DEVICE", "cpu")
