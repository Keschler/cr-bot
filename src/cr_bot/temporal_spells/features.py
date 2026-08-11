from __future__ import annotations

import cv2
import numpy as np

from .config import TemporalSpellConfig


def arena_crop(frame: np.ndarray, arena_px: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = arena_px
    return frame[y : y + height, x : x + width]


def clip_to_tensor(
    frames: list[np.ndarray],
    config: TemporalSpellConfig,
) -> np.ndarray:
    if len(frames) != config.clip_frames:
        raise ValueError(f"expected {config.clip_frames} frames, got {len(frames)}")
    rgb = np.stack(
        [
            cv2.cvtColor(
                cv2.resize(
                    frame,
                    (config.input_width, config.input_height),
                    interpolation=cv2.INTER_AREA,
                ),
                cv2.COLOR_BGR2RGB,
            )
            for frame in frames
        ]
    ).astype(np.float32) / 255.0
    differences = np.zeros_like(rgb)
    differences[1:] = np.abs(rgb[1:] - rgb[:-1])
    return np.concatenate((rgb, differences), axis=-1).transpose(0, 3, 1, 2)
