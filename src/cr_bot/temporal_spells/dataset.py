from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import random

import cv2
import numpy as np

from .config import SPELL_CLASSES, TemporalSpellConfig
from .features import clip_to_tensor


CAST_OWNERSHIPS = ("own", "enemy")


def normalize_manifest_row(row: dict, *, source: str = "manifest") -> dict:
    normalized = dict(row)
    card = str(normalized.get("card", "")).strip().lower()
    if card not in SPELL_CLASSES:
        raise ValueError(f"{source}: unsupported temporal spell card {card!r}")
    normalized["card"] = card

    ownership = normalized.get("ownership")
    if ownership is not None:
        ownership = str(ownership).strip().lower()
    if card != "background" and ownership not in CAST_OWNERSHIPS:
        raise ValueError(
            f"{source}: {card} cast must have ownership 'own' or 'enemy'"
        )
    if card == "background" and ownership not in {None, "background"}:
        raise ValueError(
            f"{source}: background rows cannot have cast ownership {ownership!r}"
        )
    normalized["ownership"] = ownership or "background"
    return normalized


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(
                    normalize_manifest_row(
                        json.loads(line),
                        source=f"{path}:{len(rows) + 1}",
                    )
                )
    return rows


def split_rows_by_session(
    rows: list[dict],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get("recording_session") or row.get("video"))
        groups[key].append(dict(row))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    test_count = round(len(keys) * test_fraction)
    val_count = round(len(keys) * val_fraction)
    assignment = {
        key: "test" if i < test_count else "val" if i < test_count + val_count else "train"
        for i, key in enumerate(keys)
    }
    result = {"train": [], "val": [], "test": []}
    for key, group in groups.items():
        result[assignment[key]].extend({**row, "split": assignment[key]} for row in group)
    return result


def clip_end_times(event_time_s: float, offsets_s: list[float]) -> list[float]:
    return [event_time_s + offset for offset in offsets_s]


def overlaps_event_window(
    clip_end_s: float,
    event_times_s: list[float],
    config: TemporalSpellConfig,
    *,
    exclusion_s: float = 0.3,
) -> bool:
    clip_start_s = clip_end_s - (config.clip_frames - 1) / config.sample_fps
    return any(clip_start_s - exclusion_s <= event <= clip_end_s + exclusion_s for event in event_times_s)


def extract_causal_clip(
    video_path: str | Path,
    clip_end_s: float,
    config: TemporalSpellConfig,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frames = []
    start_s = clip_end_s - (config.clip_frames - 1) / config.sample_fps
    try:
        for index in range(config.clip_frames):
            timestamp_s = start_s + index / config.sample_fps
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"missing frame at {timestamp_s:.3f}s in {video_path}")
            frames.append(frame)
    finally:
        capture.release()
    return frames


class TemporalSpellDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        config: TemporalSpellConfig | None = None,
        *,
        include_ownership: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        self.config = config or TemporalSpellConfig()
        self.include_ownership = include_ownership
        self.class_to_idx = {name: index for index, name in enumerate(SPELL_CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        frames = [cv2.imread(str(path)) for path in row["frame_paths"]]
        if any(frame is None for frame in frames):
            raise FileNotFoundError(f"missing clip frame for row {index}")
        inputs = torch.from_numpy(clip_to_tensor(frames, self.config))
        label = self.class_to_idx[str(row["card"])]
        target = torch.zeros((32, 18), dtype=torch.float32)
        cell = row.get("target_cell")
        has_target = cell is not None
        if has_target:
            col, grid_row = map(int, cell)
            target[grid_row, col] = 1.0
        sample = (inputs, label, target, has_target)
        if self.include_ownership:
            return (*sample, row["ownership"])
        return sample
