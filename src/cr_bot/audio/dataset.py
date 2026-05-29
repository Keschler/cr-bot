from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    AudioFeatureConfig,
    crop_or_pad,
    load_audio_window,
    read_wav_mono,
    resample_audio,
    waveform_to_log_mel,
)
from .labels import folder_to_card_keys, normalize_card_key

DEPLOY_NAME_PARTS = (
    "deploy",
    "dep",
    "spawn",
    "summon",
    "cast",
    "land",
)

NO_EVENT_CLASS = "no_event"


def collect_sfx_files(
    raw_sfx_dir: str | Path,
    *,
    deploy_only: bool = True,
    known_cards: set[str] | None = None,
) -> tuple[list[tuple[str, Path]], dict[str, list[Path]], list[Path]]:
    raw_sfx_dir = Path(raw_sfx_dir)
    samples: list[tuple[str, Path]] = []
    skipped_by_folder: dict[str, list[Path]] = {}
    unmapped: list[Path] = []

    for folder in sorted(raw_sfx_dir.iterdir()):
        if not folder.is_dir():
            continue
        cards = folder_to_card_keys(folder)
        if not cards:
            unmapped.append(folder)
            continue
        cards = [card for card in cards if known_cards is None or card in known_cards]
        if not cards:
            for card in folder_to_card_keys(folder):
                skipped_by_folder.setdefault(card, []).append(folder)
            continue

        wavs = sorted(folder.glob("*.wav"))
        deploy_wavs = [path for path in wavs if is_deploy_like(path)]
        selected = deploy_wavs if deploy_only and deploy_wavs else wavs
        for card in cards:
            samples.extend((card, path) for path in selected)

    return samples, skipped_by_folder, unmapped


def is_deploy_like(path: str | Path) -> bool:
    name = Path(path).stem.lower()
    return any(part in name for part in DEPLOY_NAME_PARTS)


class GameplayBackground:
    def __init__(
        self,
        paths: list[str | Path],
        config: AudioFeatureConfig,
        *,
        ground_truth_path: str | Path | None = None,
        fps: float = 10.0,
        exclude_before_s: float = 1.0,
        exclude_after_s: float = 1.5,
    ):
        self.config = config
        self.tracks: list[np.ndarray] = []
        self.allowed_windows: list[tuple[int, int, int]] = []

        excluded = []
        if ground_truth_path is not None:
            excluded = load_ground_truth_exclusion_ranges(
                ground_truth_path,
                fps=fps,
                before_s=exclude_before_s,
                after_s=exclude_after_s,
            )

        for path in paths:
            waveform, sample_rate = read_wav_mono(path)
            waveform = resample_audio(waveform, sample_rate, config.sample_rate)
            track_idx = len(self.tracks)
            self.tracks.append(waveform)
            self.allowed_windows.extend(
                allowed_sample_ranges(
                    len(waveform),
                    config.sample_rate,
                    excluded,
                    config.num_samples,
                    track_idx=track_idx,
                )
            )

    @property
    def available(self) -> bool:
        return bool(self.allowed_windows)

    def sample_window(self, rng: np.random.Generator) -> np.ndarray:
        if not self.allowed_windows:
            raise ValueError("No background windows available")
        track_idx, start, end = self.allowed_windows[int(rng.integers(0, len(self.allowed_windows)))]
        max_start = end - self.config.num_samples
        offset = int(rng.integers(start, max_start + 1))
        return self.tracks[track_idx][offset : offset + self.config.num_samples].copy()


class MixedSFXCardDataset(Dataset):
    def __init__(
        self,
        sfx_samples: list[tuple[str, Path]],
        classes: list[str],
        config: AudioFeatureConfig,
        *,
        background: GameplayBackground | None = None,
        samples_per_sfx: int = 8,
        no_event_count: int | None = None,
        seed: int = 0,
    ):
        if NO_EVENT_CLASS not in classes:
            raise ValueError(f"{NO_EVENT_CLASS!r} must be present in classes")
        self.sfx_samples = list(sfx_samples)
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.config = config
        self.background = background
        self.samples_per_sfx = max(1, int(samples_per_sfx))
        self.no_event_count = (
            len(self.sfx_samples) if no_event_count is None else max(0, int(no_event_count))
        )
        self.seed = int(seed)
        self.positive_count = len(self.sfx_samples) * self.samples_per_sfx

    def __len__(self) -> int:
        return self.positive_count + self.no_event_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rng = np.random.default_rng(self.seed + index)
        if index >= self.positive_count:
            waveform = self._background_or_noise(rng)
            label = self.class_to_idx[NO_EVENT_CLASS]
            return waveform_to_log_mel(waveform, self.config), label

        card, path = self.sfx_samples[index // self.samples_per_sfx]
        sfx = load_audio_window(path, self.config, random_offset=False, rng=rng)
        background = self._background_or_noise(rng)

        mixed = background * float(rng.uniform(0.35, 0.9))
        insert = int(rng.uniform(0.15, 0.45) * self.config.sample_rate)
        sfx_gain = float(rng.uniform(0.55, 1.25))
        available = min(len(sfx), len(mixed) - insert)
        if available > 0:
            mixed[insert : insert + available] += sfx[:available] * sfx_gain
        mixed += rng.normal(0.0, 0.002, size=mixed.shape).astype(np.float32)
        mixed = normalize_peak(mixed)
        label = self.class_to_idx[card]
        return waveform_to_log_mel(mixed, self.config), label

    def _background_or_noise(self, rng: np.random.Generator) -> np.ndarray:
        if self.background is not None and self.background.available:
            return self.background.sample_window(rng)
        return rng.normal(0.0, 0.01, size=self.config.num_samples).astype(np.float32)


def split_sfx_samples(
    samples: list[tuple[str, Path]],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Path]] = {}
    for card, path in samples:
        by_class.setdefault(card, []).append(path)

    train: list[tuple[str, Path]] = []
    val: list[tuple[str, Path]] = []
    for card, paths in sorted(by_class.items()):
        paths = list(paths)
        rng.shuffle(paths)
        val_count = 1 if len(paths) > 1 else 0
        val_count = max(val_count, int(round(len(paths) * val_fraction)))
        val_count = min(val_count, max(0, len(paths) - 1))
        val.extend((card, path) for path in paths[:val_count])
        train.extend((card, path) for path in paths[val_count:])
    return train, val


def build_real_event_windows(
    audio_path: str | Path,
    ground_truth_path: str | Path,
    classes: list[str],
    config: AudioFeatureConfig,
    *,
    fps: float = 10.0,
    side: str = "enemy",
    start_offset_s: float = -0.3,
) -> list[tuple[torch.Tensor, int, dict]]:
    waveform, sample_rate = read_wav_mono(audio_path)
    waveform = resample_audio(waveform, sample_rate, config.sample_rate)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    events = load_ground_truth_events(ground_truth_path, fps=fps, side=side)
    windows = []
    for event in events:
        card = normalize_card_key(event["card"])
        if card not in class_to_idx:
            continue
        start_sample = int(round((event["time_s"] + start_offset_s) * config.sample_rate))
        window = crop_or_pad(waveform, config.num_samples, start_sample=start_sample)
        windows.append((waveform_to_log_mel(window, config), class_to_idx[card], event))
    return windows


def load_ground_truth_events(
    ground_truth_path: str | Path,
    *,
    fps: float = 10.0,
    side: str | None = None,
) -> list[dict]:
    with Path(ground_truth_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    events = []
    for event in payload.get("events", []):
        if side is not None and event.get("side") != side:
            continue
        if "frame_index" not in event or "card" not in event:
            continue
        copied = dict(event)
        copied["card"] = normalize_card_key(copied["card"])
        copied["time_s"] = float(copied["frame_index"]) / fps
        events.append(copied)
    return events


def load_ground_truth_exclusion_ranges(
    ground_truth_path: str | Path,
    *,
    fps: float,
    before_s: float,
    after_s: float,
) -> list[tuple[float, float]]:
    ranges = []
    for event in load_ground_truth_events(ground_truth_path, fps=fps, side=None):
        ranges.append((max(0.0, event["time_s"] - before_s), event["time_s"] + after_s))
    return ranges


def allowed_sample_ranges(
    num_samples: int,
    sample_rate: int,
    excluded_s: list[tuple[float, float]],
    window_samples: int,
    *,
    track_idx: int,
) -> list[tuple[int, int, int]]:
    excluded = sorted(
        (
            max(0, int(round(start * sample_rate))),
            min(num_samples, int(round(end * sample_rate))),
        )
        for start, end in excluded_s
    )
    allowed = []
    cursor = 0
    for start, end in excluded:
        if start - cursor >= window_samples:
            allowed.append((track_idx, cursor, start))
        cursor = max(cursor, end)
    if num_samples - cursor >= window_samples:
        allowed.append((track_idx, cursor, num_samples))
    return allowed


def normalize_peak(waveform: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if max_abs > peak:
        waveform = waveform * (peak / max_abs)
    return waveform.astype(np.float32, copy=False)


def feature_config_to_dict(config: AudioFeatureConfig) -> dict:
    return asdict(config)
