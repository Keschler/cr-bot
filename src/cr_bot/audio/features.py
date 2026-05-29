from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16_000
    window_s: float = 1.0
    n_fft: int = 1024
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 40.0
    f_max: float = 7600.0

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate * self.window_s))


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    sample_rate, data = wavfile.read(path)
    waveform = _pcm_to_float32(data)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32, copy=False), int(sample_rate)


def resample_audio(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=False)
    divisor = gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = source_sample_rate // divisor
    return resample_poly(waveform, up, down).astype(np.float32, copy=False)


def load_audio_window(
    path: str | Path,
    config: AudioFeatureConfig,
    *,
    start_s: float | None = None,
    random_offset: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    waveform, sample_rate = read_wav_mono(path)
    waveform = resample_audio(waveform, sample_rate, config.sample_rate)
    return crop_or_pad(
        waveform,
        config.num_samples,
        start_sample=None if start_s is None else int(round(start_s * config.sample_rate)),
        random_offset=random_offset,
        rng=rng,
    )


def crop_or_pad(
    waveform: np.ndarray,
    num_samples: int,
    *,
    start_sample: int | None = None,
    random_offset: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    waveform = waveform.astype(np.float32, copy=False)
    if len(waveform) >= num_samples:
        if start_sample is not None:
            start = max(0, min(start_sample, len(waveform) - num_samples))
        elif random_offset:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, len(waveform) - num_samples + 1))
        else:
            start = max(0, (len(waveform) - num_samples) // 2)
        return waveform[start : start + num_samples]

    padded = np.zeros(num_samples, dtype=np.float32)
    if len(waveform) == 0:
        return padded
    start = max(0, (num_samples - len(waveform)) // 2)
    padded[start : start + len(waveform)] = waveform
    return padded


def waveform_to_log_mel(
    waveform: np.ndarray | torch.Tensor,
    config: AudioFeatureConfig,
) -> torch.Tensor:
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform.astype(np.float32, copy=False))
    waveform = waveform.float()
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform shape [T], got {tuple(waveform.shape)}")

    window = torch.hann_window(config.win_length, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = stft.abs().pow(2.0)
    mel_filter = mel_filterbank(config, device=waveform.device, dtype=power.dtype)
    mel = torch.matmul(mel_filter, power)
    log_mel = torch.log(torch.clamp(mel, min=1e-8))
    log_mel = (log_mel - log_mel.mean()) / torch.clamp(log_mel.std(), min=1e-5)
    return log_mel.unsqueeze(0)


def mel_filterbank(
    config: AudioFeatureConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    n_freqs = config.n_fft // 2 + 1
    min_mel = _hz_to_mel(config.f_min)
    max_mel = _hz_to_mel(config.f_max)
    mels = torch.linspace(min_mel, max_mel, config.n_mels + 2, device=device, dtype=dtype)
    hz = _mel_to_hz(mels)
    bins = torch.floor((config.n_fft + 1) * hz / config.sample_rate).long()
    bins = torch.clamp(bins, min=0, max=n_freqs - 1)

    filters = torch.zeros(config.n_mels, n_freqs, device=device, dtype=dtype)
    for mel_idx in range(config.n_mels):
        left = int(bins[mel_idx])
        center = int(bins[mel_idx + 1])
        right = int(bins[mel_idx + 2])
        if center > left:
            filters[mel_idx, left:center] = torch.linspace(
                0.0,
                1.0,
                center - left,
                device=device,
                dtype=dtype,
            )
        if right > center:
            filters[mel_idx, center:right] = torch.linspace(
                1.0,
                0.0,
                right - center,
                device=device,
                dtype=dtype,
            )
    return filters


def _pcm_to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return np.nan_to_num(
            np.clip(data.astype(np.float32, copy=False), -1.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
    if np.issubdtype(data.dtype, np.signedinteger):
        max_value = float(np.iinfo(data.dtype).max)
        return np.nan_to_num(
            np.clip(data.astype(np.float32) / max_value, -1.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
    if np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        midpoint = (info.max + 1) / 2.0
        return np.nan_to_num(
            np.clip((data.astype(np.float32) - midpoint) / midpoint, -1.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
    raise TypeError(f"Unsupported WAV dtype: {data.dtype}")


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)
