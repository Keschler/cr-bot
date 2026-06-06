from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from cr_bot.domain.constants import FULL_TOWER_HP
from cr_bot.paths import MODELS_DIR
from cr_bot.vision.model_loader import load_torch_checkpoint, torch_inference_device

DIGITS = "0123456789"
BLANK_IDX = 0
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128
TOWER_HP_OCR_PATH = MODELS_DIR / "tower_hp_crnn_best.pt"


@dataclass(frozen=True)
class TowerHPOCRPrediction:
    value: int | None
    text: str
    readable_prob: float
    char_confidence: float
    rejected_reason: str | None = None


class TowerHPCRNN(nn.Module):
    """Small CRNN for one tower HP text crop.

    The CNN turns the crop into a left-to-right feature sequence. The GRU reads
    that sequence and the CTC head emits digits without needing per-digit boxes.
    A separate readability head lets the model reject blocked/empty crops.
    """

    def __init__(self, hidden_size: int = 96, num_layers: int = 1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        out_features = hidden_size * 2
        self.ctc_head = nn.Linear(out_features, len(DIGITS) + 1)
        self.readable_head = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(x)
        features = features.mean(dim=2).permute(0, 2, 1)
        seq, _ = self.rnn(features)
        logits = self.ctc_head(seq)
        readable_logits = self.readable_head(seq.mean(dim=1)).squeeze(1)
        return logits, readable_logits


def normalize_tower_hp_crop(crop: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    if crop is None or crop.size == 0:
        raise ValueError("normalize_tower_hp_crop received an empty crop")
    if len(crop.shape) == 2:
        gray = crop
    elif crop.shape[2] == 4:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor, resized


def encode_text(text: str) -> list[int]:
    return [DIGITS.index(char) + 1 for char in text]


def decode_ctc_logits(logits: torch.Tensor) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)
    indices = probs.argmax(dim=-1)
    chars: list[str] = []
    confidences: list[float] = []
    previous = BLANK_IDX
    for idx_tensor, prob_row in zip(indices, probs):
        idx = int(idx_tensor.item())
        if idx != BLANK_IDX and idx != previous:
            chars.append(DIGITS[idx - 1])
            confidences.append(float(prob_row[idx].item()))
        previous = idx
    if not chars:
        return "", 0.0
    return "".join(chars), float(min(confidences))


def validate_tower_hp_prediction(
    *,
    text: str,
    readable_prob: float,
    char_confidence: float,
    tower_name: str,
    min_readable_prob: float,
    min_char_confidence: float,
) -> tuple[int | None, str | None]:
    if readable_prob < min_readable_prob:
        return None, "unreadable"
    if not text:
        return None, "empty_text"
    if not text.isdigit():
        return None, "non_digit_text"
    if len(text) > 4:
        return None, "too_many_digits"
    if char_confidence < min_char_confidence:
        return None, "low_char_confidence"

    value = int(text)
    max_hp = FULL_TOWER_HP[tower_name]
    if value > max_hp:
        return None, "above_max_hp"
    return value, None


class TowerHPOCR:
    def __init__(
        self,
        checkpoint_path: Path = TOWER_HP_OCR_PATH,
        *,
        device: str | None = None,
        min_readable_prob: float = 0.65,
        min_char_confidence: float = 0.45,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or _default_device())
        self.min_readable_prob = min_readable_prob
        self.min_char_confidence = min_char_confidence
        self.model = TowerHPCRNN()
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Tower HP CRNN checkpoint is missing: {self.checkpoint_path}. "
                "Train it with scripts/train_tower_hp_ocr.py or set TOWER_HP_OCR_PATH."
            )

        checkpoint = load_torch_checkpoint(self.checkpoint_path, self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(
        self,
        crops_by_tower: dict[str, np.ndarray],
        *,
        debug_steps_by_tower: dict[str, dict] | None = None,
    ) -> dict[str, TowerHPOCRPrediction]:
        if not crops_by_tower:
            return {}

        tower_names = list(crops_by_tower)
        tensors: list[torch.Tensor] = []
        normalized_images: dict[str, np.ndarray] = {}
        for tower_name in tower_names:
            tensor, normalized = normalize_tower_hp_crop(crops_by_tower[tower_name])
            tensors.append(tensor)
            normalized_images[tower_name] = normalized

        batch = torch.stack(tensors, dim=0).to(self.device)
        logits, readable_logits = self.model(batch)
        readable_probs = torch.sigmoid(readable_logits)

        predictions: dict[str, TowerHPOCRPrediction] = {}
        for idx, tower_name in enumerate(tower_names):
            text, char_confidence = decode_ctc_logits(logits[idx].cpu())
            readable_prob = float(readable_probs[idx].cpu().item())
            value, rejected_reason = validate_tower_hp_prediction(
                text=text,
                readable_prob=readable_prob,
                char_confidence=char_confidence,
                tower_name=tower_name,
                min_readable_prob=self.min_readable_prob,
                min_char_confidence=self.min_char_confidence,
            )
            prediction = TowerHPOCRPrediction(
                value=value,
                text=text,
                readable_prob=readable_prob,
                char_confidence=char_confidence,
                rejected_reason=rejected_reason,
            )
            predictions[tower_name] = prediction

            if debug_steps_by_tower is not None:
                tower_debug = debug_steps_by_tower.setdefault(tower_name, {})
                tower_debug["raw"] = crops_by_tower[tower_name]
                tower_debug["normalized"] = normalized_images[tower_name]
                tower_debug["ocr_model"] = "tower_hp_crnn"
                tower_debug["ocr_text"] = text
                tower_debug["ocr_value"] = value
                tower_debug["ocr_readable_prob"] = readable_prob
                tower_debug["ocr_confidence"] = char_confidence
                if rejected_reason is not None:
                    tower_debug["ocr_rejected_reason"] = rejected_reason

        return predictions


_OCR_CACHE: dict[tuple[str, str], TowerHPOCR] = {}


def _default_device() -> str:
    return str(torch_inference_device("TOWER_HP_OCR_DEVICE"))


def get_tower_hp_ocr(checkpoint_path: Path | None = None) -> TowerHPOCR:
    path = Path(os.environ.get("TOWER_HP_OCR_PATH") or checkpoint_path or TOWER_HP_OCR_PATH)
    device = _default_device()
    key = (str(path), device)
    if key not in _OCR_CACHE:
        _OCR_CACHE[key] = TowerHPOCR(path, device=device)
    return _OCR_CACHE[key]


def reset_tower_hp_ocr_cache() -> None:
    _OCR_CACHE.clear()
