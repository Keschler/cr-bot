from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class CardClassifier:
    model: nn.Module
    classes: list[str]
    device: torch.device


_CARD_CLASSIFIER_CACHE: dict[tuple[str, str], CardClassifier] = {}


def torch_inference_device(env_var: str, *, default_cpu: bool = False) -> torch.device:
    configured = os.environ.get(env_var)
    if configured:
        return torch.device(configured)
    if not default_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def yolo_device() -> str:
    configured = os.environ.get("YOLO_DEVICE")
    if configured:
        return configured
    return "0" if torch.cuda.is_available() else "cpu"


def load_torch_checkpoint(path: str | Path, device: torch.device):
    return torch.load(Path(path), map_location=device)


def load_card_classifier(checkpoint_path: str | Path) -> CardClassifier | None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    device = torch_inference_device("CARD_CLASSIFIER_DEVICE")
    cache_key = (str(checkpoint_path.resolve()), str(device))
    cached = _CARD_CLASSIFIER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    classes = list(checkpoint["classes"])
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    classifier = CardClassifier(model=model, classes=classes, device=device)
    _CARD_CLASSIFIER_CACHE[cache_key] = classifier
    return classifier


def reset_model_loader_caches() -> None:
    _CARD_CLASSIFIER_CACHE.clear()
