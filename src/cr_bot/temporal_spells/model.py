from __future__ import annotations

import torch
from torch import nn

from .config import TARGET_CLASSES


class TemporalSpellCNN(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 18)),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0), groups=64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.event_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.spatial_head = nn.Conv2d(64, len(TARGET_CLASSES), 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, timesteps, channels, height, width = inputs.shape
        encoded = self.encoder(inputs.reshape(batch * timesteps, channels, height, width))
        encoded = encoded.reshape(batch, timesteps, 64, 32, 18).permute(0, 2, 1, 3, 4)
        temporal = self.temporal(encoded)
        event_logits = self.event_head(temporal)
        heatmap_logits = self.spatial_head(temporal[:, :, -1])
        return event_logits, heatmap_logits
