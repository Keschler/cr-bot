from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from cr_bot.domain.events import TemporalSpellDetection

from .config import SPELL_CLASSES, TARGET_CLASSES, TemporalSpellConfig
from .features import arena_crop, clip_to_tensor


class TemporalSpellPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.config = TemporalSpellConfig()
        self.classes = list(SPELL_CLASSES)
        self.thresholds = {card: 0.5 for card in TARGET_CLASSES}
        self._frames: deque[np.ndarray] = deque(maxlen=self.config.clip_frames)
        self._model = None
        self._device = None
        self._last_emitted: dict[str, tuple[float, tuple[int, int] | None]] = {}
        self._last_sample_time_s: float | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        from .model import TemporalSpellCNN

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Temporal spell checkpoint is missing: {self.checkpoint_path}")
        self._device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self._device)
        self.classes = list(checkpoint["classes"])
        self.config = TemporalSpellConfig(**checkpoint["input_config"])
        self.thresholds.update(checkpoint.get("thresholds", {}))
        self._frames = deque(maxlen=self.config.clip_frames)
        self._model = TemporalSpellCNN(num_classes=len(self.classes)).to(self._device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

    def update(
        self,
        frame: np.ndarray,
        *,
        video_time_s: float,
        arena_px: tuple[int, int, int, int],
    ) -> list[TemporalSpellDetection]:
        self._load()
        if (
            self._last_sample_time_s is not None
            and video_time_s - self._last_sample_time_s < 1.0 / self.config.sample_fps - 1e-6
        ):
            return []
        self._last_sample_time_s = video_time_s
        self._frames.append(arena_crop(frame, arena_px).copy())
        if len(self._frames) < self.config.clip_frames:
            return []
        import torch

        inputs = torch.from_numpy(clip_to_tensor(list(self._frames), self.config))
        with torch.inference_mode():
            event_logits, heatmap_logits = self._model(inputs.unsqueeze(0).to(self._device))
            probabilities = event_logits.softmax(dim=1)[0]
            class_index = int(probabilities.argmax())
            card = self.classes[class_index]
            confidence = float(probabilities[class_index])
            if card == "background" or confidence < self.thresholds.get(card, 1.0):
                return []
            heatmap = heatmap_logits[0, TARGET_CLASSES.index(card)].sigmoid()
            flat_index = int(heatmap.argmax())
            cell = (flat_index % self.config.heatmap_width, flat_index // self.config.heatmap_width)
            heatmap_score = float(heatmap.flatten()[flat_index])
        if self._is_duplicate(card, cell, video_time_s):
            return []
        self._last_emitted[card] = (video_time_s, cell)
        return [TemporalSpellDetection(card, confidence, video_time_s, cell, heatmap_score)]

    def _is_duplicate(self, card: str, cell: tuple[int, int], now_s: float) -> bool:
        previous = self._last_emitted.get(card)
        if previous is None or now_s - previous[0] > self.config.cooldown_s:
            return False
        previous_cell = previous[1]
        if previous_cell is None:
            return True
        return max(abs(cell[0] - previous_cell[0]), abs(cell[1] - previous_cell[1])) <= self.config.cell_merge_distance
