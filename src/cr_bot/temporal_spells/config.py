from __future__ import annotations

from dataclasses import asdict, dataclass


SPELL_CLASSES = ("background", "clone", "lightning", "zap", "arrows")
TARGET_CLASSES = SPELL_CLASSES[1:]


@dataclass(frozen=True, slots=True)
class TemporalSpellConfig:
    clip_frames: int = 8
    sample_fps: float = 10.0
    input_width: int = 192
    input_height: int = 288
    heatmap_width: int = 18
    heatmap_height: int = 32
    cooldown_s: float = 1.2
    cell_merge_distance: int = 3

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
