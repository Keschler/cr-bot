from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrontendFrame:
    """One UI-facing frame with decision payload and optional JPEG preview."""

    frame_index: int
    timestamp_s: float
    jpeg_bytes: bytes | None
    record: dict = field(default_factory=dict)
    suggestions: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    in_game: bool = False
    emitted: bool = False
    # Tracker events newly confirmed on this frame (timeline markers).
    own_actions: list[dict] = field(default_factory=list)
    enemy_plays: list[dict] = field(default_factory=list)
    # Pixel dimensions of the coordinate space that detection ``center_px``
    # values refer to (the normalized frame passed to the extractor, before
    # JPEG downscaling). The UI uses these to map overlays onto the preview.
    frame_width: int | None = None
    frame_height: int | None = None
    # Raw troop rows retained for label correction (box + class + team +
    # confidence + track + hp + center). Bounded per frame by detector output.
    detections: list[dict] = field(default_factory=list)
    # What-if re-evaluation overlay (display only; trackers/history untouched).
    corrected: dict | None = None
