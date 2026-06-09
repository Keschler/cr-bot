from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace


@dataclass(slots=True)
class FrameAnalysisResult(Mapping[str, object]):
    rendered: object
    elixir: dict
    elixir_change: dict | None
    towers_hp: dict
    time: str | None
    time_left_s: float | None
    total_remaining_s: float | None
    overtime: bool
    hand_state: dict
    yolo_boxes: object
    clock_boxes: list[dict]
    emote_boxes: list[dict]
    matches: list
    arena_px: tuple[int, int, int, int]
    tower_hp_debug_steps: dict[str, dict]
    timer_debug_steps: dict[str, object]

    def __getitem__(self, key: str) -> object:
        if key == "state":
            return self.hand_state
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        yield from (
            "rendered",
            "elixir",
            "elixir_change",
            "towers_hp",
            "time",
            "time_left_s",
            "total_remaining_s",
            "overtime",
            "state",
            "yolo_boxes",
            "clock_boxes",
            "emote_boxes",
            "matches",
            "arena_px",
            "tower_hp_debug_steps",
            "timer_debug_steps",
        )

    def __len__(self) -> int:
        return 16

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default

    def with_hand_state(self, hand_state: dict) -> "FrameAnalysisResult":
        return replace(self, hand_state=hand_state)

    def with_towers_hp(self, towers_hp: dict) -> "FrameAnalysisResult":
        return replace(self, towers_hp=towers_hp)

    def with_clock(self, *, time_left_s: float | None, total_remaining_s: float | None) -> "FrameAnalysisResult":
        return replace(
            self,
            time_left_s=time_left_s,
            total_remaining_s=total_remaining_s,
        )
