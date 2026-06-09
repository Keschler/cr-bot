from __future__ import annotations

from dataclasses import dataclass


LOG_YOLO_AFTER_PENDING_WINDOW_S = 1.0
ELIXIR_CHANGE_VIDEO_TIME_OFFSET_S = 0.1
ROLLING_SPELL_UNIT_LABELS = {
    "barbarian-barrel": "barbarian-barrel",
    "log": "the-log",
}
CARD_ALIASES = {
    "old-musketeer": "musketeer",
}
PENDING_UNIT_LABELS = {
    "barbarians": {"barbarian", "barbarian-evolution"},
    "goblins": {"goblin"},
    "minions": {"minion"},
    "skeletons": {"skeleton", "skeleton-evolution"},
}


@dataclass
class PendingOwnPlay:
    card: str
    slot_idx: int
    started_at_s: float
    elixir_before: float
    confirmed: bool = False
    spell_aim_seen: bool = False
    spell_elixir_confirmed: bool = False
    spell_release_seen: bool = False
    spell_target_cell: tuple[int, int] | None = None
    elixir_change_time_s: float | None = None
    elixir_change_video_time_s: float | None = None
    numeric_elixir_drop_time_s: float | None = None
    numeric_elixir_drop_video_time_s: float | None = None
    numeric_elixir_drop_source: str | None = None
    rolling_spell_first_cell: tuple[int, int] | None = None
    rolling_spell_first_seen_s: float | None = None
    rolling_spell_first_track_id: int | None = None


@dataclass
class RecentAllyTrack:
    match: object
    first_seen_s: float
    last_seen_s: float
