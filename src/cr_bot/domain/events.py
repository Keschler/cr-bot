from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields


class DataclassMapping(Mapping[str, object]):
    """Dataclasses that still behave like read-only mappings during migration."""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            yield field.name

    def __len__(self) -> int:
        return len(fields(self))

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass(slots=True)
class OwnActionEvent(DataclassMapping):
    time_left_s: float
    video_time_s: float | None
    card: str
    slot_idx: int | None
    cell: tuple[int, int] | None
    rolling_spell_track_id: int | None = None
    played_via: str | None = None


@dataclass(frozen=True, slots=True)
class TemporalSpellDetection(DataclassMapping):
    card: str
    confidence: float
    video_time_s: float
    target_cell: tuple[int, int] | None
    heatmap_score: float


@dataclass(slots=True)
class EnemyCardPlay(DataclassMapping):
    event_id: str
    time_left_s: float
    total_remaining_s: float
    video_time_s: float | None
    card: str
    cost: int
    track_id: int | None
    cell: tuple[int, int] | None
    clock_confirmed: bool
    frame_confirmed: bool
    avg_confidence: float
    team_ratio: float
    best_class: str | None
    class_votes: dict[str, int]
    is_spell: bool
    overtime: bool
    discard_reason: str | None = None
    played_via: str | None = None
