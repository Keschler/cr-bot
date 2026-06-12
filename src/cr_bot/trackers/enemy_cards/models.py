from __future__ import annotations

from dataclasses import dataclass, field

from cr_bot.domain.constants import (
    FRAME_CONFIRM_MOVING_SPELLS,
    FRAME_CONFIRM_STATIONARY_SPELLS,
)
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD


SPELL_CARD_NAMES = {
    card
    for unit_name in FRAME_CONFIRM_MOVING_SPELLS | FRAME_CONFIRM_STATIONARY_SPELLS
    if (card := DIRECT_UNIT_TO_CARD.get(unit_name)) is not None
}

FRAME_CONFIRM_SPELL_CLASSES = (
    FRAME_CONFIRM_MOVING_SPELLS
    | FRAME_CONFIRM_STATIONARY_SPELLS
)


@dataclass
class TrackMemory:
    track_id: int | None
    first_seen_time: float
    last_seen_time: float
    first_seen_now_s: float | None = None
    class_votes: dict[str, int] = field(default_factory=dict)
    team_votes: dict[str, int] = field(default_factory=dict)
    confidence_sum: float = 0.0
    seen_frames: int = 0
    clock_confirmed: bool = False
    frame_confirmed: bool = False
    counted_as_card: bool = False
    center_x: float | None = None
    center_y: float | None = None
    deploy_clock_center_x: float | None = None
    deploy_clock_center_y: float | None = None
    last_clock_reject_reason: str | None = None
    observed_centers: list[tuple[float, float, float]] = field(default_factory=list)
    motion_centers: list[tuple[float, float, float]] = field(default_factory=list)

    def add_observation(self, class_name, team, confidence, total_remaining_s, *, center_x=None, center_y=None):
        self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
        self.team_votes[team] = self.team_votes.get(team, 0) + 1
        self.confidence_sum += confidence
        self.seen_frames += 1
        self.last_seen_time = total_remaining_s
        if center_x is not None and center_y is not None:
            self.observed_centers.append((total_remaining_s, center_x, center_y))
            del self.observed_centers[:-8]

    def add_motion_center(self, sample_time_s, center_x, center_y):
        if sample_time_s is None or center_x is None or center_y is None:
            return
        sample_time = round(float(sample_time_s), 3)
        same_frame = [
            item
            for item in self.motion_centers
            if round(item[0], 3) == sample_time
        ]
        if same_frame:
            count = len(same_frame)
            center_x = (
                sum(item[1] for item in same_frame) + center_x
            ) / (count + 1)
            center_y = (
                sum(item[2] for item in same_frame) + center_y
            ) / (count + 1)
            self.motion_centers = [
                item
                for item in self.motion_centers
                if round(item[0], 3) != sample_time
            ]
        self.motion_centers.append((float(sample_time_s), center_x, center_y))
        del self.motion_centers[:-8]

    @property
    def best_class(self) -> str | None:
        if not self.class_votes:
            return None
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def best_team(self):
        if not self.team_votes:
            return None
        return max(self.team_votes, key=self.team_votes.get)

    @property
    def avg_confidence(self):
        if self.seen_frames == 0:
            return 0.0
        return self.confidence_sum / self.seen_frames

    @property
    def best_team_ratio(self):
        if self.seen_frames == 0:
            return 0.0
        best_team = self.best_team
        if best_team is None:
            return 0.0
        return self.team_votes[best_team] / self.seen_frames


@dataclass
class RecentEnemyClock:
    seen_at_s: float | None
    center_x: float
    center_y: float
    track_id: int | None = None
    consumed_by_track_id: int | None = None


@dataclass(frozen=True)
class ProjectileTrajectoryConfig:
    update_window_s: float
    pre_observation_window_s: float
    min_early_samples: int
    corridor_width_norm: float


@dataclass
class EnemyProjectileSpellEvent:
    play_event_id: str
    card: str
    started_at_s: float
    first_track_id: int | None
    best_cell: tuple[int, int] | None
    observed_centers: list[tuple[float, float, float]] = field(default_factory=list)
    finalized: bool = False
    finalized_cell: tuple[int, int] | None = None
    claimed_observation_key: str | None = None
    best_observation_score: tuple | None = None
    last_assigned_time_left_s: float | None = None

    def append_centers(self, centers: list[tuple[float, float, float]]) -> None:
        if not centers:
            return
        seen = {(time_left_s, x, y) for time_left_s, x, y in self.observed_centers}
        for item in centers:
            if item not in seen:
                self.observed_centers.append(item)
                seen.add(item)
        self.observed_centers.sort(key=lambda item: item[0], reverse=True)
        del self.observed_centers[8:]


@dataclass
class RecentSpellTargetObservation:
    card: str
    time_left_s: float
    cell: tuple[int, int] | None
    phase: str
    quality: float
    center_x: float
    center_y: float
    key: str
    claimed_by_event_id: str | None = None


@dataclass
class RecentArenaFrame:
    time_left_s: float
    arena_bgr: object
