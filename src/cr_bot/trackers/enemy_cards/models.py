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

    def add_observation(self, class_name, team, confidence, total_remaining_s):
        self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
        self.team_votes[team] = self.team_votes.get(team, 0) + 1
        self.confidence_sum += confidence
        self.seen_frames += 1
        self.last_seen_time = total_remaining_s

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
