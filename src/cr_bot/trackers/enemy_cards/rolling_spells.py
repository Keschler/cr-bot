from dataclasses import dataclass

from cr_bot.domain.constants import (
    ENEMY_CARD_CONFIRM_FRAMES,
    ENEMY_CARD_STALE_AFTER_SECONDS,
)
from cr_bot.features.action_space import ACTION_GRID

from .models import TrackMemory


MIN_POSITIVE_STEPS = 2
MIN_ROW_DELTA = 1.0
MAX_REVERSE_ROWS = 0.35
OWN_CLAIM_WINDOW_S = 3.25
OWN_CLAIM_LANE_TOLERANCE = 2
OWN_POST_ACTION_FRAGMENT_WINDOW_S = 0.75
ENEMY_LOG_FRAGMENT_WINDOW_S = 4.0
ENEMY_LOG_FRAGMENT_LANE_TOLERANCE = 1


@dataclass(frozen=True)
class RecordedLogTrajectory:
    time_left_s: float
    lane: int
    last_row: int


class RollingSpellTracker:
    """Tracks ground spells whose owner is determined by vertical direction."""

    def __init__(self, debug):
        self.candidates: dict[int, TrackMemory] = {}
        self.own_claims: dict[tuple[float, tuple[int, int]], int | None] = {}
        self.own_source_track_ids: set[int] = set()
        self.recorded_enemy_logs: list[RecordedLogTrajectory] = []
        self._debug = debug

    def remember_own_actions(self, own_actions):
        for action in own_actions:
            if action.get("card") != "log":
                continue
            time_left_s = action.get("time_left_s")
            cell = action.get("cell")
            if time_left_s is None or cell is None:
                continue
            source_track_id = action.get("rolling_spell_track_id")
            if source_track_id is not None:
                self.own_source_track_ids.add(int(source_track_id))
            self.own_claims.setdefault((float(time_left_s), tuple(cell)), None)
        if len(self.own_claims) > 32:
            newest = sorted(self.own_claims, reverse=True)[:32]
            self.own_claims = {
                claim: self.own_claims[claim]
                for claim in newest
            }

    def observe(self, troop, *, time_left_s, now_s, arena_px):
        if arena_px is None:
            return None
        track_id = getattr(troop, "track_id", None)
        if track_id is None:
            return None
        memory = self.candidates.get(track_id)
        if memory is None:
            memory = TrackMemory(
                track_id=track_id,
                first_seen_time=time_left_s,
                last_seen_time=time_left_s,
                first_seen_now_s=now_s,
            )
            self.candidates[track_id] = memory
        memory.add_observation(
            troop.class_name,
            troop.team,
            troop.confidence,
            time_left_s,
            center_x=troop.center_x,
            center_y=troop.center_y,
        )
        memory.center_x = troop.center_x
        memory.center_y = troop.center_y
        self.assign_own_claims(arena_px)
        if memory.counted_as_card or memory.seen_frames < ENEMY_CARD_CONFIRM_FRAMES:
            return None
        trajectory = self._trajectory_metrics(memory, arena_px)
        if trajectory is None:
            return None
        row_delta, positive_steps, reverse_rows, first_cell, last_cell = trajectory
        self._debug(
            f"log trajectory track={track_id} teams={memory.team_votes} "
            f"first_cell={first_cell} last_cell={last_cell} "
            f"row_delta={row_delta:.2f} positive_steps={positive_steps} "
            f"max_reverse_rows={reverse_rows:.2f}"
        )
        if (
            row_delta < MIN_ROW_DELTA
            or positive_steps < MIN_POSITIVE_STEPS
            or reverse_rows > MAX_REVERSE_ROWS
        ):
            return None
        if self.matches_own_claim(memory, arena_px):
            self._debug(
                f"suppress direction-confirmed enemy log track={track_id}: "
                "matches claimed own log trajectory"
            )
            memory.counted_as_card = True
            return None
        return memory

    def assign_own_claims(self, arena_px):
        if arena_px is None:
            return
        for claim, track_id in list(self.own_claims.items()):
            memory = self.candidates.get(track_id)
            if (
                memory is not None
                and self._is_confirmed_enemy_direction(memory, arena_px)
                and not self._matches_nearby_own_action(
                    memory, arena_px, claims=(claim,)
                )
            ):
                self.own_claims[claim] = None
                self._debug(
                    f"released own log trajectory claim track={track_id}: "
                    "motion resolved toward increasing rows"
                )
        assigned_track_ids = {
            track_id
            for track_id in self.own_claims.values()
            if track_id is not None
        }
        for claim, claimed_track_id in self.own_claims.items():
            if claimed_track_id is not None:
                continue
            claim_time, claim_cell = claim
            candidates = []
            for track_id, memory in self.candidates.items():
                if track_id in assigned_track_ids or not memory.observed_centers:
                    continue
                if (
                    self._is_confirmed_enemy_direction(memory, arena_px)
                    and not self._matches_nearby_own_action(
                        memory, arena_px, claims=(claim,)
                    )
                ):
                    continue
                first = memory.observed_centers[0]
                first_cell = ACTION_GRID.pixel_to_cell(first[1], first[2], arena_px)
                if first_cell is None:
                    continue
                time_distance = abs(claim_time - memory.first_seen_time)
                if time_distance > OWN_CLAIM_WINDOW_S:
                    continue
                lane_distance = abs(claim_cell[0] - first_cell[0])
                if lane_distance > OWN_CLAIM_LANE_TOLERANCE:
                    continue
                candidates.append((time_distance, lane_distance, track_id))
            if not candidates:
                continue
            track_id = min(candidates)[2]
            self.own_claims[claim] = track_id
            assigned_track_ids.add(track_id)
            self._debug(
                f"claimed own log trajectory track={track_id} "
                f"action_time_left={claim_time} action_cell={claim_cell}"
            )

    def matches_own_claim(self, memory, arena_px):
        if arena_px is None or not self.own_claims:
            return False
        self.assign_own_claims(arena_px)
        shared_source_track = memory.track_id in self.own_source_track_ids
        if (
            self._matches_nearby_own_action(memory, arena_px)
            and not (
                shared_source_track
                and self._is_confirmed_enemy_direction(memory, arena_px)
            )
        ):
            return True
        if self._is_confirmed_enemy_direction(memory, arena_px):
            return False
        return memory.track_id in self.own_claims.values()

    def _matches_nearby_own_action(self, memory, arena_px, *, claims=None):
        if not memory.observed_centers:
            return False
        first = memory.observed_centers[0]
        first_cell = ACTION_GRID.pixel_to_cell(first[1], first[2], arena_px)
        if first_cell is None:
            return False
        for claim_time, claim_cell in claims or self.own_claims:
            time_distance = abs(claim_time - memory.first_seen_time)
            if time_distance > OWN_POST_ACTION_FRAGMENT_WINDOW_S:
                continue
            if abs(claim_cell[0] - first_cell[0]) <= OWN_CLAIM_LANE_TOLERANCE:
                return True
        return False

    @classmethod
    def _is_confirmed_enemy_direction(cls, memory, arena_px):
        trajectory = cls._trajectory_metrics(memory, arena_px)
        if trajectory is None:
            return False
        row_delta, positive_steps, reverse_rows, _, _ = trajectory
        return (
            row_delta >= MIN_ROW_DELTA
            and positive_steps >= MIN_POSITIVE_STEPS
            and reverse_rows <= MAX_REVERSE_ROWS
        )

    def is_later_fragment(self, memory, *, time_left_s, arena_px):
        trajectory = self._trajectory_metrics(memory, arena_px)
        if trajectory is None:
            return False
        _, _, _, first_cell, _ = trajectory
        if first_cell is None:
            return False
        first_lane, first_row = first_cell
        for recorded in reversed(self.recorded_enemy_logs):
            elapsed_s = recorded.time_left_s - time_left_s
            if elapsed_s < 0:
                continue
            if elapsed_s > ENEMY_LOG_FRAGMENT_WINDOW_S:
                break
            same_lane = (
                abs(recorded.lane - first_lane)
                <= ENEMY_LOG_FRAGMENT_LANE_TOLERANCE
            )
            if same_lane and first_row > recorded.last_row:
                return True
        return False

    def remember_enemy_log(self, memory, *, time_left_s, arena_px):
        trajectory = self._trajectory_metrics(memory, arena_px)
        if trajectory is None:
            return
        _, _, _, _, last_cell = trajectory
        if last_cell is None:
            return
        self.recorded_enemy_logs.append(
            RecordedLogTrajectory(
                time_left_s=float(time_left_s),
                lane=last_cell[0],
                last_row=last_cell[1],
            )
        )
        del self.recorded_enemy_logs[:-32]

    def cleanup(self, time_left_s):
        self.candidates = {
            track_id: memory
            for track_id, memory in self.candidates.items()
            if memory.last_seen_time - time_left_s <= ENEMY_CARD_STALE_AFTER_SECONDS
        }
        self.recorded_enemy_logs = [
            recorded
            for recorded in self.recorded_enemy_logs
            if recorded.time_left_s - time_left_s <= ENEMY_LOG_FRAGMENT_WINDOW_S
        ]

    @staticmethod
    def _trajectory_metrics(memory, arena_px):
        if len(memory.observed_centers) < ENEMY_CARD_CONFIRM_FRAMES:
            return None
        _, arena_y, _, arena_h = arena_px
        row_height = arena_h * ACTION_GRID.height / ACTION_GRID.rows
        if row_height <= 0:
            return None
        rows = [
            (center_y - arena_y) / row_height
            for _, _, center_y in memory.observed_centers
        ]
        deltas = [
            current - previous
            for previous, current in zip(rows, rows[1:])
        ]
        positive_steps = sum(delta > 0.15 for delta in deltas)
        reverse_rows = max((-delta for delta in deltas), default=0.0)
        first = memory.observed_centers[0]
        last = memory.observed_centers[-1]
        first_cell = ACTION_GRID.pixel_to_cell(first[1], first[2], arena_px)
        last_cell = ACTION_GRID.pixel_to_cell(last[1], last[2], arena_px)
        return rows[-1] - rows[0], positive_steps, reverse_rows, first_cell, last_cell
