from dataclasses import dataclass, field

from card_metadata import CARD_METADATA
from features.global_features import card_to_id
from trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

CONFIRM_FRAMES = 3
STALE_AFTER_SECONDS = 2.0
STARTING_ELIXIR_EST = 6
MAX_ELIXIR = 10.0
ELIXIR_PER_SECOND_NORMAL = 1.0 / 2.8
ELIXIR_PER_SECOND_DOUBLE = 1.0 / 1.4
ELIXIR_PER_SECOND_TRIPLE = 1.0 / 0.9
FRAME_CONFIRM_CLASSES = {
    "arrows",
    "barbarian-barrel",
    "clone",
    "earthquake",
    "electro-wizard",
    "fireball",
    "freeze",
    "giant-snowball",
    "goblin-barrel",
    "graveyard",
    "lightning",
    "poison",
    "rage",
    "rocket",
    "royal-delivery",
    "skeleton-king-skill",
    "tesla-evolution-shock",
    "the-log",
    "tornado",
    "zap",
}

@dataclass
class TrackMemory:
    """
    Memory for one tracked battlefield object.

    Answers: what do we know about this one tracked object?
    """
    track_id: int | None
    first_seen_time: float
    last_seen_time: float
    class_votes: dict[str, int] = field(default_factory=dict)
    team_votes: dict[str, int] = field(default_factory=dict)
    confidence_sum: float = 0.0
    seen_frames: int = 0
    clock_confirmed: bool = False
    frame_confirmed: bool = False
    counted_as_card: bool = False

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

class EnemyCardTracker:
    """
      Infers enemy card plays from tracked battlefield objects.

      Answers: which enemy cards have we inferred, and how does that affect
      enemy elixir/card history?
    """
    def __init__(self):
        self.tracks: dict[int, TrackMemory] = {}
        self.confirmed_seen_cards: set[int] = set()
        self.detected_card_plays: list[dict] = []
        self.elixir_enemy_est: float | None = None
        self.last_time_left_s: float | None = None
        self.last_update_monotonic_s: float | None = None

    def start_match(self, time_left_s, total_remaining_s, now_s=None):
        opening_elapsed = max(0.0, 180.0 - time_left_s)
        self.elixir_enemy_est = min(
            MAX_ELIXIR,
            STARTING_ELIXIR_EST + opening_elapsed * ELIXIR_PER_SECOND_NORMAL,
        )
        self.last_time_left_s = total_remaining_s
        self.last_update_monotonic_s = now_s


    def update(self, time_left_s, enemy_matches, clock_boxes=None, now_s=None):
        clock_boxes = clock_boxes or []
        self._regen_elixir(time_left_s, now_s=now_s)

        for match in enemy_matches:
            troop = match.troop
            track_id = getattr(troop, "track_id", None)

            if track_id is None:
                continue
            
            memory = self.tracks.get(track_id)
            if memory is None:
                memory = TrackMemory(
                    track_id=track_id,
                    first_seen_time=time_left_s,
                    last_seen_time=time_left_s,
                )
                self.tracks[track_id] = memory

            memory.add_observation(
                troop.class_name,
                troop.team,
                troop.confidence,
                time_left_s,
            )

            if self._has_nearby_clock(troop, clock_boxes):
                memory.clock_confirmed = True

            if self._should_frame_confirm(memory):
                memory.frame_confirmed = True

            if (memory.clock_confirmed or memory.frame_confirmed) and not memory.counted_as_card:
                self._maybe_record_play(memory, time_left_s)

        self._drop_stale_tracks(time_left_s)

    def _has_nearby_clock(self, troop, clock_boxes):
        for clock in clock_boxes:
            if clock["confidence"] < 0.5:
                continue
            if clock["team"] != "enemy":
                continue
            horizontal_gap = abs(clock["center_x"] - troop.center_x)
            vertical_gap = clock["center_y"] - troop.center_y

            if horizontal_gap <= 90 and -80 <= vertical_gap <= 180:
              return True
        return False 

    def _should_frame_confirm(self, memory):
        best_class = memory.best_class
        if best_class not in FRAME_CONFIRM_CLASSES:
            return False
        if memory.seen_frames < CONFIRM_FRAMES:
            return False
        if memory.avg_confidence < 0.65:
            return False

        best_votes = memory.class_votes[best_class]
        return best_votes / memory.seen_frames >= 0.6

    def _is_reliable_enemy_play(self, memory):
        if not memory.clock_confirmed and not memory.frame_confirmed:
            return False
        if memory.best_team != "enemy":
            return False
        if memory.frame_confirmed and memory.best_team_ratio < 0.8:
            return False
        if memory.frame_confirmed and memory.avg_confidence < 0.7:
            return False
        if memory.best_class is None:
            return False
        
        return True

    def _maybe_record_play(self, memory, time_left_s):
        if not self._is_reliable_enemy_play(memory):
            return
        
        unit_name = memory.best_class
        card_name = DIRECT_UNIT_TO_CARD.get(unit_name)

        if card_name is None:
            memory.counted_as_card = True
            return
        
        cost = CARD_METADATA[card_name]["elixir_cost"]
        card_id = card_to_id(card_name)

        self.detected_card_plays.append({
            "time_left_s": time_left_s,
            "card": card_name,
            "cost": cost,
            "track_id": memory.track_id,
        })
        
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)

        self.elixir_enemy_est = max(0.0, self.elixir_enemy_est - cost)
        memory.counted_as_card = True
        
    def _regen_elixir(self, time_left_s, now_s=None):
        if self.last_time_left_s is None:
            self.last_time_left_s = time_left_s
            self.last_update_monotonic_s = now_s
            return

        if now_s is not None:
            if self.last_update_monotonic_s is None:
                self.last_update_monotonic_s = now_s
                self.last_time_left_s = time_left_s
                return

            elapsed = now_s - self.last_update_monotonic_s
            if elapsed <= 0 or elapsed > 2.0:
                self.last_update_monotonic_s = now_s
                self.last_time_left_s = time_left_s
                return

            rate = self._elixir_rate(time_left_s)
            self.elixir_enemy_est = min(
                MAX_ELIXIR,
                self.elixir_enemy_est + elapsed * rate,
            )
            self.last_update_monotonic_s = now_s
            self.last_time_left_s = time_left_s
            return
        
        elapsed = self.last_time_left_s - time_left_s
        if elapsed <= 0 or elapsed > 2.0:
            self.last_time_left_s = time_left_s
            return
        
        rate = self._elixir_rate(time_left_s)
        self.elixir_enemy_est = min(
            MAX_ELIXIR,
            self.elixir_enemy_est + elapsed * rate,
        )
        self.last_time_left_s = time_left_s

    def _elixir_rate(self, time_left_s):
        if time_left_s <= 60:
            return ELIXIR_PER_SECOND_TRIPLE
        if time_left_s <= 180:
            return ELIXIR_PER_SECOND_DOUBLE
        return ELIXIR_PER_SECOND_NORMAL

    def _drop_stale_tracks(self, time_left_s):
        stale_ids = [
            track_id
            for track_id, memory in self.tracks.items()
            if memory.last_seen_time - time_left_s > STALE_AFTER_SECONDS
        ]

        for track_id in stale_ids:
            del self.tracks[track_id]
