from dataclasses import dataclass, field

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.constants import (
    ENEMY_CLOCK_FIRST_SEEN_MIN_CONF,
    ELIXIR_PER_SECOND_DOUBLE,
    ELIXIR_PER_SECOND_NORMAL,
    ELIXIR_PER_SECOND_TRIPLE,
    ENEMY_CARD_CONFIRM_FRAMES,
    ENEMY_RECENT_CLOCK_CONFIRM_SECONDS,
    ENEMY_RECENT_CLOCK_DUPLICATE_WINDOW_S,
    ENEMY_SPELL_DISTINCT_CELL_DISTANCE,
    ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S,
    ENEMY_CARD_STALE_AFTER_SECONDS,
    FRAME_CONFIRM_TROOPS,
    FRAME_CONFIRM_MOVING_SPELLS,
    FRAME_CONFIRM_STATIONARY_SPELLS,
    MAX_ELIXIR,
    STARTING_ELIXIR_EST,
)
from cr_bot.features.action_space import ACTION_GRID
from cr_bot.features.global_features import card_to_id
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
    """
    Memory for one tracked battlefield object.

    Answers: what do we know about this one tracked object?
    """
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
        self.recent_enemy_clocks: list[RecentEnemyClock] = []

    def start_match(self, time_left_s, total_remaining_s, now_s=None):
        opening_elapsed = max(0.0, 180.0 - time_left_s)
        self.elixir_enemy_est = min(
            MAX_ELIXIR,
            STARTING_ELIXIR_EST + opening_elapsed * ELIXIR_PER_SECOND_NORMAL,
        )
        self.last_time_left_s = total_remaining_s
        self.last_update_monotonic_s = now_s


    def update(
        self,
        time_left_s,
        matches,
        clock_boxes=None,
        now_s=None,
        own_actions=None,
        arena_px=None,
    ):
        clock_boxes = clock_boxes or []
        self._regen_elixir(time_left_s, now_s=now_s)
        self._remember_recent_enemy_clocks(clock_boxes, now_s)

        for match in matches:
            troop = match.troop
            if troop.team != "enemy":
                self._debug(
                    f"skip class={troop.class_name} team={troop.team} "
                    f"conf={troop.confidence:.3f}: not enemy"
                )
                continue

            track_id = getattr(troop, "track_id", None)

            if track_id is None:
                self._debug(
                    f"skip class={troop.class_name} team={troop.team} "
                    f"conf={troop.confidence:.3f}: no track_id"
                )
                continue
            
            memory = self.tracks.get(track_id)
            if memory is None:
                memory = TrackMemory(
                    track_id=track_id,
                    first_seen_time=time_left_s,
                    last_seen_time=time_left_s,
                    first_seen_now_s=now_s,
                )
                self.tracks[track_id] = memory

            memory.add_observation(
                troop.class_name,
                troop.team,
                troop.confidence,
                time_left_s,
            )
            memory.center_x = troop.center_x
            memory.center_y = troop.center_y

            if (
                not memory.clock_confirmed
                and self._has_nearby_clock(memory, troop, clock_boxes, now_s=now_s)
            ):
                memory.clock_confirmed = True
                self._debug(
                    f"clock confirmed track={memory.track_id} class={memory.best_class} "
                    f"clock_center=({memory.deploy_clock_center_x:.1f}, "
                    f"{memory.deploy_clock_center_y:.1f})"
                )

            if self._should_frame_confirm(memory):
                memory.frame_confirmed = True
                self._debug(
                    f"frame confirmed track={memory.track_id} class={memory.best_class} "
                    f"seen={memory.seen_frames} avg_conf={memory.avg_confidence:.3f} "
                    f"team_ratio={memory.best_team_ratio:.2f}"
                )

            if memory.counted_as_card:
                self._maybe_revise_recorded_play(memory, arena_px=arena_px)
                continue
            elif memory.clock_confirmed or memory.frame_confirmed:
                self._maybe_record_play(
                    memory,
                    time_left_s,
                    own_actions=own_actions,
                    arena_px=arena_px,
                )
            else:
                self._debug_waiting(memory)

        self._drop_stale_tracks(time_left_s)

    def _remember_recent_enemy_clocks(self, clock_boxes, now_s):
        if now_s is None:
            self.recent_enemy_clocks = []
            return

        self.recent_enemy_clocks = [
            clock
            for clock in self.recent_enemy_clocks
            if clock.seen_at_s is not None
            and now_s - clock.seen_at_s <= ENEMY_RECENT_CLOCK_CONFIRM_SECONDS
        ]

        for clock in clock_boxes:
            if clock["confidence"] < 0.5:
                continue
            if clock["team"] != "enemy":
                continue
            remembered = self._find_remembered_clock(clock)
            if remembered is None:
                self.recent_enemy_clocks.append(
                    RecentEnemyClock(
                        seen_at_s=now_s,
                        center_x=clock["center_x"],
                        center_y=clock["center_y"],
                        track_id=clock.get("track_id"),
                    )
                )
                continue
            remembered.seen_at_s = now_s
            remembered.center_x = clock["center_x"]
            remembered.center_y = clock["center_y"]
            if remembered.track_id is None:
                remembered.track_id = clock.get("track_id")

    def _find_remembered_clock(self, clock):
        track_id = clock.get("track_id")
        for remembered in self.recent_enemy_clocks:
            if track_id is not None and remembered.track_id == track_id:
                return remembered
            if (
                abs(remembered.center_x - clock["center_x"]) <= 12
                and abs(remembered.center_y - clock["center_y"]) <= 12
            ):
                return remembered
        return None

    def _claim_clock(self, memory, clock):
        if (
            clock.consumed_by_track_id is not None
            and clock.consumed_by_track_id != memory.track_id
        ):
            return False
        clock.consumed_by_track_id = memory.track_id
        memory.deploy_clock_center_x = clock.center_x
        memory.deploy_clock_center_y = clock.center_y
        return True

    def _claim_current_clock(self, memory, troop, clock):
        reject_reason = self._clock_troop_reject_reason(
            clock["center_x"],
            clock["center_y"],
            troop,
        )
        if reject_reason is not None:
            memory.last_clock_reject_reason = reject_reason
            self._debug_clock_candidate(
                memory,
                troop,
                source="current",
                clock_center_x=clock["center_x"],
                clock_center_y=clock["center_y"],
                clock_team=clock.get("team"),
                clock_track_id=clock.get("track_id"),
                clock_confidence=clock.get("confidence"),
                status="rejected",
                reject_reason=reject_reason,
            )
            return False
        reject_reason = self._clock_claim_reject_reason(memory, troop)
        if reject_reason is not None:
            memory.last_clock_reject_reason = reject_reason
            self._debug_clock_candidate(
                memory,
                troop,
                source="current",
                clock_center_x=clock["center_x"],
                clock_center_y=clock["center_y"],
                clock_team=clock.get("team"),
                clock_track_id=clock.get("track_id"),
                clock_confidence=clock.get("confidence"),
                status="rejected",
                reject_reason=reject_reason,
            )
            return False
        remembered = self._find_remembered_clock(clock)
        if remembered is None:
            remembered = RecentEnemyClock(
                seen_at_s=None,
                center_x=clock["center_x"],
                center_y=clock["center_y"],
                track_id=clock.get("track_id"),
            )
            self.recent_enemy_clocks.append(remembered)
        consumed_by = remembered.consumed_by_track_id
        if consumed_by is not None and consumed_by != memory.track_id:
            reject_reason = f"enemy clock already consumed by track {consumed_by}"
            memory.last_clock_reject_reason = reject_reason
            self._debug_clock_candidate(
                memory,
                troop,
                source="current",
                clock_center_x=clock["center_x"],
                clock_center_y=clock["center_y"],
                clock_team=clock.get("team"),
                clock_track_id=clock.get("track_id"),
                clock_confidence=clock.get("confidence"),
                status="consumed",
                reject_reason=reject_reason,
                consumed_by_track_id=consumed_by,
            )
            return False

        claimed = self._claim_clock(memory, remembered)
        self._debug_clock_candidate(
            memory,
            troop,
            source="current",
            clock_center_x=clock["center_x"],
            clock_center_y=clock["center_y"],
            clock_team=clock.get("team"),
            clock_track_id=clock.get("track_id"),
            clock_confidence=clock.get("confidence"),
            status="accepted" if claimed else "rejected",
            reject_reason=None if claimed else "enemy clock claim failed",
            consumed_by_track_id=remembered.consumed_by_track_id,
        )
        return claimed

    def _has_nearby_clock(self, memory, troop, clock_boxes, now_s=None):
        saw_enemy_clock = False
        for clock in clock_boxes:
            if clock["confidence"] < 0.5:
                self._debug_clock_candidate(
                    memory,
                    troop,
                    source="current",
                    clock_center_x=clock["center_x"],
                    clock_center_y=clock["center_y"],
                    clock_team=clock.get("team"),
                    clock_track_id=clock.get("track_id"),
                    clock_confidence=clock.get("confidence"),
                    status="skipped",
                    reject_reason=f"clock confidence {clock['confidence']:.3f} < 0.500",
                )
                continue
            if clock["team"] != "enemy":
                self._debug_clock_candidate(
                    memory,
                    troop,
                    source="current",
                    clock_center_x=clock["center_x"],
                    clock_center_y=clock["center_y"],
                    clock_team=clock.get("team"),
                    clock_track_id=clock.get("track_id"),
                    clock_confidence=clock.get("confidence"),
                    status="skipped",
                    reject_reason=f"clock team {clock['team']} != enemy",
                )
                continue
            saw_enemy_clock = True
            if self._claim_current_clock(memory, troop, clock):
                return True

        card_name = DIRECT_UNIT_TO_CARD.get(troop.class_name)
        if (
            now_s is None
            or memory.first_seen_now_s is None
            or card_name is None
            or troop.class_name in FRAME_CONFIRM_SPELL_CLASSES
            or troop.class_name in FRAME_CONFIRM_TROOPS
            or now_s > memory.first_seen_now_s + ENEMY_RECENT_CLOCK_CONFIRM_SECONDS
        ):
            if not saw_enemy_clock and memory.last_clock_reject_reason is None:
                memory.last_clock_reject_reason = "no current enemy clock box"
            elif now_s is None:
                memory.last_clock_reject_reason = "no monotonic time for remembered-clock lookup"
            elif memory.first_seen_now_s is None:
                memory.last_clock_reject_reason = "track has no first-seen monotonic time"
            elif card_name is None:
                memory.last_clock_reject_reason = f"class {troop.class_name} does not map to a card"
            elif troop.class_name in FRAME_CONFIRM_SPELL_CLASSES:
                memory.last_clock_reject_reason = f"class {troop.class_name} uses frame confirmation"
            elif troop.class_name in FRAME_CONFIRM_TROOPS:
                memory.last_clock_reject_reason = f"class {troop.class_name} uses frame confirmation"
            elif now_s > memory.first_seen_now_s + ENEMY_RECENT_CLOCK_CONFIRM_SECONDS:
                memory.last_clock_reject_reason = "remembered-clock window expired"
            return False

        saw_recent_clock = False
        for clock in self.recent_enemy_clocks:
            if clock.seen_at_s is None:
                continue
            if now_s - clock.seen_at_s > ENEMY_RECENT_CLOCK_CONFIRM_SECONDS:
                continue
            saw_recent_clock = True
            reject_reason = self._clock_troop_reject_reason(
                clock.center_x,
                clock.center_y,
                troop,
            )
            if reject_reason is not None:
                memory.last_clock_reject_reason = reject_reason
                self._debug_clock_candidate(
                    memory,
                    troop,
                    source="recent",
                    clock_center_x=clock.center_x,
                    clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id,
                    status="rejected",
                    reject_reason=reject_reason,
                    consumed_by_track_id=clock.consumed_by_track_id,
                )
                continue
            reject_reason = self._clock_claim_reject_reason(memory, troop)
            if reject_reason is not None:
                memory.last_clock_reject_reason = reject_reason
                self._debug_clock_candidate(
                    memory,
                    troop,
                    source="recent",
                    clock_center_x=clock.center_x,
                    clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id,
                    status="rejected",
                    reject_reason=reject_reason,
                    consumed_by_track_id=clock.consumed_by_track_id,
                )
                continue

            consumed_by = clock.consumed_by_track_id
            if consumed_by is not None and consumed_by != memory.track_id:
                reject_reason = f"enemy clock already consumed by track {consumed_by}"
                memory.last_clock_reject_reason = reject_reason
                self._debug_clock_candidate(
                    memory,
                    troop,
                    source="recent",
                    clock_center_x=clock.center_x,
                    clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id,
                    status="consumed",
                    reject_reason=reject_reason,
                    consumed_by_track_id=consumed_by,
                )
                continue

            if self._claim_clock(memory, clock):
                self._debug_clock_candidate(
                    memory,
                    troop,
                    source="recent",
                    clock_center_x=clock.center_x,
                    clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id,
                    status="accepted",
                    consumed_by_track_id=clock.consumed_by_track_id,
                )
                return True
        if memory.last_clock_reject_reason is None:
            memory.last_clock_reject_reason = (
                "no recent enemy clock box"
                if not saw_recent_clock
                else "enemy clock already consumed"
            )
        return False

    def _clock_matches_troop(self, clock_center_x, clock_center_y, troop):
        return self._clock_troop_reject_reason(clock_center_x, clock_center_y, troop) is None

    def _clock_claim_reject_reason(self, memory, troop):
        if (
            memory.seen_frames <= 1
            and troop.confidence < ENEMY_CLOCK_FIRST_SEEN_MIN_CONF
        ):
            return (
                f"first-seen confidence {troop.confidence:.3f} < "
                f"{ENEMY_CLOCK_FIRST_SEEN_MIN_CONF:.3f}"
            )
        return None

    def _clock_troop_reject_reason(self, clock_center_x, clock_center_y, troop):
        horizontal_gap = abs(clock_center_x - troop.center_x)
        vertical_gap = clock_center_y - troop.center_y
        if horizontal_gap > 90:
            return f"clock horizontal gap {horizontal_gap:.1f} > 90"
        if vertical_gap < 10:
            return f"clock vertical gap {vertical_gap:.1f} < 10"
        if vertical_gap > 140:
            return f"clock vertical gap {vertical_gap:.1f} > 140"
        return None 

    def _debug_clock_candidate(
        self,
        memory,
        troop,
        *,
        source,
        clock_center_x,
        clock_center_y,
        status,
        clock_team=None,
        clock_track_id=None,
        clock_confidence=None,
        reject_reason=None,
        consumed_by_track_id=None,
    ):
        troop_center_x = troop.center_x
        troop_center_y = troop.center_y
        dx = abs(clock_center_x - troop_center_x)
        dy = clock_center_y - troop_center_y
        clock_conf = (
            f"{clock_confidence:.3f}"
            if clock_confidence is not None
            else "-"
        )
        self._debug(
            f"clock candidate track={memory.track_id} "
            f"class={memory.best_class or troop.class_name} "
            f"source={source} status={status} "
            f"troop_center=({troop_center_x:.1f},{troop_center_y:.1f}) "
            f"clock_center=({clock_center_x:.1f},{clock_center_y:.1f}) "
            f"clock_team={clock_team or '-'} "
            f"clock_track={clock_track_id if clock_track_id is not None else '-'} "
            f"clock_conf={clock_conf} "
            f"dx={dx:.1f} dy={dy:.1f} "
            f"reject={reject_reason or '-'} "
            f"consumed_by={consumed_by_track_id if consumed_by_track_id is not None else '-'}"
        )

    def _should_frame_confirm(self, memory):
        """Allow frame-only confirmation for spell-like classes, special troops."""
        best_class = memory.best_class
        if best_class not in FRAME_CONFIRM_SPELL_CLASSES | FRAME_CONFIRM_TROOPS:
            return False
        if memory.seen_frames < ENEMY_CARD_CONFIRM_FRAMES:
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

    def _maybe_record_play(self, memory, time_left_s, own_actions=None, arena_px=None):
        if not self._is_reliable_enemy_play(memory):
            self._debug(
                f"not reliable track={memory.track_id} class={memory.best_class} "
                f"clock={memory.clock_confirmed} frame={memory.frame_confirmed} "
                f"team={memory.best_team} team_ratio={memory.best_team_ratio:.2f} "
                f"avg_conf={memory.avg_confidence:.3f}"
            )
            return
        
        unit_name = memory.best_class
        card_name = DIRECT_UNIT_TO_CARD.get(unit_name)

        if card_name is None:
            self._debug(
                f"suppress track={memory.track_id} class={unit_name}: "
                "DIRECT_UNIT_TO_CARD maps to None"
            )
            memory.counted_as_card = True
            return

        if self._is_recent_own_spell_duplicate(
            card_name,
            time_left_s,
            memory,
            own_actions or [],
            arena_px,
        ):
            self._debug(
                f"suppress enemy {card_name} track={memory.track_id}: "
                "matches recent own spell"
            )
            memory.counted_as_card = True
            return

        if self._is_recent_duplicate_play(
            card_name,
            time_left_s,
            memory,
            arena_px,
        ):
            self._debug(
                f"suppress enemy {card_name} track={memory.track_id}: "
                "recent duplicate enemy play"
            )
            memory.counted_as_card = True
            return
        
        cost = CARD_METADATA[card_name]["elixir_cost"]
        card_id = card_to_id(card_name)
        cell = self._memory_cell(memory, arena_px)

        self.detected_card_plays.append({
            "event_id": f"{card_name}_{memory.track_id}_{len(self.detected_card_plays) + 1:06d}",
            "time_left_s": time_left_s,
            "total_remaining_s": time_left_s,
            "video_time_s": memory.first_seen_now_s,
            "card": card_name,
            "cost": cost,
            "track_id": memory.track_id,
            "cell": cell,
            "clock_confirmed": memory.clock_confirmed,
            "frame_confirmed": memory.frame_confirmed,
            "avg_confidence": memory.avg_confidence,
            "team_ratio": memory.best_team_ratio,
            "best_class": memory.best_class,
            "class_votes": dict(memory.class_votes),
            "is_spell": card_name in SPELL_CARD_NAMES,
            "overtime": time_left_s <= 120.0,
            "discard_reason": None,
        })
        self._debug(
            f"recorded enemy play card={card_name} track={memory.track_id} "
            f"time_left={time_left_s} cell={cell} "
            f"clock={memory.clock_confirmed} frame={memory.frame_confirmed}"
        )
        
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)

        self.elixir_enemy_est = max(0.0, self.elixir_enemy_est - cost)
        memory.counted_as_card = True

    def reconcile_own_actions(self, own_actions, arena_px=None):
        if not own_actions or not self.detected_card_plays:
            return

        kept_plays = []
        removed_cost = 0
        for play in self.detected_card_plays:
            memory = self.tracks.get(play["track_id"])
            if self._is_recent_own_spell_duplicate(
                play["card"],
                play["time_left_s"],
                memory,
                own_actions,
                arena_px,
            ):
                removed_cost += play["cost"]
                if memory is not None:
                    memory.counted_as_card = True
                continue

            kept_plays.append(play)

        if len(kept_plays) == len(self.detected_card_plays):
            return

        self.detected_card_plays = kept_plays
        self.confirmed_seen_cards = {
            card_id
            for play in self.detected_card_plays
            if (card_id := card_to_id(play["card"])) is not None
        }
        self.elixir_enemy_est = min(MAX_ELIXIR, self.elixir_enemy_est + removed_cost)

    def _maybe_revise_recorded_play(self, memory, *, arena_px=None):
        assert memory.track_id is not None, "recorded track must have a track_id"

        if not self._is_reliable_enemy_play(memory):
            return

        play_idx = self._find_detected_play_index(memory.track_id)
        if play_idx is None:
            return

        play = self.detected_card_plays[play_idx]
        old_card = play["card"]
        old_cost = play["cost"]
        if not self._sync_detected_play_from_memory(
            play,
            memory,
            play_idx=play_idx,
            arena_px=arena_px,
        ):
            return
        new_cost = play["cost"]
        self.confirmed_seen_cards = {
            card_id
            for tracked_play in self.detected_card_plays
            if (card_id := card_to_id(tracked_play["card"])) is not None
        }
        self.elixir_enemy_est = min(
            MAX_ELIXIR,
            max(0.0, self.elixir_enemy_est + old_cost - new_cost),
        )

        if play["card"] != old_card:
            self._debug(
                f"revised enemy play track={memory.track_id} "
                f"card={old_card} -> {play['card']}"
            )

    def _sync_detected_play_from_memory(self, play, memory, *, play_idx, arena_px=None):
        updated_card = DIRECT_UNIT_TO_CARD.get(memory.best_class)
        if updated_card is None:
            return False

        play["event_id"] = f"{updated_card}_{memory.track_id}_{play_idx + 1:06d}"
        play["card"] = updated_card
        play["cost"] = CARD_METADATA[updated_card]["elixir_cost"]
        play["cell"] = self._memory_cell(memory, arena_px)
        play["clock_confirmed"] = memory.clock_confirmed
        play["frame_confirmed"] = memory.frame_confirmed
        play["avg_confidence"] = memory.avg_confidence
        play["team_ratio"] = memory.best_team_ratio
        play["best_class"] = memory.best_class
        play["class_votes"] = dict(memory.class_votes)
        play["is_spell"] = updated_card in SPELL_CARD_NAMES
        return True

    def _find_detected_play_index(self, track_id):
        for idx, play in enumerate(self.detected_card_plays):
            if play["track_id"] == track_id:
                return idx
        return None

    def _is_recent_duplicate_play(
        self,
        card_name,
        time_left_s,
        memory,
        arena_px,
    ):
        enemy_cell = self._memory_cell(memory, arena_px)
        for play in reversed(self.detected_card_plays):
            if play["card"] != card_name:
                continue
            elapsed_s = play["time_left_s"] - time_left_s
            if elapsed_s < 0:
                continue
            if elapsed_s > ENEMY_RECENT_CLOCK_DUPLICATE_WINDOW_S:
                return False
            play_cell = play.get("cell")
            if play_cell is None or enemy_cell is None:
                return True
            if not self._has_distinct_spell_cell(play_cell, enemy_cell):
                return True
        return False

    def _is_recent_own_spell_duplicate(
        self,
        card_name,
        time_left_s,
        memory,
        own_actions,
        arena_px,
    ):
        if card_name not in SPELL_CARD_NAMES:
            return False

        enemy_cell = self._memory_cell(memory, arena_px)
        for action in reversed(own_actions):
            if action["card"] != card_name:
                continue

            elapsed_s = action["time_left_s"] - time_left_s
            if elapsed_s < 0:
                continue
            if elapsed_s > ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S:
                return False

            own_cell = action.get("cell")
            if self._has_distinct_spell_cell(own_cell, enemy_cell):
                continue
            return True

        return False

    def _memory_cell(self, memory, arena_px):
        if memory is None:
            return None
        if arena_px is None:
            return None
        if (
            memory.deploy_clock_center_x is not None
            and memory.deploy_clock_center_y is not None
        ):
            return self._raise_cell_rows(
                ACTION_GRID.pixel_to_cell(
                    memory.deploy_clock_center_x,
                    memory.deploy_clock_center_y,
                    arena_px,
                ),
                rows=2,
            )
        if memory.center_x is None or memory.center_y is None:
            return None
        return self._raise_cell_rows(
            ACTION_GRID.pixel_to_cell(
                memory.center_x,
                memory.center_y,
                arena_px,
            ),
            rows=2,
        )

    def _raise_cell_rows(self, cell, *, rows):
        if cell is None:
            return None
        col, row = cell
        return col, max(0, row - rows)

    def _has_distinct_spell_cell(self, own_cell, enemy_cell):
        if own_cell is None or enemy_cell is None:
            return False

        dx = abs(own_cell[0] - enemy_cell[0])
        dy = abs(own_cell[1] - enemy_cell[1])
        return max(dx, dy) > ENEMY_SPELL_DISTINCT_CELL_DISTANCE

    def _debug_waiting(self, memory):
        if memory.seen_frames > ENEMY_CARD_CONFIRM_FRAMES:
            return
        frame_class = memory.best_class in FRAME_CONFIRM_SPELL_CLASSES | FRAME_CONFIRM_TROOPS
        self._debug(
            f"waiting track={memory.track_id} class={memory.best_class} "
            f"seen={memory.seen_frames} avg_conf={memory.avg_confidence:.3f} "
            f"team={memory.best_team} team_ratio={memory.best_team_ratio:.2f} "
            f"frame_class={frame_class} "
            f"clock_reject={memory.last_clock_reject_reason or 'not checked'}"
        )

    def _debug(self, message):
        print(f"[enemy_cards] {message}")
        
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
            if memory.last_seen_time - time_left_s > ENEMY_CARD_STALE_AFTER_SECONDS
        ]

        for track_id in stale_ids:
            del self.tracks[track_id]
