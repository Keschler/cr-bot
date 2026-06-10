from __future__ import annotations

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
    MAX_ELIXIR,
    STARTING_ELIXIR_EST,
)
from cr_bot.domain.events import EnemyCardPlay
from cr_bot.features.action_space import ACTION_GRID
from cr_bot.features.global_features import card_to_id
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

from .impact_observer import ArenaFrameSample, EnemyProjectileImpactObserver, build_quadratic_trajectory_model
from .models import (
    EnemyProjectileSpellEvent,
    FRAME_CONFIRM_SPELL_CLASSES,
    ProjectileTrajectoryConfig,
    RecentArenaFrame,
    RecentEnemyClock,
    RecentSpellTargetObservation,
    SPELL_CARD_NAMES,
    TrackMemory,
)


PROJECTILE_SPELL_CONFIGS = {
    "fireball": ProjectileTrajectoryConfig(
        update_window_s=3.0,
        pre_observation_window_s=0.5,
        min_early_samples=2,
        corridor_width_norm=0.08,
    ),
    "goblin-barrel": ProjectileTrajectoryConfig(
        update_window_s=1.8,
        pre_observation_window_s=0.5,
        min_early_samples=2,
        corridor_width_norm=0.10,
    ),
}

LOG_DIRECTION_MIN_POSITIVE_STEPS = 2
LOG_DIRECTION_MIN_ROW_DELTA = 1.0
LOG_DIRECTION_MAX_REVERSE_ROWS = 0.35
LOG_DUPLICATE_WINDOW_S = 4.0


class EnemyCardTracker:
    def __init__(self, *, debug: bool = True):
        self.tracks: dict[int, TrackMemory] = {}
        self.confirmed_seen_cards: set[int] = set()
        self.detected_card_plays: list[EnemyCardPlay] = []
        self.elixir_enemy_est: float | None = None
        self.last_time_left_s: float | None = None
        self.last_update_monotonic_s: float | None = None
        self.recent_enemy_clocks: list[RecentEnemyClock] = []
        self.ally_fireball_candidates: dict[int, TrackMemory] = {}
        self.log_trajectory_candidates: dict[int, TrackMemory] = {}
        self.projectile_spell_events: list[EnemyProjectileSpellEvent] = []
        self.recent_spell_target_observations: list[RecentSpellTargetObservation] = []
        self.recent_arena_frames: list[RecentArenaFrame] = []
        self.impact_observer = EnemyProjectileImpactObserver()
        self.debug = bool(debug)

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
        frame=None,
        claimed_spell_observation_keys=None,
    ):
        clock_boxes = clock_boxes or []
        claimed_spell_observation_keys = claimed_spell_observation_keys or set()
        self._regen_elixir(time_left_s, now_s=now_s)
        self._remember_recent_enemy_clocks(clock_boxes, now_s)
        self._forget_stale_spell_target_observations(time_left_s)
        self._forget_stale_arena_frames(time_left_s)

        for match in matches:
            troop = match.troop
            if DIRECT_UNIT_TO_CARD.get(troop.class_name) == "log":
                self._remember_log_trajectory_candidate(
                    troop,
                    time_left_s=time_left_s,
                    now_s=now_s,
                    own_actions=own_actions or [],
                    arena_px=arena_px,
                )
                continue
            if troop.team != "enemy":
                self._remember_ally_fireball_candidate(
                    troop,
                    time_left_s=time_left_s,
                    now_s=now_s,
                    own_actions=own_actions or [],
                    arena_px=arena_px,
                )
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
                center_x=troop.center_x,
                center_y=troop.center_y,
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
            if memory.clock_confirmed or memory.frame_confirmed:
                self._maybe_record_play(
                    memory,
                    time_left_s,
                    own_actions=own_actions,
                    arena_px=arena_px,
                )
            else:
                self._debug_waiting(memory)

        self._update_projectile_events_from_matches(
            matches,
            time_left_s=time_left_s,
            own_actions=own_actions or [],
            arena_px=arena_px,
        )
        self._remember_recent_arena_frame(time_left_s, frame=frame, arena_px=arena_px)
        self._remember_spell_target_observations(time_left_s, arena_px=arena_px)
        self._reconcile_projectile_spell_events(
            time_left_s,
            arena_px=arena_px,
            claimed_spell_observation_keys=claimed_spell_observation_keys,
        )
        self._drop_stale_tracks(time_left_s)
        self._drop_stale_ally_fireball_candidates(time_left_s)
        self._drop_stale_log_trajectory_candidates(time_left_s)

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
            if clock["confidence"] < 0.5 or clock["team"] != "enemy":
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
                memory, troop, source="current",
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
                memory, troop, source="current",
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
                memory, troop, source="current",
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
            memory, troop, source="current",
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
                    memory, troop, source="current",
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
                    memory, troop, source="current",
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
            elif troop.class_name in FRAME_CONFIRM_SPELL_CLASSES | FRAME_CONFIRM_TROOPS:
                memory.last_clock_reject_reason = f"class {troop.class_name} uses frame confirmation"
            elif now_s > memory.first_seen_now_s + ENEMY_RECENT_CLOCK_CONFIRM_SECONDS:
                memory.last_clock_reject_reason = "remembered-clock window expired"
            return False

        saw_recent_clock = False
        for clock in self.recent_enemy_clocks:
            if clock.seen_at_s is None or now_s - clock.seen_at_s > ENEMY_RECENT_CLOCK_CONFIRM_SECONDS:
                continue
            saw_recent_clock = True
            reject_reason = self._clock_troop_reject_reason(clock.center_x, clock.center_y, troop)
            if reject_reason is not None:
                memory.last_clock_reject_reason = reject_reason
                self._debug_clock_candidate(
                    memory, troop, source="recent",
                    clock_center_x=clock.center_x, clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id, status="rejected",
                    reject_reason=reject_reason,
                    consumed_by_track_id=clock.consumed_by_track_id,
                )
                continue
            reject_reason = self._clock_claim_reject_reason(memory, troop)
            if reject_reason is not None:
                memory.last_clock_reject_reason = reject_reason
                self._debug_clock_candidate(
                    memory, troop, source="recent",
                    clock_center_x=clock.center_x, clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id, status="rejected",
                    reject_reason=reject_reason,
                    consumed_by_track_id=clock.consumed_by_track_id,
                )
                continue
            consumed_by = clock.consumed_by_track_id
            if consumed_by is not None and consumed_by != memory.track_id:
                reject_reason = f"enemy clock already consumed by track {consumed_by}"
                memory.last_clock_reject_reason = reject_reason
                self._debug_clock_candidate(
                    memory, troop, source="recent",
                    clock_center_x=clock.center_x, clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id, status="consumed",
                    reject_reason=reject_reason,
                    consumed_by_track_id=consumed_by,
                )
                continue
            if self._claim_clock(memory, clock):
                self._debug_clock_candidate(
                    memory, troop, source="recent",
                    clock_center_x=clock.center_x, clock_center_y=clock.center_y,
                    clock_track_id=clock.track_id, status="accepted",
                    consumed_by_track_id=clock.consumed_by_track_id,
                )
                return True
        if memory.last_clock_reject_reason is None:
            memory.last_clock_reject_reason = (
                "no recent enemy clock box" if not saw_recent_clock else "enemy clock already consumed"
            )
        return False

    def _clock_claim_reject_reason(self, memory, troop):
        if memory.seen_frames <= 1 and troop.confidence < ENEMY_CLOCK_FIRST_SEEN_MIN_CONF:
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
        self, memory, troop, *, source, clock_center_x, clock_center_y, status,
        clock_team=None, clock_track_id=None, clock_confidence=None,
        reject_reason=None, consumed_by_track_id=None,
    ):
        troop_center_x = troop.center_x
        troop_center_y = troop.center_y
        dx = abs(clock_center_x - troop_center_x)
        dy = clock_center_y - troop_center_y
        clock_conf = f"{clock_confidence:.3f}" if clock_confidence is not None else "-"
        self._debug(
            f"clock candidate track={memory.track_id} "
            f"class={memory.best_class or troop.class_name} "
            f"source={source} status={status} "
            f"troop_center=({troop_center_x:.1f},{troop_center_y:.1f}) "
            f"clock_center=({clock_center_x:.1f},{clock_center_y:.1f}) "
            f"clock_team={clock_team or '-'} "
            f"clock_track={clock_track_id if clock_track_id is not None else '-'} "
            f"clock_conf={clock_conf} dx={dx:.1f} dy={dy:.1f} "
            f"reject={reject_reason or '-'} "
            f"consumed_by={consumed_by_track_id if consumed_by_track_id is not None else '-'}"
        )

    def _should_frame_confirm(self, memory):
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
        if self._is_recent_own_spell_duplicate(card_name, time_left_s, memory, own_actions or [], arena_px):
            self._debug(
                f"suppress enemy {card_name} track={memory.track_id}: "
                "matches recent own spell"
            )
            memory.counted_as_card = True
            return
        if self._should_absorb_into_existing_projectile_event(
            card_name,
            memory,
            time_left_s=time_left_s,
            own_actions=own_actions or [],
            arena_px=arena_px,
        ):
            self._debug(
                f"suppress enemy {card_name} track={memory.track_id}: "
                "fits active projectile continuation"
            )
            memory.counted_as_card = True
            return
        if self._is_recent_duplicate_play(card_name, time_left_s, memory, arena_px):
            self._debug(
                f"suppress enemy {card_name} track={memory.track_id}: "
                "recent duplicate enemy play"
            )
            memory.counted_as_card = True
            return

        cost = CARD_METADATA[card_name]["elixir_cost"]
        card_id = card_to_id(card_name)
        cell = self._memory_cell(memory, arena_px)

        self.detected_card_plays.append(
            EnemyCardPlay(
                event_id=f"{card_name}_{memory.track_id}_{len(self.detected_card_plays) + 1:06d}",
                time_left_s=time_left_s,
                total_remaining_s=time_left_s,
                video_time_s=memory.first_seen_now_s,
                card=card_name,
                cost=cost,
                track_id=memory.track_id,
                cell=cell,
                clock_confirmed=memory.clock_confirmed,
                frame_confirmed=memory.frame_confirmed,
                avg_confidence=memory.avg_confidence,
                team_ratio=memory.best_team_ratio,
                best_class=memory.best_class,
                class_votes=dict(memory.class_votes),
                is_spell=card_name in SPELL_CARD_NAMES,
                overtime=time_left_s <= 120.0,
                discard_reason=None,
            )
        )
        self._debug(
            f"recorded enemy play card={card_name} track={memory.track_id} "
            f"time_left={time_left_s} cell={cell} "
            f"clock={memory.clock_confirmed} frame={memory.frame_confirmed}"
        )
        self._register_projectile_spell_event(
            self.detected_card_plays[-1],
            memory,
        )
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)
        self.elixir_enemy_est = max(0.0, self.elixir_enemy_est - cost)
        memory.counted_as_card = True

    def _remember_ally_fireball_candidate(
        self,
        troop,
        *,
        time_left_s,
        now_s,
        own_actions,
        arena_px,
    ):
        if troop.team != "ally" or DIRECT_UNIT_TO_CARD.get(troop.class_name) != "fireball":
            return
        track_id = getattr(troop, "track_id", None)
        if track_id is None:
            return
        memory = self.ally_fireball_candidates.get(track_id)
        if memory is None:
            memory = TrackMemory(
                track_id=track_id,
                first_seen_time=time_left_s,
                last_seen_time=time_left_s,
                first_seen_now_s=now_s,
            )
            self.ally_fireball_candidates[track_id] = memory
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
        if memory.counted_as_card or not self._should_frame_confirm(memory):
            return
        if self._ally_fireball_is_explained_by_own_action(memory, own_actions, arena_px):
            self._debug(
                f"suppress ally-labelled fireball track={track_id}: "
                "matches recent own fireball target"
            )
            memory.counted_as_card = True
            return
        if self._should_absorb_into_existing_projectile_event(
            "fireball",
            memory,
            time_left_s=time_left_s,
            own_actions=own_actions,
            arena_px=arena_px,
        ):
            self._debug(
                f"suppress ally-labelled fireball track={track_id}: "
                "fits active enemy projectile continuation"
            )
            memory.counted_as_card = True
            return
        self._record_ally_labelled_enemy_fireball(memory, time_left_s, arena_px)

    def _ally_fireball_is_explained_by_own_action(self, memory, own_actions, arena_px):
        if arena_px is None:
            return False
        candidate_cell = ACTION_GRID.pixel_to_cell(memory.center_x, memory.center_y, arena_px)
        if candidate_cell is None:
            return False
        for action in reversed(own_actions):
            if action.get("card") != "fireball":
                continue
            elapsed_s = action.get("time_left_s", memory.last_seen_time) - memory.last_seen_time
            if elapsed_s < 0:
                continue
            if elapsed_s > ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S:
                break
            own_cell = action.get("cell")
            if own_cell is None:
                continue
            if max(
                abs(candidate_cell[0] - own_cell[0]),
                abs(candidate_cell[1] - own_cell[1]),
            ) <= ENEMY_SPELL_DISTINCT_CELL_DISTANCE:
                return True
        return False

    def _record_ally_labelled_enemy_fireball(self, memory, time_left_s, arena_px):
        card_name = "fireball"
        cost = CARD_METADATA[card_name]["elixir_cost"]
        play = EnemyCardPlay(
            event_id=f"{card_name}_{memory.track_id}_{len(self.detected_card_plays) + 1:06d}",
            time_left_s=time_left_s,
            total_remaining_s=time_left_s,
            video_time_s=memory.first_seen_now_s,
            card=card_name,
            cost=cost,
            track_id=memory.track_id,
            cell=self._memory_cell(memory, arena_px),
            clock_confirmed=False,
            frame_confirmed=True,
            avg_confidence=memory.avg_confidence,
            team_ratio=memory.best_team_ratio,
            best_class=memory.best_class,
            class_votes=dict(memory.class_votes),
            is_spell=True,
            overtime=time_left_s <= 120.0,
            discard_reason=None,
        )
        self.detected_card_plays.append(play)
        self._register_projectile_spell_event(play, memory)
        card_id = card_to_id(card_name)
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)
        self.elixir_enemy_est = max(0.0, self.elixir_enemy_est - cost)
        memory.counted_as_card = True
        self._debug(
            f"recorded enemy play from ally-labelled fireball track={memory.track_id} "
            f"time_left={time_left_s} cell={play.cell}"
        )

    def _drop_stale_ally_fireball_candidates(self, time_left_s):
        self.ally_fireball_candidates = {
            track_id: memory
            for track_id, memory in self.ally_fireball_candidates.items()
            if memory.last_seen_time - time_left_s <= ENEMY_CARD_STALE_AFTER_SECONDS
        }

    def _remember_log_trajectory_candidate(
        self,
        troop,
        *,
        time_left_s,
        now_s,
        own_actions,
        arena_px,
    ):
        if arena_px is None:
            return
        track_id = getattr(troop, "track_id", None)
        if track_id is None:
            return
        memory = self.log_trajectory_candidates.get(track_id)
        if memory is None:
            memory = TrackMemory(
                track_id=track_id,
                first_seen_time=time_left_s,
                last_seen_time=time_left_s,
                first_seen_now_s=now_s,
            )
            self.log_trajectory_candidates[track_id] = memory
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
        if memory.counted_as_card or memory.seen_frames < ENEMY_CARD_CONFIRM_FRAMES:
            return
        trajectory = self._log_trajectory_metrics(memory, arena_px)
        if trajectory is None:
            return
        row_delta, positive_steps, reverse_rows, first_cell, last_cell = trajectory
        self._debug(
            f"log trajectory track={track_id} teams={memory.team_votes} "
            f"first_cell={first_cell} last_cell={last_cell} "
            f"row_delta={row_delta:.2f} positive_steps={positive_steps} "
            f"max_reverse_rows={reverse_rows:.2f}"
        )
        if (
            row_delta < LOG_DIRECTION_MIN_ROW_DELTA
            or positive_steps < LOG_DIRECTION_MIN_POSITIVE_STEPS
            or reverse_rows > LOG_DIRECTION_MAX_REVERSE_ROWS
        ):
            return
        if self._is_recent_log_duplicate(time_left_s):
            self._debug(
                f"suppress direction-confirmed enemy log track={track_id}: "
                "recent duplicate enemy play"
            )
            memory.counted_as_card = True
            return
        self._record_direction_confirmed_enemy_log(memory, time_left_s, arena_px)

    def _log_trajectory_metrics(self, memory, arena_px):
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

    def _is_recent_log_duplicate(self, time_left_s):
        for play in reversed(self.detected_card_plays):
            if play.card != "log":
                continue
            elapsed_s = play.time_left_s - time_left_s
            if elapsed_s < 0:
                continue
            return elapsed_s <= LOG_DUPLICATE_WINDOW_S
        return False

    def _record_direction_confirmed_enemy_log(self, memory, time_left_s, arena_px):
        card_name = "log"
        cost = CARD_METADATA[card_name]["elixir_cost"]
        play = EnemyCardPlay(
            event_id=f"{card_name}_{memory.track_id}_{len(self.detected_card_plays) + 1:06d}",
            time_left_s=time_left_s,
            total_remaining_s=time_left_s,
            video_time_s=memory.first_seen_now_s,
            card=card_name,
            cost=cost,
            track_id=memory.track_id,
            cell=self._memory_cell(memory, arena_px),
            clock_confirmed=False,
            frame_confirmed=True,
            avg_confidence=memory.avg_confidence,
            team_ratio=memory.best_team_ratio,
            best_class=memory.best_class,
            class_votes=dict(memory.class_votes),
            is_spell=True,
            overtime=time_left_s <= 120.0,
            discard_reason=None,
        )
        self.detected_card_plays.append(play)
        card_id = card_to_id(card_name)
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)
        self.elixir_enemy_est = max(0.0, self.elixir_enemy_est - cost)
        memory.counted_as_card = True
        self._debug(
            f"recorded direction-confirmed enemy log track={memory.track_id} "
            f"teams={memory.team_votes} time_left={time_left_s} cell={play.cell}"
        )

    def _drop_stale_log_trajectory_candidates(self, time_left_s):
        self.log_trajectory_candidates = {
            track_id: memory
            for track_id, memory in self.log_trajectory_candidates.items()
            if memory.last_seen_time - time_left_s <= ENEMY_CARD_STALE_AFTER_SECONDS
        }

    def reconcile_own_actions(self, own_actions, arena_px=None):
        if not own_actions or not self.detected_card_plays:
            return
        kept_plays = []
        removed_cost = 0
        for play in self.detected_card_plays:
            memory = self.tracks.get(play.track_id)
            if self._is_recent_own_spell_duplicate(play.card, play.time_left_s, memory, own_actions, arena_px):
                removed_cost += play.cost
                if memory is not None:
                    memory.counted_as_card = True
                self.projectile_spell_events = [
                    event for event in self.projectile_spell_events
                    if event.play_event_id != play.event_id
                ]
                continue
            kept_plays.append(play)
        if len(kept_plays) == len(self.detected_card_plays):
            return
        self.detected_card_plays = kept_plays
        self.confirmed_seen_cards = {
            card_id
            for play in self.detected_card_plays
            if (card_id := card_to_id(play.card)) is not None
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
        old_card = play.card
        old_cost = play.cost
        if not self._sync_detected_play_from_memory(play, memory, play_idx=play_idx, arena_px=arena_px):
            return
        new_cost = play.cost
        self.confirmed_seen_cards = {
            card_id
            for tracked_play in self.detected_card_plays
            if (card_id := card_to_id(tracked_play.card)) is not None
        }
        self.elixir_enemy_est = min(MAX_ELIXIR, max(0.0, self.elixir_enemy_est + old_cost - new_cost))
        if play.card != old_card:
            self._debug(
                f"revised enemy play track={memory.track_id} "
                f"card={old_card} -> {play.card}"
            )

    def _sync_detected_play_from_memory(self, play, memory, *, play_idx, arena_px=None):
        updated_card = DIRECT_UNIT_TO_CARD.get(memory.best_class)
        if updated_card is None:
            return False
        old_event_id = play.event_id
        event = self._find_projectile_event(old_event_id)
        play.event_id = f"{updated_card}_{memory.track_id}_{play_idx + 1:06d}"
        play.card = updated_card
        play.cost = CARD_METADATA[updated_card]["elixir_cost"]
        if event is not None:
            event.play_event_id = play.event_id
            event.card = updated_card
            event.append_centers(memory.observed_centers)
            play.cell = self._play_cell_with_priority(play, memory, arena_px)
        else:
            play.cell = self._memory_cell(memory, arena_px)
        play.clock_confirmed = memory.clock_confirmed
        play.frame_confirmed = memory.frame_confirmed
        play.avg_confidence = memory.avg_confidence
        play.team_ratio = memory.best_team_ratio
        play.best_class = memory.best_class
        play.class_votes = dict(memory.class_votes)
        play.is_spell = updated_card in SPELL_CARD_NAMES
        return True

    def _find_detected_play_index(self, track_id):
        for idx, play in enumerate(self.detected_card_plays):
            if play.track_id == track_id:
                return idx
        return None

    def _register_projectile_spell_event(self, play, memory):
        if play.card not in PROJECTILE_SPELL_CONFIGS:
            return
        event = EnemyProjectileSpellEvent(
            play_event_id=play.event_id,
            card=play.card,
            started_at_s=play.time_left_s,
            first_track_id=memory.track_id,
            best_cell=play.cell,
        )
        event.append_centers(memory.observed_centers)
        self.projectile_spell_events.append(event)

    def _find_projectile_event(self, play_event_id):
        for event in self.projectile_spell_events:
            if event.play_event_id == play_event_id:
                return event
        return None

    def _play_cell_with_priority(self, play, memory, arena_px):
        event = self._find_projectile_event(play.event_id)
        if event is not None and event.finalized_cell is not None:
            return event.finalized_cell
        return self._memory_cell(memory, arena_px)

    def _remember_spell_target_observations(self, time_left_s, *, arena_px):
        if arena_px is None:
            return
        samples = [
            ArenaFrameSample(time_left_s=sample.time_left_s, arena_bgr=sample.arena_bgr)
            for sample in self.recent_arena_frames
        ]
        for event in self.projectile_spell_events:
            if event.finalized:
                continue
            config = PROJECTILE_SPELL_CONFIGS.get(event.card)
            if (
                config is None
                or event.card == "fireball"
                or not self.impact_observer.supports(event.card)
            ):
                continue
            for idx, sample in enumerate(samples):
                delta_s = event.started_at_s - sample.time_left_s
                if delta_s < -config.pre_observation_window_s or delta_s > config.update_window_s:
                    continue
                previous_sample = samples[idx - 1] if idx > 0 else None
                observation, debug_info = self.impact_observer.inspect_event_impact(
                    card=event.card,
                    event=event,
                    current_sample=sample,
                    previous_sample=previous_sample,
                    arena_px=arena_px,
                    config=config,
                )
                self._debug_projectile_impact_observation(event, sample, debug_info)
                if observation is None:
                    continue
                if any(existing.key == observation.key for existing in self.recent_spell_target_observations):
                    continue
                self.recent_spell_target_observations.append(observation)

    def _forget_stale_spell_target_observations(self, time_left_s):
        max_window_s = max(
            (config.update_window_s for config in PROJECTILE_SPELL_CONFIGS.values()),
            default=0.0,
        )
        self.recent_spell_target_observations = [
            observation
            for observation in self.recent_spell_target_observations
            if observation.time_left_s - time_left_s <= max_window_s + 0.5
        ]

    def _remember_recent_arena_frame(self, time_left_s, *, frame, arena_px):
        if frame is None or arena_px is None:
            return
        arena_x, arena_y, arena_w, arena_h = [int(round(value)) for value in arena_px]
        if arena_w <= 0 or arena_h <= 0:
            return
        if any(abs(sample.time_left_s - time_left_s) < 1e-6 for sample in self.recent_arena_frames):
            return
        arena_crop = frame[arena_y:arena_y + arena_h, arena_x:arena_x + arena_w]
        if arena_crop.size == 0:
            return
        self.recent_arena_frames.append(
            RecentArenaFrame(
                time_left_s=time_left_s,
                arena_bgr=arena_crop.copy(),
            )
        )
        self.recent_arena_frames.sort(key=lambda sample: sample.time_left_s, reverse=True)
        del self.recent_arena_frames[12:]

    def _forget_stale_arena_frames(self, time_left_s):
        max_window_s = max(
            (
                config.update_window_s + config.pre_observation_window_s
                for config in PROJECTILE_SPELL_CONFIGS.values()
            ),
            default=0.0,
        )
        self.recent_arena_frames = [
            sample
            for sample in self.recent_arena_frames
            if sample.time_left_s - time_left_s <= max_window_s + 0.5
        ]

    def _reconcile_projectile_spell_events(
        self,
        time_left_s,
        *,
        arena_px,
        claimed_spell_observation_keys,
    ):
        if arena_px is None:
            return
        for event in sorted(self.projectile_spell_events, key=lambda item: item.started_at_s, reverse=True):
            config = PROJECTILE_SPELL_CONFIGS.get(event.card)
            assert config, "projectile config must not be empty"
            if event.card == "fireball":
                continue
            candidates = []
            for observation in self.recent_spell_target_observations:
                if observation.card != event.card or observation.cell is None:
                    continue
                if observation.key in claimed_spell_observation_keys:
                    continue
                if observation.claimed_by_event_id not in (None, event.play_event_id):
                    continue
                delta_s = event.started_at_s - observation.time_left_s
                if delta_s < -config.pre_observation_window_s or delta_s > config.update_window_s:
                    continue
                trajectory_metrics = self._trajectory_metrics(event, observation, arena_px, config)
                if trajectory_metrics is None:
                    continue
                lateral_error = trajectory_metrics[0]
                candidates.append(
                    (
                        delta_s < 0,
                        lateral_error,
                        self._observation_phase_rank(observation.phase),
                        abs(delta_s),
                        -observation.quality,
                        observation,
                    )
                )
            if not candidates:
                continue
            chosen_score = min(candidates, key=lambda item: item[:5])
            chosen = chosen_score[5]
            score_tuple = chosen_score[:5]
            if event.best_observation_score is not None and score_tuple >= event.best_observation_score:
                continue
            if event.claimed_observation_key is not None and event.claimed_observation_key != chosen.key:
                previous = self._find_spell_target_observation(event.claimed_observation_key)
                if previous is not None and previous.claimed_by_event_id == event.play_event_id:
                    previous.claimed_by_event_id = None
            chosen.claimed_by_event_id = event.play_event_id
            event.finalized = True
            event.finalized_cell = chosen.cell
            event.best_cell = chosen.cell
            event.claimed_observation_key = chosen.key
            event.best_observation_score = score_tuple
            play = self._find_detected_play_by_event_id(event.play_event_id)
            if play is not None:
                play.cell = chosen.cell
            self._debug(
                f"finalized enemy projectile card={event.card} "
                f"track={event.first_track_id} cell={chosen.cell} "
                f"obs_phase={chosen.phase} dt={event.started_at_s - chosen.time_left_s:.2f}"
            )

    def _find_detected_play_by_event_id(self, play_event_id):
        for play in self.detected_card_plays:
            if play.event_id == play_event_id:
                return play
        return None

    def _find_spell_target_observation(self, observation_key):
        for observation in self.recent_spell_target_observations:
            if observation.key == observation_key:
                return observation
        return None

    def _observation_phase_rank(self, phase):
        if phase == "impact":
            return 0
        if phase == "aim":
            return 1
        return 2

    def _debug_projectile_impact_observation(self, event, sample, debug_info):
        if not self.debug or event.card != "fireball":
            return
        best_cell = debug_info.best_cell if debug_info.best_cell is not None else "none"
        best_score = "-" if debug_info.best_score is None else f"{debug_info.best_score:.3f}"
        status = "accepted" if debug_info.emitted else f"rejected:{debug_info.reject_reason}"
        self._debug(
            f"impact obs card={event.card} event={event.play_event_id} "
            f"event_t={event.started_at_s:.2f} sample_t={sample.time_left_s:.2f} "
            f"candidates={debug_info.candidate_count} best_cell={best_cell} "
            f"best_score={best_score} status={status}"
        )

    def _update_projectile_events_from_matches(self, matches, *, time_left_s, own_actions, arena_px):
        if arena_px is None:
            return
        own_fireball_actions = self._recent_own_projectile_actions("fireball", own_actions)
        for match in matches:
            troop = match.troop
            if DIRECT_UNIT_TO_CARD.get(troop.class_name) != "fireball":
                continue
            event, score = self._best_enemy_projectile_event_for_detection(
                card="fireball",
                time_left_s=time_left_s,
                center_x=troop.center_x,
                center_y=troop.center_y,
                arena_px=arena_px,
                own_fireball_actions=own_fireball_actions,
                include_finalized=True,
            )
            if event is None:
                event = self._recent_unclaimed_fireball_event(
                    time_left_s=time_left_s,
                    center_x=troop.center_x,
                    center_y=troop.center_y,
                    arena_px=arena_px,
                    own_fireball_actions=own_fireball_actions,
                )
                score = 0.0 if event is not None else None
            if event is None:
                continue
            self._assign_projectile_detection_to_event(
                event,
                time_left_s=time_left_s,
                center_x=troop.center_x,
                center_y=troop.center_y,
                arena_px=arena_px,
                source_team=troop.team,
                source_track_id=getattr(troop, "track_id", None),
                score=score,
            )

    def _recent_own_projectile_actions(self, card_name, own_actions):
        relevant = []
        config = PROJECTILE_SPELL_CONFIGS.get(card_name)
        if config is None:
            return relevant
        for action in reversed(own_actions):
            if action.get("card") != card_name:
                continue
            if action.get("cell") is None:
                continue
            relevant.append(action)
        return relevant

    def _should_absorb_into_existing_projectile_event(
        self,
        card_name,
        memory,
        *,
        time_left_s,
        own_actions,
        arena_px,
    ):
        if card_name != "fireball" or memory.center_x is None or memory.center_y is None:
            return False
        own_fireball_actions = self._recent_own_projectile_actions(card_name, own_actions)
        event, score = self._best_enemy_projectile_event_for_detection(
            card=card_name,
            time_left_s=time_left_s,
            center_x=memory.center_x,
            center_y=memory.center_y,
            arena_px=arena_px,
            own_fireball_actions=own_fireball_actions,
            include_finalized=True,
        )
        if event is None:
            event = self._recent_unclaimed_fireball_event(
                time_left_s=time_left_s,
                center_x=memory.center_x,
                center_y=memory.center_y,
                arena_px=arena_px,
                own_fireball_actions=own_fireball_actions,
            )
            score = 0.0 if event is not None else None
        if event is None:
            return False
        self._assign_projectile_detection_to_event(
            event,
            time_left_s=time_left_s,
            center_x=memory.center_x,
            center_y=memory.center_y,
            arena_px=arena_px,
            source_team=memory.best_team,
            source_track_id=memory.track_id,
            score=score,
        )
        return True

    def _recent_unclaimed_fireball_event(
        self,
        *,
        time_left_s,
        center_x,
        center_y,
        arena_px,
        own_fireball_actions,
    ):
        config = PROJECTILE_SPELL_CONFIGS["fireball"]
        own_score = self._best_own_projectile_score(
            card="fireball",
            time_left_s=time_left_s,
            center_x=center_x,
            center_y=center_y,
            arena_px=arena_px,
            own_actions=own_fireball_actions,
        )
        if own_score is not None and own_score <= config.corridor_width_norm:
            return None
        candidates = [
            event
            for event in self.projectile_spell_events
            if event.card == "fireball"
            and -config.pre_observation_window_s
            <= event.started_at_s - time_left_s
            <= config.update_window_s
        ]
        if not candidates:
            return None
        arena_x, _, arena_w, _ = arena_px
        cand_x = (center_x - arena_x) / arena_w
        plausible = [
            event
            for event in candidates
            if event.observed_centers
            and min(
                abs(cand_x - ((observed_x - arena_x) / arena_w))
                for _, observed_x, _ in event.observed_centers
            ) <= 0.32
        ]
        if not plausible:
            return None
        return min(plausible, key=lambda event: event.started_at_s - time_left_s)

    def _best_enemy_projectile_event_for_detection(
        self,
        *,
        card,
        time_left_s,
        center_x,
        center_y,
        arena_px,
        own_fireball_actions,
        include_finalized,
    ):
        candidates = []
        own_best_score = self._best_own_projectile_score(
            card=card,
            time_left_s=time_left_s,
            center_x=center_x,
            center_y=center_y,
            arena_px=arena_px,
            own_actions=own_fireball_actions,
        )
        for event in self.projectile_spell_events:
            if event.card != card:
                continue
            if event.finalized and not include_finalized:
                continue
            trajectory = self._trajectory_score_for_detection(
                event,
                time_left_s=time_left_s,
                center_x=center_x,
                center_y=center_y,
                arena_px=arena_px,
            )
            if trajectory is None:
                continue
            if own_best_score is not None and own_best_score <= trajectory[0]:
                continue
            candidates.append((trajectory[0], event))
        if not candidates:
            return None, None
        best_score, best_event = min(candidates, key=lambda item: item[0])
        return best_event, best_score

    def _best_own_projectile_score(self, *, card, time_left_s, center_x, center_y, arena_px, own_actions):
        if card != "fireball":
            return None
        config = PROJECTILE_SPELL_CONFIGS.get(card)
        if config is None:
            return None
        arena_x, arena_y, arena_w, arena_h = arena_px
        cand_x = (center_x - arena_x) / arena_w
        cand_y = (center_y - arena_y) / arena_h
        candidate_cell = ACTION_GRID.pixel_to_cell(center_x, center_y, arena_px)
        best = None
        for action in own_actions:
            target_cell = action.get("cell")
            action_time = action.get("time_left_s")
            if target_cell is None or action_time is None:
                continue
            delta_s = action_time - time_left_s
            if delta_s < -config.pre_observation_window_s or delta_s > config.update_window_s:
                continue
            if candidate_cell is not None and max(
                abs(candidate_cell[0] - target_cell[0]),
                abs(candidate_cell[1] - target_cell[1]),
            ) <= 1:
                return 0.0
            target_x_px, target_y_px = ACTION_GRID.cell_to_pixel_center(*target_cell, arena_px)
            target_x = (target_x_px - arena_x) / arena_w
            target_y = (target_y_px - arena_y) / arena_h
            launch_x = target_x
            launch_y = 1.04
            progress = min(1.0, max(0.0, delta_s / max(config.update_window_s, 0.01)))
            expected_x = launch_x + progress * (target_x - launch_x)
            expected_y = launch_y + progress * (target_y - launch_y)
            score = abs(cand_x - expected_x) + 0.45 * abs(cand_y - expected_y)
            if best is None or score < best:
                best = score
        return best

    def _trajectory_score_for_detection(self, event, *, time_left_s, center_x, center_y, arena_px):
        config = PROJECTILE_SPELL_CONFIGS.get(event.card)
        if config is None:
            return None
        delta_s = event.started_at_s - time_left_s
        if delta_s < -config.pre_observation_window_s or delta_s > config.update_window_s:
            return None
        model = build_quadratic_trajectory_model(event, time_left_s, arena_px)
        if model is None:
            return None
        arena_x, arena_y, arena_w, arena_h = arena_px
        cand_x = (center_x - arena_x) / arena_w
        cand_y = (center_y - arena_y) / arena_h
        if cand_y + config.corridor_width_norm < model.last_y:
            return None
        dx = model.last_x - model.first_x
        dy = model.last_y - model.first_y
        if dy > 0.01 and (cand_x - model.last_x) * dx + (cand_y - model.last_y) * dy <= 0:
            return None
        predicted_x = model.predict_x(cand_y)
        lateral_error = abs(cand_x - predicted_x)
        if lateral_error > config.corridor_width_norm * 1.5:
            return None
        return (lateral_error, cand_y)

    def _assign_projectile_detection_to_event(
        self,
        event,
        *,
        time_left_s,
        center_x,
        center_y,
        arena_px,
        source_team,
        source_track_id,
        score,
    ):
        if event.last_assigned_time_left_s is not None and time_left_s > event.last_assigned_time_left_s + 1e-6:
            return
        candidate_cell = ACTION_GRID.pixel_to_cell(center_x, center_y, arena_px)
        if candidate_cell is None:
            return
        current_row = event.finalized_cell[1] if event.finalized_cell is not None else -1
        if candidate_cell[1] < current_row:
            return
        event.append_centers([(time_left_s, center_x, center_y)])
        event.finalized = True
        event.finalized_cell = candidate_cell
        event.best_cell = candidate_cell
        event.last_assigned_time_left_s = time_left_s
        play = self._find_detected_play_by_event_id(event.play_event_id)
        if play is not None:
            play.cell = candidate_cell
        self._debug(
            f"projectile continuation card={event.card} event={event.play_event_id} "
            f"track={event.first_track_id} source_track={source_track_id if source_track_id is not None else '-'} "
            f"source_team={source_team} cell={candidate_cell} score={score:.3f}"
        )

    def _trajectory_allows_observation(self, event, observation, arena_px, config):
        return self._trajectory_metrics(event, observation, arena_px, config) is not None

    def _trajectory_metrics(self, event, observation, arena_px, config):
        if observation.cell is None:
            return None
        if not event.observed_centers:
            return (0.0,)
        obs_x, obs_y = ACTION_GRID.cell_to_pixel_center(*observation.cell, arena_px)
        arena_x, arena_y, arena_w, arena_h = arena_px
        cand_x = (obs_x - arena_x) / arena_w
        cand_y = (obs_y - arena_y) / arena_h
        model = build_quadratic_trajectory_model(event, observation.time_left_s, arena_px)
        if model is None:
            return (0.0,)
        first_x = model.first_x
        first_y = model.first_y
        last_x = model.last_x
        last_y = model.last_y
        if cand_y + config.corridor_width_norm < last_y:
            return None
        if cand_y <= first_y:
            return None
        if len(event.observed_centers) < config.min_early_samples:
            return (abs(cand_x - last_x),)
        dx = last_x - first_x
        dy = last_y - first_y
        if dy <= 0.01:
            return None
        if (cand_x - last_x) * dx + (cand_y - last_y) * dy <= 0:
            return None
        predicted_x = model.predict_x(cand_y)
        lateral_error = abs(cand_x - predicted_x)
        if lateral_error > config.corridor_width_norm:
            return None
        return (lateral_error,)

    def _is_recent_duplicate_play(self, card_name, time_left_s, memory, arena_px):
        enemy_cell = self._memory_cell(memory, arena_px)
        for play in reversed(self.detected_card_plays):
            if play.card != card_name:
                continue
            elapsed_s = play.time_left_s - time_left_s
            if elapsed_s < 0:
                continue
            if elapsed_s > ENEMY_RECENT_CLOCK_DUPLICATE_WINDOW_S:
                return False
            play_cell = play.cell
            if play_cell is None or enemy_cell is None:
                return True
            if not self._has_distinct_spell_cell(play_cell, enemy_cell):
                return True
        return False

    def _is_recent_own_spell_duplicate(self, card_name, time_left_s, memory, own_actions, arena_px):
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
        if memory is None or arena_px is None:
            return None
        if memory.deploy_clock_center_x is not None and memory.deploy_clock_center_y is not None:
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
            ACTION_GRID.pixel_to_cell(memory.center_x, memory.center_y, arena_px),
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
        if self.debug:
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
            self.elixir_enemy_est = min(MAX_ELIXIR, self.elixir_enemy_est + elapsed * rate)
            self.last_update_monotonic_s = now_s
            self.last_time_left_s = time_left_s
            return
        elapsed = self.last_time_left_s - time_left_s
        if elapsed <= 0 or elapsed > 2.0:
            self.last_time_left_s = time_left_s
            return
        rate = self._elixir_rate(time_left_s)
        self.elixir_enemy_est = min(MAX_ELIXIR, self.elixir_enemy_est + elapsed * rate)
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
