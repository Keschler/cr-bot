from __future__ import annotations

from collections import Counter

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.constants import (
    ENEMY_CARD_CONFIRM_FRAMES,
    ENEMY_RECENT_CLOCK_DUPLICATE_WINDOW_S,
    ENEMY_SPELL_DISTINCT_CELL_DISTANCE,
    ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S,
    ENEMY_CARD_STALE_AFTER_SECONDS,
    FRAME_CONFIRM_TROOPS,
)
from cr_bot.domain.events import EnemyCardPlay
from cr_bot.features.action_space import ACTION_GRID
from cr_bot.features.global_features import card_to_id
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

from .clocks import EnemyClockTracker
from .elixir import EnemyElixirEstimator
from .models import (
    FRAME_CONFIRM_SPELL_CLASSES,
    SPELL_CARD_NAMES,
    TrackMemory,
)
from .projectile_spells import (
    DIRECTION_OWNED_PROJECTILE_CARDS,
    ProjectileSpellTracker,
)
from .rolling_spells import RollingSpellTracker


LONG_VISIBLE_AREA_SPELLS = {
    "earthquake",
    "freeze",
    "rage",
    "void",
    "graveyard",
    "goblin-curse",
    "poison",
    "tornado",
}
LONG_VISIBLE_AREA_DUPLICATE_WINDOW_S = 1.0


class EnemyCardTracker:
    def __init__(self, *, debug: bool = True):
        self.tracks: dict[int, TrackMemory] = {}
        self.confirmed_seen_cards: set[int] = set()
        self.detected_card_plays: list[EnemyCardPlay] = []
        self.elixir = EnemyElixirEstimator()
        self.clocks = EnemyClockTracker(self._debug)
        self.rolling_spells = RollingSpellTracker(self._debug)
        self.projectiles = ProjectileSpellTracker(
            self._debug,
            self._find_detected_play_by_event_id,
        )
        self.debug = bool(debug)

    @property
    def elixir_enemy_est(self):
        return self.elixir.value

    @property
    def recent_enemy_clocks(self):
        return self.clocks.recent

    @property
    def log_trajectory_candidates(self):
        return self.rolling_spells.candidates

    @property
    def own_log_claims(self):
        return self.rolling_spells.own_claims

    @property
    def ally_fireball_candidates(self):
        return self.projectiles.ally_candidates

    @property
    def projectile_spell_events(self):
        return self.projectiles.events

    @property
    def recent_spell_target_observations(self):
        return self.projectiles.target_observations

    @property
    def recent_arena_frames(self):
        return self.projectiles.arena_frames

    def start_match(self, time_left_s, total_remaining_s, now_s=None):
        self.elixir.start_match(time_left_s, total_remaining_s, now_s=now_s)

    def update(
        self,
        time_left_s,
        matches,
        clock_boxes=None,
        now_s=None,
        own_actions=None,
        pending_own_spell_targets=None,
        arena_px=None,
        frame=None,
        claimed_spell_observation_keys=None,
    ):
        clock_boxes = clock_boxes or []
        pending_own_spell_targets = pending_own_spell_targets or []
        own_projectile_targets = [
            *list(own_actions or []),
            *pending_own_spell_targets,
        ]
        claimed_spell_observation_keys = claimed_spell_observation_keys or set()
        self.rolling_spells.remember_own_actions(own_actions or [])
        self.elixir.update(time_left_s, now_s=now_s)
        self.clocks.remember(clock_boxes, now_s)
        self.projectiles.forget_stale(time_left_s)

        for match in matches:
            troop = match.troop
            if DIRECT_UNIT_TO_CARD.get(troop.class_name) == "log":
                memory = self.rolling_spells.observe(
                    troop,
                    time_left_s=time_left_s,
                    now_s=now_s,
                    arena_px=arena_px,
                )
                if memory is not None:
                    if self.rolling_spells.is_later_fragment(
                        memory,
                        time_left_s=time_left_s,
                        arena_px=arena_px,
                    ):
                        self._debug(
                            f"suppress direction-confirmed enemy log track={memory.track_id}: "
                            "lower same-lane fragment of recent enemy log"
                        )
                        memory.counted_as_card = True
                    else:
                        self._record_direction_confirmed_enemy_log(
                            memory,
                            time_left_s,
                            arena_px,
                        )
                continue
            if troop.team != "enemy":
                track_id = getattr(troop, "track_id", None)
                existing_memory = self.tracks.get(track_id)
                if (
                    DIRECT_UNIT_TO_CARD.get(troop.class_name) == "fireball"
                    and existing_memory is not None
                    and DIRECT_UNIT_TO_CARD.get(existing_memory.best_class) != "fireball"
                ):
                    self._debug(
                        f"skip ally-labelled fireball track={track_id}: "
                        f"track already belongs to class={existing_memory.best_class}"
                    )
                    continue
                memory, action = self.projectiles.observe_ally_fireball(
                    troop,
                    time_left_s=time_left_s,
                    now_s=now_s,
                    own_actions=own_projectile_targets,
                    pending_own_spell_targets=pending_own_spell_targets,
                    arena_px=arena_px,
                    should_frame_confirm=self._should_frame_confirm,
                )
                if action == "own-action":
                    self._debug(
                        f"suppress ally-labelled fireball track={memory.track_id}: "
                        "matches recent own fireball target"
                    )
                elif action == "continuation":
                    self._debug(
                        f"suppress ally-labelled fireball track={memory.track_id}: "
                        "fits active enemy projectile continuation"
                    )
                elif action == "own-direction":
                    self._debug(
                        f"suppress fireball track={memory.track_id}: "
                        "projectile direction is toward enemy side"
                    )
                elif action == "waiting-direction":
                    self._debug(
                        f"waiting fireball track={memory.track_id}: "
                        "projectile direction is not resolved"
                    )
                elif action == "waiting-own-action":
                    self._debug(
                        f"waiting fireball track={memory.track_id}: "
                        "allowing own-action evidence to settle"
                    )
                elif action == "record":
                    self._record_ally_labelled_enemy_fireball(memory, time_left_s, arena_px)
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
            card_name = DIRECT_UNIT_TO_CARD.get(troop.class_name)
            if card_name in DIRECTION_OWNED_PROJECTILE_CARDS:
                memory.add_motion_center(
                    now_s if now_s is not None else time_left_s,
                    troop.center_x,
                    troop.center_y,
                )
            memory.center_x = troop.center_x
            memory.center_y = troop.center_y

            if (
                not memory.clock_confirmed
                and self.clocks.confirm(memory, troop, clock_boxes, now_s=now_s)
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
                    own_actions=own_projectile_targets,
                    arena_px=arena_px,
                )
            else:
                self._debug_waiting(memory)

        self.projectiles.update_from_matches(
            matches,
            time_left_s=time_left_s,
            own_actions=own_projectile_targets,
            arena_px=arena_px,
        )
        for memory in self.projectiles.confirmed_explosion_candidates(
            time_left_s,
            own_actions=own_projectile_targets,
            arena_px=arena_px,
            should_frame_confirm=self._should_frame_confirm,
        ):
            self._record_ally_labelled_enemy_fireball(memory, time_left_s, arena_px)
        self.projectiles.remember_arena_frame(time_left_s, frame=frame, arena_px=arena_px)
        self.projectiles.remember_target_observations(time_left_s, arena_px=arena_px)
        self.projectiles.reconcile(
            arena_px=arena_px,
            claimed_spell_observation_keys=claimed_spell_observation_keys,
        )
        self._drop_stale_tracks(time_left_s)
        self.projectiles.cleanup_ally_candidates(time_left_s)
        self.rolling_spells.cleanup(time_left_s)

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
        if self.projectiles.should_absorb(
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
        if card_name in DIRECTION_OWNED_PROJECTILE_CARDS:
            ownership = self.projectiles.projectile_ownership(memory, arena_px)
            if ownership == "own":
                self._debug(
                    f"suppress {card_name} track={memory.track_id}: "
                    "projectile direction is toward enemy side"
                )
                memory.counted_as_card = True
                return
            if ownership not in {"enemy", "explosion"}:
                self._debug(
                    f"waiting {card_name} track={memory.track_id}: "
                    "projectile direction is not resolved"
                )
                return
        overlapping_play = self._overlapping_long_visible_area_play(
            card_name,
            memory,
            arena_px,
        )
        if overlapping_play is not None:
            self._merge_long_visible_area_observations(
                overlapping_play,
                memory,
                arena_px,
            )
            self._debug(
                f"suppress enemy {card_name} track={memory.track_id}: "
                f"overlaps track={overlapping_play.track_id}"
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

        mirror_source = self._mirror_source_play(card_name, memory, arena_px)
        played_via = "mirror" if mirror_source is not None else None
        cost = CARD_METADATA[card_name]["elixir_cost"] + (1 if played_via == "mirror" else 0)
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
                played_via=played_via,
            )
        )
        self._debug(
            f"recorded enemy play card={card_name} track={memory.track_id} "
            f"time_left={time_left_s} cell={cell} "
            f"clock={memory.clock_confirmed} frame={memory.frame_confirmed}"
        )
        self.projectiles.register(
            self.detected_card_plays[-1],
            memory,
        )
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)
        if played_via == "mirror":
            mirror_id = card_to_id("mirror")
            if mirror_id is not None:
                self.confirmed_seen_cards.add(mirror_id)
        self.elixir.spend(cost)
        memory.counted_as_card = True

    def _record_ally_labelled_enemy_fireball(self, memory, time_left_s, arena_px):
        card_name = "fireball"
        mirror_source = self._mirror_source_play(card_name, memory, arena_px)
        played_via = "mirror" if mirror_source is not None else None
        cost = CARD_METADATA[card_name]["elixir_cost"] + (1 if played_via == "mirror" else 0)
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
            played_via=played_via,
        )
        self.detected_card_plays.append(play)
        self.projectiles.register(play, memory)
        card_id = card_to_id(card_name)
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)
        self._remember_mirror_seen(played_via)
        self.elixir.spend(cost)
        memory.counted_as_card = True
        self._debug(
            f"recorded enemy play from ally-labelled fireball track={memory.track_id} "
            f"time_left={time_left_s} cell={play.cell}"
        )

    def _record_direction_confirmed_enemy_log(self, memory, time_left_s, arena_px):
        card_name = "log"
        mirror_source = self._mirror_source_play(card_name, memory, arena_px)
        played_via = "mirror" if mirror_source is not None else None
        cost = CARD_METADATA[card_name]["elixir_cost"] + (1 if played_via == "mirror" else 0)
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
            played_via=played_via,
        )
        self.detected_card_plays.append(play)
        self.rolling_spells.remember_enemy_log(
            memory,
            time_left_s=time_left_s,
            arena_px=arena_px,
        )
        card_id = card_to_id(card_name)
        if card_id is not None:
            self.confirmed_seen_cards.add(card_id)
        self._remember_mirror_seen(played_via)
        self.elixir.spend(cost)
        memory.counted_as_card = True
        self._debug(
            f"recorded direction-confirmed enemy log track={memory.track_id} "
            f"teams={memory.team_votes} time_left={time_left_s} cell={play.cell}"
        )

    def reconcile_own_actions(self, own_actions, arena_px=None):
        self.rolling_spells.remember_own_actions(own_actions or [])
        self.rolling_spells.assign_own_claims(arena_px)
        if not own_actions or not self.detected_card_plays:
            return
        kept_plays = []
        removed_cost = 0
        for play in self.detected_card_plays:
            memory = self.tracks.get(play.track_id)
            if memory is None and play.card == "fireball":
                memory = self.projectiles.ally_candidates.get(play.track_id)
            log_memory = self.rolling_spells.candidates.get(play.track_id)
            matches_own_log = (
                play.card == "log"
                and log_memory is not None
                and self.rolling_spells.matches_own_claim(log_memory, arena_px)
            )
            resolved_as_own_fireball = (
                play.card == "fireball"
                and memory is not None
                and self.projectiles.fireball_ownership(memory, arena_px) == "own"
            )
            matches_own_spell = self._is_recent_own_spell_duplicate(
                play.card,
                play.time_left_s,
                memory,
                own_actions,
                arena_px,
            )
            if matches_own_log or resolved_as_own_fireball or matches_own_spell:
                removed_cost += play.cost
                if memory is not None:
                    memory.counted_as_card = True
                if log_memory is not None:
                    log_memory.counted_as_card = True
                self.projectiles.remove_event(play.event_id)
                if resolved_as_own_fireball:
                    self._debug(
                        f"removed enemy fireball track={play.track_id}: "
                        "later projectile direction resolved as own"
                    )
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
        if any(play.played_via == "mirror" for play in self.detected_card_plays):
            self._remember_mirror_seen("mirror")
        self.elixir.refund(removed_cost)

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
        if any(tracked_play.played_via == "mirror" for tracked_play in self.detected_card_plays):
            self._remember_mirror_seen("mirror")
        if new_cost > old_cost:
            self.elixir.spend(new_cost - old_cost)
        else:
            self.elixir.refund(old_cost - new_cost)
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
        event = self.projectiles.find_event(old_event_id)
        play.event_id = f"{updated_card}_{memory.track_id}_{play_idx + 1:06d}"
        play.card = updated_card
        play.cost = CARD_METADATA[updated_card]["elixir_cost"]
        if event is not None:
            event.play_event_id = play.event_id
            event.card = updated_card
            event.append_centers(memory.observed_centers)
            play.cell = self.projectiles.finalized_cell(play.event_id)
            if play.cell is None:
                play.cell = self._memory_cell(memory, arena_px)
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

    def _find_detected_play_by_event_id(self, play_event_id):
        for play in self.detected_card_plays:
            if play.event_id == play_event_id:
                return play
        return None

    def _overlapping_long_visible_area_play(self, card_name, memory, arena_px):
        if card_name not in LONG_VISIBLE_AREA_SPELLS or arena_px is None:
            return None
        candidate_cells = self._observed_cells(memory, arena_px)
        for play in reversed(self.detected_card_plays):
            if play.card != card_name:
                continue
            previous_memory = self.tracks.get(play.track_id)
            if previous_memory is None:
                continue
            gap_s = max(
                0.0,
                previous_memory.last_seen_time - memory.first_seen_time,
                memory.last_seen_time - previous_memory.first_seen_time,
            )
            if gap_s > LONG_VISIBLE_AREA_DUPLICATE_WINDOW_S:
                continue
            previous_cells = self._observed_cells(previous_memory, arena_px)
            if self._cell_sets_overlap(previous_cells, candidate_cells):
                return play
        return None

    def _merge_long_visible_area_observations(self, play, memory, arena_px):
        previous_memory = self.tracks.get(play.track_id)
        if previous_memory is None:
            return
        seen = set(previous_memory.observed_centers)
        for observation in memory.observed_centers:
            if observation not in seen:
                previous_memory.observed_centers.append(observation)
                seen.add(observation)
        previous_memory.observed_centers.sort(key=lambda item: item[0], reverse=True)
        play.cell = self._majority_observed_cell(previous_memory, arena_px)

    def _cell_sets_overlap(self, left_cells, right_cells):
        return any(
            max(abs(left[0] - right[0]), abs(left[1] - right[1]))
            <= ENEMY_SPELL_DISTINCT_CELL_DISTANCE
            for left in left_cells
            for right in right_cells
        )

    def _is_recent_duplicate_play(self, card_name, time_left_s, memory, arena_px):
        if self._mirror_source_play(card_name, memory, arena_px) is not None:
            return False
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

    def _mirror_source_play(self, card_name, memory, arena_px):
        previous_idx = None
        for idx in range(len(self.detected_card_plays) - 1, -1, -1):
            if self.detected_card_plays[idx].card == card_name:
                previous_idx = idx
                break
        if previous_idx is None:
            return None
        intervening_plays = len(self.detected_card_plays) - previous_idx - 1
        if intervening_plays >= 4:
            return None

        previous = self.detected_card_plays[previous_idx]
        candidate_cell = self._memory_cell(memory, arena_px)
        distinct_spell_target = (
            card_name in SPELL_CARD_NAMES
            and self._has_distinct_spell_cell(previous.cell, candidate_cell)
        )
        independently_confirmed = memory.clock_confirmed or distinct_spell_target
        return previous if independently_confirmed else None

    def _remember_mirror_seen(self, played_via):
        if played_via != "mirror":
            return
        mirror_id = card_to_id("mirror")
        if mirror_id is not None:
            self.confirmed_seen_cards.add(mirror_id)

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
        card_name = DIRECT_UNIT_TO_CARD.get(memory.best_class)
        if card_name in LONG_VISIBLE_AREA_SPELLS:
            majority_cell = self._majority_observed_cell(memory, arena_px)
            if majority_cell is not None:
                return majority_cell
        if memory.deploy_clock_center_x is not None and memory.deploy_clock_center_y is not None:
            return ACTION_GRID.pixel_to_cell(
                memory.deploy_clock_center_x,
                memory.deploy_clock_center_y,
                arena_px,
            )
        if memory.center_x is None or memory.center_y is None:
            return None
        return ACTION_GRID.pixel_to_cell(memory.center_x, memory.center_y, arena_px)

    def _majority_observed_cell(self, memory, arena_px):
        observed_cells = self._observed_cells(memory, arena_px)
        if not observed_cells:
            return None
        counts = Counter(observed_cells)
        max_count = max(counts.values())
        for cell in reversed(observed_cells):
            if counts[cell] == max_count:
                return cell
        return None

    def _observed_cells(self, memory, arena_px):
        return [
            ACTION_GRID.pixel_to_cell(center_x, center_y, arena_px)
            for _, center_x, center_y in memory.observed_centers
        ]

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

    def _drop_stale_tracks(self, time_left_s):
        stale_ids = [
            track_id
            for track_id, memory in self.tracks.items()
            if memory.last_seen_time - time_left_s > ENEMY_CARD_STALE_AFTER_SECONDS
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]
