from __future__ import annotations

from cr_bot.domain.constants import (
    ENEMY_CARD_STALE_AFTER_SECONDS,
    ENEMY_SPELL_DISTINCT_CELL_DISTANCE,
    ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S,
)
from cr_bot.features.action_space import ACTION_GRID
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

from .impact_observer import (
    ArenaFrameSample,
    EnemyProjectileImpactObserver,
    build_quadratic_trajectory_model,
)
from .models import (
    EnemyProjectileSpellEvent,
    ProjectileTrajectoryConfig,
    RecentArenaFrame,
    RecentSpellTargetObservation,
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

FIREBALL_DIRECTION_MIN_DELTA_NORM = 0.015
FIREBALL_DIRECTION_MIN_STEPS = 2
FIREBALL_EXPLOSION_MAX_SPAN_NORM = 0.08
FIREBALL_EXPLOSION_CONFIRM_DELAY_S = 0.6


class ProjectileSpellTracker:
    def __init__(self, debug, find_play):
        self._debug = debug
        self._find_play = find_play
        self.ally_candidates: dict[int, TrackMemory] = {}
        self.events: list[EnemyProjectileSpellEvent] = []
        self.target_observations: list[RecentSpellTargetObservation] = []
        self.arena_frames: list[RecentArenaFrame] = []
        self.impact_observer = EnemyProjectileImpactObserver()

    def register(self, play, memory):
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
        self.events.append(event)

    def find_event(self, play_event_id):
        for event in self.events:
            if event.play_event_id == play_event_id:
                return event
        return None

    def remove_event(self, play_event_id):
        self.events = [
            event for event in self.events
            if event.play_event_id != play_event_id
        ]

    def finalized_cell(self, play_event_id):
        event = self.find_event(play_event_id)
        if event is None:
            return None
        return event.finalized_cell

    def observe_ally_fireball(
        self,
        troop,
        *,
        time_left_s,
        now_s,
        own_actions,
        arena_px,
        should_frame_confirm,
    ):
        if troop.team != "ally" or DIRECT_UNIT_TO_CARD.get(troop.class_name) != "fireball":
            return None, None
        track_id = getattr(troop, "track_id", None)
        if track_id is None:
            return None, None
        memory = self.ally_candidates.get(track_id)
        if memory is None:
            memory = TrackMemory(
                track_id=track_id,
                first_seen_time=time_left_s,
                last_seen_time=time_left_s,
                first_seen_now_s=now_s,
            )
            self.ally_candidates[track_id] = memory
        memory.add_observation(
            troop.class_name,
            troop.team,
            troop.confidence,
            time_left_s,
            center_x=troop.center_x,
            center_y=troop.center_y,
        )
        memory.add_motion_center(
            now_s if now_s is not None else time_left_s,
            troop.center_x,
            troop.center_y,
        )
        memory.center_x = troop.center_x
        memory.center_y = troop.center_y
        if memory.counted_as_card or not should_frame_confirm(memory):
            return memory, None
        if self._explained_by_own_action(memory, own_actions, arena_px):
            memory.counted_as_card = True
            return memory, "own-action"
        if self.should_absorb(
            "fireball",
            memory,
            time_left_s=time_left_s,
            own_actions=own_actions,
            arena_px=arena_px,
        ):
            memory.counted_as_card = True
            return memory, "continuation"
        ownership = self.fireball_ownership(memory, arena_px)
        if ownership == "own":
            memory.counted_as_card = True
            return memory, "own-direction"
        if ownership == "explosion":
            return memory, "waiting-direction"
        if ownership != "enemy":
            return memory, "waiting-direction"
        return memory, "record"

    def fireball_ownership(self, memory, arena_px):
        if arena_px is None or len(memory.motion_centers) < FIREBALL_DIRECTION_MIN_STEPS + 1:
            return None
        arena_h = arena_px[3]
        if arena_h <= 0:
            return None
        centers = memory.motion_centers
        y_values = [center_y for _, _, center_y in centers]
        x_values = [center_x for _, center_x, _ in centers]
        arena_w = arena_px[2]
        span_x = (max(x_values) - min(x_values)) / arena_w if arena_w > 0 else 1.0
        span_y = (max(y_values) - min(y_values)) / arena_h
        steps = [
            (current_y - previous_y) / arena_h
            for previous_y, current_y in zip(y_values, y_values[1:])
        ]
        meaningful_steps = [
            step for step in steps
            if abs(step) >= FIREBALL_DIRECTION_MIN_DELTA_NORM / 3.0
        ]
        total_delta = (y_values[-1] - y_values[0]) / arena_h
        if (
            total_delta >= FIREBALL_DIRECTION_MIN_DELTA_NORM
            and sum(step > 0 for step in meaningful_steps) >= FIREBALL_DIRECTION_MIN_STEPS
        ):
            return "enemy"
        if (
            total_delta <= -FIREBALL_DIRECTION_MIN_DELTA_NORM
            and sum(step < 0 for step in meaningful_steps) >= FIREBALL_DIRECTION_MIN_STEPS
        ):
            return "own"
        if max(span_x, span_y) <= FIREBALL_EXPLOSION_MAX_SPAN_NORM:
            return "explosion"
        return None

    def confirmed_explosion_candidates(
        self,
        time_left_s,
        *,
        own_actions,
        arena_px,
        should_frame_confirm,
    ):
        confirmed = []
        for memory in self.ally_candidates.values():
            if memory.counted_as_card or not should_frame_confirm(memory):
                continue
            if memory.first_seen_time - time_left_s < FIREBALL_EXPLOSION_CONFIRM_DELAY_S:
                continue
            if self.fireball_ownership(memory, arena_px) != "explosion":
                continue
            if self._explained_by_own_action(memory, own_actions, arena_px):
                memory.counted_as_card = True
                continue
            confirmed.append(memory)
        return confirmed

    def cleanup_ally_candidates(self, time_left_s):
        self.ally_candidates = {
            track_id: memory
            for track_id, memory in self.ally_candidates.items()
            if memory.last_seen_time - time_left_s <= ENEMY_CARD_STALE_AFTER_SECONDS
        }

    def forget_stale(self, time_left_s):
        max_update_window_s = max(
            (config.update_window_s for config in PROJECTILE_SPELL_CONFIGS.values()),
            default=0.0,
        )
        self.target_observations = [
            observation
            for observation in self.target_observations
            if observation.time_left_s - time_left_s <= max_update_window_s + 0.5
        ]
        max_frame_window_s = max(
            (
                config.update_window_s + config.pre_observation_window_s
                for config in PROJECTILE_SPELL_CONFIGS.values()
            ),
            default=0.0,
        )
        self.arena_frames = [
            sample
            for sample in self.arena_frames
            if sample.time_left_s - time_left_s <= max_frame_window_s + 0.5
        ]

    def remember_arena_frame(self, time_left_s, *, frame, arena_px):
        if frame is None or arena_px is None:
            return
        arena_x, arena_y, arena_w, arena_h = [int(round(value)) for value in arena_px]
        if arena_w <= 0 or arena_h <= 0:
            return
        if any(abs(sample.time_left_s - time_left_s) < 1e-6 for sample in self.arena_frames):
            return
        arena_crop = frame[arena_y:arena_y + arena_h, arena_x:arena_x + arena_w]
        if arena_crop.size == 0:
            return
        self.arena_frames.append(
            RecentArenaFrame(
                time_left_s=time_left_s,
                arena_bgr=arena_crop.copy(),
            )
        )
        self.arena_frames.sort(key=lambda sample: sample.time_left_s, reverse=True)
        del self.arena_frames[12:]

    def remember_target_observations(self, time_left_s, *, arena_px):
        if arena_px is None:
            return
        samples = [
            ArenaFrameSample(time_left_s=sample.time_left_s, arena_bgr=sample.arena_bgr)
            for sample in self.arena_frames
        ]
        for event in self.events:
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
                self._debug_impact_observation(event, sample, debug_info)
                if observation is None:
                    continue
                if any(existing.key == observation.key for existing in self.target_observations):
                    continue
                self.target_observations.append(observation)

    def reconcile(self, *, arena_px, claimed_spell_observation_keys):
        if arena_px is None:
            return
        for event in sorted(self.events, key=lambda item: item.started_at_s, reverse=True):
            config = PROJECTILE_SPELL_CONFIGS.get(event.card)
            assert config, "projectile config must not be empty"
            if event.card == "fireball":
                continue
            candidates = []
            for observation in self.target_observations:
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
                candidates.append(
                    (
                        delta_s < 0,
                        trajectory_metrics[0],
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
                previous = self._find_target_observation(event.claimed_observation_key)
                if previous is not None and previous.claimed_by_event_id == event.play_event_id:
                    previous.claimed_by_event_id = None
            chosen.claimed_by_event_id = event.play_event_id
            event.finalized = True
            event.finalized_cell = chosen.cell
            event.best_cell = chosen.cell
            event.claimed_observation_key = chosen.key
            event.best_observation_score = score_tuple
            play = self._find_play(event.play_event_id)
            if play is not None:
                play.cell = chosen.cell
            self._debug(
                f"finalized enemy projectile card={event.card} "
                f"track={event.first_track_id} cell={chosen.cell} "
                f"obs_phase={chosen.phase} dt={event.started_at_s - chosen.time_left_s:.2f}"
            )

    def update_from_matches(self, matches, *, time_left_s, own_actions, arena_px):
        if arena_px is None:
            return
        own_fireball_actions = self._recent_own_actions("fireball", own_actions)
        for match in matches:
            troop = match.troop
            if DIRECT_UNIT_TO_CARD.get(troop.class_name) != "fireball":
                continue
            event, score = self._best_enemy_event_for_detection(
                card="fireball",
                time_left_s=time_left_s,
                center_x=troop.center_x,
                center_y=troop.center_y,
                arena_px=arena_px,
                own_actions=own_fireball_actions,
                include_finalized=True,
            )
            if event is None:
                event = self._recent_unclaimed_fireball_event(
                    time_left_s=time_left_s,
                    center_x=troop.center_x,
                    center_y=troop.center_y,
                    arena_px=arena_px,
                    own_actions=own_fireball_actions,
                )
                score = 0.0 if event is not None else None
            if event is None:
                continue
            self._assign_detection(
                event,
                time_left_s=time_left_s,
                center_x=troop.center_x,
                center_y=troop.center_y,
                arena_px=arena_px,
                source_team=troop.team,
                source_track_id=getattr(troop, "track_id", None),
                score=score,
            )

    def should_absorb(self, card_name, memory, *, time_left_s, own_actions, arena_px):
        if (
            card_name != "fireball"
            or memory.center_x is None
            or memory.center_y is None
            or arena_px is None
        ):
            return False
        own_projectile_actions = self._recent_own_actions(card_name, own_actions)
        event, score = self._best_enemy_event_for_detection(
            card=card_name,
            time_left_s=time_left_s,
            center_x=memory.center_x,
            center_y=memory.center_y,
            arena_px=arena_px,
            own_actions=own_projectile_actions,
            include_finalized=True,
        )
        if event is None:
            event = self._recent_unclaimed_fireball_event(
                time_left_s=time_left_s,
                center_x=memory.center_x,
                center_y=memory.center_y,
                arena_px=arena_px,
                own_actions=own_projectile_actions,
            )
            score = 0.0 if event is not None else None
        if event is None:
            return False
        self._assign_detection(
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

    def _explained_by_own_action(self, memory, own_actions, arena_px):
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

    def _recent_own_actions(self, card_name, own_actions):
        if card_name not in PROJECTILE_SPELL_CONFIGS:
            return []
        return [
            action
            for action in reversed(own_actions)
            if action.get("card") == card_name and action.get("cell") is not None
        ]

    def _recent_unclaimed_fireball_event(
        self,
        *,
        time_left_s,
        center_x,
        center_y,
        arena_px,
        own_actions,
    ):
        config = PROJECTILE_SPELL_CONFIGS["fireball"]
        own_score = self._best_own_projectile_score(
            card="fireball",
            time_left_s=time_left_s,
            center_x=center_x,
            center_y=center_y,
            arena_px=arena_px,
            own_actions=own_actions,
        )
        if own_score is not None and own_score <= config.corridor_width_norm:
            return None
        candidates = [
            event
            for event in self.events
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

    def _best_enemy_event_for_detection(
        self,
        *,
        card,
        time_left_s,
        center_x,
        center_y,
        arena_px,
        own_actions,
        include_finalized,
    ):
        candidates = []
        own_best_score = self._best_own_projectile_score(
            card=card,
            time_left_s=time_left_s,
            center_x=center_x,
            center_y=center_y,
            arena_px=arena_px,
            own_actions=own_actions,
        )
        for event in self.events:
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

    def _best_own_projectile_score(
        self,
        *,
        card,
        time_left_s,
        center_x,
        center_y,
        arena_px,
        own_actions,
    ):
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
        return lateral_error, cand_y

    def _assign_detection(
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
        if (
            event.last_assigned_time_left_s is not None
            and time_left_s > event.last_assigned_time_left_s + 1e-6
        ):
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
        play = self._find_play(event.play_event_id)
        if play is not None:
            play.cell = candidate_cell
        self._debug(
            f"projectile continuation card={event.card} event={event.play_event_id} "
            f"track={event.first_track_id} "
            f"source_track={source_track_id if source_track_id is not None else '-'} "
            f"source_team={source_team} cell={candidate_cell} score={score:.3f}"
        )

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

    def _find_target_observation(self, observation_key):
        for observation in self.target_observations:
            if observation.key == observation_key:
                return observation
        return None

    @staticmethod
    def _observation_phase_rank(phase):
        if phase == "impact":
            return 0
        if phase == "aim":
            return 1
        return 2

    def _debug_impact_observation(self, event, sample, debug_info):
        if event.card != "fireball":
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
