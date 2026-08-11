from __future__ import annotations

from dataclasses import dataclass

from cr_bot.app.state_builder import build_game_state
from cr_bot.domain.frame_analysis import FrameAnalysisResult
from cr_bot.trackers.enemy_cards import EnemyCardTracker
from cr_bot.trackers.hand_state_filter import HandStateFilter
from cr_bot.trackers.match_clock import MatchClockFilter
from cr_bot.trackers.own_actions import OwnActionTracker
from cr_bot.trackers.tower_hp_filter import TowerHPFilter
from cr_bot.vision.match_state import game_end_from_result, game_start
from cr_bot.vision.timer import total_remaining_seconds


@dataclass(slots=True)
class MatchSessionStep:
    analysis: FrameAnalysisResult
    game_state: object | None
    should_emit: bool
    in_game: bool
    finished_enemy_plays: list | None = None


class MatchSession:
    def __init__(self, *, tracker_debug: bool = True, temporal_spell_predictor=None) -> None:
        self.tracker_debug = tracker_debug
        self.temporal_spell_predictor = temporal_spell_predictor
        self.reset()

    def reset(self) -> None:
        self.enemy_card_tracker = EnemyCardTracker(debug=self.tracker_debug)
        self.own_action_tracker = OwnActionTracker()
        self.match_clock_filter = MatchClockFilter()
        self.tower_hp_filter = TowerHPFilter()
        self.hand_state_filter = HandStateFilter()
        self.game_started = False
        self.not_in_game_streak = 0

    def process(self, analysis: FrameAnalysisResult, *, frame, now_s: float) -> MatchSessionStep:
        temporal_spell_detections = (
            self.temporal_spell_predictor.update(
                frame,
                video_time_s=now_s,
                arena_px=analysis.arena_px,
            )
            if self.temporal_spell_predictor is not None
            else []
        )
        analysis = analysis.with_hand_state(self.hand_state_filter.update(analysis.hand_state))

        if not self.game_started and (game_start(frame) or has_visible_match_timer(analysis)):
            self.game_started = True
            analysis = analysis.with_towers_hp(
                self.tower_hp_filter.update(analysis.towers_hp)
            )
            self.match_clock_filter.initialise(analysis.time_left_s, now_s)
            self.enemy_card_tracker.start_match(
                analysis.time_left_s,
                analysis.total_remaining_s,
                now_s=now_s,
            )
            game_state = self._build_game_state(analysis)
            self._update_own_actions(game_state, analysis, frame=frame, now_s=now_s)
            return MatchSessionStep(
                analysis=analysis,
                game_state=game_state,
                should_emit=True,
                in_game=True,
            )

        if self.game_started:
            if self.match_clock_filter.initialised:
                filtered_time_left_s = self.match_clock_filter.update(
                    analysis.time_left_s,
                    now_s,
                    analysis.overtime,
                )
                analysis = analysis.with_clock(
                    time_left_s=filtered_time_left_s,
                    total_remaining_s=total_remaining_seconds(
                        filtered_time_left_s,
                        analysis.overtime,
                    ),
                )
            else:
                self.match_clock_filter.initialise(analysis.time_left_s, now_s)

            analysis = analysis.with_towers_hp(
                self.tower_hp_filter.update(analysis.towers_hp)
            )
            game_state = self._build_game_state(analysis)
            self._update_own_actions(game_state, analysis, frame=frame, now_s=now_s)
            self.enemy_card_tracker.update(
                analysis.total_remaining_s,
                analysis.matches,
                now_s=now_s,
                clock_boxes=analysis.clock_boxes,
                own_actions=self.own_action_tracker.actions,
                pending_own_spell_targets=self.own_action_tracker.pending_spell_targets,
                arena_px=analysis.arena_px,
                frame=frame,
                claimed_spell_observation_keys=set(
                    self.own_action_tracker.claimed_spell_target_observations
                ),
                temporal_spell_detections=temporal_spell_detections,
            )

            if game_end_from_result(analysis):
                self.not_in_game_streak += 1
                if self.not_in_game_streak >= 20:
                    finished_enemy_plays = list(self.enemy_card_tracker.detected_card_plays)
                    self.reset()
                    return MatchSessionStep(
                        analysis=analysis,
                        game_state=None,
                        should_emit=False,
                        in_game=False,
                        finished_enemy_plays=finished_enemy_plays,
                    )
            else:
                self.not_in_game_streak = 0

            return MatchSessionStep(
                analysis=analysis,
                game_state=game_state,
                should_emit=True,
                in_game=True,
            )

        return MatchSessionStep(
            analysis=analysis,
            game_state=None,
            should_emit=False,
            in_game=False,
        )

    def _build_game_state(self, analysis: FrameAnalysisResult):
        return build_game_state(
            analysis,
            seen_enemy_cards=list(self.enemy_card_tracker.confirmed_seen_cards),
            elixir_enemy_est=self.enemy_card_tracker.elixir_enemy_est,
            game_started=self.game_started,
        )

    def _update_own_actions(self, game_state, analysis: FrameAnalysisResult, *, frame, now_s: float) -> None:
        self.own_action_tracker.update(
            game_state,
            analysis.arena_px,
            frame=frame,
            clock_boxes=analysis.clock_boxes,
            own_actions_blocked=len(analysis.emote_boxes) >= 2,
            elixir_change=analysis.elixir_change,
            video_time_s=now_s,
        )
        self.enemy_card_tracker.reconcile_own_actions(
            self.own_action_tracker.actions,
            arena_px=analysis.arena_px,
        )


def has_visible_match_timer(analysis: FrameAnalysisResult) -> bool:
    return analysis.time_left_s is not None and float(analysis.time_left_s) > 0.0
