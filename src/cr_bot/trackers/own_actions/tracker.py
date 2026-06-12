from __future__ import annotations

from collections import Counter, deque
import os

from cr_bot.domain.card_metadata import CARD_METADATA
from cr_bot.domain.constants import (
    ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S,
    OWN_ACTION_DUPLICATE_WINDOW_S,
    OWN_ACTION_RECENT_HAND_WINDOW_S,
    OWN_ACTION_RECENT_TRACK_WINDOW_S,
    OWN_ACTION_START_TIME_LEFT_S,
    OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S,
    OWN_ACTION_TRACK_FALLBACK_CARDS,
)
from cr_bot.domain.events import OwnActionEvent
from cr_bot.features.action_space import ACTION_GRID
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD
from cr_bot.vision.spell_deploy import SpellDeployLocator

from .models import (
    CARD_ALIASES,
    ELIXIR_CHANGE_VIDEO_TIME_OFFSET_S,
    LOG_YOLO_AFTER_PENDING_WINDOW_S,
    PENDING_UNIT_LABELS,
    ROLLING_SPELL_UNIT_LABELS,
    PendingOwnPlay,
    RecentAllyTrack,
)
from .spell_targets import sense_spell_target_observations


class OwnActionTracker:
    def __init__(self) -> None:
        self.last_hand: list[str | None] = [None, None, None, None]
        self.slot_history: list[deque[str | None]] = [deque(maxlen=5) for _ in range(4)]
        self.last_elixir: float | None = None
        self.pending: list[PendingOwnPlay] = []
        self.seen_ally_tracks: set[int] = set()
        self.consumed_log_track_ids: set[int] = set()
        self.recent_ally_tracks: dict[int, RecentAllyTrack] = {}
        self.actions: list[OwnActionEvent] = []
        self.recent_hand_seen: dict[str, float] = {}
        self.spell_deploy_locator = SpellDeployLocator()
        self.claimed_spell_target_observations: dict[str, float] = {}
        self.debug = os.environ.get("DEBUG_OWN_ACTIONS") == "1"

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[own_actions] {message}")

    @property
    def pending_spell_targets(self):
        return [
            {
                "card": pending.card,
                "time_left_s": pending.started_at_s,
                "cell": pending.spell_target_cell,
            }
            for pending in self.pending
            if pending.spell_target_cell is not None
        ]

    def update(
        self,
        game_state,
        arena_px,
        frame=None,
        clock_boxes=None,
        own_actions_blocked=False,
        elixir_change=None,
        video_time_s=None,
    ):
        clock_boxes = clock_boxes or []
        now = game_state.total_remaining_s
        hand = game_state.hud.hand_cards
        elixir = game_state.hud.elixir_self
        elixir_change_detected = bool((elixir_change or {}).get("covered"))
        if own_actions_blocked:
            if self.pending:
                self._debug(f"own action detection blocked; preserving pending={len(self.pending)}")
            self.last_hand = hand[:]
            self.last_elixir = elixir
            self._remember_hand(hand, now)
            self._remember_slot_history(hand)
            return

        self._debug(
            f"update now={now} elixir={elixir:.2f} "
            f"last_elixir={self.last_elixir} hand={hand} "
            f"last_hand={self.last_hand} pending={len(self.pending)} "
            f"own_units={len(game_state.own_units)} clocks={len(clock_boxes)}"
        )
        self._forget_stale_spell_target_claims(now)

        self._detect_slot_drops(
            hand,
            elixir,
            now,
            elixir_change_time_s=now if elixir_change_detected else None,
            elixir_change_video_time_s=(
                self._calibrated_elixir_change_video_time(video_time_s)
                if elixir_change_detected else None
            ),
        )
        if elixir_change_detected:
            self._attach_elixir_change_to_pending(now, video_time_s)
        self._confirm_pending(
            game_state, arena_px, elixir, now,
            frame=frame, clock_boxes=clock_boxes, video_time_s=video_time_s,
        )
        self._remember_hand(hand, now)
        self._remember_slot_history(hand)
        self.last_hand = hand[:]
        self.last_elixir = elixir
        self._debug(
            f"state saved last_hand={self.last_hand} "
            f"last_elixir={self.last_elixir:.2f} actions={len(self.actions)}"
        )

    def _detect_slot_drops(self, hand, elixir, now, elixir_change_time_s=None, elixir_change_video_time_s=None):
        for idx, (prev, cur) in enumerate(zip(self.last_hand, hand)):
            self._debug(f"slot {idx}: prev={prev} cur={cur}")
            if prev is not None and cur is None:
                dropped_card = self._normalize_card(self._resolve_drop_card(idx, prev))
                dropped_card, played_via = self._resolve_mirror_play(dropped_card)
                self._debug(
                    f"slot {idx} drop detected: card={dropped_card} "
                    f"played_via={played_via} "
                    f"started_at={now} elixir_before={elixir:.2f}"
                )
                if dropped_card is None:
                    continue
                self.pending.append(
                    PendingOwnPlay(
                        card=dropped_card, slot_idx=idx, started_at_s=now, elixir_before=elixir,
                        elixir_change_time_s=elixir_change_time_s,
                        elixir_change_video_time_s=elixir_change_video_time_s,
                        played_via=played_via,
                    )
                )
            elif prev is not None and cur != prev:
                dropped_card = self._normalize_card(self._resolve_drop_card(idx, prev))
                dropped_card, played_via = self._resolve_mirror_play(dropped_card)
                if dropped_card is None:
                    continue
                if self._is_rolling_spell(dropped_card):
                    self._debug(
                        f"slot {idx} rolling spell change detected: prev={prev} cur={cur} "
                        f"card={dropped_card} started_at={now} elixir_before={elixir:.2f}"
                    )
                    self.pending.append(
                        PendingOwnPlay(
                            card=dropped_card, slot_idx=idx, started_at_s=now, elixir_before=elixir,
                            elixir_change_time_s=elixir_change_time_s,
                            elixir_change_video_time_s=elixir_change_video_time_s,
                            played_via=played_via,
                        )
                    )
                    continue
                self._debug(f"slot {idx} changed but not treated as drop: prev={prev} cur={cur}")

    def _attach_elixir_change_to_pending(self, now, video_time_s):
        candidates = []
        for pending in self.pending:
            if pending.elixir_change_time_s is not None:
                continue
            elapsed_s = pending.started_at_s - now
            if not 0 <= elapsed_s <= OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S:
                continue
            candidates.append((elapsed_s, pending))
        if not candidates:
            return
        pending = min(candidates, key=lambda item: item[0])[1]
        pending.elixir_change_time_s = now
        pending.elixir_change_video_time_s = self._calibrated_elixir_change_video_time(video_time_s)
        self._debug(
            f"attached elixir-change time card={pending.card} "
            f"slot={pending.slot_idx} time_left={now} video_time={video_time_s}"
        )

    def _calibrated_elixir_change_video_time(self, video_time_s):
        if video_time_s is None:
            return None
        return video_time_s + ELIXIR_CHANGE_VIDEO_TIME_OFFSET_S

    def _resolve_drop_card(self, slot_idx, prev):
        history = [card for card in self.slot_history[slot_idx] if card is not None]
        if not history:
            return prev
        counts = Counter(history)
        dominant_card, dominant_count = counts.most_common(1)[0]
        prev_count = counts.get(prev, 0)
        if dominant_card != prev and dominant_count >= 3 and prev_count <= 2:
            self._debug(
                f"slot {slot_idx} drop relabeled from {prev} to {dominant_card} "
                f"using recent history={list(self.slot_history[slot_idx])}"
            )
            return dominant_card
        return prev

    def _normalize_card(self, card):
        return CARD_ALIASES.get(card, card)

    def _resolve_mirror_play(self, card):
        if card != "mirror":
            return card, None
        if not self.actions:
            self._debug("mirror drop ignored: no previous confirmed own action")
            return None, None
        return self.actions[-1].card, "mirror"

    def _remember_slot_history(self, hand):
        for idx, card in enumerate(hand):
            self.slot_history[idx].append(card)

    def _remember_hand(self, hand, now):
        for card in hand:
            if card is not None:
                self.recent_hand_seen[card] = now
        stale_cards = [
            card for card, seen_at in self.recent_hand_seen.items()
            if seen_at - now > OWN_ACTION_RECENT_HAND_WINDOW_S
        ]
        for card in stale_cards:
            del self.recent_hand_seen[card]

    def _confirm_pending(self, game_state, arena_px, elixir, now, frame=None, clock_boxes=None, video_time_s=None):
        clock_boxes = clock_boxes or []
        new_tracks = []
        for match in game_state.own_units:
            track_id = match.troop.track_id
            if track_id is None:
                continue
            if track_id not in self.seen_ally_tracks:
                new_tracks.append(match)
                self.seen_ally_tracks.add(track_id)
                self._debug(
                    f"new ally track id={track_id} class={match.troop.class_name} "
                    f"center=({match.troop.center_x:.1f}, {match.troop.center_y:.1f})"
                )
            self._remember_ally_track(match, now)
        if not new_tracks:
            self._debug("no new ally tracks")
        self._forget_stale_ally_tracks(now)

        had_pending = bool(self.pending)
        still_pending = []
        rolling_spell_matches = self._first_visible_rolling_spells(game_state.own_units)
        rolling_spell_pending_to_confirm = self._rolling_spell_pending_for_detection(rolling_spell_matches, now)
        elixir_pending_to_confirm = self._pending_for_current_elixir_drop(
            elixir, now, preferred_pending=rolling_spell_pending_to_confirm,
        )
        for pending in self.pending:
            is_spell = self._is_spell_card(pending.card)
            cost = CARD_METADATA.get(pending.card, {}).get("elixir_cost")
            elixir_drop = None if self.last_elixir is None else self.last_elixir - elixir
            required_drop = self._required_numeric_elixir_drop(pending)
            pending_elixir_drop = pending.elixir_before - elixir
            selected_global_elixir_drop = pending is elixir_pending_to_confirm
            own_elixir_confirms = (
                required_drop is not None
                and pending_elixir_drop >= required_drop
                and 0 <= pending.started_at_s - now <= OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S
            )
            elixir_confirms = selected_global_elixir_drop or own_elixir_confirms
            if elixir_confirms and pending.numeric_elixir_drop_time_s is None:
                pending.numeric_elixir_drop_time_s = now
                pending.numeric_elixir_drop_video_time_s = video_time_s
                pending.numeric_elixir_drop_source = (
                    "global-drop" if selected_global_elixir_drop else "pending-elixir-before"
                )
                self._debug(
                    f"latched numeric elixir drop card={pending.card} "
                    f"slot={pending.slot_idx} time_left={now} video_time={video_time_s} "
                    f"source={pending.numeric_elixir_drop_source}"
                )
            if is_spell:
                if self._is_rolling_spell(pending.card):
                    placed_cell, keep_pending = self._confirm_pending_rolling_spell(
                        pending,
                        rolling_spell_matches.get(pending.card),
                        pending is rolling_spell_pending_to_confirm,
                        elixir_confirms or pending.spell_elixir_confirmed,
                        arena_px,
                        now,
                    )
                else:
                    placed_cell, keep_pending = self._confirm_pending_spell(
                        pending, arena_px, frame, elixir_confirms, now
                    )
            else:
                placed_cell = self._infer_pending_cell(
                    pending, self._recent_tracks_for_pending(pending, now), arena_px, frame, clock_boxes,
                )
                keep_pending = placed_cell is None
                if pending.started_at_s - now > OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S:
                    keep_pending = False
            self._debug(
                f"pending check card={pending.card} slot={pending.slot_idx} "
                f"cost={cost} elixir_drop={elixir_drop} pending_elixir_drop={pending_elixir_drop} "
                f"required_drop={required_drop} elixir_confirms={elixir_confirms} "
                f"placed_cell={placed_cell}"
            )
            confirms = placed_cell is not None
            if confirms:
                has_elixir_evidence = (
                    is_spell or pending.elixir_change_time_s is not None
                    or pending.numeric_elixir_drop_time_s is not None or elixir_confirms
                )
                if not has_elixir_evidence:
                    if pending.started_at_s - now > OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S:
                        self._debug(
                            f"pending cancelled card={pending.card} "
                            f"slot={pending.slot_idx}; cell found but no elixir evidence"
                        )
                    else:
                        self._debug(
                            f"pending kept card={pending.card} slot={pending.slot_idx}; "
                            "cell found but no elixir evidence"
                        )
                        still_pending.append(pending)
                    continue
                reasons = []
                if elixir_confirms and not is_spell:
                    reasons.append("elixir")
                if placed_cell is not None:
                    reasons.append("spell_release" if is_spell else "cell")
                self._debug(
                    f"confirmed own action card={pending.card} "
                    f"slot={pending.slot_idx} reasons={'+'.join(reasons)}"
                )
                self._append_action(
                    now=(
                        pending.elixir_change_time_s
                        if pending.elixir_change_time_s is not None
                        else (
                            pending.numeric_elixir_drop_time_s
                            if pending.numeric_elixir_drop_time_s is not None
                            else now
                        )
                    ),
                    card=pending.card,
                    slot_idx=pending.slot_idx,
                    cell=placed_cell,
                    video_time_s=(
                        pending.elixir_change_video_time_s
                        if pending.elixir_change_video_time_s is not None
                        else pending.numeric_elixir_drop_video_time_s
                    ),
                    rolling_spell_track_id=pending.rolling_spell_first_track_id,
                    played_via=pending.played_via,
                )
                if self._is_rolling_spell(pending.card) and pending.rolling_spell_first_track_id is not None:
                    self.consumed_log_track_ids.add(pending.rolling_spell_first_track_id)
            else:
                if keep_pending:
                    self._debug(f"pending kept card={pending.card} slot={pending.slot_idx}")
                    still_pending.append(pending)
                else:
                    self._debug(f"pending cancelled card={pending.card} slot={pending.slot_idx}")

        self.pending = still_pending
        if not had_pending and not self.pending:
            self._record_new_track_actions(new_tracks, game_state.hud.hand_cards, arena_px, now, clock_boxes)

    def _remember_ally_track(self, match, now):
        track_id = match.troop.track_id
        existing = self.recent_ally_tracks.get(track_id)
        first_seen_s = now
        if existing is not None:
            previous_class = existing.match.troop.class_name
            current_class = match.troop.class_name
            if previous_class == current_class:
                first_seen_s = existing.first_seen_s
            else:
                self._debug(
                    f"ally track id={track_id} class changed "
                    f"{previous_class}->{current_class}; refreshing recent track"
                )
        self.recent_ally_tracks[track_id] = RecentAllyTrack(match=match, first_seen_s=first_seen_s, last_seen_s=now)

    def _forget_stale_ally_tracks(self, now):
        stale_track_ids = [
            track_id for track_id, memory in self.recent_ally_tracks.items()
            if memory.last_seen_s - now > OWN_ACTION_RECENT_TRACK_WINDOW_S
        ]
        for track_id in stale_track_ids:
            del self.recent_ally_tracks[track_id]

    def _recent_tracks_for_pending(self, pending, now):
        candidates = []
        for memory in self.recent_ally_tracks.values():
            if memory.last_seen_s - now > OWN_ACTION_RECENT_TRACK_WINDOW_S:
                continue
            if memory.first_seen_s > pending.started_at_s:
                continue
            if pending.started_at_s - memory.first_seen_s > OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S:
                continue
            candidates.append(memory.match)
        return candidates

    def _infer_pending_cell(self, pending, candidate_tracks, arena_px, frame, clock_boxes):
        if self._is_spell_card(pending.card):
            return None
        return self._infer_cell_from_clock(candidate_tracks, arena_px, clock_boxes, pending.card)

    def _confirm_pending_spell(self, pending, arena_px, frame, elixir_confirms, now):
        if frame is None or arena_px is None:
            return None, True
        observations = sense_spell_target_observations(
            self.spell_deploy_locator,
            frame=frame,
            arena_px=arena_px,
            card=pending.card,
            time_left_s=now,
        )
        deploy = next((obs for obs in observations if obs.phase == "aim"), None)
        release = next((obs for obs in observations if obs.phase == "release"), None)
        if elixir_confirms:
            pending.spell_elixir_confirmed = True
        if deploy is not None:
            pending.spell_aim_seen = True
            pending.spell_target_cell = deploy.cell
            self._debug(f"spell aim ellipse visible card={pending.card} cell={pending.spell_target_cell}")
        else:
            self._debug(f"spell aim ellipse missing card={pending.card}")
        if release is not None:
            release_cell = release.cell
            pending.spell_release_seen = True
            pending.spell_target_cell = release_cell
            self._debug(f"spell release marker visible card={pending.card} cell={release_cell}")
            if pending.spell_elixir_confirmed and release_cell is not None:
                self._claim_spell_target_observations(observations)
                return release_cell, False
        if pending.spell_release_seen and pending.spell_elixir_confirmed and pending.spell_target_cell is not None:
            self._claim_spell_target_observations(observations)
            return pending.spell_target_cell, False
        if pending.spell_aim_seen and deploy is None and not pending.spell_release_seen:
            return None, False
        return None, True

    def _claim_spell_target_observations(self, observations):
        for observation in observations:
            self.claimed_spell_target_observations[observation.key] = observation.time_left_s

    def _forget_stale_spell_target_claims(self, now):
        stale_keys = [
            key
            for key, seen_at in self.claimed_spell_target_observations.items()
            if seen_at - now > ENEMY_SPELL_OWN_ACTION_VETO_WINDOW_S
        ]
        for key in stale_keys:
            del self.claimed_spell_target_observations[key]

    def _confirm_pending_rolling_spell(self, pending, rolling_spell_match, selected_for_detection, rolling_spell_elixir_confirmed, arena_px, now):
        if arena_px is None:
            return None, True
        if rolling_spell_elixir_confirmed:
            pending.spell_elixir_confirmed = True
        if rolling_spell_match is not None and pending.rolling_spell_first_cell is None:
            if not selected_for_detection:
                return None, False
            troop = rolling_spell_match.troop
            pending.rolling_spell_first_cell = self._rolling_spell_placement_cell(troop, arena_px)
            pending.rolling_spell_first_seen_s = now
            pending.rolling_spell_first_track_id = troop.track_id
            self._debug(
                f"first visible own rolling spell detected card={pending.card} "
                f"track={troop.track_id} center=({troop.center_x:.1f}, {troop.center_y:.1f}) "
                f"cell={pending.rolling_spell_first_cell}"
            )
        if pending.rolling_spell_first_cell is not None and pending.spell_elixir_confirmed:
            return pending.rolling_spell_first_cell, False
        if pending.started_at_s - now > OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S:
            return None, False
        return None, True

    def _first_visible_rolling_spells(self, own_units):
        matches_by_card = {}
        for card, unit_label in ROLLING_SPELL_UNIT_LABELS.items():
            candidates = [
                match for match in own_units
                if match.troop.class_name == unit_label
                and match.troop.team == "ally"
                and match.troop.track_id not in self.consumed_log_track_ids
            ]
            if candidates:
                matches_by_card[card] = max(candidates, key=lambda match: match.troop.confidence)
        return matches_by_card

    def _rolling_spell_pending_for_detection(self, rolling_spell_matches, now):
        if not rolling_spell_matches:
            return None
        candidates = []
        for pending in self.pending:
            if pending.card not in rolling_spell_matches:
                continue
            elapsed_s = pending.started_at_s - now
            if not 0 <= elapsed_s <= LOG_YOLO_AFTER_PENDING_WINDOW_S:
                continue
            candidates.append((elapsed_s, pending))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _pending_for_current_elixir_drop(self, elixir, now, preferred_pending=None):
        if self.last_elixir is None:
            return None
        elixir_drop = self.last_elixir - elixir
        candidates = []
        for pending in self.pending:
            if pending.spell_elixir_confirmed:
                continue
            required_drop = self._required_numeric_elixir_drop(pending)
            if required_drop is None or elixir_drop < required_drop:
                continue
            elapsed_s = pending.started_at_s - now
            if not 0 <= elapsed_s <= OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S:
                continue
            candidates.append((pending is preferred_pending, elapsed_s, pending))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (not item[0], item[1]))
        return candidates[0][2]

    def _required_numeric_elixir_drop(self, pending):
        cost = CARD_METADATA.get(pending.card, {}).get("elixir_cost")
        if cost is None:
            return None
        if pending.played_via == "mirror":
            cost += 1
        if self._is_rolling_spell(pending.card):
            return 0.8
        return max(0.5, cost - 1.5)

    def _rolling_spell_placement_cell(self, troop, arena_px):
        return ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, arena_px)

    def _is_rolling_spell(self, card):
        return card in ROLLING_SPELL_UNIT_LABELS

    def _record_new_track_actions(self, new_tracks, hand, arena_px, now, clock_boxes):
        for match in new_tracks:
            card = DIRECT_UNIT_TO_CARD.get(match.troop.class_name)
            if card is None:
                self._debug(f"new ally track not mapped to playable card: class={match.troop.class_name}")
                continue
            if self._is_spell_card(card):
                self._debug(
                    f"new ally spell track ignored; spell actions require HUD drop "
                    f"and white-radius deploy locator: class={match.troop.class_name} card={card}"
                )
                continue
            if not self._allows_track_fallback(card):
                self._debug(
                    f"new ally track ignored for card={card}; "
                    "direct track fallback is not enabled for this card"
                )
                continue
            slot_idx = self._find_recent_slot(card, hand)
            if slot_idx is None and card not in self.recent_hand_seen:
                self._debug(
                    f"new ally track card={card} ignored; "
                    "card is not in current, previous, or recent hand"
                )
                continue
            cell = self._infer_cell_from_clock([match], arena_px, clock_boxes, card)
            self._debug(f"recording own action from new ally track card={card} slot={slot_idx} cell={cell}")
            self._append_action(now=now, card=card, slot_idx=slot_idx, cell=cell)

    def _find_recent_slot(self, card, hand):
        for idx, hand_card in enumerate(hand):
            if hand_card == card:
                return idx
        for idx, hand_card in enumerate(self.last_hand):
            if hand_card == card:
                return idx
        return None

    def _append_action(
        self,
        *,
        now,
        card,
        slot_idx,
        cell,
        video_time_s=None,
        rolling_spell_track_id=None,
        played_via=None,
    ):
        if now > OWN_ACTION_START_TIME_LEFT_S:
            self._debug(f"own action ignored before start threshold card={card} time_left={now}")
            return False
        if played_via != "mirror" and self._is_duplicate_action(now, card, slot_idx):
            self._debug(f"duplicate own action ignored card={card} slot={slot_idx} time_left={now}")
            return False
        self.actions.append(
            OwnActionEvent(
                time_left_s=now,
                video_time_s=video_time_s,
                card=card,
                slot_idx=slot_idx,
                cell=cell,
                rolling_spell_track_id=rolling_spell_track_id,
                played_via=played_via,
            )
        )
        return True

    def _is_duplicate_action(self, now, card, slot_idx):
        if not self.actions:
            return False
        last_action = self.actions[-1]
        if last_action.card != card:
            return False
        elapsed_s = last_action.time_left_s - now
        if not 0 <= elapsed_s < OWN_ACTION_DUPLICATE_WINDOW_S:
            return False
        last_slot_idx = last_action.slot_idx
        return last_slot_idx == slot_idx or last_slot_idx is None or slot_idx is None

    def _infer_cell(self, tracks, arena_px):
        if not tracks:
            return None
        troop = tracks[0].troop
        cell = ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, arena_px)
        self._debug(f"inferred cell from track id={troop.track_id} class={troop.class_name}: {cell}")
        return cell

    def _infer_cell_from_clock(self, tracks, arena_px, clock_boxes, card=None):
        best = None
        matching_tracks = self._matching_pending_tracks(tracks, card)
        if card is not None and self._requires_matching_track(card) and not matching_tracks:
            self._debug(
                f"no matching recent ally track for pending card={card}; "
                "not using unrelated deploy clock"
            )
            return None
        candidate_tracks = matching_tracks or tracks
        for match in candidate_tracks:
            troop = match.troop
            for clock in clock_boxes:
                if clock["team"] != "ally" or clock["confidence"] < 0.5:
                    continue
                horizontal_gap = abs(clock["center_x"] - troop.center_x)
                vertical_gap = clock["center_y"] - troop.center_y
                if horizontal_gap > 100 or not (-40 <= vertical_gap <= 220):
                    continue
                score = horizontal_gap + abs(vertical_gap - 80) * 0.5
                if best is None or score < best[0]:
                    best = (score, clock, troop)
        if best is None:
            if self._allows_track_fallback(card):
                return self._infer_cell(candidate_tracks, arena_px)
            self._debug(
                f"no matching ally deploy clock for card={card}; "
                "not falling back to troop center"
            )
            return None
        _, clock, troop = best
        cell = ACTION_GRID.pixel_to_cell(clock["center_x"], clock["center_y"], arena_px)
        self._debug(f"inferred cell from ally clock near track id={troop.track_id} class={troop.class_name}: {cell}")
        return cell

    def _is_spell_card(self, card):
        return CARD_METADATA.get(card, {}).get("kind") == "spell"

    def _has_direct_unit_mapping(self, card):
        return any(mapped_card == card for mapped_card in DIRECT_UNIT_TO_CARD.values())

    def _matching_pending_tracks(self, tracks, card):
        if card is None:
            return []
        pending_labels = PENDING_UNIT_LABELS.get(card)
        if pending_labels is not None:
            return [match for match in tracks if match.troop.class_name in pending_labels]
        return [match for match in tracks if DIRECT_UNIT_TO_CARD.get(match.troop.class_name) == card]

    def _requires_matching_track(self, card):
        return card in PENDING_UNIT_LABELS or self._has_direct_unit_mapping(card)

    def _allows_track_fallback(self, card):
        return card in OWN_ACTION_TRACK_FALLBACK_CARDS
