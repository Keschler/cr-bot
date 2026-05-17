from dataclasses import dataclass
import os

from card_metadata import CARD_METADATA
from constants import (
    OWN_ACTION_DUPLICATE_WINDOW_S,
    OWN_ACTION_RECENT_HAND_WINDOW_S,
    OWN_ACTION_RECENT_TRACK_WINDOW_S,
    OWN_ACTION_START_TIME_LEFT_S,
    OWN_ACTION_TRACK_AFTER_DROP_WINDOW_S,
    OWN_ACTION_TRACK_FALLBACK_CARDS,
)
from extractors.spell_deploy import SpellDeployLocator
from features.action_space import ACTION_GRID
from trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD


@dataclass
class PendingOwnPlay:
    card: str
    slot_idx: int
    started_at_s: float
    elixir_before: float
    confirmed: bool = False


@dataclass
class RecentAllyTrack:
    match: object
    first_seen_s: float
    last_seen_s: float


class OwnActionTracker:
    def __init__(self) -> None:
        self.last_hand: list[str | None] = [None, None, None, None]
        self.last_elixir: float | None = None
        self.pending: list[PendingOwnPlay] = []
        self.seen_ally_tracks: set[int] = set()
        self.recent_ally_tracks: dict[int, RecentAllyTrack] = {}
        self.actions: list[dict] = []
        self.recent_hand_seen: dict[str, float] = {}
        self.spell_deploy_locator = SpellDeployLocator()
        self.debug = os.environ.get("DEBUG_OWN_ACTIONS") == "1"

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[own_actions] {message}")

    def update(self, game_state, arena_px, frame=None, clock_boxes=None):
        clock_boxes = clock_boxes or []
        now = game_state.total_remaining_s
        hand = game_state.hud.hand_cards
        elixir = game_state.hud.elixir_self
        self._debug(
            f"update now={now} elixir={elixir:.2f} "
            f"last_elixir={self.last_elixir} hand={hand} "
            f"last_hand={self.last_hand} pending={len(self.pending)} "
            f"own_units={len(game_state.own_units)} "
            f"clocks={len(clock_boxes)}"
        )

        self._detect_slot_drops(hand, elixir, now)
        self._confirm_pending(
            game_state,
            arena_px,
            elixir,
            now,
            frame=frame,
            clock_boxes=clock_boxes,
        )
        self._remember_hand(hand, now)
        self.last_hand = hand[:]
        self.last_elixir = elixir
        self._debug(
            f"state saved last_hand={self.last_hand} "
            f"last_elixir={self.last_elixir:.2f} actions={len(self.actions)}"
        )

    def _detect_slot_drops(self, hand, elixir, now):
        for idx, (prev, cur) in enumerate(zip(self.last_hand, hand)):
            self._debug(f"slot {idx}: prev={prev} cur={cur}")
            if prev is not None and cur is None:
                self._debug(
                    f"slot {idx} drop detected: card={prev} "
                    f"started_at={now} elixir_before={elixir:.2f}"
                )
                self.pending.append(PendingOwnPlay(
                    card=prev,
                    slot_idx=idx,
                    started_at_s=now,
                    elixir_before=elixir,
                ))
            elif prev is not None and cur != prev:
                self._debug(
                    f"slot {idx} changed but not treated as drop: "
                    f"prev={prev} cur={cur}"
                )

    def _remember_hand(self, hand, now):
        for card in hand:
            if card is not None:
                self.recent_hand_seen[card] = now

        stale_cards = [
            card
            for card, seen_at in self.recent_hand_seen.items()
            if seen_at - now > OWN_ACTION_RECENT_HAND_WINDOW_S
        ]
        for card in stale_cards:
            del self.recent_hand_seen[card]

    def _confirm_pending(self, game_state, arena_px, elixir, now, frame=None, clock_boxes=None):
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
                    f"new ally track id={track_id} "
                    f"class={match.troop.class_name} "
                    f"center=({match.troop.center_x:.1f}, {match.troop.center_y:.1f})"
                )
            self._remember_ally_track(match, now)
        if not new_tracks:
            self._debug("no new ally tracks")
        self._forget_stale_ally_tracks(now)

        had_pending = bool(self.pending)
        still_pending = []
        for pending in self.pending:
            is_spell = self._is_spell_card(pending.card)
            cost = CARD_METADATA.get(pending.card, {}).get("elixir_cost")
            elixir_drop = None if self.last_elixir is None else self.last_elixir - elixir
            required_drop = None if cost is None else max(1.0, cost - 1.0)
            elixir_confirms = (
                self.last_elixir is not None
                and cost is not None
                and elixir_drop >= required_drop
            )
            placed_cell = self._infer_pending_cell(
                pending,
                self._recent_tracks_for_pending(pending, now),
                arena_px,
                frame,
                clock_boxes,
            )
            self._debug(
                f"pending check card={pending.card} slot={pending.slot_idx} "
                f"cost={cost} elixir_drop={elixir_drop} "
                f"required_drop={required_drop} elixir_confirms={elixir_confirms} "
                f"placed_cell={placed_cell}"
            )
            confirms = placed_cell is not None
            if confirms:
                reasons = []
                if elixir_confirms and not is_spell:
                    reasons.append("elixir")
                if placed_cell is not None:
                    reasons.append("deploy_ui" if is_spell else "cell")
                self._debug(
                    f"confirmed own action card={pending.card} "
                    f"slot={pending.slot_idx} reasons={'+'.join(reasons)}"
                )
                self._append_action(
                    now=now,
                    card=pending.card,
                    slot_idx=pending.slot_idx,
                    cell=placed_cell,
                )
            else:
                self._debug(
                    f"pending kept card={pending.card} slot={pending.slot_idx}"
                )
                still_pending.append(pending)

        self.pending = still_pending
        if not had_pending and not self.pending:
            self._record_new_track_actions(
                new_tracks,
                game_state.hud.hand_cards,
                arena_px,
                now,
                clock_boxes,
            )

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
        self.recent_ally_tracks[track_id] = RecentAllyTrack(
            match=match,
            first_seen_s=first_seen_s,
            last_seen_s=now,
        )

    def _forget_stale_ally_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, memory in self.recent_ally_tracks.items()
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
            cost = CARD_METADATA.get(pending.card, {}).get("elixir_cost")
            deploy = self.spell_deploy_locator.locate(
                frame,
                arena_px,
                pending.card,
                cost,
            )
            if deploy is None:
                self._debug(
                    f"spell deploy locator found no cell for card={pending.card}"
                )
                return None

            cell = ACTION_GRID.pixel_to_cell(deploy.center_x, deploy.center_y, arena_px)
            self._debug(
                f"inferred spell cell from deploy UI card={pending.card}: {cell}"
            )
            return cell

        return self._infer_cell_from_clock(candidate_tracks, arena_px, clock_boxes, pending.card)

    def _record_new_track_actions(self, new_tracks, hand, arena_px, now, clock_boxes):
        for match in new_tracks:
            card = DIRECT_UNIT_TO_CARD.get(match.troop.class_name)
            if card is None:
                self._debug(
                    f"new ally track not mapped to playable card: "
                    f"class={match.troop.class_name}"
                )
                continue
            if self._is_spell_card(card):
                self._debug(
                    f"new ally spell track ignored; spell actions require HUD drop "
                    f"and deploy UI locator: class={match.troop.class_name} card={card}"
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
            self._debug(
                f"recording own action from new ally track "
                f"card={card} slot={slot_idx} cell={cell}"
            )
            self._append_action(
                now=now,
                card=card,
                slot_idx=slot_idx,
                cell=cell,
            )

    def _find_recent_slot(self, card, hand):
        for idx, hand_card in enumerate(hand):
            if hand_card == card:
                return idx
        for idx, hand_card in enumerate(self.last_hand):
            if hand_card == card:
                return idx
        return None

    def _append_action(self, *, now, card, slot_idx, cell):
        if now > OWN_ACTION_START_TIME_LEFT_S:
            self._debug(
                f"own action ignored before start threshold card={card} "
                f"time_left={now}"
            )
            return False

        if self._is_duplicate_action(now, card, slot_idx):
            self._debug(
                f"duplicate own action ignored card={card} "
                f"slot={slot_idx} time_left={now}"
            )
            return False

        self.actions.append({
            "time_left_s": now,
            "card": card,
            "slot_idx": slot_idx,
            "cell": cell,
        })
        return True

    def _is_duplicate_action(self, now, card, slot_idx):
        if not self.actions:
            return False

        last_action = self.actions[-1]
        if last_action["card"] != card:
            return False

        elapsed_s = last_action["time_left_s"] - now
        if not 0 <= elapsed_s < OWN_ACTION_DUPLICATE_WINDOW_S:
            return False

        last_slot_idx = last_action["slot_idx"]
        return (
            last_slot_idx == slot_idx
            or last_slot_idx is None
            or slot_idx is None
        )

    def _infer_cell(self, tracks, arena_px):
        if not tracks:
            return None

        troop = tracks[0].troop
        cell = ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, arena_px)
        self._debug(
            f"inferred cell from track id={troop.track_id} "
            f"class={troop.class_name}: {cell}"
        )
        return cell

    def _infer_cell_from_clock(self, tracks, arena_px, clock_boxes, card=None):
        best = None
        matching_tracks = [
            match
            for match in tracks
            if card is not None and DIRECT_UNIT_TO_CARD.get(match.troop.class_name) == card
        ]
        if card is not None and self._has_direct_unit_mapping(card) and not matching_tracks:
            self._debug(
                f"no matching recent ally track for pending card={card}; "
                "not using unrelated deploy clock"
            )
            return None
        candidate_tracks = matching_tracks or tracks

        for match in candidate_tracks:
            troop = match.troop
            for clock in clock_boxes:
                if clock["team"] != "ally":
                    continue
                if clock["confidence"] < 0.5:
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
        self._debug(
            f"inferred cell from ally clock near track id={troop.track_id} "
            f"class={troop.class_name}: {cell}"
        )
        return cell

    def _is_spell_card(self, card):
        return CARD_METADATA.get(card, {}).get("kind") == "spell"

    def _has_direct_unit_mapping(self, card):
        return any(mapped_card == card for mapped_card in DIRECT_UNIT_TO_CARD.values())

    def _allows_track_fallback(self, card):
        return card in OWN_ACTION_TRACK_FALLBACK_CARDS
