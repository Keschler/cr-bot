from dataclasses import dataclass
import os

from card_metadata import CARD_METADATA
from constants import (
    OWN_ACTION_DUPLICATE_WINDOW_S,
    OWN_ACTION_RECENT_HAND_WINDOW_S,
    OWN_ACTION_START_TIME_LEFT_S,
)
from features.action_space import ACTION_GRID
from trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD


@dataclass
class PendingOwnPlay:
    card: str
    slot_idx: int
    started_at_s: float
    elixir_before: float
    confirmed: bool = False

class OwnActionTracker:
    def __init__(self) -> None:
        self.last_hand: list[str | None] = [None, None, None, None]
        self.last_elixir: float | None = None
        self.pending: list[PendingOwnPlay] = []
        self.seen_ally_tracks: set[int] = set()
        self.actions: list[dict] = []
        self.recent_hand_seen: dict[str, float] = {}
        self.debug = os.environ.get("DEBUG_OWN_ACTIONS") == "1"

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[own_actions] {message}")

    def update(self, game_state, arena_px):
        now = game_state.total_remaining_s
        hand = game_state.hud.hand_cards
        elixir = game_state.hud.elixir_self
        self._debug(
            f"update now={now} elixir={elixir:.2f} "
            f"last_elixir={self.last_elixir} hand={hand} "
            f"last_hand={self.last_hand} pending={len(self.pending)} "
            f"own_units={len(game_state.own_units)}"
        )

        self._detect_slot_drops(hand, elixir, now)
        self._confirm_pending(game_state, arena_px, elixir, now)
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

    def _confirm_pending(self, game_state, arena_px, elixir, now):
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
        if not new_tracks:
            self._debug("no new ally tracks")

        had_pending = bool(self.pending)
        still_pending = []
        for pending in self.pending:
            if pending.started_at_s - now > 1.5:
                self._debug(
                    f"pending expired card={pending.card} slot={pending.slot_idx} "
                    f"age={pending.started_at_s - now:.2f}s"
                )
                continue

            cost = CARD_METADATA.get(pending.card, {}).get("elixir_cost")
            elixir_drop = None if self.last_elixir is None else self.last_elixir - elixir
            required_drop = None if cost is None else max(1.0, cost - 1.0)
            elixir_confirms = (
                self.last_elixir is not None
                and cost is not None
                and elixir_drop >= required_drop
            )
            placed_cell = self._infer_cell(new_tracks, arena_px)
            self._debug(
                f"pending check card={pending.card} slot={pending.slot_idx} "
                f"cost={cost} elixir_drop={elixir_drop} "
                f"required_drop={required_drop} elixir_confirms={elixir_confirms} "
                f"placed_cell={placed_cell}"
            )
            if elixir_confirms or placed_cell is not None:
                reasons = []
                if elixir_confirms:
                    reasons.append("elixir")
                if placed_cell is not None:
                    reasons.append("cell")
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
            self._record_new_track_actions(new_tracks, game_state.hud.hand_cards, arena_px, now)

    def _record_new_track_actions(self, new_tracks, hand, arena_px, now):
        for match in new_tracks:
            card = DIRECT_UNIT_TO_CARD.get(match.troop.class_name)
            if card is None:
                self._debug(
                    f"new ally track not mapped to playable card: "
                    f"class={match.troop.class_name}"
                )
                continue

            slot_idx = self._find_recent_slot(card, hand)
            if slot_idx is None and card not in self.recent_hand_seen:
                self._debug(
                    f"new ally track card={card} ignored; "
                    "card is not in current, previous, or recent hand"
                )
                continue

            cell = ACTION_GRID.pixel_to_cell(
                match.troop.center_x,
                match.troop.center_y,
                arena_px,
            )
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

    def _infer_cell(self, new_tracks, arena_px):
        if not new_tracks:
            return None

        troop = new_tracks[0].troop
        cell = ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, arena_px)
        self._debug(
            f"inferred cell from track id={troop.track_id} "
            f"class={troop.class_name}: {cell}"
        )
        return cell
