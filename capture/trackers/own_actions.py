from dataclasses import dataclass

from card_metadata import CARD_METADATA
from features.action_space import ACTION_GRID


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

    def update(self, game_state, arena_px):
        now = game_state.total_remaining_s
        hand = game_state.hud.hand_cards
        elixir = game_state.hud.elixir_self

        self._detect_slot_drops(hand, elixir, now)
        self._confirm_pending(game_state, arena_px, elixir, now)
        self.last_hand = hand[:]
        self.last_elixir = elixir

    def _detect_slot_drops(self, hand, elixir, now):
        for idx, (prev, cur) in enumerate(zip(self.last_hand, hand)):
            if prev is not None and cur is None:
                self.pending.append(PendingOwnPlay(
                    card=prev,
                    slot_idx=idx,
                    started_at_s=now,
                    elixir_before=elixir,
                ))

    def _confirm_pending(self, game_state, arena_px, elixir, now):
        new_tracks = []
        for match in game_state.own_units:
            track_id = match.troop.track_id
            if track_id is None:
                continue
            if track_id not in self.seen_ally_tracks:
                new_tracks.append(match)
                self.seen_ally_tracks.add(track_id)

        still_pending = []
        for pending in self.pending:
            if pending.started_at_s - now > 1.5:
                continue

            cost = CARD_METADATA.get(pending.card, {}).get("elixir_cost")
            elixir_confirms = (
                self.last_elixir is not None
                and cost is not None
                and self.last_elixir - elixir >= max(1.0, cost - 1.0)
            )
            placed_cell = self._infer_cell(new_tracks, arena_px)
            if elixir_confirms or placed_cell is not None:
                self.actions.append({
                    "time_left_s": now,
                    "card": pending.card,
                    "slot_idx": pending.slot_idx,
                    "cell": placed_cell,
                })
            else:
                still_pending.append(pending)

        self.pending = still_pending

    def _infer_cell(self, new_tracks, arena_px):
        if not new_tracks:
            return None

        troop = new_tracks[0].troop
        return ACTION_GRID.pixel_to_cell(troop.center_x, troop.center_y, arena_px)

