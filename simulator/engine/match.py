"""match mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class MatchMixin:
    def _resolve_tower_outcomes(self, state: BattleState, destroyed: list[EntityState]) -> None:
        if not destroyed or state.terminal:
            return
        king_deaths = {tower.owner for tower in destroyed if tower.role == "king"}
        for tower in destroyed:
            opponent = state.players[1 - tower.owner]
            if tower.role == "king":
                opponent.crowns = 3
            elif tower.owner not in king_deaths:
                # A same-batch Princess Tower death is subsumed by that
                # owner's three-crown King Tower loss.  A Princess Tower on
                # the opposite side still awards its legitimate crown in a
                # rare simultaneous cross-side resolution.
                opponent.crowns = min(3, opponent.crowns + 1)
                self._activate_king(state, tower.owner, "princess_tower_destroyed")
        if len(king_deaths) == 2:
            self._collapse_remaining_crown_towers(state, king_deaths)
            self._end_match(state, None, "simultaneous_king_destruction")
        elif king_deaths:
            self._collapse_remaining_crown_towers(state, king_deaths)
            self._end_match(state, 1 - next(iter(king_deaths)), "king_tower_destroyed")
        elif state.phase == "overtime":
            crowns = (state.players[0].crowns, state.players[1].crowns)
            if crowns[0] != crowns[1]:
                self._end_match(state, 0 if crowns[0] > crowns[1] else 1, "overtime_sudden_death")

    def _collapse_remaining_crown_towers(
        self, state: BattleState, owners: set[int]
    ) -> None:
        """Destroy each king owner's surviving Princess Towers terminally."""

        for owner in sorted(owners):
            for tower in self._towers_for(state, owner):
                if tower.role != "king" and tower.alive:
                    tower.hp = 0
        # Route the automatic collapse through the ordinary death resolver so
        # navigation invalidation and tower-destroyed events remain canonical.
        self._resolve_deaths(state)

    def _advance_match_clock(self, state: BattleState) -> None:
        if state.terminal:
            # Combat resolution may have ended the match earlier in this
            # physics tick.  ``step`` still advances the tick counter below,
            # so account for the completed interval here as well; otherwise
            # terminal states report tick N+1 with the elapsed time of tick N.
            state.elapsed_us += self.ruleset.tick_us
            return
        state.elapsed_us += self.ruleset.tick_us
        regulation = self.ruleset.match.regulation_us
        total = regulation + self.ruleset.match.overtime_us
        if state.phase == "regulation" and state.elapsed_us >= regulation:
            crowns = (state.players[0].crowns, state.players[1].crowns)
            if crowns[0] != crowns[1]:
                self._end_match(state, 0 if crowns[0] > crowns[1] else 1, "regulation_crowns")
                return
            state.phase = "overtime"
            self._emit(state, "match_phase_changed", phase="overtime")
        if state.elapsed_us >= total:
            if self.ruleset.match.tiebreak_enabled:
                self._resolve_tiebreak(state)
            else:
                self._end_match(state, None, "time_draw")

    def _resolve_tiebreak(self, state: BattleState) -> None:
        # The tiebreak transition removes all active combatants and transient
        # effects before either a tower drain or a draw is resolved.  Draws
        # need the same cleanup as wins; otherwise terminal observations can
        # contain troops and projectiles that can never act again.
        self._remove_entities_for_tiebreak(state)
        alive_towers = [entity for entity in self._alive_entities(state) if entity.kind == "tower"]
        if not alive_towers:
            self._end_match(state, None, "tiebreak_draw")
            return
        # Tiebreak drains the same raw HP from every surviving Crown Tower;
        # therefore the lowest absolute current HP falls first.
        lowest_hp = min(tower.hp for tower in alive_towers)
        minimum = [tower for tower in alive_towers if tower.hp == lowest_hp]
        owners = {tower.owner for tower in minimum}
        if len(owners) != 1:
            self._end_match(state, None, "tiebreak_equal_lowest_hp")
            return

        target = min(minimum, key=lambda tower: tower.uid)
        # The tiebreak is an equal raw-HP drain on every tower.  Applying the
        # full drain before resolving deaths keeps terminal tower snapshots and
        # damage-based rewards consistent with the visible game transition.
        for tower in alive_towers:
            tower.hp = max(0, tower.hp - lowest_hp)
        destroyed = self._resolve_deaths(state)
        if any(tower.role == "king" for tower in destroyed):
            # King destruction still has the normal automatic Princess Tower
            # collapse and three-crown semantics.
            self._resolve_tower_outcomes(state, destroyed)
            return

        for tower in destroyed:
            opponent = state.players[1 - tower.owner]
            opponent.crowns = min(3, opponent.crowns + 1)
            self._activate_king(state, tower.owner, "tiebreak_tower_destroyed")
        self._end_match(state, 1 - target.owner, "tiebreak_lowest_hp")

    def _remove_entities_for_tiebreak(self, state: BattleState) -> None:
        """Clear active combatants and transient effects before tiebreak."""

        for entity in sorted(state.entities.values(), key=lambda row: row.uid):
            if not entity.alive or entity.kind == "tower":
                continue
            entity.alive = False
            entity.hp = 0
            entity.target_uid = None
            entity.pending_target_uid = None
            entity.secondary_pending_target_uid = None
            entity.jump_target_uid = None
            entity.carried_by_uid = None
            entity.carried_offset_x_mtile = 0
            entity.carried_offset_y_mtile = 0
            entity.statuses.clear()
            entity.navigation_waypoints.clear()
            entity.navigation_cursor = 0
            self._emit(
                state,
                "tiebreak_entity_removed",
                uid=entity.uid,
                player=entity.owner,
                card_id=entity.card_id,
            )

        removed_projectiles = 0
        for projectile in sorted(state.projectiles.values(), key=lambda row: row.uid):
            if projectile.alive:
                projectile.alive = False
                removed_projectiles += 1
        if removed_projectiles:
            self._emit(
                state,
                "tiebreak_projectiles_removed",
                count=removed_projectiles,
            )

        removed_effects = 0
        for effect in sorted(state.effects.values(), key=lambda row: row.uid):
            if effect.alive:
                effect.alive = False
                effect.remaining_us = 0
                removed_effects += 1
        if removed_effects:
            self._emit(state, "tiebreak_effects_removed", count=removed_effects)

    def _activate_king(self, state: BattleState, player: int, reason: str) -> None:
        if state.players[player].king_active:
            return
        state.players[player].king_active = True
        self._emit(state, "king_activated", player=player, reason=reason)

    def _end_match(self, state: BattleState, winner: int | None, reason: str) -> None:
        if state.terminal:
            return
        state.terminal = True
        state.phase = "ended"
        state.winner = winner
        state.terminal_reason = reason
        self._emit(
            state,
            "match_ended",
            winner=winner,
            reason=reason,
            crowns_0=state.players[0].crowns,
            crowns_1=state.players[1].crowns,
        )

    @staticmethod
    def _alive_entities(state: BattleState) -> list[EntityState]:
        return [
            state.entities[uid]
            for uid in sorted(state.entities)
            if state.entities[uid].alive and state.entities[uid].hp > 0
        ]

    @staticmethod
    def _towers_for(state: BattleState, player: int) -> list[EntityState]:
        return [
            state.entities[uid]
            for uid in sorted(state.entities)
            if state.entities[uid].kind == "tower" and state.entities[uid].owner == player
        ]

    def _tower(self, state: BattleState, player: int, role: str) -> EntityState:
        return next(tower for tower in self._towers_for(state, player) if tower.role == role)

    @staticmethod
    def _allocate_uid(state: BattleState) -> int:
        uid = state.next_uid
        state.next_uid += 1
        return uid

    @staticmethod
    def _emit(state: BattleState, kind: str, **data: str | int | bool | None) -> None:
        event = SimEvent.create(state.tick, state.event_sequence, kind, **data)
        state.event_sequence += 1
        state.events.append(event)

class DeterministicCycleController:
    """Simple reproducible smoke-test controller for complete headless matches."""

    def __init__(self, *, lane: str = "alternate") -> None:
        if lane not in {"left", "right", "alternate"}:
            raise ValueError("lane must be left, right, or alternate")
        self.lane = lane

    def choose_action(self, engine: BattleEngine, state: BattleState, player: int) -> SimAction:
        hand = state.players[player].hand
        player_state = state.players[player]
        affordable = [
            slot
            for slot, card_id in enumerate(hand)
            if (
                card_id != "mirror"
                or player_state.last_played_card_id not in {None, "mirror"}
            )
            and engine._effective_card_cost(player_state, engine.ruleset.card(card_id))
            <= player_state.elixir_milli
        ]
        if not affordable:
            return WaitAction(player)
        slot = min(
            affordable,
            key=lambda index: (
                engine._effective_card_cost(player_state, engine.ruleset.card(hand[index])),
                index,
            ),
        )
        card = engine.ruleset.card(hand[slot])
        use_left = self.lane == "left" or (
            self.lane == "alternate" and state.players[player].cards_played % 2 == 0
        )
        col = 3 if use_left else 14
        if card.kind == "spell" and card.card_id == "fireball":
            row = 6 if player == 0 else 25
        elif card.kind == "spell":
            row = 19 if player == 0 else 12
        elif card.kind == "building":
            col, row = 8, (20 if player == 0 else 9)
        else:
            row = 23 if player == 0 else 8
        preferred = (col, row)
        action = PlayCardAction(player, slot, preferred)
        if engine.validate_action(state, action) is None:
            return action
        legal = engine.legal_cells(state, player, card.card_id)
        if not legal:
            return WaitAction(player)
        return PlayCardAction(player, slot, legal[0])
