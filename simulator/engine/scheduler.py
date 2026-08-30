"""scheduler mechanics for the deterministic battle engine."""

from __future__ import annotations

from ._base import *


class SchedulerMixin:
    @property
    def decision_interval_ticks(self) -> int:
        """Default 250 ms policy cadence, separate from the physics tick."""

        return max(1, 250_000 // self.ruleset.tick_us)

    def step(self, state: BattleState, actions: Iterable[SimAction] = ()) -> tuple[SimEvent, ...]:
        """Advance exactly one canonical physics tick and return new events."""

        if state.terminal:
            return ()
        self._verify_state_ruleset(state)
        event_start = len(state.events)
        self._regenerate_elixir(state)
        self._advance_card_cycle(state)
        self.apply_actions(state, actions)
        self._advance_deployments(state)
        self._advance_area_effects(state)
        self._advance_statuses_and_lifetimes(state)
        self._advance_concealment(state)
        self._sync_carried_entities(state)
        self._invalidate_and_acquire_targets(state)
        self._move_entities(state)
        self._sync_carried_entities(state)
        self._separate_entities(state)
        self._advance_attacks(state)
        self._advance_projectiles(state)
        destroyed_towers = self._resolve_deaths(state)
        self._resolve_tower_outcomes(state, destroyed_towers)
        self._advance_match_clock(state)
        state.tick += 1
        if self.validate_every_tick:
            self.validate_state(state)
        return tuple(state.events[event_start:])

    def apply_actions(self, state: BattleState, actions: Iterable[SimAction]) -> tuple[ActionResult, ...]:
        """Apply at most one action per player in stable player order."""

        by_player: dict[int, SimAction] = {}
        duplicate_players: set[int] = set()
        for action in actions:
            if action.player in by_player:
                duplicate_players.add(action.player)
            else:
                by_player[action.player] = action
        results: list[ActionResult] = []
        for player in sorted(duplicate_players):
            self._emit(state, "action_rejected", player=player, reason="multiple_actions_in_tick")
            results.append(ActionResult(player, False, "multiple_actions_in_tick"))
            by_player.pop(player, None)
        for player in sorted(by_player):
            action = by_player[player]
            reason = self.validate_action(state, action)
            if reason is not None:
                self._emit(state, "action_rejected", player=player, reason=reason)
                results.append(ActionResult(player, False, reason))
                continue
            if isinstance(action, WaitAction):
                results.append(ActionResult(player, True))
                continue
            if isinstance(action, UseAbilityAction):
                # The action exists from schema v1 so current forms can extend
                # the engine without an action-space migration. Base v0.1 has
                # no ability-bearing definition and therefore fails closed.
                self._emit(state, "action_rejected", player=player, reason="ability_not_supported")
                results.append(ActionResult(player, False, "ability_not_supported"))
                continue
            card_id = state.players[player].hand[action.card_slot]
            self._play_card(state, action, card_id)
            results.append(ActionResult(player, True, card_id=card_id))
        return tuple(results)

    def validate_action(self, state: BattleState, action: SimAction) -> str | None:
        if type(action.player) is not int or action.player not in (0, 1):
            return "invalid_player"
        if state.terminal:
            return "match_ended"
        if isinstance(action, WaitAction):
            return None
        if isinstance(action, UseAbilityAction):
            return "ability_not_supported"
        if not isinstance(action, PlayCardAction):
            return "unknown_action"
        player = state.players[action.player]
        if type(action.card_slot) is not int or not (0 <= action.card_slot < len(player.hand)):
            return "invalid_card_slot"
        card = self.ruleset.card(player.hand[action.card_slot])
        placement_card = card
        if card.card_id == "mirror":
            if player.last_played_card_id is None:
                return "mirror_no_target"
            placement_card = self.ruleset.card(player.last_played_card_id)
        effective_cost = self._effective_card_cost(player, card)
        if player.elixir_milli < effective_cost:
            return "insufficient_elixir"
        if card.card_id == "mirror" and player.last_played_card_id == "mirror":
            return "mirror_chain_not_allowed"
        if not self._legal_deployment(state, action.player, placement_card, action.cell):
            return "illegal_placement"
        return None

    def legal_cells(self, state: BattleState, player: int, card_id: str) -> tuple[tuple[int, int], ...]:
        card = self.ruleset.card(card_id)
        if card.card_id == "mirror":
            previous = state.players[player].last_played_card_id
            if previous is None or previous == "mirror":
                return ()
            card = self.ruleset.card(previous)
        return self._legal_cells_for_card(state, player, card)

    def legal_action_cells(
        self,
        state: BattleState,
        player: int,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return exact legal world cells for the first four hand slots.

        Policy observations need the legality of every hand slot, but calling
        :meth:`validate_action` once per grid cell repeats card resolution,
        elixir checks, action construction, and type dispatch.  This batched
        form performs those slot-level checks once and evaluates deployment
        legality directly.  It intentionally returns world coordinates; the
        observation adapter handles viewer-local mirroring.
        """

        if type(player) is not int or player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player!r}")
        if state.terminal:
            return ((), (), (), ())

        player_state = state.players[player]
        # These predicates are shared by every hand slot in one observation.
        # The old per-cell validate_action path recomputed them independently
        # for each card and each candidate coordinate.
        territory_cells = self._deployment_territory_cells(state, player)
        deployment_obstacles = self._deployment_obstacles(state)
        legal_by_slot: list[tuple[tuple[int, int], ...]] = []
        for raw_card_id in player_state.hand[:4]:
            card = self.ruleset.card(raw_card_id)
            placement_card = card
            if card.card_id == "mirror":
                previous = player_state.last_played_card_id
                if previous is None or previous == "mirror":
                    legal_by_slot.append(())
                    continue
                placement_card = self.ruleset.card(previous)
            if player_state.elixir_milli < self._effective_card_cost(player_state, card):
                legal_by_slot.append(())
                continue
            legal_by_slot.append(
                self._legal_cells_for_card(
                    state,
                    player,
                    placement_card,
                    territory_cells=territory_cells,
                    deployment_obstacles=deployment_obstacles,
                )
            )

        while len(legal_by_slot) < 4:
            legal_by_slot.append(())
        return tuple(legal_by_slot)

    def run_match(
        self,
        state: BattleState,
        controllers: tuple[Controller | Callable[["BattleEngine", BattleState, int], SimAction], Controller | Callable[["BattleEngine", BattleState, int], SimAction]] | None = None,
        *,
        decision_interval_ticks: int | None = None,
        max_ticks: int | None = None,
    ) -> BattleState:
        """Run until a King falls, regulation resolves, or tiebreak completes."""

        cadence = self.decision_interval_ticks if decision_interval_ticks is None else decision_interval_ticks
        if cadence <= 0:
            raise ValueError("decision_interval_ticks must be positive")
        total_us = self.ruleset.match.regulation_us + self.ruleset.match.overtime_us
        hard_limit = total_us // self.ruleset.tick_us + 2 if max_ticks is None else max_ticks
        if hard_limit <= 0:
            raise ValueError("max_ticks must be positive")
        while not state.terminal and state.tick < hard_limit:
            actions: list[SimAction] = []
            if controllers is not None and state.tick % cadence == 0:
                for player, controller in enumerate(controllers):
                    chooser = controller.choose_action if hasattr(controller, "choose_action") else controller
                    actions.append(chooser(self, state, player))
            self.step(state, actions)
        if not state.terminal:
            self._end_match(state, None, "runner_tick_limit")
        return state

    def _effective_card_cost(self, player: PlayerState, card: CardDefinition) -> int:
        """Return a card's payable cost, including Mirror's dynamic surcharge."""

        if card.card_id != "mirror" or player.last_played_card_id is None:
            return card.elixir_milli
        previous = self.ruleset.card(player.last_played_card_id)
        return min(self.ruleset.match.max_elixir_milli, previous.elixir_milli + 1_000)

    def _card_cycle_cooldown_us(self, state: BattleState) -> int:
        """Return the hand-loading delay for a newly exposed Next card."""

        if self._elixir_interval(state.elapsed_us) != self.ruleset.match.normal_elixir_interval_us:
            return _CARD_CYCLE_COOLDOWN_ACCELERATED_US
        return _CARD_CYCLE_COOLDOWN_SINGLE_US

    def _advance_card_cycle(self, state: BattleState) -> None:
        """Advance each player's Next-card loading timer by one physics tick."""

        dt = self.ruleset.tick_us
        hand_size = self.ruleset.match.hand_size
        for player in state.players:
            if player.next_card_cooldown_us > 0:
                player.next_card_cooldown_us = max(
                    0, player.next_card_cooldown_us - dt
                )
            if (
                player.next_card_cooldown_us == 0
                and len(player.hand) < hand_size
                and player.draw_pile
            ):
                player.hand.append(player.draw_pile.pop(0))
                player.next_card_cooldown_us = self._card_cycle_cooldown_us(state)

    def _regenerate_elixir(self, state: BattleState) -> None:
        interval = self._elixir_interval(state.elapsed_us)
        previous_interval = self._elixir_interval(
            max(0, state.elapsed_us - self.ruleset.tick_us)
        )
        for player in state.players:
            if player.elixir_milli >= self.ruleset.match.max_elixir_milli:
                player.elixir_milli = self.ruleset.match.max_elixir_milli
                player.elixir_remainder = 0
                continue
            if previous_interval != interval:
                # The remainder is a fraction of the old regeneration period,
                # not an absolute time.  Rescale it at 2x/3x transitions so
                # the in-flight fractional elixir is conserved instead of
                # granting or losing up to one milli-elixir at each boundary.
                player.elixir_remainder = (
                    player.elixir_remainder * interval // previous_interval
                )
            numerator = self.ruleset.tick_us * ELIXIR_SCALE + player.elixir_remainder
            gain, player.elixir_remainder = divmod(numerator, interval)
            player.elixir_milli = min(self.ruleset.match.max_elixir_milli, player.elixir_milli + gain)
            if player.elixir_milli == self.ruleset.match.max_elixir_milli:
                player.elixir_remainder = 0

    def _elixir_interval(self, elapsed_us: int) -> int:
        regulation = self.ruleset.match.regulation_us
        total = regulation + self.ruleset.match.overtime_us
        if elapsed_us >= total - 60 * SECOND_US:
            return self.ruleset.match.triple_elixir_interval_us
        if elapsed_us >= regulation - 60 * SECOND_US:
            return self.ruleset.match.double_elixir_interval_us
        return self.ruleset.match.normal_elixir_interval_us
