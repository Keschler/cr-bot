"""Training-only expert policies for the pinned deterministic opponent.

The expert in this module is a teacher/counter-policy, not an actor input.  It
receives authoritative simulator state while generating demonstrations, while
the recurrent actor is trained only on the public V2 observation that led to
the teacher action.  Keeping the teacher separate makes the information
boundary explicit and allows the expert to be used for evaluation audits.
"""

from __future__ import annotations

from typing import Any

try:
    from ..actions import PlayCardAction, SimAction, WaitAction
except ImportError:  # pragma: no cover - top-level ``rl`` layout
    from simulator.actions import PlayCardAction, SimAction, WaitAction


class DeterministicCounterController:
    """A conservative Hog-cycle counter for ``DeterministicCycleController``.

    The deterministic enemy spends elixir on the cheapest card available and
    alternates lanes.  This policy exploits only that documented behavior:

    * defend a crossed river push with a central Cannon or ranged support;
    * use Fireball on the weakest enemy princess tower when available;
    * save elixir for Hog Rider rather than starving it behind cheap cards;
    * otherwise cycle the remaining deck with the existing deterministic
      controller's legal placement policy.

    It is deliberately used as a teacher, so its authoritative-state access
    never becomes part of the public actor observation.
    """

    def __init__(self) -> None:
        try:
            from ..engine import DeterministicCycleController
        except ImportError:  # pragma: no cover - top-level ``rl`` layout
            from simulator.engine import DeterministicCycleController

        self._fallback = DeterministicCycleController(lane="alternate")

    @staticmethod
    def _play_if_affordable(
        engine: Any,
        state: Any,
        player: int,
        card_id: str,
        preferred_cell: tuple[int, int],
    ) -> SimAction | None:
        player_state = state.players[player]
        for slot, hand_card in enumerate(player_state.hand):
            if hand_card != card_id:
                continue
            card = engine.ruleset.card(hand_card)
            if engine._effective_card_cost(player_state, card) > player_state.elixir_milli:
                return None
            action = PlayCardAction(player, slot, preferred_cell)
            if engine.validate_action(state, action) is None:
                return action
            legal_cells = engine.legal_cells(state, player, card_id)
            if not legal_cells:
                return None
            return PlayCardAction(player, slot, legal_cells[0])
        return None

    @staticmethod
    def _enemy_troops(state: Any, player: int) -> list[Any]:
        return [
            entity
            for entity in state.entities.values()
            if entity.alive and entity.owner != player and entity.kind == "troop"
        ]

    @staticmethod
    def _has_live_card_entity(state: Any, player: int, card_id: str) -> bool:
        return any(
            entity.alive and entity.owner == player and entity.card_id == card_id
            for entity in state.entities.values()
        )

    def choose_action(self, engine: Any, state: Any, player: int) -> SimAction:
        enemy_troops = self._enemy_troops(state, player)
        crossed_troops = [
            entity
            for entity in enemy_troops
            if (entity.y_mtile >= 15_000 if player == 0 else entity.y_mtile <= 17_000)
        ]
        cannon_alive = self._has_live_card_entity(state, player, "cannon")

        # The enemy alternates lanes, so a central building is the highest
        # value defensive answer once units cross the river.
        if crossed_troops and not cannon_alive:
            defensive_cells = {
                "cannon": (8, 21) if player == 0 else (8, 10),
                "musketeer": (8, 23) if player == 0 else (8, 8),
                "skeletons": (8, 22) if player == 0 else (8, 9),
                "log": (3, 21) if player == 0 else (14, 10),
                "fireball": (3, 21) if player == 0 else (14, 10),
            }
            for card_id in ("cannon", "musketeer", "skeletons", "log", "fireball"):
                action = self._play_if_affordable(
                    engine,
                    state,
                    player,
                    card_id,
                    defensive_cells[card_id],
                )
                if action is not None:
                    return action

        # Convert every available Fireball into direct tower pressure.  The
        # engine's tower coordinates are already world coordinates, so there is
        # no viewer-relative mirroring in this teacher.
        enemy_towers = [
            entity
            for entity in state.entities.values()
            if entity.alive and entity.owner != player and entity.kind == "tower" and entity.role != "king"
        ]
        if enemy_towers:
            weakest = min(enemy_towers, key=lambda entity: (entity.hp, entity.role))
            action = self._play_if_affordable(
                engine,
                state,
                player,
                "fireball",
                (weakest.x_mtile // 1_000, weakest.y_mtile // 1_000),
            )
            if action is not None:
                return action

        # Do not spend a cheap card when Hog is in hand: doing so keeps the
        # four-elixir win condition permanently unaffordable in this ruleset.
        action = self._play_if_affordable(
            engine,
            state,
            player,
            "hog-rider",
            (3, 17) if player == 0 else (14, 14),
        )
        if action is not None:
            return action
        if "hog-rider" in state.players[player].hand and not crossed_troops:
            return WaitAction(player)

        # Cycle the other cards only when Hog is not waiting in hand.  The
        # fallback also applies the simulator's legal-cell recovery logic.
        return self._fallback.choose_action(engine, state, player)


def deterministic_counter_action(environment: Any, _public_observation: Any, player: int) -> SimAction:
    """Callback adapter for :class:`rl.collector.RecurrentRolloutCollector`."""

    state = getattr(environment, "state", None)
    if state is None:
        raise RuntimeError("counter expert received an uninitialized environment")
    # The controller is cached on the environment so its fallback cycle state
    # is isolated between lanes without leaking anything into actor tensors.
    controller = getattr(environment, "_counter_expert_controller", None)
    if controller is None:
        controller = DeterministicCounterController()
        setattr(environment, "_counter_expert_controller", controller)
    return controller.choose_action(environment.engine, state, player)


__all__ = [
    "DeterministicCounterController",
    "deterministic_counter_action",
]
