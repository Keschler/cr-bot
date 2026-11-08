from __future__ import annotations

import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.engine import BattleEngine
from simulator.fixed import distance_mtile
from simulator.geometry import cell_center_mtile
from simulator.rl.foundation.opponent_pool import (
    AggressivePressureController,
    BeatdownTankSupportController,
    DefensiveCycleController,
    SiegeBaitController,
)
from simulator.ruleset import load_fixed_ruleset


_GOLEM_DECK = (
    "golem",
    "baby-dragon",
    "zap",
    "mega-minion",
    "lightning",
    "tornado",
    "night-witch",
    "barbarian-barrel",
)
_AIR_DECK = (
    "lava-hound",
    "balloon",
    "baby-dragon",
    "minions",
    "lightning",
    "arrows",
    "guards",
    "tombstone",
)
_X_BOW_DECK = (
    "x-bow",
    "tesla",
    "archers",
    "skeletons",
    "fireball",
    "log",
    "knight",
    "ice-spirit",
)
_MINER_DECK = (
    "miner",
    "golem",
    "lightning",
    "tornado",
    "night-witch",
    "baby-dragon",
    "mega-minion",
    "barbarian-barrel",
)
_AGGRESSIVE_DECK = (
    "hog-rider",
    "ice-spirit",
    "skeletons",
    "cannon",
    "musketeer",
    "fireball",
    "log",
    "ice-golem",
)
_SIEGE_DECK = (
    "x-bow",
    "goblin-barrel",
    "rocket",
    "tesla",
    "log",
    "ice-spirit",
    "knight",
    "skeletons",
)


def _battle(deck: tuple[str, ...], player: int, hand: tuple[str, ...], elixir: int):
    ruleset = load_fixed_ruleset()
    engine = BattleEngine(ruleset)
    state = engine.new_battle((deck, deck), seed=81, shuffle_decks=False)
    state.players[player].hand = list(hand)
    state.players[player].elixir_milli = elixir
    return engine, state


@pytest.mark.parametrize("player", (0, 1))
@pytest.mark.parametrize("tank_id", ("golem", "lava-hound"))
def test_quiet_beatdown_waits_for_held_primary_tank(player: int, tank_id: str) -> None:
    deck = _GOLEM_DECK if tank_id == "golem" else _AIR_DECK
    hand = (
        (tank_id, "baby-dragon", "zap")
        if tank_id == "golem"
        else (tank_id, "balloon", "baby-dragon")
    )
    engine, state = _battle(deck, player, hand, 5_000)

    action = BeatdownTankSupportController().choose_action(engine, state, player)

    assert isinstance(action, WaitAction)


@pytest.mark.parametrize("player", (0, 1))
def test_beatdown_uses_balloon_as_primary_and_lava_hound_followup(player: int) -> None:
    engine, state = _battle(_AIR_DECK, player, ("balloon", "baby-dragon"), 10_000)
    controller = BeatdownTankSupportController()

    primary = controller.choose_action(engine, state, player)
    assert isinstance(primary, PlayCardAction)
    assert state.players[player].hand[primary.card_slot] == "balloon"
    assert engine.validate_action(state, primary) is None

    state = engine.new_battle((_AIR_DECK, _AIR_DECK), seed=82, shuffle_decks=False)
    state.players[player].hand = ["balloon", "baby-dragon"]
    state.players[player].elixir_milli = 10_000
    state.players[player].cards_played = 1
    lava_hound = engine._spawn_single_at(
        state,
        engine.ruleset.card("lava-hound"),
        owner=player,
        x_mtile=3_500,
        y_mtile=25_000 if player == 0 else 7_000,
    )

    followup = controller.choose_action(engine, state, player)
    assert isinstance(followup, PlayCardAction)
    assert state.players[player].hand[followup.card_slot] == "balloon"
    assert followup.cell[0] == lava_hound.x_mtile // 1_000
    if player == 0:
        assert followup.cell[1] > lava_hound.y_mtile // 1_000
    else:
        assert followup.cell[1] < lava_hound.y_mtile // 1_000
    assert engine.validate_action(state, followup) is None


@pytest.mark.parametrize("player", (0, 1))
def test_beatdown_support_uses_live_golem_lane(player: int) -> None:
    engine, state = _battle(_GOLEM_DECK, player, ("baby-dragon", "zap"), 10_000)
    state.players[player].cards_played = 1
    golem = engine._spawn_single_at(
        state,
        engine.ruleset.card("golem"),
        owner=player,
        x_mtile=3_500,
        y_mtile=25_000 if player == 0 else 7_000,
    )

    action = BeatdownTankSupportController().choose_action(engine, state, player)

    assert isinstance(action, PlayCardAction)
    assert state.players[player].hand[action.card_slot] == "baby-dragon"
    assert action.cell[0] == golem.x_mtile // 1_000
    if player == 0:
        assert action.cell[1] > golem.y_mtile // 1_000
    else:
        assert action.cell[1] < golem.y_mtile // 1_000
    assert engine.validate_action(state, action) is None


@pytest.mark.parametrize("player", (0, 1))
def test_aggressive_pressure_saves_for_held_hog_on_quiet_board(player: int) -> None:
    engine, state = _battle(_AGGRESSIVE_DECK, player, ("hog-rider", "ice-spirit"), 3_000)

    action = AggressivePressureController().choose_action(engine, state, player)

    assert isinstance(action, WaitAction)


@pytest.mark.parametrize("player", (0, 1))
@pytest.mark.parametrize("primary_id", ("goblin-barrel", "x-bow"))
def test_siege_bait_saves_for_held_primary_win_condition(
    player: int,
    primary_id: str,
) -> None:
    engine, state = _battle(_SIEGE_DECK, player, (primary_id, "log"), 2_000)

    action = SiegeBaitController().choose_action(engine, state, player)

    assert isinstance(action, WaitAction)


@pytest.mark.parametrize("player", (0, 1))
def test_defensive_cycle_saves_for_xbow_then_places_it_in_tower_range(player: int) -> None:
    engine, state = _battle(_X_BOW_DECK, player, ("x-bow", "ice-spirit"), 4_000)
    controller = DefensiveCycleController()

    waiting = controller.choose_action(engine, state, player)
    assert waiting == WaitAction(player)

    state.players[player].elixir_milli = 10_000
    action = controller.choose_action(engine, state, player)
    _assert_tower_connected_xbow(engine, state, player, action)


@pytest.mark.parametrize("player", (0, 1))
def test_siege_bait_xbow_cell_connects_to_a_tower(player: int) -> None:
    engine, state = _battle(_X_BOW_DECK, player, ("x-bow", "ice-spirit"), 10_000)

    action = SiegeBaitController().choose_action(engine, state, player)

    _assert_tower_connected_xbow(engine, state, player, action)


@pytest.mark.parametrize("player", (0, 1))
@pytest.mark.parametrize(
    "controller_type",
    (AggressivePressureController, DefensiveCycleController, SiegeBaitController),
)
def test_crossed_enemy_keeps_reactive_building_priority(
    player: int,
    controller_type,
) -> None:
    engine, state = _battle(_X_BOW_DECK, player, ("x-bow", "tesla"), 10_000)
    engine._spawn_single_at(
        state,
        engine.ruleset.card("giant"),
        owner=1 - player,
        x_mtile=4_500,
        y_mtile=18_000 if player == 0 else 14_000,
    )

    action = controller_type().choose_action(engine, state, player)

    assert isinstance(action, PlayCardAction)
    assert state.players[player].hand[action.card_slot] == "tesla"
    assert engine.validate_action(state, action) is None


@pytest.mark.parametrize("player", (0, 1))
def test_beatdown_fallback_sends_miner_to_enemy_tower_side(player: int) -> None:
    engine, state = _battle(_MINER_DECK, player, ("miner",), 10_000)

    action = BeatdownTankSupportController().choose_action(engine, state, player)

    assert isinstance(action, PlayCardAction)
    assert state.players[player].hand[action.card_slot] == "miner"
    if player == 0:
        assert action.cell[1] <= 14
    else:
        assert action.cell[1] >= 17
    assert engine.validate_action(state, action) is None


def _assert_tower_connected_xbow(
    engine: BattleEngine,
    state,
    player: int,
    action,
) -> None:
    assert isinstance(action, PlayCardAction)
    assert state.players[player].hand[action.card_slot] == "x-bow"
    assert engine.validate_action(state, action) is None
    card = engine.ruleset.card("x-bow")
    x_mtile, y_mtile = cell_center_mtile(action.cell)
    towers = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.owner != player and entity.kind == "tower"
    ]
    tower = min(towers, key=lambda candidate: abs(candidate.x_mtile - x_mtile))
    edge_distance = (
        distance_mtile(x_mtile, y_mtile, tower.x_mtile, tower.y_mtile)
        - int(card.collision_radius_mtile or 0)
        - engine._tower_collision_radii[tower.card_id]
    )
    assert edge_distance <= int(card.range_mtile or 0)
