from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.engine import (
    BASE_HOG_CYCLE_DECK,
    ENGINE_VERSION,
    BattleEngine,
    DeterministicCycleController,
)
from simulator.fixed import distance_mtile
from simulator.geometry import mirror_cell, mirror_position
from simulator.runner import run_scenario, run_scenario_with_snapshots
from simulator.scenario import ScheduledAction, Scenario, load_scenario
from simulator.state import (
    EntityState,
    ProjectileState,
    StatusState,
    battle_state_from_primitive,
)
from simulator.ruleset import load_ruleset


def _deck_with_first(card_id: str) -> tuple[str, ...]:
    return (card_id, *(card for card in BASE_HOG_CYCLE_DECK if card != card_id))


def _tower(state, *, owner: int, role: str):
    return next(
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == owner and entity.role == role
    )


def _unit(state, card_id: str, *, owner: int):
    return next(
        entity
        for entity in state.entities.values()
        if entity.card_id == card_id and entity.owner == owner
    )


def _advance_until(engine: BattleEngine, state, predicate, *, limit: int = 500) -> None:
    for _ in range(limit):
        if predicate():
            return
        engine.step(state)
    pytest.fail(f"condition was not reached within {limit} ticks")


def _engine_with_test_air_card() -> BattleEngine:
    base = load_ruleset()
    hog = base.cards["hog-rider"]
    mechanics = dict(hog.mechanics)
    mechanics.update(
        {
            "movement_layer": "air",
            "building_only": False,
            "placement_class": "own_ground",
        }
    )
    air = replace(
        hog,
        card_id="air-test",
        name="Air Test",
        aliases=("air-test",),
        targets=("air", "ground", "building", "crown_tower"),
        mechanics=mechanics,
    )
    cards = dict(base.cards)
    cards[air.card_id] = air
    aliases = dict(base._card_aliases)
    aliases["air-test"] = air.card_id
    return BattleEngine(
        replace(
            base,
            cards=MappingProxyType(cards),
            _card_aliases=MappingProxyType(aliases),
        )
    )


def test_air_layer_flies_directly_across_river_and_ignores_ground_collision() -> None:
    engine = _engine_with_test_air_card()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    target = _tower(state, owner=1, role="king")
    air = EntityState(
        uid=state.next_uid,
        card_id="air-test",
        owner=0,
        kind="troop",
        x_mtile=9_000,
        y_mtile=23_000,
        hp=100,
        max_hp=100,
        spawn_tick=state.tick,
        target_uid=target.uid,
    )
    state.next_uid += 1
    state.entities[air.uid] = air
    ground = EntityState(
        uid=state.next_uid,
        card_id="hog-rider",
        owner=0,
        kind="troop",
        x_mtile=air.x_mtile,
        y_mtile=air.y_mtile,
        hp=100,
        max_hp=100,
        spawn_tick=state.tick,
    )
    state.next_uid += 1
    state.entities[ground.uid] = ground

    before = (air.x_mtile, air.y_mtile)
    engine._move_entities(state)
    assert (air.x_mtile, air.y_mtile) == (9_000, 22_880)
    assert air.navigation_waypoints == [(target.x_mtile, target.y_mtile)]
    engine._separate_entities(state)
    assert (air.x_mtile, air.y_mtile) == (9_000, 22_880)
    assert before != (air.x_mtile, air.y_mtile)


def test_ground_targeting_rejects_air_and_air_targeting_accepts_air() -> None:
    engine = _engine_with_test_air_card()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    tower = _tower(state, owner=1, role="king")
    air = EntityState(
        uid=state.next_uid,
        card_id="air-test",
        owner=1,
        kind="troop",
        x_mtile=9_000,
        y_mtile=8_000,
        hp=100,
        max_hp=100,
        spawn_tick=state.tick,
    )
    state.next_uid += 1
    state.entities[air.uid] = air
    hog = EntityState(
        uid=state.next_uid,
        card_id="hog-rider",
        owner=0,
        kind="troop",
        x_mtile=9_000,
        y_mtile=9_000,
        hp=100,
        max_hp=100,
        spawn_tick=state.tick,
    )
    state.next_uid += 1
    state.entities[hog.uid] = hog

    assert engine._target_allowed(hog, air) is False
    assert engine._target_allowed(air, air) is True  # same owner is rejected by acquisition, not legality
    assert engine._target_allowed(air, tower) is True


def test_complete_headless_match_reaches_a_declared_terminal_outcome() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=23)

    result = engine.run_match(
        state,
        (
            DeterministicCycleController(lane="left"),
            DeterministicCycleController(lane="right"),
        ),
    )

    assert result is state
    assert state.terminal
    assert state.phase == "ended"
    assert state.winner in {0, 1, None}
    assert state.terminal_reason in {
        "king_tower_destroyed",
        "regulation_crowns",
        "overtime_sudden_death",
        "tiebreak_lowest_hp",
        "tiebreak_equal_lowest_hp",
        "simultaneous_king_destruction",
    }
    assert state.elapsed_us <= (
        engine.ruleset.match.regulation_us + engine.ruleset.match.overtime_us
    )
    assert state.events[-1].kind == "match_ended"
    engine.validate_state(state)


def test_repeated_execution_has_identical_state_hashes_and_events() -> None:
    first_engine = BattleEngine()
    second_engine = BattleEngine()
    first = first_engine.new_battle(seed=918_251, shuffle_decks=False)
    second = second_engine.new_battle(seed=918_251, shuffle_decks=False)

    for tick in range(180):
        actions = (
            (PlayCardAction(0, 0, (3, 17)), PlayCardAction(1, 1, (8, 11)))
            if tick == 0
            else ()
        )
        first_events = first_engine.step(first, actions)
        second_events = second_engine.step(second, actions)
        assert first.state_hash() == second.state_hash()
        assert first_events == second_events

    assert first.canonical_json(include_events=True) == second.canonical_json(include_events=True)
    assert first.event_log_hash() == second.event_log_hash()
    assert first.replay_hash() == second.replay_hash()


def test_fast_training_engine_is_bit_identical_to_strict_reference() -> None:
    strict = BattleEngine(validate_every_tick=True)
    fast = BattleEngine(validate_every_tick=False)
    first = strict.new_battle(seed=77, shuffle_decks=False)
    second = fast.new_battle(seed=77, shuffle_decks=False)

    for tick in range(300):
        actions = (
            (PlayCardAction(0, 0, (3, 17)), PlayCardAction(1, 1, (8, 11)))
            if tick == 0
            else ()
        )
        strict.step(first, actions)
        fast.step(second, actions)

    fast.validate_state(second)
    assert first.replay_hash() == second.replay_hash()


def test_mirrored_hog_deployments_have_mirrored_trajectories_and_damage() -> None:
    engine = BattleEngine()
    player_zero = engine.new_battle(seed=0, shuffle_decks=False)
    player_one = engine.new_battle(seed=0, shuffle_decks=False)
    cell = (3, 17)

    for tick in range(120):
        engine.step(
            player_zero,
            (PlayCardAction(0, 0, cell),) if tick == 0 else (),
        )
        engine.step(
            player_one,
            (PlayCardAction(1, 0, mirror_cell(cell)),) if tick == 0 else (),
        )

    first_hog = _unit(player_zero, "hog-rider", owner=0)
    mirrored_hog = _unit(player_one, "hog-rider", owner=1)
    assert mirror_position(first_hog.x_mtile, first_hog.y_mtile) == (
        mirrored_hog.x_mtile,
        mirrored_hog.y_mtile,
    )
    assert (first_hog.hp, first_hog.alive, first_hog.attack_count) == (
        mirrored_hog.hp,
        mirrored_hog.alive,
        mirrored_hog.attack_count,
    )
    assert _tower(player_zero, owner=1, role="right").hp == _tower(
        player_one, owner=0, role="right"
    ).hp


def test_elixir_regeneration_is_exact_and_card_returns_after_four_more_plays() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    player = state.players[0]

    for _ in range(engine.ruleset.match.normal_elixir_interval_us // engine.ruleset.tick_us):
        engine.step(state)
    assert player.elixir_milli == engine.ruleset.match.initial_elixir_milli + 1_000
    assert player.elixir_remainder == 0

    original_card = player.hand[0]
    expected_cards = [
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
    ]
    placements = [(3, 20), (8, 20), (4, 20), (5, 20), (6, 20)]
    for expected_card, cell in zip(expected_cards, placements, strict=True):
        player.elixir_milli = engine.ruleset.match.max_elixir_milli
        cost = engine.ruleset.card(expected_card).elixir_milli
        result = engine.apply_actions(state, (PlayCardAction(0, 0, cell),))[0]
        assert result.accepted and result.card_id == expected_card
        assert player.elixir_milli == engine.ruleset.match.max_elixir_milli - cost

    assert original_card == "hog-rider"
    assert player.hand[-1] == original_card
    assert player.cards_played == 5
    assert sorted(player.hand + player.draw_pile) == sorted(player.deck)


def test_invalid_and_duplicate_actions_do_not_spend_or_spawn() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    player = state.players[0]
    hand_before = list(player.hand)
    entity_uids_before = set(state.entities)

    player.elixir_milli = 0
    result = engine.apply_actions(state, (PlayCardAction(0, 0, (3, 20)),))[0]
    assert not result.accepted and result.reason == "insufficient_elixir"

    player.elixir_milli = engine.ruleset.match.max_elixir_milli
    result = engine.apply_actions(state, (PlayCardAction(0, 9, (3, 20)),))[0]
    assert not result.accepted and result.reason == "invalid_card_slot"
    result = engine.apply_actions(state, (PlayCardAction(0, 0, (3, 14)),))[0]
    assert not result.accepted and result.reason == "illegal_placement"

    duplicate = engine.apply_actions(
        state,
        (WaitAction(0), PlayCardAction(0, 0, (3, 20))),
    )
    assert len(duplicate) == 1
    assert not duplicate[0].accepted and duplicate[0].reason == "multiple_actions_in_tick"
    assert player.hand == hand_before
    assert player.elixir_milli == engine.ruleset.match.max_elixir_milli
    assert set(state.entities) == entity_uids_before


def test_air_and_ground_troops_cannot_be_placed_on_either_owners_building() -> None:
    engine = _engine_with_test_air_card()
    state = engine.new_battle(seed=8, shuffle_decks=False)
    cell = (4, 20)
    x_mtile, y_mtile = cell[0] * 1_000 + 500, cell[1] * 1_000 + 500

    for owner in (0, 1):
        building = EntityState(
            uid=state.next_uid,
            card_id="cannon",
            owner=owner,
            kind="building",
            x_mtile=x_mtile,
            y_mtile=y_mtile,
            hp=100,
            max_hp=100,
            spawn_tick=state.tick,
        )
        state.next_uid += 1
        state.entities[building.uid] = building
        assert not engine._legal_deployment(state, 0, engine.ruleset.card("hog-rider"), cell)
        assert not engine._legal_deployment(state, 0, engine.ruleset.card("air-test"), cell)
        del state.entities[building.uid]


def test_hog_ignores_troops_and_cannon_pulls_it_off_the_tower_path() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)

    engine.step(
        state,
        (
            PlayCardAction(0, 0, (3, 18)),
            PlayCardAction(1, 1, (8, 12)),
        ),
    )
    # Add an enemy troop close to the Hog without spending a second action in
    # the deployment tick; this is a reachable Musketeer play on the next tick.
    state.players[1].elixir_milli = engine.ruleset.match.max_elixir_milli
    musketeer_slot = state.players[1].hand.index("musketeer")
    engine.step(state, (PlayCardAction(1, musketeer_slot, (4, 13)),))
    _advance_until(
        engine,
        state,
        lambda: _unit(state, "hog-rider", owner=0).target_uid is not None,
        limit=30,
    )

    hog = _unit(state, "hog-rider", owner=0)
    cannon = _unit(state, "cannon", owner=1)
    musketeer = _unit(state, "musketeer", owner=1)
    assert hog.target_uid == cannon.uid
    assert hog.target_uid != musketeer.uid
    _advance_until(engine, state, lambda: hog.x_mtile > 3_500, limit=60)
    assert hog.x_mtile > 3_500
    _advance_until(
        engine,
        state,
        lambda: any(
            event.kind == "damage_applied"
            and event.get("source_card_id") == "cannon"
            and event.get("target_uid") == hog.uid
            for event in state.events
        ),
        limit=30,
    )
    assert cannon.attack_count >= 1


def test_deploying_cannon_can_immediately_pull_but_cannot_attack() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    state.players[1].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(1, 0, (3, 14)),))
    hog = _unit(state, "hog-rider", owner=1)
    hog.deploy_remaining_us = 0
    hog.x_mtile, hog.y_mtile = 7_500, 17_000
    hog.target_uid = _tower(state, owner=0, role="left").uid

    engine.step(state, (PlayCardAction(0, 1, (8, 20)),))

    cannon = _unit(state, "cannon", owner=0)
    assert cannon.deploy_remaining_us == 950_000
    assert hog.target_uid == cannon.uid
    assert cannon.attack_count == 0
    assert cannon.lifetime_remaining_us == 29_950_000
    assert cannon.hp == 823


def test_cannon_loses_hp_linearly_over_its_declared_lifetime() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli

    engine.step(state, (PlayCardAction(0, 1, (8, 20)),))
    cannon = _unit(state, "cannon", owner=0)
    # Lifetime and HP decay begin at placement, while the Cannon still needs
    # its full deployment delay before it can attack.
    assert cannon.deploy_remaining_us == 950_000
    assert cannon.lifetime_remaining_us == 29_950_000
    assert cannon.hp == 823
    for _ in range(299):
        engine.step(state)

    assert cannon.lifetime_remaining_us == 15_000_000
    assert cannon.hp == cannon.max_hp // 2 == 412
    assert cannon.alive

    for _ in range(300):
        engine.step(state)

    assert not cannon.alive
    assert cannon.hp == 0
    assert any(
        event.kind == "building_expired" and event.get("uid") == cannon.uid
        for event in state.events
    )


def test_cannon_lifetime_decay_is_additive_with_combat_damage() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 1, (8, 20)),))
    cannon = _unit(state, "cannon", owner=0)
    for _ in range(299):
        engine.step(state)
    assert cannon.hp == 412

    cannon.hp -= 100
    for _ in range(10):
        engine.step(state)

    # Ten 50 ms ticks consume 13 HP plus a retained fixed-point remainder.
    # Combat damage does not reset the lifetime curve or get overwritten by it.
    assert cannon.hp == 299
    assert cannon.lifetime_remaining_us == 14_500_000


def test_hog_routes_around_a_friendly_cannon_without_intersection() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 1, (3, 20)),))
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 0, (3, 22)),))
    cannon = _unit(state, "cannon", owner=0)
    hog = _unit(state, "hog-rider", owner=0)

    trajectory = []
    for _ in range(80):
        engine.step(state)
        trajectory.append((hog.x_mtile, hog.y_mtile))

    minimum = engine._collision_radius(hog) + engine._collision_radius(cannon)
    assert all(
        ((x - cannon.x_mtile) ** 2 + (y - cannon.y_mtile) ** 2) >= minimum**2
        for x, y in trajectory
    )
    assert max(abs(x - 3_500) for x, _ in trajectory) > 1_000


def test_cached_hog_route_replans_when_a_building_changes_topology() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 0, (3, 22)),))
    hog = _unit(state, "hog-rider", owner=0)
    for _ in range(25):
        engine.step(state)
    cached_revision = hog.navigation_revision
    assert hog.navigation_waypoints

    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 0, (3, 20)),))
    cannon = _unit(state, "cannon", owner=0)
    assert state.navigation_revision > cached_revision

    trajectory = []
    for _ in range(80):
        engine.step(state)
        trajectory.append((hog.x_mtile, hog.y_mtile))

    minimum = engine._collision_radius(hog) + engine._collision_radius(cannon)
    assert hog.navigation_revision == state.navigation_revision
    assert all(
        ((x - cannon.x_mtile) ** 2 + (y - cannon.y_mtile) ** 2) >= minimum**2
        for x, y in trajectory
    )


def test_local_collision_resolution_moves_both_units_symmetrically() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    definition = engine.ruleset.card("skeletons")
    first = EntityState(
        uid=state.next_uid,
        card_id="skeletons",
        owner=0,
        kind="troop",
        x_mtile=9_000,
        y_mtile=22_000,
        hp=definition.hitpoints,
        max_hp=definition.hitpoints,
        spawn_tick=state.tick,
    )
    state.next_uid += 1
    second = replace(first, uid=state.next_uid)
    state.next_uid += 1
    state.entities[first.uid] = first
    state.entities[second.uid] = second

    engine._separate_entities(state)

    assert first.x_mtile + second.x_mtile == 18_000
    assert first.y_mtile + second.y_mtile == 44_000
    assert abs(first.x_mtile - second.x_mtile) == (
        engine._collision_radius(first) + engine._collision_radius(second)
    )


def test_collision_mass_makes_ice_golem_displace_skeleton_farther() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    entities = []
    for card_id in ("ice-golem", "skeletons"):
        definition = engine.ruleset.card(card_id)
        entity = EntityState(
            uid=state.next_uid,
            card_id=card_id,
            owner=0,
            kind="troop",
            x_mtile=9_000,
            y_mtile=22_000,
            hp=int(definition.hitpoints or 0),
            max_hp=int(definition.hitpoints or 0),
            spawn_tick=state.tick,
        )
        state.next_uid += 1
        state.entities[entity.uid] = entity
        entities.append(entity)
    golem, skeleton = entities

    engine._separate_entities(state)

    golem_displacement = distance_mtile(9_000, 22_000, golem.x_mtile, golem.y_mtile)
    skeleton_displacement = distance_mtile(
        9_000,
        22_000,
        skeleton.x_mtile,
        skeleton.y_mtile,
    )
    assert skeleton_displacement > golem_displacement * 5
    assert distance_mtile(
        golem.x_mtile,
        golem.y_mtile,
        skeleton.x_mtile,
        skeleton.y_mtile,
    ) == engine._collision_radius(golem) + engine._collision_radius(skeleton)


def test_skeleton_card_uses_measured_leader_and_rear_pair_formation() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("skeletons"), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000

    engine.step(state, (PlayCardAction(0, 0, (8, 20)),))

    spawned = sorted(
        (
            (entity.x_mtile, entity.y_mtile)
            for entity in state.entities.values()
            if entity.card_id == "skeletons" and entity.owner == 0
        )
    )
    assert spawned == sorted(((8_500, 20_500), (7_750, 21_000), (9_250, 21_000)))


def test_knockback_stops_at_structure_collision_and_invalidates_route() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    state.players[0].elixir_milli = engine.ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 1, (4, 18)),))
    for _ in range(20):
        engine.step(state)
    cannon = _unit(state, "cannon", owner=0)
    definition = engine.ruleset.card("ice-golem")
    golem = EntityState(
        uid=state.next_uid,
        card_id="ice-golem",
        owner=0,
        kind="troop",
        x_mtile=3_500,
        y_mtile=17_500,
        hp=definition.hitpoints,
        max_hp=definition.hitpoints,
        spawn_tick=state.tick,
        navigation_revision=state.navigation_revision,
        navigation_waypoints=[(3_500, 6_500)],
    )
    state.next_uid += 1
    state.entities[golem.uid] = golem
    engine.validate_state(state)


def test_long_knockback_cannot_tunnel_through_a_structure() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("ice-golem"), _deck_with_first("cannon")),
        seed=0,
        shuffle_decks=False,
    )
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (6, 20)),
            PlayCardAction(1, 0, (8, 13)),
        ),
    )
    golem = _unit(state, "ice-golem", owner=0)
    cannon = _unit(state, "cannon", owner=1)
    golem.x_mtile, golem.y_mtile = 6_000, 20_500
    cannon.x_mtile, cannon.y_mtile = 8_500, 20_500

    engine._apply_knockback(state, golem, 5_000, 20_500, 5_000)

    minimum = engine._collision_radius(golem) + engine._collision_radius(cannon)
    assert golem.x_mtile < cannon.x_mtile
    assert cannon.x_mtile - golem.x_mtile >= minimum
    engine.validate_state(state)

    engine._apply_knockback(state, golem, 2_500, 16_500, 1_000)

    minimum = engine._collision_radius(golem) + engine._collision_radius(cannon)
    assert (golem.x_mtile, golem.y_mtile) != (3_500, 17_500)
    assert (
        (golem.x_mtile - cannon.x_mtile) ** 2
        + (golem.y_mtile - cannon.y_mtile) ** 2
    ) >= minimum**2
    assert golem.navigation_revision == -1
    assert golem.navigation_waypoints == []
    engine.validate_state(state)


def test_ranged_projectile_applies_card_damage_to_princess_tower() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("musketeer"), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    target = _tower(state, owner=1, role="right")
    damage = engine.ruleset.card("musketeer").damage

    engine.step(state, (PlayCardAction(0, 0, (3, 17)),))
    _advance_until(engine, state, lambda: target.hp < target.max_hp, limit=150)

    assert target.max_hp - target.hp == damage
    projectile_events = [event for event in state.events if event.kind == "projectile_spawned"]
    damage_events = [
        event
        for event in state.events
        if event.kind == "damage_applied"
        and event.get("source_card_id") == "musketeer"
        and event.get("target_uid") == target.uid
    ]
    assert projectile_events
    assert len(damage_events) == 1
    assert damage_events[0].get("damage") == damage


def test_official_current_cannon_damage_is_applied_to_a_troop() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("cannon"), _deck_with_first("ice-golem")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 20)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    cannon = _unit(state, "cannon", owner=0)
    target = _unit(state, "ice-golem", owner=1)
    cannon.deploy_remaining_us = 0
    target.deploy_remaining_us = 0
    cannon.x_mtile, cannon.y_mtile = 8_500, 17_500
    target.x_mtile, target.y_mtile = 8_500, 14_500
    target.statuses.append(StatusState("freeze", 10_000_000, 0))

    _advance_until(engine, state, lambda: target.hp < target.max_hp, limit=30)

    assert target.max_hp - target.hp == 202
    assert any(
        event.kind == "damage_applied"
        and event.get("source_uid") == cannon.uid
        and event.get("target_uid") == target.uid
        and event.get("damage") == 202
        for event in state.events
    )


@pytest.mark.parametrize(
    ("card_id", "expected_damage", "expected_interval_ticks"),
    (("hog-rider", 317, 32), ("musketeer", 217, 20)),
)
def test_current_level11_tower_damage_and_repeat_interval_are_exact(
    card_id: str,
    expected_damage: int,
    expected_interval_ticks: int,
) -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first(card_id), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    target = _tower(state, owner=1, role="right")
    for tower in (
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == 1
    ):
        tower.statuses.append(StatusState("freeze", 20_000_000, 0))

    engine.step(state, (PlayCardAction(0, 0, (3, 17)),))
    attacker = _unit(state, card_id, owner=0)
    attacker.x_mtile = target.x_mtile
    attacker.y_mtile = target.y_mtile + 2_000
    attacker.deploy_remaining_us = 0
    attacker.target_uid = target.uid
    attacker.navigation_revision = -1
    attacker.navigation_waypoints = []

    _advance_until(
        engine,
        state,
        lambda: sum(
            event.kind == "attack_started" and event.get("uid") == attacker.uid
            for event in state.events
        )
        >= 3,
        limit=100,
    )

    starts = [
        event.tick
        for event in state.events
        if event.kind == "attack_started" and event.get("uid") == attacker.uid
    ][:3]
    assert [right - left for left, right in zip(starts, starts[1:])] == [
        expected_interval_ticks,
        expected_interval_ticks,
    ]
    damage = [
        event.get("damage")
        for event in state.events
        if event.kind == "damage_applied"
        and event.get("source_card_id") == card_id
        and event.get("target_uid") == target.uid
    ]
    assert damage
    assert set(damage) == {expected_damage}


def test_ranged_projectile_starts_at_sourced_muzzle_offset() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("musketeer"), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    engine.step(state, (PlayCardAction(0, 0, (5, 20)),))
    musketeer = _unit(state, "musketeer", owner=0)
    target = _tower(state, owner=1, role="left")
    musketeer.x_mtile, musketeer.y_mtile = 5_000, 10_000
    target.x_mtile, target.y_mtile = 8_000, 10_000
    musketeer.deploy_remaining_us = 0
    musketeer.pending_target_uid = target.uid

    engine._resolve_attack(state, musketeer)

    projectile = state.projectiles[max(state.projectiles)]
    assert (projectile.x_mtile, projectile.y_mtile) == (5_450, 10_000)
    assert (projectile.target_x_mtile, projectile.target_y_mtile) == (8_000, 10_000)


@pytest.mark.parametrize(
    ("card_id", "cell", "expected_damage"),
    (("fireball", (3, 6), 172), ("log", (3, 17), 35)),
)
def test_spells_apply_reduced_crown_tower_damage(
    card_id: str,
    cell: tuple[int, int],
    expected_damage: int,
) -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first(card_id), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    target = _tower(state, owner=1, role="right")

    engine.step(state, (PlayCardAction(0, 0, cell),))
    _advance_until(engine, state, lambda: target.hp < target.max_hp, limit=100)

    assert target.max_hp - target.hp == expected_damage
    assert any(
        event.kind == "damage_applied"
        and event.get("source_card_id") == card_id
        and event.get("damage") == expected_damage
        for event in state.events
    )


def test_fireball_hits_every_unit_in_its_area_once() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("fireball"), _deck_with_first("skeletons")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 14)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    skeletons = [
        entity
        for entity in state.entities.values()
        if entity.card_id == "skeletons" and entity.owner == 1
    ]
    assert len(skeletons) == 3
    for skeleton in skeletons:
        skeleton.x_mtile = 8_500
        skeleton.y_mtile = 14_500
        skeleton.deploy_remaining_us = 5_000_000

    _advance_until(engine, state, lambda: all(not unit.alive for unit in skeletons), limit=40)

    damage_events = [
        event
        for event in state.events
        if event.kind == "damage_applied"
        and event.get("source_card_id") == "fireball"
        and event.get("target_uid") in {unit.uid for unit in skeletons}
    ]
    assert len(damage_events) == 3
    assert {event.get("target_uid") for event in damage_events} == {
        unit.uid for unit in skeletons
    }


def test_fireball_area_uses_target_collision_edge_at_exact_integer_boundary() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("fireball"), _deck_with_first("skeletons")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[1].elixir_milli = 10_000
    engine.step(state, (PlayCardAction(1, 0, (8, 14)),))
    skeletons = sorted(
        (entity for entity in state.entities.values() if entity.card_id == "skeletons"),
        key=lambda entity: entity.uid,
    )
    assert len(skeletons) == 3
    impact_x, impact_y = 8_500, 14_500
    boundary = engine.ruleset.card("fireball").area_radius_mtile + engine._collision_radius(
        skeletons[0]
    )
    skeletons[0].x_mtile, skeletons[0].y_mtile = impact_x + boundary, impact_y
    skeletons[1].x_mtile, skeletons[1].y_mtile = impact_x + boundary + 1, impact_y
    skeletons[2].x_mtile, skeletons[2].y_mtile = impact_x + boundary + 2_000, impact_y

    engine._impact_area(
        state,
        owner=0,
        source_uid=None,
        source_card_id="fireball",
        x=impact_x,
        y=impact_y,
        damage=688,
        crown_damage=172,
        radius=engine.ruleset.card("fireball").area_radius_mtile,
        status=None,
        knockback=0,
        primary_target_uid=None,
    )

    assert skeletons[0].hp == 0
    assert skeletons[1].hp == skeletons[1].max_hp
    assert skeletons[2].hp == skeletons[2].max_hp


def test_fireball_damage_activates_the_king_tower() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("fireball"), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    king = _tower(state, owner=1, role="king")

    engine.step(state, (PlayCardAction(0, 0, (8, 3)),))
    _advance_until(engine, state, lambda: king.hp < king.max_hp, limit=60)

    assert king.max_hp - king.hp == 172
    assert state.players[1].king_active
    assert any(
        event.kind == "king_activated"
        and event.get("player") == 1
        and event.get("reason") == "damaged"
        for event in state.events
    )


def test_log_pierces_multiple_ground_units_but_hits_each_only_once() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("log"), _deck_with_first("skeletons")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 17)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    skeletons = sorted(
        (
            entity
            for entity in state.entities.values()
            if entity.card_id == "skeletons" and entity.owner == 1
        ),
        key=lambda entity: entity.uid,
    )
    assert len(skeletons) == 3
    for skeleton, y_mtile in zip(skeletons, (15_000, 14_000, 13_000), strict=True):
        skeleton.x_mtile = 8_500
        skeleton.y_mtile = y_mtile
        skeleton.deploy_remaining_us = 0
        skeleton.statuses.append(StatusState("freeze", 10_000_000, 0))

    _advance_until(engine, state, lambda: all(not unit.alive for unit in skeletons), limit=40)

    hit_uids = [
        event.get("target_uid")
        for event in state.events
        if event.kind == "damage_applied"
        and event.get("source_card_id") == "log"
        and event.get("target_uid") in {unit.uid for unit in skeletons}
    ]
    assert sorted(hit_uids) == sorted(unit.uid for unit in skeletons)


def test_log_continuous_collision_has_an_exact_lateral_boundary() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("log"), _deck_with_first("skeletons")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[1].elixir_milli = 10_000
    engine.step(state, (PlayCardAction(1, 0, (8, 14)),))
    skeletons = sorted(
        (entity for entity in state.entities.values() if entity.card_id == "skeletons"),
        key=lambda entity: entity.uid,
    )
    assert len(skeletons) == 3
    radius = engine.ruleset.card("log").area_radius_mtile
    boundary = radius + engine._collision_radius(skeletons[0])
    skeletons[0].x_mtile, skeletons[0].y_mtile = 8_500 + boundary, 14_500
    skeletons[1].x_mtile, skeletons[1].y_mtile = 8_500 + boundary + 1, 14_500
    skeletons[2].x_mtile, skeletons[2].y_mtile = 13_500, 14_500
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=None,
        source_card_id="log",
        owner=0,
        x_mtile=8_500,
        y_mtile=14_500,
        target_x_mtile=8_500,
        target_y_mtile=4_400,
        damage=266,
        crown_damage=35,
        speed_mtile_per_s=4_000,
        radius_mtile=radius,
        knockback_mtile=700,
        piercing=True,
    )

    engine._impact_piercing_projectile(state, projectile)

    assert skeletons[0].hp == 0
    assert skeletons[1].hp == skeletons[1].max_hp
    assert skeletons[2].hp == skeletons[2].max_hp
    assert projectile.hit_uids == [skeletons[0].uid]


def test_log_knockback_follows_roll_direction_without_radial_sideways_drift() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("log"), _deck_with_first("ice-golem")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 17)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    target = _unit(state, "ice-golem", owner=1)
    target.x_mtile = 9_500
    target.y_mtile = 15_000
    target.deploy_remaining_us = 0
    target.statuses.append(StatusState("freeze", 10_000_000, 0))
    original_x, original_y = target.x_mtile, target.y_mtile

    _advance_until(engine, state, lambda: target.hp < target.max_hp, limit=30)

    assert target.x_mtile == original_x
    assert target.y_mtile == original_y - 700


def test_unassisted_august_ice_spirit_does_not_connect_to_crown_tower() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("ice-spirit"), BASE_HOG_CYCLE_DECK),
        seed=0,
        shuffle_decks=False,
    )
    target = _tower(state, owner=1, role="right")

    engine.step(state, (PlayCardAction(0, 0, (3, 17)),))
    spirit = _unit(state, "ice-spirit", owner=0)
    # The August ruleset removes the bare Crown Tower from Spirit target
    # acquisition.  With no other enemy body present the Spirit therefore
    # remains idle instead of self-destructing on an unassisted tower hit.
    for _ in range(150):
        engine.step(state)

    assert spirit.alive
    assert engine._choose_target(state, spirit) is None
    assert spirit.attack_count == 0
    assert target.hp == target.max_hp


def test_ice_spirit_jump_damages_and_freezes_a_surviving_target() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("ice-spirit"), _deck_with_first("ice-golem")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 17)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    spirit = _unit(state, "ice-spirit", owner=0)
    golem = _unit(state, "ice-golem", owner=1)
    spirit.x_mtile, spirit.y_mtile = 8_500, 16_000
    golem.x_mtile, golem.y_mtile = 8_500, 14_500
    spirit.deploy_remaining_us = 0
    golem.deploy_remaining_us = 0

    _advance_until(engine, state, lambda: golem.hp < golem.max_hp, limit=20)

    assert not spirit.alive
    assert golem.max_hp - golem.hp == 110
    freeze = next(status for status in golem.statuses if status.kind == "freeze")
    assert 0 < freeze.remaining_us <= 1_100_000
    assert freeze.magnitude_permille == 0


def test_ice_golem_death_damage_and_slow_affect_adjacent_enemy_troop() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("ice-golem"), _deck_with_first("musketeer")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 17)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    golem = _unit(state, "ice-golem", owner=0)
    musketeer = _unit(state, "musketeer", owner=1)
    golem.x_mtile = musketeer.x_mtile
    golem.y_mtile = musketeer.y_mtile
    golem.hp = 0

    engine.step(state)

    assert musketeer.max_hp - musketeer.hp == 84
    assert any(status.kind == "slow" for status in musketeer.statuses)


def test_freeze_pauses_an_attack_already_in_windup() -> None:
    engine = BattleEngine()
    state = engine.new_battle(
        (_deck_with_first("musketeer"), _deck_with_first("skeletons")),
        seed=0,
        shuffle_decks=False,
    )
    state.players[0].elixir_milli = 10_000
    state.players[1].elixir_milli = 10_000
    engine.step(
        state,
        (
            PlayCardAction(0, 0, (8, 17)),
            PlayCardAction(1, 0, (8, 14)),
        ),
    )
    musketeer = _unit(state, "musketeer", owner=0)
    _advance_until(engine, state, lambda: musketeer.windup_remaining_us > 0, limit=40)
    initial_windup = musketeer.windup_remaining_us
    musketeer.statuses.append(StatusState("freeze", 1_200_000, 0))

    for _ in range(10):
        engine.step(state)

    assert musketeer.attack_count == 0
    assert musketeer.windup_remaining_us == initial_windup
    _advance_until(engine, state, lambda: musketeer.attack_count == 1, limit=40)
    assert musketeer.attack_count == 1


def test_destroyed_enemy_princess_opens_forward_hog_cannon_and_log_cells() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    enemy_right = _tower(state, owner=1, role="right")
    enemy_right.alive = False
    enemy_right.hp = 0
    forward = (3, 13)

    assert engine._legal_deployment(state, 0, engine.ruleset.card("hog-rider"), forward)
    assert engine._legal_deployment(state, 0, engine.ruleset.card("cannon"), forward)
    assert engine._legal_deployment(state, 0, engine.ruleset.card("log"), forward)


def test_state_json_round_trip_and_resume_are_bit_identical() -> None:
    engine = BattleEngine()
    uninterrupted = engine.new_battle(seed=71, shuffle_decks=False)
    engine.step(uninterrupted, (PlayCardAction(0, 0, (3, 17)),))
    for _ in range(79):
        engine.step(uninterrupted)

    encoded = json.loads(uninterrupted.canonical_json(include_events=True))
    resumed = battle_state_from_primitive(encoded)
    engine.validate_state(resumed)
    assert resumed.canonical_json(include_events=True) == uninterrupted.canonical_json(
        include_events=True
    )

    for _ in range(120):
        assert engine.step(uninterrupted) == engine.step(resumed)
        assert uninterrupted.state_hash() == resumed.state_hash()
    assert resumed.canonical_json(include_events=True) == uninterrupted.canonical_json(
        include_events=True
    )


def test_restored_state_rejects_a_next_uid_that_would_overwrite_an_entity() -> None:
    engine = BattleEngine()
    raw = json.loads(
        engine.new_battle(seed=0, shuffle_decks=False).canonical_json(include_events=True)
    )
    raw["next_uid"] = 1
    restored = battle_state_from_primitive(raw)

    with pytest.raises(ValueError, match="next_uid"):
        engine.validate_state(restored)


def test_restored_state_rejects_cards_outside_the_interaction_set() -> None:
    engine = BattleEngine()
    raw = json.loads(
        engine.new_battle(seed=0, shuffle_decks=False).canonical_json(include_events=True)
    )
    raw["players"][0]["deck"][0] = "not-a-real-card"
    raw["players"][0]["hand"][0] = "not-a-real-card"
    restored = battle_state_from_primitive(raw)

    with pytest.raises(ValueError, match="interaction set|unknown card"):
        engine.validate_state(restored)


def test_restored_state_rejects_duplicate_uids_and_non_integer_fixed_point() -> None:
    engine = BattleEngine()
    raw = json.loads(
        engine.new_battle(seed=0, shuffle_decks=False).canonical_json(include_events=True)
    )
    raw["entities"].append(dict(raw["entities"][0]))
    with pytest.raises(ValueError, match="duplicate entity UID"):
        battle_state_from_primitive(raw)

    raw = json.loads(
        engine.new_battle(seed=0, shuffle_decks=False).canonical_json(include_events=True)
    )
    raw["entities"][0]["x_mtile"] = 3_500.0
    restored = battle_state_from_primitive(raw)
    with pytest.raises(ValueError, match="fixed-point fields must be integers"):
        engine.validate_state(restored)


def test_saved_scenario_runs_with_pinned_ruleset_and_is_reproducible(tmp_path) -> None:
    engine = BattleEngine()
    scenario = Scenario(
        scenario_id="fireball-princess-smoke",
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        seed=12,
        decks=(_deck_with_first("fireball"), BASE_HOG_CYCLE_DECK),
        actions=(ScheduledAction(0, PlayCardAction(0, 0, (3, 6))),),
        max_ticks=50,
        shuffle_decks=False,
        tags=("fireball", "tower-damage"),
    )
    path = tmp_path / "scenario.json"
    scenario.save(path)
    loaded = load_scenario(path)

    first = run_scenario(engine, loaded)
    second = run_scenario(engine, loaded)

    assert first.tick == 50
    assert first.state_hash() == second.state_hash()
    assert first.canonical_json(include_events=True) == second.canonical_json(include_events=True)
    assert _tower(first, owner=1, role="right").hp == (
        _tower(first, owner=1, role="right").max_hp - 172
    )
    with pytest.raises(ValueError, match="engine version"):
        run_scenario(engine, replace(loaded, engine_version="reference-future"))


def test_saved_state_is_pinned_to_the_engine_algorithm_version() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)

    assert state.engine_version == ENGINE_VERSION
    state.engine_version = "reference-future"
    with pytest.raises(ValueError, match="engine version"):
        engine.validate_state(state)


def test_midmatch_initial_state_scenario_and_snapshots_resume_exactly() -> None:
    engine = BattleEngine()
    initial = engine.new_battle(seed=91, shuffle_decks=False)
    engine.step(initial, (PlayCardAction(0, 0, (3, 17)),))
    for _ in range(19):
        engine.step(initial)
    scenario = Scenario(
        scenario_id="midmatch-hog-trace",
        ruleset_id=engine.ruleset.ruleset_id,
        ruleset_hash=engine.ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        seed=initial.seed,
        decks=tuple(player.deck for player in initial.players),
        initial_state=initial.to_primitive(include_events=False),
        max_ticks=40,
        shuffle_decks=False,
        split="heldout",
    )

    result, snapshots = run_scenario_with_snapshots(
        engine,
        scenario,
        snapshot_ticks=(20, 30, 40),
    )
    direct = battle_state_from_primitive(initial.to_primitive(include_events=False))
    while direct.tick < 40:
        engine.step(direct)

    assert snapshots[20].state_hash() == initial.state_hash()
    assert snapshots[30].tick == 30
    assert snapshots[40].state_hash() == result.state_hash() == direct.state_hash()


def test_tiebreak_uses_lowest_absolute_tower_hp() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    # Tiebreak drains the same raw HP from all towers.  A 3,000-HP Princess
    # Tower therefore falls before a 4,500-HP King Tower even though the King
    # has the lower HP percentage.
    _tower(state, owner=0, role="left").hp = 3_000
    _tower(state, owner=1, role="king").hp = 4_500
    state.phase = "overtime"
    state.elapsed_us = (
        engine.ruleset.match.regulation_us
        + engine.ruleset.match.overtime_us
        - engine.ruleset.tick_us
    )

    engine.step(state)

    assert state.terminal
    assert state.winner == 1
    assert state.terminal_reason == "tiebreak_lowest_hp"


def test_match_runner_rejects_zero_decision_cadence() -> None:
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)

    with pytest.raises(ValueError, match="decision_interval_ticks must be positive"):
        engine.run_match(state, decision_interval_ticks=0, max_ticks=1)
