from __future__ import annotations

import numpy as np
import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.catalog import SIGHT_RANGE_MOBILE
from simulator.env import SimulatorEnv
from simulator.engine import BattleEngine
from simulator.fixed import distance_mtile
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import AreaEffectState, EntityState, ProjectileState


RULESET = load_ruleset("v1")


def _state() -> tuple[BattleEngine, object]:
    engine = BattleEngine(RULESET)
    return engine, engine.new_battle(
        decks=(PLAYER_DECK, PLAYER_DECK), seed=1901, shuffle_decks=False
    )


def _entity(state, card_id: str, owner: int, x: int, y: int, *, hp: int | None = None):
    card = RULESET.card(card_id)
    maximum = int(card.hitpoints or 1)
    entity = EntityState(
        uid=state.next_uid,
        card_id=card_id,
        owner=owner,
        kind=card.kind,
        x_mtile=x,
        y_mtile=y,
        hp=maximum if hp is None else hp,
        max_hp=maximum,
        spawn_tick=state.tick,
    )
    state.next_uid += 1
    state.entities[entity.uid] = entity
    return entity


def test_clones_keep_one_hp_shields_and_clone_death_children() -> None:
    engine, state = _state()
    original = _entity(state, "guards", 0, 9_000, 15_000)
    engine._impact_clone(
        state,
        owner=0,
        source_uid=None,
        source_card_id="clone",
        x=9_000,
        y=15_000,
        radius=3_000,
        raw_clone=RULESET.card("clone").mechanics["clone"],
    )
    cloned = [
        entity
        for entity in state.entities.values()
        if entity.uid != original.uid and entity.card_id == "guards"
    ][0]
    assert cloned.is_clone
    assert (cloned.hp, cloned.max_hp) == (1, 1)
    assert (cloned.shield_hp, cloned.shield_max_hp) == (1, 1)

    # A cloned death payload must not become a full-stat child stream.
    golem = _entity(state, "golem", 0, 12_000, 15_000)
    engine._impact_clone(
        state,
        owner=0,
        source_uid=None,
        source_card_id="clone",
        x=12_000,
        y=15_000,
        radius=3_000,
        raw_clone=RULESET.card("clone").mechanics["clone"],
    )
    clone_golem = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "golem" and entity.uid != golem.uid
    )
    clone_golem.hp = 0
    engine._resolve_deaths(state)
    children = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "golemite"
    ]
    assert children and all(child.is_clone and child.hp == child.max_hp == 1 for child in children)


def test_clone_bypasses_deploy_and_preserves_carrier_payload() -> None:
    engine, state = _state()
    engine._spawn_card_entities(state, 0, RULESET.card("goblin-giant"), (9, 15))
    original = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "goblin-giant" and not entity.is_clone
    )
    original_children = [
        entity
        for entity in state.entities.values()
        if entity.carried_by_uid == original.uid
    ]
    assert len(original_children) == 2

    engine._impact_clone(
        state,
        owner=0,
        source_uid=None,
        source_card_id="clone",
        x=original.x_mtile,
        y=original.y_mtile,
        radius=3_000,
        raw_clone=RULESET.card("clone").mechanics["clone"],
    )

    cloned_giant = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "goblin-giant" and entity.is_clone
    )
    assert cloned_giant.deploy_remaining_us == 0
    cloned_children = [
        entity
        for entity in state.entities.values()
        if entity.carried_by_uid == cloned_giant.uid
    ]
    assert len(cloned_children) == 2
    assert all(
        child.is_clone and child.hp == child.max_hp == 1
        for child in cloned_children
    )
    assert not any(
        entity.is_clone
        and entity.card_id == "spear-goblin"
        and entity.carried_by_uid is None
        for entity in state.entities.values()
    )

    # The attached pair is released from the clone; the legacy death payload
    # must not create a second pair.
    cloned_giant.hp = 0
    engine._resolve_deaths(state)
    assert len(
        [
            entity
            for entity in state.entities.values()
            if entity.is_clone and entity.card_id == "spear-goblin" and entity.alive
        ]
    ) == 2
    assert all(child.carried_by_uid is None for child in cloned_children)


def test_cloned_spawner_starts_immediately_with_a_fresh_spawn_delay() -> None:
    engine, state = _state()
    original = _entity(state, "witch", 0, 9_000, 15_000)
    original.deploy_remaining_us = RULESET.card("witch").deploy_time_us
    assert original.deploy_remaining_us > 0

    engine._impact_clone(
        state,
        owner=0,
        source_uid=None,
        source_card_id="clone",
        x=original.x_mtile,
        y=original.y_mtile,
        radius=3_000,
        raw_clone=RULESET.card("clone").mechanics["clone"],
    )
    cloned = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "witch" and entity.is_clone
    )
    assert cloned.deploy_remaining_us == 0
    assert cloned.spawn_cooldown_us == 1_000_000

    engine._advance_spawners(state, 500_000)
    assert not any(
        entity.parent_uid == cloned.uid and entity.card_id == "skeletons"
        for entity in state.entities.values()
    )
    engine._advance_spawners(state, 500_000)
    skeletons = [
        entity
        for entity in state.entities.values()
        if entity.parent_uid == cloned.uid and entity.card_id == "skeletons"
    ]
    assert len(skeletons) == 4
    assert all(skeleton.is_clone and skeleton.hp == skeleton.max_hp == 1 for skeleton in skeletons)


def test_rage_buffs_friendly_buildings_and_crown_towers() -> None:
    engine, state = _state()
    tower = next(
        entity
        for entity in state.entities.values()
        if entity.owner == 0 and entity.kind == "tower" and entity.role == "left"
    )
    building = engine._spawn_single_at(
        state,
        RULESET.card("cannon"),
        owner=0,
        x_mtile=tower.x_mtile,
        y_mtile=tower.y_mtile - 2_000,
        deploy_remaining_us=0,
    )
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="rage",
        x_mtile=tower.x_mtile,
        y_mtile=tower.y_mtile,
        default_radius=3_000,
        default_damage=0,
        default_crown_damage=0,
        default_status=None,
        default_knockback=0,
        raw_effect=RULESET.card("rage").mechanics["persistent_effect"],
    )

    for target in (tower, building):
        rage = [status for status in target.statuses if status.kind == "rage"]
        assert len(rage) == 1
        assert rage[0].hit_speed_magnitude_permille == 1_300


def test_cloned_deploy_damage_is_suppressed() -> None:
    engine, state = _state()
    target = _entity(state, "giant", 1, 9_000, 15_000, hp=4_000)
    knight = _entity(state, "mega-knight", 0, 9_000, 15_000)
    knight.is_clone = True
    knight.hp = knight.max_hp = 1
    knight.deploy_remaining_us = 1
    engine._advance_deployments(state)
    assert target.hp == 4_000
    assert not any(event.kind == "landing_attack" for event in state.events)


def test_goblin_giant_main_targets_buildings_while_spear_children_target_air() -> None:
    engine, state = _state()
    giant = _entity(state, "goblin-giant", 0, 9_000, 15_000)
    giant.deploy_remaining_us = 0
    air_target = _entity(state, "bats", 1, 9_000, 16_000)

    # With no legal troop/building target the normal engine fallback is an
    # enemy Crown Tower; importantly, it must not select the nearby air troop.
    assert engine._choose_target(state, giant) != air_target.uid
    assert set(RULESET.card("goblin-giant").targets) == {"building", "crown_tower"}
    assert set(RULESET.card("spear-goblin").targets) == {"air", "ground"}

    building = _entity(state, "cannon", 1, 9_000, 16_000)
    assert engine._choose_target(state, giant) == building.uid
    assert air_target.alive


def test_goblin_giant_backpack_children_are_sheltered_until_release() -> None:
    engine, state = _state()
    engine._spawn_card_entities(state, 0, RULESET.card("goblin-giant"), (9, 15))
    giant = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "goblin-giant"
    )
    children = [
        entity
        for entity in state.entities.values()
        if entity.carried_by_uid == giant.uid
    ]
    assert len(children) == 2
    child = children[0]
    child.deploy_remaining_us = 0
    hp_before = child.hp

    assert not engine._targetable_for_acquisition(state, child)
    engine._impact_area(
        state,
        owner=1,
        source_uid=None,
        source_card_id="test-spell",
        x=child.x_mtile,
        y=child.y_mtile,
        damage=999_999,
        crown_damage=999_999,
        radius=1,
        status=None,
        knockback=0,
        primary_target_uid=None,
        allowed_targets=("ground",),
    )
    assert child.hp == hp_before

    giant.alive = False
    engine._release_carried_children(state, giant)
    assert child.carried_by_uid is None
    assert engine._targetable_for_acquisition(state, child)
    engine._impact_area(
        state,
        owner=1,
        source_uid=None,
        source_card_id="test-spell",
        x=child.x_mtile,
        y=child.y_mtile,
        damage=1,
        crown_damage=1,
        radius=1,
        status=None,
        knockback=0,
        primary_target_uid=None,
        allowed_targets=("ground",),
    )
    assert child.hp == hp_before - 1


def test_mega_knight_can_start_a_jump_on_a_crown_tower() -> None:
    engine, state = _state()
    knight = _entity(state, "mega-knight", 0, 3_500, 10_500)
    tower = min(
        (
            entity
            for entity in state.entities.values()
            if entity.kind == "tower" and entity.owner == 1 and entity.role != "king"
        ),
        key=lambda entity: abs(entity.x_mtile - knight.x_mtile)
        + abs(entity.y_mtile - knight.y_mtile),
    )
    knight.target_uid = tower.uid
    engine._move_entities(state)
    assert knight.jump_remaining_us == 400_000
    assert knight.jump_target_uid == tower.uid


def test_fisherman_hook_cancels_an_active_mega_knight_jump() -> None:
    engine, state = _state()
    fisherman = _entity(state, "fisherman", 0, 8_000, 15_000)
    knight = _entity(state, "mega-knight", 1, 13_000, 15_000)
    knight.jump_remaining_us = 200_000
    knight.jump_target_uid = fisherman.uid
    knight.jump_landing_x_mtile = 15_000
    knight.jump_landing_y_mtile = 15_000
    engine._apply_hook(state, fisherman, knight, RULESET.card("fisherman").mechanics["hook"])
    assert knight.jump_remaining_us == 0
    assert knight.jump_target_uid is None
    assert any(event.kind == "jump_cancelled" for event in state.events)


def test_cannon_cart_preserves_target_when_transforming_to_building() -> None:
    engine, state = _state()
    cart = _entity(state, "cannon-cart", 0, 9_000, 15_000)
    target = next(
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == 1 and entity.role != "king"
    )
    cart.target_uid = target.uid
    cart.attack_cooldown_us = 250_000
    engine._deal_damage(state, cart, 905, source_uid=None, source_card_id="test")
    assert cart.card_id == "cannon-cart-building"
    assert cart.kind == "building"
    assert cart.target_uid == target.uid
    assert cart.attack_cooldown_us == 250_000
    engine._invalidate_and_acquire_targets(state)
    assert cart.target_uid == target.uid


def test_king_tower_destruction_collapses_remaining_crown_towers() -> None:
    engine, state = _state()
    king = engine._tower(state, 1, "king")
    king.hp = 0

    destroyed = engine._resolve_deaths(state)
    engine._resolve_tower_outcomes(state, destroyed)

    assert state.terminal
    assert state.winner == 0
    assert state.players[0].crowns == 3
    assert all(
        not tower.alive and tower.hp == 0
        for tower in engine._towers_for(state, 1)
    )
    assert sum(
        event.kind == "tower_destroyed" and event.get("player") == 1
        for event in state.events
    ) == 3
    engine.validate_state(state)


def test_simultaneous_king_and_princess_destruction_caps_crowns() -> None:
    engine, state = _state()
    for tower in engine._towers_for(state, 1):
        tower.hp = 0

    destroyed = engine._resolve_deaths(state)
    engine._resolve_tower_outcomes(state, destroyed)

    assert state.terminal
    assert state.winner == 0
    assert state.players[0].crowns == 3
    engine.validate_state(state)


def test_terminal_tower_destruction_accounts_for_the_completed_physics_tick() -> None:
    engine, state = _state()
    king = engine._tower(state, 1, "king")
    king.hp = 0

    engine.step(state)

    assert state.terminal
    assert state.tick == 1
    assert state.elapsed_us == RULESET.tick_us
    engine.validate_state(state)


def test_elixir_remainder_is_rescaled_at_double_elixir_transition() -> None:
    engine, state = _state()
    player = state.players[0]
    player.elixir_milli = 0
    player.elixir_remainder = 2_700_000
    state.elapsed_us = RULESET.match.regulation_us - 60 * 1_000_000

    engine._regenerate_elixir(state)

    # The in-flight 2.7m/2.8m normal-elixir fraction becomes 1.35m/1.4m
    # before the boundary tick, yielding 36 milli-elixir and 950k remainder.
    assert player.elixir_milli == 36
    assert player.elixir_remainder == 950_000


def test_tiebreak_destroys_lowest_tower_and_clears_combatants() -> None:
    engine, state = _state()
    state.phase = "overtime"
    low = engine._tower(state, 1, "left")
    low.hp = 1_500
    engine._tower(state, 0, "left").hp = 2_000
    troop = _entity(state, "knight", 0, 9_000, 15_000)
    building = _entity(state, "cannon", 1, 9_000, 14_000)

    engine._resolve_tiebreak(state)

    assert state.terminal
    assert state.winner == 0
    assert state.terminal_reason == "tiebreak_lowest_hp"
    assert not low.alive and low.hp == 0
    assert engine._tower(state, 0, "left").hp == 500
    assert not troop.alive and troop.hp == 0
    assert not building.alive and building.hp == 0
    assert any(
        event.kind == "tower_destroyed" and event.get("uid") == low.uid
        for event in state.events
    )
    assert any(
        event.kind == "tiebreak_entity_removed" and event.get("uid") == troop.uid
        for event in state.events
    )
    assert any(
        event.kind == "tiebreak_entity_removed" and event.get("uid") == building.uid
        for event in state.events
    )
    engine.validate_state(state)


def test_phoenix_egg_uses_troop_damage_and_tower_targeting_but_not_building_only_targets() -> None:
    engine, state = _state()
    egg = _entity(state, "phoenix-egg", 1, 9_000, 15_000)
    tower = engine._tower(state, 0, "king")
    hog = _entity(state, "hog-rider", 0, 9_000, 15_000)

    assert engine._target_allowed(tower, egg)
    assert engine._spell_can_hit("earthquake", egg)
    assert not engine._target_allowed(hog, egg)

    earthquake = RULESET.card("earthquake")
    before = egg.hp
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="earthquake",
        x_mtile=egg.x_mtile,
        y_mtile=egg.y_mtile,
        default_radius=int(earthquake.area_radius_mtile or 0),
        default_damage=int(earthquake.damage or 0),
        default_crown_damage=int(earthquake.crown_tower_damage or 0),
        default_status=None,
        default_knockback=0,
        raw_effect=earthquake.mechanics["persistent_effect"],
    )
    assert before - egg.hp == 82


def test_phoenix_egg_is_pullable_by_current_tornado_and_fisherman_rules() -> None:
    engine, state = _state()
    egg = _entity(state, "phoenix-egg", 1, 10_000, 15_000)
    tornado = RULESET.card("tornado")
    old_x = egg.x_mtile
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="tornado",
        x_mtile=9_000,
        y_mtile=15_000,
        default_radius=int(tornado.area_radius_mtile or 0),
        default_damage=int(tornado.damage or 0),
        default_crown_damage=int(tornado.crown_tower_damage or 0),
        default_status=None,
        default_knockback=0,
        raw_effect=tornado.mechanics["persistent_effect"],
    )
    assert egg.x_mtile < old_x

    fisherman = _entity(state, "fisherman", 0, 7_000, 15_000)
    egg.x_mtile = 13_000
    old_fisherman_x = fisherman.x_mtile
    engine._apply_hook(
        state,
        fisherman,
        egg,
        RULESET.card("fisherman").mechanics["hook"],
    )
    assert egg.x_mtile < 13_000
    assert fisherman.x_mtile == old_fisherman_x


def test_rage_accelerates_phoenix_egg_hatching() -> None:
    engine, state = _state()
    egg = engine._spawn_single_at(
        state,
        RULESET.card("phoenix-egg"),
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        deploy_remaining_us=0,
    )
    engine._apply_status(
        state,
        egg,
        {
            "kind": "rage",
            "duration_us": 100_000,
            "speed_multiplier_milli": 1_300,
            "hit_speed_multiplier_milli": 1_300,
        },
    )

    engine._advance_statuses_and_lifetimes(state)

    assert egg.lifetime_remaining_us == 3_735_000


def test_clone_can_copy_phoenix_egg_and_preserves_clone_hp_on_hatch() -> None:
    engine, state = _state()
    egg = engine._spawn_single_at(
        state,
        RULESET.card("phoenix-egg"),
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        deploy_remaining_us=0,
    )
    engine._impact_clone(
        state,
        owner=0,
        source_uid=None,
        source_card_id="clone",
        x=egg.x_mtile,
        y=egg.y_mtile,
        radius=3_000,
        raw_clone=RULESET.card("clone").mechanics["clone"],
    )
    cloned_egg = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "phoenix-egg" and entity.is_clone
    )
    assert (cloned_egg.hp, cloned_egg.max_hp) == (1, 1)

    cloned_egg.lifetime_remaining_us = RULESET.tick_us
    engine._advance_statuses_and_lifetimes(state)
    engine._resolve_deaths(state)

    reborn = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "phoenix" and entity.parent_uid == cloned_egg.uid
    )
    assert reborn.is_clone
    assert (reborn.hp, reborn.max_hp) == (1, 1)


def test_tiebreak_with_equal_lowest_towers_is_a_draw_without_destruction() -> None:
    engine, state = _state()
    state.phase = "overtime"
    engine._tower(state, 0, "left").hp = 1_500
    engine._tower(state, 1, "left").hp = 1_500
    troop = _entity(state, "knight", 0, 9_000, 15_000)
    projectile_uid = engine._allocate_uid(state)
    state.projectiles[projectile_uid] = ProjectileState(
        uid=projectile_uid,
        source_uid=troop.uid,
        source_card_id="fireball",
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        target_x_mtile=9_000,
        target_y_mtile=14_000,
        damage=1,
        crown_damage=1,
        speed_mtile_per_s=1_000,
    )
    effect_uid = engine._allocate_uid(state)
    state.effects[effect_uid] = AreaEffectState(
        uid=effect_uid,
        source_uid=troop.uid,
        source_card_id="poison",
        owner=0,
        x_mtile=9_000,
        y_mtile=14_000,
        radius_mtile=1_000,
        remaining_us=1_000_000,
        tick_interval_us=1_000_000,
    )

    engine._resolve_tiebreak(state)

    assert state.terminal
    assert state.winner is None
    assert state.terminal_reason == "tiebreak_equal_lowest_hp"
    assert engine._tower(state, 0, "left").alive
    assert engine._tower(state, 1, "left").alive
    assert not troop.alive and troop.hp == 0
    assert not state.projectiles[projectile_uid].alive
    assert not state.effects[effect_uid].alive
    assert any(event.kind == "tiebreak_projectiles_removed" for event in state.events)
    assert any(event.kind == "tiebreak_effects_removed" for event in state.events)


def test_sight_ranges_use_card_specific_hidden_values_and_keep_official_overrides() -> None:
    for card_id, expected in SIGHT_RANGE_MOBILE.items():
        assert RULESET.card(card_id).sight_range_mtile == expected, card_id
    assert RULESET.card("dart-goblin").sight_range_mtile == 7_000
    assert RULESET.card("firecracker").sight_range_mtile == 8_000


def test_royal_recruits_use_full_width_line_and_documented_split_positions() -> None:
    engine, state = _state()
    card = RULESET.card("royal-recruits")

    engine._spawn_card_entities(state, 0, card, (8, 20))
    central = sorted(
        entity.x_mtile
        for entity in state.entities.values()
        if entity.card_id == "royal-recruits" and entity.owner == 0
    )
    assert central == [3_500, 5_500, 7_500, 9_500, 11_500, 13_500]
    assert len({
        entity.y_mtile
        for entity in state.entities.values()
        if entity.card_id == "royal-recruits" and entity.owner == 0
    }) == 1
    assert sum(x < 9_000 for x in central) == 3
    assert sum(x > 9_000 for x in central) == 3

    shifted_engine, shifted_state = _state()
    shifted_engine._spawn_card_entities(state=shifted_state, player=0, card=card, cell=(7, 20))
    shifted = sorted(
        entity.x_mtile
        for entity in shifted_state.entities.values()
        if entity.card_id == "royal-recruits" and entity.owner == 0
    )
    assert sum(x < 9_000 for x in shifted) == 4
    assert sum(x > 9_000 for x in shifted) == 2


def test_unlocked_troop_reacquires_nearer_target_but_locked_troop_does_not() -> None:
    engine, state = _state()
    attacker = _entity(state, "skeletons", 0, 9_000, 15_000)
    farther = _entity(state, "giant", 1, 9_000, 10_000)
    nearer = _entity(state, "giant", 1, 9_000, 13_000)
    attacker.target_uid = farther.uid

    engine._invalidate_and_acquire_targets(state)
    assert attacker.target_uid == nearer.uid

    attacker.target_uid = farther.uid
    attacker.windup_remaining_us = 100_000
    engine._invalidate_and_acquire_targets(state)
    assert attacker.target_uid == farther.uid


def test_building_targeter_rechecks_closest_building_after_lock() -> None:
    engine, state = _state()
    attacker = _entity(state, "giant", 0, 9_000, 12_000)
    farther = _entity(state, "cannon", 1, 9_000, 13_000)
    nearer = _entity(state, "cannon", 1, 9_000, 12_500)
    attacker.target_uid = farther.uid

    engine._invalidate_and_acquire_targets(state)
    assert attacker.target_uid == nearer.uid


def test_load_state_requires_public_event_history_and_round_trips_observation() -> None:
    env = SimulatorEnv(
        engine=BattleEngine(RULESET),
        decision_interval_us=50_000,
    )
    env.reset(decks=(PLAYER_DECK, PLAYER_DECK), shuffle_decks=False)
    state = env.state
    assert state is not None
    slot = state.players[1].hand.index("skeletons")
    cell = env.engine.legal_action_cells(state, 1)[slot][0]
    env.step((WaitAction(0), PlayCardAction(1, slot, cell)))
    expected = env.observe()

    with pytest.raises(ValueError, match="event-inclusive"):
        env.load_state(state.to_primitive(include_events=False))

    restored = SimulatorEnv(
        engine=BattleEngine(RULESET),
        decision_interval_us=50_000,
    )
    actual = restored.load_state(env.save_state())
    for expected_view, actual_view in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(expected_view.board, actual_view.board)
        np.testing.assert_array_equal(expected_view.global_vector, actual_view.global_vector)
        np.testing.assert_array_equal(expected_view.spatial_masks, actual_view.spatial_masks)
        np.testing.assert_array_equal(expected_view.legal_play, actual_view.legal_play)
        assert expected_view.legal_wait == actual_view.legal_wait


def test_tiebreak_caps_crowns_when_equal_lowest_towers_fall_together() -> None:
    engine, state = _state()
    state.phase = "overtime"
    state.players[0].crowns = 2
    engine._tower(state, 1, "left").hp = 1_500
    engine._tower(state, 1, "right").hp = 1_500

    engine._resolve_tiebreak(state)

    assert state.terminal
    assert state.winner == 0
    assert state.players[0].crowns == 3
    assert not engine._tower(state, 1, "left").alive
    assert not engine._tower(state, 1, "right").alive
    engine.validate_state(state)


def test_diagonal_one_unit_overlap_makes_separation_progress() -> None:
    engine, state = _state()
    left = _entity(state, "knight", 0, 9_000, 18_000)
    right = _entity(state, "knight", 0, 9_565, 18_565)
    minimum = engine._collision_radius(left) + engine._collision_radius(right)

    engine._separate_entities(state)

    assert distance_mtile(left.x_mtile, left.y_mtile, right.x_mtile, right.y_mtile) >= minimum
    engine.validate_state(state)


def test_suspicious_bush_uses_authored_long_trigger_and_child_hp() -> None:
    engine, state = _state()
    bush = _entity(state, "suspicious-bush", 0, 9_000, 15_000, hp=81)
    building = _entity(state, "cannon", 1, 9_000, 17_000)
    bush.target_uid = building.uid
    engine._move_entities(state)
    assert bush.hp == 0
    engine._resolve_deaths(state)
    children = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "bush-goblin"
    ]
    assert len(children) == 2
    assert {child.hp for child in children} == {337}
