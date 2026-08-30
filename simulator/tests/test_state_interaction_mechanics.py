from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.fixed import distance_mtile
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState


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

    engine._resolve_tiebreak(state)

    assert state.terminal
    assert state.winner == 0
    assert state.terminal_reason == "tiebreak_lowest_hp"
    assert not low.alive and low.hp == 0
    assert engine._tower(state, 0, "left").hp == 500
    assert not troop.alive and troop.hp == 0
    assert any(
        event.kind == "tower_destroyed" and event.get("uid") == low.uid
        for event in state.events
    )
    assert any(
        event.kind == "tiebreak_entity_removed" and event.get("uid") == troop.uid
        for event in state.events
    )
    engine.validate_state(state)


def test_tiebreak_with_equal_lowest_towers_is_a_draw_without_destruction() -> None:
    engine, state = _state()
    state.phase = "overtime"
    engine._tower(state, 0, "left").hp = 1_500
    engine._tower(state, 1, "left").hp = 1_500

    engine._resolve_tiebreak(state)

    assert state.terminal
    assert state.winner is None
    assert state.terminal_reason == "tiebreak_equal_lowest_hp"
    assert engine._tower(state, 0, "left").alive
    assert engine._tower(state, 1, "left").alive


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
