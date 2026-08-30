from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.fixed import distance_mtile
from simulator.navigation import plan_route, point_is_walkable, segment_is_walkable
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState, ProjectileState
from simulator.state import StatusState


RULESET = load_ruleset("v1")


def _state() -> tuple[BattleEngine, object]:
    engine = BattleEngine(RULESET)
    return engine, engine.new_battle(
        decks=(PLAYER_DECK, PLAYER_DECK), seed=8211, shuffle_decks=False
    )


def _entity(
    state,
    card_id: str,
    owner: int,
    x_mtile: int,
    y_mtile: int,
    *,
    hp: int | None = None,
    parent_uid: int | None = None,
) -> EntityState:
    card = RULESET.card(card_id)
    maximum = int(card.hitpoints or 1)
    entity = EntityState(
        uid=state.next_uid,
        card_id=card_id,
        owner=owner,
        kind=card.kind,
        x_mtile=x_mtile,
        y_mtile=y_mtile,
        hp=maximum if hp is None else hp,
        max_hp=maximum,
        spawn_tick=state.tick,
        parent_uid=parent_uid,
    )
    state.next_uid += 1
    state.entities[entity.uid] = entity
    return entity


def test_seed_aliases_have_one_canonical_replay_identity() -> None:
    engine = BattleEngine(RULESET)
    low = engine.new_battle(
        decks=(PLAYER_DECK, PLAYER_DECK), seed=0, shuffle_decks=True
    )
    aliased = engine.new_battle(
        decks=(PLAYER_DECK, PLAYER_DECK), seed=1 << 64, shuffle_decks=True
    )

    assert low.seed == aliased.seed == 0
    assert low.state_hash() == aliased.state_hash()
    assert low.event_log_hash() == aliased.event_log_hash()
    assert low.replay_hash() == aliased.replay_hash()


def test_poison_and_earthquake_slow_statuses_change_movement_only() -> None:
    expected = {"poison": 850, "earthquake": 500}
    for card_id, movement_multiplier in expected.items():
        engine, state = _state()
        target = _entity(state, "giant", 1, 9_000, 15_000)
        status = RULESET.card(card_id).mechanics["persistent_effect"]["status"]

        engine._apply_status(state, target, status)

        assert target.statuses[-1].kind == f"{card_id}-slow"
        assert engine._speed_multiplier(target) == movement_multiplier
        assert engine._hit_speed_multiplier(target) == 1_000


def test_high_severity_card_overlays_are_in_the_runtime_ruleset() -> None:
    expected_projectiles = {
        "arrows": (22_000, True),
        "archers": (12_000, True),
        "baby-dragon": (10_000, True),
        "bomber": (8_000, False),
        "bowler": (3_400, False),
        "hunter": (11_000, False),
        "mother-witch": (12_000, True),
        "goblin-demolisher": (8_000, False),
        "lava-pup": (10_000, True),
        "firecracker": (10_000, False),
        "x-bow": (32_000, True),
        "fireball": (12_000, False),
        "mortar": (6_000, False),
        "rocket": (7_000, False),
        "royal-delivery": (100_000, False),
        "goblin-barrel": (8_000, False),
    }
    for card_id, expected in expected_projectiles.items():
        projectile = RULESET.card(card_id).projectile
        assert projectile is not None
        assert (projectile.speed_mtile_per_s, projectile.homing) == expected

    assert RULESET.card("archers").first_hit_delay_us == 500_000
    assert RULESET.card("hunter").first_hit_delay_us == 700_000
    assert RULESET.card("mother-witch").first_hit_delay_us == 300_000
    assert RULESET.card("goblin-demolisher").first_hit_delay_us == 500_000
    assert RULESET.card("lava-pup").first_hit_delay_us == 1_000_000
    assert RULESET.card("firecracker").first_hit_delay_us == 650_000
    assert RULESET.card("night-witch").first_hit_delay_us == 750_000
    assert RULESET.card("zappies").first_hit_delay_us == 800_000
    for card_id, expected_first_hit in {
        "knight": 500_000,
        "barbarians": 400_000,
        "battle-ram": 350_000,
        "electro-spirit": 200_000,
        "giant": 500_000,
        "golem": 1_000_000,
        "mortar": 1_000_000,
        "skeleton-army": 500_000,
        "valkyrie": 100_000,
        "wall-breakers": 200_000,
    }.items():
        assert RULESET.card(card_id).first_hit_delay_us == expected_first_hit
    assert RULESET.card("goblin-barrel").damage == 0
    assert RULESET.card("phoenix").mechanics["death"]["crown_tower_damage"] == 163
    assert RULESET.card("night-witch").mechanics["death"]["spawn_count"] == 1
    assert RULESET.card("royal-delivery").mechanics["impact_targets"] == (
        "air",
        "ground",
    )
    assert RULESET.card("tesla").mechanics["building_footprint_size"] == 2
    for card_id in ("log", "earthquake", "bowler"):
        assert RULESET.card(card_id).mechanics["cannot_hit_jumping"] is True


def test_hunter_close_range_fan_lands_more_pellets_than_long_range_fan() -> None:
    def damage_at_distance(distance: int) -> int:
        engine, state = _state()
        hunter = _entity(state, "hunter", 0, 9_000, 15_000)
        target = _entity(
            state,
            "giant",
            1,
            9_000 + distance,
            15_000,
            hp=10_000,
        )
        hunter.pending_target_uid = target.uid

        engine._resolve_attack(state, hunter)
        projectiles = list(state.projectiles.values())
        assert len(projectiles) == 10
        for projectile in projectiles:
            projectile.x_mtile = projectile.target_x_mtile
            projectile.y_mtile = projectile.target_y_mtile
            engine._impact_projectile(state, projectile)
        return 10_000 - target.hp

    pellet_damage = int(RULESET.card("hunter").damage or 0)
    close_damage = damage_at_distance(500)
    far_damage = damage_at_distance(3_500)

    assert close_damage == pellet_damage * 10
    assert 0 < far_damage < close_damage


def test_goblin_barrel_has_no_impact_damage_but_spawns_three_goblins() -> None:
    engine, state = _state()
    tower = engine._tower(state, 1, "king")
    before_hp = tower.hp
    card = RULESET.card("goblin-barrel")
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=None,
        source_card_id="goblin-barrel",
        owner=0,
        x_mtile=tower.x_mtile,
        y_mtile=tower.y_mtile,
        target_x_mtile=tower.x_mtile,
        target_y_mtile=tower.y_mtile,
        damage=int(card.damage or 0),
        crown_damage=int(card.crown_tower_damage or 0),
        speed_mtile_per_s=0,
        radius_mtile=int(card.projectile.radius_mtile),
    )
    state.next_uid += 1
    state.projectiles[projectile.uid] = projectile

    engine._advance_projectiles(state)

    assert tower.hp == before_hp
    assert sum(
        entity.alive and entity.card_id == "goblin" and entity.parent_uid is None
        for entity in state.entities.values()
    ) == 3
    goblins = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "goblin" and entity.parent_uid is None
    ]
    assert {entity.deploy_remaining_us for entity in goblins} == {1_100_000}


def test_each_spawner_has_an_independent_max_alive_cap() -> None:
    engine, state = _state()
    first = _entity(state, "barbarian-hut", 0, 5_000, 20_000, hp=1_000)
    second = _entity(state, "barbarian-hut", 0, 13_000, 20_000, hp=1_000)
    first.deploy_remaining_us = second.deploy_remaining_us = 0
    first.spawn_cooldown_us = second.spawn_cooldown_us = 0

    engine._advance_spawners(state, RULESET.tick_us)
    first.spawn_cooldown_us = 0
    second.spawn_cooldown_us = 10_000_000
    engine._advance_spawners(state, RULESET.tick_us)

    first_children = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.parent_uid == first.uid
    ]
    assert len(first_children) == 6

    second.spawn_cooldown_us = 0
    engine._advance_spawners(state, RULESET.tick_us)
    second_children = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.parent_uid == second.uid
    ]
    assert len(second_children) == 6


def test_first_attack_preloads_before_a_troop_reaches_attack_range() -> None:
    engine, state = _state()
    attacker = _entity(state, "knight", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 10_000, 15_000)
    attacker.target_uid = target.uid

    for _ in range(10):
        engine._advance_attacks(state)
    assert attacker.attack_count == 0
    assert attacker.attack_load_remaining_us == 0

    # The target now moves into the Knight's 1.2-tile edge range.  Since the
    # first-hit clock was loaded while the Knight approached, this tick fires
    # immediately instead of waiting another 0.5 seconds.
    target.x_mtile = 8_800
    before = target.hp
    engine._advance_attacks(state)
    assert attacker.attack_count == 1
    assert target.hp < before


def test_nonzero_first_hit_delay_loads_while_target_is_out_of_range() -> None:
    engine, state = _state()
    attacker = _entity(state, "archers", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 13_000, 15_000)
    attacker.target_uid = target.uid

    engine._advance_attacks(state)
    assert attacker.attack_count == 0
    assert attacker.attack_load_remaining_us == 450_000

    for _ in range(9):
        engine._advance_attacks(state)
    assert attacker.attack_count == 0
    assert attacker.attack_load_remaining_us == 0

    target.x_mtile = 8_800
    engine._advance_attacks(state)
    assert attacker.attack_count == 1
    assert any(
        projectile.source_uid == attacker.uid
        and projectile.target_uid == target.uid
        for projectile in state.projectiles.values()
    )


def test_piercing_projectile_only_hits_each_moving_segment_once() -> None:
    engine, state = _state()
    first = _entity(state, "giant", 1, 8_000, 15_000)
    second = _entity(state, "giant", 1, 11_000, 15_000)
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=None,
        source_card_id="magic-archer",
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        target_x_mtile=12_000,
        target_y_mtile=15_000,
        damage=10,
        crown_damage=10,
        speed_mtile_per_s=0,
        piercing=True,
        origin_x_mtile=7_000,
        origin_y_mtile=15_000,
        line_end_x_mtile=12_000,
        line_end_y_mtile=15_000,
    )
    state.next_uid += 1
    state.projectiles[projectile.uid] = projectile

    engine._impact_piercing_projectile(state, projectile)
    first_after_first_segment = first.hp
    second_after_first_segment = second.hp
    projectile.x_mtile = 12_000
    engine._impact_piercing_projectile(state, projectile)

    assert first.hp == first_after_first_segment < first.max_hp
    assert second.hp < second_after_first_segment == second.max_hp


def test_graveyard_skeletons_are_bumped_to_legal_ground_positions() -> None:
    engine, state = _state()
    card = RULESET.card("graveyard")
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="graveyard",
        x_mtile=9_000,
        y_mtile=16_000,
        default_radius=int(card.area_radius_mtile or 0),
        default_damage=int(card.damage or 0),
        default_crown_damage=int(card.crown_tower_damage or 0),
        default_status=None,
        default_knockback=0,
        raw_effect=card.mechanics["persistent_effect"],
    )
    effect = next(iter(state.effects.values()))
    engine._apply_area_effect_tick(state, effect)
    skeleton = next(entity for entity in state.entities.values() if entity.card_id == "skeletons")

    assert point_is_walkable(
        RULESET.arena,
        skeleton.x_mtile,
        skeleton.y_mtile,
        int(RULESET.card("skeletons").collision_radius_mtile or 0),
    )
    assert distance_mtile(skeleton.x_mtile, skeleton.y_mtile, 9_000, 15_000) > 0
    engine.validate_state(state)


def test_hog_rider_repeat_hits_use_hit_speed_without_first_hit_windup() -> None:
    engine, state = _state()
    hog = _entity(state, "hog-rider", 0, 9_000, 15_000)
    target = _entity(state, "giant", 1, 10_500, 15_000)
    hog.target_uid = target.uid

    for _ in range(RULESET.card("hog-rider").first_hit_delay_us // RULESET.tick_us):
        engine._advance_attacks(state)
    assert hog.attack_count == 1

    for _ in range(RULESET.card("hog-rider").attack_interval_us // RULESET.tick_us):
        engine._advance_attacks(state)
    assert hog.attack_count == 2
    assert hog.windup_remaining_us == 0


def test_hunter_fan_reflects_once_per_attack_instance() -> None:
    engine, state = _state()
    hunter = _entity(state, "hunter", 0, 9_000, 15_000)
    giant = _entity(state, "electro-giant", 1, 9_500, 15_000)
    hunter.pending_target_uid = giant.uid

    before_hunter = hunter.hp
    engine._resolve_attack(state, hunter)
    for projectile in list(state.projectiles.values()):
        projectile.x_mtile = projectile.target_x_mtile
        projectile.y_mtile = projectile.target_y_mtile
        engine._impact_projectile(state, projectile)

    assert before_hunter - hunter.hp == int(
        RULESET.card("electro-giant").mechanics["reflection"]["damage"]
    )
    assert sum(event.kind == "reflected_damage" for event in state.events) == 1


def test_frozen_electro_giant_does_not_reflect_damage() -> None:
    engine, state = _state()
    hunter = _entity(state, "hunter", 0, 9_000, 15_000)
    giant = _entity(state, "electro-giant", 1, 9_500, 15_000)
    giant.statuses.append(StatusState(kind="freeze", remaining_us=1_000_000))
    hunter.pending_target_uid = giant.uid
    before_hunter = hunter.hp

    engine._resolve_attack(state, hunter)
    projectile = next(iter(state.projectiles.values()))
    projectile.x_mtile = projectile.target_x_mtile
    projectile.y_mtile = projectile.target_y_mtile
    engine._impact_projectile(state, projectile)

    assert giant.hp < giant.max_hp
    assert hunter.hp == before_hunter
    assert not any(event.kind == "reflected_damage" for event in state.events)


def test_crown_tower_does_not_target_a_placed_building() -> None:
    engine, state = _state()
    tower = engine._tower(state, 1, "left")
    building = _entity(state, "cannon", 0, tower.x_mtile - 2_000, tower.y_mtile)

    assert not engine._valid_target(state, tower, building.uid)
    assert not engine._spell_can_hit(tower.card_id, building)
    assert engine._choose_target(state, tower) is None


def test_fisherman_reels_himself_to_building_melee_range() -> None:
    engine, state = _state()
    fisherman = _entity(state, "fisherman", 0, 7_000, 15_000)
    building = _entity(state, "cannon", 1, 13_000, 15_000)
    hook = RULESET.card("fisherman").mechanics["hook"]

    engine._apply_hook(state, fisherman, building, hook)

    assert fisherman.x_mtile > 7_000
    assert engine._edge_distance(fisherman, building) <= int(
        RULESET.card("fisherman").range_mtile
    )
    assert any(event.kind == "hook_pulled" for event in state.events)
    assert not any(event.kind == "hook_noop" for event in state.events)


def test_lethal_in_flight_projectile_frees_attacker_to_retarget() -> None:
    engine, state = _state()
    attacker = _entity(state, "musketeer", 0, 9_000, 15_000)
    projectile_source = _entity(state, "archers", 0, 8_000, 15_000)
    reserved = _entity(state, "giant", 1, 10_000, 15_000, hp=100)
    replacement = _entity(state, "giant", 1, 11_000, 15_000)
    attacker.target_uid = reserved.uid
    state.projectiles[state.next_uid] = ProjectileState(
        uid=state.next_uid,
        source_uid=projectile_source.uid,
        source_card_id="archers",
        owner=0,
        x_mtile=8_500,
        y_mtile=15_000,
        target_x_mtile=reserved.x_mtile,
        target_y_mtile=reserved.y_mtile,
        target_uid=reserved.uid,
        damage=100,
        crown_damage=100,
        speed_mtile_per_s=12_000,
    )
    state.next_uid += 1

    engine._invalidate_and_acquire_targets(state)

    assert attacker.target_uid == replacement.uid
    in_flight = next(iter(state.projectiles.values()))
    in_flight.x_mtile = in_flight.target_x_mtile
    in_flight.y_mtile = in_flight.target_y_mtile
    engine._impact_projectile(state, in_flight)
    assert reserved.hp == 0


def test_bridge_walkability_accounts_for_unit_radius() -> None:
    start, end = RULESET.arena.bridge_x_ranges_mtile[0]
    y = (RULESET.arena.river_y_min_mtile + RULESET.arena.river_y_max_mtile) // 2
    radius = 500

    assert not point_is_walkable(RULESET.arena, start + radius - 1, y, radius)
    assert point_is_walkable(RULESET.arena, start + radius, y, radius)
    assert point_is_walkable(RULESET.arena, end - radius, y, radius)
    assert not point_is_walkable(RULESET.arena, end - radius + 1, y, radius)


def test_large_ground_units_can_route_through_the_three_tile_bridge() -> None:
    for card_id in ("ice-golem", "giant-skeleton"):
        radius = int(RULESET.card(card_id).collision_radius_mtile or 0)
        assert point_is_walkable(RULESET.arena, 3_500, 16_000, radius)
        route = plan_route(
            RULESET.arena,
            (3_500, 14_000),
            (3_500, 18_000),
            agent_radius_mtile=radius,
        )
        assert route
        assert route[0] == (3_500, 14_000)
        assert route[-1] == (3_500, 18_000)
        assert all(
            segment_is_walkable(
                RULESET.arena,
                start,
                end,
                agent_radius_mtile=radius,
            )
            for start, end in zip(route, route[1:])
        )


def test_tesla_uses_two_by_two_legality_in_engine_and_policy_cells() -> None:
    from simulator.geometry import building_footprint_fits

    engine, state = _state()
    cell = next(
        cell
        for row in range(17, 32)
        for col in range(18)
        for cell in ((col, row),)
        if building_footprint_fits(0, cell, 2)
        and not building_footprint_fits(0, cell, 3)
    )
    tesla = RULESET.card("tesla")

    assert engine._legal_deployment(state, 0, tesla, cell)
    assert cell in engine.legal_cells(state, 0, "tesla")


def test_log_earthquake_and_bowler_cannot_hit_a_jumping_mega_knight() -> None:
    engine, state = _state()
    knight = _entity(state, "mega-knight", 1, 9_000, 15_000)
    knight.jump_remaining_us = 200_000

    for card_id in ("log", "earthquake", "bowler"):
        assert not engine._spell_can_hit(card_id, knight)
        before = knight.hp
        engine._impact_area(
            state,
            owner=0,
            source_uid=None,
            source_card_id=card_id,
            x=knight.x_mtile,
            y=knight.y_mtile,
            damage=100,
            crown_damage=100,
            radius=1_000,
            status=None,
            knockback=0,
            primary_target_uid=None,
            allowed_targets=("ground",),
        )
        assert knight.hp == before

    assert engine._spell_can_hit("mortar", knight)
    assert engine._spell_can_hit("sparky", knight)


def test_bandit_dash_ignores_damage_and_hard_control_until_landing() -> None:
    engine, state = _state()
    bandit = _entity(state, "bandit", 0, 8_000, 15_000)
    target = _entity(state, "giant", 1, 11_000, 15_000)
    bandit.target_uid = target.uid

    engine._move_entities(state)
    assert bandit.dash_attack_active
    assert bandit.dash_remaining_us > 0
    before = bandit.hp
    engine._deal_damage(state, bandit, 1_000, target.uid, "giant")
    engine._apply_status(state, bandit, {"kind": "freeze", "duration_us": 1_000_000})
    assert bandit.hp == before
    assert bandit.dash_remaining_us > 0

    for _ in range(10):
        engine._advance_statuses_and_lifetimes(state)
        if bandit.dash_remaining_us == 0:
            break
    assert bandit.dash_remaining_us == 0


def test_bandit_dash_damage_resolves_immediately_after_landing() -> None:
    engine, state = _state()
    bandit = _entity(state, "bandit", 0, 8_000, 15_000)
    target = _entity(state, "giant", 1, 11_000, 15_000)
    bandit.target_uid = target.uid

    engine._move_entities(state)
    for _ in range(4):
        engine._advance_statuses_and_lifetimes(state)
    assert bandit.dash_remaining_us == 0

    engine._advance_attacks(state)

    assert target.max_hp - target.hp == int(RULESET.card("bandit").mechanics["dash"]["dash_damage"])
    assert bandit.attack_count == 1
    assert not bandit.dash_attack_active


def test_bandit_dash_misses_when_target_leaves_landing_range() -> None:
    engine, state = _state()
    bandit = _entity(state, "bandit", 0, 8_000, 15_000)
    target = _entity(state, "giant", 1, 11_000, 15_000)
    bandit.target_uid = target.uid

    engine._move_entities(state)
    target.x_mtile = 16_000
    for _ in range(4):
        engine._advance_statuses_and_lifetimes(state)
    before = target.hp
    engine._advance_attacks(state)

    assert target.hp == before
    assert not bandit.dash_attack_active

    target.x_mtile = 11_000
    engine._move_entities(state)
    for _ in range(8):
        engine._advance_attacks(state)
    assert target.max_hp - target.hp == int(RULESET.card("bandit").damage or 0)


def test_lumberjack_death_rage_uses_rage_damage_on_its_first_pulse() -> None:
    engine, state = _state()
    lumberjack = _entity(state, "lumberjack", 0, 9_000, 15_000)
    target = _entity(state, "giant", 1, 10_000, 15_000)
    lumberjack.hp = 0

    engine._resolve_deaths(state)

    assert target.max_hp - target.hp == int(RULESET.card("rage").damage or 0)
    effect = next(
        effect
        for effect in state.effects.values()
        if effect.source_uid == lumberjack.uid
    )
    assert effect.damage_schedule == (int(RULESET.card("rage").damage or 0),)
