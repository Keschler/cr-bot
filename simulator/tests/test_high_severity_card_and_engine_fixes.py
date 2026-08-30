from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.fixed import distance_mtile
from simulator.navigation import point_is_walkable
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState, ProjectileState


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
