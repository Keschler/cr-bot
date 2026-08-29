from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState, ProjectileState, battle_state_from_primitive


RULESET = load_ruleset("v1")


def _state():
    engine = BattleEngine(RULESET)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=930, shuffle_decks=False)
    return engine, state


def _entity(state, card_id: str, owner: int, x: int, y: int, hp: int | None = None):
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


def test_goblin_hut_latches_only_on_visible_enemy_proximity() -> None:
    engine, state = _state()
    hut = engine._spawn_single_at(state, RULESET.card("goblin-hut"), owner=0, x_mtile=9_000, y_mtile=15_000)
    hut.deploy_remaining_us = 0
    hidden = _entity(state, "royal-ghost", 1, 9_000, 18_000)
    hidden.stealth_active = True
    engine._advance_spawners(state, RULESET.tick_us)
    assert not hut.spawner_active

    hidden.stealth_active = False
    engine._advance_spawners(state, RULESET.tick_us)
    assert hut.spawner_active
    assert any(event.kind == "spawner_activation_changed" for event in state.events)
    restored = battle_state_from_primitive(state.to_primitive())
    assert restored.entities[hut.uid].spawner_active


def test_elixir_golem_family_rewards_opponent_by_stage() -> None:
    engine, state = _state()
    state.players[1].elixir_milli = 0
    expected = {"elixir-golem": 1_000, "elixir-golemite": 500, "elixir-blob": 500}
    for index, (card_id, reward) in enumerate(expected.items()):
        body = _entity(state, card_id, 0, 4_000 + index * 2_000, 15_000, hp=1)
        body.hp = 0
        before = state.players[1].elixir_milli
        engine._resolve_deaths(state)
        assert state.players[1].elixir_milli == before + reward


def test_electro_spirit_chain_is_delayed_and_stuns_each_target() -> None:
    engine, state = _state()
    targets = [_entity(state, "giant", 1, 8_000 + i * 1_500, 15_000) for i in range(3)]
    projectile = ProjectileState(
        uid=state.next_uid, source_uid=None, source_card_id="electro-spirit", owner=0,
        x_mtile=8_000, y_mtile=15_000, target_x_mtile=8_000, target_y_mtile=15_000,
        target_uid=targets[0].uid, damage=99, crown_damage=99,
        speed_mtile_per_s=0, status_kind="stun", status_duration_us=500_000,
        status_magnitude_permille=0,
    )
    state.next_uid += 1
    state.projectiles[projectile.uid] = projectile
    engine._advance_projectiles(state)
    assert targets[0].hp == targets[0].max_hp - 99
    assert targets[1].hp == targets[1].max_hp
    for _ in range(5):
        engine._advance_projectiles(state)
    assert targets[1].hp == targets[1].max_hp - 99
    assert targets[0].statuses and targets[1].statuses
    restored = battle_state_from_primitive(state.to_primitive())
    assert restored.state_hash() == state.state_hash()
    engine.validate_state(restored)


def test_state_round_trip_preserves_navigation_revision() -> None:
    engine, state = _state()
    state.navigation_revision = 17

    restored = battle_state_from_primitive(state.to_primitive())

    assert restored.navigation_revision == 17


def test_tesla_concealment_has_earthquake_and_freeze_exceptions() -> None:
    engine, state = _state()
    tesla = engine._spawn_single_at(state, RULESET.card("tesla"), owner=0, x_mtile=9_000, y_mtile=15_000)
    tesla.deploy_remaining_us = 0
    assert tesla.concealed_active
    assert not engine._spell_can_hit("fireball", tesla)
    assert engine._spell_can_hit("earthquake", tesla)
    assert engine._spell_can_hit("freeze", tesla)
    _entity(state, "giant", 1, 9_000, 19_000)
    engine._advance_concealment(state)
    assert not tesla.concealed_active


def test_spell_dispatch_uses_authored_origin_and_continuous_path() -> None:
    engine, state = _state()
    selected_cell = (9, 10)
    selected_x, selected_y = 9_500, 10_500

    engine._spawn_spell(state, 0, RULESET.card("fireball"), selected_cell)
    fireball = next(projectile for projectile in state.projectiles.values())
    king = engine._tower(state, 0, "king")
    assert (fireball.x_mtile, fireball.y_mtile) == (king.x_mtile, king.y_mtile)
    assert (fireball.target_x_mtile, fireball.target_y_mtile) == (selected_x, selected_y)
    assert (fireball.origin_x_mtile, fireball.origin_y_mtile) == (king.x_mtile, king.y_mtile)
    assert fireball.piercing is False

    engine._spawn_spell(state, 0, RULESET.card("log"), selected_cell)
    log = max(state.projectiles.values(), key=lambda projectile: projectile.uid)
    assert (log.x_mtile, log.y_mtile) == (selected_x, selected_y)
    assert log.origin_x_mtile == selected_x
    assert log.origin_y_mtile == selected_y
    assert log.piercing is True
    assert log.target_y_mtile < selected_y

    path_target = _entity(state, "giant", 1, selected_x, 9_000)
    before = path_target.hp
    for _ in range(20):
        engine._advance_projectiles(state)
        if not log.alive:
            break
    assert path_target.hp < before


def test_archers_one_shot_a_spirit_target() -> None:
    engine, state = _state()
    archer = _entity(state, "archers", 0, 7_000, 15_000)
    spirit = _entity(state, "fire-spirit", 1, 9_000, 15_000)
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=archer.uid,
        source_card_id="archers",
        owner=0,
        x_mtile=spirit.x_mtile,
        y_mtile=spirit.y_mtile,
        target_x_mtile=spirit.x_mtile,
        target_y_mtile=spirit.y_mtile,
        target_uid=spirit.uid,
        damage=112,
        crown_damage=112,
        speed_mtile_per_s=0,
    )
    state.next_uid += 1
    state.projectiles[projectile.uid] = projectile
    engine._advance_projectiles(state)
    assert spirit.hp == 0
    damage = next(event for event in state.events if event.kind == "damage_applied")
    assert damage.get("damage") == spirit.max_hp


def test_arrows_pulse_three_times_and_earthquake_uses_building_damage() -> None:
    engine, state = _state()
    victim = _entity(state, "giant", 1, 9_000, 15_000)
    arrows = RULESET.card("arrows")
    projectile = ProjectileState(
        uid=state.next_uid, source_uid=None, source_card_id="arrows", owner=0,
        x_mtile=9_000, y_mtile=15_000, target_x_mtile=9_000, target_y_mtile=15_000,
        damage=int(arrows.damage or 0), crown_damage=int(arrows.crown_tower_damage or 0),
        speed_mtile_per_s=0, radius_mtile=int(arrows.area_radius_mtile or 0),
    )
    state.next_uid += 1
    engine._impact_projectile(state, projectile)
    for _ in range(8):
        engine._advance_area_effects(state)
    assert victim.hp == victim.max_hp - 369

    building = _entity(state, "cannon", 1, 12_000, 15_000)
    engine._create_area_effect(
        state, owner=0, source_uid=None, source_card_id="earthquake",
        x_mtile=12_000, y_mtile=15_000, default_radius=3_500,
        default_damage=82, default_crown_damage=49, default_status=None,
        default_knockback=0, raw_effect=RULESET.card("earthquake").mechanics["persistent_effect"],
    )
    assert building.hp == building.max_hp - 287
    assert building.statuses and building.statuses[0].magnitude_permille == 500


def test_building_death_payloads_and_barbarian_barrel_endpoint_spawn() -> None:
    engine, state = _state()
    for card_id, child_id, count in (
        ("tombstone", "skeletons", 4),
        ("barbarian-hut", "barbarian", 1),
        ("goblin-hut", "spear-goblin", 1),
    ):
        parent = _entity(state, card_id, 0, 5_000, 15_000, hp=1)
        before = sum(entity.card_id == child_id for entity in state.entities.values())
        parent.hp = 0
        engine._resolve_deaths(state)
        after = sum(entity.card_id == child_id for entity in state.entities.values())
        assert after - before == count

    projectile = ProjectileState(
        uid=state.next_uid, source_uid=None, source_card_id="barbarian-barrel", owner=0,
        x_mtile=9_000, y_mtile=10_000, target_x_mtile=9_000, target_y_mtile=10_000,
        damage=233, crown_damage=0, speed_mtile_per_s=0, radius_mtile=1_300,
        piercing=True,
    )
    state.next_uid += 1
    state.projectiles[projectile.uid] = projectile
    before = sum(entity.card_id == "barbarian" for entity in state.entities.values())
    engine._advance_projectiles(state)
    assert sum(entity.card_id == "barbarian" for entity in state.entities.values()) == before + 1


def test_bomb_tower_death_bomb_zappies_stun_and_river_jump_routing() -> None:
    engine, state = _state()
    bomb_tower = _entity(state, "bomb-tower", 0, 9_000, 15_000, hp=1)
    victim = _entity(state, "giant", 1, 9_000, 16_000)
    bomb_tower.hp = 0
    engine._resolve_deaths(state)
    assert victim.hp == victim.max_hp
    for _ in range(60):
        engine._advance_area_effects(state)
    assert victim.hp == victim.max_hp - 222

    zappy = _entity(state, "zappies", 0, 5_000, 10_000)
    target = _entity(state, "giant", 1, 5_500, 10_000)
    zappy.pending_target_uid = target.uid
    engine._resolve_attack(state, zappy)
    assert target.statuses and target.statuses[0].kind == "stun"

    jumper = _entity(state, "hog-rider", 0, 9_000, 14_000)
    across = _entity(state, "cannon", 1, 9_000, 19_000)
    assert engine._movement_waypoint(state, jumper, across) == (
        across.x_mtile,
        across.y_mtile,
    )
    ordinary = _entity(state, "giant", 0, 8_000, 14_000)
    assert engine._movement_waypoint(state, ordinary, across) != (
        across.x_mtile,
        across.y_mtile,
    )
