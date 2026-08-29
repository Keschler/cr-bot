"""Focused regression coverage for transport and swept projectile mechanics."""

from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState


RULESET = load_ruleset("v1")


def _state() -> tuple[BattleEngine, object]:
    engine = BattleEngine(RULESET)
    return engine, engine.new_battle(
        decks=(PLAYER_DECK, PLAYER_DECK),
        seed=1201,
        shuffle_decks=False,
    )


def _entity(
    state,
    card_id: str,
    owner: int,
    x_mtile: int,
    y_mtile: int,
    *,
    hp: int | None = None,
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
    )
    state.next_uid += 1
    state.entities[entity.uid] = entity
    return entity


def test_bowler_boulder_hits_ground_bodies_across_its_swept_path() -> None:
    engine, state = _state()
    bowler = _entity(state, "bowler", 0, 7_000, 15_000)
    first = _entity(state, "giant", 1, 9_000, 15_000)
    second = _entity(state, "giant", 1, 10_500, 15_000)
    bowler.pending_target_uid = first.uid

    engine._resolve_attack(state, bowler)
    projectile = next(iter(state.projectiles.values()))
    assert projectile.piercing is True
    # Resolve the boulder at its endpoint.  The swept resolver must still
    # inspect every body between the source and endpoint, not only the target
    # acquired by the attack scheduler.
    projectile.x_mtile = projectile.target_x_mtile
    projectile.y_mtile = projectile.target_y_mtile
    engine._impact_piercing_projectile(state, projectile)

    expected_damage = int(RULESET.card("bowler").damage or 0)
    assert first.hp == first.max_hp - expected_damage
    assert second.hp == second.max_hp - expected_damage
    assert sum(event.kind == "piercing_hit" for event in state.events) == 2
    engine.validate_state(state)


def test_firecracker_emits_five_behind_target_shrapnel_projectiles() -> None:
    engine, state = _state()
    firecracker = _entity(state, "firecracker", 0, 7_000, 15_000)
    primary = _entity(state, "giant", 1, 9_000, 15_000)
    behind = _entity(state, "giant", 1, 11_000, 15_000)
    firecracker.pending_target_uid = primary.uid

    engine._resolve_attack(state, firecracker)
    primary_projectile = next(iter(state.projectiles.values()))
    primary_projectile.x_mtile = primary_projectile.target_x_mtile
    primary_projectile.y_mtile = primary_projectile.target_y_mtile
    engine._impact_projectile(state, primary_projectile)

    shrapnels = [
        projectile
        for projectile in state.projectiles.values()
        if projectile.source_card_id == "firecracker" and projectile.target_uid is None
    ]
    assert len(shrapnels) == 5
    assert all(projectile.piercing for projectile in shrapnels)
    assert all(primary.uid in projectile.hit_uids for projectile in shrapnels)
    assert len({(p.target_x_mtile, p.target_y_mtile) for p in shrapnels}) == 5

    # The central fragment travels through the body behind the primary target;
    # the primary splash radius alone does not reach this target.
    for projectile in shrapnels:
        projectile.x_mtile = projectile.target_x_mtile
        projectile.y_mtile = projectile.target_y_mtile
        engine._impact_piercing_projectile(state, projectile)
    assert behind.hp < behind.max_hp
    assert primary.hp == primary.max_hp - int(RULESET.card("firecracker").damage or 0)
    engine.validate_state(state)


def test_skeleton_barrel_drops_on_building_contact_and_spawns_seven_skeletons() -> None:
    engine, state = _state()
    tower = next(
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == 1 and entity.role == "king"
    )
    barrel = _entity(state, "skeleton-barrel", 0, tower.x_mtile, tower.y_mtile + 1_800)
    barrel.target_uid = tower.uid

    engine._move_entities(state)
    assert barrel.hp == 0
    assert any(
        event.kind == "entity_triggered"
        and event.get("uid") == barrel.uid
        and event.get("target_uid") == tower.uid
        for event in state.events
    )
    engine._resolve_deaths(state)

    skeletons = [
        entity
        for entity in state.entities.values()
        if entity.card_id == "skeletons" and entity.owner == barrel.owner
    ]
    assert len(skeletons) == 7
    assert tower.hp == tower.max_hp - int(RULESET.card("skeleton-barrel").mechanics["death"]["crown_tower_damage"])
    engine.validate_state(state)


def test_skeleton_barrel_crosses_legacy_melee_edge_before_contact_trigger() -> None:
    engine, state = _state()
    barrel = _entity(state, "skeleton-barrel", 0, 3_500, 14_500)
    target = _entity(state, "cannon", 1, 3_500, 18_500)
    barrel.target_uid = target.uid

    for _ in range(200):
        engine._move_entities(state)
        if any(
            event.kind == "entity_triggered"
            and event.get("uid") == barrel.uid
            for event in state.events
        ):
            break

    assert any(
        event.kind == "entity_triggered"
        and event.get("uid") == barrel.uid
        and event.get("target_uid") == target.uid
        for event in state.events
    )
    assert engine._edge_distance(barrel, target) <= 250
