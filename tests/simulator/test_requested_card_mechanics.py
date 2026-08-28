"""Deterministic regression coverage for the remaining V1 card components.

These tests intentionally exercise the engine component boundary directly.
The same component events are emitted when a card is played through the
headless match loop, while direct setup keeps each interaction small enough to
pinpoint a first divergence in generated or video-mined scenarios.
"""

from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState


RULESET = load_ruleset("v1")


def _state() -> tuple[BattleEngine, object]:
    engine = BattleEngine(RULESET)
    state = engine.new_battle(
        decks=(PLAYER_DECK, PLAYER_DECK),
        seed=801,
        shuffle_decks=False,
    )
    return engine, state


def _entity(
    state,
    card_id: str,
    owner: int,
    x: int,
    y: int,
    *,
    hp: int | None = None,
) -> EntityState:
    card = RULESET.card(card_id)
    uid = state.next_uid
    state.next_uid += 1
    maximum = int(card.hitpoints or 1)
    entity = EntityState(
        uid=uid,
        card_id=card_id,
        owner=owner,
        kind=card.kind,
        x_mtile=x,
        y_mtile=y,
        hp=maximum if hp is None else hp,
        max_hp=maximum,
        spawn_tick=state.tick,
    )
    state.entities[uid] = entity
    return entity


def _spawn_card(engine: BattleEngine, state, card_id: str, owner: int, cell=(9, 20)) -> list[EntityState]:
    before = set(state.entities)
    engine._spawn_card_entities(state, owner, RULESET.card(card_id), cell)
    return [state.entities[uid] for uid in sorted(set(state.entities) - before)]


def test_shield_layers_absorb_a_hit_without_body_damage_then_break() -> None:
    engine, state = _state()
    expected = {"dark-prince": 240, "guards": 256, "royal-recruits": 240}
    spawned_count = 0

    for index, (card_id, shield_hp) in enumerate(expected.items()):
        spawned = _spawn_card(engine, state, card_id, 0, (2 + index * 4, 20))
        spawned_count += len(spawned)
        assert spawned
        for entity in spawned:
            assert entity.shield_hp == shield_hp
            body_before = entity.hp
            engine._deal_damage(state, entity, shield_hp + 100, None, "test")
            assert entity.shield_hp == 0
            assert entity.hp == body_before
            engine._deal_damage(state, entity, 1, None, "test")
            assert entity.hp == body_before - 1

    assert sum(event.kind == "shield_broken" for event in state.events) == spawned_count
    engine.validate_state(state)


def test_royal_ghost_reveals_on_attack_and_recloaks_after_delay() -> None:
    engine, state = _state()
    ghost = _spawn_card(engine, state, "royal-ghost", 0)[0]
    ghost.deploy_remaining_us = 0
    target = _entity(state, "giant", 1, ghost.x_mtile + 1_000, ghost.y_mtile)
    ghost.pending_target_uid = target.uid

    engine._resolve_attack(state, ghost)
    assert ghost.stealth_active is False
    assert ghost.stealth_remaining_us == 1_500_000
    assert any(event.kind == "stealth_broken" for event in state.events)

    for _ in range(30):
        engine._advance_statuses_and_lifetimes(state)
    assert ghost.stealth_active is True
    assert any(event.kind == "stealth_started" for event in state.events)
    engine.validate_state(state)


def test_miner_can_tunnel_to_any_ground_cell_but_is_hidden_until_emergence() -> None:
    engine, state = _state()
    miner = RULESET.card("miner")
    assert engine._legal_deployment(state, 0, miner, (9, 5))
    # The anywhere exception still cannot overlap a structure.  Use a
    # ground cell away from the fixed towers so the failure is specifically
    # the building-footprint rule rather than arena terrain.
    building_cell = (9, 5)
    _entity(state, "cannon", 1, 9_500, 5_500)
    assert not engine._legal_deployment(state, 0, miner, building_cell)

    spawned = _spawn_card(engine, state, "miner", 0, (9, 6))
    miner_entity = spawned[0]
    assert miner_entity.burrow_active is True
    assert miner_entity.deploy_remaining_us == 1_000_000
    assert engine._targetable_for_acquisition(state, miner_entity) is False

    for _ in range(20):
        engine._advance_deployments(state)
    assert miner_entity.burrow_active is False
    assert miner_entity.deploy_remaining_us == 0
    assert any(event.kind == "burrow_emerged" for event in state.events)
    engine.validate_state(state)


def test_mixed_swarm_cards_spawn_their_real_child_compositions() -> None:
    engine, state = _state()
    gang = _spawn_card(engine, state, "goblin-gang", 1, (9, 10))
    assert [entity.card_id for entity in gang].count("goblin") == 3
    assert [entity.card_id for entity in gang].count("spear-goblin") == 3
    assert all(entity.max_hp in {133, 202} for entity in gang)

    rascals = _spawn_card(engine, state, "rascals", 1, (9, 11))
    assert [entity.card_id for entity in rascals].count("rascal-boy") == 1
    assert [entity.card_id for entity in rascals].count("rascal-girl") == 2
    assert next(entity for entity in rascals if entity.card_id == "rascal-boy").max_hp == 1_940
    assert all(
        entity.max_hp == 202
        for entity in rascals
        if entity.card_id == "rascal-girl"
    )
    engine.validate_state(state)


def test_magic_archer_hits_only_bodies_on_its_piercing_line() -> None:
    engine, state = _state()
    archer = _entity(state, "magic-archer", 0, 7_000, 15_000)
    aligned_a = _entity(state, "giant", 1, 9_000, 15_000, hp=1_000)
    aligned_b = _entity(state, "giant", 1, 11_000, 15_000, hp=1_000)
    off_line = _entity(state, "giant", 1, 11_000, 16_500, hp=1_000)
    archer.pending_target_uid = aligned_a.uid
    engine._resolve_attack(state, archer)
    projectile = next(iter(state.projectiles.values()))
    for _ in range(30):
        engine._advance_projectiles(state)
        if not projectile.alive:
            break

    assert aligned_a.hp == 1_000 - int(RULESET.card("magic-archer").damage or 0)
    assert aligned_b.hp == aligned_a.hp
    assert off_line.hp == 1_000
    assert sum(event.kind == "piercing_hit" for event in state.events) == 2
    engine.validate_state(state)


def test_executioner_axe_hits_on_outbound_and_return_passes() -> None:
    engine, state = _state()
    executioner = _entity(state, "executioner", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 9_000, 15_000, hp=4_000)
    executioner.pending_target_uid = target.uid
    engine._resolve_attack(state, executioner)
    projectile = next(iter(state.projectiles.values()))
    for _ in range(100):
        engine._advance_projectiles(state)
        if not projectile.alive:
            break

    assert target.hp == 4_000 - 2 * int(RULESET.card("executioner").damage or 0)
    assert any(event.kind == "projectile_return_started" for event in state.events)
    assert sum(event.kind == "piercing_hit" for event in state.events) == 2
    engine.validate_state(state)


def test_hunter_emits_a_deterministic_ten_pellet_fan() -> None:
    engine, state = _state()
    hunter = _entity(state, "hunter", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 9_000, 15_000)
    hunter.pending_target_uid = target.uid
    engine._resolve_attack(state, hunter)
    projectiles = sorted(state.projectiles.values(), key=lambda projectile: projectile.uid)
    assert len(projectiles) == 10
    assert [projectile.pellet_index for projectile in projectiles] == list(range(10))
    assert len({(projectile.target_x_mtile, projectile.target_y_mtile) for projectile in projectiles}) > 1
    before_targets = [
        (projectile.target_x_mtile, projectile.target_y_mtile) for projectile in projectiles
    ]
    engine._advance_projectiles(state)
    assert [
        (projectile.target_x_mtile, projectile.target_y_mtile)
        for projectile in projectiles
    ] == before_targets
    assert all(projectile.homing is False for projectile in projectiles)
    engine.validate_state(state)


def test_bowler_knockback_follows_projectile_direction() -> None:
    engine, state = _state()
    bowler = _entity(state, "bowler", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 9_000, 15_000)
    bowler.pending_target_uid = target.uid
    engine._resolve_attack(state, bowler)
    projectile = next(iter(state.projectiles.values()))
    engine._impact_projectile(state, projectile)
    assert target.x_mtile == 10_500
    assert target.y_mtile == 15_000
    engine.validate_state(state)


def test_mega_knight_jumps_and_lands_with_area_damage() -> None:
    engine, state = _state()
    knight = _entity(state, "mega-knight", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 9_500, 15_000, hp=4_000)
    knight.target_uid = target.uid
    engine._move_entities(state)
    assert knight.jump_remaining_us == 400_000
    assert any(event.kind == "jump_started" for event in state.events)
    for _ in range(8):
        engine._advance_statuses_and_lifetimes(state)
    assert knight.jump_remaining_us == 0
    assert target.hp == 4_000 - 268
    assert any(event.kind == "jump_landed" for event in state.events)
    engine.validate_state(state)


def test_electro_and_ice_wizard_deployment_control_pulse() -> None:
    for card_id, status_kind, duration in (
        ("electro-wizard", "stun", 500_000),
        ("ice-wizard", "freeze", 1_500_000),
    ):
        engine, state = _state()
        wizard = _spawn_card(engine, state, card_id, 0)[0]
        wizard.deploy_remaining_us = 1
        target = _entity(state, "giant", 1, wizard.x_mtile + 500, wizard.y_mtile)
        engine._advance_deployments(state)
        status = next(status for status in target.statuses if status.kind == status_kind)
        assert status.remaining_us == duration
        assert any(
            event.kind == "deployment_effect" and event.get("card_id") == card_id
            for event in state.events
        )
        engine.validate_state(state)


def test_lumberjack_death_creates_a_friendly_rage_area() -> None:
    engine, state = _state()
    lumberjack = _entity(state, "lumberjack", 0, 9_000, 15_000, hp=1)
    ally = _entity(state, "giant", 0, 9_500, 15_000)
    lumberjack.hp = 0
    engine._resolve_deaths(state)
    assert len(state.effects) == 1
    assert any(status.kind == "rage" for status in ally.statuses)
    assert any(event.kind == "death_rage_created" for event in state.events)
    engine.validate_state(state)


def test_mother_witch_curse_converts_a_lethal_target_to_one_cursed_hog() -> None:
    engine, state = _state()
    mother = _entity(state, "mother-witch", 0, 7_000, 15_000)
    victim = _entity(state, "skeletons", 1, 9_000, 15_000, hp=1)
    mother.pending_target_uid = victim.uid
    engine._resolve_attack(state, mother)
    projectile = next(iter(state.projectiles.values()))
    engine._impact_projectile(state, projectile)
    engine._resolve_deaths(state)
    cursed = [entity for entity in state.entities.values() if entity.alive and entity.card_id == "cursed-hog"]
    assert len(cursed) == 1
    assert cursed[0].owner == mother.owner
    assert any(event.kind == "death_transform" and event.get("source_card_id") == "mother-witch" for event in state.events)
    engine.validate_state(state)


def test_ram_rider_applies_a_movement_snare_on_hit() -> None:
    engine, state = _state()
    rider = _entity(state, "ram-rider", 0, 7_000, 15_000)
    target = _entity(state, "giant", 1, 8_500, 15_000)
    # The Ram's primary channel is building-only.  The rider's bola is an
    # independent troop-targeting weapon with its own wind-up/projectile.
    rider.secondary_pending_target_uid = target.uid
    engine._resolve_secondary_attack(state, rider)
    projectile = next(iter(state.projectiles.values()))
    engine._impact_projectile(state, projectile)
    snare = next(status for status in target.statuses if status.kind == "slow")
    assert snare.remaining_us == 1_500_000
    assert snare.magnitude_permille == 300
    assert snare.hit_speed_magnitude_permille == 1_000
    engine.validate_state(state)


def test_witch_death_spawns_three_skeletons_in_addition_to_her_wave_stream() -> None:
    engine, state = _state()
    witch = _entity(state, "witch", 0, 9_000, 15_000, hp=1)
    witch.hp = 0
    engine._resolve_deaths(state)
    skeletons = [entity for entity in state.entities.values() if entity.alive and entity.card_id == "skeletons"]
    assert len(skeletons) == 3
    assert any(
        event.kind == "death_spawn"
        and event.get("parent_card_id") == "witch"
        and event.get("child_count") == 3
        for event in state.events
    )
    engine.validate_state(state)
