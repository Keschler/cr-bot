from __future__ import annotations

from simulator.actions import PlayCardAction
from simulator.engine import BattleEngine
from simulator.ruleset import load_ruleset
from simulator.state import EntityState, ProjectileState, battle_state_from_primitive
from simulator.roster import PLAYER_DECK


ROSTER = load_ruleset("v1")


def _entity(
    state,
    *,
    card_id: str,
    owner: int,
    x: int,
    y: int,
    hp: int | None = None,
) -> EntityState:
    card = ROSTER.card(card_id)
    uid = state.next_uid
    state.next_uid += 1
    maximum = int(card.hitpoints or 1)
    row = EntityState(
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
    state.entities[uid] = row
    return row


def test_battle_healer_heals_nearby_allies_but_not_self_or_another_healer() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=2, shuffle_decks=False)
    healer = _entity(state, card_id="battle-healer", owner=0, x=7_000, y=20_000)
    healer.target_uid = _entity(
        state, card_id="skeletons", owner=1, x=8_500, y=20_000, hp=50
    ).uid
    healer.pending_target_uid = healer.target_uid
    ally = _entity(state, card_id="hog-rider", owner=0, x=7_500, y=20_500, hp=1_000)
    other_healer = _entity(
        state, card_id="battle-healer", owner=0, x=7_700, y=20_500, hp=1_000
    )
    before_self = healer.hp

    engine._resolve_attack(state, healer)

    assert ally.hp == 1_100
    assert other_healer.hp == 1_000
    assert healer.hp == before_self
    assert any(
        event.kind == "healing_applied" and event.get("target_uid") == ally.uid
        for event in state.events
    )


def test_heal_spirit_impact_heals_friendly_air_and_ground_troops_after_damage() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=201, shuffle_decks=False)
    spirit = _entity(state, card_id="heal-spirit", owner=0, x=9_000, y=15_000, hp=1)
    spirit.hp = 0
    spirit.alive = False
    ground = _entity(state, card_id="giant", owner=0, x=9_200, y=15_000, hp=1_000)
    air = _entity(state, card_id="baby-dragon", owner=0, x=9_400, y=15_000, hp=1_000)
    full_health = _entity(state, card_id="musketeer", owner=0, x=9_500, y=15_000)
    building = _entity(state, card_id="cannon", owner=0, x=12_000, y=15_000, hp=500)
    enemy = _entity(state, card_id="giant", owner=1, x=9_000, y=15_000, hp=1_000)
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=spirit.uid,
        source_card_id="heal-spirit",
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        target_x_mtile=9_000,
        target_y_mtile=15_000,
        target_uid=enemy.uid,
        damage=110,
        crown_damage=110,
        speed_mtile_per_s=12_000,
        radius_mtile=1_500,
    )
    state.next_uid += 1

    engine._impact_projectile(state, projectile)

    assert enemy.hp == 890
    assert ground.hp == 1_532
    assert air.hp == air.max_hp == 1_152
    assert full_health.hp == full_health.max_hp
    assert building.hp == 500
    impact_events = [
        event
        for event in state.events
        if event.kind == "healing_impact_resolved"
    ]
    assert len(impact_events) == 1
    assert impact_events[0].get("recipient_count") == 2
    assert [event.kind for event in state.events].index("damage_applied") < [
        event.kind for event in state.events
    ].index("healing_applied")
    assert all(
        event.get("source_card_id") == "heal-spirit"
        for event in state.events
        if event.kind in {"healing_applied", "healing_impact_resolved"}
    )
    engine.validate_state(state)


def test_cannon_cart_transforms_in_place_at_shared_half_health_and_decays_as_building() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=202, shuffle_decks=False)
    cart = _entity(state, card_id="cannon-cart", owner=1, x=9_000, y=15_000)
    uid = cart.uid

    # 905 damage leaves 904/1809 HP, which is at or below the official 50%
    # transform threshold.  The event must preserve the UID and shared pool.
    engine._deal_damage(state, cart, 905, source_uid=None, source_card_id="test")

    assert cart.uid == uid
    assert cart.card_id == "cannon-cart-building"
    assert cart.kind == "building"
    assert cart.hp == 904
    assert cart.max_hp == 1_809
    assert cart.lifetime_remaining_us == 30_000_000
    assert any(
        event.kind == "entity_transformed"
        and event.get("source_card_id") == "cannon-cart"
        and event.get("target_card_id") == "cannon-cart-building"
        for event in state.events
    )

    # The stationary form no longer moves and follows the normal linear
    # building decay path.  A second damage event must not create a second
    # transform.
    before_position = (cart.x_mtile, cart.y_mtile)
    engine._advance_statuses_and_lifetimes(state)
    assert (cart.x_mtile, cart.y_mtile) == before_position
    assert cart.hp < 904
    assert sum(event.kind == "entity_transformed" for event in state.events) == 1
    engine.validate_state(state)


def test_golem_death_splits_into_two_golemites_with_nested_child_death_damage() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=203, shuffle_decks=False)
    golem = _entity(state, card_id="golem", owner=1, x=9_000, y=15_000, hp=1)
    golem.hp = 0

    engine._resolve_deaths(state)
    golemites = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "golemite"
    ]
    assert len(golemites) == 2
    assert all(entity.owner == golem.owner for entity in golemites)
    assert any(
        event.kind == "death_spawn"
        and event.get("parent_card_id") == "golem"
        and event.get("child_card_id") == "golemite"
        and event.get("child_count") == 2
        for event in state.events
    )

    # The child form has its own death burst.  Killing one child must not
    # respawn a Golem; it resolves only the child's 99-damage area effect.
    target = _entity(state, card_id="giant", owner=0, x=9_000, y=15_800, hp=200)
    golemites[0].hp = 0
    engine._resolve_deaths(state)
    assert target.hp == 101
    assert sum(event.kind == "death_spawn" for event in state.events) == 1
    engine.validate_state(state)


def test_goblin_cage_death_releases_one_brawler_not_the_goblins_formation() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=206, shuffle_decks=False)
    cage = _entity(state, card_id="goblin-cage", owner=1, x=9_000, y=15_000, hp=1)
    cage.hp = 0

    engine._resolve_deaths(state)

    brawlers = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "goblin-brawler"
    ]
    generic_goblins = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "goblins"
    ]
    assert len(brawlers) == 1
    assert not generic_goblins
    assert brawlers[0].owner == cage.owner
    assert brawlers[0].deploy_remaining_us == 0
    assert any(
        event.kind == "death_spawn"
        and event.get("parent_card_id") == "goblin-cage"
        and event.get("child_card_id") == "goblin-brawler"
        and event.get("child_count") == 1
        for event in state.events
    )
    engine.validate_state(state)


def test_elixir_golem_split_recurses_into_four_elixir_blobs() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=204, shuffle_decks=False)
    parent = _entity(state, card_id="elixir-golem", owner=1, x=9_000, y=15_000, hp=1)
    parent.hp = 0
    engine._resolve_deaths(state)
    mid_forms = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "elixir-golemite"
    ]
    assert len(mid_forms) == 2

    for entity in mid_forms:
        entity.hp = 0
    engine._resolve_deaths(state)
    blobs = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "elixir-blob"
    ]
    assert len(blobs) == 4
    assert sum(
        event.kind == "death_spawn" and event.get("child_card_id") == "elixir-blob"
        for event in state.events
    ) == 2
    engine.validate_state(state)


def test_lava_hound_death_releases_six_airborne_lava_pups() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=205, shuffle_decks=False)
    hound = _entity(state, card_id="lava-hound", owner=1, x=9_000, y=15_000, hp=1)
    hound.hp = 0
    engine._resolve_deaths(state)
    pups = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "lava-pup"
    ]
    assert len(pups) == 6
    assert all(entity.kind == "troop" for entity in pups)
    assert all(engine._movement_layer(entity) == "air" for entity in pups)
    engine.validate_state(state)


def test_goblin_giant_death_releases_two_single_spear_goblins() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=206, shuffle_decks=False)
    giant = _entity(state, card_id="goblin-giant", owner=1, x=9_000, y=15_000, hp=1)
    giant.hp = 0
    engine._resolve_deaths(state)
    children = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "spear-goblin"
    ]
    assert len(children) == 2
    assert all(entity.max_hp == 133 for entity in children)
    assert all(entity.owner == giant.owner for entity in children)
    engine.validate_state(state)


def test_goblin_giant_carrier_children_attack_while_attached_and_release_without_duplicates() -> None:
    engine = BattleEngine(ROSTER, validate_every_tick=True)
    opponent_deck = ("goblin-giant",) + tuple(
        card for card in ROSTER.interaction_set if card != "goblin-giant"
    )[:7]
    state = engine.new_battle(
        decks=(PLAYER_DECK, opponent_deck), seed=208, shuffle_decks=False
    )
    state.players[1].elixir_milli = ROSTER.match.max_elixir_milli
    cell = engine.legal_cells(state, 1, "goblin-giant")[0]
    engine.step(state, (PlayCardAction(1, 0, cell),))

    giant = next(entity for entity in state.entities.values() if entity.card_id == "goblin-giant")
    children = [
        entity
        for entity in state.entities.values()
        if entity.card_id == "spear-goblin" and entity.carried_by_uid == giant.uid
    ]
    assert len(children) == 2
    assert all(child.deploy_remaining_us == giant.deploy_remaining_us for child in children)

    # Attached children follow the carrier but do not enter the independent
    # collision/navigation solver.  Their positions remain deterministic.
    giant.x_mtile += 250
    giant.y_mtile += 125
    engine._sync_carried_entities(state)
    assert all(
        (child.x_mtile, child.y_mtile)
        == (
            giant.x_mtile + child.carried_offset_x_mtile,
            giant.y_mtile + child.carried_offset_y_mtile,
        )
        for child in children
    )

    # A carried Spear Goblin still owns its normal ranged attack channel.
    target = next(
        tower for tower in state.entities.values() if tower.kind == "tower" and tower.owner == 0 and tower.role == "left"
    )
    child = children[0]
    child.x_mtile = max(0, target.x_mtile - 4_000)
    child.y_mtile = target.y_mtile
    child.target_uid = target.uid
    child.pending_target_uid = target.uid
    engine._resolve_attack(state, child)
    assert any(
        event.kind == "projectile_spawned"
        and event.get("source_uid") == child.uid
        and event.get("card_id") == "spear-goblin"
        for event in state.events
    )

    giant.hp = 0
    engine._resolve_deaths(state)
    released = [
        entity
        for entity in state.entities.values()
        if entity.card_id == "spear-goblin" and entity.uid in {row.uid for row in children}
    ]
    assert len(released) == 2
    assert all(entity.carried_by_uid is None for entity in released)
    assert sum(
        event.kind == "carrier_child_released" and event.get("parent_uid") == giant.uid
        for event in state.events
    ) == 2
    assert not any(
        event.kind == "death_spawn"
        and event.get("parent_card_id") == "goblin-giant"
        for event in state.events
    )
    engine.validate_state(state)
    restored = battle_state_from_primitive(state.to_primitive(include_events=True))
    engine.validate_state(restored)
    assert restored.canonical_json(include_events=True) == state.canonical_json(include_events=True)


def test_battle_ram_is_consumed_on_building_impact_and_releases_two_barbarians() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=207, shuffle_decks=False)
    # Place the Ram at a valid building-contact edge rather than overlapping
    # the structure; released children are then required to remain clear of
    # the Cannon under strict per-tick validation.
    ram = _entity(state, card_id="battle-ram", owner=0, x=9_000, y=14_000)
    target = _entity(state, card_id="cannon", owner=1, x=9_000, y=15_000)
    ram.target_uid = target.uid
    ram.pending_target_uid = target.uid
    ram.attack_cooldown_us = 0

    engine._resolve_attack(state, ram)
    assert ram.hp == 0
    engine._resolve_deaths(state)

    children = [
        entity
        for entity in state.entities.values()
        if entity.alive and entity.card_id == "barbarian"
    ]
    assert len(children) == 2
    assert all(entity.owner == ram.owner for entity in children)
    assert any(
        event.kind == "death_spawn"
        and event.get("parent_card_id") == "battle-ram"
        and event.get("child_card_id") == "barbarian"
        and event.get("child_count") == 2
        for event in state.events
    )
    engine.validate_state(state)


def test_balloon_death_bomb_damages_nearby_air_ground_and_building_targets() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=101, shuffle_decks=False)
    balloon = _entity(state, card_id="balloon", owner=0, x=9_000, y=15_000, hp=1)
    ground = _entity(state, card_id="giant", owner=1, x=10_000, y=15_000, hp=1_000)
    air = _entity(state, card_id="bats", owner=1, x=9_000, y=16_000, hp=300)
    building = _entity(state, card_id="cannon", owner=1, x=9_000, y=14_000, hp=1_000)

    balloon.hp = 0
    engine._resolve_deaths(state)

    # Current Balloon death damage is delayed by three seconds.
    assert ground.hp == 1_000
    assert air.hp == 300
    assert building.hp == 1_000
    for _ in range(60):
        engine._advance_area_effects(state)
    assert ground.hp == 760
    assert air.hp == 60
    assert building.hp == 760
    assert any(
        event.kind == "damage_applied" and event.get("source_card_id") == "balloon"
        for event in state.events
    )
    engine.validate_state(state)


def test_wall_breakers_resolve_splash_at_building_contact_without_projectile_flight() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=102, shuffle_decks=False)
    breakers = _entity(state, card_id="wall-breakers", owner=0, x=9_000, y=15_000)
    cannon = _entity(state, card_id="cannon", owner=1, x=9_500, y=15_000, hp=1_000)
    nearby = _entity(state, card_id="skeletons", owner=1, x=9_500, y=16_000, hp=81)
    breakers.pending_target_uid = cannon.uid

    engine._resolve_attack(state, breakers)
    engine._resolve_deaths(state)

    assert cannon.hp == 719
    assert nearby.hp == 0
    assert breakers.alive is False
    assert not state.projectiles
    assert any(
        event.kind == "damage_applied" and event.get("source_card_id") == "wall-breakers"
        for event in state.events
    )
    engine.validate_state(state)


def test_sparky_has_a_four_second_charge_and_recharges_without_an_extra_cooldown() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=103, shuffle_decks=False)
    sparky = _entity(state, card_id="sparky", owner=0, x=9_000, y=15_000)
    target = _entity(state, card_id="giant", owner=1, x=10_000, y=15_000, hp=3_617)
    sparky.target_uid = target.uid

    engine._advance_attacks(state)
    assert sparky.windup_remaining_us == 4_000_000
    assert sparky.attack_cooldown_us == 4_000_000
    assert sparky.attack_count == 0

    for _ in range(80):
        engine._advance_attacks(state)
    assert sparky.attack_count == 1
    assert sparky.windup_remaining_us == 0
    assert sparky.attack_cooldown_us == 0
    assert len(state.projectiles) == 1

    # A hard crowd-control effect interrupts the next charge rather than
    # allowing the pending shot to fire after the status expires.
    sparky.windup_remaining_us = 2_000_000
    sparky.attack_cooldown_us = 2_000_000
    sparky.pending_target_uid = target.uid
    engine._apply_status(
        state,
        sparky,
        {"kind": "stun", "duration_us": 500_000, "speed_multiplier_milli": 0},
    )
    assert sparky.windup_remaining_us == 0
    assert sparky.attack_cooldown_us == 0
    assert sparky.pending_target_uid is None
    engine.validate_state(state)


def test_goblin_machine_runs_melee_and_blind_range_rocket_as_independent_weapons() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=104, shuffle_decks=False)
    machine = _entity(state, card_id="goblin-machine", owner=0, x=9_000, y=15_000)
    melee_target = _entity(state, card_id="giant", owner=1, x=10_000, y=15_000)
    rocket_target = _entity(state, card_id="baby-dragon", owner=1, x=12_500, y=15_000)
    machine.target_uid = melee_target.uid

    engine._advance_attacks(state)
    secondary = [
        event
        for event in state.events
        if event.kind == "secondary_attack_started"
        and event.get("uid") == machine.uid
    ]
    assert len(secondary) == 1
    assert secondary[0].get("target_uid") == rocket_target.uid
    assert melee_target.hp == 3_736
    assert len(state.projectiles) == 1
    projectile = next(iter(state.projectiles.values()))
    assert projectile.allowed_targets == ("air", "ground", "building", "crown_tower")
    assert projectile.speed_mtile_per_s == 350_000

    engine._advance_projectiles(state)
    assert rocket_target.hp == 761
    engine._resolve_deaths(state)
    assert any(
        event.kind == "damage_applied"
        and event.get("source_card_id") == "goblin-machine"
        and event.get("target_uid") == rocket_target.uid
        and event.get("damage") == 391
        for event in state.events
    )
    restored = battle_state_from_primitive(state.to_primitive())
    engine.validate_state(restored)
    assert restored.state_hash() == state.state_hash()

    # A target inside the 2.5-tile blind range is not eligible for the rocket,
    # even when it is the only air unit on the field.
    close_air = _entity(state, card_id="bats", owner=1, x=10_000, y=15_000)
    machine.secondary_attack_cooldown_us = 0
    engine._advance_secondary_attacks(state, engine.ruleset.tick_us)
    assert not any(
        event.kind == "secondary_attack_started"
        and event.get("target_uid") == close_air.uid
        for event in state.events
    )
    engine.validate_state(state)


def test_void_uses_official_target_count_tiers_and_stops_after_three_pulses() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=3, shuffle_decks=False)
    targets = [
        _entity(state, card_id="giant", owner=1, x=9_000 + i * 300, y=15_000, hp=4_000)
        for i in range(5)
    ]
    raw_effect = ROSTER.card("void").mechanics["persistent_effect"]
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="void",
        x_mtile=9_000,
        y_mtile=15_000,
        default_radius=2_500,
        default_damage=696,
        default_crown_damage=97,
        default_status=None,
        default_knockback=0,
        raw_effect=raw_effect,
    )
    # Five targets select the official 5+ tier for the immediate pulse.
    assert all(target.hp == 3_847 for target in targets)
    effect = next(iter(state.effects.values()))
    assert effect.pulses_applied == 1

    engine._apply_area_effect_tick(state, effect)
    engine._apply_area_effect_tick(state, effect)
    before_fourth = [target.hp for target in targets]
    engine._apply_area_effect_tick(state, effect)
    assert effect.pulses_applied == 3
    assert effect.alive is True
    assert [target.hp for target in targets] == before_fourth
    assert all(target.hp == 3_541 for target in targets)


def test_suspicious_bush_is_untargetable_and_releases_two_bush_goblins_on_contact() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=4, shuffle_decks=False)
    bush = _entity(state, card_id="suspicious-bush", owner=0, x=9_000, y=17_000, hp=81)
    tower = next(
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == 1 and entity.role == "king"
    )
    bush.x_mtile = tower.x_mtile
    bush.y_mtile = tower.y_mtile - 1_000
    bush.target_uid = tower.uid

    assert engine._targetable_for_acquisition(bush) is False
    engine._move_entities(state)
    assert bush.hp == 0
    assert tower.hp == tower.max_hp
    assert not any(
        event.kind == "damage_applied"
        and event.get("source_card_id") == "suspicious-bush"
        and event.get("target_uid") == tower.uid
        for event in state.events
    )
    engine._resolve_deaths(state)

    children = [entity for entity in state.entities.values() if entity.card_id == "bush-goblin"]
    assert len(children) == 2
    assert {child.hp for child in children} == {337}


def test_plain_persistent_status_has_no_death_transform_owner_and_is_cleared_on_death() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=405, shuffle_decks=False)
    target = _entity(state, card_id="hog-rider", owner=1, x=3_500, y=16_500, hp=1_000)
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="poison",
        x_mtile=target.x_mtile,
        y_mtile=target.y_mtile,
        default_radius=3_500,
        default_damage=0,
        default_crown_damage=0,
        default_status=None,
        default_knockback=0,
        raw_effect={
            **dict(ROSTER.card("poison").mechanics["persistent_effect"]),
            "status": dict(ROSTER.card("poison").mechanics["status"]),
        },
    )
    assert target.statuses
    assert target.statuses[0].on_death_spawn_owner is None
    target.hp = 0
    engine._resolve_deaths(state)
    assert target.statuses == []
    engine.validate_state(state)


def test_goblin_demolisher_latches_charge_and_explodes_when_fuse_expires() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=5, shuffle_decks=False)
    demolisher = _entity(
        state,
        card_id="goblin-demolisher",
        owner=0,
        x=9_000,
        y=17_000,
        hp=650,
    )
    _entity(state, card_id="skeletons", owner=1, x=9_000, y=17_000, hp=500)
    engine._advance_statuses_and_lifetimes(state)
    assert demolisher.charge_active is True
    assert demolisher.charge_remaining_us == 9_950_000

    demolisher.charge_remaining_us = engine.ruleset.tick_us
    engine._advance_statuses_and_lifetimes(state)
    assert demolisher.hp == 0
    engine._resolve_deaths(state)
    assert any(
        event.kind == "damage_applied" and event.get("source_card_id") == "goblin-demolisher"
        for event in state.events
    )


def test_clone_copies_friendly_troops_as_one_hp_entities_but_not_buildings_enemies_or_clones() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=6, shuffle_decks=False)
    original = _entity(state, card_id="hog-rider", owner=0, x=9_000, y=15_000)
    _entity(state, card_id="cannon", owner=0, x=9_000, y=16_800)
    _entity(state, card_id="musketeer", owner=1, x=9_000, y=15_000)
    already_clone = _entity(state, card_id="ice-golem", owner=0, x=9_500, y=15_000)
    already_clone.is_clone = True

    engine._impact_clone(
        state,
        owner=0,
        source_uid=None,
        source_card_id="clone",
        x=9_000,
        y=15_000,
        radius=3_000,
        raw_clone=ROSTER.card("clone").mechanics["clone"],
    )

    cloned_uids = [event.get("uid") for event in state.events if event.kind == "entity_cloned"]
    clones = [state.entities[uid] for uid in cloned_uids]
    assert len(clones) == 1
    assert {entity.card_id for entity in clones} == {"hog-rider"}
    assert all(entity.is_clone and entity.hp == 1 and entity.max_hp == 1 for entity in clones)
    assert all(entity.card_id != "cannon" for entity in clones)
    assert all(entity.owner == 0 for entity in clones)
    impact = next(event for event in state.events if event.kind == "clone_impact")
    assert impact.get("cloned_count") == 1

    # A canonical round-trip must preserve clone provenance and the resulting
    # deterministic entity stream for replay/fidelity tooling.
    restored = battle_state_from_primitive(state.to_primitive())
    engine.validate_state(restored)
    assert restored.state_hash() == state.state_hash()


def test_lightning_selects_three_highest_current_hp_targets_and_resets_them() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=7, shuffle_decks=False)
    targets = [
        _entity(state, card_id="giant", owner=1, x=9_000 + index * 300, y=15_000, hp=hp)
        for index, hp in enumerate((4_000, 3_000, 2_000, 1_500))
    ]
    for target in targets:
        target.attack_cooldown_us = 900_000
        target.windup_remaining_us = 200_000
        target.pending_target_uid = target.uid
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=None,
        source_card_id="lightning",
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        target_x_mtile=9_000,
        target_y_mtile=15_000,
        damage=1_057,
        crown_damage=265,
        speed_mtile_per_s=1,
        radius_mtile=3_500,
        status_kind="stun",
        status_duration_us=500_000,
        status_magnitude_permille=0,
    )
    state.next_uid += 1
    state.projectiles[projectile.uid] = projectile
    engine._impact_projectile(state, projectile)

    assert [target.hp for target in targets] == [2_943, 1_943, 943, 1_500]
    assert all(target.attack_cooldown_us == 0 for target in targets[:3])
    assert all(target.windup_remaining_us == 0 for target in targets[:3])
    assert all(any(status.kind == "stun" for status in target.statuses) for target in targets[:3])
    assert targets[3].attack_cooldown_us == 900_000
    assert not targets[3].statuses
    engine.validate_state(state)


def test_electro_dragon_chains_to_two_nearest_additional_targets_without_splashing_everything() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=8, shuffle_decks=False)
    first = _entity(state, card_id="giant", owner=1, x=9_000, y=15_000, hp=3_968)
    second = _entity(state, card_id="giant", owner=1, x=11_000, y=15_000, hp=3_968)
    third = _entity(state, card_id="giant", owner=1, x=13_000, y=15_000, hp=3_968)
    outside = _entity(state, card_id="giant", owner=1, x=17_500, y=15_000, hp=3_968)
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=None,
        source_card_id="electro-dragon",
        owner=0,
        x_mtile=9_000,
        y_mtile=15_000,
        target_x_mtile=first.x_mtile,
        target_y_mtile=first.y_mtile,
        target_uid=first.uid,
        damage=ROSTER.card("electro-dragon").damage or 0,
        crown_damage=ROSTER.card("electro-dragon").damage or 0,
        speed_mtile_per_s=1,
        status_kind="stun",
        status_duration_us=500_000,
        status_magnitude_permille=0,
    )
    state.next_uid += 1
    engine._impact_projectile(state, projectile)

    hits = [
        event.get("target_uid")
        for event in state.events
        if event.kind == "chain_hit"
    ]
    assert hits == [first.uid, second.uid, third.uid]
    assert [target.hp for target in (first, second, third)] == [3_776, 3_776, 3_776]
    assert outside.hp == 3_968
    assert all(any(status.kind == "stun" for status in target.statuses) for target in (first, second, third))
    engine.validate_state(state)


def test_electro_wizard_hits_two_discrete_targets_not_everything_in_visual_radius() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=9, shuffle_decks=False)
    wizard = _entity(state, card_id="electro-wizard", owner=0, x=9_000, y=15_000)
    primary = _entity(state, card_id="giant", owner=1, x=9_500, y=15_000, hp=3_968)
    nearest = _entity(state, card_id="giant", owner=1, x=10_000, y=16_000, hp=3_968)
    third = _entity(state, card_id="giant", owner=1, x=12_000, y=15_000, hp=3_968)
    wizard.pending_target_uid = primary.uid

    engine._resolve_attack(state, wizard)

    assert primary.hp == 3_850
    assert nearest.hp == 3_850
    assert third.hp == 3_968
    assert len([event for event in state.events if event.kind == "multi_target_hit"]) == 2
    assert all(any(status.kind == "stun" for status in target.statuses) for target in (primary, nearest))
    engine.validate_state(state)


def test_electro_giant_reflects_damage_and_stuns_the_nearby_attacker_once() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=10, shuffle_decks=False)
    giant = _entity(state, card_id="electro-giant", owner=1, x=9_000, y=15_000)
    attacker = _entity(state, card_id="hog-rider", owner=0, x=10_500, y=15_000)
    attacker.attack_cooldown_us = 800_000
    attacker.windup_remaining_us = 200_000
    attacker.pending_target_uid = giant.uid
    before_giant = giant.hp
    before_attacker = attacker.hp

    engine._deal_damage(
        state,
        giant,
        100,
        source_uid=attacker.uid,
        source_card_id="hog-rider",
    )

    assert giant.hp == before_giant - 100
    assert attacker.hp == before_attacker - 192
    assert attacker.attack_cooldown_us == 0
    assert attacker.windup_remaining_us == 0
    assert attacker.pending_target_uid is None
    assert any(event.kind == "reflected_damage" for event in state.events)
    # Reflection tags are terminal for the reactive path, so two Electro
    # Giants do not recurse forever when one attacks the other.
    engine.validate_state(state)


def test_charge_attack_component_runs_prince_and_consumes_the_charged_hit() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=12, shuffle_decks=False)
    prince = _entity(state, card_id="prince", owner=0, x=4_000, y=15_000)
    target = _entity(state, card_id="giant", owner=1, x=14_000, y=15_000, hp=4_000)
    prince.target_uid = target.uid

    # The canonical 50 ms tick and medium 1,200 milli-tile/s speed require
    # roughly 2 seconds to cross Prince's provisional 2.5-tile charge distance.
    for _ in range(80):
        engine._move_entities(state)
        if prince.attack_charge_active:
            break
    assert prince.attack_charge_active is True
    assert prince.attack_charge_distance_mtile >= 2_500
    assert any(event.kind == "charge_started" for event in state.events)

    # Put the target in melee range while preserving the latched charge, then
    # resolve the hit directly.  The charge damage is distinct from Prince's
    # ordinary 391 body damage and the component resets after impact.
    target.x_mtile = prince.x_mtile + 500
    target.y_mtile = prince.y_mtile
    prince.pending_target_uid = target.uid
    before = target.hp
    engine._resolve_attack(state, prince)
    assert target.hp == before - 783
    assert prince.attack_charge_active is False
    assert prince.attack_charge_distance_mtile == 0
    assert any(event.kind == "charge_reset" and event.get("reason") == "hit_consumed" for event in state.events)
    engine.validate_state(state)


def test_all_v1_charge_cards_have_source_level11_damage_and_hard_cc_resets_run() -> None:
    engine = BattleEngine(ROSTER)
    expected = {
        "prince": (391, 783, 2_500),
        "dark-prince": (266, 532, 3_000),
        "battle-ram": (192, 573, 3_500),
        "ram-rider": (250, 501, 2_500),
    }
    for card_id, (normal_damage, charge_damage, charge_distance) in expected.items():
        card = ROSTER.card(card_id)
        component = card.mechanics["charge_attack"]
        assert card.damage == normal_damage
        assert component["charge_damage"] == charge_damage
        assert component["charge_distance_mtile"] == charge_distance
        assert component["charged_speed_mtile_per_s"] == 2_400

        state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=13, shuffle_decks=False)
        entity = _entity(state, card_id=card_id, owner=0, x=8_000, y=15_000)
        entity.attack_charge_active = True
        entity.attack_charge_distance_mtile = charge_distance
        engine._apply_status(
            state,
            entity,
            {
                "kind": "stun",
                "duration_us": 500_000,
                "speed_multiplier_milli": 0,
                "hit_speed_multiplier_milli": 0,
            },
        )
        assert entity.attack_charge_active is False
        assert entity.attack_charge_distance_mtile == 0
        engine.validate_state(state)


def test_bandit_dashes_into_range_and_uses_separate_dash_damage() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=14, shuffle_decks=False)
    bandit = _entity(state, card_id="bandit", owner=0, x=8_000, y=15_000)
    target = _entity(state, card_id="giant", owner=1, x=13_000, y=15_000, hp=4_000)
    bandit.target_uid = target.uid
    old_position = (bandit.x_mtile, bandit.y_mtile)

    engine._move_entities(state)

    assert bandit.dash_attack_active is True
    assert (bandit.x_mtile, bandit.y_mtile) != old_position
    assert any(event.kind == "dash_started" for event in state.events)
    bandit.pending_target_uid = target.uid
    engine._resolve_attack(state, bandit)
    assert target.hp == 4_000 - 388
    assert bandit.dash_attack_active is False
    assert any(event.kind == "dash_reset" and event.get("reason") == "hit_consumed" for event in state.events)
    engine.validate_state(state)


def test_fisherman_hook_pulls_a_ground_troop_before_his_impact() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=15, shuffle_decks=False)
    fisherman = _entity(state, card_id="fisherman", owner=0, x=8_000, y=15_000)
    target = _entity(state, card_id="giant", owner=1, x=13_000, y=15_000, hp=4_000)
    fisherman.pending_target_uid = target.uid
    before_position = (target.x_mtile, target.y_mtile)

    engine._resolve_attack(state, fisherman)

    assert (target.x_mtile, target.y_mtile) != before_position
    assert target.hp == 4_000 - 194
    assert any(event.kind == "hook_pulled" and event.get("target_uid") == target.uid for event in state.events)
    engine.validate_state(state)


def test_inferno_dragon_ramps_damage_and_resets_when_target_is_stunned() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=16, shuffle_decks=False)
    dragon = _entity(state, card_id="inferno-dragon", owner=0, x=8_000, y=15_000)
    target = _entity(state, card_id="giant", owner=1, x=10_500, y=15_000, hp=4_000)
    dragon.target_uid = target.uid
    component = ROSTER.card("inferno-dragon").mechanics["ramp_attack"]
    assert component["damage_schedule"] == (35, 120, 422)
    assert component["stage_thresholds_us"] == (0, 2_000_000, 4_000_000)

    for _ in range(39):
        engine._advance_attack_ramps(state, engine.ruleset.tick_us)
    assert dragon.ramp_elapsed_us == 1_950_000
    assert dragon.ramp_stage == 0
    engine._advance_attack_ramps(state, engine.ruleset.tick_us)
    assert dragon.ramp_elapsed_us == 2_000_000
    assert dragon.ramp_stage == 1
    dragon.pending_target_uid = target.uid
    before = target.hp
    engine._resolve_attack(state, dragon)
    assert target.hp == before - 120

    engine._apply_status(
        state,
        target,
        {
            "kind": "stun",
            "duration_us": 500_000,
            "speed_multiplier_milli": 0,
            "hit_speed_multiplier_milli": 0,
        },
    )
    assert dragon.ramp_elapsed_us == 0
    assert dragon.ramp_stage == 0
    assert any(
        event.kind == "ramp_reset" and event.get("reason") == "target_stun"
        for event in state.events
    )

    engine._apply_status(
        state,
        dragon,
        {
            "kind": "stun",
            "duration_us": 500_000,
            "speed_multiplier_milli": 0,
            "hit_speed_multiplier_milli": 0,
        },
    )
    assert dragon.ramp_elapsed_us == 0
    assert dragon.ramp_stage == 0
    dragon.ramp_elapsed_us = 1_250_000
    dragon.ramp_stage = 0
    restored = battle_state_from_primitive(state.to_primitive())
    assert restored.entities[dragon.uid].ramp_elapsed_us == 1_250_000
    assert restored.entities[dragon.uid].ramp_stage == 0
    assert restored.state_hash() == state.state_hash()
    engine.validate_state(state)


def test_inferno_tower_uses_its_longer_ramp_schedule_and_target_loss_resets_it() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=17, shuffle_decks=False)
    tower = _entity(state, card_id="inferno-tower", owner=0, x=8_000, y=15_000)
    target = _entity(state, card_id="giant", owner=1, x=10_500, y=15_000, hp=4_000)
    tower.target_uid = target.uid
    for _ in range(30):
        engine._advance_attack_ramps(state, engine.ruleset.tick_us)
    assert tower.ramp_elapsed_us == 1_500_000
    assert tower.ramp_stage == 1
    tower.pending_target_uid = target.uid
    before = target.hp
    engine._resolve_attack(state, tower)
    assert target.hp == before - 158

    tower.target_uid = None
    engine._advance_attack_ramps(state, engine.ruleset.tick_us)
    assert tower.ramp_elapsed_us == 0
    assert tower.ramp_stage == 0
    assert any(event.kind == "ramp_reset" and event.get("reason") == "target_lost" for event in state.events)
    engine.validate_state(state)


def test_phoenix_death_creates_targetable_egg_that_hatches_once_with_full_stats() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=18, shuffle_decks=False)
    phoenix = _entity(state, card_id="phoenix", owner=0, x=8_000, y=15_000, hp=1)
    nearby = _entity(state, card_id="giant", owner=1, x=9_000, y=15_000, hp=500)
    engine._deal_damage(state, phoenix, 1, source_uid=None, source_card_id="test")
    engine._resolve_deaths(state)

    egg = next(entity for entity in state.entities.values() if entity.card_id == "phoenix-egg" and entity.alive)
    assert egg.kind == "building"
    assert egg.hp == 317
    assert egg.lifetime_remaining_us == 3_800_000
    assert nearby.hp == 500 - 163
    assert any(event.kind == "phoenix_egg_created" for event in state.events)

    egg.lifetime_remaining_us = engine.ruleset.tick_us
    engine._advance_statuses_and_lifetimes(state)
    assert egg.hatch_due is True
    engine._resolve_deaths(state)
    reborn = next(
        entity
        for entity in state.entities.values()
        if entity.card_id == "phoenix" and entity.alive
    )
    assert reborn.hp == 1_052
    assert reborn.max_hp == 1_052
    assert reborn.revive_eligible is False
    assert any(event.kind == "phoenix_egg_hatched" for event in state.events)
    engine.validate_state(state)


def test_destroyed_phoenix_egg_does_not_hatch() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=19, shuffle_decks=False)
    phoenix = _entity(state, card_id="phoenix", owner=0, x=8_000, y=15_000, hp=1)
    engine._deal_damage(state, phoenix, 1, source_uid=None, source_card_id="test")
    engine._resolve_deaths(state)
    egg = next(entity for entity in state.entities.values() if entity.card_id == "phoenix-egg" and entity.alive)
    engine._deal_damage(state, egg, egg.hp, source_uid=None, source_card_id="test")
    engine._resolve_deaths(state)
    assert not any(entity.card_id == "phoenix" and entity.alive for entity in state.entities.values())
    assert not any(event.kind == "phoenix_egg_hatched" for event in state.events)
    engine.validate_state(state)


def test_firecracker_projectile_splashes_and_recoils_its_source() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle(decks=(PLAYER_DECK, PLAYER_DECK), seed=16, shuffle_decks=False)
    firecracker = _entity(state, card_id="firecracker", owner=0, x=8_000, y=15_000)
    primary = _entity(state, card_id="giant", owner=1, x=10_000, y=15_000, hp=4_000)
    nearby = _entity(state, card_id="giant", owner=1, x=10_500, y=15_000, hp=4_000)
    firecracker.pending_target_uid = primary.uid
    before_position = (firecracker.x_mtile, firecracker.y_mtile)

    engine._resolve_attack(state, firecracker)
    projectile = next(iter(state.projectiles.values()))
    engine._impact_projectile(state, projectile)

    assert primary.hp == 4_000 - 64
    assert nearby.hp == 4_000 - 64
    assert (firecracker.x_mtile, firecracker.y_mtile) != before_position
    assert any(event.kind == "recoil_applied" for event in state.events)
    engine.validate_state(state)
