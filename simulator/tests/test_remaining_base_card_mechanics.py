from __future__ import annotations

import pytest

from simulator.actions import PlayCardAction
from simulator.engine import BattleEngine
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState, ProjectileState


RULESET = load_ruleset("v1")


def _state(
    deck: tuple[str, ...] = PLAYER_DECK,
) -> tuple[BattleEngine, object]:
    engine = BattleEngine(RULESET)
    state = engine.new_battle((deck, PLAYER_DECK), seed=947, shuffle_decks=False)
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
        deploy_remaining_us=0,
    )
    state.next_uid += 1
    state.entities[entity.uid] = entity
    return entity


def _instant_projectile(
    state,
    card_id: str,
    owner: int,
    x: int,
    y: int,
) -> ProjectileState:
    card = RULESET.card(card_id)
    status = card.mechanics.get("status")
    projectile = ProjectileState(
        uid=state.next_uid,
        source_uid=None,
        source_card_id=card_id,
        owner=owner,
        x_mtile=x,
        y_mtile=y,
        target_x_mtile=x,
        target_y_mtile=y,
        damage=int(card.damage or 0),
        crown_damage=int(card.crown_tower_damage or card.damage or 0),
        speed_mtile_per_s=0,
        radius_mtile=int(card.area_radius_mtile or 0),
        status_kind=None if not status else str(status.get("kind")),
        status_duration_us=0 if not status else int(status.get("duration_us") or 0),
        status_magnitude_permille=(
            1_000 if not status else int(status.get("speed_multiplier_milli") or 1_000)
        ),
        knockback_mtile=int(card.mechanics.get("knockback_mtile") or 0),
    )
    state.next_uid += 1
    return projectile


def test_mirror_is_excluded_from_opening_hand_and_charges_previous_cost_plus_one() -> None:
    deck = (
        "mirror",
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
    )
    engine, state = _state(deck)
    player = state.players[0]
    assert "mirror" not in player.hand

    player.hand = ["mirror", "cannon", "musketeer", "skeletons"]
    player.draw_pile = ["ice-golem", "ice-spirit", "fireball", "hog-rider"]
    player.last_played_card_id = "hog-rider"
    player.elixir_milli = 10_000
    before_uids = set(state.entities)
    action = PlayCardAction(player=0, card_slot=0, cell=(8, 20))
    assert engine.validate_action(state, action) is None
    engine._play_card(state, action, "mirror")
    assert player.elixir_milli == 5_000
    mirrored_hogs = [
        entity for uid, entity in state.entities.items()
        if uid not in before_uids and entity.card_id == "hog-rider"
    ]
    assert len(mirrored_hogs) == 1
    assert mirrored_hogs[0].max_hp > int(RULESET.card("hog-rider").hitpoints or 0)
    assert mirrored_hogs[0].level_multiplier_permille == 1_100
    assert any(
        event.kind == "card_mirrored"
        and event.get("source_card_id") == "hog-rider"
        for event in state.events
    )


def test_elixir_collector_death_grants_final_elixir_and_uses_current_cadence() -> None:
    engine, state = _state()
    collector_definition = RULESET.card("elixir-collector")
    generation = collector_definition.mechanics["elixir_generation"]
    assert generation["interval_us"] == 13_000_000
    assert collector_definition.lifetime_us == 93_000_000

    collector = _entity(state, "elixir-collector", 0, 9_000, 20_000, hp=1)
    state.players[0].elixir_milli = 0
    collector.hp = 0
    engine._resolve_deaths(state)
    assert state.players[0].elixir_milli == 1_000


def test_ram_rider_bola_is_an_independent_air_targeting_snare() -> None:
    engine, state = _state()
    ram = _entity(state, "ram-rider", 0, 9_000, 15_000)
    air_target = _entity(state, "baby-dragon", 1, 9_000, 19_000)
    primary_before = ram.attack_count

    for _ in range(80):
        engine._advance_secondary_attacks(state, RULESET.tick_us)
        engine._advance_projectiles(state)
        if air_target.hp < air_target.max_hp:
            break

    assert air_target.hp < air_target.max_hp
    assert any(status.kind == "slow" for status in air_target.statuses)
    assert ram.secondary_attack_count == 1
    assert ram.attack_count == primary_before


def test_three_musketeers_add_bayonet_damage_at_close_range() -> None:
    engine, state = _state()
    musketeer = _entity(state, "three-musketeers", 0, 9_000, 15_000)
    target = _entity(state, "giant", 1, 9_000, 15_700)
    musketeer.pending_target_uid = target.uid
    ordinary_shot = int(RULESET.card("three-musketeers").damage or 0)
    engine._resolve_attack(state, musketeer)
    for _ in range(40):
        engine._advance_projectiles(state)
        if not any(projectile.alive for projectile in state.projectiles.values()):
            break
    assert target.max_hp - target.hp > ordinary_shot


def test_goblin_drill_emergence_and_death_payloads_are_distinct() -> None:
    engine, state = _state()
    victim = _entity(state, "giant", 1, 9_000, 19_500)
    drill = engine._spawn_single_at(
        state,
        RULESET.card("goblin-drill"),
        owner=0,
        x_mtile=9_000,
        y_mtile=18_000,
    )
    initial_hp = victim.hp
    while drill.deploy_remaining_us > 0:
        engine._advance_deployments(state)
    assert victim.hp < initial_hp
    assert (victim.x_mtile, victim.y_mtile) != (9_000, 19_500)

    before = sum(
        entity.alive and entity.card_id == "goblin"
        for entity in state.entities.values()
    )
    drill.hp = 0
    engine._resolve_deaths(state)
    after = sum(
        entity.alive and entity.card_id == "goblin"
        for entity in state.entities.values()
    )
    assert after - before == 2


@pytest.mark.parametrize(
    "card_id",
    ("fire-spirit", "ice-spirit", "heal-spirit", "electro-spirit"),
)
def test_spirits_do_not_unassisted_connect_to_towers(card_id: str) -> None:
    engine, state = _state()
    spirit = _entity(state, card_id, 0, 14_500, 8_000)
    # August 2026 makes the no-unassisted-connection rule explicit.  The
    # 215-HP value remains independently pinned, but it must not be used as a
    # substitute for the Crown-Tower target filter.
    assert spirit.max_hp == 215
    assert engine._choose_target(state, spirit) is None


def test_mortar_respects_its_minimum_range_blind_spot() -> None:
    engine, state = _state()
    mortar = _entity(state, "mortar", 0, 9_000, 15_000)
    too_close = _entity(state, "giant", 1, 9_000, 17_000)
    valid = _entity(state, "giant", 1, 9_000, 23_000)
    assert not engine._in_attack_range(mortar, too_close)
    assert engine._in_attack_range(mortar, valid)


def test_giant_snowball_pushes_targets_and_barbarian_barrel_remains_ground_only() -> None:
    engine, state = _state()
    ground = _entity(state, "giant", 1, 9_000, 15_000)
    snowball = _instant_projectile(state, "giant-snowball", 0, 9_000, 15_000)
    engine._impact_projectile(state, snowball)
    assert ground.hp < ground.max_hp
    assert (ground.x_mtile, ground.y_mtile) != (9_000, 15_000)

    ground = _entity(state, "giant", 1, 12_000, 15_000)
    air = _entity(state, "baby-dragon", 1, 12_000, 15_000)
    barrel = _instant_projectile(state, "barbarian-barrel", 0, 12_000, 15_000)
    engine._impact_projectile(state, barrel)
    assert ground.hp < ground.max_hp
    assert air.hp == air.max_hp


@pytest.mark.parametrize("card_id", ("balloon", "giant-skeleton"))
def test_large_death_bombs_resolve_after_three_seconds(card_id: str) -> None:
    engine, state = _state()
    body = _entity(state, card_id, 0, 9_000, 15_000, hp=1)
    victim = _entity(state, "giant", 1, 9_000, 16_000)
    body.hp = 0
    engine._resolve_deaths(state)
    assert victim.hp == victim.max_hp
    for _ in range(59):
        engine._advance_area_effects(state)
    assert victim.hp == victim.max_hp
    engine._advance_area_effects(state)
    assert victim.hp < victim.max_hp


def test_giant_skeleton_uses_the_current_level_eleven_body_footprint() -> None:
    card = RULESET.card("giant-skeleton")
    assert card.hitpoints == 1_313
    assert card.collision_radius_mtile == 750
    death = card.mechanics["death"]
    assert death["damage"] == 269
    assert death["crown_tower_damage"] == 269
    assert death["radius_mtile"] == 3_000
    assert death["delay_us"] == 3_000_000


def test_graveyard_uses_a_deterministic_noncentral_spawn_pattern() -> None:
    def positions() -> list[tuple[int, int]]:
        engine, state = _state()
        card = RULESET.card("graveyard")
        engine._create_area_effect(
            state,
            owner=0,
            source_uid=None,
            source_card_id="graveyard",
            x_mtile=9_000,
            y_mtile=15_000,
            default_radius=int(card.area_radius_mtile or 0),
            default_damage=0,
            default_crown_damage=0,
            default_status=None,
            default_knockback=0,
            raw_effect=card.mechanics["persistent_effect"],
        )
        # Current Graveyard keeps the fixed eight-location pattern introduced
        # in February 2026, but the first Skeleton does not rise until the
        # published 2.2-second deployment delay.  Subsequent pulses are 0.5 s
        # apart and cycle those eight locations until the current twelve-body
        # cap is reached.
        for _ in range(155):
            engine._advance_area_effects(state)
        return [
            (entity.x_mtile, entity.y_mtile)
            for entity in state.entities.values()
            if entity.card_id == "skeletons" and entity.owner == 0
        ]

    first = positions()
    second = positions()
    assert first == second
    assert len(first) == 12
    assert len(set(first)) == 8
    assert (9_000, 15_000) not in first


def test_minion_deployment_is_staggered_and_bush_releases_without_parent_damage() -> None:
    engine, state = _state()
    before_uids = set(state.entities)
    engine._spawn_card_entities(state, 0, RULESET.card("minions"), (8, 20))
    minions = [
        entity for uid, entity in state.entities.items()
        if uid not in before_uids and entity.card_id == "minions"
    ]
    assert len(minions) == 3
    delays = sorted(entity.deploy_remaining_us for entity in minions)
    assert delays[1] - delays[0] == 100_000
    assert delays[2] - delays[1] == 100_000

    victim = _entity(state, "cannon", 1, 9_000, 15_000)
    bush = _entity(state, "suspicious-bush", 0, 9_000, 15_000, hp=1)
    before_hp = victim.hp
    bush.hp = 0
    engine._resolve_deaths(state)
    children = [
        entity for entity in state.entities.values()
        if entity.alive and entity.card_id == "bush-goblin"
    ]
    assert victim.hp == before_hp
    assert len(children) == 2
    assert sorted(
        (child.x_mtile - bush.x_mtile, child.y_mtile - bush.y_mtile)
        for child in children
    ) == [(-1_600, 0), (1_600, 0)]
