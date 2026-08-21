from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_ruleset
from simulator.state import EntityState, battle_state_from_primitive


ROSTER = load_ruleset("2026-08-04-roster")


def _unit(state, uid: int, card_id: str, owner: int, x: int, y: int, hp: int = 1_000):
    state.entities[uid] = EntityState(
        uid=uid,
        card_id=card_id,
        owner=owner,
        kind="troop",
        x_mtile=x,
        y_mtile=y,
        hp=hp,
        max_hp=hp,
        spawn_tick=state.tick,
    )


def test_persistent_area_pulses_newly_entering_units_and_round_trips() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=9, shuffle_decks=False)
    _unit(state, state.next_uid, "hog-rider", 1, 3_500, 16_500, hp=500)
    state.next_uid += 1
    _unit(state, state.next_uid, "hog-rider", 1, 14_500, 16_500, hp=500)
    state.next_uid += 1

    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="poison",
        x_mtile=3_500,
        y_mtile=16_500,
        default_radius=3_500,
        default_damage=1,
        default_crown_damage=1,
        default_status=None,
        default_knockback=0,
        raw_effect={
            "duration_us": 1_000_000,
            "tick_interval_us": 50_000,
            "radius_mtile": 3_500,
            "damage_per_tick": 10,
            "crown_damage_per_tick": 10,
            "targets": ["ground"],
        },
    )
    first = state.entities[7]
    second = state.entities[8]
    assert first.hp == 490
    assert second.hp == 500

    # Move the previously out-of-area unit into the effect before the next
    # pulse.  A persistent zone must not freeze its victim set at impact time.
    second.x_mtile, second.y_mtile = 3_500, 16_500
    engine.step(state)
    assert second.hp == 490
    assert any(event.kind == "area_effect_pulse" for event in state.events)

    restored = battle_state_from_primitive(state.to_primitive())
    assert restored.state_hash() == state.state_hash()
    engine.validate_state(restored)

    for _ in range(30):
        engine.step(state)
    assert not next(iter(state.effects.values())).alive
    assert any(event.kind == "area_effect_expired" for event in state.events)


def test_persistent_spawn_effect_is_temporal_and_capped() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=10, shuffle_decks=False)
    engine._create_area_effect(
        state,
        owner=1,
        source_uid=None,
        source_card_id="graveyard",
        x_mtile=9_000,
        y_mtile=15_000,
        default_radius=4_000,
        default_damage=0,
        default_crown_damage=0,
        default_status=None,
        default_knockback=0,
        raw_effect={
            "duration_us": 500_000,
            "tick_interval_us": 100_000,
            "radius_mtile": 4_000,
            "damage_per_tick": 0,
            "crown_damage_per_tick": 0,
            "targets": ["ground"],
            "spawn": {"card_id": "skeletons", "count": 1, "max_spawns": 3},
        },
    )
    assert len([row for row in state.entities.values() if row.card_id == "skeletons"]) == 1
    for _ in range(20):
        engine.step(state)
    assert len([row for row in state.entities.values() if row.card_id == "skeletons"]) == 3
    effect = next(iter(state.effects.values()))
    assert effect.spawned_count == 3
    assert not effect.alive
    engine.validate_state(state)


def test_rage_deals_one_impact_pulse_and_refreshes_a_friendly_aura() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=12, shuffle_decks=False)
    _unit(state, state.next_uid, "hog-rider", 0, 9_000, 15_000, hp=500)
    state.next_uid += 1
    building_uid = state.next_uid
    state.next_uid += 1
    building = ROSTER.card("cannon")
    state.entities[building_uid] = EntityState(
        uid=building_uid,
        card_id="cannon",
        owner=0,
        kind="building",
        x_mtile=10_500,
        y_mtile=15_000,
        hp=int(building.hitpoints or 1),
        max_hp=int(building.hitpoints or 1),
        spawn_tick=state.tick,
    )
    _unit(state, state.next_uid, "giant", 1, 12_000, 15_000, hp=1_000)
    state.next_uid += 1

    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="rage",
        x_mtile=9_000,
        y_mtile=15_000,
        default_radius=3_000,
        default_damage=179,
        default_crown_damage=45,
        default_status=None,
        default_knockback=0,
        raw_effect=ROSTER.card("rage").mechanics["persistent_effect"],
    )
    hog = state.entities[7]
    giant = state.entities[9]
    assert giant.hp == 821
    assert any(status.kind == "rage" for status in hog.statuses)
    assert engine._speed_multiplier(hog) == 1_300

    # Later pulses refresh the friendly aura but do not re-apply Rage damage.
    for _ in range(10):
        engine._advance_area_effects(state)
    assert giant.hp == 821
    hog.x_mtile = 15_000
    for _ in range(20):
        engine._advance_area_effects(state)
        engine._advance_statuses_and_lifetimes(state)
    assert not any(status.kind == "rage" for status in hog.statuses)
    engine.validate_state(state)


def test_tornado_uses_two_damage_pulses_then_pull_only_tail() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=11, shuffle_decks=False)
    target_uid = state.next_uid
    _unit(state, target_uid, "hog-rider", 1, 11_000, 15_000, hp=500)
    state.next_uid += 1
    target = state.entities[target_uid]
    raw_effect = ROSTER.card("tornado").mechanics["persistent_effect"]

    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="tornado",
        x_mtile=9_000,
        y_mtile=15_000,
        default_radius=5_500,
        default_damage=84,
        default_crown_damage=27,
        default_status=None,
        default_knockback=0,
        raw_effect=raw_effect,
    )
    effect = next(iter(state.effects.values()))
    assert effect.damage_schedule == (42, 42)
    assert effect.crown_damage_schedule == (14, 13)
    assert target.hp == 458
    assert target.x_mtile == 10_000

    # The reference clock is 50 ms.  At 0.5 s the second damage pulse and
    # another deterministic pull occur; the 1.0 s pulse keeps the pull alive
    # but has no scheduled damage.
    for _ in range(10):
        engine._advance_area_effects(state)
    assert target.hp == 416
    assert target.x_mtile == 9_000
    for _ in range(10):
        engine._advance_area_effects(state)
    assert target.hp == 416
    assert target.x_mtile == 9_000
    assert effect.pulses_applied == 3
    assert effect.alive is True

    for _ in range(2):
        engine._advance_area_effects(state)
    assert effect.alive is False
    pulses = [event for event in state.events if event.kind == "area_effect_pulse"]
    assert [event.get("damage") for event in pulses] == [42, 42, 0]
    assert [event.get("pulse_index") for event in pulses] == [0, 1, 2]
    engine.validate_state(state)


def test_goblin_curse_slows_and_transforms_a_lethal_troop() -> None:
    engine = BattleEngine(ROSTER)
    state = engine.new_battle((PLAYER_DECK, PLAYER_DECK), seed=13, shuffle_decks=False)
    target_uid = state.next_uid
    _unit(state, target_uid, "hog-rider", 1, 9_000, 15_000, hp=35)
    state.next_uid += 1

    raw_effect = ROSTER.card("goblin-curse").mechanics["persistent_effect"]
    engine._create_area_effect(
        state,
        owner=0,
        source_uid=None,
        source_card_id="goblin-curse",
        x_mtile=9_000,
        y_mtile=15_000,
        default_radius=3_000,
        default_damage=35,
        default_crown_damage=7,
        default_status=None,
        default_knockback=0,
        raw_effect=raw_effect,
    )
    target = state.entities[target_uid]
    assert target.hp == 0
    assert target.statuses[0].kind == "slow"
    assert target.statuses[0].magnitude_permille == 850
    assert engine._speed_multiplier(target) == 850
    assert engine._hit_speed_multiplier(target) == 1_000
    effect = next(iter(state.effects.values()))
    for _ in range(100):  # five seconds after the immediate pulse
        engine._advance_area_effects(state)
    assert effect.pulses_applied == 6
    assert effect.alive is False

    # Death resolution happens at the deterministic end-of-tick boundary;
    # the spawned child is owned by the caster, not the cursed victim.
    engine.step(state)
    transformed = [
        entity
        for entity in state.entities.values()
        if entity.card_id == "goblin" and entity.alive
    ]
    assert len(transformed) == 1
    assert transformed[0].owner == 0
    assert transformed[0].hp == 202
    assert any(event.kind == "death_transform" for event in state.events)
    restored = battle_state_from_primitive(state.to_primitive())
    assert restored.state_hash() == state.state_hash()
    engine.validate_state(state)
