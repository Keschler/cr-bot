from __future__ import annotations

import json

from simulator.actions import PlayCardAction
from simulator.catalog import ROSTER_RULESET_ID, build_roster_ruleset_raw
from simulator.engine import BattleEngine
from simulator.roster import PLAYER_DECK, load_opponent_roster
from simulator.ruleset import calculate_content_hash, load_ruleset, ruleset_path


def test_roster_ruleset_is_reproducible_and_contains_every_eligible_card() -> None:
    path = ruleset_path(ROSTER_RULESET_ID)
    raw = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = build_roster_ruleset_raw()
    assert calculate_content_hash(raw) == raw["content_hash"]
    assert rebuilt == raw

    ruleset = load_ruleset(ROSTER_RULESET_ID)
    roster = load_opponent_roster()
    assert set(roster.eligible_cards) == set(ruleset.interaction_set)
    assert set(PLAYER_DECK) <= set(ruleset.cards)
    assert ruleset.metadata["training_ready"] is False


def test_every_eligible_card_can_be_deployed_and_advanced_under_strict_validation() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    engine = BattleEngine(ruleset, validate_every_tick=True)
    roster = load_opponent_roster()

    for card_id in roster.eligible_cards:
        opponent_deck = (card_id,) + tuple(
            card for card in roster.eligible_cards if card != card_id
        )[:7]
        state = engine.new_battle(
            decks=(PLAYER_DECK, opponent_deck),
            seed=7,
            shuffle_decks=False,
        )
        state.players[1].elixir_milli = ruleset.match.max_elixir_milli
        legal = engine.legal_cells(state, 1, card_id)
        assert legal, card_id
        action = PlayCardAction(1, 0, legal[0])
        assert engine.validate_action(state, action) is None
        engine.step(state, (action,))
        for _ in range(120):
            engine.step(state)


def test_roster_spawner_and_spell_spawn_components_emit_deterministic_events() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    engine = BattleEngine(ruleset, validate_every_tick=True)
    for card_id in ("tombstone", "goblin-barrel", "elixir-collector"):
        opponent_deck = (card_id,) + tuple(
            card for card in ruleset.interaction_set if card != card_id
        )[:7]
        state = engine.new_battle(
            decks=(PLAYER_DECK, opponent_deck), seed=11, shuffle_decks=False
        )
        state.players[1].elixir_milli = ruleset.match.max_elixir_milli
        cell = engine.legal_cells(state, 1, card_id)[0]
        engine.step(state, (PlayCardAction(1, 0, cell),))
        # Collector generation is pinned to the current 12-second Level-11
        # cycle (older fixtures used an 8-second placeholder).
        for _ in range(260):
            engine.step(state)
        kinds = {event.kind for event in state.events}
        if card_id == "elixir-collector":
            assert "elixir_generated" in kinds
        else:
            assert "entity_spawned" in kinds


def test_goblin_cage_uses_one_hidden_goblin_brawler_child() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    cage = ruleset.card("goblin-cage")
    brawler = ruleset.card("goblin-brawler")

    assert cage.mechanics["death"]["spawn_card_id"] == "goblin-brawler"
    assert cage.mechanics["death"]["spawn_count"] == 1
    assert cage.damage is None
    assert cage.attack_interval_us is None
    assert cage.range_mtile is None
    assert cage.projectile is None
    assert "goblin-brawler" not in ruleset.interaction_set
    assert brawler.spawn_count == 1
    assert brawler.hitpoints == 1_080
    assert brawler.damage == 337
    assert brawler.attack_interval_us == 1_100_000
    assert brawler.first_hit_delay_us == 200_000
    assert brawler.range_mtile == 800
    assert brawler.move_speed_mtile_per_s == 1_800
    assert brawler.targets == ("ground",)
    assert brawler.provenance["level_11_stats"] == (
        "deckmelon-goblin-cage-level11-2026-08-16",
    )


def test_goblin_drill_is_a_passive_spawner_not_a_repeating_turret() -> None:
    drill = load_ruleset(ROSTER_RULESET_ID).card("goblin-drill")

    assert drill.kind == "building"
    assert drill.damage is None
    assert drill.attack_interval_us is None
    assert drill.range_mtile is None
    assert drill.projectile is None
    assert drill.mechanics["tower_spawn_damage"] == 0
    assert drill.mechanics["spawn"] == {
        "card_id": "goblins",
        "interval_us": 3_500_000,
        "start_delay_us": 1_000_000,
        "max_alive": 6,
        "count": 1,
    }


def test_furnace_is_a_moving_troop_that_spawns_fire_spirits() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    furnace = ruleset.card("furnace")
    assert furnace.kind == "troop"
    assert furnace.hitpoints == 727
    assert furnace.damage == 135
    assert furnace.attack_interval_us == 1_700_000
    assert furnace.range_mtile == 5_500
    assert furnace.move_speed_mtile_per_s == 1_200
    assert furnace.lifetime_us is None
    # The rework is a ranged ground troop that can acquire both air and
    # ground targets; it is not a building-only spawner with a ground-only
    # attack channel.
    assert furnace.targets == ("air", "ground")
    assert furnace.mechanics["spawn"] == {
        "card_id": "fire-spirit",
        "interval_us": 5_000_000,
        "start_delay_us": 1_000_000,
        "max_alive": None,
        "count": 1,
    }
    assert ruleset.card("goblin").spawn_count == 1
    assert ruleset.card("goblin").hitpoints == ruleset.card("goblins").hitpoints

    engine = BattleEngine(ruleset, validate_every_tick=True)
    opponent_deck = ("furnace",) + tuple(
        card for card in ruleset.interaction_set if card != "furnace"
    )[:7]
    state = engine.new_battle(
        decks=(PLAYER_DECK, opponent_deck), seed=17, shuffle_decks=False
    )
    state.players[1].elixir_milli = ruleset.match.max_elixir_milli
    cell = engine.legal_cells(state, 1, "furnace")[0]
    action = PlayCardAction(1, 0, cell)
    assert engine.validate_action(state, action) is None
    engine.step(state, (action,))
    furnace_uid = next(
        uid for uid, entity in state.entities.items() if entity.card_id == "furnace"
    )
    assert state.entities[furnace_uid].kind == "troop"
    initial_position = (state.entities[furnace_uid].x_mtile, state.entities[furnace_uid].y_mtile)

    # One-second deploy plus a five-second cadence means the first child is
    # emitted by tick 120 at the canonical 50 ms simulation step.
    for _ in range(125):
        engine.step(state)
    spirits = [
        event
        for event in state.events
        if event.kind == "entity_spawned" and event.get("card_id") == "fire-spirit"
    ]
    assert spirits
    assert spirits[0].get("parent_uid") == furnace_uid
    moved_position = (state.entities[furnace_uid].x_mtile, state.entities[furnace_uid].y_mtile)
    assert moved_position != initial_position

    # The reworked Furnace is an active troop, so movement must not suppress
    # its normal single-target cauldron attack.  Put the live body at a known
    # in-range position beside the opponent's Princess Tower and advance the
    # ordinary attack scheduler; this intentionally exercises the projectile
    # channel rather than calling the impact helper directly.
    furnace = state.entities[furnace_uid]
    target = next(
        entity
        for entity in state.entities.values()
        if entity.kind == "tower" and entity.owner == 0 and entity.role == "right"
    )
    furnace.deploy_remaining_us = 0
    furnace.x_mtile = target.x_mtile
    furnace.y_mtile = target.y_mtile + 4_000
    furnace.target_uid = target.uid
    furnace.attack_cooldown_us = 0
    engine._advance_attacks(state)
    assert furnace.attack_count == 1
    assert any(
        event.kind == "projectile_spawned"
        and event.get("card_id") == "furnace"
        and event.get("source_uid") == furnace_uid
        and event.get("target_uid") == target.uid
        for event in state.events
    )


def test_furnace_spawner_does_not_apply_an_undocumented_four_spirit_cap() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    engine = BattleEngine(ruleset, validate_every_tick=True)
    opponent_deck = ("furnace",) + tuple(
        card for card in ruleset.interaction_set if card != "furnace"
    )[:7]
    state = engine.new_battle(
        decks=(PLAYER_DECK, opponent_deck), seed=19, shuffle_decks=False
    )
    state.players[1].elixir_milli = ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(1, 0, engine.legal_cells(state, 1, "furnace")[0]),))
    furnace = next(entity for entity in state.entities.values() if entity.card_id == "furnace")
    furnace.deploy_remaining_us = 0
    furnace.spawn_cooldown_us = 0

    # Isolate the spawner clock from combat/death resolution.  Six complete
    # cadences must create six live children when the definition is unbounded;
    # a stale four-child cap would fail this regression immediately.
    for _ in range(6):
        engine._advance_spawners(state, 5_000_000)
    spirits = [
        entity
        for entity in state.entities.values()
        if entity.card_id == "fire-spirit" and entity.alive
    ]
    assert len(spirits) == 6


def test_cannon_cart_has_official_shared_health_transform_and_hidden_building_form() -> None:
    ruleset = load_ruleset("v1")
    cart = ruleset.card("cannon-cart")
    broken = ruleset.card("cannon-cart-building")

    assert cart.mechanics["health_transform"] == {
        "threshold_permille": 500,
        "target_card_id": "cannon-cart-building",
        "preserve_hp": True,
        "preserve_max_hp": True,
        "lifetime_us": 30_000_000,
    }
    assert broken.kind == "building"
    assert broken.damage == cart.damage == 212
    assert broken.attack_interval_us == cart.attack_interval_us == 900_000
    assert broken.lifetime_us == 30_000_000
    assert broken.mechanics["lifetime_decay"] == "linear_hp"
    assert broken.mechanics["lifetime_start"] == "transform"


def test_split_cards_are_hidden_level11_forms_with_nested_death_components() -> None:
    ruleset = load_ruleset("v1")
    assert "golemite" not in ruleset.interaction_set
    assert ruleset.card("golem").mechanics["death"]["spawn_children"] == (
        {"card_id": "golemite", "count": 2},
    )
    assert ruleset.card("golemite").hitpoints == 1_039
    assert ruleset.card("golemite").damage == 84
    assert ruleset.card("golemite").mechanics["death"]["damage"] == 99
    assert ruleset.card("elixir-golem").mechanics["death"]["spawn_children"] == (
        {"card_id": "elixir-golemite", "count": 2},
    )
    assert ruleset.card("elixir-golemite").mechanics["death"]["spawn_children"] == (
        {"card_id": "elixir-blob", "count": 2},
    )
    assert ruleset.card("lava-hound").mechanics["death"]["spawn_children"] == (
        {"card_id": "lava-pup", "count": 6},
    )
    assert ruleset.card("goblin-giant").mechanics["death"]["spawn_children"] == (
        {"card_id": "spear-goblin", "count": 2},
    )
    assert ruleset.card("goblin-giant").mechanics["carrier"] == {
        "child_card_id": "spear-goblin",
        "count": 2,
        "offsets_mtile": ((-450, 0), (450, 0)),
        "release_on_death": True,
    }


def test_mirror_replays_the_previous_card_without_becoming_an_inert_projectile() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    engine = BattleEngine(ruleset, validate_every_tick=True)
    deck = ("fireball", "mirror", "ice-spirit", "log", "hog-rider", "cannon", "musketeer", "skeletons")
    state = engine.new_battle(decks=(deck, PLAYER_DECK), seed=2, shuffle_decks=False)
    state.players[0].elixir_milli = ruleset.match.max_elixir_milli
    engine.step(state, (PlayCardAction(0, 0, (3, 19)),))
    engine.step(state, (PlayCardAction(0, 0, (3, 19)),))
    assert any(event.kind == "card_mirrored" for event in state.events)
    assert not any(
        event.kind == "projectile_spawned" and event.get("card_id") == "mirror"
        for event in state.events
    )


def test_mirror_cannot_chain_after_a_mirror() -> None:
    ruleset = load_ruleset(ROSTER_RULESET_ID)
    engine = BattleEngine(ruleset, validate_every_tick=True)
    deck = ("mirror", "fireball", "ice-spirit", "log", "hog-rider", "cannon", "musketeer", "skeletons")
    state = engine.new_battle(decks=(deck, PLAYER_DECK), seed=4, shuffle_decks=False)
    state.players[0].elixir_milli = ruleset.match.max_elixir_milli
    state.players[0].last_played_card_id = "mirror"

    result = engine.apply_actions(state, (PlayCardAction(0, 0, (3, 19)),))[0]

    assert not result.accepted
    assert result.reason == "mirror_chain_not_allowed"
    assert state.players[0].hand[0] == "mirror"
