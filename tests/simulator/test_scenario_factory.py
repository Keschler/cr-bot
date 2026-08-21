from __future__ import annotations

import json

from simulator.cli import main as simulator_main
from simulator.engine import BattleEngine
from simulator.generated_validation import validate_generated_scenarios
from simulator.ruleset import load_ruleset
from simulator.scenario_factory import (
    card_mechanics,
    generate_card_scenarios,
    generate_interaction_scenarios,
    generate_opponent_pair_scenarios,
    generate_roster_scenarios,
)


def test_card_mechanics_include_shared_components() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    assert "air_navigation" in card_mechanics(ruleset, "baby-dragon")
    assert "periodic_spawn" in card_mechanics(ruleset, "tombstone")
    assert "resource_generation" in card_mechanics(ruleset, "elixir-collector")
    assert "impact_spawn" in card_mechanics(ruleset, "goblin-barrel")
    assert "persistent_area_effect" in card_mechanics(ruleset, "poison")
    assert "persistent_area_effect" in card_mechanics(ruleset, "graveyard")
    assert "health_threshold_transform" in card_mechanics(ruleset, "cannon-cart")


def test_passive_buildings_generate_stream_cases_not_turret_cases() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    for card_id in ("barbarian-hut", "goblin-cage", "goblin-drill", "goblin-hut", "tombstone"):
        mechanics = set(card_mechanics(ruleset, card_id))
        assert "passive_spawner" in mechanics
        assert "building_navigation" in mechanics
        assert "lifetime" in mechanics
        assert "attack" not in mechanics
        assert "target_acquisition" not in mechanics
    furnace_mechanics = set(card_mechanics(ruleset, "furnace"))
    assert {"attack", "target_acquisition", "periodic_spawn"} <= furnace_mechanics


def test_interaction_matrix_keeps_fixed_player_deck_and_exercises_both_cards() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    rows = generate_interaction_scenarios(
        ruleset,
        opponent_card_ids=("cannon", "three-musketeers"),
        player_card_ids=("hog-rider", "fireball"),
    )
    assert len(rows) == 4
    for row in rows:
        assert set(row.scenario.decks[0]) == {
            "hog-rider", "musketeer", "ice-golem", "ice-spirit",
            "cannon", "skeletons", "fireball", "log",
        }
        assert row.scenario.decks[0][0] == row.scenario.oracle["player_card_id"]
        assert row.scenario.decks[1][0] == row.scenario.oracle["opponent_card_id"]
        assert row.scenario.oracle["required_card_plays"]
        assert row.scenario.actions == tuple(sorted(
            row.scenario.actions,
            key=lambda scheduled: (scheduled.tick, scheduled.action.player),
        ))


def test_every_roster_card_gets_reproducible_runnable_cases() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    rows = generate_roster_scenarios(ruleset, per_mechanic=1)
    assert len({row.card_id for row in rows}) == len(ruleset.interaction_set)
    assert rows == generate_roster_scenarios(ruleset, per_mechanic=1)

    engine = BattleEngine(ruleset)
    for row in rows[:12]:
        state = engine.new_battle(
            decks=row.scenario.decks,
            seed=row.scenario.seed,
            shuffle_decks=row.scenario.shuffle_decks,
        )
        for _ in range(row.scenario.max_ticks or 1):
            actions = tuple(
                scheduled.action
                for scheduled in row.scenario.actions
                if scheduled.tick == state.tick
            )
            engine.step(state, actions)
            if state.terminal:
                break


def test_generated_support_fixtures_require_branch_events() -> None:
    """A generated card case must exercise its mechanic, not only play it."""

    ruleset = load_ruleset("2026-08-04-roster")
    selected = (
        next(row for row in generate_card_scenarios(ruleset, "cannon") if row.mechanic == "attack"),
        next(row for row in generate_card_scenarios(ruleset, "fireball") if row.mechanic == "victim_selection"),
        next(row for row in generate_card_scenarios(ruleset, "furnace") if row.mechanic == "periodic_spawn"),
        next(row for row in generate_card_scenarios(ruleset, "clone") if row.mechanic == "clone_component"),
        next(row for row in generate_card_scenarios(ruleset, "battle-ram") if row.mechanic == "attack"),
        next(row for row in generate_card_scenarios(ruleset, "suspicious-bush") if row.mechanic == "attack"),
        next(row for row in generate_card_scenarios(ruleset, "poison") if row.mechanic == "status_effect"),
        next(row for row in generate_card_scenarios(ruleset, "guards") if row.mechanic == "shield"),
        next(row for row in generate_card_scenarios(ruleset, "goblin-gang") if row.mechanic == "spawn_composition"),
    )
    for row in selected:
        assert row.scenario.oracle["required_event_kinds"]
        report = validate_generated_scenarios(
            BattleEngine(ruleset, validate_every_tick=False),
            (row,),
            repeats=2,
        )
        assert report["failed_count"] == 0, report["failures"]


def test_generated_variants_change_geometry_and_timing_but_remain_reproducible() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    rows = generate_card_scenarios(ruleset, "cannon", per_mechanic=4)
    attack_rows = [row for row in rows if row.mechanic == "attack"]
    assert len(attack_rows) == 4
    assert rows == generate_card_scenarios(ruleset, "cannon", per_mechanic=4)
    action_signatures = {
        tuple(
            (scheduled.tick, scheduled.action.cell)
            for scheduled in row.scenario.actions
            if hasattr(scheduled.action, "cell")
        )
        for row in attack_rows
    }
    assert len(action_signatures) == 4
    assert attack_rows[0].scenario.actions != attack_rows[1].scenario.actions
    assert attack_rows[0].scenario.seed != attack_rows[1].scenario.seed


def test_interaction_variants_are_not_seed_only_duplicates() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    rows = generate_interaction_scenarios(
        ruleset,
        opponent_card_ids=("cannon",),
        player_card_ids=("hog-rider",),
        variants=4,
    )
    assert len(rows) == 4
    assert len({tuple((item.tick, item.action.cell) for item in row.scenario.actions) for row in rows}) == 4
    assert rows == generate_interaction_scenarios(
        ruleset,
        opponent_card_ids=("cannon",),
        player_card_ids=("hog-rider",),
        variants=4,
    )


def test_opponent_pair_matrix_keeps_pairs_unordered_and_plays_both_cards() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    rows = generate_opponent_pair_scenarios(
        ruleset,
        opponent_card_ids=("cannon", "fireball", "balloon"),
        variants=2,
    )
    assert len(rows) == 6  # three unordered pairs × two deterministic lanes
    assert len({row.scenario.scenario_id for row in rows}) == len(rows)
    for row in rows:
        assert row.scenario.decks[0] == tuple(
            ("hog-rider", "musketeer", "ice-golem", "ice-spirit", "cannon", "skeletons", "fireball", "log")
        )
        assert row.scenario.oracle["mechanic"] == "opponent_pair_interaction"
        required = row.scenario.oracle["required_card_plays"]
        assert [item["card_id"] for item in required] == [
            "hog-rider",
            row.scenario.oracle["first_opponent_card_id"],
            row.scenario.oracle["second_opponent_card_id"],
        ]
        assert row.scenario.actions[-1].tick > row.scenario.actions[0].tick
    assert rows == generate_opponent_pair_scenarios(
        ruleset,
        opponent_card_ids=("cannon", "fireball", "balloon"),
        variants=2,
    )


def test_opponent_pair_matrix_runs_a_mixed_building_spell_air_fixture() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    rows = generate_opponent_pair_scenarios(
        ruleset,
        opponent_card_ids=("goblin-drill", "fireball", "balloon"),
    )
    report = validate_generated_scenarios(
        BattleEngine(ruleset, validate_every_tick=False),
        rows,
        repeats=2,
    )
    assert report["failed_count"] == 0, report["failures"]


def test_generate_opponent_pairs_cli_writes_the_pinned_pair_manifest(tmp_path) -> None:
    path = tmp_path / "opponent-pairs.json"
    assert simulator_main(
        [
            "--ruleset",
            "v1",
            "generate-opponent-pairs",
            "--opponent-card",
            "cannon",
            "--opponent-card",
            "fireball",
            "--opponent-card",
            "balloon",
            "--variants",
            "2",
            "--json-out",
            str(path),
        ]
    ) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "simulator_generated_scenario_manifest"
    assert payload["summary"]["scenario_count"] == 6
    assert payload["summary"]["card_count"] == 3
    assert payload["summary"]["unordered_pair_count"] == 3
    assert len(payload["cases"]) == 6
    assert all(case["ruleset_id"] == "v1" for case in payload["cases"])


def test_generate_scenarios_cli_is_versioned(tmp_path) -> None:
    path = tmp_path / "scenarios.json"
    assert simulator_main(
        [
            "--ruleset",
            "2026-08-04-roster",
            "generate-scenarios",
            "--card",
            "baby-dragon",
            "--json-out",
            str(path),
        ]
    ) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "simulator_generated_scenario_manifest"
    assert payload["summary"]["card_count"] == 1
    assert payload["cases"][0]["ruleset_id"] == "2026-08-04-roster"
