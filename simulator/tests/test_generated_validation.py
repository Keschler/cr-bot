"""Regression coverage for non-vacuous generated roster fixtures."""

from __future__ import annotations

from simulator.engine import BattleEngine
from simulator.generated_validation import validate_generated_scenarios
from simulator.ruleset import load_ruleset
from simulator.scenario_factory import generate_card_scenarios


def test_generated_roster_reproductions_exercise_the_authored_branch() -> None:
    """Keep the seven strict-gate reproductions from regressing silently."""

    expected = {
        ("cannon", "projectile_speed"),
        ("firecracker", "damage"),
        ("ice-spirit", "projectile_speed"),
        ("lava-hound", "projectile_speed"),
        ("princess", "death"),
        ("rascals", "projectile_speed"),
        ("royal-giant", "projectile_speed"),
    }
    ruleset = load_ruleset("v1")
    scenarios = tuple(
        generated.scenario
        for card_id, mechanic in sorted(expected)
        for generated in generate_card_scenarios(ruleset, card_id)
        if generated.mechanic == mechanic
    )

    report = validate_generated_scenarios(
        BattleEngine(ruleset),
        scenarios,
        repeats=2,
    )

    assert {(row["card_id"], row["mechanic"]) for row in report["cases"]} == expected
    assert report["failed_count"] == 0, report["failures"]
    assert report["determinism_failures"] == 0


def test_generated_fixture_edge_cases_remain_non_vacuous() -> None:
    """Protect hand-cycle, short-range, and passive-target fixture paths."""

    expected = {
        ("bomb-tower", "projectile_speed"),
        ("electro-wizard", "multi_targeting"),
        ("elixir-collector", "building_navigation"),
        ("elixir-collector", "death_effect"),
        ("elixir-collector", "deployment"),
        ("elixir-collector", "lifetime"),
        ("elixir-collector", "passive_spawner"),
        ("elixir-collector", "resource_generation"),
        ("elixir-collector", "target_legality"),
        ("fire-spirit", "projectile_speed"),
        ("firecracker", "area_damage"),
        ("firecracker", "line_piercing"),
        ("heal-spirit", "projectile_speed"),
        ("princess", "area_damage"),
    }
    ruleset = load_ruleset("v1")
    scenarios = tuple(
        generated.scenario
        for card_id, mechanic in sorted(expected)
        for generated in generate_card_scenarios(ruleset, card_id)
        if generated.mechanic == mechanic
    )

    report = validate_generated_scenarios(
        BattleEngine(ruleset),
        scenarios,
        repeats=2,
    )

    assert {(row["card_id"], row["mechanic"]) for row in report["cases"]} == expected
    assert report["failed_count"] == 0, report["failures"]
    assert report["determinism_failures"] == 0
