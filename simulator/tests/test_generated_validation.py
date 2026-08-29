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
