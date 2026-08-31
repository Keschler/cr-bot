from __future__ import annotations

import json

import pytest

from simulator.cli import main as simulator_main
from simulator.engine import BattleEngine
from simulator.generated_validation import (
    load_generated_manifest,
    validate_complete_generated_coverage,
    validate_generated_behavioral_obligations,
    validate_generated_scenarios,
    write_generated_validation_report,
)
from simulator.ruleset import load_ruleset
from simulator.scenario_factory import (
    generate_card_scenarios,
    generate_interaction_scenarios,
    generate_roster_scenarios,
    generated_manifest,
)


def test_generated_scenario_validation_reports_hash_identity(tmp_path) -> None:
    ruleset = load_ruleset("2026-08-04")
    generated = generate_card_scenarios(ruleset, "hog-rider")[:1]
    report = validate_generated_scenarios(BattleEngine(ruleset), generated, repeats=2)
    assert report["scenario_count"] == 1
    assert report["passed_count"] == 1
    assert report["failed_count"] == 0
    assert report["determinism_failures"] == 0
    assert report["revision_guard"]["status"] == "stable"
    assert report["code_revision"]["commit"]
    assert report["run_code_revision"] == report["revision_guard"]["start"]
    assert len(report["cases"][0]["hashes"]) == 2

    output = tmp_path / "generated-validation.json"
    write_generated_validation_report(output, report)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["report_hash"].startswith("sha256:")


def test_generated_validation_cli_runs_manifest(tmp_path) -> None:
    ruleset = load_ruleset("2026-08-04")
    generated = generate_card_scenarios(ruleset, "cannon")[:1]
    manifest_path = tmp_path / "generated.json"
    manifest_path.write_text(json.dumps(generated_manifest(generated)), encoding="utf-8")
    report_path = tmp_path / "report.json"
    assert simulator_main(
        [
            "--ruleset",
            "2026-08-04",
            "validate-generated",
            str(manifest_path),
            "--json-out",
            str(report_path),
        ]
    ) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["failed_count"] == 0
    _, scenarios = load_generated_manifest(manifest_path)
    assert len(scenarios) == 1


def test_generated_high_cost_card_must_be_exercised_not_rejected() -> None:
    ruleset = load_ruleset("2026-08-04-roster")
    generated = generate_card_scenarios(ruleset, "three-musketeers")
    assert generated
    scenario = generated[0].scenario
    assert scenario.actions[0].tick >= 400
    assert scenario.oracle["required_card_plays"] == [
        {"player": 1, "card_id": "three-musketeers"},
    ]
    report = validate_generated_scenarios(BattleEngine(ruleset), generated[:1], repeats=2)
    assert report["failed_count"] == 0


def test_parallel_generated_validation_matches_serial_rows() -> None:
    ruleset = load_ruleset("2026-08-04")
    generated = generate_card_scenarios(ruleset, "hog-rider")[:2]
    serial = validate_generated_scenarios(
        BattleEngine(ruleset, validate_every_tick=False),
        generated,
        repeats=2,
        workers=1,
    )
    parallel = validate_generated_scenarios(
        BattleEngine(ruleset, validate_every_tick=False),
        generated,
        repeats=2,
        workers=2,
    )
    assert parallel["failed_count"] == 0
    assert parallel["determinism_failures"] == 0
    assert parallel["cases"] == serial["cases"]
    assert parallel["workers"] == 2


def test_complete_generated_coverage_gate_rejects_focused_manifest() -> None:
    ruleset = load_ruleset("v1")
    generated = generate_card_scenarios(ruleset, "cannon")[:1]
    payload = generated_manifest(generated)

    gate = validate_complete_generated_coverage(
        payload,
        (row.scenario for row in generated),
        ruleset=ruleset,
    )

    assert gate["passed"] is False
    assert gate["scope"] == "roster_mechanics"
    assert gate["expected_card_count"] == 109
    assert gate["missing_cards"]
    assert gate["missing_coverage_count"] > 0


def test_behavioral_obligation_gate_accepts_source_specific_deployment_cases() -> None:
    ruleset = load_ruleset("v1")
    deployment = next(
        row for row in generate_card_scenarios(ruleset, "cannon")
        if row.mechanic == "deployment"
    )
    attack = next(
        row for row in generate_card_scenarios(ruleset, "cannon")
        if row.mechanic == "attack"
    )

    gate = validate_generated_behavioral_obligations(
        (deployment.scenario, attack.scenario)
    )

    assert gate["passed"] is True
    assert gate["behavioral_obligation_gap_count"] == 0


def test_complete_roster_obligation_audit_closes_deployment_and_lifecycle_gaps() -> None:
    """Every generated roster case has a source-specific event contract."""

    ruleset = load_ruleset("v1")
    generated = generate_roster_scenarios(ruleset, per_mechanic=1)
    gate = validate_generated_behavioral_obligations(
        tuple(row.scenario for row in generated)
    )

    assert gate["passed"] is True
    assert gate["behavioral_obligation_gap_count"] == 0
    assert gate["gaps"] == []


@pytest.mark.parametrize(
    ("card_id", "mechanic"),
    (
        ("hog-rider", "deployment"),
        ("goblin-gang", "lifecycle"),
        ("cannon", "deployment"),
        ("fireball", "lifecycle"),
    ),
)
def test_deployment_and_lifecycle_obligations_execute_for_each_card_kind(
    card_id: str,
    mechanic: str,
) -> None:
    ruleset = load_ruleset("v1")
    generated = next(
        row for row in generate_card_scenarios(ruleset, card_id)
        if row.mechanic == mechanic
    )
    assert generated.scenario.oracle["required_event_kinds"]
    assert generated.scenario.oracle["required_event_matches"]

    report = validate_generated_scenarios(
        BattleEngine(ruleset, validate_every_tick=False),
        (generated,),
        repeats=2,
    )
    assert report["failed_count"] == 0, report["failures"]


@pytest.mark.parametrize(
    ("card_id", "mechanic"),
    (
        ("goblin-machine", "secondary_attack"),
        ("goblin-demolisher", "threshold_charge"),
        ("battle-healer", "healing"),
        ("sparky", "recharge_windup"),
        ("x-bow", "projectile_speed"),
    ),
)
def test_high_impact_component_scenarios_have_exercised_obligations(
    card_id: str,
    mechanic: str,
) -> None:
    ruleset = load_ruleset("v1")
    generated = next(
        row for row in generate_card_scenarios(ruleset, card_id)
        if row.mechanic == mechanic
    )

    assert generated.scenario.oracle["required_event_matches"]
    report = validate_generated_scenarios(
        BattleEngine(ruleset, validate_every_tick=True),
        (generated,),
        repeats=2,
    )
    assert report["failed_count"] == 0, report["failures"]


def test_behavioral_obligation_gate_is_not_applicable_to_action_boundary_scope() -> None:
    ruleset = load_ruleset("v1")
    interaction = generate_interaction_scenarios(
        ruleset,
        opponent_card_ids=("cannon",),
        player_card_ids=("hog-rider",),
    )[0]

    gate = validate_generated_behavioral_obligations((interaction.scenario,))

    assert gate["passed"] is True
    assert gate["applicable"] is False
    assert gate["scope"] == "fixed_deck_interactions"
