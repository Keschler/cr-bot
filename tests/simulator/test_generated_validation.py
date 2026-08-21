from __future__ import annotations

import json

from simulator.cli import main as simulator_main
from simulator.engine import BattleEngine
from simulator.generated_validation import (
    load_generated_manifest,
    validate_generated_scenarios,
    write_generated_validation_report,
)
from simulator.ruleset import load_ruleset
from simulator.scenario_factory import generate_card_scenarios, generated_manifest


def test_generated_scenario_validation_reports_hash_identity(tmp_path) -> None:
    ruleset = load_ruleset("2026-08-04")
    generated = generate_card_scenarios(ruleset, "hog-rider")[:1]
    report = validate_generated_scenarios(BattleEngine(ruleset), generated, repeats=2)
    assert report["scenario_count"] == 1
    assert report["passed_count"] == 1
    assert report["failed_count"] == 0
    assert report["determinism_failures"] == 0
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
