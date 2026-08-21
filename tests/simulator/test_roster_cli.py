from __future__ import annotations

import json

from simulator.cli import main


def test_roster_cli_reports_fail_closed_coverage(tmp_path) -> None:
    output = tmp_path / "roster.json"
    assert main(["roster", "--json-out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["coverage"]["all_cards_implemented"] is False


def test_roster_cli_can_require_exact_release_and_full_coverage(tmp_path) -> None:
    output = tmp_path / "roster-strict.json"
    assert main(
        [
            "roster",
            "--require-release-verification",
            "--require-coverage",
            "--json-out",
            str(output),
        ]
    ) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["catalog"]["release_date_unverified"]
    assert payload["coverage"]["missing_card_count"] > 0


def test_roster_complete_ruleset_is_executable_but_not_training_ready(tmp_path) -> None:
    output = tmp_path / "roster-full.json"
    assert main(
        [
            "--ruleset",
            "2026-08-04-roster",
            "roster",
            "--require-coverage",
            "--json-out",
            str(output),
        ]
    ) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["coverage"]["all_cards_implemented"] is True
    assert payload["coverage"]["all_cards_fidelity_ready"] is False
    assert payload["coverage"]["fidelity_not_ready_cards"]
