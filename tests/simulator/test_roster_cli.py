from __future__ import annotations

import json

from simulator.cli import main


def test_roster_cli_reports_fail_closed_coverage(tmp_path) -> None:
    output = tmp_path / "roster.json"
    assert main(["roster", "--json-out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["ruleset_id"] == "v1"
    assert payload["coverage"]["all_cards_implemented"] is True
    assert payload["coverage"]["all_cards_fidelity_ready"] is False


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
    assert payload["coverage"]["missing_card_count"] == 0
    assert payload["coverage"]["all_cards_fidelity_ready"] is False


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


def test_reconcile_data_audits_the_requested_ruleset(tmp_path) -> None:
    """The report must identify the ruleset the caller asked for.

    The command substituted the 124-card roster artifact when ``--ruleset
    2026-08-04`` was passed, so the eight-card base set could never be audited
    and the report's identity did not match the request.
    """

    base_output = tmp_path / "base.json"
    assert main(["--ruleset", "2026-08-04", "reconcile-data", "--json-out", str(base_output)]) == 0
    base = json.loads(base_output.read_text(encoding="utf-8"))
    assert base["ruleset_id"] == "2026-08-04"
    assert base["summary"]["card_count"] == 8

    v1_output = tmp_path / "v1.json"
    assert main(["--ruleset", "v1", "reconcile-data", "--json-out", str(v1_output)]) == 0
    fixed = json.loads(v1_output.read_text(encoding="utf-8"))
    assert fixed["ruleset_id"] == "v1"
    assert fixed["summary"]["card_count"] == 109
    # The roster artifact is reachable explicitly and is no longer what the
    # eight-card flag loads.
    roster_output = tmp_path / "roster.json"
    assert main(
        ["--ruleset", "2026-08-04-roster", "reconcile-data", "--json-out", str(roster_output)]
    ) == 0
    roster = json.loads(roster_output.read_text(encoding="utf-8"))
    assert roster["ruleset_id"] == "2026-08-04-roster"
