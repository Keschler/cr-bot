from __future__ import annotations

from simulator.roster import (
    EVOLUTION_CUTOFF,
    OPPONENT_RELEASE_CUTOFF,
    PLAYER_DECK,
    build_mechanic_coverage,
    load_opponent_roster,
    validate_roster_against_catalog,
)


def test_v1_roster_has_fixed_player_deck_and_cutoffs() -> None:
    roster = load_opponent_roster()
    assert roster.release_cutoff_exclusive == OPPONENT_RELEASE_CUTOFF
    assert EVOLUTION_CUTOFF.isoformat() == "2023-06-19"
    assert PLAYER_DECK == (
        "hog-rider", "musketeer", "ice-golem", "ice-spirit",
        "cannon", "skeletons", "fireball", "log",
    )
    assert len(roster.eligible_cards) > 100
    assert set(PLAYER_DECK) <= set(roster.eligible_cards)


def test_roster_classifies_current_card_catalog_without_unknowns() -> None:
    roster = load_opponent_roster()
    from cr_bot.domain.card_metadata import CARD_METADATA

    report = validate_roster_against_catalog(roster, CARD_METADATA)
    assert report["missing_catalog_classification"] == []
    assert report["unknown_roster_cards"] == []
    assert report["missing_player_cards"] == []
    assert report["complete"] is True


def test_release_verification_remains_fail_closed_until_exact_dates_exist() -> None:
    roster = load_opponent_roster()
    from cr_bot.domain.card_metadata import CARD_METADATA

    report = validate_roster_against_catalog(
        roster, CARD_METADATA, require_release_verification=True
    )
    assert report["complete"] is False
    assert len(report["release_date_unverified"]) == len(roster.eligible_cards)


def test_coverage_graph_marks_current_eight_and_missing_opponents() -> None:
    roster = load_opponent_roster()
    coverage = build_mechanic_coverage(
        roster,
        {card: {"kind": "troop"} for card in roster.eligible_cards},
        set(PLAYER_DECK),
    )
    assert coverage["card_count"] == len(roster.eligible_cards)
    assert coverage["implemented_card_count"] == len(PLAYER_DECK)
    assert coverage["all_cards_implemented"] is False


def test_coverage_graph_exposes_component_mechanics_when_definitions_supply_them() -> None:
    roster = load_opponent_roster()
    coverage = build_mechanic_coverage(
        roster,
        {
            "prince": {
                "kind": "troop",
                "mechanics": {"charge_attack": {"charge_damage": 783}},
            }
        },
        {"prince"},
    )
    row = next(item for item in coverage["cards"] if item["card_id"] == "prince")
    assert "charge_attack" in row["required_mechanics"]
    assert "charge_movement" in row["required_mechanics"]
