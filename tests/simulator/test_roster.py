from __future__ import annotations

from simulator.roster import (
    EVOLUTION_CUTOFF,
    OPPONENT_RELEASE_CUTOFF,
    PLAYER_DECK,
    build_mechanic_coverage,
    load_opponent_roster,
    validate_roster_against_catalog,
)
from simulator.engine import BASE_HOG_CYCLE_DECK
from simulator.physical_lab.workflow.automation import FIXED_HOG_CYCLE_DECK


def test_v1_roster_has_fixed_player_deck_and_cutoffs() -> None:
    roster = load_opponent_roster()
    assert roster.release_cutoff_exclusive == OPPONENT_RELEASE_CUTOFF
    assert EVOLUTION_CUTOFF.isoformat() == "2023-06-19"
    assert PLAYER_DECK == (
        "hog-rider", "cannon", "musketeer", "skeletons",
        "ice-golem", "ice-spirit", "fireball", "log",
    )
    assert len(roster.eligible_cards) > 100
    assert set(PLAYER_DECK) <= set(roster.eligible_cards)


def test_fixed_order_is_shared_by_policy_engine_and_physical_lab() -> None:
    assert PLAYER_DECK == BASE_HOG_CYCLE_DECK == FIXED_HOG_CYCLE_DECK
    assert PLAYER_DECK[:4] == (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
    )


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


def test_coverage_graph_exposes_all_exceptional_v1_components() -> None:
    roster = load_opponent_roster()
    from simulator.ruleset import load_fixed_ruleset

    ruleset = load_fixed_ruleset()
    definitions = {
        card_id: {
            "kind": definition.kind,
            "is_air": definition.mechanics.get("movement_layer") == "air",
            "is_splash": definition.area_radius_mtile is not None,
            "mechanics": definition.mechanics,
        }
        for card_id, definition in ruleset.cards.items()
        if card_id in roster.eligible_cards
    }
    coverage = build_mechanic_coverage(roster, definitions, set(ruleset.cards))
    by_card = {row["card_id"]: set(row["required_mechanics"]) for row in coverage["cards"]}

    assert "secondary_attack" in by_card["goblin-machine"]
    assert "threshold_charge" in by_card["goblin-demolisher"]
    assert "healing" in by_card["battle-healer"]
    assert "recharge_windup" in by_card["sparky"]
    assert "projectile_speed" in by_card["x-bow"]
    assert {"shield"} <= by_card["guards"]
    assert {"line_piercing"} <= by_card["magic-archer"]
    assert {"returning_projectile"} <= by_card["executioner"]


def test_roster_ruleset_id_resolves_to_a_ruleset_carrying_its_cards() -> None:
    """The declared ruleset must be able to run the roster it describes.

    The roster pointed at the eight-card base artifact while describing 109
    opponent cards. The base set contains exactly the eight player-deck cards,
    so the declaration was not merely cosmetic: any use of the field would have
    raised on the first opponent card.
    """

    from simulator.ruleset import load_ruleset

    roster = load_opponent_roster()
    ruleset = load_ruleset(roster.ruleset_id)

    assert ruleset.ruleset_id == roster.ruleset_id
    missing = [card for card in roster.eligible_cards if card not in ruleset.cards]
    assert missing == []
    assert len(ruleset.cards) >= len(roster.eligible_cards)


def test_excluded_releases_absent_from_the_catalog_are_reported_separately() -> None:
    """A card excluded because it postdates the catalog is not an unknown.

    Ronin and Minion Giant are real releases with no catalog entry, and the
    exclusion list is how the roster records that they are deliberately out of
    scope. They must not be reported as unknown eligible cards, but they must
    stay visible so an incomplete catalog is still auditable.
    """

    from cr_bot.domain.card_metadata import CARD_METADATA

    roster = load_opponent_roster()
    report = validate_roster_against_catalog(roster, CARD_METADATA)

    assert report["unknown_roster_cards"] == []
    assert "ronin" in report["excluded_absent_from_catalog"]
    assert "minion-giant" in report["excluded_absent_from_catalog"]
    # They are classified, not silently absent from the roster's own scope.
    assert {"ronin", "minion-giant"} <= roster.all_classified_cards
    assert not {"ronin", "minion-giant"} & set(roster.eligible_cards)
