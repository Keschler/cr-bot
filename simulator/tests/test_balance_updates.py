from __future__ import annotations

from simulator.catalog import build_fixed_ruleset_raw
from simulator.data_reconciliation import official_override_rows


BALANCE_SOURCE = "user-supplied-balance-update-2026-08-26"


def _override_values(card_id: str) -> dict[str, object]:
    return {
        str(row["field"]): row["value"]
        for row in official_override_rows(card_id)
        if row.get("source_id") == BALANCE_SOURCE
    }


def test_requested_special_variant_values_are_pinned_without_claiming_runtime_support() -> None:
    assert _override_values("berserker") == {
        "mechanics.hero_ability.duration_us": 3_500_000,
    }
    assert _override_values("elite-barbarians") == {
        "mechanics.evolution.spear_damage": 220,
        "mechanics.evolution.rage_duration_us": 2_500_000,
    }
    assert _override_values("goblinstein") == {
        "mechanics.monster.hitpoints": 2_240,
    }
    assert _override_values("archer-queen") == {"damage": 232}
    assert _override_values("little-prince") == {
        "mechanics.ability.charge_damage": 320,
    }


def test_requested_active_values_are_applied_to_the_generated_level11_ruleset() -> None:
    cards = build_fixed_ruleset_raw()["cards"]

    elite_barbarians = cards["elite-barbarians"]
    assert elite_barbarians["damage"] == 384
    assert elite_barbarians["mechanics"]["evolution"] == {
        "rage_duration_us": 2_500_000,
        "spear_damage": 220,
    }

    assert cards["electro-giant"]["damage"] == 184
    assert cards["electro-giant"]["mechanics"]["reflection"]["damage"] == 192
    assert cards["electro-spirit"]["mechanics"]["chain_attack"]["chain_range_mtile"] == 3_000
    assert cards["void"]["attack_interval_us"] == 1_000_000
    assert cards["void"]["mechanics"]["persistent_effect"]["tick_interval_us"] == 1_000_000


def test_balance_update_source_is_retained_in_generated_provenance() -> None:
    raw = build_fixed_ruleset_raw()
    assert raw["sources"][BALANCE_SOURCE]["confidence_tier"] == "E"
    assert raw["cards"]["electro-giant"]["provenance"]["damage"] == [BALANCE_SOURCE]
    assert raw["cards"]["electro-spirit"]["provenance"]["mechanics.chain_attack.chain_range_mtile"] == [
        BALANCE_SOURCE
    ]
    assert raw["cards"]["void"]["provenance"]["attack_interval_us"] == [BALANCE_SOURCE]
    assert raw["cards"]["void"]["provenance"]["mechanics.persistent_effect.tick_interval_us"] == [
        BALANCE_SOURCE
    ]
