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


def test_current_card_balance_values_are_applied_to_the_fixed_ruleset() -> None:
    cards = build_fixed_ruleset_raw()["cards"]

    assert cards["bats"]["attack_interval_us"] == 1_200_000
    assert cards["barbarians"]["attack_interval_us"] == 1_400_000
    assert cards["royal-giant"]["attack_interval_us"] == 1_800_000
    assert cards["dart-goblin"]["sight_range_mtile"] == 7_000
    assert cards["dart-goblin"]["range_mtile"] == 6_500
    assert cards["electro-dragon"]["hitpoints"] == 1_049
    assert cards["executioner"]["damage"] == 179
    assert cards["goblin-giant"]["hitpoints"] == 3_110
    assert cards["goblin-giant"]["targets"] == ["building", "crown_tower"]
    assert cards["goblin-giant"]["mechanics"]["building_only"] is True
    assert cards["rascals"]["hitpoints"] == 1_832
    assert cards["rascals"]["damage"] == 125
    assert cards["rascal-boy"]["hitpoints"] == 1_832
    assert cards["rascal-girl"]["damage"] == 125
    assert cards["royal-ghost"]["mechanics"]["stealth_recloak_us"] == 2_000_000
    assert cards["ice-wizard"]["mechanics"]["status"]["speed_multiplier_milli"] == 700
    assert cards["ice-wizard"]["mechanics"]["status"]["hit_speed_multiplier_milli"] == 700
    assert cards["ice-golem"]["mechanics"]["death"]["status"] == {
        "duration_us": 2_000_000,
        "hit_speed_multiplier_milli": 700,
        "kind": "slow",
        "speed_multiplier_milli": 700,
    }
    assert cards["firecracker"]["sight_range_mtile"] == 8_000
    assert cards["firecracker"]["mechanics"]["recoil_mtile"] == 1_000
    assert cards["furnace"]["damage"] == 179
    assert cards["goblin-hut"]["lifetime_us"] == 30_000_000
    assert cards["goblin-curse"]["crown_tower_damage"] == 60
    assert cards["goblin-curse"]["mechanics"]["persistent_effect"]["crown_damage_per_tick"] == 10
    assert cards["hog-rider"]["first_hit_delay_us"] == 600_000
    assert cards["lumberjack"]["mechanics"]["death_rage"] == {
            "duration_us": 5_500_000,
        "tick_interval_us": 100_000,
        "radius_mtile": 3_000,
        "speed_multiplier_milli": 1_300,
        "hit_speed_multiplier_milli": 1_300,
        "targets": ["air", "ground", "building", "crown_tower"],
    }


def test_freeze_spell_does_not_inherit_ice_spirit_duration() -> None:
    cards = build_fixed_ruleset_raw()["cards"]

    assert cards["freeze"]["mechanics"]["status"]["duration_us"] == 4_000_000
    assert cards["ice-spirit"]["mechanics"]["status"]["duration_us"] == 1_100_000


def test_spawn_source_specific_first_hit_values_are_kept_separate() -> None:
    cards = build_fixed_ruleset_raw()["cards"]

    assert cards["goblin-machine"]["first_hit_delay_us"] == 500_000
    assert cards["goblin-machine"]["mechanics"]["secondary_attack"]["damage"] == 304
    assert cards["goblin-machine"]["mechanics"]["secondary_attack"]["crown_tower_damage"] == 152
    assert cards["guards"]["first_hit_delay_us"] == 500_000
    assert cards["spear-goblins"]["first_hit_delay_us"] == 500_000
    assert cards["spear-goblin"]["first_hit_delay_us"] == 500_000
    assert cards["goblins"]["first_hit_delay_us"] == 600_000
    assert cards["goblin-gang"]["first_hit_delay_us"] == 600_000
    assert cards["goblin-gang"]["mechanics"]["spawn_children"] == [
        {
            "card_id": "goblin-gang-goblin",
            "count": 3,
            "offsets_mtile": [[-800, 400], [0, 400], [800, 400]],
        },
        {
            "card_id": "spear-goblin",
            "count": 3,
            "offsets_mtile": [[-800, -400], [0, -400], [800, -400]],
        },
    ]
    assert cards["goblin-gang-goblin"]["first_hit_delay_us"] == 600_000
    assert cards["goblin"]["first_hit_delay_us"] == 400_000
    assert cards["barbarian-barrel"]["damage"] == 232
