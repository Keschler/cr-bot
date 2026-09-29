from __future__ import annotations

import json

from simulator.catalog import build_fixed_ruleset_raw, build_roster_ruleset_raw
from simulator.data_reconciliation import (
    DECKSHOP_CORE_SOURCE_PATH,
    DECKSHOP_SOURCE_PATH,
    apply_official_overrides,
    load_level11_source,
    reconcile_ruleset,
    source_payload_sha256,
)
from simulator.ruleset import calculate_content_hash, load_ruleset, ruleset_path


def test_pinned_level11_source_is_level11_and_has_expected_core_rows() -> None:
    source = load_level11_source()
    assert source["level"] == 11
    assert source["cards"]["hog-rider"]["damage"] == 317
    assert source["cards"]["hog-rider"]["hitpoints"] == 1697
    assert source["cards"]["cannon"]["duration_s"] == 30.0
    assert source["cards"]["fireball"]["tower_damage"] == 207


def test_deckshop_level11_snapshot_is_available_as_independent_corroboration() -> None:
    source = load_level11_source(DECKSHOP_SOURCE_PATH)
    assert source["source_id"] == "deckshop-battle-healer-2026-08-14"
    assert source["cards"]["battle-healer"] == {
        "attack_interval_us": 2_000_000,
        "damage": 268,
        "hitpoints": 1920,
        "move_speed": "medium",
    }


def test_fixed_ruleset_applies_deckshop_core_spell_scalars_without_overriding_official_tower_scaling() -> None:
    source = load_level11_source(DECKSHOP_CORE_SOURCE_PATH)
    assert source["cards"]["fireball"]["damage"] == 688
    assert source["cards"]["log"]["damage"] == 268
    assert source["cards"]["tornado"] == {
        "damage": 84,
        "tower_damage": 27,
        "radius_tiles": 5.5,
        "duration_s": 1.1,
    }

    cards = build_fixed_ruleset_raw()["cards"]
    assert cards["fireball"]["damage"] == 688
    assert cards["fireball"]["crown_tower_damage"] == 172
    assert cards["log"]["damage"] == 268
    assert cards["log"]["crown_tower_damage"] == 35
    assert cards["tornado"]["damage"] == 84
    assert cards["tornado"]["crown_tower_damage"] == 27
    assert cards["tornado"]["area_radius_mtile"] == 5_500
    assert cards["tornado"]["mechanics"]["persistent_effect"]["damage_schedule"] == [84]
    assert cards["tornado"]["mechanics"]["persistent_effect"]["crown_damage_schedule"] == [27]


def test_official_overrides_are_field_level_and_do_not_mutate_input() -> None:
    original = {
        "hitpoints": 230,
        "damage": 110,
        "mechanics": {"crown_tower_connection": "normal"},
        "provenance": {},
    }
    resolved, rows = apply_official_overrides("ice-spirit", original)

    assert original["hitpoints"] == 230
    assert resolved["hitpoints"] == 215
    assert resolved["mechanics"]["crown_tower_connection"] == (
        "expected-no-unassisted-connection"
    )
    assert {row["field"] for row in rows} >= {
        "hitpoints",
        "mechanics.crown_tower_connection",
    }
    assert resolved["provenance"]["hitpoints"] == ["official-august-2026"]


def test_roster_contains_current_explicit_patch_values() -> None:
    raw = build_roster_ruleset_raw()
    assert raw["content_hash"] == calculate_content_hash(raw)
    cards = raw["cards"]
    assert cards["baby-dragon"]["damage"] == 168
    assert cards["electro-giant"]["hitpoints"] == 3952
    assert cards["mortar"]["attack_interval_us"] == 4_700_000
    assert cards["tesla"]["hitpoints"] == 1182
    assert cards["tesla"]["lifetime_us"] == 25_000_000
    assert cards["void"]["elixir_milli"] == 5_000
    assert cards["void"]["damage"] == 696
    assert cards["void"]["mechanics"]["damage_by_target_count"]["5+"] == 153
    assert cards["poison"]["mechanics"]["persistent_effect"]["damage_per_tick"] == 92
    assert cards["poison"]["mechanics"]["persistent_effect"]["crown_damage_per_tick"] == 21
    assert cards["graveyard"]["mechanics"]["persistent_effect"]["spawn"]["max_spawns"] == 12
    assert cards["goblin-curse"]["damage"] == 210
    assert cards["goblin-curse"]["mechanics"]["persistent_effect"]["damage_per_tick"] == 35
    assert cards["goblin-curse"]["mechanics"]["persistent_effect"]["crown_damage_per_tick"] == 10
    assert cards["goblin-curse"]["mechanics"]["persistent_effect"]["status"]["on_death_spawn_card_id"] == "goblin"


def test_generated_cards_prefer_level11_formation_speed_target_and_projectile_fields() -> None:
    cards = build_fixed_ruleset_raw()["cards"]

    # These values are explicit in the pinned Level-11 structured source.  A
    # regression to the old level-16 metadata fallback would change both
    # entity counts and path timings while leaving basic schema tests green.
    assert cards["goblins"]["spawn_count"] == 4
    assert cards["goblin-gang"]["spawn_count"] == 6
    assert cards["hog-rider"]["targets"] == ["building", "crown_tower"]
    assert cards["hog-rider"]["move_speed_mtile_per_s"] == 2_400
    assert cards["bats"]["move_speed_mtile_per_s"] == 2_400
    assert cards["battle-healer"]["projectile"] is None
    assert cards["ice-spirit"]["mechanics"]["suicide_on_attack"] is True


def test_generated_card_builder_limits_area_and_does_not_turn_huts_into_turrets() -> None:
    cards = build_fixed_ruleset_raw()["cards"]
    # Firecracker is the explicit exception: its projectile has a bounded
    # splash burst and recoil, while the generic range fallback remains
    # disabled for non-area ranged cards.
    assert cards["firecracker"]["area_radius_mtile"] == 1_500
    # The August 2025 official scalar reduced the recoil from the 1.5-tile
    # authored default to one tile.
    assert cards["firecracker"]["mechanics"]["recoil_mtile"] == 1_000
    assert cards["electro-dragon"]["area_radius_mtile"] is None
    for card_id in ("barbarian-hut", "goblin-hut", "tombstone"):
        assert cards[card_id]["attack_interval_us"] is None
        assert cards[card_id]["damage"] is None
    furnace = cards["furnace"]
    assert furnace["kind"] == "troop"
    assert furnace["hitpoints"] == 727
    assert furnace["damage"] == 179
    assert furnace["attack_interval_us"] == 1_700_000
    assert furnace["range_mtile"] == 5_500
    assert furnace["move_speed_mtile_per_s"] == 1_200
    assert furnace["mechanics"]["spawn"]["card_id"] == "fire-spirit"


def test_v1_is_the_single_explicit_fixed_runtime_artifact() -> None:
    ruleset = load_ruleset("v1")
    assert ruleset.ruleset_id == "v1"
    assert ruleset.metadata["fixed_data"] is True
    assert ruleset.metadata["versioning_policy"] == "constant-until-v2"
    assert len(ruleset.interaction_set) == 109
    assert json.loads(ruleset_path("v1").read_text()) == build_fixed_ruleset_raw()


def test_reconciliation_is_fail_closed_for_missing_and_conflicting_fields() -> None:
    report = reconcile_ruleset(load_ruleset("2026-08-04-roster"))
    summary = report["summary"]

    assert summary["card_count"] == 109
    assert summary["field_count"] > summary["card_count"]
    assert summary["counts_by_status"]["verified_official"] > 0
    assert summary["unresolved_count"] > 0
    assert summary["fully_verified"] is False
    assert summary["training_ready"] is False
    # Troop ``duration_s`` values are not body lifetimes (see the
    # reconciliation comment); remaining unresolved rows must therefore be
    # genuine missing/conflicting fields, not polymorphic duration noise.
    assert all(
        row["status"].startswith("unresolved_") for row in report["unresolved"]
    )
    # The Furnace's structured disagreement after the August 2025 moving-troop
    # rework is now resolved by an official August 2026 scalar (5.5-tile
    # range), so it is no longer an unresolved row.  The disagreement must
    # still be retained and visible rather than dropped.
    furnace_range = next(
        row
        for row in report["rows"]
        if row["card_id"] == "furnace" and row["field"] == "range_mtile"
    )
    assert furnace_range["status"] == "official_override_conflict"
    assert furnace_range["simulator_value"] == 5_500
    assert furnace_range["structured_value"] == 6_000
    assert furnace_range["official_value"] == 5_500


def test_additional_structured_source_never_resolves_a_primary_conflict() -> None:
    """One dissenting additional row must not promote a conflicting value.

    The primary structured snapshot and the executable value disagree for
    ``fireball.damage`` (689 versus 688).  Supplying an additional source that
    happens to agree with the simulator value is not independent evidence
    against the primary source, so the row must stay fail-closed.
    """

    base = reconcile_ruleset(load_ruleset("2026-08-04-roster"))
    conflict = next(
        row
        for row in base["rows"]
        if row["card_id"] == "fireball" and row["field"] == "damage"
    )
    assert conflict["status"] == "unresolved_source_conflict"

    corroborating = {"cards": {"fireball": {"damage": conflict["simulator_value"]}}}
    report = reconcile_ruleset(
        load_ruleset("2026-08-04-roster"),
        additional_sources=(("corroborating", corroborating),),
    )
    row = next(
        row
        for row in report["rows"]
        if row["card_id"] == "fireball" and row["field"] == "damage"
    )
    assert row["status"] == "unresolved_source_conflict"
    assert row["additional_structured_values"]["corroborating"] == conflict["simulator_value"]
    assert any(
        entry["card_id"] == "fireball" and entry["field"] == "damage"
        for entry in report["unresolved"]
    )


def test_reconciliation_can_include_deckshop_as_an_additional_source() -> None:
    report = reconcile_ruleset(
        load_ruleset("2026-08-04-roster"),
        additional_sources=(
            ("deckshop-battle-healer-2026-08-14", load_level11_source(DECKSHOP_SOURCE_PATH)),
        ),
    )
    row = next(
        row
        for row in report["rows"]
        if row["card_id"] == "battle-healer" and row["field"] == "hitpoints"
    )
    assert row["additional_structured_values"]["deckshop-battle-healer-2026-08-14"] == 1920


def test_passive_spawner_legacy_attack_columns_are_not_applicable() -> None:
    report = reconcile_ruleset(load_ruleset("2026-08-04-roster"))
    for card_id in ("barbarian-hut", "goblin-cage", "goblin-drill", "goblin-hut", "tombstone"):
        rows = {
            row["field"]: row
            for row in report["rows"]
            if row["card_id"] == card_id
        }
        for field in ("damage", "attack_interval_us", "range_mtile"):
            assert rows[field]["status"] == "not_applicable"
            assert rows[field]["structured_value"] is None
        assert not any(
            row["card_id"] == card_id
            and row["status"] == "unresolved_source_conflict"
            for row in report["rows"]
        )


def test_in_memory_reconciliation_reports_the_payload_hash() -> None:
    source = load_level11_source()
    report = reconcile_ruleset(load_ruleset("v1"), source_payload=source)
    assert report["source"]["sha256"] == source_payload_sha256(source)


def test_movement_speed_is_compared_against_the_structured_speed_tier() -> None:
    """A wrong speed conversion has to show up as a reconciliation conflict.

    The reconciler compared damage, hitpoints, and cadence but not movement, so
    a conversion that was wrong for twelve cards produced no report at all.
    Comparing the qualitative tier through the same table the catalog uses
    makes the conversion auditable, and spells and buildings stay exempt
    because they never path and the source fills a default tier on their rows.
    """

    report = reconcile_ruleset(load_ruleset("v1"))
    rows = [row for row in report["rows"] if row["field"] == "move_speed_mtile_per_s"]
    assert rows, "movement speed must be part of the compared scalar set"

    by_status: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_status.setdefault(str(row["status"]), []).append(row)
    assert by_status.get("unresolved_source_conflict") is None
    assert len(by_status.get("structured_match", [])) > 50
    for row in by_status.get("not_applicable", []):
        assert row["simulator_value"] is None

    # A conversion error in either direction is reported, not absorbed.
    giant = next(row for row in rows if row["card_id"] == "giant")
    assert giant["simulator_value"] == 900
    assert giant["structured_value"] == 900


def test_unapplied_mechanics_override_blocks_full_verification() -> None:
    """An override the executable value does not match is a defect.

    A mechanics row whose value differs from the official override means the
    override was never applied. The scalar case where official simply wins
    over a third-party snapshot is a different situation and keeps its own
    non-blocking name, so the two cannot be confused.
    """

    report = reconcile_ruleset(load_ruleset("v1"))
    statuses = {str(row["status"]) for row in report["rows"]}
    assert "official_override_unapplied" not in statuses

    for row in report["rows"]:
        if row["status"] == "official_override_unapplied":
            assert row["simulator_value"] != row["official_value"]
            assert row in report["unresolved"]


def test_balance_pending_record_is_valid_and_named_by_the_ruleset() -> None:
    """The staleness gap must be auditable, not inferred from a date.

    The generated ruleset is a pinned snapshot, so any patch released after it
    is absent by construction. A reader previously had to compare
    ``generated_at`` against the outside world to notice. The record names the
    gap explicitly and the ruleset points at it.
    """

    from simulator.data_reconciliation import (
        BALANCE_PENDING_FILENAME,
        load_balance_pending,
    )

    record = load_balance_pending()
    assert record["kind"] == "simulator_balance_pending"
    assert isinstance(record["items"], list)
    # The record must state that it is a reminder rather than a source.
    assert "never be promoted" in record["instructions"]

    ruleset = load_ruleset("v1")
    assert ruleset.metadata["balance_pending_record"] == BALANCE_PENDING_FILENAME
    # Still not a release: the gap is recorded, not closed.
    assert ruleset.metadata["training_ready"] is False
    assert ruleset.metadata["training_blockers"]


def test_balance_pending_record_does_not_assert_unverified_values() -> None:
    """Nothing in the record may masquerade as applied simulator truth.

    Every entry is a reminder that a note exists and is unapplied. A value
    recorded here would be an unsourced override, which is exactly what the
    reconciliation gate exists to prevent.
    """

    from simulator.data_reconciliation import load_balance_pending

    record = load_balance_pending()
    for item in record["items"]:
        assert "status" in item, item
        assert item["status"] == "unapplied", item
        assert "source" in item or "note" in item, item
        # A recorded value would need an official override row to take effect;
        # the record must not carry one on its own.
        assert "value" not in item or item.get("applied_via_override") is None, item
