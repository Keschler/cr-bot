from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.ruleset import (
    DEFAULT_RULESET_ID,
    RulesetError,
    available_rulesets,
    calculate_content_hash,
    load_ruleset,
    ruleset_path,
)


def test_august_ruleset_is_pinned_level_11_hog_cycle() -> None:
    ruleset = load_ruleset()

    assert ruleset.ruleset_id == DEFAULT_RULESET_ID
    assert ruleset.level == 11
    assert ruleset.tick_us == 50_000
    assert ruleset.interaction_set == (
        "hog-rider",
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
    assert set(ruleset.towers) == {"princess-tower", "king-tower"}
    princess = ruleset.tower("princess-tower")
    king = ruleset.tower("king-tower")
    assert (princess.hitpoints, princess.damage) == (3_052, 109)
    assert (king.hitpoints, king.damage) == (4_824, 109)
    assert princess.provenance["level_11_stats"] == (
        "local-level11-hog-video-2026-08-12",
        "royaleapi-tower-princess-2024",
    )
    assert available_rulesets() == (DEFAULT_RULESET_ID,)
    assert "2026-08-04-roster" in available_rulesets(include_provisional=True)
    ruleset.verify_hash()


def test_aliases_resolve_to_stable_canonical_ids() -> None:
    ruleset = load_ruleset()

    assert ruleset.resolve_card_id("The Log") == "log"
    assert ruleset.resolve_card_id("the_log") == "log"
    assert ruleset.resolve_card_id("HogRider") == "hog-rider"
    assert ruleset.resolve_card_id("ice golem") == "ice-golem"
    assert ruleset.resolve_tower_id("king_tower") == "king-tower"
    with pytest.raises(KeyError, match="unknown card"):
        ruleset.card("hero-musketeer")


def test_official_spirit_override_wins_and_remains_a_testable_outcome() -> None:
    ruleset = load_ruleset()
    spirit = ruleset.card("ice-spirit")

    assert spirit.hitpoints == 215
    assert spirit.damage == 110
    assert spirit.mechanics["crown_tower_connection"] == "expected-no-unassisted-connection"
    assert spirit.mechanics["status"]["duration_us"] == 1_100_000
    assert spirit.provenance["hitpoints"] == ("official-august-2026",)
    assert spirit.provenance["freeze_duration_us"] == (
        "official-july-2025-ice-spirit",
    )
    assert ruleset.sources["official-august-2026"].confidence_tier == "A"
    assert any(item.field == "mechanics.crown_tower_connection" for item in spirit.uncertainties)


def test_spell_placement_classes_match_the_existing_policy_boundary() -> None:
    ruleset = load_ruleset()

    assert ruleset.card("fireball").mechanics["placement_class"] == "spell_anywhere"
    log = ruleset.card("log")
    assert log.mechanics["placement_class"] == "restricted_spell"
    assert log.provenance["placement_class"] == ("policy-grid-2026-08-11",)
    assert any(item.field == "mechanics.placement_class" for item in log.uncertainties)


def test_official_june_spell_crown_damage_is_applied_to_august_ruleset() -> None:
    ruleset = load_ruleset()
    source_id = "official-june-2026-spell-crown-damage"

    assert ruleset.card("fireball").crown_tower_damage == 172
    assert ruleset.card("log").crown_tower_damage == 35
    assert ruleset.card("fireball").provenance["crown_tower_damage"] == (source_id,)
    assert ruleset.card("log").provenance["crown_tower_damage"] == (source_id,)
    assert ruleset.sources[source_id].confidence_tier == "A"
    conflicts = {
        row["field"]: (row["lower_confidence_value"], row["resolved_value"])
        for row in ruleset.metadata["source_conflicts"]
    }
    assert conflicts["cards.fireball.crown_tower_damage"] == (207, 172)
    assert conflicts["cards.log.crown_tower_damage"] == (40, 35)
    assert conflicts["cards.ice-spirit.mechanics.status.duration_us"] == (
        1_200_000,
        1_100_000,
    )


def test_cannon_declares_linear_lifetime_decay_as_versioned_data() -> None:
    ruleset = load_ruleset()

    cannon = ruleset.card("cannon")
    assert cannon.damage == 202
    assert cannon.provenance["damage"] == ("official-april-2026-cannon",)
    assert cannon.mechanics["lifetime_decay"] == "linear_hp"
    assert cannon.mechanics["lifetime_start"] == "placement"
    assert cannon.mechanics["targetable_during_deploy"] is True
    assert any(item.field == "lifetime_us.start" for item in cannon.uncertainties)
    assert "lifetime_decay" not in ruleset.card("hog-rider").mechanics
    conflict = next(
        row
        for row in ruleset.metadata["source_conflicts"]
        if row["field"] == "cards.cannon.damage"
    )
    assert (conflict["lower_confidence_value"], conflict["resolved_value"]) == (
        212,
        202,
    )


def test_declared_troop_masses_match_structured_collision_data() -> None:
    ruleset = load_ruleset()

    assert {
        card_id: ruleset.card(card_id).mass
        for card_id in ("ice-golem", "musketeer", "hog-rider", "skeletons", "ice-spirit")
    } == {
        "ice-golem": 6,
        "musketeer": 5,
        "hog-rider": 4,
        "skeletons": 1,
        "ice-spirit": 1,
    }
    assert ruleset.card("cannon").mass == 0


def test_projectile_muzzle_offsets_match_structured_game_data() -> None:
    ruleset = load_ruleset()

    assert {
        "cannon": ruleset.card("cannon").projectile.start_radius_mtile,
        "musketeer": ruleset.card("musketeer").projectile.start_radius_mtile,
        "princess-tower": ruleset.tower("princess-tower").projectile.start_radius_mtile,
        "king-tower": ruleset.tower("king-tower").projectile.start_radius_mtile,
    } == {
        "cannon": 1_000,
        "musketeer": 450,
        "princess-tower": 300,
        "king-tower": 750,
    }
    assert ruleset.card("ice-spirit").projectile.start_radius_mtile == 0
    assert ruleset.card("fireball").projectile.start_radius_mtile == 0
    assert ruleset.card("log").projectile.start_radius_mtile == 0


def test_current_video_resolves_hog_and_musketeer_damage_rounding() -> None:
    ruleset = load_ruleset()
    hog = ruleset.card("hog-rider")
    musketeer = ruleset.card("musketeer")
    source_id = "local-level11-spell3-tower-hp-2026-08-12"

    assert (hog.damage, musketeer.damage) == (317, 217)
    assert hog.provenance["damage"][0] == source_id
    assert hog.provenance["attack_interval_us"][0] == source_id
    assert musketeer.provenance["damage"][0] == source_id
    assert ruleset.sources[source_id].confidence_tier == "C"
    assert not any(item.field == "damage" for item in musketeer.uncertainties)
    conflicts = {
        row["field"]: (row["lower_confidence_value"], row["resolved_value"])
        for row in ruleset.metadata["source_conflicts"]
    }
    assert conflicts["cards.hog-rider.damage"] == (318, 317)
    assert conflicts["cards.musketeer.damage"] == (218, 217)


def test_high_frame_rate_video_pins_musketeer_repeat_interval() -> None:
    ruleset = load_ruleset()
    musketeer = ruleset.card("musketeer")
    source_id = "local-level11-spell2-musketeer-timing-2026-08-12"

    assert musketeer.attack_interval_us == 1_000_000
    assert musketeer.provenance["attack_interval_us"][0] == source_id
    assert ruleset.sources[source_id].confidence_tier == "C"


def test_isolated_tracks_calibrate_hog_movement_speed() -> None:
    ruleset = load_ruleset()
    hog = ruleset.card("hog-rider")
    source_id = "local-level11-hog-movement-2026-08-12"

    assert hog.move_speed_mtile_per_s == 2_400
    assert hog.provenance["move_speed_mtile_per_s"][0] == source_id
    assert ruleset.sources[source_id].confidence_tier == "C"
    speed_uncertainty = next(
        item for item in hog.uncertainties if item.field == "move_speed_mtile_per_s"
    )
    assert speed_uncertainty.impact == "medium"


def test_independent_tracks_calibrate_log_rolling_speed() -> None:
    ruleset = load_ruleset()
    log = ruleset.card("log")
    sources = log.provenance["projectile_speed_conversion"]

    assert log.projectile.speed_mtile_per_s == 4_000
    assert sources[:2] == (
        "local-level11-spell-log-motion-2026-08-12",
        "local-level11-spell3-log-motion-2026-08-12",
    )
    assert all(ruleset.sources[source].confidence_tier == "C" for source in sources[:2])
    speed_uncertainty = next(
        item for item in log.uncertainties if item.field == "projectile.speed_mtile_per_s"
    )
    assert speed_uncertainty.impact == "medium"


def test_localized_casts_validate_combined_fireball_flight_timing() -> None:
    ruleset = load_ruleset()
    fireball = ruleset.card("fireball")
    source_id = "local-level11-spell-fireball-flight-2026-08-13"

    assert fireball.provenance["flight_timing"] == (source_id,)
    assert ruleset.sources[source_id].confidence_tier == "C"
    assert "127 ms MAE" in ruleset.sources[source_id].note


def test_skeleton_spawn_layout_is_pinned_to_local_level11_frames() -> None:
    skeletons = load_ruleset().card("skeletons")

    assert skeletons.mechanics["spawn_layout_mtile"] == (
        (0, 0),
        (-750, 500),
        (750, 500),
    )
    assert skeletons.provenance["spawn_layout_mtile"] == (
        "local-level11-hog-video-2026-08-12",
    )


def test_all_ruleset_numbers_are_integer_canonical_units() -> None:
    raw = json.loads(ruleset_path().read_text(encoding="utf-8"))

    def visit(value: object) -> None:
        assert not isinstance(value, float)
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)
    assert raw["units"] == {
        "time": "microseconds",
        "position": "milli-tiles",
        "elixir": "milli-elixir",
        "health": "integer-hitpoints",
        "damage": "integer-hitpoints",
        "multiplier": "permille",
    }


def test_loaded_nested_data_is_immutable() -> None:
    ruleset = load_ruleset()

    with pytest.raises(TypeError):
        ruleset.cards["new-card"] = ruleset.card("hog-rider")  # type: ignore[index]
    with pytest.raises(TypeError):
        ruleset.card("hog-rider").mechanics["building_only"] = False  # type: ignore[index]


def test_tampering_is_rejected_before_simulation(tmp_path: Path) -> None:
    raw = json.loads(ruleset_path().read_text(encoding="utf-8"))
    raw["cards"]["hog-rider"]["damage"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RulesetError, match="content hash mismatch"):
        load_ruleset(path)


def test_canonical_hash_excludes_only_hash_field() -> None:
    raw = json.loads(ruleset_path().read_text(encoding="utf-8"))
    declared = raw["content_hash"]

    assert calculate_content_hash(raw) == declared
    raw["content_hash"] = "sha256:" + "f" * 64
    assert calculate_content_hash(raw) == declared
    raw["level"] = 12
    assert calculate_content_hash(raw) != declared
