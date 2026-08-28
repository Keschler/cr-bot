from __future__ import annotations

from pathlib import Path

from simulator.physical_lab.evaluation import evaluate_stored_cases, write_stored_evaluation
from simulator.physical_lab.extraction import (
    _apply_event_identity,
    _canonical_position_mtile,
    _extractor_command,
    _internal_match_us,
    _merge_cross_phone_events,
    _merge_singleton_action_entities,
    _reconcile_singleton_lifecycle_events,
    _run_action_events,
    _select_primary_view_entities,
)
from simulator.physical_lab.identity import KnownCardIdentity, KnownPlacement
from simulator.physical_lab.replay import action_match_time_us


def test_extractor_preserves_native_phone_geometry_and_selects_b_profile() -> None:
    command_a, transform_a, profile_a = _extractor_command(
        Path("/repo"), Path("/tmp/a.mp4"), Path("/tmp/a.pkl.gz"), side="A", sample_interval_s=0.1
    )
    command_b, transform_b, profile_b = _extractor_command(
        Path("/repo"), Path("/tmp/b.mp4"), Path("/tmp/b.pkl.gz"), side="B", sample_interval_s=0.1
    )

    assert "--no-normalize" not in command_a
    assert "--alternative-rois" not in command_a
    assert transform_a == "native_1080x2400_identity"
    assert profile_a == "standard_asus"
    assert "--no-normalize" not in command_b
    assert "--alternative-rois" in command_b
    assert transform_b == "normalized_1080x2400_from_native_1080x2280"
    assert profile_b == "alternative_samsung_on_calibrated_canvas"


def test_known_placement_overrides_wrong_detector_class() -> None:
    identity = KnownCardIdentity(
        decks={"B": ("hog-rider", "cannon")},
        placements=(
            KnownPlacement(
                action_id="b-hog",
                owner="B",
                card_id="hog-rider",
                match_time_us=1_000_000,
            ),
        ),
    )

    decision = identity.resolve(
        owner="B",
        raw_card_id="hunter",
        match_time_us=1_100_000,
    )

    assert decision.accepted is True
    assert decision.card_id == "hog-rider"
    assert decision.raw_card_id == "hunter"
    assert decision.source == "placement_receipt_override"
    assert decision.matched_action_id == "b-hog"


def test_known_deck_rejects_unreceipted_impossible_detector_class() -> None:
    identity = KnownCardIdentity(decks={"B": ("hog-rider", "cannon")})

    decision = identity.resolve(
        owner="B",
        raw_card_id="hunter",
        match_time_us=6_238_049,
    )

    assert decision.accepted is False
    assert decision.card_id is None
    assert "declared deck" in decision.reason


def test_authoritative_actions_reject_declared_but_unplayed_card() -> None:
    identity = KnownCardIdentity(
        decks={"B": ("hog-rider", "musketeer")},
        placements=(),
        placements_authoritative=True,
    )

    decision = identity.resolve(
        owner="B",
        raw_card_id="hog-rider",
        match_time_us=6_000_000,
    )

    assert decision.accepted is False
    assert "authoritative action log" in decision.reason


def test_authoritative_placement_lineage_outlives_identity_override_window() -> None:
    identity = KnownCardIdentity(
        decks={"A": ("hog-rider",)},
        placements=(
            KnownPlacement(
                action_id="deploy-hog",
                owner="A",
                card_id="hog-rider",
                match_time_us=1_000_000,
            ),
        ),
        placements_authoritative=True,
    )

    decision = identity.resolve(
        owner="A",
        raw_card_id="hog-rider",
        match_time_us=20_000_000,
    )

    assert decision.accepted is True
    assert decision.source == "placement_lineage"
    assert decision.matched_action_id == "deploy-hog"


def test_phone_b_position_is_rotated_into_phone_a_canonical_frame() -> None:
    assert _canonical_position_mtile("A", 3_324, 21_408) == (3_324, 21_408)
    assert _canonical_position_mtile("B", 14_676, 10_592) == (3_324, 21_408)


def test_event_identity_filter_keeps_rejection_auditable() -> None:
    identity = KnownCardIdentity(decks={"B": ("hog-rider",)})
    event = {
        "event_id": "B-spawn-1",
        "kind": "unit_spawn_observed",
        "owner": "B",
        "card_id": "hunter",
        "match_time_us": 6_238_049,
        "confidence": 0.804,
        "values": {},
    }

    filtered, rejection = _apply_event_identity(
        event,
        identity_context=identity,
    )

    assert filtered is None
    assert rejection is not None
    assert rejection["raw_card_id"] == "hunter"


def test_internal_match_time_uses_capture_and_battle_monotonic_receipts() -> None:
    assert _internal_match_us(
        2.0,
        battle_start_video_s=0.0,
        capture_start_monotonic_us=1_000_000,
        battle_start_monotonic_us=1_500_000,
    ) == 1_500_000


def test_internal_match_time_applies_device_alignment_offset() -> None:
    assert _internal_match_us(
        2.0,
        battle_start_video_s=0.0,
        capture_start_monotonic_us=1_000_000,
        battle_start_monotonic_us=1_500_000,
        stream_offset_us=125_000,
    ) == 1_375_000


def test_action_time_prefers_placement_receipt_over_post_capture_timestamp() -> None:
    run = {"clock_provenance": {"battle_start_monotonic_us": 1_000_000}}
    action = {
        "actual_match_time_us": 9_000_000,
        "placement_receipt": {"accepted": True, "completed_at_monotonic_us": 1_750_000},
    }
    assert action_match_time_us(run, action) == 750_000


def test_runner_action_receipts_are_direct_timing_rows() -> None:
    events = _run_action_events(
        {
            "actions": [
                {
                    "action_id": "deploy-hog",
                    "side": "A",
                    "card_id": "hog-rider",
                    "card_slot": 0,
                    "arena_cell": [3, 20],
                    "accepted": True,
                    "actual_match_time_us": 123_456,
                }
            ]
        },
        stream_payloads={"A": {"timeline": []}},
    )
    assert events[0]["certainty"] == "direct"
    assert events[0]["match_time_us"] == 123_456
    assert events[0]["card_id"] == "hog-rider"


def test_runner_action_events_use_placement_receipt_time_when_available() -> None:
    events = _run_action_events(
        {
            "clock_provenance": {"battle_start_monotonic_us": 1_000_000},
            "actions": [
                {
                    "action_id": "deploy-hog",
                    "side": "A",
                    "card_id": "hog-rider",
                    "card_slot": 0,
                    "arena_cell": [3, 20],
                    "accepted": True,
                    "actual_match_time_us": 9_000_000,
                    "placement_receipt": {"accepted": True, "completed_at_monotonic_us": 1_750_000},
                }
            ],
        },
        stream_payloads={"A": {"timeline": []}},
    )
    assert events[0]["match_time_us"] == 750_000


def test_cross_phone_corroboration_merges_overlapping_tentative_transition() -> None:
    event_a = {
        "event_id": "A-death-1",
        "kind": "unit_disappearance_observed",
        "card_id": "hog-rider",
        "owner": "A",
        "video_time_us": 1_000,
        "match_time_us": 1_000,
        "confidence": 0.70,
        "certainty": "tentative",
        "source_frame_indices": [10],
        "evidence_refs": ["capture:A:10"],
        "values": {"confirmation_state": "tentative"},
    }
    event_b = {
        **event_a,
        "event_id": "B-death-1",
        "video_time_us": 1_100,
        "match_time_us": 1_100,
        "source_frame_indices": [11],
        "evidence_refs": ["capture:B:11"],
    }

    merged = _merge_cross_phone_events({"A": [event_a], "B": [event_b]})

    assert len(merged) == 1
    assert merged[0]["certainty"] == "inferred"
    assert merged[0]["values"]["cross_phone_corroborated"] is True
    assert merged[0]["values"]["cross_phone_sides"] == "A,B"
    assert merged[0]["source_frame_indices"] == [10, 11]


def test_cross_phone_corroboration_keeps_non_overlapping_events_separate() -> None:
    base = {
        "kind": "unit_spawn_observed",
        "card_id": "hog-rider",
        "owner": "A",
        "video_time_us": 1_000,
        "match_time_us": 1_000,
        "confidence": 0.70,
        "certainty": "inferred",
        "source_frame_indices": [10],
        "evidence_refs": ["capture:A:10"],
        "values": {},
    }
    merged = _merge_cross_phone_events(
        {"A": [{**base, "event_id": "A-spawn-1"}], "B": [{**base, "event_id": "B-spawn-1", "match_time_us": 1_000_000, "video_time_us": 1_000_000}]}
    )

    assert len(merged) == 2
    assert not any(event["values"].get("cross_phone_corroborated") for event in merged)


def test_singleton_action_stitches_primary_view_track_fragments() -> None:
    identity = KnownCardIdentity(
        placements=(KnownPlacement("deploy-hog", "A", "hog-rider", 1_000_000),),
        placements_authoritative=True,
    )
    streams = {
        "A": {
            "entities": [
                {
                    "stable_observation_id": "A-1",
                    "card_id": "hog-rider",
                    "owner": "A",
                    "confidence": 0.8,
                    "samples": [
                        {
                            "frame_index": 10,
                            "video_time_us": 1_100_000,
                            "match_time_us": 1_100_000,
                            "x_mtile": 3_000,
                            "y_mtile": 20_000,
                            "confidence": 0.8,
                            "source_capture_id": "capture-A",
                            "uncertainty_us": 100_000,
                        }
                    ],
                },
                {
                    "stable_observation_id": "A-2",
                    "card_id": "hog-rider",
                    "owner": "A",
                    "confidence": 0.9,
                    "samples": [
                        {
                            "frame_index": 20,
                            "video_time_us": 2_100_000,
                            "match_time_us": 2_100_000,
                            "x_mtile": 3_100,
                            "y_mtile": 18_000,
                            "confidence": 0.9,
                            "source_capture_id": "capture-A",
                            "uncertainty_us": 100_000,
                        }
                    ],
                },
            ]
        },
        "B": {
            "entities": [
                {
                    "stable_observation_id": "B-view",
                    "card_id": "hog-rider",
                    "owner": "A",
                    "confidence": 0.95,
                    "samples": [
                        {
                            "frame_index": 11,
                            "video_time_us": 1_150_000,
                            "match_time_us": 1_150_000,
                            "x_mtile": 3_050,
                            "y_mtile": 19_900,
                            "confidence": 0.95,
                            "source_capture_id": "capture-B",
                            "uncertainty_us": 100_000,
                        }
                    ],
                }
            ]
        },
    }

    primary = _select_primary_view_entities(streams)
    merged = _merge_singleton_action_entities(
        primary,
        identity_context=identity,
        singleton_card_ids=frozenset({"hog-rider"}),
    )

    assert len(primary) == 2
    assert len(merged) == 1
    assert merged[0]["stable_observation_id"] == "action-deploy-hog-entity-000"
    assert len(merged[0]["samples"]) == 2


def test_singleton_lifecycle_keeps_first_spawn_and_final_death() -> None:
    identity = KnownCardIdentity(
        placements=(KnownPlacement("deploy-hog", "A", "hog-rider", 1_000_000),),
        placements_authoritative=True,
    )
    events = [
        {
            "event_id": "spawn-1",
            "kind": "unit_spawn_observed",
            "owner": "A",
            "card_id": "hog-rider",
            "match_time_us": 1_100_000,
            "video_time_us": 1_100_000,
            "certainty": "inferred",
            "values": {},
        },
        {
            "event_id": "death-1",
            "kind": "unit_disappearance_observed",
            "owner": "A",
            "card_id": "hog-rider",
            "match_time_us": 2_000_000,
            "video_time_us": 2_000_000,
            "certainty": "inferred",
            "values": {},
        },
        {
            "event_id": "spawn-2",
            "kind": "unit_spawn_observed",
            "owner": "A",
            "card_id": "hog-rider",
            "match_time_us": 2_100_000,
            "video_time_us": 2_100_000,
            "certainty": "inferred",
            "values": {},
        },
        {
            "event_id": "death-2",
            "kind": "unit_disappearance_observed",
            "owner": "A",
            "card_id": "hog-rider",
            "match_time_us": 5_000_000,
            "video_time_us": 5_000_000,
            "certainty": "inferred",
            "values": {},
        },
    ]

    reconciled = _reconcile_singleton_lifecycle_events(
        events,
        identity_context=identity,
        singleton_card_ids=frozenset({"hog-rider"}),
    )

    assert reconciled[0]["certainty"] == "inferred"
    assert reconciled[1]["certainty"] == "tentative"
    assert reconciled[2]["certainty"] == "tentative"
    assert reconciled[3]["certainty"] == "inferred"


def test_stored_evaluation_is_write_once(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    payload = evaluate_stored_cases(cases_root)
    destination = tmp_path / "evaluation.json"
    write_stored_evaluation(destination, payload)
    write_stored_evaluation(destination, payload)
    assert destination.is_file()
