from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from simulator.physical_lab import (
    EvidenceSplit,
    EvidenceStatus,
    NormalizedEvent,
    ObservationCertainty,
    ObservationManifest,
    hog_cannon_probe,
    run_simulator_replay,
)
from simulator.physical_lab.comparison import (
    _first_divergence,
    _follow_up,
    _minimized_scenario,
    _stable_sim_event,
    compare_observation_to_replay,
)
from simulator.physical_lab.schema import canonical_hash


_SUPPORTED_PARAMETERS = (
    ("targeting", "building_acquisition_radius_mtile"),
    ("projectile", "projectile_speed_mtile_per_s"),
    ("damage", "damage_or_crown_damage"),
    ("lifecycle", "spawn_delay_us"),
    ("status", "status_duration_us"),
    ("timing", "first_hit_delay_us"),
)
_EXPECTED_OFFSETS = (-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)


@pytest.mark.parametrize("subsystem, parameter", _SUPPORTED_PARAMETERS)
def test_supported_divergence_returns_the_bounded_ordered_sweep(
    subsystem: str,
    parameter: str,
) -> None:
    follow_up = _follow_up({"subsystem": subsystem, "confidence": 0.91})

    assert follow_up == {
        "parameter": parameter,
        "strategy": "local_sweep",
        "offsets": list(_EXPECTED_OFFSETS),
        "reason": "first decision-relevant divergence; keep held-out evidence sealed",
    }
    assert len(follow_up["offsets"]) == 9  # type: ignore[index]
    assert min(follow_up["offsets"]) >= -0.50  # type: ignore[arg-type]
    assert max(follow_up["offsets"]) <= 0.50  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "divergence",
    (
        {"subsystem": "future_mechanic", "confidence": 0.99},
        {"subsystem": "targeting"},
        {"subsystem": "targeting", "confidence": 0.49},
        {"subsystem": "targeting", "confidence": float("nan")},
        {"subsystem": [], "confidence": 0.99},
    ),
)
def test_unsupported_or_unverified_divergence_has_no_follow_up(
    divergence: dict[str, object],
) -> None:
    assert _follow_up(divergence) is None


def test_unclassified_event_does_not_fall_through_to_timing() -> None:
    event = NormalizedEvent(
        event_id="event-unknown",
        kind="card_mirrored",
        video_time_us=1_000,
        match_time_us=1_000,
        confidence=0.99,
        certainty=ObservationCertainty.DIRECT,
        source_frame_indices=(1,),
    )

    divergence = _first_divergence((event,), (), tick_us=1_000)

    assert divergence is not None
    assert divergence["subsystem"] is None
    assert _follow_up(divergence) is None


def test_comparison_maps_observed_spawn_to_simulator_entity_creation() -> None:
    event = NormalizedEvent(
        event_id="spawn",
        kind="unit_spawn_observed",
        video_time_us=1_000,
        match_time_us=1_000,
        confidence=0.99,
        certainty=ObservationCertainty.DIRECT,
        source_frame_indices=(1,),
        card_id="hog-rider",
        owner="A",
    )
    simulated = (
        {"kind": "match_started", "tick": 0, "values": {}},
        {"kind": "entity_created", "tick": 1, "card_id": "hog-rider", "owner": 0},
        {"kind": "entity_deployed", "tick": 2, "card_id": "hog-rider", "owner": 0},
    )

    # _first_divergence receives stable simulator rows in production; this
    # assertion covers the semantic mapping and bookkeeping filtering with
    # the same wire shape directly.
    assert _first_divergence((event,), simulated, tick_us=1_000) is None


def test_tentative_observation_is_excluded_from_first_divergence() -> None:
    event = NormalizedEvent(
        event_id="tentative-death",
        kind="unit_disappearance_observed",
        video_time_us=1_000,
        match_time_us=1_000,
        confidence=0.70,
        certainty=ObservationCertainty.TENTATIVE,
        source_frame_indices=(1,),
        card_id="hog-rider",
        owner="A",
    )

    assert _first_divergence(
        (event,),
        ({"kind": "entity_died", "tick": 1, "card_id": "hog-rider", "owner": 0},),
        tick_us=1_000,
    ) is None


def test_simulator_damage_to_tower_is_normalized_as_tower_damage() -> None:
    event = SimpleNamespace(
        kind="damage_applied",
        tick=292,
        data={
            "damage": 317,
            "hp_after": 2735,
            "source_card_id": "hog-rider",
            "source_uid": 7,
            "target_uid": 6,
        },
    )
    replay = SimpleNamespace(
        final_state=SimpleNamespace(
            entities={
                6: SimpleNamespace(
                    kind="tower",
                    owner=1,
                    card_id="princess-tower",
                )
            }
        )
    )

    row = _stable_sim_event(event, replay)

    assert row["kind"] == "tower_damage_observed"
    assert row["owner"] == 1
    assert row["source_card_id"] == "hog-rider"
    assert row["target_card_id"] == "princess-tower"
    assert row["values"]["tower_damage"] == 317


def test_minimized_scenario_has_no_legacy_follow_up_tail() -> None:
    source = inspect.getsource(_minimized_scenario)

    assert "parameter_by_subsystem" not in source
    assert source.count("return scenario") == 1


def _comparison_report_for_event(kind: str):
    spec = hog_cannon_probe()
    replay = run_simulator_replay(spec, action_times={"deploy-cannon": 17_000})
    observation = ObservationManifest(
        run_id="comparison-follow-up",
        experiment_hash=spec.experiment_hash(),
        capture_group_id=spec.capture_group_id,
        evidence_split=EvidenceSplit.CALIBRATION,
        status=EvidenceStatus.CALIBRATED_ONLY,
        events=(
            NormalizedEvent(
                event_id="event-1",
                kind=kind,
                video_time_us=1_000,
                match_time_us=1_000,
                confidence=0.91,
                certainty=ObservationCertainty.DIRECT,
                source_frame_indices=(1,),
                card_id="hog-rider",
                owner="A",
                target_role="princess-tower",
            ),
        ),
    )
    return compare_observation_to_replay(observation, replay)


def test_supported_report_is_hash_stable_and_contains_follow_up() -> None:
    first = _comparison_report_for_event("target_changed")
    second = _comparison_report_for_event("target_changed")

    assert first.eligible
    assert first.follow_up is not None
    assert first.follow_up["parameter"] == "building_acquisition_radius_mtile"
    assert first.follow_up["offsets"] == list(_EXPECTED_OFFSETS)
    assert first.comparison_hash == second.comparison_hash

    payload = first.to_dict()
    declared_hash = payload.pop("comparison_hash")
    # ComparisonReport adds the wire-format kind after hashing the canonical
    # comparison payload; preserve that established hash boundary.
    payload.pop("kind")
    assert declared_hash == canonical_hash(payload)


def test_unsupported_report_is_ineligible_and_emits_no_follow_up() -> None:
    report = _comparison_report_for_event("card_mirrored")

    assert not report.eligible
    assert report.follow_up is None
    assert "unsupported" in " ".join(report.rejection_reasons)
