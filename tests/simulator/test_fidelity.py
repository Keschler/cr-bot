from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from simulator.fidelity import (
    ComparisonTolerance,
    DatasetSplit,
    ObservationEvidence,
    ObservedEvent,
    ObservedMechanicSample,
    ObservedTrace,
    SimulatedMechanicSample,
    build_fidelity_report,
    compare_sample,
    compare_samples,
    compare_trace,
    compare_traces,
    normalize_event,
)
from simulator.events import SimEvent


EVIDENCE = ObservationEvidence(
    source_id="capture-60fps-001",
    method="offline_detector_ensemble",
    group_id="match-001",
    confidence=0.98,
)


def _observed(
    sample_id: str,
    value: float | str,
    *,
    mechanic: str = "hog_movement",
    split: DatasetSplit = DatasetSplit.HELDOUT,
    absolute: float = 0.0,
    relative: float = 0.0,
    tick: int | None = None,
    tick_tolerance: int = 0,
) -> ObservedMechanicSample:
    return ObservedMechanicSample(
        sample_id=sample_id,
        mechanic=mechanic,
        split=split,
        observed_value=value,
        evidence=EVIDENCE,
        tolerance=ComparisonTolerance(
            absolute=absolute,
            relative=relative,
            ticks=tick_tolerance,
        ),
        observed_tick=tick,
    )


def test_numeric_comparison_respects_absolute_relative_and_tick_tolerances() -> None:
    absolute = compare_sample(
        _observed("absolute", 10.0, absolute=0.2, tick=100, tick_tolerance=2),
        SimulatedMechanicSample("absolute", 10.2, tick=102),
    )
    relative = compare_sample(
        _observed("relative", 100.0, relative=0.02),
        SimulatedMechanicSample("relative", 101.9),
    )
    late = compare_sample(
        _observed("late", 10.0, absolute=1.0, tick=100, tick_tolerance=2),
        SimulatedMechanicSample("late", 10.0, tick=103),
    )

    assert absolute.agrees
    assert absolute.absolute_error == pytest.approx(0.2)
    assert absolute.tick_absolute_error == 2
    assert relative.agrees
    assert not late.agrees
    assert late.reason == "tick_outside_tolerance"


def test_categorical_comparison_is_exact_and_missing_results_are_failures() -> None:
    observed = _observed("target", "cannon", mechanic="hog_target")

    mismatch = compare_sample(observed, SimulatedMechanicSample("target", "tower"))
    missing = compare_sample(observed, None)

    assert not mismatch.agrees
    assert mismatch.absolute_error is None
    assert mismatch.reason == "value_outside_tolerance"
    assert not missing.agrees
    assert missing.reason == "missing_simulation"


def test_boolean_measurements_do_not_compare_equal_to_integer_flags() -> None:
    observed = ObservedMechanicSample(
        sample_id="connected",
        mechanic="ice_spirit_tower_connection",
        split=DatasetSplit.HELDOUT,
        observed_value=True,
        evidence=EVIDENCE,
    )

    comparison = compare_sample(
        observed,
        SimulatedMechanicSample("connected", 1),
    )

    assert not comparison.agrees


def test_compare_samples_pairs_deterministically_and_rejects_duplicate_ids() -> None:
    observations = [_observed("b", 2.0), _observed("a", 1.0)]
    comparisons = compare_samples(
        observations,
        [SimulatedMechanicSample("a", 1.0)],
    )

    assert [item.sample_id for item in comparisons] == ["a", "b"]
    assert comparisons[0].agrees
    assert comparisons[1].reason == "missing_simulation"
    with pytest.raises(ValueError, match="duplicate observed sample"):
        compare_samples([observations[0], observations[0]], [])


@dataclass(frozen=True, slots=True)
class DamageApplied:
    tick: int
    target_uid: int
    amount: int


def test_event_normalization_accepts_dicts_and_typed_records() -> None:
    dictionary = normalize_event(
        {"tick": 4, "type": "target_changed", "values": {"target_uid": 7}}
    )
    typed = normalize_event(DamageApplied(tick=5, target_uid=7, amount=317))
    engine_event = normalize_event(
        SimEvent.create(6, 3, "damage_applied", target_uid=8, amount=316)
    )

    assert dictionary.kind == "target_changed"
    assert dictionary.values == {"target_uid": 7}
    assert typed.kind == "damage_applied"
    assert typed.values == {"target_uid": 7, "amount": 317}
    assert engine_event.kind == "damage_applied"
    assert engine_event.values == {"sequence": 3, "target_uid": 8, "amount": 316}


def test_trace_comparison_uses_observation_tolerances_and_ignores_other_kinds() -> None:
    observed = ObservedTrace(
        trace_id="hog-hit-1",
        mechanic="hog_attack_timing",
        split=DatasetSplit.HELDOUT,
        evidence=EVIDENCE,
        events=(
            ObservedEvent(
                tick=100,
                kind="damage_applied",
                values={"amount": 317},
                tick_tolerance=2,
                value_tolerances={"amount": ComparisonTolerance(absolute=1)},
            ),
        ),
    )

    comparison = compare_trace(
        observed,
        [
            {"tick": 90, "kind": "movement", "x": 8_000},
            DamageApplied(tick=102, target_uid=9, amount=318),
        ],
    )

    assert comparison.agrees
    assert comparison.matched_event_count == 1
    assert comparison.simulated_event_count == 1
    assert comparison.divergence_tick is None


def test_trace_reports_first_decision_relevant_divergence_tick() -> None:
    observed = ObservedTrace(
        trace_id="pull-1",
        mechanic="hog_cannon_pull",
        split=DatasetSplit.HELDOUT,
        evidence=EVIDENCE,
        events=(
            ObservedEvent(30, "target_changed", {"target_uid": 5}),
            ObservedEvent(80, "damage_applied", {"target_uid": 5}),
        ),
    )

    comparison = compare_trace(
        observed,
        [
            {"tick": 30, "kind": "target_changed", "target_uid": 9},
            {"tick": 80, "kind": "damage_applied", "target_uid": 5},
        ],
    )

    assert not comparison.agrees
    assert comparison.divergence_tick == 30
    assert comparison.reason == "event_field_outside_tolerance:target_uid"
    assert comparison.matched_event_count == 0


def test_empty_observation_can_assert_that_an_event_does_not_occur() -> None:
    no_connection = ObservedTrace(
        trace_id="spirit-no-connect",
        mechanic="ice_spirit_tower_connection",
        split=DatasetSplit.HELDOUT,
        evidence=EVIDENCE,
        events=(),
        included_event_kinds=frozenset({"damage_applied"}),
    )

    comparison = compare_trace(
        no_connection,
        [{"tick": 40, "kind": "damage_applied", "target_uid": "princess-tower"}],
    )

    assert not comparison.agrees
    assert comparison.divergence_tick == 40
    assert comparison.reason == "unexpected_simulator_event"


def test_compare_traces_treats_an_absent_simulator_trace_as_empty() -> None:
    observed = ObservedTrace(
        trace_id="missing",
        mechanic="projectile_timing",
        split=DatasetSplit.HELDOUT,
        evidence=EVIDENCE,
        events=(ObservedEvent(12, "projectile_spawned"),),
    )

    comparison = compare_traces([observed], {})[0]

    assert comparison.reason == "missing_simulator_event"
    assert comparison.divergence_tick == 12


def test_report_is_heldout_only_and_contains_per_mechanic_metrics_and_ci() -> None:
    observations = [
        _observed("h1", 1.0, absolute=0.25),
        _observed("h2", 2.0, absolute=0.25),
        _observed("h3", 3.0, absolute=0.25),
        _observed(
            "calibration",
            100.0,
            split=DatasetSplit.CALIBRATION,
            absolute=100.0,
        ),
    ]
    samples = compare_samples(
        observations,
        [
            SimulatedMechanicSample("h1", 1.1),
            SimulatedMechanicSample("h2", 2.2),
            SimulatedMechanicSample("h3", 4.0),
            SimulatedMechanicSample("calibration", 100.0),
        ],
    )
    divergent_trace = ObservedTrace(
        trace_id="trace-b",
        mechanic="hog_movement",
        split=DatasetSplit.HELDOUT,
        evidence=EVIDENCE,
        events=(ObservedEvent(42, "target_changed", {"target_uid": 1}),),
    )
    traces = [
        compare_trace(
            divergent_trace,
            [{"tick": 42, "kind": "target_changed", "target_uid": 2}],
        )
    ]

    report = build_fidelity_report(
        ruleset_id="2026-08-04-l11",
        sample_comparisons=samples,
        trace_comparisons=traces,
    ).to_dict()
    metric = report["mechanics"]["hog_movement"]["samples"]

    assert report["dataset_split"] == "heldout"
    assert report["canonical_units"]["position"] == "milli_tile"
    assert "fallible observed measurements" in report["evidence_notice"]
    assert metric["count"] == 3
    assert metric["agreement_count"] == 2
    assert metric["mae"] == pytest.approx((0.1 + 0.2 + 1.0) / 3)
    assert metric["p95_absolute_error"] == 1.0
    interval = metric["agreement_confidence_interval"]
    assert interval["lower"] < 2 / 3 < interval["upper"]
    assert report["excluded_by_split"] == {
        "calibration": {"sample_comparisons": 1, "trace_comparisons": 0}
    }
    assert report["overall"]["traces"]["first_divergence_tick"] == 42
    assert report["trace_divergences"] == [
        {
            "trace_id": "trace-b",
            "mechanic": "hog_movement",
            "observation_source_id": "capture-60fps-001",
            "observation_group_id": "match-001",
            "observation_method": "offline_detector_ensemble",
            "divergence_tick": 42,
            "reason": "event_field_outside_tolerance:target_uid",
        }
    ]


def test_report_json_is_deterministic_valid_and_atomically_writable(tmp_path) -> None:
    comparison = compare_sample(
        _observed("one", 1.0),
        SimulatedMechanicSample("one", 1.0),
    )
    report = build_fidelity_report(
        ruleset_id="ruleset-sha256:abc",
        sample_comparisons=[comparison],
    )

    first = report.to_json()
    second = report.to_json()
    output = tmp_path / "nested" / "fidelity.json"
    report.write_json(output)

    assert first == second == output.read_text(encoding="utf-8")
    assert json.loads(first)["schema_version"] == 1


def test_report_rejects_invalid_tick_duration() -> None:
    with pytest.raises(ValueError, match="tick_us"):
        build_fidelity_report(ruleset_id="ruleset", tick_us=0)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: ComparisonTolerance(absolute=float("nan")), "absolute tolerance"),
        (
            lambda: ObservationEvidence("source", "method", confidence=1.1),
            "confidence",
        ),
        (lambda: SimulatedMechanicSample("sample", float("inf")), "finite"),
    ],
)
def test_non_finite_or_invalid_measurements_are_rejected(constructor, message) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()
