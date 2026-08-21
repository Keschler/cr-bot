"""Tolerance-aware sim-to-observation fidelity reporting.

This module deliberately calls its inputs *observations*, not ground truth.
Video-derived measurements can be quantized, occluded, or incorrectly
classified.  Callers must therefore attach provenance and tolerances to every
observation.  Confidence is retained as evidence metadata; it is never used to
silently weight a score.

The default report split is ``heldout``.  Inputs from calibration, validation,
or regression splits are counted as excluded and cannot leak into held-out
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias


Scalar: TypeAlias = int | float | str | bool

_EVIDENCE_NOTICE = (
    "Metrics compare simulator output with fallible observed measurements; "
    "they do not establish exact game truth. Wilson intervals are sample-level "
    "descriptives and do not assume samples from one capture group are independent."
)
_MISSING = object()


class DatasetSplit(str, Enum):
    """Purpose assigned before an observed sample is evaluated."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"
    REGRESSION = "regression"
    HELDOUT = "heldout"


def _as_split(value: DatasetSplit | str) -> DatasetSplit:
    try:
        return value if isinstance(value, DatasetSplit) else DatasetSplit(value)
    except ValueError as exc:
        choices = ", ".join(split.value for split in DatasetSplit)
        raise ValueError(f"unknown dataset split {value!r}; expected one of {choices}") from exc


def _require_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_tick(value: int | None, field_name: str) -> None:
    if value is None:
        return
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer tick")


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


def _validate_scalar(value: object, field_name: str) -> None:
    if not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"{field_name} must be a bool, int, float, or str")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _immutable_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ObservationEvidence:
    """Provenance for a fallible observed measurement.

    ``confidence`` describes the extraction system's confidence, not the
    probability that the observation is game truth.  Reports do not use it as
    a statistical weight.
    """

    source_id: str
    method: str
    group_id: str | None = None
    confidence: float | None = None
    notes: str | None = None
    media_hash: str | None = None
    frame_start: int | None = None
    frame_end: int | None = None

    def __post_init__(self) -> None:
        _require_name(self.source_id, "source_id")
        _require_name(self.method, "method")
        if self.group_id is not None:
            _require_name(self.group_id, "group_id")
        if self.media_hash is not None:
            _require_name(self.media_hash, "media_hash")
        _require_tick(self.frame_start, "frame_start")
        _require_tick(self.frame_end, "frame_end")
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and self.frame_end < self.frame_start
        ):
            raise ValueError("frame_end must not precede frame_start")
        if self.confidence is not None:
            if (
                type(self.confidence) not in (int, float)
                or not math.isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class ComparisonTolerance:
    """Allowed measurement error for a value and its event time.

    Numeric values agree when their absolute error is at most the larger of
    ``absolute`` and ``relative * abs(observed)``.  Non-numeric values use exact
    equality.  Tick tolerance applies only when the observation has a tick.
    """

    absolute: float = 0.0
    relative: float = 0.0
    ticks: int = 0

    def __post_init__(self) -> None:
        for name in ("absolute", "relative"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} tolerance must be finite and non-negative")
        if type(self.ticks) is not int or self.ticks < 0:
            raise ValueError("tick tolerance must be a non-negative integer")

    def allowed_error(self, observed: int | float) -> float:
        return max(float(self.absolute), float(self.relative) * abs(float(observed)))


@dataclass(frozen=True, slots=True)
class ObservedMechanicSample:
    """One independently sourced observation of a mechanic."""

    sample_id: str
    mechanic: str
    split: DatasetSplit
    observed_value: Scalar
    evidence: ObservationEvidence
    tolerance: ComparisonTolerance = field(default_factory=ComparisonTolerance)
    observed_tick: int | None = None

    def __post_init__(self) -> None:
        _require_name(self.sample_id, "sample_id")
        _require_name(self.mechanic, "mechanic")
        object.__setattr__(self, "split", _as_split(self.split))
        _validate_scalar(self.observed_value, "observed_value")
        _require_tick(self.observed_tick, "observed_tick")


@dataclass(frozen=True, slots=True)
class SimulatedMechanicSample:
    """Simulator output paired to an observation by ``sample_id``."""

    sample_id: str
    value: Scalar
    tick: int | None = None

    def __post_init__(self) -> None:
        _require_name(self.sample_id, "sample_id")
        _validate_scalar(self.value, "value")
        _require_tick(self.tick, "tick")


@dataclass(frozen=True, slots=True)
class SampleComparison:
    """Result of comparing one observation with simulator output."""

    observed: ObservedMechanicSample
    simulated: SimulatedMechanicSample | None
    agrees: bool
    absolute_error: float | None
    tick_absolute_error: int | None
    reason: str

    @property
    def sample_id(self) -> str:
        return self.observed.sample_id

    @property
    def mechanic(self) -> str:
        return self.observed.mechanic

    @property
    def split(self) -> DatasetSplit:
        return self.observed.split


def _numeric_error(observed: object, simulated: object) -> float | None:
    if not (_is_number(observed) and _is_number(simulated)):
        return None
    return abs(float(simulated) - float(observed))


def _value_agrees(
    observed: object,
    simulated: object,
    tolerance: ComparisonTolerance,
) -> bool:
    error = _numeric_error(observed, simulated)
    if error is None:
        if isinstance(observed, bool) or isinstance(simulated, bool):
            return type(observed) is type(simulated) and observed == simulated
        return observed == simulated
    allowed = tolerance.allowed_error(observed)  # type: ignore[arg-type]
    return error <= allowed or math.isclose(error, allowed, rel_tol=1e-12, abs_tol=1e-12)


def compare_sample(
    observed: ObservedMechanicSample,
    simulated: SimulatedMechanicSample | None,
) -> SampleComparison:
    """Compare one observed sample with one simulator result."""

    if simulated is None:
        return SampleComparison(
            observed=observed,
            simulated=None,
            agrees=False,
            absolute_error=None,
            tick_absolute_error=None,
            reason="missing_simulation",
        )
    if observed.sample_id != simulated.sample_id:
        raise ValueError(
            f"sample ID mismatch: {observed.sample_id!r} != {simulated.sample_id!r}"
        )

    absolute_error = _numeric_error(observed.observed_value, simulated.value)
    value_agrees = _value_agrees(
        observed.observed_value,
        simulated.value,
        observed.tolerance,
    )

    tick_error: int | None = None
    tick_agrees = True
    if observed.observed_tick is not None:
        if simulated.tick is None:
            tick_agrees = False
        else:
            tick_error = abs(simulated.tick - observed.observed_tick)
            tick_agrees = tick_error <= observed.tolerance.ticks

    agrees = value_agrees and tick_agrees
    if not value_agrees:
        reason = "value_outside_tolerance"
    elif not tick_agrees:
        reason = "missing_simulation_tick" if simulated.tick is None else "tick_outside_tolerance"
    else:
        reason = "within_tolerance"

    return SampleComparison(
        observed=observed,
        simulated=simulated,
        agrees=agrees,
        absolute_error=absolute_error,
        tick_absolute_error=tick_error,
        reason=reason,
    )


def compare_samples(
    observed_samples: Iterable[ObservedMechanicSample],
    simulated_samples: Iterable[SimulatedMechanicSample],
) -> tuple[SampleComparison, ...]:
    """Pair samples by ID, retaining missing simulator results as failures."""

    observed_by_id: dict[str, ObservedMechanicSample] = {}
    for sample in observed_samples:
        if sample.sample_id in observed_by_id:
            raise ValueError(f"duplicate observed sample ID {sample.sample_id!r}")
        observed_by_id[sample.sample_id] = sample

    simulated_by_id: dict[str, SimulatedMechanicSample] = {}
    for sample in simulated_samples:
        if sample.sample_id in simulated_by_id:
            raise ValueError(f"duplicate simulated sample ID {sample.sample_id!r}")
        simulated_by_id[sample.sample_id] = sample

    return tuple(
        compare_sample(observed_by_id[sample_id], simulated_by_id.get(sample_id))
        for sample_id in sorted(observed_by_id)
    )


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Normalized simulator event used by trace comparison."""

    tick: int
    kind: str
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tick(self.tick, "event tick")
        _require_name(self.kind, "event kind")
        object.__setattr__(self, "values", _immutable_mapping(self.values, "event values"))


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    """An observed event with explicit timing and field tolerances."""

    tick: int
    kind: str
    values: Mapping[str, Any] = field(default_factory=dict)
    tick_tolerance: int = 0
    value_tolerances: Mapping[str, ComparisonTolerance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tick(self.tick, "observed event tick")
        _require_name(self.kind, "observed event kind")
        if type(self.tick_tolerance) is not int or self.tick_tolerance < 0:
            raise ValueError("event tick_tolerance must be a non-negative integer")
        immutable_values = _immutable_mapping(self.values, "observed event values")
        immutable_tolerances = _immutable_mapping(
            self.value_tolerances,
            "event value tolerances",
        )
        unknown = set(immutable_tolerances).difference(immutable_values)
        if unknown:
            raise ValueError(
                "value tolerances reference absent fields: " + ", ".join(sorted(unknown))
            )
        for key, tolerance in immutable_tolerances.items():
            if not isinstance(tolerance, ComparisonTolerance):
                raise TypeError(f"value tolerance for {key!r} must be ComparisonTolerance")
        object.__setattr__(self, "values", immutable_values)
        object.__setattr__(self, "value_tolerances", immutable_tolerances)


@dataclass(frozen=True, slots=True)
class ObservedTrace:
    """Ordered observed events for one mechanic in one evidence split.

    By default simulator events are filtered to kinds present in the observed
    trace.  Set ``included_event_kinds`` explicitly when an empty observation
    is intended to assert that specific event kinds do not occur.
    """

    trace_id: str
    mechanic: str
    split: DatasetSplit
    events: tuple[ObservedEvent, ...]
    evidence: ObservationEvidence
    included_event_kinds: frozenset[str] | None = None

    def __post_init__(self) -> None:
        _require_name(self.trace_id, "trace_id")
        _require_name(self.mechanic, "mechanic")
        object.__setattr__(self, "split", _as_split(self.split))
        object.__setattr__(self, "events", tuple(self.events))
        if any(not isinstance(event, ObservedEvent) for event in self.events):
            raise TypeError("trace events must be ObservedEvent instances")
        ticks = [event.tick for event in self.events]
        if ticks != sorted(ticks):
            raise ValueError("observed trace events must be ordered by non-decreasing tick")
        if self.included_event_kinds is None and self.events:
            object.__setattr__(
                self,
                "included_event_kinds",
                frozenset(event.kind for event in self.events),
            )
        elif self.included_event_kinds is not None:
            kinds = frozenset(self.included_event_kinds)
            if any(not isinstance(kind, str) or not kind for kind in kinds):
                raise ValueError("included event kinds must be non-empty strings")
            missing = {event.kind for event in self.events}.difference(kinds)
            if missing:
                raise ValueError(
                    "included_event_kinds excludes observed kinds: "
                    + ", ".join(sorted(missing))
                )
            object.__setattr__(self, "included_event_kinds", kinds)


@dataclass(frozen=True, slots=True)
class TraceComparison:
    trace_id: str
    mechanic: str
    split: DatasetSplit
    agrees: bool
    divergence_tick: int | None
    reason: str
    observed_event_count: int
    simulated_event_count: int
    matched_event_count: int
    observation_source_id: str
    observation_group_id: str | None = None
    observation_method: str | None = None


def _camel_to_snake(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _event_mapping(event: object) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    if is_dataclass(event) and not isinstance(event, type):
        return {item.name: getattr(event, item.name) for item in fields(event)}
    asdict = getattr(event, "_asdict", None)
    if callable(asdict):
        value = asdict()
        if isinstance(value, Mapping):
            return dict(value)
    try:
        return dict(vars(event))
    except TypeError:
        slots = getattr(type(event), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        result = {name: getattr(event, name) for name in slots if hasattr(event, name)}
        if result:
            return result
    raise TypeError("simulator events must be mappings or simple typed records")


def normalize_event(event: object) -> EventRecord:
    """Normalize a dict, dataclass, named tuple, or simple record event.

    Recognized kind keys are ``kind``, ``event_type``, and ``type``.  A typed
    record without one uses its class name converted to snake case.  Remaining
    public fields are available for observed-field comparison.
    """

    if isinstance(event, EventRecord):
        return event
    payload = _event_mapping(event)

    tick = payload.pop("tick", _MISSING)
    if tick is _MISSING:
        raise ValueError("simulator event is missing tick")

    kind: object = _MISSING
    for key in ("kind", "event_type", "type"):
        if key in payload:
            kind = payload.pop(key)
            break
    if kind is _MISSING:
        kind = _camel_to_snake(type(event).__name__)
    if isinstance(kind, Enum):
        kind = kind.value
    if not isinstance(kind, str):
        raise TypeError("simulator event kind must be a string or string enum")

    nested_values: dict[str, Any] = {}
    for key in ("values", "fields", "attributes", "data"):
        if key in payload:
            nested = payload.pop(key)
            if isinstance(nested, Mapping):
                nested_values.update(nested)
                continue
            try:
                nested_values.update(dict(nested))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"simulator event {key} must be a mapping or key/value pairs"
                ) from exc
    nested_values.update(
        (key, value)
        for key, value in payload.items()
        if not key.startswith("_")
    )
    return EventRecord(tick=tick, kind=kind, values=nested_values)  # type: ignore[arg-type]


def compare_trace(
    observed: ObservedTrace,
    simulated_events: Iterable[object],
) -> TraceComparison:
    """Compare an observed event sequence with an engine event sequence."""

    simulated = tuple(normalize_event(event) for event in simulated_events)
    if any(a.tick > b.tick for a, b in zip(simulated, simulated[1:])):
        raise ValueError("simulator events must be ordered by non-decreasing tick")
    if observed.included_event_kinds is not None:
        simulated = tuple(
            event for event in simulated if event.kind in observed.included_event_kinds
        )

    matched = 0
    for expected, actual in zip(observed.events, simulated):
        divergence_tick = min(expected.tick, actual.tick)
        if expected.kind != actual.kind:
            return TraceComparison(
                observed.trace_id,
                observed.mechanic,
                observed.split,
                False,
                divergence_tick,
                "event_kind_mismatch",
                len(observed.events),
                len(simulated),
                matched,
                observed.evidence.source_id,
                observed.evidence.group_id,
                observed.evidence.method,
            )
        if abs(actual.tick - expected.tick) > expected.tick_tolerance:
            return TraceComparison(
                observed.trace_id,
                observed.mechanic,
                observed.split,
                False,
                divergence_tick,
                "event_tick_outside_tolerance",
                len(observed.events),
                len(simulated),
                matched,
                observed.evidence.source_id,
                observed.evidence.group_id,
                observed.evidence.method,
            )
        for key in sorted(expected.values):
            if key not in actual.values:
                return TraceComparison(
                    observed.trace_id,
                    observed.mechanic,
                    observed.split,
                    False,
                    divergence_tick,
                    f"missing_event_field:{key}",
                    len(observed.events),
                    len(simulated),
                    matched,
                    observed.evidence.source_id,
                    observed.evidence.group_id,
                    observed.evidence.method,
                )
            tolerance = expected.value_tolerances.get(key, ComparisonTolerance())
            if not _value_agrees(expected.values[key], actual.values[key], tolerance):
                return TraceComparison(
                    observed.trace_id,
                    observed.mechanic,
                    observed.split,
                    False,
                    divergence_tick,
                    f"event_field_outside_tolerance:{key}",
                    len(observed.events),
                    len(simulated),
                    matched,
                    observed.evidence.source_id,
                    observed.evidence.group_id,
                    observed.evidence.method,
                )
        matched += 1

    if len(observed.events) > len(simulated):
        missing = observed.events[len(simulated)]
        return TraceComparison(
            observed.trace_id,
            observed.mechanic,
            observed.split,
            False,
            missing.tick,
            "missing_simulator_event",
            len(observed.events),
            len(simulated),
            matched,
            observed.evidence.source_id,
            observed.evidence.group_id,
            observed.evidence.method,
        )
    if len(simulated) > len(observed.events):
        extra = simulated[len(observed.events)]
        return TraceComparison(
            observed.trace_id,
            observed.mechanic,
            observed.split,
            False,
            extra.tick,
            "unexpected_simulator_event",
            len(observed.events),
            len(simulated),
            matched,
            observed.evidence.source_id,
            observed.evidence.group_id,
            observed.evidence.method,
        )
    return TraceComparison(
        observed.trace_id,
        observed.mechanic,
        observed.split,
        True,
        None,
        "within_tolerance",
        len(observed.events),
        len(simulated),
        matched,
        observed.evidence.source_id,
        observed.evidence.group_id,
        observed.evidence.method,
    )


def compare_traces(
    observed_traces: Iterable[ObservedTrace],
    simulated_events_by_trace: Mapping[str, Iterable[object]],
) -> tuple[TraceComparison, ...]:
    """Compare traces by ID; an absent simulator trace is treated as empty."""

    observed_by_id: dict[str, ObservedTrace] = {}
    for trace in observed_traces:
        if trace.trace_id in observed_by_id:
            raise ValueError(f"duplicate observed trace ID {trace.trace_id!r}")
        observed_by_id[trace.trace_id] = trace
    return tuple(
        compare_trace(trace, simulated_events_by_trace.get(trace_id, ()))
        for trace_id, trace in sorted(observed_by_id.items())
    )


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _wilson_interval(successes: int, total: int, confidence: float) -> dict[str, float] | None:
    if total == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "confidence_level": confidence,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _sample_summary(
    comparisons: list[SampleComparison],
    confidence: float,
) -> dict[str, object]:
    errors = [
        item.absolute_error
        for item in comparisons
        if item.absolute_error is not None
    ]
    tick_errors = [
        float(item.tick_absolute_error)
        for item in comparisons
        if item.tick_absolute_error is not None
    ]
    agreements = sum(item.agrees for item in comparisons)
    total = len(comparisons)
    return {
        "count": total,
        "simulated_count": sum(item.simulated is not None for item in comparisons),
        "numeric_error_count": len(errors),
        "agreement_count": agreements,
        "agreement_rate": agreements / total if total else None,
        "agreement_confidence_interval": _wilson_interval(agreements, total, confidence),
        "mae": sum(errors) / len(errors) if errors else None,
        "p95_absolute_error": _nearest_rank(errors, 0.95),
        "tick_error_count": len(tick_errors),
        "tick_mae": sum(tick_errors) / len(tick_errors) if tick_errors else None,
        "tick_p95_absolute_error": _nearest_rank(tick_errors, 0.95),
    }


def _trace_summary(
    comparisons: list[TraceComparison],
    confidence: float,
) -> dict[str, object]:
    agreements = sum(item.agrees for item in comparisons)
    total = len(comparisons)
    divergences = [
        item.divergence_tick
        for item in comparisons
        if item.divergence_tick is not None
    ]
    return {
        "count": total,
        "agreement_count": agreements,
        "agreement_rate": agreements / total if total else None,
        "agreement_confidence_interval": _wilson_interval(agreements, total, confidence),
        "observed_event_count": sum(item.observed_event_count for item in comparisons),
        "simulated_event_count": sum(item.simulated_event_count for item in comparisons),
        "matched_event_count": sum(item.matched_event_count for item in comparisons),
        "diverged_count": total - agreements,
        "first_divergence_tick": min(divergences) if divergences else None,
    }


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """A deterministic, JSON-serializable fidelity report."""

    ruleset_id: str
    split: DatasetSplit
    confidence_level: float
    mechanics: Mapping[str, Mapping[str, object]]
    overall: Mapping[str, object]
    excluded_by_split: Mapping[str, Mapping[str, int]]
    divergences: tuple[Mapping[str, object], ...]
    sample_failures: tuple[Mapping[str, object], ...] = ()
    gate: Mapping[str, object] | None = None
    case_results: tuple[Mapping[str, object], ...] = ()
    ruleset_hash: str | None = None
    corpus_id: str | None = None
    corpus_hash: str | None = None
    engine_version: str | None = None
    tick_us: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "ruleset_id": self.ruleset_id,
            "dataset_split": self.split.value,
            "evidence_notice": _EVIDENCE_NOTICE,
            "confidence_level": self.confidence_level,
            "canonical_units": {
                "position": "milli_tile",
                "hitpoints": "level_11_integer_hp",
                "elixir": "milli_elixir",
                "time": "simulation_tick",
                "tick_duration": "microseconds",
            },
            "overall": dict(self.overall),
            "mechanics": {
                mechanic: {
                    section: dict(metrics) if isinstance(metrics, Mapping) else metrics
                    for section, metrics in sections.items()
                }
                for mechanic, sections in sorted(self.mechanics.items())
            },
            "excluded_by_split": {
                split: dict(counts)
                for split, counts in sorted(self.excluded_by_split.items())
            },
            "trace_divergences": [dict(item) for item in self.divergences],
            "sample_failures": [dict(item) for item in self.sample_failures],
            "case_results": [dict(item) for item in self.case_results],
        }
        if self.corpus_id is not None:
            result["corpus_id"] = self.corpus_id
        if self.ruleset_hash is not None:
            result["ruleset_hash"] = self.ruleset_hash
        if self.corpus_hash is not None:
            result["corpus_hash"] = self.corpus_hash
        if self.engine_version is not None:
            result["engine_version"] = self.engine_version
        if self.tick_us is not None:
            result["tick_us"] = self.tick_us
        if self.gate is not None:
            result["gate"] = dict(self.gate)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(self.to_json(indent=indent), encoding="utf-8")
        temporary.replace(destination)


def _excluded_counts(
    sample_comparisons: tuple[SampleComparison, ...],
    trace_comparisons: tuple[TraceComparison, ...],
    target: DatasetSplit,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in DatasetSplit:
        if split is target:
            continue
        sample_count = sum(item.split is split for item in sample_comparisons)
        trace_count = sum(item.split is split for item in trace_comparisons)
        if sample_count or trace_count:
            result[split.value] = {
                "sample_comparisons": sample_count,
                "trace_comparisons": trace_count,
            }
    return result


def build_fidelity_report(
    *,
    ruleset_id: str,
    sample_comparisons: Iterable[SampleComparison] = (),
    trace_comparisons: Iterable[TraceComparison] = (),
    split: DatasetSplit | str = DatasetSplit.HELDOUT,
    confidence_level: float = 0.95,
    ruleset_hash: str | None = None,
    corpus_id: str | None = None,
    corpus_hash: str | None = None,
    engine_version: str | None = None,
    tick_us: int | None = None,
    excluded_by_split_counts: Mapping[str, Mapping[str, int]] | None = None,
    case_results: Iterable[Mapping[str, object]] = (),
) -> FidelityReport:
    """Aggregate one split without allowing other splits into its metrics."""

    _require_name(ruleset_id, "ruleset_id")
    if ruleset_hash is not None:
        _require_name(ruleset_hash, "ruleset_hash")
    if corpus_id is not None:
        _require_name(corpus_id, "corpus_id")
    if corpus_hash is not None:
        _require_name(corpus_hash, "corpus_hash")
    if engine_version is not None:
        _require_name(engine_version, "engine_version")
    if tick_us is not None and (type(tick_us) is not int or tick_us <= 0):
        raise ValueError("tick_us must be a positive integer")
    target = _as_split(split)
    if (
        type(confidence_level) not in (int, float)
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")

    all_samples = tuple(sample_comparisons)
    all_traces = tuple(trace_comparisons)
    selected_samples = [item for item in all_samples if item.split is target]
    selected_traces = [item for item in all_traces if item.split is target]

    mechanic_names = sorted(
        {item.mechanic for item in selected_samples}
        | {item.mechanic for item in selected_traces}
    )
    mechanic_reports: dict[str, dict[str, object]] = {}
    for mechanic in mechanic_names:
        mechanic_samples = [item for item in selected_samples if item.mechanic == mechanic]
        mechanic_traces = [item for item in selected_traces if item.mechanic == mechanic]
        sample_evidence = [item.observed.evidence for item in mechanic_samples]
        mechanic_reports[mechanic] = {
            "samples": _sample_summary(
                mechanic_samples,
                float(confidence_level),
            ),
            "traces": _trace_summary(
                mechanic_traces,
                float(confidence_level),
            ),
            "evidence": {
                "source_ids": sorted(
                    {item.source_id for item in sample_evidence}
                    | {item.observation_source_id for item in mechanic_traces}
                ),
                "group_ids": sorted(
                    {item.group_id for item in sample_evidence if item.group_id is not None}
                    | {
                        item.observation_group_id
                        for item in mechanic_traces
                        if item.observation_group_id is not None
                    }
                ),
                "methods": sorted(
                    {item.method for item in sample_evidence}
                    | {
                        item.observation_method
                        for item in mechanic_traces
                        if item.observation_method is not None
                    }
                ),
            },
        }

    divergences = tuple(
        {
            "trace_id": item.trace_id,
            "mechanic": item.mechanic,
            "observation_source_id": item.observation_source_id,
            "observation_group_id": item.observation_group_id,
            "observation_method": item.observation_method,
            "divergence_tick": item.divergence_tick,
            "reason": item.reason,
        }
        for item in sorted(selected_traces, key=lambda row: (row.trace_id, row.mechanic))
        if not item.agrees
    )
    overall = {
        "samples": _sample_summary(selected_samples, float(confidence_level)),
        "traces": _trace_summary(selected_traces, float(confidence_level)),
        "mechanic_count": len(mechanic_names),
        "observation_source_count": len(
            {
                item.observed.evidence.source_id
                for item in selected_samples
            }
            | {item.observation_source_id for item in selected_traces}
        ),
        "observation_group_count": len(
            {
                item.observed.evidence.group_id
                for item in selected_samples
                if item.observed.evidence.group_id is not None
            }
            | {
                item.observation_group_id
                for item in selected_traces
                if item.observation_group_id is not None
            }
        ),
    }
    sample_failures = tuple(
        {
            "sample_id": item.sample_id,
            "mechanic": item.mechanic,
            "source_id": item.observed.evidence.source_id,
            "group_id": item.observed.evidence.group_id,
            "method": item.observed.evidence.method,
            "confidence": item.observed.evidence.confidence,
            "observed_value": item.observed.observed_value,
            "simulated_value": None if item.simulated is None else item.simulated.value,
            "observed_tick": item.observed.observed_tick,
            "simulated_tick": None if item.simulated is None else item.simulated.tick,
            "absolute_error": item.absolute_error,
            "tick_absolute_error": item.tick_absolute_error,
            "reason": item.reason,
        }
        for item in selected_samples
        if not item.agrees
    )
    return FidelityReport(
        ruleset_id=ruleset_id,
        split=target,
        confidence_level=float(confidence_level),
        mechanics=MappingProxyType(mechanic_reports),
        overall=MappingProxyType(overall),
        excluded_by_split=MappingProxyType(
            dict(excluded_by_split_counts)
            if excluded_by_split_counts is not None
            else _excluded_counts(all_samples, all_traces, target)
        ),
        divergences=divergences,
        sample_failures=sample_failures,
        case_results=tuple(dict(item) for item in case_results),
        ruleset_hash=ruleset_hash,
        corpus_id=corpus_id,
        corpus_hash=corpus_hash,
        engine_version=engine_version,
        tick_us=tick_us,
    )


__all__ = [
    "ComparisonTolerance",
    "DatasetSplit",
    "EventRecord",
    "FidelityReport",
    "ObservationEvidence",
    "ObservedEvent",
    "ObservedMechanicSample",
    "ObservedTrace",
    "SampleComparison",
    "SimulatedMechanicSample",
    "TraceComparison",
    "build_fidelity_report",
    "compare_sample",
    "compare_samples",
    "compare_trace",
    "compare_traces",
    "normalize_event",
]
