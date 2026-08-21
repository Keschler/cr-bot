"""Network-free execution of pre-split sim-to-real fidelity corpora.

The corpus is an immutable input, not a labeling workflow.  Every case is
assigned to ``calibration``, ``validation``, ``regression``, or ``heldout``
before it is run and carries explicit observation provenance.  Extractor
confidence is retained in :mod:`simulator.fidelity`; it never changes an
observed value, fills a missing simulator result, or weights an agreement.

Corpus schema version 1::

    {
      "schema_version": 1,
      "corpus_id": "hog-cycle-controlled-001",
      "engine_version": "reference-0.28.0",
      "ruleset_id": "2026-08-04",
      "ruleset_hash": "sha256:...",
      "cases": [
        {
          "case_id": "hog-left-princess",
          "split": "heldout",
          "evidence": {
            "source_id": "capture-120fps-0042",
            "method": "controlled_capture_v1",
            "confidence": 0.99,
            "notes": "optional"
          },
          "scenario_path": "scenarios/hog-left-princess.json",
          "measurements": [
            {
              "sample_id": "hog-left-princess:hits",
              "mechanic": "hog_tower_hits",
              "observed_value": 2,
              "extractor": {
                "type": "event_count",
                "event_kind": "damage_applied",
                "filters": {
                  "source_card_id": "hog-rider",
                  "target_role": "left"
                }
              }
            }
          ],
          "traces": []
        }
      ]
    }

Exactly one of ``scenario`` (an inline canonical scenario) and
``scenario_path`` is required per case.  Paths are relative to the corpus
file, are constrained to that directory tree, and are never fetched.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias

from .engine import ENGINE_VERSION, BattleEngine
from .fidelity import (
    ComparisonTolerance,
    DatasetSplit,
    EventRecord,
    FidelityReport,
    ObservationEvidence,
    ObservedEvent,
    ObservedMechanicSample,
    ObservedTrace,
    SampleComparison,
    SimulatedMechanicSample,
    TraceComparison,
    build_fidelity_report,
    compare_samples,
    compare_trace,
    normalize_event,
)
from .fixed import SECOND_US, distance_mtile
from .runner import run_scenario_with_snapshots
from .ruleset import Ruleset
from .scenario import SCENARIO_SCHEMA_VERSION, Scenario, scenario_from_dict
from .state import BattleState, EntityState


CORPUS_SCHEMA_VERSION = 1
Scalar: TypeAlias = bool | int | float | str

_CONTENT_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CASE_SPLITS = frozenset(DatasetSplit)
_EVENT_EXTRACTORS = frozenset(
    {
        "event_count",
        "first_event_tick",
        "last_event_tick",
        "first_event_field",
    }
)
_FINAL_EXTRACTORS = frozenset(
    {
        "final_tower_hp",
        "final_tower_alive",
        "final_entity_hp",
        "final_entity_alive",
    }
)
_SNAPSHOT_EXTRACTORS = frozenset(
    {
        "entity_x_mtile_at_tick",
        "entity_y_mtile_at_tick",
        "entity_hp_at_tick",
        "entity_alive_at_tick",
        "tower_hp_at_tick",
        "tower_alive_at_tick",
    }
)
_DUAL_SNAPSHOT_EXTRACTORS = frozenset(
    {"entity_displacement_speed_mtile_per_s"}
)
_CARD_DEFINITION_EXTRACTORS = frozenset({"card_move_speed_mtile_per_s"})
_OUTCOME_EXTRACTORS = frozenset({"outcome_winner", "outcome_reason"})
_EXTRACTOR_TYPES = (
    _EVENT_EXTRACTORS
    | _FINAL_EXTRACTORS
    | _SNAPSHOT_EXTRACTORS
    | _DUAL_SNAPSHOT_EXTRACTORS
    | _CARD_DEFINITION_EXTRACTORS
    | _OUTCOME_EXTRACTORS
)
_ENTITY_FILTER_FIELDS = frozenset(
    {"uid", "owner", "card_id", "role", "kind", "alive", "spawn_tick"}
)
_MISSING = object()


class ValidationCorpusError(ValueError):
    """Raised when a fidelity corpus or deterministic extractor is invalid."""


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationCorpusError(f"{field_name} must be a non-empty string")
    return value


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationCorpusError(f"{field_name} must be an integer >= {minimum}")
    return value


def _object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationCorpusError(f"{field_name} must be an object")
    return value


def _array(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationCorpusError(f"{field_name} must be an array")
    return value


def _fields(
    row: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    unknown = sorted(set(row) - required - optional)
    if unknown:
        raise ValidationCorpusError(f"unknown fields at {context}: {unknown}")
    missing = sorted(required - set(row))
    if missing:
        raise ValidationCorpusError(f"missing fields at {context}: {missing}")


def _scalar(value: object, field_name: str, *, allow_none: bool = False) -> Any:
    if value is None and allow_none:
        return None
    if not isinstance(value, (bool, int, float, str)):
        raise ValidationCorpusError(f"{field_name} must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationCorpusError(f"{field_name} must be finite")
    return value


def _freeze_scalar_filters(value: object, context: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    row = _object(value, context)
    result: dict[str, Any] = {}
    for key, item in row.items():
        _name(key, f"{context} key")
        result[key] = _scalar(item, f"{context}.{key}", allow_none=True)
    return MappingProxyType(result)


def _split(value: object, field_name: str = "split") -> DatasetSplit:
    try:
        split = DatasetSplit(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in DatasetSplit)
        raise ValidationCorpusError(
            f"{field_name} must be one of {choices}; got {value!r}"
        ) from error
    if split not in _CASE_SPLITS:  # defensive if DatasetSplit is extended
        raise ValidationCorpusError(f"unsupported case split: {split.value}")
    return split


@dataclass(frozen=True, slots=True)
class ExtractorSpec:
    """A deterministic projection from one final battle and its event log."""

    extractor_type: str
    event_kind: str | None = None
    field_name: str | None = None
    tick: int | None = None
    start_tick: int | None = None
    end_tick: int | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.extractor_type not in _EXTRACTOR_TYPES:
            choices = ", ".join(sorted(_EXTRACTOR_TYPES))
            raise ValidationCorpusError(
                f"unknown extractor type {self.extractor_type!r}; expected one of {choices}"
            )
        object.__setattr__(
            self,
            "filters",
            _freeze_scalar_filters(dict(self.filters), "extractor.filters"),
        )
        if self.extractor_type in _EVENT_EXTRACTORS:
            _name(self.event_kind, "extractor.event_kind")
        elif self.event_kind is not None:
            raise ValidationCorpusError(
                f"{self.extractor_type} does not accept event_kind"
            )
        if self.extractor_type == "first_event_field":
            _name(self.field_name, "extractor.field")
        elif self.field_name is not None:
            raise ValidationCorpusError(f"{self.extractor_type} does not accept field")
        if self.extractor_type in _SNAPSHOT_EXTRACTORS:
            _integer(self.tick, "extractor.tick")
        elif self.tick is not None:
            raise ValidationCorpusError(f"{self.extractor_type} does not accept tick")
        if self.extractor_type in _DUAL_SNAPSHOT_EXTRACTORS:
            start_tick = _integer(self.start_tick, "extractor.start_tick")
            end_tick = _integer(self.end_tick, "extractor.end_tick")
            if end_tick <= start_tick:
                raise ValidationCorpusError(
                    "extractor.end_tick must be greater than extractor.start_tick"
                )
        elif self.start_tick is not None or self.end_tick is not None:
            raise ValidationCorpusError(
                f"{self.extractor_type} does not accept start_tick/end_tick"
            )
        if self.extractor_type in _OUTCOME_EXTRACTORS and self.filters:
            raise ValidationCorpusError(
                f"{self.extractor_type} does not accept filters"
            )
        if self.extractor_type in (
            _FINAL_EXTRACTORS
            | _SNAPSHOT_EXTRACTORS
            | _DUAL_SNAPSHOT_EXTRACTORS
            | _CARD_DEFINITION_EXTRACTORS
        ):
            unknown = sorted(set(self.filters) - _ENTITY_FILTER_FIELDS)
            if unknown:
                raise ValidationCorpusError(
                    f"unsupported final entity filters: {unknown}"
                )
            _validate_entity_filters(self.filters)


@dataclass(frozen=True, slots=True)
class ScalarObservationSpec:
    observed: ObservedMechanicSample
    extractor: ExtractorSpec


@dataclass(frozen=True, slots=True)
class TraceObservationSpec:
    observed: ObservedTrace
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filters",
            _freeze_scalar_filters(dict(self.filters), "trace.filters"),
        )


@dataclass(frozen=True, slots=True)
class ValidationCase:
    case_id: str
    split: DatasetSplit
    evidence: ObservationEvidence
    measurements: tuple[ScalarObservationSpec, ...]
    traces: tuple[TraceObservationSpec, ...]
    scenario: Scenario | None = None
    scenario_path: str | None = None

    def __post_init__(self) -> None:
        _name(self.case_id, "case_id")
        object.__setattr__(self, "split", _split(self.split, "case split"))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "traces", tuple(self.traces))
        if (self.scenario is None) == (self.scenario_path is None):
            raise ValidationCorpusError(
                f"case {self.case_id!r} requires exactly one of scenario and scenario_path"
            )
        if not self.measurements and not self.traces:
            raise ValidationCorpusError(
                f"case {self.case_id!r} has no measurements or event traces"
            )


@dataclass(frozen=True, slots=True)
class ValidationCorpus:
    corpus_id: str
    engine_version: str
    ruleset_id: str
    ruleset_hash: str
    cases: tuple[ValidationCase, ...]
    base_dir: Path
    content_hash: str
    schema_version: int = CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != CORPUS_SCHEMA_VERSION:
            raise ValidationCorpusError(
                f"unsupported corpus schema {self.schema_version!r}; "
                f"expected {CORPUS_SCHEMA_VERSION}"
            )
        _name(self.corpus_id, "corpus_id")
        _name(self.engine_version, "engine_version")
        _name(self.ruleset_id, "ruleset_id")
        if not isinstance(self.ruleset_hash, str) or not _CONTENT_HASH_RE.fullmatch(
            self.ruleset_hash
        ):
            raise ValidationCorpusError(
                "ruleset_hash must be sha256:<64 lowercase hex characters>"
            )
        if not isinstance(self.content_hash, str) or not _CONTENT_HASH_RE.fullmatch(
            self.content_hash
        ):
            raise ValidationCorpusError(
                "content_hash must be sha256:<64 lowercase hex characters>"
            )
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "base_dir", self.base_dir.resolve())
        _require_unique_case_and_observation_ids(self.cases)


def _validate_entity_filters(filters: Mapping[str, Any]) -> None:
    if "uid" in filters:
        _integer(filters["uid"], "filters.uid", minimum=1)
    if "owner" in filters:
        owner = _integer(filters["owner"], "filters.owner")
        if owner not in (0, 1):
            raise ValidationCorpusError("filters.owner must be 0 or 1")
    if "spawn_tick" in filters:
        _integer(filters["spawn_tick"], "filters.spawn_tick")
    if "alive" in filters and type(filters["alive"]) is not bool:
        raise ValidationCorpusError("filters.alive must be boolean")
    for key in ("card_id", "kind"):
        if key in filters:
            _name(filters[key], f"filters.{key}")
    if "role" in filters and filters["role"] is not None:
        _name(filters["role"], "filters.role")


def _require_unique_case_and_observation_ids(cases: Iterable[ValidationCase]) -> None:
    case_ids: set[str] = set()
    observation_ids: set[str] = set()
    for case in cases:
        if case.case_id in case_ids:
            raise ValidationCorpusError(f"duplicate case_id {case.case_id!r}")
        case_ids.add(case.case_id)
        for measurement in case.measurements:
            sample_id = measurement.observed.sample_id
            if sample_id in observation_ids:
                raise ValidationCorpusError(f"duplicate observation ID {sample_id!r}")
            observation_ids.add(sample_id)
        for trace in case.traces:
            trace_id = trace.observed.trace_id
            if trace_id in observation_ids:
                raise ValidationCorpusError(f"duplicate observation ID {trace_id!r}")
            observation_ids.add(trace_id)


def _parse_evidence(value: object, context: str) -> ObservationEvidence:
    row = _object(value, context)
    _fields(
        row,
        required={"source_id", "group_id", "method", "confidence"},
        optional={"notes", "media_hash", "frame_start", "frame_end"},
        context=context,
    )
    confidence = row["confidence"]
    if confidence is not None:
        _scalar(confidence, f"{context}.confidence")
    notes = row.get("notes")
    if notes is not None:
        _name(notes, f"{context}.notes")
    for field_name in ("frame_start", "frame_end"):
        if row.get(field_name) is not None:
            _integer(row[field_name], f"{context}.{field_name}")
    media_hash = row.get("media_hash")
    if media_hash is not None and (
        not isinstance(media_hash, str) or not _CONTENT_HASH_RE.fullmatch(media_hash)
    ):
        raise ValidationCorpusError(
            f"{context}.media_hash must be sha256:<64 lowercase hex characters>"
        )
    try:
        return ObservationEvidence(
            source_id=_name(row["source_id"], f"{context}.source_id"),
            group_id=_name(row["group_id"], f"{context}.group_id"),
            method=_name(row["method"], f"{context}.method"),
            confidence=confidence,
            notes=notes,
            media_hash=media_hash,
            frame_start=row.get("frame_start"),
            frame_end=row.get("frame_end"),
        )
    except (TypeError, ValueError) as error:
        raise ValidationCorpusError(f"invalid {context}: {error}") from error


def _parse_tolerance(value: object, context: str) -> ComparisonTolerance:
    if value is None:
        return ComparisonTolerance()
    row = _object(value, context)
    _fields(
        row,
        required=set(),
        optional={"absolute", "relative", "ticks"},
        context=context,
    )
    try:
        return ComparisonTolerance(
            absolute=row.get("absolute", 0.0),
            relative=row.get("relative", 0.0),
            ticks=row.get("ticks", 0),
        )
    except (TypeError, ValueError) as error:
        raise ValidationCorpusError(f"invalid {context}: {error}") from error


def _parse_extractor(value: object, context: str) -> ExtractorSpec:
    row = _object(value, context)
    extractor_type = _name(row.get("type"), f"{context}.type")
    if extractor_type not in _EXTRACTOR_TYPES:
        choices = ", ".join(sorted(_EXTRACTOR_TYPES))
        raise ValidationCorpusError(
            f"unknown extractor type {extractor_type!r}; expected one of {choices}"
        )
    if extractor_type in _OUTCOME_EXTRACTORS:
        allowed = {"type"}
    elif extractor_type == "first_event_field":
        allowed = {"type", "event_kind", "field", "filters"}
    elif extractor_type in _EVENT_EXTRACTORS:
        allowed = {"type", "event_kind", "filters"}
    elif extractor_type in _SNAPSHOT_EXTRACTORS:
        allowed = {"type", "tick", "filters"}
    elif extractor_type in _DUAL_SNAPSHOT_EXTRACTORS:
        allowed = {"type", "start_tick", "end_tick", "filters"}
    else:
        allowed = {"type", "filters"}
    _fields(row, required={"type"}, optional=allowed - {"type"}, context=context)
    filters = _freeze_scalar_filters(row.get("filters"), f"{context}.filters")
    return ExtractorSpec(
        extractor_type=extractor_type,
        event_kind=row.get("event_kind"),
        field_name=row.get("field"),
        tick=row.get("tick"),
        start_tick=row.get("start_tick"),
        end_tick=row.get("end_tick"),
        filters=filters,
    )


def _parse_measurement(
    value: object,
    *,
    case_split: DatasetSplit,
    evidence: ObservationEvidence,
    context: str,
) -> ScalarObservationSpec:
    row = _object(value, context)
    _fields(
        row,
        required={"sample_id", "mechanic", "observed_value", "extractor"},
        optional={"observed_tick", "tolerance"},
        context=context,
    )
    observed_tick = row.get("observed_tick")
    if observed_tick is not None:
        _integer(observed_tick, f"{context}.observed_tick")
    observed_value = _scalar(row["observed_value"], f"{context}.observed_value")
    try:
        observed = ObservedMechanicSample(
            sample_id=_name(row["sample_id"], f"{context}.sample_id"),
            mechanic=_name(row["mechanic"], f"{context}.mechanic"),
            split=case_split,
            observed_value=observed_value,
            observed_tick=observed_tick,
            evidence=evidence,
            tolerance=_parse_tolerance(row.get("tolerance"), f"{context}.tolerance"),
        )
    except (TypeError, ValueError) as error:
        raise ValidationCorpusError(f"invalid {context}: {error}") from error
    return ScalarObservationSpec(
        observed=observed,
        extractor=_parse_extractor(row["extractor"], f"{context}.extractor"),
    )


def _parse_observed_event(value: object, context: str) -> ObservedEvent:
    row = _object(value, context)
    _fields(
        row,
        required={"tick", "kind"},
        optional={"values", "tick_tolerance", "value_tolerances"},
        context=context,
    )
    values = _freeze_scalar_filters(row.get("values"), f"{context}.values")
    tolerances_raw = _object(
        row.get("value_tolerances", {}), f"{context}.value_tolerances"
    )
    tolerances = {
        key: _parse_tolerance(item, f"{context}.value_tolerances.{key}")
        for key, item in tolerances_raw.items()
    }
    try:
        return ObservedEvent(
            tick=_integer(row["tick"], f"{context}.tick"),
            kind=_name(row["kind"], f"{context}.kind"),
            values=values,
            tick_tolerance=row.get("tick_tolerance", 0),
            value_tolerances=tolerances,
        )
    except (TypeError, ValueError) as error:
        raise ValidationCorpusError(f"invalid {context}: {error}") from error


def _parse_trace(
    value: object,
    *,
    case_split: DatasetSplit,
    evidence: ObservationEvidence,
    context: str,
) -> TraceObservationSpec:
    row = _object(value, context)
    _fields(
        row,
        required={"trace_id", "mechanic", "events"},
        optional={"included_event_kinds", "filters"},
        context=context,
    )
    events = tuple(
        _parse_observed_event(item, f"{context}.events[{index}]")
        for index, item in enumerate(_array(row["events"], f"{context}.events"))
    )
    included_raw = row.get("included_event_kinds")
    included: frozenset[str] | None = None
    if included_raw is not None:
        included_items = _array(included_raw, f"{context}.included_event_kinds")
        included = frozenset(
            _name(item, f"{context}.included_event_kinds[{index}]")
            for index, item in enumerate(included_items)
        )
        if len(included) != len(included_items):
            raise ValidationCorpusError(
                f"{context}.included_event_kinds contains duplicates"
            )
    try:
        observed = ObservedTrace(
            trace_id=_name(row["trace_id"], f"{context}.trace_id"),
            mechanic=_name(row["mechanic"], f"{context}.mechanic"),
            split=case_split,
            events=events,
            evidence=evidence,
            included_event_kinds=included,
        )
    except (TypeError, ValueError) as error:
        raise ValidationCorpusError(f"invalid {context}: {error}") from error
    return TraceObservationSpec(
        observed=observed,
        filters=_freeze_scalar_filters(row.get("filters"), f"{context}.filters"),
    )


_SCENARIO_ALLOWED = {
    "schema_version",
    "scenario_id",
    "ruleset_id",
    "ruleset_hash",
    "engine_version",
    "seed",
    "decks",
    "actions",
    "max_ticks",
    "shuffle_decks",
    "split",
    "tags",
    "oracle",
    "initial_state",
}


def _parse_inline_scenario(value: object, context: str) -> Scenario:
    row = _object(value, context)
    _fields(
        row,
        required={
            "schema_version",
            "scenario_id",
            "ruleset_id",
            "ruleset_hash",
            "engine_version",
            "seed",
            "decks",
            "split",
        },
        optional=_SCENARIO_ALLOWED
        - {
            "schema_version",
            "scenario_id",
            "ruleset_id",
            "ruleset_hash",
            "engine_version",
            "seed",
            "decks",
            "split",
        },
        context=context,
    )
    schema_version = _integer(row["schema_version"], f"{context}.schema_version", minimum=1)
    if schema_version != SCENARIO_SCHEMA_VERSION:
        raise ValidationCorpusError(
            f"{context}.schema_version must be {SCENARIO_SCHEMA_VERSION}"
        )
    _integer(row["seed"], f"{context}.seed")
    max_ticks = row.get("max_ticks")
    if max_ticks is not None:
        _integer(max_ticks, f"{context}.max_ticks", minimum=1)
    shuffle = row.get("shuffle_decks", False)
    if type(shuffle) is not bool:
        raise ValidationCorpusError(f"{context}.shuffle_decks must be boolean")
    try:
        return scenario_from_dict(dict(row))
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationCorpusError(f"invalid {context}: {error}") from error


def _safe_relative_scenario_path(value: object, base_dir: Path, context: str) -> str:
    raw = _name(value, context)
    if "\\" in raw or "://" in raw:
        raise ValidationCorpusError(f"{context} must be a local POSIX-style relative path")
    relative = Path(raw)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValidationCorpusError(f"{context} must not be absolute or traverse directories")
    if relative.suffix.casefold() != ".json":
        raise ValidationCorpusError(f"{context} must reference a .json file")
    root = base_dir.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValidationCorpusError(f"{context} escapes the corpus directory")
    return relative.as_posix()


def _parse_case(value: object, *, base_dir: Path, context: str) -> ValidationCase:
    row = _object(value, context)
    _fields(
        row,
        required={"case_id", "split", "evidence", "measurements"},
        optional={"scenario", "scenario_path", "traces"},
        context=context,
    )
    if ("scenario" in row) == ("scenario_path" in row):
        raise ValidationCorpusError(
            f"{context} requires exactly one of scenario and scenario_path"
        )
    case_split = _split(row["split"], f"{context}.split")
    evidence = _parse_evidence(row["evidence"], f"{context}.evidence")
    measurements = tuple(
        _parse_measurement(
            item,
            case_split=case_split,
            evidence=evidence,
            context=f"{context}.measurements[{index}]",
        )
        for index, item in enumerate(
            _array(row["measurements"], f"{context}.measurements")
        )
    )
    traces = tuple(
        _parse_trace(
            item,
            case_split=case_split,
            evidence=evidence,
            context=f"{context}.traces[{index}]",
        )
        for index, item in enumerate(_array(row.get("traces", []), f"{context}.traces"))
    )
    return ValidationCase(
        case_id=_name(row["case_id"], f"{context}.case_id"),
        split=case_split,
        evidence=evidence,
        measurements=measurements,
        traces=traces,
        scenario=(
            _parse_inline_scenario(row["scenario"], f"{context}.scenario")
            if "scenario" in row
            else None
        ),
        scenario_path=(
            _safe_relative_scenario_path(
                row["scenario_path"], base_dir, f"{context}.scenario_path"
            )
            if "scenario_path" in row
            else None
        ),
    )


def validation_corpus_from_dict(
    raw: dict[str, Any],
    *,
    base_dir: str | Path = ".",
) -> ValidationCorpus:
    """Validate a decoded version-1 corpus without executing it."""

    row = _object(raw, "corpus")
    _fields(
        row,
        required={
            "schema_version",
            "corpus_id",
            "engine_version",
            "ruleset_id",
            "ruleset_hash",
            "cases",
        },
        context="corpus",
    )
    schema = _integer(row["schema_version"], "corpus.schema_version", minimum=1)
    if schema != CORPUS_SCHEMA_VERSION:
        raise ValidationCorpusError(
            f"unsupported corpus schema {schema}; expected {CORPUS_SCHEMA_VERSION}"
        )
    root = Path(base_dir).resolve()
    parsed_cases = tuple(
        _parse_case(item, base_dir=root, context=f"corpus.cases[{index}]")
        for index, item in enumerate(_array(row["cases"], "corpus.cases"))
    )
    group_splits: dict[str, DatasetSplit] = {}
    for case in parsed_cases:
        assert case.evidence.group_id is not None
        prior = group_splits.setdefault(case.evidence.group_id, case.split)
        if prior is not case.split:
            raise ValidationCorpusError(
                f"evidence group {case.evidence.group_id!r} occurs in both "
                f"{prior.value!r} and {case.split.value!r} splits"
            )
    expanded_raw = json.loads(json.dumps(raw, allow_nan=False))
    sealed_cases: list[ValidationCase] = []
    for index, case in enumerate(parsed_cases):
        if case.scenario is not None:
            sealed_cases.append(case)
            continue
        assert case.scenario_path is not None
        relative = _safe_relative_scenario_path(
            case.scenario_path,
            root,
            f"corpus.cases[{index}].scenario_path",
        )
        scenario_path = (root / relative).resolve()
        if not scenario_path.is_file():
            raise ValidationCorpusError(
                f"case {case.case_id!r} scenario is not a regular file: {scenario_path}"
            )
        scenario_raw = _load_json_object(scenario_path, "scenario")
        scenario = _parse_inline_scenario(
            scenario_raw,
            f"case {case.case_id!r} scenario",
        )
        sealed_cases.append(replace(case, scenario=scenario, scenario_path=None))
        expanded_case = expanded_raw["cases"][index]
        expanded_case.pop("scenario_path", None)
        expanded_case["scenario"] = scenario.to_dict()
    encoded = json.dumps(
        expanded_raw,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    content_hash = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    corpus = ValidationCorpus(
        corpus_id=_name(row["corpus_id"], "corpus.corpus_id"),
        engine_version=_name(row["engine_version"], "corpus.engine_version"),
        ruleset_id=_name(row["ruleset_id"], "corpus.ruleset_id"),
        ruleset_hash=_name(row["ruleset_hash"], "corpus.ruleset_hash"),
        cases=tuple(sealed_cases),
        base_dir=root,
        content_hash=content_hash,
        schema_version=schema,
    )
    for case in corpus.cases:
        if case.scenario is not None:
            _verify_scenario_contract(case.scenario, case, corpus)
    return corpus


def _load_json_object(path: Path, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationCorpusError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except FileNotFoundError as error:
        raise ValidationCorpusError(f"missing {context} file: {path}") from error
    except OSError as error:
        raise ValidationCorpusError(f"cannot read {context} file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValidationCorpusError(f"invalid JSON in {context} file {path}: {error}") from error
    return _object(value, context)


def load_validation_corpus(path: str | Path) -> ValidationCorpus:
    """Load a strict corpus from disk without accessing the network."""

    source = Path(path).resolve()
    return validation_corpus_from_dict(
        _load_json_object(source, "corpus"),
        base_dir=source.parent,
    )


def load_validation_corpus_pinned(
    path: str | Path,
    *,
    expected_hash: str | None = None,
) -> ValidationCorpus:
    """Load a corpus and optionally require an out-of-band sealed hash."""

    corpus = load_validation_corpus(path)
    if expected_hash is not None and corpus.content_hash != expected_hash:
        raise ValidationCorpusError(
            f"corpus hash does not match expected pin: expected={expected_hash}, "
            f"actual={corpus.content_hash}"
        )
    return corpus


def _verify_scenario_contract(
    scenario: Scenario,
    case: ValidationCase,
    corpus: ValidationCorpus,
) -> None:
    if scenario.ruleset_id != corpus.ruleset_id:
        raise ValidationCorpusError(
            f"case {case.case_id!r} scenario ruleset ID does not match corpus"
        )
    if scenario.ruleset_hash != corpus.ruleset_hash:
        raise ValidationCorpusError(
            f"case {case.case_id!r} scenario ruleset hash does not match corpus"
        )
    if scenario.engine_version != corpus.engine_version:
        raise ValidationCorpusError(
            f"case {case.case_id!r} scenario engine version does not match corpus"
        )
    if scenario.split != case.split.value:
        raise ValidationCorpusError(
            f"case {case.case_id!r} split {case.split.value!r} does not match "
            f"scenario split {scenario.split!r}"
        )


def _load_case_scenario(case: ValidationCase, corpus: ValidationCorpus) -> Scenario:
    if case.scenario is not None:
        scenario = case.scenario
    else:
        assert case.scenario_path is not None
        relative = _safe_relative_scenario_path(
            case.scenario_path,
            corpus.base_dir,
            f"case {case.case_id!r} scenario_path",
        )
        path = (corpus.base_dir / relative).resolve()
        if not path.is_file():
            raise ValidationCorpusError(
                f"case {case.case_id!r} scenario is not a regular file: {path}"
            )
        scenario = _parse_inline_scenario(
            _load_json_object(path, "scenario"),
            f"case {case.case_id!r} scenario",
        )
    _verify_scenario_contract(scenario, case, corpus)
    return scenario


def _entity_metadata(
    values: dict[str, Any],
    state: BattleState,
    *,
    uid_field: str,
    prefix: str,
    generic: bool = False,
) -> None:
    uid = values.get(uid_field)
    if type(uid) is not int:
        return
    entity = state.entities.get(uid)
    if entity is None:
        return
    metadata = {
        "card_id": entity.card_id,
        "owner": entity.owner,
        "role": entity.role,
        "kind": entity.kind,
    }
    for key, value in metadata.items():
        destination = f"{prefix}_{key}" if prefix else key
        values.setdefault(destination, value)
    if generic:
        values.setdefault("entity_kind", entity.kind)


def normalized_state_events(state: BattleState) -> tuple[EventRecord, ...]:
    """Normalize events and add stable entity card/owner/role lookup fields.

    For an event's primary ``uid``, missing ``card_id``, ``owner``, and
    ``role`` fields are filled from the retained entity.  ``source_uid`` and
    ``target_uid`` similarly expose ``source_*`` and ``target_*`` fields.
    This lets a corpus select a Hog-to-left-Princess damage event without
    depending on run-specific UIDs.
    """

    result: list[EventRecord] = []
    for raw_event in state.events:
        event = normalize_event(raw_event)
        values = dict(event.values)
        _entity_metadata(values, state, uid_field="uid", prefix="", generic=True)
        _entity_metadata(values, state, uid_field="source_uid", prefix="source")
        _entity_metadata(values, state, uid_field="target_uid", prefix="target")
        _entity_metadata(values, state, uid_field="old_target", prefix="old_target")
        result.append(EventRecord(event.tick, event.kind, values))
    return tuple(result)


def _filter_equals(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    return actual == expected


def _event_matches(
    event: EventRecord,
    *,
    event_kind: str | None,
    filters: Mapping[str, Any],
) -> bool:
    if event_kind is not None and event.kind != event_kind:
        return False
    return all(
        key in event.values and _filter_equals(event.values[key], expected)
        for key, expected in filters.items()
    )


def _matching_events(
    events: Iterable[EventRecord],
    *,
    event_kind: str | None,
    filters: Mapping[str, Any],
) -> tuple[EventRecord, ...]:
    return tuple(
        event
        for event in events
        if _event_matches(event, event_kind=event_kind, filters=filters)
    )


def _entity_matches(entity: EntityState, filters: Mapping[str, Any]) -> bool:
    return all(
        hasattr(entity, key)
        and _filter_equals(getattr(entity, key), expected)
        for key, expected in filters.items()
    )


def _unique_final_entity(
    state: BattleState,
    filters: Mapping[str, Any],
    *,
    towers_only: bool,
    sample_id: str,
) -> EntityState | None:
    matches = [
        state.entities[uid]
        for uid in sorted(state.entities)
        if (not towers_only or state.entities[uid].kind == "tower")
        and _entity_matches(state.entities[uid], filters)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValidationCorpusError(
            f"extractor for sample {sample_id!r} matched {len(matches)} final entities; "
            "add filters (normally uid, owner, card_id, or role)"
        )
    return matches[0]


def _outcome_tick(events: Iterable[EventRecord]) -> int | None:
    matches = [event.tick for event in events if event.kind == "match_ended"]
    return matches[-1] if matches else None


def extract_simulated_measurement(
    measurement: ScalarObservationSpec,
    state: BattleState,
    *,
    events: tuple[EventRecord, ...] | None = None,
    snapshots: Mapping[int, BattleState] | None = None,
    ruleset: Ruleset | None = None,
) -> SimulatedMechanicSample | None:
    """Extract one simulator value; return ``None`` when output is absent.

    Returning ``None`` is intentional: :func:`compare_samples` records it as
    ``missing_simulation``.  No value is inferred from observation confidence.
    Ambiguous final-entity selectors are configuration errors rather than an
    arbitrary first/last choice.
    """

    event_rows = normalized_state_events(state) if events is None else events
    spec = measurement.extractor
    sample_id = measurement.observed.sample_id
    value: object = _MISSING
    tick: int | None = None

    if spec.extractor_type == "outcome_winner":
        if state.terminal:
            value = "draw" if state.winner is None else state.winner
            tick = _outcome_tick(event_rows)
    elif spec.extractor_type == "outcome_reason":
        if state.terminal and state.terminal_reason is not None:
            value = state.terminal_reason
            tick = _outcome_tick(event_rows)
    elif spec.extractor_type in _EVENT_EXTRACTORS:
        matches = _matching_events(
            event_rows,
            event_kind=spec.event_kind,
            filters=spec.filters,
        )
        if spec.extractor_type == "event_count":
            value = len(matches)
            tick = state.tick
        elif matches:
            selected = matches[-1] if spec.extractor_type == "last_event_tick" else matches[0]
            tick = selected.tick
            if spec.extractor_type in {"first_event_tick", "last_event_tick"}:
                value = selected.tick
            else:
                assert spec.field_name is not None
                value = selected.values.get(spec.field_name, _MISSING)
    elif spec.extractor_type in _CARD_DEFINITION_EXTRACTORS:
        if ruleset is None:
            raise ValidationCorpusError(
                f"extractor for sample {sample_id!r} requires a ruleset"
            )
        entity = _unique_final_entity(
            state,
            spec.filters,
            towers_only=False,
            sample_id=sample_id,
        )
        if entity is not None:
            value = ruleset.card(entity.card_id).move_speed_mtile_per_s
            # A card stat is timeless. Associate it with the observation's
            # endpoint so normal tick-tolerance accounting remains explicit.
            tick = measurement.observed.observed_tick
    elif spec.extractor_type in _FINAL_EXTRACTORS:
        towers_only = spec.extractor_type.startswith("final_tower_")
        entity = _unique_final_entity(
            state,
            spec.filters,
            towers_only=towers_only,
            sample_id=sample_id,
        )
        if entity is not None:
            value = entity.hp if spec.extractor_type.endswith("_hp") else entity.alive
            tick = state.tick
    elif spec.extractor_type in _SNAPSHOT_EXTRACTORS:
        assert spec.extractor_type in _SNAPSHOT_EXTRACTORS
        assert spec.tick is not None
        snapshot = None if snapshots is None else snapshots.get(spec.tick)
        if snapshot is not None:
            towers_only = spec.extractor_type.startswith("tower_")
            entity = _unique_final_entity(
                snapshot,
                spec.filters,
                towers_only=towers_only,
                sample_id=sample_id,
            )
            if entity is not None:
                field = spec.extractor_type.removeprefix("entity_").removeprefix("tower_")
                field = field.removesuffix("_at_tick")
                value = getattr(entity, field)
                tick = spec.tick
    else:
        assert spec.extractor_type in _DUAL_SNAPSHOT_EXTRACTORS
        assert spec.start_tick is not None and spec.end_tick is not None
        start = None if snapshots is None else snapshots.get(spec.start_tick)
        end = None if snapshots is None else snapshots.get(spec.end_tick)
        if start is not None and end is not None:
            start_entity = _unique_final_entity(
                start,
                spec.filters,
                towers_only=False,
                sample_id=sample_id,
            )
            end_entity = _unique_final_entity(
                end,
                spec.filters,
                towers_only=False,
                sample_id=sample_id,
            )
            if start_entity is not None and end_entity is not None:
                duration_us = end.elapsed_us - start.elapsed_us
                if duration_us <= 0:
                    raise ValidationCorpusError(
                        f"extractor for sample {sample_id!r} has non-positive "
                        "snapshot duration"
                    )
                displacement = distance_mtile(
                    start_entity.x_mtile,
                    start_entity.y_mtile,
                    end_entity.x_mtile,
                    end_entity.y_mtile,
                )
                value = (displacement * SECOND_US + duration_us // 2) // duration_us
                tick = spec.end_tick

    if value is _MISSING or value is None:
        return None
    try:
        checked = _scalar(value, f"simulated value for {sample_id}")
        return SimulatedMechanicSample(sample_id=sample_id, value=checked, tick=tick)
    except (TypeError, ValueError) as error:
        raise ValidationCorpusError(
            f"extractor for sample {sample_id!r} produced an invalid scalar: {error}"
        ) from error


def _compare_case_traces(
    case: ValidationCase,
    events: tuple[EventRecord, ...],
) -> tuple[TraceComparison, ...]:
    result: list[TraceComparison] = []
    for trace in case.traces:
        selected = _matching_events(
            events,
            event_kind=None,
            filters=trace.filters,
        )
        result.append(compare_trace(trace.observed, selected))
    return tuple(result)


def _verify_engine_contract(engine: BattleEngine, corpus: ValidationCorpus) -> None:
    if corpus.engine_version != ENGINE_VERSION:
        raise ValidationCorpusError(
            f"corpus engine version does not match {ENGINE_VERSION!r}"
        )
    if engine.ruleset.ruleset_id != corpus.ruleset_id:
        raise ValidationCorpusError("engine ruleset ID does not match corpus")
    if engine.ruleset.content_hash != corpus.ruleset_hash:
        raise ValidationCorpusError("engine ruleset hash does not match corpus")
    engine.ruleset.verify_hash()


def evaluate_fidelity_corpus(
    engine: BattleEngine,
    corpus: ValidationCorpus,
    *,
    split: DatasetSplit | str = DatasetSplit.HELDOUT,
    confidence_level: float = 0.95,
) -> tuple[FidelityReport, tuple[SampleComparison, ...], tuple[TraceComparison, ...]]:
    """Run all preassigned cases, then aggregate only the requested split."""

    target = _split(split, "requested report split")
    _verify_engine_contract(engine, corpus)
    observed_samples: list[ObservedMechanicSample] = []
    simulated_samples: list[SimulatedMechanicSample] = []
    trace_comparisons: list[TraceComparison] = []
    case_results: list[dict[str, object]] = []

    # Case order cannot influence a battle because each scenario constructs a
    # new seeded state.  Sorting makes execution and error selection canonical.
    excluded_by_split: dict[str, dict[str, int]] = {}
    for case in corpus.cases:
        if case.split is target:
            continue
        counts = excluded_by_split.setdefault(
            case.split.value,
            {"sample_comparisons": 0, "trace_comparisons": 0},
        )
        counts["sample_comparisons"] += len(case.measurements)
        counts["trace_comparisons"] += len(case.traces)

    for case in sorted(
        (item for item in corpus.cases if item.split is target),
        key=lambda item: item.case_id,
    ):
        scenario = _load_case_scenario(case, corpus)
        snapshot_ticks = tuple(
            sorted(
                {
                tick
                for measurement in case.measurements
                for tick in (
                    measurement.extractor.tick,
                    measurement.extractor.start_tick,
                    measurement.extractor.end_tick,
                )
                if tick is not None
                }
            )
        )
        state, snapshots = run_scenario_with_snapshots(
            engine,
            scenario,
            snapshot_ticks=snapshot_ticks,
        )
        events = normalized_state_events(state)
        case_results.append(
            {
                "case_id": case.case_id,
                "scenario_id": scenario.scenario_id,
                "source_id": case.evidence.source_id,
                "group_id": case.evidence.group_id,
                "media_hash": case.evidence.media_hash,
                "frame_start": case.evidence.frame_start,
                "frame_end": case.evidence.frame_end,
                "final_tick": state.tick,
                "terminal": state.terminal,
                "state_hash": state.state_hash(),
                "event_log_hash": state.event_log_hash(),
                "replay_hash": state.replay_hash(),
                "sample_count": len(case.measurements),
                "trace_count": len(case.traces),
            }
        )
        for measurement in case.measurements:
            observed_samples.append(measurement.observed)
            simulated = extract_simulated_measurement(
                measurement,
                state,
                events=events,
                snapshots=snapshots,
                ruleset=engine.ruleset,
            )
            if simulated is not None:
                simulated_samples.append(simulated)
        trace_comparisons.extend(_compare_case_traces(case, events))

    sample_comparisons = compare_samples(observed_samples, simulated_samples)
    traces = tuple(trace_comparisons)
    report = build_fidelity_report(
        ruleset_id=corpus.ruleset_id,
        ruleset_hash=corpus.ruleset_hash,
        sample_comparisons=sample_comparisons,
        trace_comparisons=traces,
        split=target,
        confidence_level=confidence_level,
        corpus_id=corpus.corpus_id,
        corpus_hash=corpus.content_hash,
        engine_version=ENGINE_VERSION,
        tick_us=engine.ruleset.tick_us,
        excluded_by_split_counts=excluded_by_split,
        case_results=case_results,
    )
    return report, sample_comparisons, traces


def run_fidelity_corpus(
    engine: BattleEngine,
    path: str | Path,
    *,
    split: DatasetSplit | str = DatasetSplit.HELDOUT,
    confidence_level: float = 0.95,
    expected_corpus_hash: str | None = None,
) -> FidelityReport:
    """CLI-facing one-shot loader, runner, comparator, and report builder."""

    corpus = load_validation_corpus_pinned(path, expected_hash=expected_corpus_hash)
    report, _, _ = evaluate_fidelity_corpus(
        engine,
        corpus,
        split=split,
        confidence_level=confidence_level,
    )
    return report


def apply_fidelity_gate(
    report: FidelityReport,
    *,
    min_observations: int = 1,
    min_agreement_rate: float | None = None,
    required_mechanics: Iterable[str] = (),
) -> FidelityReport:
    """Attach deterministic CI acceptance criteria to a held-out report."""

    if type(min_observations) is not int or min_observations < 1:
        raise ValueError("min_observations must be a positive integer")
    if min_agreement_rate is not None and (
        type(min_agreement_rate) not in (int, float)
        or not math.isfinite(min_agreement_rate)
        or not 0.0 <= min_agreement_rate <= 1.0
    ):
        raise ValueError("min_agreement_rate must be between 0 and 1")
    required = tuple(sorted({_name(item, "required mechanic") for item in required_mechanics}))
    sample_metrics = report.overall["samples"]
    trace_metrics = report.overall["traces"]
    assert isinstance(sample_metrics, Mapping) and isinstance(trace_metrics, Mapping)
    observation_count = int(sample_metrics["count"]) + int(trace_metrics["count"])
    agreement_count = int(sample_metrics["agreement_count"]) + int(
        trace_metrics["agreement_count"]
    )
    agreement_rate = agreement_count / observation_count if observation_count else None
    missing_mechanics = sorted(set(required) - set(report.mechanics))
    failures: list[str] = []
    if observation_count < min_observations:
        failures.append(
            f"observation_count {observation_count} is below required {min_observations}"
        )
    if (
        min_agreement_rate is not None
        and (agreement_rate is None or agreement_rate < min_agreement_rate)
    ):
        failures.append(
            f"agreement_rate {agreement_rate!r} is below required {min_agreement_rate}"
        )
    if missing_mechanics:
        failures.append("missing required mechanics: " + ", ".join(missing_mechanics))
    return replace(
        report,
        gate=MappingProxyType(
            {
                "passed": not failures,
                "observation_count": observation_count,
                "agreement_count": agreement_count,
                "agreement_rate": agreement_rate,
                "min_observations": min_observations,
                "min_agreement_rate": min_agreement_rate,
                "required_mechanics": list(required),
                "failures": failures,
            }
        ),
    )


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "ExtractorSpec",
    "ScalarObservationSpec",
    "TraceObservationSpec",
    "ValidationCase",
    "ValidationCorpus",
    "ValidationCorpusError",
    "apply_fidelity_gate",
    "evaluate_fidelity_corpus",
    "extract_simulated_measurement",
    "load_validation_corpus",
    "load_validation_corpus_pinned",
    "normalized_state_events",
    "run_fidelity_corpus",
    "validation_corpus_from_dict",
]
