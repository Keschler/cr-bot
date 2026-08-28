"""Deterministic active-learning helpers for selecting the next physical probe."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from ..engine import BASE_HOG_CYCLE_DECK, ENGINE_VERSION
from ..ruleset import load_fixed_ruleset
from .schema import (
    DeviceSpec,
    EvidenceSplit,
    ExperimentSpec,
    InitialConditions,
    MeasurementSpec,
    PhysicalAction,
    PhysicalLabError,
    Trigger,
    TriggerType,
    canonical_hash,
)


@dataclass(frozen=True, slots=True)
class PriorityInputs:
    frequency: float
    decision_impact: float
    uncertainty: float
    failure_rate: float
    estimated_run_cost: float

    def __post_init__(self) -> None:
        for field_name in (
            "frequency",
            "decision_impact",
            "uncertainty",
            "failure_rate",
            "estimated_run_cost",
        ):
            value = getattr(self, field_name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise PhysicalLabError(f"priority.{field_name} must be finite and non-negative")
        if self.estimated_run_cost <= 0:
            raise PhysicalLabError("priority.estimated_run_cost must be positive")

    @property
    def score(self) -> float:
        return (
            self.frequency
            * self.decision_impact
            * self.uncertainty
            * self.failure_rate
            / self.estimated_run_cost
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frequency": self.frequency,
            "decision_impact": self.decision_impact,
            "uncertainty": self.uncertainty,
            "failure_rate": self.failure_rate,
            "estimated_run_cost": self.estimated_run_cost,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class PlannedExperiment:
    spec: ExperimentSpec
    priority: PriorityInputs
    source_question: str

    def to_dict(self) -> dict[str, object]:
        result = self.spec.to_dict(include_hash=True)
        result["planning"] = {
            "priority": self.priority.to_dict(),
            "source_question": self.source_question,
        }
        return result


def _safe_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return value or "probe"


def _offline_devices() -> dict[str, DeviceSpec]:
    return ExperimentSpec.offline_default_devices()


def hog_cannon_probe(
    *,
    experiment_id: str = "hog_cannon_pull_0142",
    capture_group_id: str = "lab-session-offline-calibration",
    evidence_split: EvidenceSplit | str = EvidenceSplit.CALIBRATION,
    seed: int = 0,
    source_question: str = "hog-cannon-pull",
    metadata: Mapping[str, Any] | None = None,
) -> ExperimentSpec:
    """Return the canonical Phase-0 probe from the lab specification."""

    ruleset = load_fixed_ruleset()
    deck = tuple(BASE_HOG_CYCLE_DECK)
    opening_slots = {
        card_id: slot for slot, card_id in enumerate(deck[:4])
    }
    initial = InitialConditions(
        tower_state="default",
        requested_elixir_milli={"A": 10_000, "B": 10_000},
        decks={"A": deck, "B": deck},
        # The physical Testspiel fixed-deck option makes the first four deck
        # entries the opening hand.  Preserve the complete contract in the
        # experiment instead of only recording cards used by this probe.
        hand_slots={"A": opening_slots, "B": opening_slots},
    )
    actions = (
        PhysicalAction(
            action_id="deploy-hog",
            side="A",
            card_id="hog-rider",
            arena_cell=(3, 20),
            card_slot=0,
            trigger=Trigger(TriggerType.MATCH_TIME_US, value=0),
        ),
        PhysicalAction(
            action_id="deploy-cannon",
            side="B",
            card_id="cannon",
            arena_cell=(8, 13),
            card_slot=1,
            trigger=Trigger(
                TriggerType.AFTER_OBSERVATION,
                value=17_000,
                event="hog_crosses_y_mtile",
            ),
        ),
    )
    measurements = (
        MeasurementSpec("hog_isolated_movement"),
        MeasurementSpec("hog_cannon_targeting", requires_direct_timing=True),
        MeasurementSpec("hog_cannon_pull_trajectory"),
        MeasurementSpec("cannon_lifetime_hp_decay"),
        MeasurementSpec("tower_hit_count"),
    )
    merged_metadata = {
        "planner": "physical_lab_phase0",
        "source_question": source_question,
        "offline_ready": True,
    }
    if metadata:
        merged_metadata.update(metadata)
    return ExperimentSpec(
        experiment_id=experiment_id,
        ruleset_id=ruleset.ruleset_id,
        ruleset_hash=ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        capture_group_id=capture_group_id,
        evidence_split=evidence_split,
        devices=_offline_devices(),
        initial_conditions=initial,
        actions=actions,
        measurements=measurements,
        seed=seed,
        provenance={
            "game_version": "unknown-unconnected",
            "patch_id": "unknown-unconnected",
            "operator_id": "offline-harness",
            "capture_transport": "fake",
        },
        metadata=merged_metadata,
    )


def _priority_from_question(item: Mapping[str, Any]) -> PriorityInputs:
    def value(name: str, default: float) -> float:
        raw = item.get(name, default)
        try:
            return float(raw)
        except (TypeError, ValueError) as error:
            raise PhysicalLabError(f"unknown-behavior {name} must be numeric") from error

    return PriorityInputs(
        frequency=value("frequency", 1.0),
        decision_impact=value("decision_impact", 1.0),
        uncertainty=value("uncertainty", 1.0),
        failure_rate=value("failure_rate", 1.0),
        estimated_run_cost=value("estimated_run_cost", 1.0),
    )


def load_questions(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalLabError(f"cannot load unknown behaviors {source}: {error}") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("items"), list):
        raise PhysicalLabError("unknown behaviors document must contain an items array")
    return tuple(item for item in raw["items"] if isinstance(item, Mapping) and item.get("status") == "open")


def load_readiness_questions(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Extract unresolved non-held-out edges from a readiness report.

    Held-out rows are intentionally ignored: the active-learning planner may
    propose a new independent probe for a known subsystem, but it cannot tune
    the simulator from the sealed held-out result itself.
    """

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicalLabError(f"cannot load readiness report {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise PhysicalLabError("readiness report must be an object")
    rows: list[Mapping[str, Any]] = []
    for key in ("mechanics", "requirements", "failures"):
        candidate = raw.get(key)
        if isinstance(candidate, list):
            rows.extend(item for item in candidate if isinstance(item, Mapping))
    summary = raw.get("summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("failures"), list):
        rows.extend(item if isinstance(item, Mapping) else {"id": str(item)} for item in summary["failures"])
    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        split = row.get("split") or row.get("evidence_split")
        if split == EvidenceSplit.HELDOUT.value:
            continue
        ready = row.get("ready")
        status = str(row.get("status", "")).casefold()
        if ready is True or status in {"ready", "passed", "complete"}:
            continue
        identifier = row.get("id") or row.get("mechanic") or row.get("name") or f"readiness-failure-{index:04d}"
        result.append({**dict(row), "id": str(identifier), "status": "open"})
    return tuple(result)


def _question_measurement_id(question_id: str) -> str:
    return _safe_id(question_id.replace(".", "-"))[:70]


def plan_from_questions(
    questions: Iterable[Mapping[str, Any]],
    *,
    capture_group_id: str = "lab-session-offline-calibration",
    evidence_split: EvidenceSplit | str = EvidenceSplit.CALIBRATION,
    limit: int | None = None,
) -> tuple[PlannedExperiment, ...]:
    """Turn open questions into deterministic, reviewable probe records."""

    planned: list[PlannedExperiment] = []
    for question in questions:
        question_id = str(question.get("id", "open-question"))
        priority = _priority_from_question(question)
        if question_id == "cannon.first-hit":
            spec = hog_cannon_probe(
                experiment_id=f"probe-{_question_measurement_id(question_id)}",
                capture_group_id=capture_group_id,
                evidence_split=evidence_split,
                source_question=question_id,
                metadata={"question_id": question_id},
            )
        else:
            ruleset = load_fixed_ruleset()
            measurement_name = _question_measurement_id(question_id)
            spec = ExperimentSpec(
                experiment_id=f"probe-{measurement_name}",
                ruleset_id=ruleset.ruleset_id,
                ruleset_hash=ruleset.content_hash,
                engine_version=ENGINE_VERSION,
                capture_group_id=capture_group_id,
                evidence_split=evidence_split,
                devices=_offline_devices(),
                actions=(),
                measurements=(MeasurementSpec(measurement_name),),
                provenance={
                    "game_version": "unknown-unconnected",
                    "patch_id": "unknown-unconnected",
                    "operator_id": "offline-harness",
                    "capture_transport": "fake",
                },
                metadata={
                    "planner": "physical_lab_phase0",
                    "question_id": question_id,
                    "requires_template_review": True,
                },
            )
        planned.append(
            PlannedExperiment(
                spec=spec,
                priority=priority,
                source_question=str(question.get("id", question_id)),
            )
        )
    planned.sort(key=lambda item: (-item.priority.score, item.spec.experiment_id))
    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise PhysicalLabError("plan limit must be a non-negative integer")
        planned = planned[:limit]
    return tuple(planned)


def plan_from_readiness(
    path: str | Path,
    *,
    capture_group_id: str = "lab-session-offline-calibration",
    evidence_split: EvidenceSplit | str = EvidenceSplit.CALIBRATION,
    limit: int | None = None,
) -> tuple[PlannedExperiment, ...]:
    return plan_from_questions(
        load_readiness_questions(path),
        capture_group_id=capture_group_id,
        evidence_split=evidence_split,
        limit=limit,
    )


def write_plan(path: str | Path, planned: Iterable[PlannedExperiment | ExperimentSpec]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in planned:
        if isinstance(item, ExperimentSpec):
            rows.append(item.to_dict(include_hash=True))
        else:
            rows.append(item.to_dict())
    lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in rows]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_plan_line(path: str | Path, line_number: int = 1) -> ExperimentSpec:
    source = Path(path)
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= line_number <= len(lines):
        raise PhysicalLabError(f"plan line {line_number} is not present in {source}")
    try:
        raw = json.loads(lines[line_number - 1])
    except json.JSONDecodeError as error:
        raise PhysicalLabError(f"invalid JSON in plan line {line_number}: {error}") from error
    if isinstance(raw, Mapping) and isinstance(raw.get("planning"), Mapping):
        raw = {key: value for key, value in raw.items() if key != "planning"}
    return ExperimentSpec.from_dict(raw)


__all__ = [
    "PlannedExperiment",
    "PriorityInputs",
    "hog_cannon_probe",
    "load_plan_line",
    "load_readiness_questions",
    "load_questions",
    "plan_from_readiness",
    "plan_from_questions",
    "write_plan",
]
