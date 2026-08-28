"""Serializable behavior-cloning curriculum configuration.

This module describes *what a future learner should consume*; it does not
load demonstrations, train a model, or claim that any teacher is expert.  A
curriculum makes the confidence policy and phase boundaries explicit so a
rollout/training service can apply the same decisions after a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping


CURRICULUM_SCHEMA_VERSION = 1
ConfidenceBand = Literal["high", "medium", "low", "rejected"]


class CurriculumConfigurationError(ValueError):
    """Raised when a curriculum is ambiguous or internally inconsistent."""


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CurriculumConfigurationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise CurriculumConfigurationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise CurriculumConfigurationError(f"{name} must be >= {minimum}")
    return value


def _finite_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurriculumConfigurationError(f"{name} must be a finite number in [0, 1]")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise CurriculumConfigurationError(f"{name} must be a finite number in [0, 1]")
    return converted


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurriculumConfigurationError(f"{name} must be a finite non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise CurriculumConfigurationError(f"{name} must be a finite non-negative number")
    return converted


@dataclass(frozen=True, slots=True)
class BCTeacherConfidencePolicy:
    """Map per-example teacher confidence to an explicit BC loss weight.

    Confidence is metadata supplied by the demonstration producer.  It is
    not inferred here and is not evidence that a teacher is optimal.  The
    default policy keeps high-confidence labels at full weight, discounts
    medium/low-confidence labels, and drops labels below the low threshold.
    """

    high_threshold: float = 0.90
    medium_threshold: float = 0.75
    low_threshold: float = 0.50
    high_weight: float = 1.00
    medium_weight: float = 0.50
    low_weight: float = 0.25

    def __post_init__(self) -> None:
        thresholds = (
            ("high_threshold", self.high_threshold),
            ("medium_threshold", self.medium_threshold),
            ("low_threshold", self.low_threshold),
        )
        for name, value in thresholds:
            _finite_unit(value, name)
        if not (
            self.low_threshold <= self.medium_threshold <= self.high_threshold
        ):
            raise CurriculumConfigurationError(
                "confidence thresholds must satisfy low <= medium <= high"
            )

        weights = (
            ("high_weight", self.high_weight),
            ("medium_weight", self.medium_weight),
            ("low_weight", self.low_weight),
        )
        for name, value in weights:
            _finite_unit(value, name)
        if not self.low_weight <= self.medium_weight <= self.high_weight:
            raise CurriculumConfigurationError(
                "confidence weights must satisfy low <= medium <= high"
            )

    def band(self, confidence: float) -> ConfidenceBand:
        """Return the configured band for one finite confidence value."""

        confidence = _finite_unit(confidence, "teacher confidence")
        if confidence >= self.high_threshold:
            return "high"
        if confidence >= self.medium_threshold:
            return "medium"
        if confidence >= self.low_threshold:
            return "low"
        return "rejected"

    def weight(self, confidence: float) -> float:
        """Return the confidence multiplier used by a BC objective."""

        return {
            "high": self.high_weight,
            "medium": self.medium_weight,
            "low": self.low_weight,
            "rejected": 0.0,
        }[self.band(confidence)]

    def as_dict(self) -> dict[str, object]:
        return {
            "high_threshold": self.high_threshold,
            "medium_threshold": self.medium_threshold,
            "low_threshold": self.low_threshold,
            "high_weight": self.high_weight,
            "medium_weight": self.medium_weight,
            "low_weight": self.low_weight,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BCTeacherConfidencePolicy":
        if not isinstance(raw, Mapping):
            raise CurriculumConfigurationError("confidence policy must be an object")
        defaults = cls()
        return cls(
            high_threshold=raw.get("high_threshold", defaults.high_threshold),
            medium_threshold=raw.get("medium_threshold", defaults.medium_threshold),
            low_threshold=raw.get("low_threshold", defaults.low_threshold),
            high_weight=raw.get("high_weight", defaults.high_weight),
            medium_weight=raw.get("medium_weight", defaults.medium_weight),
            low_weight=raw.get("low_weight", defaults.low_weight),
        )


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    """One inclusive/exclusive phase in a BC-to-RL curriculum schedule."""

    phase_id: str
    start_step: int
    end_step: int | None = None
    teacher_id: str = "heuristic"
    bc_loss_coefficient: float = 1.0
    confidence_policy: BCTeacherConfidencePolicy = field(
        default_factory=BCTeacherConfidencePolicy
    )
    description: str = ""

    def __post_init__(self) -> None:
        _string(self.phase_id, "phase_id")
        _integer(self.start_step, "start_step", minimum=0)
        if self.end_step is not None:
            _integer(self.end_step, "end_step", minimum=1)
            if self.end_step <= self.start_step:
                raise CurriculumConfigurationError(
                    "end_step must be greater than start_step"
                )
        _string(self.teacher_id, "teacher_id")
        _finite_nonnegative(self.bc_loss_coefficient, "bc_loss_coefficient")
        if not isinstance(self.confidence_policy, BCTeacherConfidencePolicy):
            raise CurriculumConfigurationError(
                "confidence_policy must be a BCTeacherConfidencePolicy"
            )
        if not isinstance(self.description, str):
            raise CurriculumConfigurationError("description must be a string")

    def contains(self, step: int) -> bool:
        """Return whether ``step`` belongs to this phase."""

        _integer(step, "step", minimum=0)
        return self.start_step <= step and (
            self.end_step is None or step < self.end_step
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "teacher_id": self.teacher_id,
            "bc_loss_coefficient": self.bc_loss_coefficient,
            "confidence_policy": self.confidence_policy.as_dict(),
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CurriculumPhase":
        if not isinstance(raw, Mapping):
            raise CurriculumConfigurationError("curriculum phase must be an object")
        policy_raw = raw.get("confidence_policy", {})
        return cls(
            phase_id=raw.get("phase_id", ""),
            start_step=raw.get("start_step", 0),
            end_step=raw.get("end_step"),
            teacher_id=raw.get("teacher_id", "heuristic"),
            bc_loss_coefficient=raw.get("bc_loss_coefficient", 1.0),
            confidence_policy=BCTeacherConfidencePolicy.from_mapping(policy_raw),
            description=raw.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class BCDecision:
    """Serializable resolution of one teacher label under one phase."""

    phase_id: str
    teacher_id: str
    confidence: float
    confidence_band: ConfidenceBand
    confidence_weight: float
    loss_weight: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "teacher_id": self.teacher_id,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band,
            "confidence_weight": self.confidence_weight,
            "loss_weight": self.loss_weight,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    """Contiguous, restartable curriculum phases for a future learner."""

    schedule_id: str
    phases: tuple[CurriculumPhase, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        _string(self.schedule_id, "schedule_id")
        _integer(self.seed, "seed")
        if not self.phases:
            raise CurriculumConfigurationError("curriculum must contain at least one phase")
        previous_end: int | None = None
        for index, phase in enumerate(self.phases):
            if not isinstance(phase, CurriculumPhase):
                raise CurriculumConfigurationError(
                    f"phases[{index}] must be a CurriculumPhase"
                )
            if index == 0 and phase.start_step != 0:
                raise CurriculumConfigurationError(
                    "the first curriculum phase must start at step 0"
                )
            if previous_end is not None and phase.start_step != previous_end:
                raise CurriculumConfigurationError(
                    "curriculum phases must be contiguous with no gaps or overlaps"
                )
            if previous_end is None and index > 0:
                raise CurriculumConfigurationError(
                    "an open-ended phase must be the final curriculum phase"
                )
            previous_end = phase.end_step

    def phase_at(self, step: int) -> CurriculumPhase:
        """Resolve the phase at a non-negative learner step."""

        _integer(step, "step", minimum=0)
        for phase in self.phases:
            if phase.contains(step):
                return phase
        raise CurriculumConfigurationError(
            f"curriculum {self.schedule_id!r} has no phase covering step {step}"
        )

    def decide(self, step: int, confidence: float) -> BCDecision:
        """Resolve confidence and phase into an explicit BC weighting decision."""

        phase = self.phase_at(step)
        band = phase.confidence_policy.band(confidence)
        confidence_weight = phase.confidence_policy.weight(confidence)
        return BCDecision(
            phase_id=phase.phase_id,
            teacher_id=phase.teacher_id,
            confidence=float(confidence),
            confidence_band=band,
            confidence_weight=confidence_weight,
            loss_weight=phase.bc_loss_coefficient * confidence_weight,
            accepted=confidence_weight > 0.0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CURRICULUM_SCHEMA_VERSION,
            "schedule_id": self.schedule_id,
            "seed": self.seed,
            "phases": [phase.as_dict() for phase in self.phases],
        }

    def to_json(self) -> str:
        """Return stable JSON without writing or claiming a training artifact."""

        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CurriculumSchedule":
        if not isinstance(raw, Mapping):
            raise CurriculumConfigurationError("curriculum schedule must be an object")
        schema_version = raw.get("schema_version", CURRICULUM_SCHEMA_VERSION)
        if schema_version != CURRICULUM_SCHEMA_VERSION:
            raise CurriculumConfigurationError(
                f"unsupported curriculum schema: {schema_version!r}"
            )
        raw_phases = raw.get("phases")
        if not isinstance(raw_phases, (list, tuple)):
            raise CurriculumConfigurationError("phases must be a list of objects")
        return cls(
            schedule_id=raw.get("schedule_id", ""),
            seed=raw.get("seed", 0),
            phases=tuple(CurriculumPhase.from_mapping(item) for item in raw_phases),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CurriculumSchedule":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CurriculumConfigurationError(
                f"cannot load curriculum schedule {source}: {error}"
            ) from error
        return cls.from_mapping(raw)


__all__ = [
    "BCDecision",
    "BCTeacherConfidencePolicy",
    "CURRICULUM_SCHEMA_VERSION",
    "CurriculumConfigurationError",
    "CurriculumPhase",
    "CurriculumSchedule",
]
