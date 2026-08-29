"""Serializable behavior-cloning and strategic curriculum configuration.

The behavior-cloning objects describe *what a future learner should consume*;
the strategic objects describe which opponent distribution a current learner
should face. This module does not load demonstrations, train a model, or claim
that any teacher is expert. A curriculum makes confidence/phase boundaries
explicit so a rollout/training service can reproduce them after a restart.
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


STRATEGIC_CURRICULUM_SCHEMA_VERSION = 1


def _name_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise CurriculumConfigurationError(f"{name} must be a list of names")
    names = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not names:
        raise CurriculumConfigurationError(f"{name} must not be empty")
    if len(set(names)) != len(names):
        raise CurriculumConfigurationError(f"{name} must not contain duplicates")
    return names


@dataclass(frozen=True, slots=True)
class StrategicCurriculumStage:
    """One rollout-segment stage in the teacher-free strategic curriculum.

    These fields choose the opponent distribution only.  They do not prescribe
    the learner's card, timing, lane, or placement.  The stage vocabulary is
    intentionally kept as names so the simulator-specific generalized runner
    can validate it against the active opponent pool.
    """

    stage_id: str
    start_segment: int
    end_segment: int | None
    archetypes: tuple[str, ...]
    strategies: tuple[str, ...]
    description: str = ""
    # Ordered source names and normalized weights.  These labels describe
    # opponent/scenario sampling only; they never prescribe the learner's
    # action.  An empty tuple keeps compatibility with caller-defined stages
    # that only constrain archetypes and strategies.
    sampling_mix: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _string(self.stage_id, "stage_id")
        _integer(self.start_segment, "start_segment", minimum=0)
        if self.end_segment is not None:
            _integer(self.end_segment, "end_segment", minimum=1)
            if self.end_segment <= self.start_segment:
                raise CurriculumConfigurationError(
                    "end_segment must be greater than start_segment"
                )
        _name_tuple(self.archetypes, "archetypes")
        _name_tuple(self.strategies, "strategies")
        if not isinstance(self.description, str):
            raise CurriculumConfigurationError("description must be a string")
        if not isinstance(self.sampling_mix, tuple):
            raise CurriculumConfigurationError("sampling_mix must be a tuple")
        if self.sampling_mix:
            total = 0.0
            seen: set[str] = set()
            for index, item in enumerate(self.sampling_mix):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise CurriculumConfigurationError(
                        f"sampling_mix[{index}] must be a (source, weight) tuple"
                    )
                source, weight = item
                _string(source, f"sampling_mix[{index}].source")
                if source in seen:
                    raise CurriculumConfigurationError(
                        "sampling_mix sources must not contain duplicates"
                    )
                seen.add(source)
                total += _finite_unit(weight, f"sampling_mix[{index}].weight")
                if float(weight) <= 0.0:
                    raise CurriculumConfigurationError(
                        f"sampling_mix[{index}].weight must be greater than zero"
                    )
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise CurriculumConfigurationError(
                    "sampling_mix weights must sum to 1"
                )

    def contains(self, segment: int) -> bool:
        _integer(segment, "segment", minimum=0)
        return self.start_segment <= segment and (
            self.end_segment is None or segment < self.end_segment
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "archetypes": list(self.archetypes),
            "strategies": list(self.strategies),
            "description": self.description,
            "sampling_mix": [
                {"source": source, "weight": weight}
                for source, weight in self.sampling_mix
            ],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StrategicCurriculumStage":
        if not isinstance(raw, Mapping):
            raise CurriculumConfigurationError(
                "strategic curriculum stage must be an object"
            )
        raw_mix = raw.get("sampling_mix", ())
        if isinstance(raw_mix, Mapping):
            sampling_mix = tuple(
                (source, weight) for source, weight in raw_mix.items()
            )
        elif isinstance(raw_mix, (list, tuple)):
            parsed_mix: list[tuple[str, float]] = []
            for index, item in enumerate(raw_mix):
                if isinstance(item, Mapping):
                    parsed_mix.append(
                        (item.get("source", ""), item.get("weight", 0.0))
                    )
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    parsed_mix.append((item[0], item[1]))
                else:
                    raise CurriculumConfigurationError(
                        f"sampling_mix[{index}] must be an object or pair"
                    )
            sampling_mix = tuple(parsed_mix)
        else:
            raise CurriculumConfigurationError(
                "sampling_mix must be an object or list of objects"
            )
        return cls(
            stage_id=raw.get("stage_id", ""),
            start_segment=raw.get("start_segment", 0),
            end_segment=raw.get("end_segment"),
            archetypes=_name_tuple(raw.get("archetypes", []), "archetypes"),
            strategies=_name_tuple(raw.get("strategies", []), "strategies"),
            description=raw.get("description", ""),
            sampling_mix=sampling_mix,
        )


@dataclass(frozen=True, slots=True)
class StrategicCurriculum:
    """Restartable, non-prescriptive opponent curriculum for generalized PPO."""

    schedule_id: str
    stages: tuple[StrategicCurriculumStage, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        _string(self.schedule_id, "schedule_id")
        _integer(self.seed, "seed")
        if not self.stages:
            raise CurriculumConfigurationError(
                "strategic curriculum must contain at least one stage"
            )
        previous_end: int | None = None
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, StrategicCurriculumStage):
                raise CurriculumConfigurationError(
                    f"stages[{index}] must be a StrategicCurriculumStage"
                )
            if index == 0 and stage.start_segment != 0:
                raise CurriculumConfigurationError(
                    "the first strategic curriculum stage must start at segment 0"
                )
            if previous_end is not None and stage.start_segment != previous_end:
                raise CurriculumConfigurationError(
                    "strategic curriculum stages must be contiguous"
                )
            if previous_end is None and index > 0:
                raise CurriculumConfigurationError(
                    "an open-ended strategic stage must be final"
                )
            previous_end = stage.end_segment

    def stage_at(self, segment: int) -> StrategicCurriculumStage:
        _integer(segment, "segment", minimum=0)
        for stage in self.stages:
            if stage.contains(segment):
                return stage
        raise CurriculumConfigurationError(
            f"strategic curriculum {self.schedule_id!r} has no stage at segment {segment}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRATEGIC_CURRICULUM_SCHEMA_VERSION,
            "schedule_id": self.schedule_id,
            "seed": self.seed,
            "stages": [stage.as_dict() for stage in self.stages],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StrategicCurriculum":
        if not isinstance(raw, Mapping):
            raise CurriculumConfigurationError("strategic curriculum must be an object")
        schema_version = raw.get(
            "schema_version", STRATEGIC_CURRICULUM_SCHEMA_VERSION
        )
        if schema_version != STRATEGIC_CURRICULUM_SCHEMA_VERSION:
            raise CurriculumConfigurationError(
                f"unsupported strategic curriculum schema: {schema_version!r}"
            )
        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, (list, tuple)):
            raise CurriculumConfigurationError("stages must be a list of objects")
        return cls(
            schedule_id=raw.get("schedule_id", ""),
            seed=raw.get("seed", 0),
            stages=tuple(
                StrategicCurriculumStage.from_mapping(item) for item in raw_stages
            ),
        )


def default_strategic_curriculum() -> StrategicCurriculum:
    """Return the documented local-first curriculum schedule."""

    all_archetypes = (
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "air-beatdown",
        "siege-bait",
        "random-legal",
    )
    all_strategies = (
        "aggressive-pressure",
        "defensive-cycle",
        "beatdown",
        "siege-bait",
        "random-legal",
    )
    basic_mechanics_mix = (
        ("isolated-offense", 0.25),
        ("ground-defense", 0.25),
        ("air-defense", 0.20),
        ("spell-situations", 0.15),
        ("kiting-cycling-elixir", 0.15),
    )
    scripted_curriculum_mix = (
        ("phase-1-rehearsal", 0.20),
        ("passive-random-legal", 0.20),
        ("simple-win-condition", 0.20),
        ("reactive-defensive-aggressive", 0.20),
        ("randomized-tempo-placement", 0.20),
    )
    meta_deck_mix = (
        ("uniform-archetypes", 0.35),
        ("weakness-prioritized-matchups", 0.30),
        ("earlier-curriculum-rehearsal", 0.20),
        ("randomized-variants", 0.15),
    )
    historical_mix = (
        ("scripted-meta-anchors", 0.30),
        ("pfsp-historical", 0.30),
        ("newest-frozen-main", 0.20),
        ("random-historical-checkpoint", 0.10),
        ("exploiters-adversarial", 0.10),
    )
    league_mix = (
        ("main-learner", 0.25),
        ("main-exploiter", 0.15),
        ("league-exploiter", 0.15),
        ("historical-frozen", 0.30),
        ("scripted-anchor", 0.15),
    )
    return StrategicCurriculum(
        schedule_id="strategic-hog-v1",
        stages=(
            StrategicCurriculumStage(
                "mechanics-foundation",
                0,
                4,
                all_archetypes,
                all_strategies,
                "Short generated scenarios for action/placement and basic defense.",
                basic_mechanics_mix,
            ),
            StrategicCurriculumStage(
                "scripted-threat-expansion",
                4,
                12,
                all_archetypes,
                all_strategies,
                "Mix pressure, air, beatdown, siege/bait, and random legal controllers.",
                scripted_curriculum_mix,
            ),
            StrategicCurriculumStage(
                "meta-deck-diversity",
                12,
                32,
                all_archetypes,
                all_strategies,
                "Expand validated archetypes and held-out deck variants with rehearsal.",
                meta_deck_mix,
            ),
            StrategicCurriculumStage(
                "historical-league",
                32,
                64,
                all_archetypes,
                all_strategies,
                "Use frozen checkpoints/PFSP and later league opponents without forgetting anchors.",
                historical_mix,
            ),
            StrategicCurriculumStage(
                "small-league",
                64,
                None,
                all_archetypes,
                all_strategies,
                "Train main and exploiter roles against frozen history and fixed anchors.",
                league_mix,
            ),
        ),
    )


__all__ = [
    "BCDecision",
    "BCTeacherConfidencePolicy",
    "CURRICULUM_SCHEMA_VERSION",
    "CurriculumConfigurationError",
    "CurriculumPhase",
    "CurriculumSchedule",
    "STRATEGIC_CURRICULUM_SCHEMA_VERSION",
    "StrategicCurriculum",
    "StrategicCurriculumStage",
    "default_strategic_curriculum",
]
