"""Confidence-gated normalized observations emitted by a physical run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    EvidenceSplit,
    EvidenceStatus,
    ExperimentSpec,
    PhysicalLabError,
    canonical_hash,
    canonical_json,
)
from .sync import SynchronizationResult


OBSERVATION_SCHEMA_VERSION = 1


class ObservationCertainty(str, Enum):
    DIRECT = "direct"
    INFERRED = "inferred"


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalLabError(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhysicalLabError(f"{field_name} must be an integer >= {minimum}")
    return value


def _confidence(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise PhysicalLabError(f"{field_name} must be finite and between 0 and 1")
    return float(value)


def _side(value: object, field_name: str) -> str:
    value = _name(value, field_name).upper()
    if value not in {"A", "B"}:
        raise PhysicalLabError(f"{field_name} must be A or B")
    return value


def _number(value: object, field_name: str) -> int | float:
    if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
        raise PhysicalLabError(f"{field_name} must be a finite number")
    return value


@dataclass(frozen=True, slots=True)
class EntitySample:
    frame_index: int
    video_time_us: int
    match_time_us: int | None
    x_mtile: int
    y_mtile: int
    confidence: float
    source_capture_id: str
    hp_observed: int | None = None
    uncertainty_us: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "sample.frame_index"))
        object.__setattr__(self, "video_time_us", _integer(self.video_time_us, "sample.video_time_us"))
        if self.match_time_us is not None:
            object.__setattr__(self, "match_time_us", _integer(self.match_time_us, "sample.match_time_us"))
        object.__setattr__(self, "x_mtile", int(_number(self.x_mtile, "sample.x_mtile")))
        object.__setattr__(self, "y_mtile", int(_number(self.y_mtile, "sample.y_mtile")))
        object.__setattr__(self, "confidence", _confidence(self.confidence, "sample.confidence"))
        object.__setattr__(self, "source_capture_id", _name(self.source_capture_id, "sample.source_capture_id"))
        if self.hp_observed is not None:
            object.__setattr__(self, "hp_observed", _integer(self.hp_observed, "sample.hp_observed"))
        object.__setattr__(self, "uncertainty_us", _integer(self.uncertainty_us, "sample.uncertainty_us"))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_index": self.frame_index,
            "video_time_us": self.video_time_us,
            "x_mtile": self.x_mtile,
            "y_mtile": self.y_mtile,
            "confidence": self.confidence,
            "source_capture_id": self.source_capture_id,
            "uncertainty_us": self.uncertainty_us,
        }
        if self.match_time_us is not None:
            result["match_time_us"] = self.match_time_us
        if self.hp_observed is not None:
            result["hp_observed"] = self.hp_observed
        return result


@dataclass(frozen=True, slots=True)
class EntityObservation:
    """Stable card/owner observation; run-local detector UIDs are excluded."""

    stable_observation_id: str
    card_id: str
    owner: str
    samples: tuple[EntitySample, ...]
    role: str | None = None
    confidence: float = 0.0
    source_card_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stable_observation_id", _name(self.stable_observation_id, "stable_observation_id"))
        object.__setattr__(self, "card_id", _name(self.card_id, "entity.card_id").lower())
        object.__setattr__(self, "owner", _side(self.owner, "entity.owner"))
        if not self.samples:
            raise PhysicalLabError("entity observation requires at least one sample")
        if any(not isinstance(sample, EntitySample) for sample in self.samples):
            raise PhysicalLabError("entity samples must be EntitySample records")
        ordered = tuple(sorted(self.samples, key=lambda sample: (sample.video_time_us, sample.frame_index)))
        if ordered != self.samples:
            raise PhysicalLabError("entity samples must be ordered by video time/frame")
        object.__setattr__(self, "confidence", _confidence(self.confidence, "entity.confidence"))
        if self.role is not None:
            object.__setattr__(self, "role", _name(self.role, "entity.role"))
        if self.source_card_id is not None:
            object.__setattr__(self, "source_card_id", _name(self.source_card_id, "entity.source_card_id").lower())

    def selector(self) -> tuple[str, str, str | None, str | None]:
        return self.card_id, self.owner, self.role, self.source_card_id

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "stable_observation_id": self.stable_observation_id,
            "card_id": self.card_id,
            "owner": self.owner,
            "confidence": self.confidence,
            "samples": [sample.to_dict() for sample in self.samples],
        }
        for key in ("role", "source_card_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """A normalized event with direct/inferred provenance and uncertainty."""

    event_id: str
    kind: str
    video_time_us: int
    match_time_us: int | None
    confidence: float
    certainty: ObservationCertainty
    source_frame_indices: tuple[int, ...]
    evidence_refs: tuple[str, ...] = ()
    card_id: str | None = None
    owner: str | None = None
    source_card_id: str | None = None
    target_role: str | None = None
    target_card_id: str | None = None
    values: Mapping[str, bool | int | float | str | None] = field(default_factory=dict)
    uncertainty_us: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _name(self.event_id, "event.event_id"))
        object.__setattr__(self, "kind", _name(self.kind, "event.kind"))
        object.__setattr__(self, "video_time_us", _integer(self.video_time_us, "event.video_time_us"))
        if self.match_time_us is not None:
            object.__setattr__(self, "match_time_us", _integer(self.match_time_us, "event.match_time_us"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, "event.confidence"))
        try:
            certainty = self.certainty if isinstance(self.certainty, ObservationCertainty) else ObservationCertainty(self.certainty)
        except (TypeError, ValueError) as error:
            raise PhysicalLabError(f"unsupported event certainty: {self.certainty!r}") from error
        object.__setattr__(self, "certainty", certainty)
        frames = tuple(_integer(frame, "event.source_frame_indices") for frame in self.source_frame_indices)
        object.__setattr__(self, "source_frame_indices", frames)
        object.__setattr__(self, "evidence_refs", tuple(_name(ref, "event.evidence_ref") for ref in self.evidence_refs))
        if self.card_id is not None:
            object.__setattr__(self, "card_id", _name(self.card_id, "event.card_id").lower())
        if self.owner is not None:
            object.__setattr__(self, "owner", _side(self.owner, "event.owner"))
        for key in ("source_card_id", "target_role", "target_card_id"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, _name(value, f"event.{key}"))
        clean_values: dict[str, bool | int | float | str | None] = {}
        for key, value in dict(self.values).items():
            _name(key, "event.values key")
            if not isinstance(value, (bool, int, float, str)) and value is not None:
                raise PhysicalLabError("event values must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise PhysicalLabError("event values must be finite")
            clean_values[key] = value
        object.__setattr__(self, "values", dict(sorted(clean_values.items())))
        object.__setattr__(self, "uncertainty_us", _integer(self.uncertainty_us, "event.uncertainty_us"))

    def selector(self) -> tuple[str | None, str | None, str | None, str | None]:
        return self.card_id, self.owner, self.source_card_id, self.target_role

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "event_id": self.event_id,
            "kind": self.kind,
            "video_time_us": self.video_time_us,
            "confidence": self.confidence,
            "certainty": self.certainty.value,
            "source_frame_indices": list(self.source_frame_indices),
            "evidence_refs": list(self.evidence_refs),
            "values": dict(self.values),
            "uncertainty_us": self.uncertainty_us,
        }
        if self.match_time_us is not None:
            result["match_time_us"] = self.match_time_us
        for key in ("card_id", "owner", "source_card_id", "target_role", "target_card_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True, slots=True)
class RejectedObservation:
    record_id: str
    record_type: str
    reason: str
    confidence: float | None = None
    source_frame_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _name(self.record_id, "rejected.record_id"))
        object.__setattr__(self, "record_type", _name(self.record_type, "rejected.record_type"))
        object.__setattr__(self, "reason", _name(self.reason, "rejected.reason"))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _confidence(self.confidence, "rejected.confidence"))
        object.__setattr__(
            self,
            "source_frame_indices",
            tuple(_integer(frame, "rejected.source_frame_indices") for frame in self.source_frame_indices),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "reason": self.reason,
            "source_frame_indices": list(self.source_frame_indices),
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass(frozen=True, slots=True)
class ObservationManifest:
    """Sealed physical observation corpus input; never called ground truth."""

    run_id: str
    experiment_hash: str
    capture_group_id: str
    evidence_split: EvidenceSplit | str
    status: EvidenceStatus | str
    entities: tuple[EntityObservation, ...] = ()
    events: tuple[NormalizedEvent, ...] = ()
    rejected: tuple[RejectedObservation, ...] = ()
    capture_ids: tuple[str, ...] = ()
    media_hashes: Mapping[str, str] = field(default_factory=dict)
    synchronization: Mapping[str, Any] = field(default_factory=dict)
    replay_cache_hash: str | None = None
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise PhysicalLabError(f"unsupported observation schema: {self.schema_version}")
        for field_name in ("run_id", "capture_group_id"):
            object.__setattr__(self, field_name, _name(getattr(self, field_name), field_name))
        object.__setattr__(self, "experiment_hash", _name(self.experiment_hash, "experiment_hash"))
        try:
            split = self.evidence_split if isinstance(self.evidence_split, EvidenceSplit) else EvidenceSplit(self.evidence_split)
            status = self.status if isinstance(self.status, EvidenceStatus) else EvidenceStatus(self.status)
        except (TypeError, ValueError) as error:
            raise PhysicalLabError("invalid observation split or status") from error
        object.__setattr__(self, "evidence_split", split)
        object.__setattr__(self, "status", status)
        if any(not isinstance(entity, EntityObservation) for entity in self.entities):
            raise PhysicalLabError("entities must contain EntityObservation records")
        if any(not isinstance(event, NormalizedEvent) for event in self.events):
            raise PhysicalLabError("events must contain NormalizedEvent records")
        if any(not isinstance(item, RejectedObservation) for item in self.rejected):
            raise PhysicalLabError("rejected must contain RejectedObservation records")
        ids = [entity.stable_observation_id for entity in self.entities]
        if len(set(ids)) != len(ids):
            raise PhysicalLabError("stable observation IDs must be unique")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise PhysicalLabError("event IDs must be unique")
        object.__setattr__(self, "capture_ids", tuple(_name(item, "capture_id") for item in self.capture_ids))
        clean_hashes: dict[str, str] = {}
        for key, value in dict(self.media_hashes).items():
            clean_hashes[_name(key, "media_hash key")] = _name(value, "media_hash")
        object.__setattr__(self, "media_hashes", dict(sorted(clean_hashes.items())))
        object.__setattr__(self, "synchronization", json.loads(canonical_json(dict(self.synchronization))))
        if self.replay_cache_hash is not None:
            object.__setattr__(self, "replay_cache_hash", _name(self.replay_cache_hash, "replay_cache_hash"))

    def to_dict(self, *, include_hash: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": "physical_lab_observation_manifest",
            "run_id": self.run_id,
            "experiment_hash": self.experiment_hash,
            "capture_group_id": self.capture_group_id,
            "evidence_split": self.evidence_split.value,
            "status": self.status.value,
            "capture_ids": list(self.capture_ids),
            "media_hashes": dict(self.media_hashes),
            "synchronization": self.synchronization,
            "entities": [entity.to_dict() for entity in self.entities],
            "events": [event.to_dict() for event in self.events],
            "rejected": [item.to_dict() for item in self.rejected],
        }
        if self.replay_cache_hash is not None:
            result["replay_cache_hash"] = self.replay_cache_hash
        if include_hash:
            result["manifest_hash"] = canonical_hash(result)
        return result

    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(include_hash=True), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationManifest":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError("observation manifest must be an object")
        allowed = {
            "schema_version",
            "kind",
            "run_id",
            "experiment_hash",
            "capture_group_id",
            "evidence_split",
            "status",
            "capture_ids",
            "media_hashes",
            "synchronization",
            "entities",
            "events",
            "rejected",
            "replay_cache_hash",
            "manifest_hash",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown observation fields: {unknown}")
        for collection_name in ("entities", "events", "rejected"):
            if not isinstance(raw.get(collection_name, []), list):
                raise PhysicalLabError(f"observation {collection_name} must be an array")
        entities = tuple(_parse_entity(item, index) for index, item in enumerate(raw.get("entities", [])))
        events = tuple(_parse_event(item, index) for index, item in enumerate(raw.get("events", [])))
        rejected = tuple(_parse_rejected(item, index) for index, item in enumerate(raw.get("rejected", [])))
        manifest = cls(
            schema_version=raw.get("schema_version", OBSERVATION_SCHEMA_VERSION),
            run_id=raw.get("run_id"),
            experiment_hash=raw.get("experiment_hash"),
            capture_group_id=raw.get("capture_group_id"),
            evidence_split=raw.get("evidence_split"),
            status=raw.get("status"),
            capture_ids=tuple(raw.get("capture_ids", [])),
            media_hashes=raw.get("media_hashes", {}),
            synchronization=raw.get("synchronization", {}),
            entities=entities,
            events=events,
            rejected=rejected,
            replay_cache_hash=raw.get("replay_cache_hash"),
        )
        declared = raw.get("manifest_hash")
        if declared is not None and declared != manifest.manifest_hash():
            raise PhysicalLabError(
                f"manifest_hash mismatch: declared={declared!r}, actual={manifest.manifest_hash()!r}"
            )
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "ObservationManifest":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PhysicalLabError(f"cannot load observation manifest {source}: {error}") from error
        return cls.from_dict(raw)


def _parse_entity(raw: object, index: int) -> EntityObservation:
    if not isinstance(raw, Mapping):
        raise PhysicalLabError(f"entities[{index}] must be an object")
    samples_raw = raw.get("samples", [])
    if not isinstance(samples_raw, list):
        raise PhysicalLabError(f"entities[{index}].samples must be an array")
    samples = tuple(
        EntitySample(
            frame_index=item.get("frame_index"),
            video_time_us=item.get("video_time_us"),
            match_time_us=item.get("match_time_us"),
            x_mtile=item.get("x_mtile"),
            y_mtile=item.get("y_mtile"),
            hp_observed=item.get("hp_observed"),
            confidence=item.get("confidence"),
            source_capture_id=item.get("source_capture_id"),
            uncertainty_us=item.get("uncertainty_us", 0),
        )
        for item in samples_raw
    )
    return EntityObservation(
        stable_observation_id=raw.get("stable_observation_id"),
        card_id=raw.get("card_id"),
        owner=raw.get("owner"),
        samples=samples,
        role=raw.get("role"),
        confidence=raw.get("confidence"),
        source_card_id=raw.get("source_card_id"),
    )


def _parse_event(raw: object, index: int) -> NormalizedEvent:
    if not isinstance(raw, Mapping):
        raise PhysicalLabError(f"events[{index}] must be an object")
    return NormalizedEvent(
        event_id=raw.get("event_id"),
        kind=raw.get("kind"),
        video_time_us=raw.get("video_time_us"),
        match_time_us=raw.get("match_time_us"),
        confidence=raw.get("confidence"),
        certainty=raw.get("certainty"),
        source_frame_indices=tuple(raw.get("source_frame_indices", [])),
        evidence_refs=tuple(raw.get("evidence_refs", [])),
        card_id=raw.get("card_id"),
        owner=raw.get("owner"),
        source_card_id=raw.get("source_card_id"),
        target_role=raw.get("target_role"),
        target_card_id=raw.get("target_card_id"),
        values=raw.get("values", {}),
        uncertainty_us=raw.get("uncertainty_us", 0),
    )


def _parse_rejected(raw: object, index: int) -> RejectedObservation:
    if not isinstance(raw, Mapping):
        raise PhysicalLabError(f"rejected[{index}] must be an object")
    return RejectedObservation(
        record_id=raw.get("record_id"),
        record_type=raw.get("record_type"),
        reason=raw.get("reason"),
        confidence=raw.get("confidence"),
        source_frame_indices=tuple(raw.get("source_frame_indices", [])),
    )


def ingest_extracted_observations(
    raw: Mapping[str, Any],
    *,
    run_id: str,
    experiment_hash: str,
    capture_group_id: str,
    evidence_split: EvidenceSplit | str,
    synchronization: SynchronizationResult | Mapping[str, Any],
    confidence_threshold: float = 0.98,
    measurement_requires_direct_timing: bool = False,
    capture_ids: tuple[str, ...] = (),
    media_hashes: Mapping[str, str] | None = None,
    replay_cache_hash: str | None = None,
    replay_cache_error: str | None = None,
) -> ObservationManifest:
    """Ingest detector rows without promoting a candidate to truth.

    Rows below the confidence threshold, rows without source provenance, and
    inferred timing rows for a direct-timing measurement are retained in
    ``rejected``.  They are never silently discarded and cannot make a
    held-out gate pass.
    """

    if not isinstance(raw, Mapping):
        raise PhysicalLabError("extracted observations must be an object")
    for collection_name in ("entities", "events"):
        if not isinstance(raw.get(collection_name, []), list):
            raise PhysicalLabError(f"extracted {collection_name} must be an array")
    if type(confidence_threshold) not in (int, float) or not 0 <= confidence_threshold <= 1:
        raise PhysicalLabError("confidence_threshold must be between zero and one")
    if isinstance(synchronization, SynchronizationResult):
        sync_payload = synchronization.to_dict()
        sync_accepted = synchronization.accepted
    else:
        sync_payload = json.loads(canonical_json(dict(synchronization)))
        sync_accepted = bool(sync_payload.get("accepted", False))

    rejected: list[RejectedObservation] = []
    entities: list[EntityObservation] = []
    for index, item in enumerate(raw.get("entities", [])):
        record_id = str(item.get("stable_observation_id", f"entity-{index:04d}")) if isinstance(item, Mapping) else f"entity-{index:04d}"
        try:
            entity = _parse_entity(item, index)
            if entity.confidence < confidence_threshold:
                raise PhysicalLabError(
                    f"confidence {entity.confidence:.3f} below threshold {float(confidence_threshold):.3f}"
                )
            if any(not sample.source_capture_id for sample in entity.samples):
                raise PhysicalLabError("entity sample lacks capture provenance")
            entities.append(entity)
        except (PhysicalLabError, AttributeError, TypeError, KeyError) as error:
            confidence = None
            if isinstance(item, Mapping) and item.get("confidence") is not None:
                try:
                    confidence = _confidence(item["confidence"], "rejected.confidence")
                except PhysicalLabError:
                    confidence = None
            rejected.append(RejectedObservation(record_id, "entity", str(error), confidence))

    events: list[NormalizedEvent] = []
    for index, item in enumerate(raw.get("events", [])):
        record_id = str(item.get("event_id", f"event-{index:04d}")) if isinstance(item, Mapping) else f"event-{index:04d}"
        try:
            event = _parse_event(item, index)
            if event.confidence < confidence_threshold:
                raise PhysicalLabError(
                    f"confidence {event.confidence:.3f} below threshold {float(confidence_threshold):.3f}"
                )
            if not event.evidence_refs:
                raise PhysicalLabError("event has no evidence references")
            if not event.source_frame_indices:
                raise PhysicalLabError("event has no source frame provenance")
            if measurement_requires_direct_timing and event.certainty is ObservationCertainty.INFERRED:
                raise PhysicalLabError("inferred event is not eligible for direct timing measurement")
            events.append(event)
        except (PhysicalLabError, AttributeError, TypeError, KeyError) as error:
            confidence = None
            if isinstance(item, Mapping) and item.get("confidence") is not None:
                try:
                    confidence = _confidence(item["confidence"], "rejected.confidence")
                except PhysicalLabError:
                    confidence = None
            rejected.append(RejectedObservation(record_id, "event", str(error), confidence))

    if not sync_accepted:
        rejected.append(
            RejectedObservation(
                record_id="synchronization",
                record_type="run",
                reason="capture synchronization failed its declared timing gate",
            )
        )
    if replay_cache_error is not None:
        rejected.append(
            RejectedObservation(
                record_id="replay-cache",
                record_type="replay_cache",
                reason=replay_cache_error,
            )
        )
    status = EvidenceStatus.CANDIDATE_ONLY
    try:
        split = evidence_split if isinstance(evidence_split, EvidenceSplit) else EvidenceSplit(evidence_split)
    except (TypeError, ValueError) as error:
        raise PhysicalLabError(f"invalid evidence split: {evidence_split!r}") from error
    if rejected or not entities and not events or not sync_accepted:
        status = EvidenceStatus.REJECTED
    elif split is EvidenceSplit.VALIDATION:
        status = EvidenceStatus.VALIDATION
    elif split is EvidenceSplit.HELDOUT:
        status = EvidenceStatus.HELDOUT
    elif split is EvidenceSplit.REGRESSION:
        status = EvidenceStatus.REGRESSION
    else:
        status = EvidenceStatus.CALIBRATED_ONLY
    return ObservationManifest(
        run_id=run_id,
        experiment_hash=experiment_hash,
        capture_group_id=capture_group_id,
        evidence_split=split,
        status=status,
        entities=tuple(entities),
        events=tuple(events),
        rejected=tuple(rejected),
        capture_ids=tuple(capture_ids),
        media_hashes=dict(media_hashes or {}),
        synchronization=sync_payload,
        replay_cache_hash=replay_cache_hash,
    )


def ingest_for_experiment(
    raw: Mapping[str, Any],
    *,
    spec: ExperimentSpec,
    run_id: str,
    synchronization: SynchronizationResult | Mapping[str, Any],
    confidence_threshold: float = 0.98,
    capture_ids: tuple[str, ...] = (),
    media_hashes: Mapping[str, str] | None = None,
    replay_cache_hash: str | None = None,
    replay_cache_error: str | None = None,
    force_direct_timing: bool = False,
) -> ObservationManifest:
    """Apply the experiment's measurement directness contract automatically."""

    return ingest_extracted_observations(
        raw,
        run_id=run_id,
        experiment_hash=spec.experiment_hash(),
        capture_group_id=spec.capture_group_id,
        evidence_split=spec.evidence_split,
        synchronization=synchronization,
        confidence_threshold=confidence_threshold,
        measurement_requires_direct_timing=force_direct_timing or any(
            measurement.requires_direct_timing for measurement in spec.measurements
        ),
        capture_ids=capture_ids,
        media_hashes=media_hashes,
        replay_cache_hash=replay_cache_hash,
        replay_cache_error=replay_cache_error,
    )


__all__ = [
    "EntityObservation",
    "EntitySample",
    "NormalizedEvent",
    "OBSERVATION_SCHEMA_VERSION",
    "ObservationCertainty",
    "ObservationManifest",
    "RejectedObservation",
    "ingest_extracted_observations",
    "ingest_for_experiment",
]
