"""Versioned, canonical records for the physical-fidelity lab.

The lab deliberately keeps its wire format independent from a phone vendor or
capture transport.  A physical run, an offline fake run, and a later ADB run
therefore share the same experiment hash and action boundary.

Only JSON-compatible primitive values are included in the canonical records.
Binary screenshots and videos are referenced by hashes in the artifact layer;
they are never embedded in an experiment specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


PHYSICAL_EXPERIMENT_SCHEMA_VERSION = 1
PHYSICAL_RUN_SCHEMA_VERSION = 1
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


class PhysicalLabError(ValueError):
    """Raised when a physical-lab record violates its versioned contract."""


class EvidenceSplit(str, Enum):
    """Evidence purpose assigned before capture inspection."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"
    HELDOUT = "heldout"
    REGRESSION = "regression"


class EvidenceStatus(str, Enum):
    """Fail-closed status for a run or an observation set."""

    CALIBRATED_ONLY = "calibrated_only"
    CANDIDATE_ONLY = "candidate_only"
    VALIDATION = "validation"
    HELDOUT = "heldout"
    REGRESSION = "regression"
    REJECTED = "rejected"


class TriggerType(str, Enum):
    MATCH_TIME_US = "match_time_us"
    AFTER_OBSERVATION = "after_observation"


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalLabError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, field_name: str) -> str:
    value = _name(value, field_name)
    if not _ID_RE.fullmatch(value):
        raise PhysicalLabError(f"{field_name} contains unsupported characters: {value!r}")
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise PhysicalLabError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PhysicalLabError(f"{field_name} must be a positive integer")
    return value


def _hash(value: object, field_name: str) -> str:
    value = _name(value, field_name)
    if not _HASH_RE.fullmatch(value):
        raise PhysicalLabError(f"{field_name} must be sha256:<64 lowercase hex characters>")
    return value


def _finite_number(value: object, field_name: str) -> int | float:
    if type(value) not in (int, float):
        raise PhysicalLabError(f"{field_name} must be an integer or float")
    if isinstance(value, float):
        import math

        if not math.isfinite(value):
            raise PhysicalLabError(f"{field_name} must be finite")
    return value


def canonical_json(value: object) -> str:
    """Encode a JSON-compatible value with the lab's canonical settings."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PhysicalLabError(f"value is not canonical JSON: {error}") from error


def canonical_hash(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('ascii')).hexdigest()}"


def _copy_json(value: object, field_name: str) -> Any:
    """Round-trip a mapping to detach it and reject non-JSON values."""

    try:
        copied = json.loads(canonical_json(value))
    except PhysicalLabError as error:
        raise PhysicalLabError(f"{field_name} must be JSON-compatible: {error}") from error
    return copied


def _cell(value: object, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PhysicalLabError(f"{field_name} must contain [column, row]")
    col, row = value
    if type(col) is not int or type(row) is not int:
        raise PhysicalLabError(f"{field_name} coordinates must be integers")
    if not (0 <= col < 18 and 0 <= row < 32):
        raise PhysicalLabError(f"{field_name} is outside the 18x32 action grid")
    return col, row


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Sealed device provenance without exposing a raw serial number."""

    serial_hash: str
    role: str
    device_label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "serial_hash", _hash(self.serial_hash, "serial_hash"))
        object.__setattr__(self, "role", _name(self.role, "role"))
        if self.device_label is not None:
            object.__setattr__(self, "device_label", _name(self.device_label, "device_label"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"serial_hash": self.serial_hash, "role": self.role}
        if self.device_label is not None:
            result["device_label"] = self.device_label
        return result


@dataclass(frozen=True, slots=True)
class Trigger:
    """Logical action trigger; no screen or device time is stored here."""

    type: TriggerType | str
    value: int = 0
    event: str | None = None

    def __post_init__(self) -> None:
        try:
            trigger_type = self.type if isinstance(self.type, TriggerType) else TriggerType(self.type)
        except (TypeError, ValueError) as error:
            raise PhysicalLabError(f"unsupported action trigger: {self.type!r}") from error
        object.__setattr__(self, "type", trigger_type)
        object.__setattr__(self, "value", _nonnegative_int(self.value, "trigger.value"))
        if trigger_type is TriggerType.MATCH_TIME_US:
            if self.event is not None:
                raise PhysicalLabError("match_time_us trigger cannot contain event")
        else:
            if self.event is None:
                raise PhysicalLabError("after_observation trigger requires event")
            object.__setattr__(self, "event", _name(self.event, "trigger.event"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value, "value": self.value}
        if self.event is not None:
            result["event"] = self.event
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Trigger":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError("trigger must be an object")
        allowed = {"type", "value", "event"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown trigger fields: {unknown}")
        return cls(raw.get("type"), raw.get("value", 0), raw.get("event"))


@dataclass(frozen=True, slots=True)
class PhysicalAction:
    """A logical card action shared by physical and simulator runners."""

    action_id: str
    side: str
    card_id: str
    arena_cell: tuple[int, int]
    trigger: Trigger
    card_slot: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _identifier(self.action_id, "action_id"))
        side = _name(self.side, "action.side").upper()
        if side not in {"A", "B"}:
            raise PhysicalLabError("action.side must be A or B")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "card_id", _identifier(self.card_id, "action.card_id").lower())
        object.__setattr__(self, "arena_cell", _cell(self.arena_cell, "action.arena_cell"))
        if not isinstance(self.trigger, Trigger):
            raise PhysicalLabError("action.trigger must be a Trigger")
        if self.card_slot is not None:
            object.__setattr__(self, "card_slot", _nonnegative_int(self.card_slot, "action.card_slot"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_id": self.action_id,
            "side": self.side,
            "card_id": self.card_id,
            "arena_cell": list(self.arena_cell),
            "trigger": self.trigger.to_dict(),
        }
        if self.card_slot is not None:
            result["card_slot"] = self.card_slot
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, index: int = 0) -> "PhysicalAction":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError(f"actions[{index}] must be an object")
        allowed = {"action_id", "side", "card_id", "arena_cell", "trigger", "card_slot"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown fields at actions[{index}]: {unknown}")
        return cls(
            action_id=raw.get("action_id", f"action-{index:04d}"),
            side=raw.get("side"),
            card_id=raw.get("card_id"),
            arena_cell=_cell(raw.get("arena_cell"), f"actions[{index}].arena_cell"),
            trigger=Trigger.from_dict(raw.get("trigger", {})),
            card_slot=raw.get("card_slot"),
        )


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """One declared measurement and its timing acceptance boundary."""

    name: str
    timing_tolerance_us: int = 10_000
    requires_direct_timing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "measurement.name"))
        object.__setattr__(
            self,
            "timing_tolerance_us",
            _nonnegative_int(self.timing_tolerance_us, "measurement.timing_tolerance_us"),
        )
        if type(self.requires_direct_timing) is not bool:
            raise PhysicalLabError("measurement.requires_direct_timing must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timing_tolerance_us": self.timing_tolerance_us,
            "requires_direct_timing": self.requires_direct_timing,
        }

    @classmethod
    def from_value(cls, value: object, *, index: int = 0) -> "MeasurementSpec":
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise PhysicalLabError(f"measurements[{index}] must be a name or object")
        allowed = {"name", "timing_tolerance_us", "requires_direct_timing"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown fields at measurements[{index}]: {unknown}")
        return cls(
            name=value.get("name"),
            timing_tolerance_us=value.get("timing_tolerance_us", 10_000),
            requires_direct_timing=value.get("requires_direct_timing", False),
        )


@dataclass(frozen=True, slots=True)
class InitialConditions:
    """Logical starting assumptions; device-specific coordinates stay elsewhere."""

    tower_state: str = "default"
    requested_elixir_milli: Mapping[str, int] = field(
        default_factory=lambda: {"A": 10_000, "B": 10_000}
    )
    decks: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    hand_slots: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tower_state", _name(self.tower_state, "initial_conditions.tower_state"))
        elixir: dict[str, int] = {}
        for side, value in dict(self.requested_elixir_milli).items():
            side = _name(side, "requested_elixir_milli side").upper()
            if side not in {"A", "B"}:
                raise PhysicalLabError("requested_elixir_milli keys must be A or B")
            elixir[side] = _nonnegative_int(value, f"requested_elixir_milli.{side}")
        for side in ("A", "B"):
            elixir.setdefault(side, 10_000)
        object.__setattr__(self, "requested_elixir_milli", dict(sorted(elixir.items())))

        decks: dict[str, tuple[str, ...]] = {}
        for raw_side, raw_deck in dict(self.decks).items():
            side = _name(raw_side, "initial_conditions.decks side").upper()
            if side not in {"A", "B"}:
                raise PhysicalLabError("decks keys must be A or B")
            if not isinstance(raw_deck, (list, tuple)) or len(raw_deck) != 8:
                raise PhysicalLabError(f"initial_conditions.decks.{side} must contain eight cards")
            deck = tuple(_identifier(card, f"decks.{side} card").lower() for card in raw_deck)
            if len(set(deck)) != 8:
                raise PhysicalLabError(f"initial_conditions.decks.{side} must contain unique cards")
            decks[side] = deck
        object.__setattr__(self, "decks", dict(sorted(decks.items())))

        slots: dict[str, dict[str, int]] = {}
        for raw_side, raw_slots in dict(self.hand_slots).items():
            side = _name(raw_side, "initial_conditions.hand_slots side").upper()
            if side not in {"A", "B"} or not isinstance(raw_slots, Mapping):
                raise PhysicalLabError("hand_slots must map A/B to card-slot objects")
            parsed = {
                _identifier(card, f"hand_slots.{side} card").lower(): _nonnegative_int(
                    slot, f"hand_slots.{side}.{card}"
                )
                for card, slot in raw_slots.items()
            }
            if any(slot >= 8 for slot in parsed.values()):
                raise PhysicalLabError("hand slot must be between 0 and 7")
            if len(set(parsed.values())) != len(parsed):
                raise PhysicalLabError(f"hand_slots.{side} contains duplicate slots")
            slots[side] = dict(sorted(parsed.items()))
        object.__setattr__(self, "hand_slots", dict(sorted(slots.items())))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tower_state": self.tower_state,
            "requested_elixir_milli": dict(self.requested_elixir_milli),
        }
        if self.decks:
            result["decks"] = {side: list(deck) for side, deck in self.decks.items()}
        if self.hand_slots:
            result["hand_slots"] = {
                side: dict(card_slots) for side, card_slots in self.hand_slots.items()
            }
        return result


def _offline_serial_hash(label: str) -> str:
    return canonical_hash({"offline_device": label})


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Canonical input shared by the physical and simulator runners."""

    experiment_id: str
    ruleset_id: str
    ruleset_hash: str
    engine_version: str
    capture_group_id: str
    evidence_split: EvidenceSplit | str
    devices: Mapping[str, DeviceSpec]
    actions: tuple[PhysicalAction, ...]
    measurements: tuple[MeasurementSpec, ...]
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    seed: int = 0
    duration_us: int = 30_000_000
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PHYSICAL_EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_EXPERIMENT_SCHEMA_VERSION:
            raise PhysicalLabError(f"unsupported physical experiment schema: {self.schema_version}")
        object.__setattr__(self, "experiment_id", _identifier(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "ruleset_id", _identifier(self.ruleset_id, "ruleset_id").lower())
        object.__setattr__(self, "ruleset_hash", _hash(self.ruleset_hash, "ruleset_hash"))
        object.__setattr__(self, "engine_version", _name(self.engine_version, "engine_version"))
        object.__setattr__(self, "capture_group_id", _identifier(self.capture_group_id, "capture_group_id"))
        try:
            split = self.evidence_split if isinstance(self.evidence_split, EvidenceSplit) else EvidenceSplit(self.evidence_split)
        except (TypeError, ValueError) as error:
            raise PhysicalLabError(f"unsupported evidence split: {self.evidence_split!r}") from error
        object.__setattr__(self, "evidence_split", split)
        if type(self.seed) is not int:
            raise PhysicalLabError("seed must be an integer")
        object.__setattr__(self, "duration_us", _positive_int(self.duration_us, "duration_us"))
        if not isinstance(self.initial_conditions, InitialConditions):
            raise PhysicalLabError("initial_conditions must be InitialConditions")

        parsed_devices: dict[str, DeviceSpec] = {}
        for raw_side, device in dict(self.devices).items():
            side = _name(raw_side, "devices side").upper()
            if side not in {"A", "B"}:
                raise PhysicalLabError("devices must use A and B keys")
            if not isinstance(device, DeviceSpec):
                raise PhysicalLabError(f"devices.{side} must be DeviceSpec")
            parsed_devices[side] = device
        if set(parsed_devices) != {"A", "B"}:
            raise PhysicalLabError("an experiment requires both device A and device B")
        object.__setattr__(self, "devices", dict(sorted(parsed_devices.items())))

        parsed_actions = tuple(self.actions)
        if any(not isinstance(action, PhysicalAction) for action in parsed_actions):
            raise PhysicalLabError("actions must contain PhysicalAction records")
        action_ids = [action.action_id for action in parsed_actions]
        if len(set(action_ids)) != len(action_ids):
            raise PhysicalLabError("action IDs must be unique")
        object.__setattr__(self, "actions", parsed_actions)

        parsed_measurements = tuple(self.measurements)
        if not parsed_measurements:
            raise PhysicalLabError("an experiment requires at least one measurement")
        if any(not isinstance(item, MeasurementSpec) for item in parsed_measurements):
            raise PhysicalLabError("measurements must contain MeasurementSpec records")
        measurement_names = [item.name for item in parsed_measurements]
        if len(set(measurement_names)) != len(measurement_names):
            raise PhysicalLabError("measurement names must be unique")
        object.__setattr__(self, "measurements", parsed_measurements)
        object.__setattr__(self, "metadata", _copy_json(dict(self.metadata), "metadata"))
        object.__setattr__(self, "provenance", _copy_json(dict(self.provenance), "provenance"))

    @classmethod
    def offline_default_devices(cls) -> dict[str, DeviceSpec]:
        return {
            "A": DeviceSpec(_offline_serial_hash("A"), "player", "offline-A"),
            "B": DeviceSpec(_offline_serial_hash("B"), "opponent", "offline-B"),
        }

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "ruleset_id": self.ruleset_id,
            "ruleset_hash": self.ruleset_hash,
            "engine_version": self.engine_version,
            "capture_group_id": self.capture_group_id,
            "evidence_split": self.evidence_split.value,
            "devices": {side: device.to_dict() for side, device in self.devices.items()},
            "initial_conditions": self.initial_conditions.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "seed": self.seed,
            "duration_us": self.duration_us,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }
        if include_hash:
            result["experiment_hash"] = self.experiment_hash()
        return result

    def canonical_payload(self) -> dict[str, Any]:
        return self.to_dict(include_hash=False)

    def experiment_hash(self) -> str:
        return canonical_hash(self.canonical_payload())

    def dumps(self, *, include_hash: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_hash=include_hash),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"

    def save(self, path: str | Path, *, include_hash: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.dumps(include_hash=include_hash), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentSpec":
        if not isinstance(raw, Mapping):
            raise PhysicalLabError("experiment document must be an object")
        allowed = {
            "schema_version",
            "experiment_id",
            "ruleset_id",
            "ruleset_hash",
            "engine_version",
            "capture_group_id",
            "evidence_split",
            "devices",
            "initial_conditions",
            "actions",
            "measurements",
            "seed",
            "duration_us",
            "provenance",
            "metadata",
            "experiment_hash",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PhysicalLabError(f"unknown experiment fields: {unknown}")
        devices_raw = raw.get("devices")
        if not isinstance(devices_raw, Mapping):
            raise PhysicalLabError("experiment.devices must be an object")
        devices: dict[str, DeviceSpec] = {}
        for side, value in devices_raw.items():
            if not isinstance(value, Mapping):
                raise PhysicalLabError(f"devices.{side} must be an object")
            device_allowed = {"serial_hash", "role", "device_label"}
            device_unknown = sorted(set(value) - device_allowed)
            if device_unknown:
                raise PhysicalLabError(f"unknown fields at devices.{side}: {device_unknown}")
            devices[str(side).upper()] = DeviceSpec(
                serial_hash=value.get("serial_hash"),
                role=value.get("role"),
                device_label=value.get("device_label"),
            )

        initial_raw = raw.get("initial_conditions", {})
        if not isinstance(initial_raw, Mapping):
            raise PhysicalLabError("initial_conditions must be an object")
        initial_allowed = {"tower_state", "requested_elixir_milli", "decks", "hand_slots"}
        initial_unknown = sorted(set(initial_raw) - initial_allowed)
        if initial_unknown:
            raise PhysicalLabError(f"unknown initial_conditions fields: {initial_unknown}")
        initial = InitialConditions(
            tower_state=initial_raw.get("tower_state", "default"),
            requested_elixir_milli=initial_raw.get("requested_elixir_milli", {"A": 10_000, "B": 10_000}),
            decks=initial_raw.get("decks", {}),
            hand_slots=initial_raw.get("hand_slots", {}),
        )
        actions_raw = raw.get("actions", [])
        if not isinstance(actions_raw, list):
            raise PhysicalLabError("actions must be an array")
        actions = tuple(
            PhysicalAction.from_dict(value, index=index)
            for index, value in enumerate(actions_raw)
        )
        measurements_raw = raw.get("measurements", [])
        if not isinstance(measurements_raw, list):
            raise PhysicalLabError("measurements must be an array")
        measurements = tuple(
            MeasurementSpec.from_value(value, index=index)
            for index, value in enumerate(measurements_raw)
        )
        spec = cls(
            schema_version=raw.get("schema_version", PHYSICAL_EXPERIMENT_SCHEMA_VERSION),
            experiment_id=raw.get("experiment_id"),
            ruleset_id=raw.get("ruleset_id"),
            ruleset_hash=raw.get("ruleset_hash"),
            engine_version=raw.get("engine_version"),
            capture_group_id=raw.get("capture_group_id"),
            evidence_split=raw.get("evidence_split"),
            devices=devices,
            initial_conditions=initial,
            actions=actions,
            measurements=measurements,
            seed=raw.get("seed", 0),
            duration_us=raw.get("duration_us", 30_000_000),
            provenance=raw.get("provenance", {}),
            metadata=raw.get("metadata", {}),
        )
        declared_hash = raw.get("experiment_hash")
        if declared_hash is not None and declared_hash != spec.experiment_hash():
            raise PhysicalLabError(
                f"experiment_hash mismatch: declared={declared_hash!r}, actual={spec.experiment_hash()!r}"
            )
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PhysicalLabError(f"cannot load experiment {source}: {error}") from error
        return cls.from_dict(raw)


__all__ = [
    "DeviceSpec",
    "EvidenceSplit",
    "EvidenceStatus",
    "ExperimentSpec",
    "InitialConditions",
    "MeasurementSpec",
    "PHYSICAL_EXPERIMENT_SCHEMA_VERSION",
    "PHYSICAL_RUN_SCHEMA_VERSION",
    "PhysicalAction",
    "PhysicalLabError",
    "Trigger",
    "TriggerType",
    "canonical_hash",
    "canonical_json",
]
