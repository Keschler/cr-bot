"""Bounded packed-batch transport for authoritative simulator states.

This module is an opt-in ABI prototype.  It deliberately does not change
``SimulatorEnv`` or the reference engine.  A caller can use it as the state
transport for a future vector worker or a shared-memory lane store:

* each lane has one fixed byte capacity (``PackLimits.max_state_bytes``);
* a batch is a fixed-stride byte buffer plus a length for every lane;
* lane values use a deterministic tagged binary encoding rather than pickle;
* strings and integers are bounded, and all relevant variable-length state
  collections have explicit capacities;
* the encoded value is the complete
  ``BattleState.to_primitive(include_events=True)`` representation.

The wire format is intentionally small and self-contained.  Its 16-byte
header is ``>4sHHII``: magic, ABI version, reserved flags, lane count, and
lane capacity.  It is followed by one big-endian ``uint32`` length per lane
and then ``lane_count * lane_capacity`` bytes.  Only the prefix described by a
lane length is decoded; the rest must be zero padding.  The binary value
codec preserves lists and tuples separately, so the normal state
reconstructor can restore canonical tuple fields exactly.

This is a bounded transport, not a complete SoA/JIT backend.  The default
limits are deliberately conservative prototype limits: 32 lanes, 2 MiB per
lane, 1024 entities/projectiles, 512 effects, 32,768 events, and signed
128-bit scalar integers.  A workload that needs more capacity must construct a
different :class:`PackLimits` explicitly; silently truncating a state is never
allowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Final

from .state import BattleState, battle_state_from_primitive


ABI_MAGIC: Final[bytes] = b"CRPB"
ABI_VERSION: Final[int] = 1
_HEADER = struct.Struct(">4sHHII")
_U32 = struct.Struct(">I")

_TAG_NONE = 0
_TAG_FALSE = 1
_TAG_TRUE = 2
_TAG_INT = 3
_TAG_STRING = 4
_TAG_LIST = 5
_TAG_TUPLE = 6
_TAG_DICT = 7


class PackedBatchError(ValueError):
    """Base error for invalid or unrepresentable packed state data."""


class CapacityError(PackedBatchError):
    """Raised when a state or batch exceeds a declared ABI capacity."""


class WireFormatError(PackedBatchError):
    """Raised when a packed batch is malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class PackLimits:
    """Fixed capacities for the prototype ABI.

    Capacities apply before encoding, so a malformed payload cannot use the
    byte budget to hide an unbounded collection.  The simulator currently
    uses two players; ``max_players`` is retained as an explicit ABI field so
    a future format can negotiate a different player count.
    """

    max_batch_size: int = 32
    max_state_bytes: int = 2 * 1024 * 1024
    max_players: int = 2
    max_cards_per_player: int = 64
    max_entities: int = 1024
    max_projectiles: int = 1024
    max_effects: int = 512
    max_statuses_per_entity: int = 64
    max_navigation_waypoints: int = 256
    max_projectile_hits: int = 1024
    max_effect_schedule: int = 512
    max_events: int = 32_768
    max_event_fields: int = 128
    max_container_items: int = 65_536
    max_string_bytes: int = 4_096
    max_integer_bits: int = 128
    max_depth: int = 64

    def __post_init__(self) -> None:
        for name in (
            "max_batch_size",
            "max_state_bytes",
            "max_players",
            "max_cards_per_player",
            "max_entities",
            "max_projectiles",
            "max_effects",
            "max_statuses_per_entity",
            "max_navigation_waypoints",
            "max_projectile_hits",
            "max_effect_schedule",
            "max_events",
            "max_event_fields",
            "max_container_items",
            "max_string_bytes",
            "max_integer_bits",
            "max_depth",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_LIMITS: Final[PackLimits] = PackLimits()


def _require_limits(limits: PackLimits | None) -> PackLimits:
    if limits is None:
        return DEFAULT_LIMITS
    if not isinstance(limits, PackLimits):
        raise TypeError("limits must be a PackLimits instance")
    return limits


def _sequence(value: object, field_name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise PackedBatchError(f"{field_name} must be a list or tuple")
    return value


def _bounded_sequence(
    value: object,
    field_name: str,
    limit: int,
    limits: PackLimits,
) -> Sequence[Any]:
    result = _sequence(value, field_name)
    if len(result) > limit:
        raise CapacityError(
            f"{field_name} has {len(result)} items; capacity is {limit}"
        )
    if len(result) > limits.max_container_items:
        raise CapacityError(
            f"{field_name} has {len(result)} items; generic capacity is "
            f"{limits.max_container_items}"
        )
    return result


def _bounded_mapping(value: object, field_name: str, limits: PackLimits) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackedBatchError(f"{field_name} must be an object")
    if len(value) > limits.max_container_items:
        raise CapacityError(
            f"{field_name} has {len(value)} fields; capacity is "
            f"{limits.max_container_items}"
        )
    return value  # type: ignore[return-value]


def _validate_state_shape(raw: Mapping[str, Any], limits: PackLimits) -> None:
    """Check state-specific collection capacities before binary encoding."""

    players = _bounded_sequence(
        raw.get("players"), "players", limits.max_players, limits
    )
    entities = _bounded_sequence(
        raw.get("entities"), "entities", limits.max_entities, limits
    )
    projectiles = _bounded_sequence(
        raw.get("projectiles"), "projectiles", limits.max_projectiles, limits
    )
    effects = _bounded_sequence(
        raw.get("effects"), "effects", limits.max_effects, limits
    )
    events = _bounded_sequence(
        raw.get("events"), "events", limits.max_events, limits
    )

    for index, player in enumerate(players):
        player_map = _bounded_mapping(player, f"players[{index}]", limits)
        for field_name in ("deck", "hand", "draw_pile", "seen_enemy_cards"):
            _bounded_sequence(
                player_map.get(field_name),
                f"players[{index}].{field_name}",
                limits.max_cards_per_player,
                limits,
            )

    for index, entity in enumerate(entities):
        entity_map = _bounded_mapping(entity, f"entities[{index}]", limits)
        _bounded_sequence(
            entity_map.get("statuses"),
            f"entities[{index}].statuses",
            limits.max_statuses_per_entity,
            limits,
        )
        _bounded_sequence(
            entity_map.get("navigation_waypoints"),
            f"entities[{index}].navigation_waypoints",
            limits.max_navigation_waypoints,
            limits,
        )

    for index, projectile in enumerate(projectiles):
        projectile_map = _bounded_mapping(projectile, f"projectiles[{index}]", limits)
        _bounded_sequence(
            projectile_map.get("hit_uids"),
            f"projectiles[{index}].hit_uids",
            limits.max_projectile_hits,
            limits,
        )

    for index, effect in enumerate(effects):
        effect_map = _bounded_mapping(effect, f"effects[{index}]", limits)
        for field_name in ("damage_schedule", "crown_damage_schedule"):
            _bounded_sequence(
                effect_map.get(field_name),
                f"effects[{index}].{field_name}",
                limits.max_effect_schedule,
                limits,
            )

    for index, event in enumerate(events):
        event_map = _bounded_mapping(event, f"events[{index}]", limits)
        data = _bounded_mapping(event_map.get("data"), f"events[{index}].data", limits)
        if len(data) > limits.max_event_fields:
            raise CapacityError(
                f"events[{index}].data has {len(data)} fields; capacity is "
                f"{limits.max_event_fields}"
            )


def _write_varuint(value: int, output: bytearray) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("varuint value must be a non-negative integer")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def _read_varuint(reader: "_Reader") -> int:
    result = 0
    shift = 0
    count = 0
    while True:
        byte = reader.read_byte()
        count += 1
        if count > reader.max_varint_bytes:
            raise WireFormatError("variable-length integer exceeds ABI limit")
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if count > 1 and result < (1 << (7 * (count - 1))):
                raise WireFormatError("non-canonical variable-length integer")
            return result
        shift += 7


def _zigzag_encode(value: int) -> int:
    return value * 2 if value >= 0 else (-value * 2) - 1


def _zigzag_decode(value: int) -> int:
    return value // 2 if value % 2 == 0 else -((value // 2) + 1)


def _integer_bounds(limits: PackLimits) -> tuple[int, int]:
    minimum = -(1 << (limits.max_integer_bits - 1))
    maximum = (1 << (limits.max_integer_bits - 1)) - 1
    return minimum, maximum


def _encode_value(
    value: Any,
    output: bytearray,
    limits: PackLimits,
    *,
    depth: int = 0,
) -> None:
    if depth > limits.max_depth:
        raise CapacityError(f"value nesting exceeds capacity {limits.max_depth}")
    if value is None:
        output.append(_TAG_NONE)
        return
    if type(value) is bool:
        output.append(_TAG_TRUE if value else _TAG_FALSE)
        return
    if type(value) is int:
        minimum, maximum = _integer_bounds(limits)
        if not minimum <= value <= maximum:
            raise CapacityError(
                f"integer {value} is outside signed {limits.max_integer_bits}-bit ABI range"
            )
        output.append(_TAG_INT)
        _write_varuint(_zigzag_encode(value), output)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) > limits.max_string_bytes:
            raise CapacityError(
                f"string has {len(encoded)} bytes; capacity is {limits.max_string_bytes}"
            )
        output.append(_TAG_STRING)
        _write_varuint(len(encoded), output)
        output.extend(encoded)
        return
    if isinstance(value, list) or isinstance(value, tuple):
        if len(value) > limits.max_container_items:
            raise CapacityError(
                f"sequence has {len(value)} items; capacity is "
                f"{limits.max_container_items}"
            )
        output.append(_TAG_LIST if isinstance(value, list) else _TAG_TUPLE)
        _write_varuint(len(value), output)
        for item in value:
            _encode_value(item, output, limits, depth=depth + 1)
        return
    if isinstance(value, Mapping):
        if len(value) > limits.max_container_items:
            raise CapacityError(
                f"mapping has {len(value)} fields; capacity is "
                f"{limits.max_container_items}"
            )
        keys = sorted(value)
        if any(type(key) is not str for key in keys):
            raise PackedBatchError("ABI mappings require string keys")
        output.append(_TAG_DICT)
        _write_varuint(len(keys), output)
        for key in keys:
            _encode_value(key, output, limits, depth=depth + 1)
            _encode_value(value[key], output, limits, depth=depth + 1)
        return
    raise PackedBatchError(f"unsupported value type in packed state: {type(value).__name__}")


class _Reader:
    __slots__ = ("data", "position", "max_varint_bytes")

    def __init__(self, data: bytes, limits: PackLimits) -> None:
        self.data = memoryview(data)
        self.position = 0
        self.max_varint_bytes = max(1, (limits.max_integer_bits + 6) // 7)

    @property
    def remaining(self) -> int:
        return len(self.data) - self.position

    def read_byte(self) -> int:
        if self.position >= len(self.data):
            raise WireFormatError("unexpected end of packed lane")
        value = self.data[self.position]
        self.position += 1
        return int(value)

    def read_bytes(self, count: int) -> bytes:
        if type(count) is not int or count < 0 or count > self.remaining:
            raise WireFormatError("packed lane length exceeds available bytes")
        start = self.position
        self.position += count
        return self.data[start:self.position].tobytes()


def _decode_value(reader: _Reader, limits: PackLimits, *, depth: int = 0) -> Any:
    if depth > limits.max_depth:
        raise WireFormatError(f"value nesting exceeds capacity {limits.max_depth}")
    tag = reader.read_byte()
    if tag == _TAG_NONE:
        return None
    if tag == _TAG_FALSE:
        return False
    if tag == _TAG_TRUE:
        return True
    if tag == _TAG_INT:
        encoded = _read_varuint(reader)
        value = _zigzag_decode(encoded)
        minimum, maximum = _integer_bounds(limits)
        if not minimum <= value <= maximum:
            raise WireFormatError("packed integer exceeds ABI range")
        return value
    if tag == _TAG_STRING:
        length = _read_varuint(reader)
        if length > limits.max_string_bytes:
            raise WireFormatError("packed string exceeds ABI capacity")
        try:
            return reader.read_bytes(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise WireFormatError("packed string is not valid UTF-8") from error
    if tag in (_TAG_LIST, _TAG_TUPLE):
        count = _read_varuint(reader)
        if count > limits.max_container_items:
            raise WireFormatError("packed sequence exceeds ABI capacity")
        values = [
            _decode_value(reader, limits, depth=depth + 1)
            for _ in range(count)
        ]
        return values if tag == _TAG_LIST else tuple(values)
    if tag == _TAG_DICT:
        count = _read_varuint(reader)
        if count > limits.max_container_items:
            raise WireFormatError("packed mapping exceeds ABI capacity")
        result: dict[str, Any] = {}
        previous_key: str | None = None
        for _ in range(count):
            key = _decode_value(reader, limits, depth=depth + 1)
            if type(key) is not str:
                raise WireFormatError("packed mapping key is not a string")
            if previous_key is not None and key <= previous_key:
                raise WireFormatError("packed mapping keys are not canonical")
            previous_key = key
            if key in result:
                raise WireFormatError("packed mapping contains a duplicate key")
            result[key] = _decode_value(reader, limits, depth=depth + 1)
        return result
    raise WireFormatError(f"unknown packed value tag: {tag}")


def _encode_state(state: BattleState, limits: PackLimits) -> bytes:
    raw = state.to_primitive(include_events=True)
    raw_map = _bounded_mapping(raw, "state", limits)
    _validate_state_shape(raw_map, limits)
    output = bytearray()
    _encode_value(raw_map, output, limits)
    if len(output) > limits.max_state_bytes:
        raise CapacityError(
            f"encoded state has {len(output)} bytes; capacity is "
            f"{limits.max_state_bytes}"
        )
    return bytes(output)


def _decode_state(lane: bytes, limits: PackLimits) -> BattleState:
    reader = _Reader(lane, limits)
    raw = _decode_value(reader, limits)
    if reader.remaining:
        raise WireFormatError("packed lane contains trailing bytes")
    if not isinstance(raw, dict):
        raise WireFormatError("packed state root is not an object")
    try:
        state = battle_state_from_primitive(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise WireFormatError(f"packed state does not satisfy BattleState ABI: {error}") from error
    # Re-encoding is a canonicality check as well as a protection against
    # silently changing tuple/list or default-field semantics in a future
    # state reconstructor.
    canonical = _encode_state(state, limits)
    if canonical != lane:
        raise WireFormatError("packed state is not canonical after reconstruction")
    return state


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """A fixed-stride batch of packed BattleState lanes."""

    buffer: bytes
    lengths: tuple[int, ...]
    slot_bytes: int
    abi_version: int = ABI_VERSION

    def __post_init__(self) -> None:
        if type(self.abi_version) is not int or self.abi_version != ABI_VERSION:
            raise WireFormatError(f"unsupported packed ABI version: {self.abi_version!r}")
        if type(self.slot_bytes) is not int or self.slot_bytes <= 0:
            raise WireFormatError("slot_bytes must be a positive integer")
        if not self.lengths:
            raise WireFormatError("a packed batch must contain at least one lane")
        if any(type(length) is not int or length <= 0 or length > self.slot_bytes for length in self.lengths):
            raise WireFormatError("lane length is outside its fixed slot")
        try:
            normalized_buffer = bytes(self.buffer)
        except (TypeError, ValueError) as error:
            raise WireFormatError("packed buffer must be bytes-like") from error
        object.__setattr__(self, "buffer", normalized_buffer)
        expected = len(self.lengths) * self.slot_bytes
        if len(normalized_buffer) != expected:
            raise WireFormatError(
                f"packed buffer has {len(normalized_buffer)} bytes; expected {expected}"
            )
        for index, length in enumerate(self.lengths):
            start = index * self.slot_bytes
            if any(normalized_buffer[start + length : start + self.slot_bytes]):
                raise WireFormatError(f"lane {index} has non-zero padding")

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    def lane_bytes(self, index: int) -> bytes:
        if type(index) is not int or not 0 <= index < self.batch_size:
            raise IndexError(f"lane index out of range: {index}")
        start = index * self.slot_bytes
        return self.buffer[start : start + self.lengths[index]]

    def to_bytes(self) -> bytes:
        if self.batch_size > 0xFFFFFFFF or self.slot_bytes > 0xFFFFFFFF:
            raise CapacityError("packed batch exceeds wire uint32 dimensions")
        header = _HEADER.pack(
            ABI_MAGIC,
            self.abi_version,
            0,
            self.batch_size,
            self.slot_bytes,
        )
        lengths = b"".join(_U32.pack(length) for length in self.lengths)
        return header + lengths + self.buffer

    @classmethod
    def from_bytes(
        cls,
        payload: bytes | bytearray | memoryview,
        *,
        limits: PackLimits | None = None,
    ) -> "PackedBatch":
        configured = _require_limits(limits)
        try:
            encoded = bytes(payload)
        except (TypeError, ValueError) as error:
            raise WireFormatError("packed batch payload must be bytes-like") from error
        if len(encoded) < _HEADER.size:
            raise WireFormatError("packed batch is shorter than its header")
        magic, version, flags, batch_size, slot_bytes = _HEADER.unpack_from(encoded)
        if magic != ABI_MAGIC:
            raise WireFormatError("packed batch magic does not match")
        if version != ABI_VERSION:
            raise WireFormatError(f"unsupported packed ABI version: {version}")
        if flags != 0:
            raise WireFormatError("packed batch has unsupported flags")
        if batch_size <= 0 or batch_size > configured.max_batch_size:
            raise CapacityError(
                f"batch has {batch_size} lanes; capacity is {configured.max_batch_size}"
            )
        if slot_bytes <= 0 or slot_bytes > configured.max_state_bytes:
            raise CapacityError(
                f"lane capacity is {slot_bytes}; configured maximum is "
                f"{configured.max_state_bytes}"
            )
        lengths_start = _HEADER.size
        lengths_end = lengths_start + batch_size * _U32.size
        expected = lengths_end + batch_size * slot_bytes
        if len(encoded) != expected:
            raise WireFormatError(
                f"packed batch has {len(encoded)} bytes; expected {expected}"
            )
        lengths = tuple(
            _U32.unpack_from(encoded, lengths_start + index * _U32.size)[0]
            for index in range(batch_size)
        )
        return cls(
            buffer=encoded[lengths_end:],
            lengths=lengths,
            slot_bytes=slot_bytes,
            abi_version=version,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_bytes(self.to_bytes())
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        limits: PackLimits | None = None,
    ) -> "PackedBatch":
        return cls.from_bytes(Path(path).read_bytes(), limits=limits)

    def unpack(self, *, limits: PackLimits | None = None) -> tuple[BattleState, ...]:
        return unpack_batch(self, limits=limits)


def pack_batch(
    states: Sequence[BattleState],
    *,
    limits: PackLimits | None = None,
) -> PackedBatch:
    """Pack a non-empty batch into fixed-capacity, fixed-stride lanes."""

    configured = _require_limits(limits)
    if not isinstance(states, Sequence):
        raise TypeError("states must be a sequence of BattleState objects")
    if not states:
        raise CapacityError("a packed batch must contain at least one state")
    if len(states) > configured.max_batch_size:
        raise CapacityError(
            f"batch has {len(states)} states; capacity is {configured.max_batch_size}"
        )
    lanes: list[bytes] = []
    for index, state in enumerate(states):
        if not isinstance(state, BattleState):
            raise TypeError(f"states[{index}] is not a BattleState")
        lanes.append(_encode_state(state, configured))
    buffer = bytearray(len(lanes) * configured.max_state_bytes)
    for index, lane in enumerate(lanes):
        start = index * configured.max_state_bytes
        buffer[start : start + len(lane)] = lane
    return PackedBatch(
        buffer=bytes(buffer),
        lengths=tuple(len(lane) for lane in lanes),
        slot_bytes=configured.max_state_bytes,
    )


def pack_state(state: BattleState, *, limits: PackLimits | None = None) -> PackedBatch:
    """Pack one state as a one-lane :class:`PackedBatch`."""

    return pack_batch((state,), limits=limits)


def _coerce_batch(
    packed: PackedBatch | bytes | bytearray | memoryview,
    limits: PackLimits,
) -> PackedBatch:
    if isinstance(packed, PackedBatch):
        if packed.batch_size > limits.max_batch_size:
            raise CapacityError(
                f"batch has {packed.batch_size} lanes; capacity is {limits.max_batch_size}"
            )
        if packed.slot_bytes > limits.max_state_bytes:
            raise CapacityError(
                f"lane capacity is {packed.slot_bytes}; configured maximum is "
                f"{limits.max_state_bytes}"
            )
        return packed
    return PackedBatch.from_bytes(packed, limits=limits)


def unpack_batch(
    packed: PackedBatch | bytes | bytearray | memoryview,
    *,
    limits: PackLimits | None = None,
) -> tuple[BattleState, ...]:
    """Decode and reconstruct every lane, rejecting non-canonical data."""

    configured = _require_limits(limits)
    batch = _coerce_batch(packed, configured)
    return tuple(_decode_state(batch.lane_bytes(index), configured) for index in range(batch.batch_size))


def unpack_state(
    packed: PackedBatch | bytes | bytearray | memoryview,
    *,
    limits: PackLimits | None = None,
) -> BattleState:
    """Decode exactly one state from a one-lane packed batch."""

    states = unpack_batch(packed, limits=limits)
    if len(states) != 1:
        raise ValueError(f"unpack_state requires one lane, got {len(states)}")
    return states[0]


__all__ = [
    "ABI_MAGIC",
    "ABI_VERSION",
    "CapacityError",
    "DEFAULT_LIMITS",
    "PackLimits",
    "PackedBatch",
    "PackedBatchError",
    "WireFormatError",
    "pack_batch",
    "pack_state",
    "unpack_batch",
    "unpack_state",
]
