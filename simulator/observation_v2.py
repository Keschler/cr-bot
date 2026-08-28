"""Immutable public hybrid observations for future policy models.

This module is additive to :mod:`simulator.observation`.  It does not project
an authoritative ``BattleState`` and deliberately has no dependency on that
type.  Callers first obtain the existing public ``PolicyObservationV1`` and
may then attach public entity rows produced by the same observation boundary
(for example, by a visual tracker or a public-state adapter).

The bounded entity-token contract is:

* ``NMAX = 128`` rows per observation;
* ``D = 32`` float features per row;
* rows beyond the supplied public entities are zero padded and marked false
  in ``entity_mask``.

The 32 slots are normalized public features, in this stable order:

``card_id, side, x, y, hp_fraction, is_air, is_building, is_tower,
is_spell, is_visible, is_targetable, lane, distance_to_own_tower,
distance_to_enemy_tower, distance_to_own_king, distance_to_enemy_king,
velocity_x, velocity_y, state_idle, state_moving, state_attacking,
state_stunned, state_slowed, state_frozen, state_invisible, has_target,
recent_damage, recent_deploy, confidence, age, reserved_0, reserved_1``.

The module validates shape, exact dtype, and finiteness, but does not infer or
clamp feature semantics.  Upstream producers are responsible for using the
documented normalized representation and for excluding private opponent
fields.  Array inputs are copied into read-only snapshots, so the frozen
dataclass cannot be bypassed by mutating an input or output tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Final

import numpy as np

from .observation import (
    ACTION_MASK_SHAPE,
    BOARD_SHAPE,
    GLOBAL_VECTOR_SHAPE,
    PINNED_OBSERVATION_CONTRACT_HASH,
    PolicyObservationV1,
)


OBSERVATION_V2_SCHEMA_VERSION: Final = "public-hybrid-v2-1"
"""Version identifier for the additive public hybrid observation contract."""

ENTITY_TOKEN_MAX: Final = 128
"""Maximum number of public entities represented in one observation."""

ENTITY_TOKEN_DIM: Final = 32
"""Number of normalized public features carried by each entity token."""

# Short aliases make the tensor contract easy to consume without importing
# the legacy module's implementation details.
NMAX: Final = ENTITY_TOKEN_MAX
D: Final = ENTITY_TOKEN_DIM
LEGAL_PLAY_SHAPE: Final = ACTION_MASK_SHAPE
ENTITY_TOKEN_SHAPE: Final = (ENTITY_TOKEN_MAX, ENTITY_TOKEN_DIM)

ENTITY_TOKEN_FEATURES: Final[tuple[str, ...]] = (
    "card_id",
    "side",
    "x",
    "y",
    "hp_fraction",
    "is_air",
    "is_building",
    "is_tower",
    "is_spell",
    "is_visible",
    "is_targetable",
    "lane",
    "distance_to_own_tower",
    "distance_to_enemy_tower",
    "distance_to_own_king",
    "distance_to_enemy_king",
    "velocity_x",
    "velocity_y",
    "state_idle",
    "state_moving",
    "state_attacking",
    "state_stunned",
    "state_slowed",
    "state_frozen",
    "state_invisible",
    "has_target",
    "recent_damage",
    "recent_deploy",
    "confidence",
    "age",
    "reserved_0",
    "reserved_1",
)

if len(ENTITY_TOKEN_FEATURES) != ENTITY_TOKEN_DIM:  # pragma: no cover - schema guard
    raise RuntimeError("entity-token feature names must match ENTITY_TOKEN_DIM")


def observation_v2_contract_manifest() -> dict[str, object]:
    """Return the canonical, JSON-compatible V2 schema manifest."""

    return {
        "schema_version": OBSERVATION_V2_SCHEMA_VERSION,
        "parent": {
            "schema_version": "vision-v1-exact-1",
            "contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
        },
        "tensors": {
            "board": {"shape": list(BOARD_SHAPE), "dtype": "float32"},
            "global_vector": {"shape": list(GLOBAL_VECTOR_SHAPE), "dtype": "float32"},
            "entity_tokens": {
                "shape": list(ENTITY_TOKEN_SHAPE),
                "dtype": "float32",
                "features": list(ENTITY_TOKEN_FEATURES),
                "padding": "zero",
            },
            "entity_mask": {
                "shape": [ENTITY_TOKEN_MAX],
                "dtype": "bool",
                "meaning": "true for a supplied public entity row",
            },
            "legal_play": {"shape": list(LEGAL_PLAY_SHAPE), "dtype": "bool"},
            "legal_wait": {"shape": [], "dtype": "bool"},
        },
        "array_order": "C-contiguous snapshots",
        "visibility": "public observation only; no BattleState private fields",
    }


def calculate_observation_v2_contract_hash() -> str:
    """Calculate the content hash identifying this exact tensor contract."""

    encoded = json.dumps(
        observation_v2_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


OBSERVATION_V2_CONTRACT_HASH: Final = calculate_observation_v2_contract_hash()
PINNED_OBSERVATION_V2_CONTRACT_HASH: Final = OBSERVATION_V2_CONTRACT_HASH


def _snapshot_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Validate an input tensor and return an immutable contiguous copy."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if dtype.kind == "f" and not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")

    snapshot = np.array(value, dtype=dtype, order="C", copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _normalize_public_entity_rows(
    public_entity_rows: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a bounded public-row array and construct its validity mask."""

    tokens = np.zeros(ENTITY_TOKEN_SHAPE, dtype=np.float32)
    mask = np.zeros((ENTITY_TOKEN_MAX,), dtype=bool)
    if public_entity_rows is None:
        tokens.setflags(write=False)
        mask.setflags(write=False)
        return tokens, mask

    if not isinstance(public_entity_rows, np.ndarray):
        raise TypeError("public_entity_rows must be a numpy array or None")
    if public_entity_rows.ndim != 2:
        raise ValueError(
            "public_entity_rows must have shape (num_entities, ENTITY_TOKEN_DIM)"
        )
    if public_entity_rows.shape[1:] != (ENTITY_TOKEN_DIM,):
        raise ValueError(
            "public_entity_rows must have shape "
            f"(num_entities, {ENTITY_TOKEN_DIM}), got {public_entity_rows.shape}"
        )
    if public_entity_rows.shape[0] > ENTITY_TOKEN_MAX:
        raise ValueError(
            f"public_entity_rows cannot contain more than {ENTITY_TOKEN_MAX} entities"
        )
    if public_entity_rows.dtype != np.dtype(np.float32):
        raise TypeError(
            "public_entity_rows must have dtype float32, "
            f"got {public_entity_rows.dtype}"
        )
    if not bool(np.isfinite(public_entity_rows).all()):
        raise ValueError("public_entity_rows must contain only finite values")

    count = public_entity_rows.shape[0]
    tokens[:count] = public_entity_rows
    mask[:count] = True
    tokens.setflags(write=False)
    mask.setflags(write=False)
    return tokens, mask


@dataclass(frozen=True, slots=True)
class PolicyObservationV2:
    """Immutable public hybrid observation for future structured policies.

    ``entity_tokens`` and ``entity_mask`` are independent of the V1 raster
    tensors.  A row is valid exactly when the corresponding mask entry is
    true; invalid rows are required to be zero in normal constructor output.
    Direct construction validates shape and finiteness but intentionally does
    not impose a policy on the values of masked rows, allowing serialized
    batches to be checked without silently rewriting them.
    """

    board: np.ndarray
    global_vector: np.ndarray
    entity_tokens: np.ndarray
    entity_mask: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    schema_version: str = OBSERVATION_V2_SCHEMA_VERSION
    contract_hash: str = OBSERVATION_V2_CONTRACT_HASH

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "board",
            _snapshot_array("board", self.board, BOARD_SHAPE, np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "global_vector",
            _snapshot_array(
                "global_vector",
                self.global_vector,
                GLOBAL_VECTOR_SHAPE,
                np.dtype(np.float32),
            ),
        )
        object.__setattr__(
            self,
            "entity_tokens",
            _snapshot_array(
                "entity_tokens",
                self.entity_tokens,
                ENTITY_TOKEN_SHAPE,
                np.dtype(np.float32),
            ),
        )
        object.__setattr__(
            self,
            "entity_mask",
            _snapshot_array(
                "entity_mask",
                self.entity_mask,
                (ENTITY_TOKEN_MAX,),
                np.dtype(bool),
            ),
        )
        object.__setattr__(
            self,
            "legal_play",
            _snapshot_array(
                "legal_play", self.legal_play, LEGAL_PLAY_SHAPE, np.dtype(bool)
            ),
        )

        if not isinstance(self.legal_wait, (bool, np.bool_)):
            raise TypeError("legal_wait must be boolean")
        object.__setattr__(self, "legal_wait", bool(self.legal_wait))

        if not isinstance(self.schema_version, str):
            raise TypeError("schema_version must be a string")
        if self.schema_version != OBSERVATION_V2_SCHEMA_VERSION:
            raise ValueError(f"unsupported observation schema version: {self.schema_version!r}")
        if not isinstance(self.contract_hash, str):
            raise TypeError("contract_hash must be a string")
        if self.contract_hash != OBSERVATION_V2_CONTRACT_HASH:
            raise ValueError("observation V2 contract hash does not match the schema")

    @classmethod
    def from_v1(
        cls,
        observation: PolicyObservationV1,
        public_entity_rows: np.ndarray | None = None,
        *,
        entity_rows: np.ndarray | None = None,
    ) -> "PolicyObservationV2":
        """Extend a public V1 observation with optional public entity rows.

        ``public_entity_rows`` must be a float32 array of shape
        ``[num_entities, ENTITY_TOKEN_DIM]`` with at most ``NMAX`` rows.  The
        optional ``entity_rows`` keyword is a readable alias; providing both
        aliases is rejected.  No authoritative state is consulted.
        """

        if not isinstance(observation, PolicyObservationV1):
            raise TypeError("observation must be a PolicyObservationV1")
        if public_entity_rows is not None and entity_rows is not None:
            raise TypeError("provide only one of public_entity_rows and entity_rows")
        rows = public_entity_rows if public_entity_rows is not None else entity_rows
        tokens, mask = _normalize_public_entity_rows(rows)
        return cls(
            board=observation.board,
            global_vector=observation.global_vector,
            entity_tokens=tokens,
            entity_mask=mask,
            legal_play=observation.legal_play,
            legal_wait=observation.legal_wait,
        )

    @classmethod
    def from_policy_observation_v1(
        cls,
        observation: PolicyObservationV1,
        public_entity_rows: np.ndarray | None = None,
        *,
        entity_rows: np.ndarray | None = None,
    ) -> "PolicyObservationV2":
        """Named alias for :meth:`from_v1`."""

        return cls.from_v1(
            observation,
            public_entity_rows,
            entity_rows=entity_rows,
        )

    def as_dict(self, *, copy: bool = False) -> dict[str, np.ndarray | bool | str]:
        """Return model-facing tensors and schema metadata."""

        maybe_copy = (lambda value: value.copy()) if copy else (lambda value: value)
        return {
            "board": maybe_copy(self.board),
            "global_vector": maybe_copy(self.global_vector),
            "entity_tokens": maybe_copy(self.entity_tokens),
            "entity_mask": maybe_copy(self.entity_mask),
            "legal_play": maybe_copy(self.legal_play),
            "legal_wait": bool(self.legal_wait),
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
        }

    def structured_action_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive WAIT/PLAY, card, and placement masks for autoregressive heads.

        The returned arrays are independent read-only snapshots.  Mode index
        zero is WAIT and mode index one is PLAY; a card is legal exactly when
        at least one cell in its center-based ``legal_play`` slice is legal.
        """

        card_mask = np.ascontiguousarray(self.legal_play.reshape(4, -1).any(axis=1), dtype=bool)
        mode_mask = np.asarray((self.legal_wait, bool(card_mask.any())), dtype=bool)
        placement_mask = np.ascontiguousarray(self.legal_play, dtype=bool)
        for array in (mode_mask, card_mask, placement_mask):
            array.setflags(write=False)
        return mode_mask, card_mask, placement_mask


def from_policy_observation_v1(
    observation: PolicyObservationV1,
    public_entity_rows: np.ndarray | None = None,
    *,
    entity_rows: np.ndarray | None = None,
) -> PolicyObservationV2:
    """Construct :class:`PolicyObservationV2` from the public V1 boundary."""

    return PolicyObservationV2.from_v1(
        observation,
        public_entity_rows,
        entity_rows=entity_rows,
    )


# A descriptive alias for callers that do not use the legacy ``Policy`` name.
PublicObservationV2 = PolicyObservationV2


__all__ = [
    "D",
    "ENTITY_TOKEN_DIM",
    "ENTITY_TOKEN_FEATURES",
    "ENTITY_TOKEN_MAX",
    "ENTITY_TOKEN_SHAPE",
    "LEGAL_PLAY_SHAPE",
    "NMAX",
    "OBSERVATION_V2_CONTRACT_HASH",
    "OBSERVATION_V2_SCHEMA_VERSION",
    "PINNED_OBSERVATION_V2_CONTRACT_HASH",
    "PolicyObservationV2",
    "PublicObservationV2",
    "calculate_observation_v2_contract_hash",
    "from_policy_observation_v1",
    "observation_v2_contract_manifest",
]
