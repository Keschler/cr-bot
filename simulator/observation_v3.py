"""Immutable public V3 observations for the V4 policy generation.

``PolicyObservationV3`` extends the V2 tensor contract with the three
interfaces the V4 paper requires, while preserving the public-only boundary:

* ``hand_tokens``: the four hand cards as structured per-card features
  (identity, cost, slot, affordability) instead of an opaque global slice;
* ``opponent belief``: elixir interval plus possible-hand/out-of-cycle
  distributions derived from public evidence only;
* ``event_history``: a short ring buffer of recent public events
  (deployments, damage, deaths, target changes, action timing).

Array inputs are copied into read-only snapshots.  No authoritative
``BattleState`` is consulted here; the deterministic bookkeeping lives in
:mod:`simulator.public_state_estimator`, which is the only producer allowed
to fill the belief/event fields from the public event stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Final

import numpy as np

from .geometry import GRID_COLS, GRID_ROWS
from .observation import (
    ACTION_MASK_SHAPE,
    BOARD_SHAPE,
    GLOBAL_VECTOR_SHAPE,
    PINNED_OBSERVATION_CONTRACT_HASH,
    PolicyObservationV1,
)
from .observation_v2 import (
    ENTITY_TOKEN_MAX,
    ENTITY_TOKEN_DIM,
    ENTITY_TOKEN_FEATURES,
    ENTITY_TOKEN_SHAPE,
    OBSERVATION_V2_CONTRACT_HASH,
)


OBSERVATION_V3_SCHEMA_VERSION: Final = "public-v3-1"
"""Version identifier for the public V3 observation contract."""

HAND_SLOTS: Final = 4
"""Number of hand cards represented as conditional query tokens."""

HAND_TOKEN_DIM: Final = 16
"""Features per hand token (identity, cost, slot, affordability, class)."""

HAND_TOKEN_FEATURES: Final[tuple[str, ...]] = (
    "card_id_norm",
    "elixir_cost_norm",
    "slot_index_norm",
    "affordable",
    "playable_now",
    "is_win_condition",
    "is_spell",
    "is_building",
    "is_swarm",
    "is_air_attacker",
    "cycle_position_norm",
    "next_card_indicator",
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
)

HAND_TOKEN_SHAPE: Final = (HAND_SLOTS, HAND_TOKEN_DIM)

BELIEF_CARD_COUNT: Final = 128
"""Belief vocabulary size (matches the V3 belief-head convention)."""

EVENT_HISTORY_LEN: Final = 16
"""Ring-buffer length for the public event summary."""

EVENT_DIM: Final = 8
"""Features per event row (type one-hot of 6 + normalized tick + magnitude)."""

EVENT_HISTORY_SHAPE: Final = (EVENT_HISTORY_LEN, EVENT_DIM)

EVENT_TYPES: Final[tuple[str, ...]] = (
    "own_deploy",
    "enemy_deploy",
    "own_tower_damaged",
    "enemy_tower_damaged",
    "unit_died",
    "action_taken",
)

if len(HAND_TOKEN_FEATURES) != HAND_TOKEN_DIM:  # pragma: no cover - schema guard
    raise RuntimeError("hand-token feature names must match HAND_TOKEN_DIM")
if len(EVENT_TYPES) != 6:  # pragma: no cover - schema guard
    raise RuntimeError("event types must fill EVENT_DIM - 2")


def observation_v3_contract_manifest() -> dict[str, object]:
    """Return the canonical, JSON-compatible V3 schema manifest."""

    return {
        "schema_version": OBSERVATION_V3_SCHEMA_VERSION,
        "parent": {
            "schema_version": "public-hybrid-v2-1",
            "contract_hash": OBSERVATION_V2_CONTRACT_HASH,
            "grandparent_contract_hash": PINNED_OBSERVATION_CONTRACT_HASH,
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
            "entity_mask": {"shape": [ENTITY_TOKEN_MAX], "dtype": "bool"},
            "hand_tokens": {
                "shape": list(HAND_TOKEN_SHAPE),
                "dtype": "float32",
                "features": list(HAND_TOKEN_FEATURES),
            },
            "opp_hand_probs": {"shape": [BELIEF_CARD_COUNT], "dtype": "float32"},
            "opp_out_of_cycle": {"shape": [BELIEF_CARD_COUNT], "dtype": "bool"},
            "opp_elixir_interval": {"shape": [2], "dtype": "float32"},
            "event_history": {"shape": list(EVENT_HISTORY_SHAPE), "dtype": "float32"},
            "legal_play": {"shape": list(ACTION_MASK_SHAPE), "dtype": "bool"},
            "legal_wait": {"shape": [], "dtype": "bool"},
        },
        "array_order": "C-contiguous snapshots",
        "visibility": "public observation only; no BattleState private fields",
    }


def calculate_observation_v3_contract_hash() -> str:
    """Calculate the content hash identifying this exact tensor contract."""

    encoded = json.dumps(
        observation_v3_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


OBSERVATION_V3_CONTRACT_HASH: Final = calculate_observation_v3_contract_hash()
PINNED_OBSERVATION_V3_CONTRACT_HASH: Final = OBSERVATION_V3_CONTRACT_HASH


def _snapshot_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
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


def empty_hand_tokens() -> np.ndarray:
    """Return a zero hand-token table (no card knowledge)."""

    return np.zeros(HAND_TOKEN_SHAPE, dtype=np.float32)


def uniform_hand_belief() -> tuple[np.ndarray, np.ndarray]:
    """Return an uninformative opponent-hand belief (uniform possible)."""

    probs = np.full((BELIEF_CARD_COUNT,), 1.0 / BELIEF_CARD_COUNT, dtype=np.float32)
    out_of_cycle = np.zeros((BELIEF_CARD_COUNT,), dtype=bool)
    return probs, out_of_cycle


def empty_event_history() -> np.ndarray:
    """Return a zero event ring buffer (no history)."""

    return np.zeros(EVENT_HISTORY_SHAPE, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PolicyObservationV3:
    """Immutable public observation consumed by the V4 actor."""

    board: np.ndarray
    global_vector: np.ndarray
    entity_tokens: np.ndarray
    entity_mask: np.ndarray
    hand_tokens: np.ndarray
    opp_hand_probs: np.ndarray
    opp_out_of_cycle: np.ndarray
    opp_elixir_interval: np.ndarray
    event_history: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    schema_version: str = OBSERVATION_V3_SCHEMA_VERSION
    contract_hash: str = OBSERVATION_V3_CONTRACT_HASH

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "board", _snapshot_array("board", self.board, BOARD_SHAPE, np.dtype(np.float32))
        )
        object.__setattr__(
            self,
            "global_vector",
            _snapshot_array(
                "global_vector", self.global_vector, GLOBAL_VECTOR_SHAPE, np.dtype(np.float32)
            ),
        )
        object.__setattr__(
            self,
            "entity_tokens",
            _snapshot_array(
                "entity_tokens", self.entity_tokens, ENTITY_TOKEN_SHAPE, np.dtype(np.float32)
            ),
        )
        object.__setattr__(
            self,
            "entity_mask",
            _snapshot_array(
                "entity_mask", self.entity_mask, (ENTITY_TOKEN_MAX,), np.dtype(bool)
            ),
        )
        object.__setattr__(
            self,
            "hand_tokens",
            _snapshot_array(
                "hand_tokens", self.hand_tokens, HAND_TOKEN_SHAPE, np.dtype(np.float32)
            ),
        )
        object.__setattr__(
            self,
            "opp_hand_probs",
            _snapshot_array(
                "opp_hand_probs",
                self.opp_hand_probs,
                (BELIEF_CARD_COUNT,),
                np.dtype(np.float32),
            ),
        )
        if bool((self.opp_hand_probs < 0.0).any()) or bool((self.opp_hand_probs > 1.0).any()):
            raise ValueError("opp_hand_probs must be in [0, 1]")
        object.__setattr__(
            self,
            "opp_out_of_cycle",
            _snapshot_array(
                "opp_out_of_cycle",
                self.opp_out_of_cycle,
                (BELIEF_CARD_COUNT,),
                np.dtype(bool),
            ),
        )
        object.__setattr__(
            self,
            "opp_elixir_interval",
            _snapshot_array(
                "opp_elixir_interval",
                self.opp_elixir_interval,
                (2,),
                np.dtype(np.float32),
            ),
        )
        elixir = self.opp_elixir_interval
        if bool((elixir < 0.0).any()) or bool((elixir > 10.0).any()) or not bool(elixir[0] <= elixir[1]):
            raise ValueError("opp_elixir_interval must satisfy 0 <= lo <= hi <= 10")
        object.__setattr__(
            self,
            "event_history",
            _snapshot_array(
                "event_history", self.event_history, EVENT_HISTORY_SHAPE, np.dtype(np.float32)
            ),
        )
        object.__setattr__(
            self,
            "legal_play",
            _snapshot_array(
                "legal_play", self.legal_play, ACTION_MASK_SHAPE, np.dtype(bool)
            ),
        )
        if not isinstance(self.legal_wait, (bool, np.bool_)):
            raise TypeError("legal_wait must be boolean")
        object.__setattr__(self, "legal_wait", bool(self.legal_wait))
        if not isinstance(self.schema_version, str):
            raise TypeError("schema_version must be a string")
        if self.schema_version != OBSERVATION_V3_SCHEMA_VERSION:
            raise ValueError(f"unsupported observation schema version: {self.schema_version!r}")
        if not isinstance(self.contract_hash, str):
            raise TypeError("contract_hash must be a string")
        if self.contract_hash != OBSERVATION_V3_CONTRACT_HASH:
            raise ValueError("observation V3 contract hash does not match the schema")

    @classmethod
    def from_v2(
        cls,
        observation_v1: PolicyObservationV1,
        public_entity_rows: np.ndarray | None = None,
        *,
        hand_tokens: np.ndarray | None = None,
        opp_hand_probs: np.ndarray | None = None,
        opp_out_of_cycle: np.ndarray | None = None,
        opp_elixir_interval: tuple[float, float] | np.ndarray | None = None,
        event_history: np.ndarray | None = None,
    ) -> "PolicyObservationV3":
        """Build a V3 observation from the public V1 boundary plus V3 fields.

        Only the V3 additions (hand, belief, events) are new inputs; the
        raster/global/entity/mask tensors follow the V2 construction path so
        no private state can enter through this builder.
        """

        from .observation_v2 import PolicyObservationV2

        v2 = PolicyObservationV2.from_v1(observation_v1, public_entity_rows)
        if hand_tokens is None:
            hand_tokens = empty_hand_tokens()
        if opp_hand_probs is None or opp_out_of_cycle is None:
            default_probs, default_out = uniform_hand_belief()
            if opp_hand_probs is None:
                opp_hand_probs = default_probs
            if opp_out_of_cycle is None:
                opp_out_of_cycle = default_out
        if opp_elixir_interval is None:
            interval = np.asarray([0.0, 10.0], dtype=np.float32)
        else:
            interval = np.asarray(opp_elixir_interval, dtype=np.float32)
        if event_history is None:
            event_history = empty_event_history()
        return cls(
            board=v2.board,
            global_vector=v2.global_vector,
            entity_tokens=v2.entity_tokens,
            entity_mask=v2.entity_mask,
            hand_tokens=np.asarray(hand_tokens, dtype=np.float32),
            opp_hand_probs=np.asarray(opp_hand_probs, dtype=np.float32),
            opp_out_of_cycle=np.asarray(opp_out_of_cycle, dtype=bool),
            opp_elixir_interval=interval,
            event_history=np.asarray(event_history, dtype=np.float32),
            legal_play=v2.legal_play,
            legal_wait=v2.legal_wait,
        )

    def structured_action_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive WAIT/PLAY, card, and placement masks (mode 0=WAIT, 1=PLAY)."""

        card_mask = np.ascontiguousarray(
            self.legal_play.reshape(HAND_SLOTS, -1).any(axis=1), dtype=bool
        )
        mode_mask = np.asarray((self.legal_wait, bool(card_mask.any())), dtype=bool)
        placement_mask = np.ascontiguousarray(self.legal_play, dtype=bool)
        for array in (mode_mask, card_mask, placement_mask):
            array.setflags(write=False)
        return mode_mask, card_mask, placement_mask

    def grid_shape(self) -> tuple[int, int]:
        return (GRID_ROWS, GRID_COLS)


PublicObservationV3 = PolicyObservationV3


__all__ = [
    "BELIEF_CARD_COUNT",
    "EVENT_DIM",
    "EVENT_HISTORY_LEN",
    "EVENT_HISTORY_SHAPE",
    "EVENT_TYPES",
    "HAND_SLOTS",
    "HAND_TOKEN_DIM",
    "HAND_TOKEN_FEATURES",
    "HAND_TOKEN_SHAPE",
    "OBSERVATION_V3_CONTRACT_HASH",
    "OBSERVATION_V3_SCHEMA_VERSION",
    "PINNED_OBSERVATION_V3_CONTRACT_HASH",
    "PolicyObservationV3",
    "PublicObservationV3",
    "calculate_observation_v3_contract_hash",
    "empty_event_history",
    "empty_hand_tokens",
    "observation_v3_contract_manifest",
    "uniform_hand_belief",
]
