from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from simulator.observation import (
    ACTION_MASK_SHAPE,
    BOARD_SHAPE,
    GLOBAL_VECTOR_SHAPE,
    PolicyObservationV1,
)
from simulator.observation_v2 import (
    D,
    ENTITY_TOKEN_DIM,
    ENTITY_TOKEN_FEATURES,
    ENTITY_TOKEN_MAX,
    ENTITY_TOKEN_SHAPE,
    LEGAL_PLAY_SHAPE,
    OBSERVATION_V2_CONTRACT_HASH,
    OBSERVATION_V2_SCHEMA_VERSION,
    PolicyObservationV2,
    calculate_observation_v2_contract_hash,
    from_policy_observation_v1,
    observation_v2_contract_manifest,
)


def _v1() -> PolicyObservationV1:
    board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    global_vector = np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float32)
    spatial_masks = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    legal_play = np.zeros(ACTION_MASK_SHAPE, dtype=bool)
    board[0, 0, 0] = 0.25
    global_vector[0] = 0.5
    legal_play[0, 20, 9] = True
    return PolicyObservationV1(
        board=board,
        global_vector=global_vector,
        spatial_masks=spatial_masks,
        legal_play=legal_play,
        legal_wait=True,
    )


def _valid_v2_kwargs() -> dict[str, object]:
    return {
        "board": np.zeros(BOARD_SHAPE, dtype=np.float32),
        "global_vector": np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float32),
        "entity_tokens": np.zeros(ENTITY_TOKEN_SHAPE, dtype=np.float32),
        "entity_mask": np.zeros((ENTITY_TOKEN_MAX,), dtype=bool),
        "legal_play": np.zeros(LEGAL_PLAY_SHAPE, dtype=bool),
        "legal_wait": True,
    }


def test_v2_contract_is_documented_and_hashes_deterministically() -> None:
    manifest = observation_v2_contract_manifest()

    assert OBSERVATION_V2_SCHEMA_VERSION == "public-hybrid-v2-1"
    assert ENTITY_TOKEN_DIM == D == len(ENTITY_TOKEN_FEATURES) == 32
    assert manifest["schema_version"] == OBSERVATION_V2_SCHEMA_VERSION
    assert manifest["tensors"]["entity_tokens"]["shape"] == [ENTITY_TOKEN_MAX, D]
    assert calculate_observation_v2_contract_hash() == OBSERVATION_V2_CONTRACT_HASH
    assert OBSERVATION_V2_CONTRACT_HASH.startswith("sha256:")


def test_from_v1_copies_public_tensors_and_pads_entity_rows() -> None:
    rows = np.zeros((2, ENTITY_TOKEN_DIM), dtype=np.float32)
    rows[0, 0] = 7.0
    rows[1, 3] = -0.25

    observation = PolicyObservationV2.from_v1(_v1(), public_entity_rows=rows)

    assert observation.board.shape == BOARD_SHAPE
    assert observation.global_vector.shape == GLOBAL_VECTOR_SHAPE
    assert observation.entity_tokens.shape == ENTITY_TOKEN_SHAPE
    assert observation.entity_mask.shape == (ENTITY_TOKEN_MAX,)
    assert observation.legal_play.shape == LEGAL_PLAY_SHAPE
    assert observation.board.dtype == np.dtype(np.float32)
    assert observation.global_vector.dtype == np.dtype(np.float32)
    assert observation.entity_tokens.dtype == np.dtype(np.float32)
    assert observation.entity_mask.dtype == np.dtype(bool)
    assert observation.legal_play.dtype == np.dtype(bool)
    assert observation.entity_mask[:2].tolist() == [True, True]
    assert not observation.entity_mask[2]
    np.testing.assert_array_equal(observation.entity_tokens[:2], rows)
    np.testing.assert_array_equal(
        observation.entity_tokens[2:], np.zeros((ENTITY_TOKEN_MAX - 2, D), dtype=np.float32)
    )
    assert observation.legal_wait is True
    assert observation.schema_version == OBSERVATION_V2_SCHEMA_VERSION
    assert observation.contract_hash == OBSERVATION_V2_CONTRACT_HASH

    rows[0, 0] = 99.0
    assert observation.entity_tokens[0, 0] == np.float32(7.0)
    assert not observation.board.flags.writeable
    assert not observation.global_vector.flags.writeable
    assert not observation.entity_tokens.flags.writeable
    assert not observation.entity_mask.flags.writeable
    assert not observation.legal_play.flags.writeable


def test_structured_action_masks_derive_autoregressive_legality() -> None:
    observation = PolicyObservationV2.from_v1(_v1())

    mode, card, placement = observation.structured_action_masks()

    assert mode.tolist() == [True, True]
    assert card.shape == (4,)
    assert card.tolist() == [True, False, False, False]
    assert placement[0, 20, 9]
    assert not mode.flags.writeable
    assert not card.flags.writeable
    assert not placement.flags.writeable


def test_module_constructor_and_dict_alias_match_class_constructor() -> None:
    rows = np.ones((1, ENTITY_TOKEN_DIM), dtype=np.float32)
    from_module = from_policy_observation_v1(_v1(), entity_rows=rows)
    from_class = PolicyObservationV2.from_policy_observation_v1(_v1(), rows)

    np.testing.assert_array_equal(from_module.entity_tokens, from_class.entity_tokens)
    np.testing.assert_array_equal(from_module.entity_mask, from_class.entity_mask)
    values = from_module.as_dict(copy=True)
    assert set(values) == {
        "board",
        "global_vector",
        "entity_tokens",
        "entity_mask",
        "legal_play",
        "legal_wait",
        "schema_version",
        "contract_hash",
    }
    assert values["entity_tokens"].flags.writeable


def test_frozen_dataclass_and_arrays_are_immutable() -> None:
    observation = PolicyObservationV2.from_v1(_v1())

    with pytest.raises(FrozenInstanceError):
        observation.legal_wait = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        observation.board[0, 0, 0] = 1.0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("board", np.zeros((1,), dtype=np.float32), ValueError),
        ("global_vector", np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float64), TypeError),
        (
            "entity_tokens",
            np.full(ENTITY_TOKEN_SHAPE, np.nan, dtype=np.float32),
            ValueError,
        ),
        ("entity_mask", np.zeros((ENTITY_TOKEN_MAX,), dtype=np.uint8), TypeError),
        ("legal_play", np.zeros((4, 32, 17), dtype=bool), ValueError),
        ("legal_wait", 1, TypeError),
    ],
)
def test_direct_constructor_rejects_invalid_shape_dtype_or_finite_values(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    kwargs = _valid_v2_kwargs()
    kwargs[field] = value

    with pytest.raises(error):
        PolicyObservationV2(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (np.zeros((D,), dtype=np.float32), ValueError),
        (np.zeros((1, D), dtype=np.float64), TypeError),
        (np.full((1, D), np.inf, dtype=np.float32), ValueError),
        (np.zeros((ENTITY_TOKEN_MAX + 1, D), dtype=np.float32), ValueError),
    ],
)
def test_entity_row_constructor_is_strict(rows: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        PolicyObservationV2.from_v1(_v1(), public_entity_rows=rows)


def test_metadata_and_entity_row_aliases_cannot_be_ambiguous() -> None:
    kwargs = _valid_v2_kwargs()
    kwargs["schema_version"] = "wrong-version"
    with pytest.raises(ValueError, match="schema version"):
        PolicyObservationV2(**kwargs)  # type: ignore[arg-type]

    kwargs = _valid_v2_kwargs()
    kwargs["contract_hash"] = "sha256:wrong"
    with pytest.raises(ValueError, match="contract hash"):
        PolicyObservationV2(**kwargs)  # type: ignore[arg-type]

    rows = np.zeros((1, D), dtype=np.float32)
    with pytest.raises(TypeError, match="only one"):
        PolicyObservationV2.from_v1(_v1(), rows, entity_rows=rows)
