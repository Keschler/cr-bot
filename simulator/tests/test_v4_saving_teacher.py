"""Focused semantic regressions for the opt-in saving teacher.

The fixtures deliberately use the public V2 entity-token contract.  Entity
card IDs are normalized by the public catalog maximum (124); learner-hand
tokens use the separate V4 hand vocabulary and are therefore built through
``build_hand_tokens``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from cr_bot.domain.card_metadata import CARD_METADATA
from simulator.observation_v2 import ENTITY_TOKEN_DIM, ENTITY_TOKEN_MAX
from simulator.rl.v4.hand_features import build_hand_tokens
from simulator.rl.v4.simulator_teacher import saving_teacher_label


_CARD_ID = 0
_SIDE = 1
_X = 2
_Y = 3
_HP = 4
_IS_AIR = 5
_IS_BUILDING = 6
_IS_TOWER = 7
_IS_SPELL = 8
_IS_VISIBLE = 9
_IS_TARGETABLE = 10

_MAX_PUBLIC_CARD_ID = max(
    int(metadata["id"])
    for metadata in CARD_METADATA.values()
    if isinstance(metadata, dict) and type(metadata.get("id")) is int
)


def _entity(
    card: str,
    *,
    side: int = 1,
    x: float = 0.50,
    y: float = 0.60,
    hp: float = 1.0,
    is_air: bool | None = None,
    is_building: bool | None = None,
    is_tower: bool = False,
    is_spell: bool | None = None,
    visible: bool = True,
    targetable: bool = True,
) -> np.ndarray:
    """Build one normalized public entity row with native metadata flags."""

    metadata = CARD_METADATA[card]
    row = np.zeros((ENTITY_TOKEN_DIM,), dtype=np.float32)
    row[_CARD_ID] = np.float32(float(metadata["id"]) / _MAX_PUBLIC_CARD_ID)
    row[_SIDE] = np.float32(side)
    row[_X] = np.float32(x)
    row[_Y] = np.float32(y)
    row[_HP] = np.float32(hp)
    row[_IS_AIR] = np.float32(metadata["is_air"] if is_air is None else is_air)
    row[_IS_BUILDING] = np.float32(
        metadata["kind"] == "building" if is_building is None else is_building
    )
    row[_IS_TOWER] = np.float32(is_tower)
    row[_IS_SPELL] = np.float32(
        metadata["kind"] == "spell" if is_spell is None else is_spell
    )
    row[_IS_VISIBLE] = np.float32(visible)
    row[_IS_TARGETABLE] = np.float32(targetable)
    return row


def _entities(rows: Iterable[np.ndarray] = ()) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((ENTITY_TOKEN_MAX, ENTITY_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((ENTITY_TOKEN_MAX,), dtype=bool)
    for index, row in enumerate(rows):
        if index >= ENTITY_TOKEN_MAX:
            raise AssertionError("fixture exceeded the public entity-token limit")
        tokens[index] = row
        mask[index] = True
    return tokens, mask


def _legal(*, disabled_slots: Iterable[int] = ()) -> np.ndarray:
    legal = np.ones((4, 32, 18), dtype=bool)
    for slot in disabled_slots:
        legal[int(slot)] = False
    return legal


def _label(
    hand: list[str],
    elixir: float,
    *,
    entities: Iterable[np.ndarray] = (),
    legal_play: np.ndarray | None = None,
    legal_wait: bool = True,
):
    entity_tokens, entity_mask = _entities(entities)
    return saving_teacher_label(
        hand_tokens=build_hand_tokens(hand, elixir),
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=_legal() if legal_play is None else legal_play,
        legal_wait=legal_wait,
        own_elixir=elixir,
    )


def test_public_entity_fixture_uses_catalog_124_and_hand_127_contracts() -> None:
    assert _MAX_PUBLIC_CARD_ID == 124
    hand = build_hand_tokens(["hog-rider", "cannon", "musketeer", "log"], 4.0)
    assert hand[0, 0] == pytest.approx(49.0 / 127.0)
    assert _entity("hog-rider")[_CARD_ID] == pytest.approx(49.0 / 124.0)


def test_air_only_ground_hand_waits() -> None:
    target = _label(
        ["cannon", "ice-golem", "skeletons", "log"],
        4.0,
        entities=[_entity("balloon", y=0.60)],
    )

    assert target.mode == 0


def test_air_defense_falls_back_to_fireball() -> None:
    target = _label(
        ["fireball", "skeletons", "ice-golem", "log"],
        4.0,
        entities=[_entity("balloon", y=0.70)],
    )

    assert (target.mode, target.card_slot) == (1, 0)


def test_pure_air_threat_does_not_choose_log_before_fireball() -> None:
    target = _label(
        ["log", "fireball", "skeletons", "ice-golem"],
        4.0,
        entities=[_entity("balloon", y=0.60)],
    )

    assert (target.mode, target.card_slot) == (1, 1)


def test_ground_cluster_prefers_log() -> None:
    target = _label(
        ["log", "fireball", "skeletons", "ice-golem"],
        4.0,
        entities=[
            _entity("skeletons", x=0.42, y=0.58),
            _entity("skeletons", x=0.45, y=0.59),
        ],
    )

    assert (target.mode, target.card_slot) == (1, 0)


def test_urgent_air_threat_precedes_distant_ground_cluster() -> None:
    target = _label(
        ["log", "musketeer", "fireball", "ice-golem"],
        4.0,
        entities=[
            _entity("balloon", x=0.75, y=0.90),
            _entity("skeletons", x=0.20, y=0.50),
            _entity("skeletons", x=0.22, y=0.51),
        ],
    )

    assert (target.mode, target.card_slot) == (1, 1)
    assert (target.row, target.col) == (30, 13)


def test_urgent_ground_threat_precedes_distant_air_cluster() -> None:
    target = _label(
        ["fireball", "cannon", "musketeer", "log"],
        4.0,
        entities=[
            _entity("hog-rider", x=0.75, y=0.90),
            _entity("minions", x=0.20, y=0.50),
            _entity("minions", x=0.22, y=0.51),
        ],
    )

    assert (target.mode, target.card_slot) == (1, 1)
    assert (target.row, target.col) == (30, 13)


@pytest.mark.parametrize("cover_card", ["cannon", "musketeer"])
def test_healthy_same_lane_cannon_or_musketeer_covers_nonurgent_ground_threat(
    cover_card: str,
) -> None:
    target = _label(
        ["skeletons", "ice-golem", "ice-spirit", "log"],
        3.0,
        entities=[
            _entity("skeletons", x=0.50, y=0.60),
            _entity(cover_card, side=0, x=0.50, y=0.57),
        ],
    )

    assert target.mode == 0


def test_urgent_ground_threat_bypasses_own_body_coverage_at_y_070() -> None:
    target = _label(
        ["cannon", "musketeer", "fireball", "log"],
        3.0,
        entities=[
            _entity("hog-rider", x=0.50, y=0.70),
            _entity("musketeer", side=0, x=0.50, y=0.67),
        ],
    )

    assert (target.mode, target.card_slot) == (1, 0)


def test_skeletons_do_not_count_as_air_coverage() -> None:
    target = _label(
        ["musketeer", "ice-golem", "skeletons", "log"],
        4.0,
        entities=[
            _entity("balloon", x=0.50, y=0.60),
            _entity("skeletons", side=0, x=0.50, y=0.57),
        ],
    )

    assert (target.mode, target.card_slot) == (1, 0)


def test_hog_is_saved_for_four_elixir() -> None:
    target = _label(["hog-rider", "skeletons", "ice-golem", "log"], 3.0)

    assert target.mode == 0


def test_safe_affordable_hog_is_played() -> None:
    target = _label(["hog-rider", "skeletons", "ice-golem", "log"], 4.0)

    assert (target.mode, target.card_slot) == (1, 0)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_entity("skeletons", side=0, y=0.90), id="own-player"),
        pytest.param(_entity("skeletons", visible=False, y=0.90), id="invisible"),
        pytest.param(_entity("skeletons", hp=0.0, y=0.90), id="dead"),
        pytest.param(_entity("skeletons", targetable=False, y=0.90), id="untargetable"),
        pytest.param(_entity("skeletons", is_tower=True, y=0.90), id="tower"),
        pytest.param(_entity("fireball", is_spell=True, y=0.90), id="spell"),
    ],
)
def test_non_threat_public_rows_are_filtered(row: np.ndarray) -> None:
    target = _label(
        ["hog-rider", "skeletons", "ice-golem", "log"],
        4.0,
        entities=[row],
    )

    assert (target.mode, target.card_slot) == (1, 0)
