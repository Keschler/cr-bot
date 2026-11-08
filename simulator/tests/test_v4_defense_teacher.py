"""Public-only regressions for the opt-in defense teacher.

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
from simulator.rl.v4.simulator_teacher import (
    TeacherError, defense_teacher_context, defense_teacher_label, saving_teacher_label,
)


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
    return defense_teacher_label(
        hand_tokens=build_hand_tokens(hand, elixir),
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=_legal() if legal_play is None else legal_play,
        legal_wait=legal_wait,
        own_elixir=elixir,
    )


def _context(*entities: np.ndarray):
    tokens, mask = _entities(entities)
    return defense_teacher_context(tokens, mask)


def test_bridge_xbow_below_old_cutoff_receives_same_lane_hog() -> None:
    threat = _entity("x-bow", x=14.5 / 18, y=13.5 / 32)
    target = _label(["hog-rider", "skeletons", "ice-golem", "log"], 4, entities=[threat])
    assert _context(threat).siege_threats == 1
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 0, 17, 14)


def test_remote_rear_xbow_is_not_in_own_tower_range() -> None:
    threat = _entity("x-bow", x=14.5 / 18, y=10.5 / 32)
    target = _label(["hog-rider", "skeletons", "ice-golem", "log"], 4, entities=[threat])
    assert _context(threat).siege_threats == 0
    assert (target.mode, target.card_slot, target.col) == (1, 0, 4)


def test_supplied_public_tower_positions_control_siege_range() -> None:
    threat = _entity("x-bow", x=14.5 / 18, y=13.5 / 32)
    # The one supplied live tower is outside this X-Bow's public range.
    tower = _entity("cannon", side=0, x=1 / 18, y=30 / 32, is_tower=True)
    assert _context(threat, tower).siege_threats == 0


def test_weakened_bridge_xbow_receives_fireball() -> None:
    target = _label(
        ["hog-rider", "fireball", "musketeer", "log"], 4,
        entities=[_entity("x-bow", x=14.5 / 18, y=13.5 / 32, hp=.35)],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 1, 13, 14)


def test_siege_answer_uses_affordable_ranged_fallback() -> None:
    target = _label(
        ["musketeer", "cannon", "fireball", "log"], 3,
        entities=[_entity("x-bow", x=3.5 / 18, y=13.5 / 32)],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 1, 18, 3)


def test_masked_siege_answers_fall_through_legally() -> None:
    target = _label(
        ["hog-rider", "musketeer", "fireball", "log"], 4,
        entities=[_entity("x-bow", x=14.5 / 18, y=13.5 / 32)],
        legal_play=_legal(disabled_slots=[0, 1]),
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 2, 13, 14)


def test_urgent_air_precedes_remote_xbow() -> None:
    target = _label(
        ["hog-rider", "musketeer", "fireball", "log"], 4,
        entities=[
            _entity("x-bow", x=3.5 / 18, y=13.5 / 32),
            _entity("balloon", x=14.5 / 18, y=.80),
        ],
    )
    assert (target.mode, target.card_slot) == (1, 1)
    assert (target.row, target.col) == (26, 11)


def test_air_before_old_cutoff_deploys_sustained_defender() -> None:
    target = _label(
        ["ice-spirit", "musketeer", "fireball", "cannon"], 4,
        entities=[_entity("balloon", x=14.5 / 18, y=14 / 32)],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 1, 18, 11)


def test_air_ground_only_hand_waits_with_short_revisit() -> None:
    target = _label(
        ["cannon", "ice-golem", "skeletons", "log"], 4,
        entities=[_entity("balloon", y=.60)],
    )
    assert (target.mode, target.wait_duration_idx) == (0, 0)


def test_urgent_air_fireball_precedes_one_shot_ice_spirit() -> None:
    target = _label(
        ["ice-spirit", "fireball", "skeletons", "log"], 4,
        entities=[_entity("balloon", x=.75, y=.75)],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 1, 24, 13)


def test_lone_remote_tank_reserves_elixir_for_sustained_answer() -> None:
    target = _label(
        ["fireball", "hog-rider", "ice-golem", "log"], 4,
        entities=[_entity("lava-hound", x=.75, y=.50)],
    )
    assert target.mode == 0


def test_fragile_aerial_cluster_prefers_fireball() -> None:
    target = _label(
        ["musketeer", "fireball", "ice-spirit", "log"], 4,
        entities=[
            _entity("minions", x=.75, y=.60),
            _entity("minions", x=.76, y=.61),
            _entity("skeletons", x=.20, y=.55),
        ],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 1, 19, 13)


def test_durable_air_push_uses_musketeer_before_fireball() -> None:
    target = _label(
        ["musketeer", "fireball", "ice-spirit", "log"], 4,
        entities=[_entity("balloon", x=.75, y=.60), _entity("lava-hound", x=.76, y=.61)],
    )
    assert (target.mode, target.card_slot) == (1, 0)


def test_healthy_nearby_musketeer_covers_isolated_nonurgent_air() -> None:
    balloon = _entity("balloon", x=.75, y=.60)
    musketeer = _entity("musketeer", side=0, x=.75, y=.65)
    target = _label(
        ["musketeer", "hog-rider", "fireball", "log"], 4,
        entities=[balloon, musketeer],
    )
    context = _context(balloon, musketeer)
    assert context.covered_air_threats == 1
    assert context.covered_threats == 1
    assert target.mode == 0


@pytest.mark.parametrize("defender", [
    pytest.param(_entity("cannon", side=0, x=.75, y=.65), id="ground-only"),
    pytest.param(_entity("ice-spirit", side=0, x=.75, y=.65), id="one-shot"),
    pytest.param(_entity("musketeer", side=0, x=.20, y=.65), id="wrong-lane"),
    pytest.param(_entity("musketeer", side=0, x=.75, y=.85), id="distant"),
    pytest.param(_entity("musketeer", side=0, x=.75, y=.65, hp=.20), id="low-hp"),
    pytest.param(_entity("musketeer", side=0, x=.75, y=.65, visible=False), id="invisible"),
])
def test_inadequate_body_does_not_suppress_air_answer(defender: np.ndarray) -> None:
    balloon = _entity("balloon", x=.75, y=.60)
    target = _label(["musketeer", "hog-rider", "fireball", "log"], 4, entities=[balloon, defender])
    assert _context(balloon, defender).covered_air_threats == 0
    assert (target.mode, target.card_slot) == (1, 0)


def test_crowded_air_lane_retains_response_with_existing_musketeer() -> None:
    balloon = _entity("balloon", x=.75, y=.60)
    hound = _entity("lava-hound", x=.75, y=.62)
    musketeer = _entity("musketeer", side=0, x=.75, y=.65)
    assert _context(balloon, hound, musketeer).covered_air_threats == 0
    target = _label(
        ["musketeer", "hog-rider", "fireball", "log"], 4,
        entities=[balloon, hound, musketeer],
    )
    assert (target.mode, target.card_slot) == (1, 0)


def test_urgent_air_receives_supplemental_fireball_with_existing_musketeer() -> None:
    balloon = _entity("balloon", x=.75, y=.75)
    musketeer = _entity("musketeer", side=0, x=.75, y=.78)
    assert _context(balloon, musketeer).covered_air_threats == 0
    target = _label(
        ["musketeer", "hog-rider", "fireball", "log"], 4,
        entities=[balloon, musketeer],
    )
    assert (target.mode, target.card_slot) == (1, 2)


def test_urgent_balloon_can_be_pulled_when_existing_musketeer_provides_damage() -> None:
    target = _label(
        ["cannon", "hog-rider", "ice-golem", "log"], 3,
        entities=[
            _entity("balloon", x=.75, y=.70),
            _entity("musketeer", side=0, x=.75, y=.73),
        ],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 0, 23, 12)


@pytest.mark.parametrize("x,col", [(3.5 / 18, 7), (14.5 / 18, 10)])
def test_cannon_ground_building_targeter_uses_central_pull(x: float, col: int) -> None:
    target = _label(
        ["cannon", "musketeer", "fireball", "log"], 3,
        entities=[_entity("hog-rider", x=x, y=.60)],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 0, 20, col)


def test_unrelated_ground_cluster_preserves_saving_teacher_choice() -> None:
    tokens, mask = _entities([
        _entity("skeletons", x=.42, y=.58),
        _entity("skeletons", x=.45, y=.59),
    ])
    kwargs = dict(
        hand_tokens=build_hand_tokens(["log", "fireball", "musketeer", "ice-golem"], 4),
        entity_tokens=tokens, entity_mask=mask, legal_play=_legal(), legal_wait=True, own_elixir=4,
    )
    assert defense_teacher_label(**kwargs) == saving_teacher_label(**kwargs)


@pytest.mark.parametrize("row", [
    _entity("x-bow", side=0, y=.90),
    _entity("x-bow", visible=False, y=.90),
    _entity("x-bow", hp=0, y=.90),
    _entity("x-bow", targetable=False, y=.90),
    _entity("x-bow", is_tower=True, y=.90),
    _entity("fireball", is_spell=True, y=.90),
])
def test_non_threat_rows_are_filtered(row: np.ndarray) -> None:
    target = _label(["hog-rider", "skeletons", "ice-golem", "log"], 4, entities=[row])
    assert (target.mode, target.card_slot) == (1, 0)


def test_no_legal_answer_or_wait_raises() -> None:
    with pytest.raises(TeacherError):
        _label(
            ["musketeer", "fireball", "hog-rider", "log"], 4,
            entities=[_entity("balloon", y=.75)],
            legal_play=np.zeros((4, 32, 18), dtype=bool), legal_wait=False,
        )


@pytest.mark.parametrize("col", [6, 7, 8, 9, 10, 11])
def test_inward_musketeer_anchor_does_not_cross_lane_midline(col: int) -> None:
    target = _label(
        ["musketeer", "fireball", "ice-golem", "log"], 4,
        entities=[_entity("balloon", x=(col + .5) / 18, y=.60)],
    )
    assert target.card_slot == 0
    assert (target.col < 9) == (col < 9)


@pytest.mark.parametrize("x,col", [(3.5 / 18, 4), (14.5 / 18, 13)])
def test_advanced_ground_pull_moves_cannon_toward_threat_lane(x: float, col: int) -> None:
    target = _label(
        ["cannon", "musketeer", "fireball", "log"], 3,
        entities=[_entity("hog-rider", x=x, y=22.5 / 32)],
    )
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 0, 23, col)


def test_mixed_push_does_not_assume_musketeer_can_cover_air_body() -> None:
    balloon = _entity("balloon", x=.75, y=.60)
    knight = _entity("knight", x=.75, y=.63)
    musketeer = _entity("musketeer", side=0, x=.75, y=.65)
    assert _context(balloon, knight, musketeer).covered_air_threats == 0
