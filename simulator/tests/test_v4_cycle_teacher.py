"""Public-state regressions for the opt-in V2 air-cycle rescue."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.rl.v4.hand_features import build_hand_tokens
from simulator.rl.v4.simulator_teacher import (
    TeacherError,
    cycle_teacher_label,
    defense_teacher_label,
)
from simulator.tests.test_v4_defense_teacher import _entities, _entity, _legal


def _state(
    hand: list[str],
    elixir: float,
    *,
    entities: tuple[np.ndarray, ...] = (),
    legal_play: np.ndarray | None = None,
    legal_wait: bool = True,
) -> dict[str, object]:
    tokens, mask = _entities(entities)
    return dict(
        hand_tokens=build_hand_tokens(hand, elixir),
        entity_tokens=tokens,
        entity_mask=mask,
        legal_play=_legal() if legal_play is None else legal_play,
        legal_wait=legal_wait,
        own_elixir=elixir,
    )


@pytest.mark.parametrize("elixir,mode", [(5.99, 0), (6.00, 1)])
def test_documented_ground_hand_stall_keeps_four_elixir(elixir: float, mode: int) -> None:
    state = _state(
        ["cannon", "log", "ice-golem", "hog-rider"], elixir,
        entities=(_entity("balloon", x=14.5 / 18, y=20.5 / 32),),
    )
    target = cycle_teacher_label(**state)
    assert target.mode == mode
    if mode:
        assert (target.card_slot, target.row, target.col) == (2, 29, 5)
        assert elixir - 2 >= 4
    else:
        assert target.wait_duration_idx == 0
    # Calling the original API remains frozen even at the rescue threshold.
    assert defense_teacher_label(**state).mode == 0


@pytest.mark.parametrize("elixir,mode", [(4.99, 0), (5.00, 1), (10.00, 1)])
def test_one_elixir_cycle_precedes_more_expensive_cards(elixir: float, mode: int) -> None:
    state = _state(
        ["cannon", "log", "ice-golem", "skeletons"], elixir,
        entities=(_entity("balloon"),),
    )
    target = cycle_teacher_label(**state)
    assert target.mode == mode
    if mode:
        assert target.card_slot == 3
        assert elixir - 1 >= 4
    assert defense_teacher_label(**state).mode == 0


@pytest.mark.parametrize("slot,cost", [(0, 3), (1, 2), (2, 2), (3, 4)])
@pytest.mark.parametrize("above_threshold", [False, True])
def test_every_fallback_requires_its_cost_plus_reserve(
    slot: int, cost: int, above_threshold: bool,
) -> None:
    elixir = cost + 4 if above_threshold else cost + 3.99
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[slot] = True
    state = _state(
        ["cannon", "log", "ice-golem", "hog-rider"], elixir,
        entities=(_entity("balloon"),), legal_play=legal,
    )
    target = cycle_teacher_label(**state)
    assert target.mode == int(above_threshold)
    if above_threshold:
        assert target.card_slot == slot
        assert elixir - cost >= 4


@pytest.mark.parametrize("unusable", ["masked", "distant-cell", "unplayable", "unaffordable"])
def test_unusable_cheapest_card_falls_through_legally(unusable: str) -> None:
    state = _state(
        ["skeletons", "log", "ice-golem", "cannon"], 10,
        entities=(_entity("balloon"),),
    )
    if unusable == "masked":
        state["legal_play"][0] = False
    elif unusable == "distant-cell":
        state["legal_play"][0] = False
        state["legal_play"][0, 0, 0] = True
    else:
        # Public per-card availability must also constrain a masked action.
        state["hand_tokens"][0, 4 if unusable == "unplayable" else 3] = 0
    target = cycle_teacher_label(**state)
    assert (target.mode, target.card_slot) == (1, 2)
    assert state["legal_play"][target.card_slot, target.row, target.col]


def test_nearest_legal_cycle_placement_is_used() -> None:
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[0, 28, 6] = True
    target = cycle_teacher_label(**_state(
        ["skeletons", "log", "ice-golem", "cannon"], 5,
        entities=(_entity("balloon"),), legal_play=legal,
    ))
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 0, 28, 6)


@pytest.mark.parametrize("legal_wait", [False, True])
def test_no_legal_cycle_honors_wait_legality(legal_wait: bool) -> None:
    state = _state(
        ["skeletons", "log", "ice-golem", "cannon"], 10,
        entities=(_entity("balloon"),),
        legal_play=np.zeros((4, 32, 18), dtype=bool), legal_wait=legal_wait,
    )
    if legal_wait:
        assert cycle_teacher_label(**state).mode == 0
    else:
        with pytest.raises(TeacherError):
            cycle_teacher_label(**state)


@pytest.mark.parametrize("answer", ["musketeer", "fireball", "ice-spirit"])
def test_actual_air_answer_keeps_ordinary_priority(answer: str) -> None:
    state = _state(
        ["skeletons", "ice-golem", "log", answer], 10,
        entities=(_entity("balloon", y=.75),),
    )
    target = cycle_teacher_label(**state)
    assert target == defense_teacher_label(**state)
    assert (target.mode, target.card_slot) == (1, 3)


@pytest.mark.parametrize("answer", ["musketeer", "fireball", "ice-spirit"])
def test_air_answer_present_but_masked_does_not_trigger_ground_cycle(answer: str) -> None:
    state = _state(
        ["skeletons", "ice-golem", "log", answer], 10,
        entities=(_entity("balloon", y=.75),), legal_play=_legal(disabled_slots=[3]),
    )
    target = cycle_teacher_label(**state)
    assert target == defense_teacher_label(**state)
    assert target.mode == 0


def test_urgent_pull_with_existing_air_damage_precedes_rescue() -> None:
    state = _state(
        ["skeletons", "ice-golem", "log", "cannon"], 10,
        entities=(
            _entity("balloon", x=.75, y=.75),
            _entity("musketeer", side=0, x=.75, y=.78),
        ),
    )
    target = cycle_teacher_label(**state)
    assert target == defense_teacher_label(**state)
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 3, 23, 12)


def test_urgent_ground_defense_precedes_cycling_for_remote_air() -> None:
    state = _state(
        ["skeletons", "ice-golem", "log", "cannon"], 10,
        entities=(_entity("balloon", y=.50), _entity("hog-rider", x=.75, y=.75)),
    )
    target = cycle_teacher_label(**state)
    assert target == defense_teacher_label(**state)
    assert (target.mode, target.card_slot, target.row, target.col) == (1, 3, 23, 12)


@pytest.mark.parametrize("entities", [
    pytest.param((), id="no-air"),
    pytest.param((_entity("balloon", y=12.5 / 32),), id="air-before-threat-cutoff"),
    pytest.param((_entity("balloon", side=0),), id="own-air"),
    pytest.param((_entity("balloon", visible=False),), id="invisible-air"),
    pytest.param((_entity("balloon", targetable=False),), id="untargetable-air"),
    pytest.param((_entity("balloon", hp=0),), id="dead-air"),
    pytest.param((_entity("balloon", is_tower=True),), id="tower-row"),
    pytest.param((_entity("balloon", is_spell=True),), id="spell-row"),
    pytest.param((
        _entity("balloon", x=.75, y=.60),
        _entity("musketeer", side=0, x=.75, y=.65),
    ), id="already-covered-air"),
    pytest.param((
        _entity("goblin-barrel", x=.42, y=.80, is_spell=False),
        _entity("goblin-barrel", x=.45, y=.81, is_spell=False),
    ), id="ground-bait-cluster"),
])
def test_unrelated_public_states_keep_v1_label(entities: tuple[np.ndarray, ...]) -> None:
    state = _state(["log", "skeletons", "ice-golem", "hog-rider"], 10, entities=entities)
    assert cycle_teacher_label(**state) == defense_teacher_label(**state)


def test_masked_air_entity_cannot_trigger_rescue() -> None:
    state = _state(
        ["cannon", "log", "ice-golem", "hog-rider"], 6,
        entities=(_entity("balloon"),),
    )
    state["entity_mask"][:] = False
    target = cycle_teacher_label(**state)
    assert target == defense_teacher_label(**state)
    assert (target.mode, target.card_slot, target.row) == (1, 3, 16)


def test_cycle_teacher_does_not_mutate_public_inputs() -> None:
    state = _state(
        ["skeletons", "log", "ice-golem", "cannon"], 5,
        entities=(_entity("balloon"),),
    )
    snapshots = {key: value.copy() for key, value in state.items() if isinstance(value, np.ndarray)}
    assert cycle_teacher_label(**state).mode == 1
    for key, snapshot in snapshots.items():
        np.testing.assert_array_equal(state[key], snapshot)
