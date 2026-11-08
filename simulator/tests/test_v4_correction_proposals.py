"""Public hypotheses retain factual actions and require legal alternatives."""

from __future__ import annotations

import numpy as np
import pytest

from cr_bot.domain.card_metadata import CARD_METADATA
from simulator.rl.v4.correction_proposals import (
    candidate_categories,
    propose_correction_branches,
)
from simulator.rl.v4.frontier_eval import BranchSpec, action_key
from simulator.rl.v4.hand_features import build_hand_tokens
from simulator.rl.v4.simulator_teacher import cycle_teacher_label, defense_teacher_label


_MAX_ID = max(int(metadata["id"]) for metadata in CARD_METADATA.values())


def _entity(name: str, *, x: float = 4.5, y: float = 13.5, side: int = 1,
            hp: float = 1, visible: bool = True, targetable: bool = True) -> np.ndarray:
    metadata = CARD_METADATA[name]
    row = np.zeros(32, dtype=np.float32)
    row[:11] = (int(metadata["id"]) / _MAX_ID, side, x / 18, y / 32, hp,
                metadata["is_air"], metadata["kind"] == "building", False,
                metadata["kind"] == "spell", visible, targetable)
    return row


def _inputs(*, hand: tuple[str, ...] = ("cannon", "musketeer", "ice-golem", "skeletons"),
            elixir: float = 8, entities: tuple[np.ndarray, ...] = (),
            legal_wait: bool = True) -> dict[str, object]:
    tokens = np.zeros((128, 32), dtype=np.float32)
    mask = np.zeros(128, dtype=bool)
    for index, entity in enumerate(entities):
        tokens[index], mask[index] = entity, True
    hand_tokens = build_hand_tokens(hand, elixir)
    legal = np.zeros((4, 32, 18), dtype=bool)
    for slot, name in enumerate(hand):
        if hand_tokens[slot, 3] > .5:
            legal[slot, 0 if name in ("log", "fireball") else 17:] = True
    global_features = np.zeros(768, dtype=np.float32)
    global_features[0] = elixir / 10
    return dict(entities=tokens, entity_mask=mask, hand_tokens=hand_tokens,
                legal_play=legal, legal_wait=legal_wait, global_features=global_features)


def _keys(branches: list[BranchSpec]) -> set[tuple[int, int, int, int]]:
    return {action_key(branch) for branch in branches}


def test_recorded_action_is_first_exact_and_wait_is_legal_and_distinct() -> None:
    inputs = _inputs(entities=(_entity("x-bow"),))
    recorded = BranchSpec("original", 1, 2, 25, 13)
    branches = propose_correction_branches(inputs, recorded)
    assert branches[0] == BranchSpec("sim", 1, 2, 25, 13)
    assert (0, 0, 0, 0) in _keys(branches)
    assert len(branches) == len(_keys(branches))
    assert propose_correction_branches(inputs, {"mode": 0})[0] == BranchSpec("sim", 0)
    no_wait = _inputs(legal_wait=False, entities=(_entity("x-bow"),))
    assert all(branch.mode == 1 for branch in propose_correction_branches(no_wait, recorded))


def test_current_teachers_remain_candidates_when_distinct_and_legal() -> None:
    inputs = _inputs(hand=("hog-rider", "cannon", "musketeer", "log"),
                     entities=(_entity("x-bow"),))
    branches = propose_correction_branches(inputs, {"mode": 0})
    kwargs = dict(hand_tokens=inputs["hand_tokens"], entity_tokens=inputs["entities"],
                  entity_mask=inputs["entity_mask"], legal_play=inputs["legal_play"],
                  legal_wait=True, own_elixir=8)
    for labeler in (cycle_teacher_label, defense_teacher_label):
        target = labeler(**kwargs)
        assert (target.mode, target.card_slot, target.row, target.col) in _keys(branches)


@pytest.mark.parametrize("limit", (2, 8, 10, 12))
def test_proposals_are_deterministic_bounded_legal_and_do_not_mutate_inputs(limit: int) -> None:
    inputs = _inputs(entities=(_entity("x-bow"), _entity("balloon", x=13.5, y=19.5)))
    original = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in inputs.items()}
    first = propose_correction_branches(inputs, {"mode": 0}, max_branches=limit)
    assert first == propose_correction_branches(inputs, {"mode": 0}, max_branches=limit)
    assert 1 <= len(first) <= limit
    assert len(first) == len(_keys(first))
    for branch in first:
        if branch.mode:
            assert inputs["legal_play"][branch.slot, branch.row, branch.col]
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], original[key])


@pytest.mark.parametrize("limit", (0, 1, -1, 2.0, True))
def test_invalid_bounds_are_rejected(limit: int) -> None:
    with pytest.raises(ValueError, match="max_branches"):
        propose_correction_branches(_inputs(), {"mode": 0}, max_branches=limit)


@pytest.mark.parametrize("recorded", (
    {"mode": 3}, {"mode": 1, "slot": -1},
    {"mode": 1, "slot": 0, "row": 32},
    {"mode": 1, "slot": 0, "row": 3},
))
def test_invalid_factual_action_is_rejected(recorded: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="recorded"):
        propose_correction_branches(_inputs(), recorded)


def test_bridge_siege_adds_cannon_interception_and_ranged_answer_ahead_of_hog_teacher() -> None:
    inputs = _inputs(hand=("hog-rider", "cannon", "musketeer", "ice-golem"),
                     entities=(_entity("x-bow", x=14.5),))
    branches = propose_correction_branches(inputs, {"mode": 0}, max_branches=8)
    geometry = [branch for branch in branches if branch.source == "siege-geometry"]
    assert {branch.slot for branch in geometry} >= {1, 2}
    for branch in geometry:
        assert branch.col >= 9
        if branch.slot in (1, 2):
            name = "cannon" if branch.slot == 1 else "musketeer"
            assert np.hypot(branch.col + .5 - 14.5, branch.row + .5 - 13.5) <= CARD_METADATA[name]["range"]
    assert candidate_categories(inputs, {"mode": 0}) == ("siege",)


def test_lanes_mirror_for_siege_geometric_interception() -> None:
    proposals = [propose_correction_branches(_inputs(entities=(_entity("x-bow", x=x),)), {"mode": 0})
                 for x in (4.5, 13.5)]
    cannon = [next(branch for branch in branches if branch.slot == 0 and branch.mode == 1
                   and branch.source == "siege-geometry") for branches in proposals]
    assert cannon[0].row == cannon[1].row
    assert cannon[0].col == 17 - cannon[1].col


def test_princess_has_useful_log_counter_without_claiming_out_of_range_cannon_damage() -> None:
    inputs = _inputs(hand=("cannon", "log", "musketeer", "ice-golem"),
                     entities=(_entity("princess", y=12.5),))
    inputs["legal_play"][0, :19] = False
    branches = propose_correction_branches(inputs, {"mode": 1, "slot": 0, "row": 22, "col": 12})
    geometry = [branch for branch in branches if branch.source == "ranged-ground-geometry"]
    assert any(branch.slot == 1 for branch in geometry)
    assert not any(branch.slot == 0 for branch in geometry)
    # The factual Cannon remains a hypothesis: tanking/cycle value needs the
    # evaluator and is never removed by this placement proposal module.
    assert branches[0] == BranchSpec("sim", 1, 0, 22, 12)
    assert "ranged-ground" in candidate_categories(inputs, {"mode": 0})


def test_cannon_geometry_remains_available_when_it_can_reach_ground_target() -> None:
    inputs = _inputs(hand=("cannon", "log", "musketeer", "ice-golem"),
                     entities=(_entity("princess", y=20.5),))
    assert any(branch.slot == 0 and branch.mode == 1 and branch.source == "ranged-ground-geometry"
               for branch in propose_correction_branches(inputs, {"mode": 0}))


def test_air_answers_and_stalls_are_proposed_without_ground_damage_claims() -> None:
    inputs = _inputs(hand=("musketeer", "fireball", "ice-spirit", "cannon"),
                     entities=(_entity("balloon", y=19.5),))
    branches = propose_correction_branches(inputs, {"mode": 0}, max_branches=10)
    assert {branch.slot for branch in branches if branch.source == "air-geometry"} >= {0, 1, 2, 3}
    assert candidate_categories(inputs, {"mode": 0}) == ("air",)


def test_unaffordable_air_answers_are_never_commanded_from_merely_in_hand() -> None:
    inputs = _inputs(hand=("skeletons", "musketeer", "fireball", "ice-golem"), elixir=1.125,
                     entities=(_entity("balloon", y=22.5),))
    recorded = {"mode": 1, "slot": 0, "row": 20, "col": 4}
    branches = propose_correction_branches(inputs, recorded)
    assert branches[0] == BranchSpec("sim", 1, 0, 20, 4)
    assert all(branch.mode == 0 or branch.slot == 0 for branch in branches)
    assert (0, 0, 0, 0) in _keys(branches)
    assert candidate_categories(inputs, recorded) == ("air", "upstream-spend")


def test_real_elixir_overrides_stale_affordability_flags_and_legal_mask() -> None:
    inputs = _inputs(hand=("musketeer", "fireball", "ice-spirit", "cannon"),
                     entities=(_entity("balloon", y=20.5),))
    branches = propose_correction_branches(inputs, {"mode": 0}, own_elixir=1.1)
    assert all(branch.mode == 0 or branch.slot == 2 for branch in branches)


def test_low_elixir_golem_is_retained_even_when_it_is_a_resource_review_candidate() -> None:
    inputs = _inputs(hand=("ice-golem", "cannon", "musketeer", "log"), elixir=2.178,
                     entities=(_entity("x-bow"), _entity("archers", y=21.5)))
    recorded = BranchSpec("actor", 1, 0, 20, 4)
    assert propose_correction_branches(inputs, recorded)[0] == BranchSpec("sim", 1, 0, 20, 4)
    assert "upstream-spend" in candidate_categories(inputs, recorded)


def test_public_metadata_namespace_and_unused_entity_columns_cannot_supply_locks() -> None:
    inputs = _inputs(entities=(_entity("x-bow"), _entity("balloon", x=13.5, y=20.5)))
    baseline = propose_correction_branches(inputs, {"mode": 0})
    inputs["entities"][:, 11:] = np.arange(21, dtype=np.float32)
    inputs.update(enemy_hand=object(), exact_target_ids=object(), combat_timers=object())
    assert propose_correction_branches(inputs, {"mode": 0}) == baseline


@pytest.mark.parametrize("entity", (
    _entity("balloon", visible=False), _entity("balloon", targetable=False),
    _entity("balloon", hp=0), _entity("balloon", side=0),
))
def test_nonvisible_dead_or_friendly_air_is_not_a_candidate_category(entity: np.ndarray) -> None:
    assert candidate_categories(_inputs(entities=(entity,)), {"mode": 0}) == ()


def test_candidate_categories_are_distinct_and_do_not_require_future_events() -> None:
    entities = (_entity("x-bow"), _entity("balloon", y=20.5), _entity("princess", y=18.5))
    inputs = _inputs(elixir=3.5, entities=entities)
    recorded = {"mode": 1, "slot": 2, "row": 20, "col": 4}
    assert candidate_categories(inputs, recorded) == (
        "siege", "air", "ranged-ground", "upstream-spend", "multi-threat-spend")
    assert candidate_categories(_inputs(), {"mode": 1, "slot": 2}) == ()


def test_tight_budget_covers_distinct_threat_families_cards_and_roles() -> None:
    inputs = _inputs(hand=("cannon", "musketeer", "ice-golem", "ice-spirit"), entities=(
        _entity("balloon", x=13.5, y=21.5),
        _entity("princess", x=4.5, y=21.5),
        _entity("x-bow", x=4.5, y=13.5),
    ))
    branches = propose_correction_branches(inputs, {"mode": 0}, max_branches=10)
    geometry = [branch for branch in branches if branch.source.endswith("-geometry")]
    assert {branch.source for branch in geometry} >= {
        "air-geometry", "ranged-ground-geometry", "siege-geometry"}
    assert {branch.slot for branch in geometry} == {0, 1, 2, 3}
    # Cannon's distinct pull hypothesis must survive alongside its ground
    # answer, while Ice Golem tanking and Spirit stalls remain available.
    assert any(branch.slot == 0 and branch.source == "air-geometry" for branch in geometry)
    assert any(branch.slot == 0 and branch.source == "siege-geometry" for branch in geometry)


def test_multiple_same_family_targets_receive_options_before_neighbor_fills() -> None:
    inputs = _inputs(hand=("cannon", "musketeer", "ice-golem", "ice-spirit"), entities=(
        _entity("x-bow", x=4.5, y=13.5),
        _entity("x-bow", x=13.5, y=13.5),
        _entity("x-bow", x=4.5, y=22.5),
        _entity("x-bow", x=13.5, y=22.5),
    ))
    branches = propose_correction_branches(inputs, {"mode": 0}, max_branches=10)
    geometry = [branch for branch in branches if branch.source == "siege-geometry"]
    # Both early interception and advanced answers in each lane survive;
    # several placements around the first sorted X-Bow cannot fill the cap.
    assert {(branch.col < 9, branch.row >= 20) for branch in geometry} == {
        (True, True), (True, False), (False, True), (False, False)}
    assert {branch.slot for branch in geometry} == {0, 1, 2, 3}


def test_adjacent_placements_remain_legal_and_reachable() -> None:
    inputs = _inputs(hand=("cannon", "musketeer", "fireball", "hog-rider"), elixir=3,
                     entities=(_entity("princess", y=20.5),))
    factual = {"mode": 1, "slot": 0, "row": 22, "col": 4}
    geometry = [branch for branch in propose_correction_branches(inputs, factual, max_branches=8)
                if branch.source == "ranged-ground-geometry"]
    assert len(geometry) >= 3
    for branch in geometry:
        assert branch.slot == 0
        assert inputs["legal_play"][branch.slot, branch.row, branch.col]
        assert max(abs(branch.row - 22), abs(branch.col - 4)) <= 1
        assert np.hypot(branch.row + .5 - 20.5, branch.col + .5 - 4.5) <= CARD_METADATA["cannon"]["range"]


def test_mixed_air_and_ground_proposals_preserve_ground_emergency_options() -> None:
    inputs = _inputs(hand=("cannon", "skeletons", "ice-golem", "ice-spirit"), elixir=3,
                     entities=(_entity("balloon", y=20.5), _entity("knight", x=13.5, y=21.5)))
    branches = propose_correction_branches(inputs, {"mode": 0}, max_branches=10)
    assert any(branch.source == "ground-geometry" and branch.slot == 1 for branch in branches)
    assert any(branch.source == "ground-geometry" and branch.slot == 2 for branch in branches)
    assert any(branch.source == "air-geometry" and branch.slot == 0 for branch in branches)
    assert any(branch.source == "air-geometry" and branch.slot == 3 for branch in branches)


def test_quiet_upstream_spend_has_cycle_and_setup_options_without_future_enemies() -> None:
    inputs = _inputs(hand=("cannon", "musketeer", "ice-golem", "ice-spirit"), elixir=3)
    factual = BranchSpec("actor", 1, 0, 18, 4)
    branches = propose_correction_branches(inputs, factual, max_branches=8)
    assert branches[0] == BranchSpec("sim", 1, 0, 18, 4)
    assert (0, 0, 0, 0) in _keys(branches)
    setups = [branch for branch in branches if branch.source == "quiet-setup"]
    assert {branch.slot for branch in setups} == {0, 2, 3}
    assert all(inputs["legal_play"][branch.slot, branch.row, branch.col] for branch in setups)
    assert candidate_categories(inputs, factual) == ()


def test_cannon_princess_flag_selects_actual_placements_without_judging_them() -> None:
    inputs = _inputs(entities=(_entity("princess", y=20.5),))
    for col in (4, 13):  # reachable and remote placements receive the same review flag
        categories = candidate_categories(inputs, {"mode": 1, "slot": 0, "row": 22, "col": col})
        assert "cannon-princess-placement" in categories
    assert "cannon-princess-placement" not in candidate_categories(inputs, {"mode": 0})
    assert "cannon-princess-placement" not in candidate_categories(
        inputs, {"mode": 1, "slot": 2, "row": 22, "col": 4})


def test_multithreat_spend_flag_keeps_simultaneous_ground_request_without_counting_swarm_members() -> None:
    mixed = _inputs(entities=(_entity("balloon", y=20.5), _entity("knight", y=20.5)))
    action = {"mode": 1, "slot": 2, "row": 22, "col": 4}
    assert "multi-threat-spend" in candidate_categories(mixed, action)
    assert "multi-threat-spend" not in candidate_categories(mixed, {"mode": 0})
    swarm = _inputs(entities=(_entity("skeletons", x=4.5, y=20.5),
                              _entity("skeletons", x=5.5, y=20.5)))
    assert "multi-threat-spend" not in candidate_categories(swarm, action)


def test_context_flags_and_neighbor_proposals_ignore_all_nonpublic_entity_columns() -> None:
    inputs = _inputs(entities=(_entity("princess", y=20.5), _entity("balloon", x=13.5, y=20.5)))
    action = {"mode": 1, "slot": 0, "row": 22, "col": 4}
    branches = propose_correction_branches(inputs, action, max_branches=10)
    categories = candidate_categories(inputs, action)
    inputs["entities"][:, 11:] = np.nan
    inputs.update(target_uid=object(), target_lock=object(), future_deployments=object())
    assert propose_correction_branches(inputs, action, max_branches=10) == branches
    assert candidate_categories(inputs, action) == categories
