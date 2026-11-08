"""Native-cadence short defenses use only fresh, executable public facts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cr_bot.domain.card_metadata import CARD_METADATA
from simulator.observation_v2 import ENTITY_TOKEN_DIM, ENTITY_TOKEN_MAX
from simulator.rl.v4.hand_features import build_hand_tokens
from simulator.rl.v4.simulator_teacher import TeacherTarget, cycle_teacher_label
from simulator.rl.v4.tactical_plans import (
    DECISION_INTERVAL_US, DefensePlanController, PlanPublicState, PlanStatus,
    PlanTeacherLabeler, choose, propose_defense_plans,
)


_MAX_ID = max(int(row["id"]) for row in CARD_METADATA.values())
_WAIT = TeacherTarget(0, 0, 0, 0, 0)
_HAND = ("cannon", "musketeer", "ice-spirit", "skeletons")


def _entity(name: str, *, side: int = 1, x: float = .25, y: float = .55,
            hp: float = 1, **flags: bool) -> np.ndarray:
    metadata = CARD_METADATA[name]
    row = np.zeros(ENTITY_TOKEN_DIM, dtype=np.float32)
    row[:11] = (
        int(metadata["id"]) / _MAX_ID, side, x, y, hp,
        metadata["is_air"], metadata["kind"] == "building", False,
        metadata["kind"] == "spell", True, True,
    )
    columns = dict(is_air=5, is_building=6, is_tower=7, is_spell=8,
                   visible=9, targetable=10)
    for name, value in flags.items():
        row[columns[name]] = value
    return row


def _state(step: int = 0, *, hand: tuple[str, ...] = _HAND, elixir: float = 7,
           entities: tuple[np.ndarray, ...] | None = None,
           legal: np.ndarray | None = None, legal_wait: bool = True,
           tokens: np.ndarray | None = None) -> PlanPublicState:
    bodies = (_entity("giant"),) if entities is None else entities
    entity_tokens = np.zeros((ENTITY_TOKEN_MAX, ENTITY_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(ENTITY_TOKEN_MAX, dtype=bool)
    for index, body in enumerate(bodies):
        entity_tokens[index] = body
        mask[index] = True
    return PlanPublicState(
        step, build_hand_tokens(hand, elixir) if tokens is None else tokens,
        entity_tokens, mask, np.ones((4, 32, 18), dtype=bool) if legal is None else legal,
        legal_wait, elixir,
    )


def _kwargs(state: PlanPublicState) -> dict[str, object]:
    return dict(hand_tokens=state.hand_tokens, entity_tokens=state.entity_tokens,
                entity_mask=state.entity_mask, legal_play=state.legal_play,
                legal_wait=state.legal_wait, own_elixir=state.own_elixir)


def _controller(state: PlanPublicState) -> DefensePlanController:
    proposal = choose(state)
    assert proposal is not None
    return DefensePlanController(proposal)


def _setup(*, name: str = "giant", y: float = .55) -> tuple[np.ndarray, ...]:
    return (_entity(name, y=y), _entity("cannon", side=0, x=7.5 / 18, y=20.5 / 32))


def test_snapshot_copies_public_inputs_and_ignores_unselected_mapping_fields() -> None:
    state = _state()
    inputs = dict(_kwargs(state), entities=state.entity_tokens, enemy_hand=object(),
                  target_lock=object(), combat_timers=object())
    copied = PlanPublicState.from_inputs(inputs, 0, 7)
    assert copied.decision_interval_us == 250_000
    for name in ("hand_tokens", "entity_tokens", "entity_mask", "legal_play"):
        assert not getattr(copied, name).flags.writeable
        assert not np.shares_memory(getattr(copied, name), getattr(state, name))
    assert choose(copied) == choose(state)


@pytest.mark.parametrize("changes", [
    {"decision_interval_us": 100_000}, {"decision_step": -1}, {"decision_step": .0},
    {"own_elixir": float("nan")}, {"own_elixir": 11},
    {"hand_tokens": np.zeros((4, 15))},
    {"entity_tokens": np.full((128, 32), float("nan"))},
])
def test_invalid_observation_contract_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_state(), **changes)


def test_public_entity_id_and_learner_hand_id_are_separate_abis() -> None:
    state = _state(entities=(_entity("balloon"),))
    proposals = propose_defense_plans(state)
    assert [p.plan_id for p in proposals] == ["air-pull-intercept", "air-intercept-stall"]
    assert proposals[0].threat.card_id == CARD_METADATA["balloon"]["id"]
    assert round(float(state.hand_tokens[0, 0]) * 127) == 114
    assert round(float(state.entity_tokens[0, 0]) * _MAX_ID) == CARD_METADATA["balloon"]["id"]


@pytest.mark.parametrize("flag,value", [
    ("visible", False), ("targetable", False), ("is_tower", True),
    ("is_spell", True), ("is_air", False), ("is_building", True),
])
def test_nonlive_or_conflicting_air_flags_do_not_propose(flag: str, value: bool) -> None:
    assert choose(_state(entities=(_entity("balloon", **{flag: value}),))) is None


def test_masked_dead_unknown_and_spell_entities_do_not_propose() -> None:
    assert choose(_state(entities=(_entity("balloon", hp=0),))) is None
    assert choose(_state(entities=(_entity("fireball", is_spell=False, is_air=True),))) is None
    state = _state(entities=(_entity("balloon"),))
    assert choose(replace(state, entity_mask=np.zeros(128, dtype=bool))) is None
    tokens = np.array(state.entity_tokens)
    tokens[0, 0] = 999
    assert choose(replace(state, entity_tokens=tokens)) is None


def test_hidden_targets_state_and_timing_columns_cannot_change_labels() -> None:
    state = _state()
    tokens = np.array(state.entity_tokens)
    tokens[0, 11:] = np.linspace(-20, 20, 21)
    changed = replace(state, entity_tokens=tokens)
    assert choose(state) == choose(changed)
    assert _controller(state).step(state).target == _controller(changed).step(changed).target


@pytest.mark.parametrize("x,col", [(.25, 7), (.75, 10)])
def test_cannon_anchor_mirrors_between_lanes(x: float, col: int) -> None:
    state = _state(entities=(_entity("giant", x=x),))
    assert _controller(state).step(state).target == TeacherTarget(1, 0, 20, col, 0)


def test_cheap_isolated_hog_requires_only_cannon_budget_and_no_extra_support() -> None:
    state = _state(hand=("cannon", "hog-rider", "ice-golem", "log"), elixir=3,
                   entities=(_entity("hog-rider"),))
    controller = _controller(state)
    decision = controller.step(state)
    assert decision.target == TeacherTarget(1, 0, 20, 7, 0)
    controller.acknowledge(decision)
    done = controller.step(_state(1, elixir=0, entities=(_entity("hog-rider"),)))
    assert (done.status, done.reason) == (PlanStatus.COMPLETE, "support_not_needed")
    assert controller.counters.accepted_plays == 1


def test_support_setup_requires_known_hand_budget_flags_and_nearby_legal_cells() -> None:
    assert choose(_state(elixir=6.99)) is None
    assert choose(_state(hand=("cannon", "hog-rider", "ice-golem", "log"))) is None
    for slot in (0, 1):
        for column in (3, 4):
            tokens = build_hand_tokens(_HAND, 7)
            tokens[slot, column] = 0
            assert choose(_state(tokens=tokens)) is None
        legal = np.ones((4, 32, 18), dtype=bool)
        legal[slot] = False
        legal[slot, 31, 17] = True
        assert choose(_state(legal=legal)) is None
    stale = build_hand_tokens(_HAND, 10)
    assert choose(_state(tokens=stale, elixir=6)) is None


def test_cannon_then_musketeer_waits_exactly_one_second_and_needs_real_setup() -> None:
    state = _state()
    controller = _controller(state)
    first = controller.step(state)
    controller.acknowledge(first)
    for step in range(1, 4):
        decision = controller.step(_state(step, elixir=4, entities=_setup()))
        assert (decision.status, decision.reason, decision.target) == (PlanStatus.WAIT, "minimum_delay", _WAIT)
    followup = controller.step(_state(4, elixir=4, entities=_setup()))
    assert followup.status == PlanStatus.PLAY
    assert followup.target == TeacherTarget(1, 1, 20, 7, 0)
    controller.acknowledge(followup)
    assert not controller.active
    assert controller.counters.accepted_plays == 2
    assert controller.counters.waits == 3
    assert controller.step(_state(5)).reason == "plan_finished"


def test_refill_and_fresh_hand_are_checked_before_followup() -> None:
    controller = _controller(_state())
    controller.acknowledge(controller.step(_state()))
    stale = build_hand_tokens(_HAND, 10)
    for step in range(1, 6):
        decision = controller.step(_state(step, tokens=stale, elixir=3.99, entities=_setup()))
        assert decision.status == PlanStatus.WAIT
    # Refill alone is insufficient if a fresh public hand has no Musketeer.
    decision = controller.step(_state(6, hand=("hog-rider", "log", "ice-golem", "skeletons"),
                                      elixir=4, entities=_setup()))
    assert decision.reason == "awaiting_legal_card"
    decision = controller.step(_state(7, hand=("ice-golem", "log", "musketeer", "skeletons"),
                                      elixir=4, entities=_setup()))
    assert decision.target is not None and decision.target.card_slot == 2


def test_air_musketeer_can_spend_four_but_spirit_requires_new_public_elixir() -> None:
    initial = _state(elixir=4, entities=(_entity("balloon", y=.60),))
    controller = _controller(initial)
    assert controller.proposal.plan_id == "air-intercept-stall"
    first = controller.step(initial)
    assert first.target is not None and first.target.card_slot == 1
    controller.acknowledge(first)
    bodies = (_entity("balloon", y=.62), _entity("musketeer", side=0, y=.70))
    for step in range(1, 5):
        decision = controller.step(_state(step, elixir=0, entities=bodies))
        assert decision.status == PlanStatus.WAIT
    spirit = controller.step(_state(5, elixir=1, entities=bodies))
    assert spirit.target == TeacherTarget(1, 2, 20, 4, 0)
    controller.acknowledge(spirit)
    assert controller.counters.accepted_plays == 2


def test_no_speculative_cycle_or_fireball_followup() -> None:
    state = _state(hand=("cannon", "hog-rider", "ice-golem", "log"),
                   entities=(_entity("balloon"),), elixir=10)
    assert propose_defense_plans(state) == ()
    initial = _state(hand=("musketeer", "fireball", "hog-rider", "log"),
                     entities=(_entity("balloon", y=.62),), elixir=8)
    controller = _controller(initial)
    controller.acknowledge(controller.step(initial))
    bodies = (_entity("balloon", y=.65), _entity("musketeer", side=0, y=.70))
    for step in range(1, 11):
        assert controller.step(_state(step, hand=("cannon", "fireball", "hog-rider", "log"),
                                      elixir=4, entities=bodies)).status == PlanStatus.WAIT
    assert controller.step(_state(11, entities=bodies)).reason == "step_expired"
    assert controller.counters.accepted_plays == 1


@pytest.mark.parametrize("bodies,reason", [
    ((), "threat_gone"),
    ((_entity("giant", hp=.1),), "support_not_needed"),
    ((_entity("giant"), _entity("musketeer", side=0, y=.65)), "support_not_needed"),
])
def test_resolved_threat_completes_without_extra_spending(bodies: tuple[np.ndarray, ...], reason: str) -> None:
    controller = _controller(_state())
    controller.acknowledge(controller.step(_state()))
    decision = controller.step(_state(1, entities=bodies))
    assert (decision.status, decision.reason, decision.target) == (PlanStatus.COMPLETE, reason, None)


def test_distant_healthy_air_threat_does_not_receive_unneeded_spirit() -> None:
    initial = _state(elixir=4, entities=(_entity("lava-hound", y=.50),))
    controller = _controller(initial)
    controller.acknowledge(controller.step(initial))
    bodies = (_entity("lava-hound", y=.51), _entity("musketeer", side=0, y=.64))
    for step in range(1, 4):
        assert controller.step(_state(step, entities=bodies)).status == PlanStatus.WAIT
    decision = controller.step(_state(4, entities=bodies))
    assert (decision.status, decision.reason) == (PlanStatus.COMPLETE, "stall_not_needed")


@pytest.mark.parametrize("bodies,reason", [
    ((_entity("giant", y=.70),), "urgent_threat"),
    ((_entity("giant"), _entity("hog-rider", x=.8, y=.7)), "urgent_threat"),
    ((_entity("hog-rider", x=.8, y=.7),), "urgent_threat"),
    ((_entity("giant"), _entity("hog-rider", x=.8, y=.5)), "new_threat"),
    ((_entity("giant"), _entity("giant", x=.26)), "ambiguous_threat"),
])
def test_changed_or_urgent_threat_interrupts_before_spending(bodies: tuple[np.ndarray, ...], reason: str) -> None:
    controller = _controller(_state())
    controller.acknowledge(controller.step(_state()))
    decision = controller.step(_state(1, entities=bodies))
    assert (decision.status, decision.reason, decision.target) == (PlanStatus.INTERRUPTED, reason, None)
    assert controller.counters.accepted_plays == 1


def test_threat_crossing_into_defense_zone_interrupts() -> None:
    initial = _state(entities=(_entity("giant"), _entity("hog-rider", x=.8, y=.44)))
    controller = _controller(initial)
    controller.acknowledge(controller.step(initial))
    changed = _state(1, entities=(_entity("giant"), _entity("hog-rider", x=.8, y=.46)))
    assert controller.step(changed).reason == "new_threat"


def test_proposals_refuse_urgent_ambiguous_and_already_covered_threats() -> None:
    for bodies in (
        (_entity("giant", y=.70),),
        (_entity("giant"), _entity("giant", x=.26)),
        (_entity("giant"), _entity("musketeer", side=0, y=.65)),
        _setup(),
    ):
        assert choose(_state(entities=bodies)) is None


def test_missing_visible_setup_waits_only_until_step_expiry() -> None:
    controller = _controller(_state())
    controller.acknowledge(controller.step(_state()))
    for step in range(1, 13):
        decision = controller.step(_state(step))
        assert decision.status == PlanStatus.WAIT
    decision = controller.step(_state(13))
    assert (decision.status, decision.reason) == (PlanStatus.INTERRUPTED, "step_expired")


def test_observation_gap_and_total_plan_expiry_release_control() -> None:
    initial = _state()
    controller = _controller(initial)
    controller.acknowledge(controller.step(initial))
    assert controller.step(_state(2)).reason == "observation_gap"
    proposal = choose(initial)
    assert proposal is not None
    controller = DefensePlanController(replace(proposal, max_decisions=1))
    controller.acknowledge(controller.step(initial))
    assert controller.step(_state(1)).reason == "plan_expired"


def test_wait_illegality_releases_control() -> None:
    controller = _controller(_state())
    controller.acknowledge(controller.step(_state()))
    assert controller.step(_state(1, legal_wait=False)).reason == "wait_illegal"


def test_same_observation_is_inert_and_unacknowledged_play_never_advances() -> None:
    initial = _state()
    controller = _controller(initial)
    decision = controller.step(initial)
    assert controller.step(initial) is decision
    assert controller.counters.accepted_plays == 0
    assert controller.step(_state(1)).reason == "unacknowledged_play"
    assert controller.counters.accepted_plays == 0
    assert controller.counters.interruptions == 1
    controller.step(_state(2))
    assert controller.counters.interruptions == 1


def test_rejected_execution_and_duplicate_acknowledgement_are_explicit() -> None:
    controller = _controller(_state())
    decision = controller.step(_state())
    controller.acknowledge(decision, accepted=False)
    assert not controller.active
    assert controller.counters.rejected_plays == 1
    assert controller.counters.accepted_plays == 0
    with pytest.raises(ValueError):
        controller.acknowledge(decision)
    controller = _controller(_state())
    decision = controller.step(_state())
    controller.acknowledge(decision)
    with pytest.raises(ValueError):
        controller.acknowledge(decision)


def test_labeler_fallback_matches_cycle_teacher_and_requires_one_execution_report() -> None:
    state = _state(entities=())
    teacher = PlanTeacherLabeler()
    target = teacher.label(decision_step=0, **_kwargs(state))
    assert target == cycle_teacher_label(**_kwargs(state))
    assert teacher.last_source == "cycle"
    teacher.acknowledge(target)
    with pytest.raises(ValueError):
        teacher.acknowledge(target)


@pytest.mark.parametrize("accepted,actual", [(False, None), (True, _WAIT)])
def test_labeler_does_not_progress_nonexecuted_plan(accepted: bool, actual: TeacherTarget | None) -> None:
    teacher = PlanTeacherLabeler()
    first = teacher.label(decision_step=0, **_kwargs(_state()))
    assert teacher.last_source == "plan" and first.mode == 1
    teacher.acknowledge(actual, accepted=accepted)
    assert teacher.counters.accepted_plays == 0
    assert teacher.counters.interruptions == 1
    next_target = teacher.label(decision_step=1, **_kwargs(_state(1)))
    assert teacher.last_decision is not None and teacher.last_decision.step_index == 0
    assert next_target.card_slot == 0


def test_labeler_only_retains_followup_after_matching_accepted_execution() -> None:
    teacher = PlanTeacherLabeler()
    first = teacher.label(decision_step=0, **_kwargs(_state()))
    assert teacher.label(decision_step=0, **_kwargs(_state())) == first
    teacher.acknowledge(first)
    for step in range(1, 4):
        target = teacher.label(decision_step=step, **_kwargs(_state(step, entities=_setup())))
        assert target == _WAIT
        teacher.acknowledge(target)
    target = teacher.label(decision_step=4, **_kwargs(_state(4, entities=_setup())))
    assert target.mode == 1 and target.card_slot == 1
    teacher.acknowledge(target)
    assert teacher.counters.accepted_plays == 2
    teacher.reset()
    assert teacher.counters.accepted_plays == 0
    assert teacher.last_decision is None


def test_missing_execution_report_discards_intent_and_uses_fallback() -> None:
    teacher = PlanTeacherLabeler()
    teacher.label(decision_step=0, **_kwargs(_state()))
    following = _state(1, entities=_setup())
    target = teacher.label(decision_step=1, **_kwargs(following))
    assert teacher.last_source == "cycle"
    assert target == cycle_teacher_label(**_kwargs(following))
    assert teacher.counters.accepted_plays == 0
    assert teacher.counters.interruptions == 1


def test_alternative_during_planned_wait_ends_intent() -> None:
    teacher = PlanTeacherLabeler()
    first = teacher.label(decision_step=0, **_kwargs(_state()))
    teacher.acknowledge(first)
    assert teacher.label(decision_step=1, **_kwargs(_state(1, entities=_setup()))) == _WAIT
    teacher.acknowledge(TeacherTarget(1, 3, 20, 4, 0))
    assert teacher.counters.interruptions == 1
    assert teacher.counters.accepted_plays == 1


def test_plan_cadence_constant_is_native_250ms() -> None:
    assert DECISION_INTERVAL_US == 250_000


@pytest.mark.parametrize("changes", [
    {"max_decisions": 1.5}, {"created_step": -.1},
])
def test_plan_lifetime_uses_integer_observation_steps(changes: dict[str, object]) -> None:
    proposal = choose(_state())
    assert proposal is not None
    with pytest.raises(ValueError):
        DefensePlanController(replace(proposal, **changes))


@pytest.mark.parametrize("changes", [
    {"min_delay_steps": .5}, {"max_delay_steps": 2.5},
    {"reserve_elixir": float("nan")}, {"reserve_elixir": -1},
])
def test_plan_steps_reject_fractional_delays_and_invalid_reserve(changes: dict[str, object]) -> None:
    proposal = choose(_state())
    assert proposal is not None
    step = replace(proposal.steps[0], **changes)
    with pytest.raises(ValueError):
        DefensePlanController(replace(proposal, steps=(step,)))


class _ExecutionEnvironment:
    def __init__(self, *, accepted: bool = True, event_player: int = 0) -> None:
        own = SimpleNamespace(elixir_milli=7000, hand=list(_HAND))

        class OwnOnlyPlayers:
            def __getitem__(self, index):
                assert index == 0, "plan adapter accessed private enemy state"
                return own

        self.state = SimpleNamespace(players=OwnOnlyPlayers(), events=[])
        self.engine = SimpleNamespace(ruleset=SimpleNamespace(tick_us=50_000))
        self.decision_interval_ticks = 5
        self.accepted, self.event_player = accepted, event_player

    def step_v2(self, actions):
        from simulator.actions import PlayCardAction
        from simulator.events import SimEvent
        action = actions[0]
        if isinstance(action, PlayCardAction):
            self.state.events.append(SimEvent.create(
                0, len(self.state.events), "card_played" if self.accepted else "action_rejected",
                player=self.event_player, card_id=_HAND[action.card_slot],
            ))
        return SimpleNamespace(terminated=False, truncated=False)


def _policy_inputs(state: PlanPublicState):
    import torch
    from simulator.rl.v4.model_v4 import masks_from_legal_play
    args = dict(
        hand_tokens=torch.as_tensor(np.array(state.hand_tokens))[None, None],
        entities=torch.as_tensor(np.array(state.entity_tokens))[None, None],
        entity_mask=torch.as_tensor(np.array(state.entity_mask))[None, None],
    )
    masks = masks_from_legal_play(torch.as_tensor(np.array(state.legal_play))[None, None], state.legal_wait)
    return args, masks


@pytest.mark.parametrize("accepted,event_player,accepted_plays", [
    (True, 0, 1), (False, 0, 0), (True, 1, 0),
])
def test_development_plan_arm_waits_for_actual_own_execution(accepted: bool, event_player: int, accepted_plays: int) -> None:
    from simulator.actions import PlayCardAction, WaitAction
    from simulator.rl.v4.policy_development import DevelopmentArm, _PublicTeacherPolicy, _TraceEnvironment
    assert DevelopmentArm("short_plans", "plans").kind == "plans"
    trace = _TraceEnvironment(_ExecutionEnvironment(accepted=accepted, event_player=event_player))
    policy = _PublicTeacherPolicy("plans", [trace])
    action = policy.act_deterministic(*_policy_inputs(_state()))
    assert policy.plan_teacher.counters.accepted_plays == 0
    assert action.mode.tolist() == [[1]]
    assert action.card_slot.tolist() == [[0]]
    row, col = action.placement[0, 0].tolist()
    assert (row, col) == (20, 7)
    # Native actions use (column, row); labels use (row, column).
    trace.step_v2((PlayCardAction(0, 0, (col, row)), WaitAction(1)))
    assert policy.plan_teacher.counters.accepted_plays == accepted_plays
    assert trace.decisions == 1
    assert policy.plan_teacher.counters.interruptions == int(not accepted_plays)
    if accepted_plays:
        followup = policy.act_deterministic(*_policy_inputs(_state(1, entities=_setup())))
        assert followup.mode.tolist() == [[0]]
        assert policy.plan_teacher.last_decision.reason == "minimum_delay"
        trace.step_v2((WaitAction(0), WaitAction(1)))
        assert policy.plan_teacher.counters.accepted_plays == 1
        assert policy.plan_teacher.counters.waits == 1


def test_development_plan_arm_rejects_non_native_environment_cadence() -> None:
    from simulator.rl.v4.policy_development import _PublicTeacherPolicy, _TraceEnvironment
    env = _ExecutionEnvironment()
    env.decision_interval_ticks = 10
    policy = _PublicTeacherPolicy("plans", [_TraceEnvironment(env)])
    with pytest.raises(ValueError, match="250000us"):
        policy.act_deterministic(*_policy_inputs(_state()))


@pytest.mark.parametrize("kind", ["defense", "cycle"])
def test_historical_teacher_arms_keep_same_labels_without_execution_hook(kind: str) -> None:
    from simulator.rl.v4.policy_development import _PublicTeacherPolicy, _TraceEnvironment
    from simulator.rl.v4.simulator_teacher import defense_teacher_label
    trace = _TraceEnvironment(_ExecutionEnvironment())
    policy = _PublicTeacherPolicy(kind, [trace])
    state = _state()
    expected = {"defense": defense_teacher_label, "cycle": cycle_teacher_label}[kind](**_kwargs(state))
    actual = policy.act_deterministic(*_policy_inputs(state))
    assert actual.mode.tolist() == [[expected.mode]]
    assert actual.card_slot.tolist() == [[expected.card_slot]]
    assert actual.placement.tolist() == [[[expected.row, expected.col]]]
    assert trace.on_action_executed is None


def test_development_plan_match_diagnostics_are_recorded_and_summed(monkeypatch) -> None:
    from dataclasses import asdict
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.frontier_ppo import evaluation as native
    from simulator.rl.v4.tactical_plans import PlanCounters, TACTICAL_PLANS_VERSION
    from simulator.tests.test_v4_policy_development import _job, _row

    job = _job(development.training_opponents()[0], arm="plans")
    base = _row(job)

    def fake_record(*, policy, **kwargs):
        native.make_frontier_env()
        return [base["native_record"]]

    monkeypatch.setattr(native, "make_frontier_env", _ExecutionEnvironment)
    monkeypatch.setattr(native, "full_match_record", fake_record)
    monkeypatch.setattr(development, "validate_match", lambda row, job: None)
    row = development.execute_match(job)
    assert row["diagnostics"]["tactical_plans_version"] == TACTICAL_PLANS_VERSION
    assert row["diagnostics"]["tactical_plan_counters"] == asdict(PlanCounters())
    row["diagnostics"]["tactical_plan_counters"]["accepted_plays"] = 2
    assert development.summarize([row, row])["tactical_plan_counters"]["accepted_plays"] == 4


def test_development_plan_match_validation_rejects_missing_or_inconsistent_counters() -> None:
    from dataclasses import asdict
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.tactical_plans import PlanCounters, TACTICAL_PLANS_VERSION
    from simulator.tests.test_v4_policy_development import _job, _row
    job = _job(development.training_opponents()[0], arm="plans")
    row = _row(job)
    with pytest.raises(ValueError, match="tactical-plan"):
        development.validate_match(row, job)
    row["diagnostics"].update(tactical_plans_version=TACTICAL_PLANS_VERSION,
                              tactical_plan_counters=asdict(PlanCounters(accepted_plays=1)))
    development.validate_match(row, job)
    row["diagnostics"]["tactical_plan_counters"]["accepted_plays"] = 2
    with pytest.raises(ValueError, match="tactical-plan"):
        development.validate_match(row, job)


def _frozen_arm(tmp_path, name="frozen_actor"):
    from simulator.rl.v4 import policy_development as development
    checkpoint = tmp_path / f"{name}.pt"
    checkpoint.write_bytes(f"frozen actor fixture {name}".encode())
    return development.DevelopmentArm(
        name, "actor", str(checkpoint), development.file_sha(checkpoint), {"hidden_size": 1},
    )


def _reserved_job(arm, *, spec=None, declared=True):
    from dataclasses import asdict
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    if spec is None:
        spec = catalog_opponents()["eval"][0]
    job = dict(cell_id=f"{spec['deck_id']}__fixed__2026101901", arm=asdict(arm),
               opponent=dict(spec), seed=2026101901)
    if declared:
        job["catalog_split"] = "eval"
    return job


def _reserved_row(job):
    from simulator.tests.test_v4_policy_development import _row
    row = _row(job)
    row["native_record"]["real_deck_split"] = job["opponent"]["real_deck_split"]
    row["native_record"]["deterministic_policy"] = job["arm"]["kind"] != "actor"
    return row


def test_reserved_opponents_require_explicit_eval_split_and_exact_catalog_identity() -> None:
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    catalog = catalog_opponents()
    development.validate_opponents(catalog["eval"], catalog_split="eval")
    development.validate_opponents(catalog["train"])
    with pytest.raises(ValueError, match="pinned catalog"):
        development.validate_opponents(catalog["eval"])
    with pytest.raises(ValueError, match="pinned catalog"):
        development.validate_opponents(catalog["train"], catalog_split="eval")


@pytest.mark.parametrize("mutation", [
    {"style": "tempo"}, {"style_config": {}}, {"style_identity": {}},
    {"style_split": "eval"}, {"extra_execution_field": 1},
    {"real_deck_split": "train"}, {"controller_seed": -1},
    {"deck": ["skeletons"] * 8}, {"strategy": "another-strategy"},
])
def test_reserved_opponents_reject_styles_relabels_and_execution_mutations(mutation) -> None:
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    spec = dict(catalog_opponents()["eval"][0], **mutation)
    with pytest.raises(ValueError):
        development.validate_opponents([spec], catalog_split="eval")


@pytest.mark.parametrize("split", [None, "dev", "reserved", "evaluation"])
def test_unknown_catalog_split_is_rejected(split) -> None:
    from simulator.rl.v4 import policy_development as development
    with pytest.raises(ValueError, match="catalog_split"):
        development.validate_opponents([], catalog_split=split)


def test_exact_reserved_actor_record_passes_validation(tmp_path) -> None:
    from simulator.rl.v4 import policy_development as development
    job = _reserved_job(_frozen_arm(tmp_path))
    development.validate_match(_reserved_row(job), job)


@pytest.mark.parametrize("kind", ["defense", "cycle", "plans"])
def test_reserved_teacher_records_and_direct_execution_are_rejected(kind, monkeypatch) -> None:
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.frontier_ppo import evaluation as native
    job = _reserved_job(development.DevelopmentArm(kind, kind))
    with pytest.raises(ValueError, match="frozen actor"):
        development.validate_match(_reserved_row(job), job)
    monkeypatch.setattr(native, "make_frontier_env", lambda: pytest.fail("reserved teacher started a match"))
    with pytest.raises(ValueError, match="frozen actor"):
        development.execute_match(job)


@pytest.mark.parametrize("mutation", ["undeclared", "relabel", "train-in-eval", "record-split", "style"])
def test_reserved_record_split_cannot_bypass_catalog_verification(tmp_path, mutation) -> None:
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    job = _reserved_job(_frozen_arm(tmp_path))
    if mutation == "undeclared":
        job.pop("catalog_split")
    elif mutation == "relabel":
        job.pop("catalog_split")
        job["opponent"]["real_deck_split"] = "train"
    elif mutation == "train-in-eval":
        job["opponent"] = dict(catalog_opponents()["train"][0])
    elif mutation == "style":
        job["opponent"]["style_config"] = {}
    row = _reserved_row(job)
    if mutation == "record-split":
        row["native_record"]["real_deck_split"] = "train"
    with pytest.raises(ValueError):
        development.validate_match(row, job)


@pytest.mark.parametrize("kind", ["defense", "cycle", "plans"])
def test_reserved_screen_rejects_teacher_arms_before_execution(tmp_path, monkeypatch, kind) -> None:
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    monkeypatch.setattr(development, "runtime_source_fingerprint", lambda: pytest.fail("invalid reserved screen started"))
    with pytest.raises(ValueError, match="frozen actor"):
        development.run_development_screen(
            out=tmp_path, arms=[development.DevelopmentArm(kind, kind)],
            opponents=catalog_opponents()["eval"], seeds=[2026101901], catalog_split="eval",
        )
    assert not (tmp_path / "manifest.json").exists()


def test_reserved_screen_rejects_adaptive_selection_gate(tmp_path, monkeypatch) -> None:
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    monkeypatch.setattr(development, "runtime_source_fingerprint", lambda: pytest.fail("adaptive reserved screen started"))
    with pytest.raises(ValueError, match="no selection gate"):
        development.run_development_screen(
            out=tmp_path, arms=[_frozen_arm(tmp_path)], opponents=catalog_opponents()["eval"],
            seeds=[2026101901], catalog_split="eval", gate={},
        )


def test_reserved_final_screen_records_explicit_split_and_resumes_exact_grid(tmp_path, monkeypatch) -> None:
    import json
    from copy import deepcopy
    from simulator.rl.v4 import policy_development as development
    from simulator.rl.v4.catalog_evaluation import catalog_opponents
    from simulator.tests.test_v4_policy_development import _SynchronousPool
    calls = []

    def execute(job):
        calls.append(deepcopy(job))
        return _reserved_row(job)

    monkeypatch.setattr(development, "execute_match", execute)
    monkeypatch.setattr(development, "ProcessPoolExecutor", _SynchronousPool)
    monkeypatch.setattr(development, "runtime_source_fingerprint", lambda: {"sha256": "reserved-fixture"})
    kwargs = dict(out=tmp_path / "final", arms=[_frozen_arm(tmp_path, "baseline"), _frozen_arm(tmp_path, "eligible")],
                  opponents=catalog_opponents()["eval"], seeds=[2026101901], workers=1, catalog_split="eval")
    first = development.run_development_screen(**kwargs)
    assert len(calls) == first["raw_records"] == 20
    assert all(job["catalog_split"] == "eval" for job in calls)
    assert first["arms"]["baseline"]["games"] == first["arms"]["eligible"]["games"] == 10
    assert first["comparisons"]["eligible"]["pairs"] == 10
    assert first["eligibility"] == {}
    assert not first["selected_actor_changed"]
    manifest = json.loads((kwargs["out"] / "manifest.json").read_text())
    assert manifest["identity"]["catalog_split"] == "eval"
    second = development.run_development_screen(**kwargs)
    assert len(calls) == 20
    assert second["raw_digest_sha256"] == first["raw_digest_sha256"]
