"""Whole-sequence credit, deadline budgets and actual branch chronology."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rl.v4 import correction_sequences as sequences
from simulator.rl.v4.frontier_eval import BranchSpec
from simulator.rl.v4.hand_features import build_hand_tokens
from simulator.rl.v4.tactical_plans import PlanPublicState, PlanStatus


def public(elixir=2., *, air=True, hand=("musketeer", "ice-spirit", "ice-golem", "log"), y=.50):
    from cr_bot.domain.card_metadata import CARD_METADATA
    maximum = max(row["id"] for row in CARD_METADATA.values())
    entities = np.zeros((128, 32), np.float32)
    mask = np.zeros(128, bool)
    if air:
        entities[0, :11] = [CARD_METADATA["balloon"]["id"] / maximum, 1, .25, y, 1, 1, 0, 0, 0, 1, 1]
        mask[0] = True
    return dict(hand_tokens=build_hand_tokens(hand, elixir), entities=entities, entity_mask=mask,
                legal_play=np.ones((4, 32, 18), bool), legal_wait=True)


def state(step, elixir, **kwargs):
    return PlanPublicState.from_inputs(public(elixir, **kwargs), step, elixir)


def test_controller_waits_until_affordable_and_checks_defense_before_urgency():
    spec = sequences.SavingPlanSpec("musketeer", 4, sequences.SavingPlanConfig(allow_spirit_stall=False))
    controller = sequences.SavingPlanController(spec)
    first = controller.step(state(4, 2))
    assert first.status == PlanStatus.WAIT and first.reason == "initial_wait"
    controller.acknowledge(first)
    waiting = controller.step(state(5, 3.9))
    assert waiting.status == PlanStatus.WAIT
    controller.acknowledge(waiting)
    play = controller.step(state(6, 4., y=.71))
    assert play.status == PlanStatus.PLAY and play.target.card_slot == 0
    controller.acknowledge(play)
    assert controller.completed and controller.defender_plays == 1
    assert controller.step(state(7, .1)).status == PlanStatus.COMPLETE


def test_fireball_saving_uses_a_visible_air_threat_and_unaffordable_in_hand_answer():
    inputs = public(2., hand=("fireball", "ice-spirit", "ice-golem", "log"))
    plans = sequences.propose_saving_plans(inputs, own_elixir=2., decision_step=0)
    assert [p.defender for p in plans] == ["fireball"]
    controller = sequences.SavingPlanController(plans[0])
    first = controller.step(PlanPublicState.from_inputs(inputs, 0, 2))
    controller.acknowledge(first)
    play = controller.step(state(1, 4, hand=("fireball", "ice-spirit", "ice-golem", "log")))
    assert play.reason == "saved_defender"


def test_proposals_distinguish_hand_membership_from_affordability():
    inputs = public(2.)
    assert inputs["hand_tokens"][0, 3] == 0
    assert [plan.defender for plan in sequences.propose_saving_plans(
        inputs, own_elixir=2., decision_step=0)] == ["musketeer"]
    assert not sequences.propose_saving_plans(public(4.), own_elixir=4., decision_step=0)
    assert not sequences.propose_saving_plans(public(2., air=False), own_elixir=2., decision_step=0)
    assert not sequences.propose_saving_plans(public(2., hand=("hog-rider", "ice-spirit", "ice-golem", "log")),
                                              own_elixir=2., decision_step=0)


def test_spirit_stall_uses_deadline_budget_not_blanket_four_elixir_reserve():
    roomy = sequences.SavingPlanController(sequences.SavingPlanSpec("musketeer", 0))
    first = roomy.step(state(0, 3))
    roomy.acknowledge(first)
    stall = roomy.step(state(1, 3))
    assert stall.reason == "budgeted_spirit_stall" and stall.target.card_slot == 1
    roomy.acknowledge(stall)
    assert roomy.step(state(2, 2.1)).status == PlanStatus.WAIT
    tight = sequences.SavingPlanController(sequences.SavingPlanSpec("musketeer", 0,
        sequences.SavingPlanConfig(max_decisions=4, deadline_decisions=3)))
    first = tight.step(state(0, 3))
    tight.acknowledge(first)
    assert tight.step(state(1, 3)).status == PlanStatus.WAIT


def test_actions_need_acknowledgement_and_rejection_never_completes():
    controller = sequences.SavingPlanController(sequences.SavingPlanSpec("musketeer", 0))
    first = controller.step(state(0, 3))
    assert controller.step(state(0, 3)) is first
    assert controller.step(state(1, 3)).reason == "unacknowledged_action"
    controller = sequences.SavingPlanController(sequences.SavingPlanSpec("musketeer", 0))
    controller.acknowledge(controller.step(state(0, 3)))
    play = controller.step(state(1, 4))
    controller.acknowledge(play, accepted=False)
    assert not controller.completed
    assert controller.step(state(2, 4)).reason == "execution_rejected"
    with pytest.raises(ValueError, match="acknowledgement"):
        controller.acknowledge(play)


def test_gaps_expiry_and_urgent_ground_release_control():
    controller = sequences.SavingPlanController(sequences.SavingPlanSpec("musketeer", 0))
    controller.acknowledge(controller.step(state(0, 2)))
    assert controller.step(state(2, 2)).reason == "observation_gap"
    controller = sequences.SavingPlanController(sequences.SavingPlanSpec("musketeer", 0,
        sequences.SavingPlanConfig(max_decisions=3, deadline_decisions=2)))
    for step in range(2):
        controller.acknowledge(controller.step(state(step, 2, air=False)))
    assert controller.step(state(2, 2, air=False)).reason == "defense_deadline"
    inputs = public(4)
    from cr_bot.domain.card_metadata import CARD_METADATA
    maximum = max(row["id"] for row in CARD_METADATA.values())
    inputs["entities"][1, :11] = [CARD_METADATA["knight"]["id"] / maximum, 1, .75, .8, 1, 0, 0, 0, 0, 1, 1]
    inputs["entity_mask"][1] = True
    assert not sequences.propose_saving_plans(inputs, own_elixir=4, decision_step=0)


@pytest.fixture
def certificate(monkeypatch):
    def run(**kwargs):
        value = dict(factual=0., wait=.1, plan=1.)[kwargs["arm"]]
        return dict(value=value, winner=None, terminal=False, truncated=False, decisions_run=kwargs["horizon"],
                    plan_completed=kwargs["arm"] == "plan", defender_plays=int(kwargs["arm"] == "plan"))
    monkeypatch.setattr(sequences, "_run", run)
    result = sequences.evaluate_saving_sequence(env=None, hidden=None, policy=None,
        candidate=dict(t=0, state_hash="origin"), factual=BranchSpec("sim", 1, 2, 18, 4),
        plan=sequences.SavingPlanSpec("musketeer", 0), opponent_spec={}, origin_inputs=public(), own_elixir=2,
        config=SimpleNamespace(discovery_replicas=2, confirmation_replicas=3, discovery_horizon=4,
                               confirmation_horizon=8, seed=7, min_lcb=.05))
    return result


def test_sequence_certificate_preserves_separate_wait_credit(certificate):
    assert certificate["accepted"] and certificate["target"] is None
    assert certificate["target_kind"] == "sequence"
    assert certificate["wait_only_accepted"]
    stats = sequences.validate_sequence_certificate(certificate, min_lcb=.05,
        expected_state_hash="origin", expected_plan_id=certificate["plan_id"])
    assert stats["plan_vs_wait"]["mean"] == pytest.approx(.9)


@pytest.mark.parametrize("mutation", ["seed", "overlap", "plan", "score", "execution", "origin"])
def test_sequence_certificate_fails_closed_on_forged_evidence(certificate, mutation):
    value = deepcopy(certificate)
    if mutation == "seed":
        value["arms"]["wait"]["confirmation_replicas"][0]["replica_seed"] += 1
    elif mutation == "overlap":
        for arm in value["arms"].values():
            arm["confirmation_replicas"][0]["replica_seed"] = arm["discovery_replicas"][0]["replica_seed"]
    elif mutation == "plan":
        value["plan_identity"]["defender"] = "fireball"
    elif mutation == "score":
        value["comparisons"]["plan_vs_factual"]["mean"] += 1
    elif mutation == "execution":
        value["arms"]["plan"]["confirmation_replicas"][0]["plan_completed"] = False
    else:
        value["state_hash"] = "wrong"
    with pytest.raises(ValueError):
        sequences.validate_sequence_certificate(value, min_lcb=.05, expected_state_hash="origin")


def test_plan_only_gain_never_certifies_wait_alone(monkeypatch, certificate):
    original = sequences._run
    def run(**kwargs):
        result = original(**kwargs)
        if kwargs["arm"] == "wait":
            result["value"] = -.3
        return result
    monkeypatch.setattr(sequences, "_run", run)
    config = SimpleNamespace(discovery_replicas=2, confirmation_replicas=3, discovery_horizon=4,
                             confirmation_horizon=8, seed=7, min_lcb=.05)
    result = sequences.evaluate_saving_sequence(env=None, hidden=None, policy=None,
        candidate=dict(t=0, state_hash="origin"), factual=BranchSpec("sim", 1, 2, 18, 4),
        plan=sequences.SavingPlanSpec("musketeer", 0), opponent_spec={}, origin_inputs=public(), own_elixir=2,
        config=config)
    assert result["accepted"] and not result["wait_only_accepted"] and result["target"] is None


def test_real_branch_capture_keeps_prefix_and_advances_memory_once_on_forced_waits(monkeypatch, tmp_path):
    import torch
    from simulator.actions import WaitAction
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.rl.v4.frontier_eval import _fresh_env
    from simulator.rl.v4.tactical_data import PUBLIC_KEYS, active_public_mask
    from simulator.rl.v4.v4_ppo.runtime import observe_v4_inputs
    from simulator.roster import PLAYER_DECK
    env = _fresh_env(dict(config=dict(match_seed=4, learner_deck=PLAYER_DECK, opponent_deck=PLAYER_DECK)))
    estimator = PublicStateEstimator()
    records = []
    for _ in range(17):
        _, inputs, _, elixir = observe_v4_inputs(env, estimator)
        air, siege = active_public_mask(inputs)
        records.append(dict(**{key: np.array(inputs[key], copy=True) for key in PUBLIC_KEYS},
            state_hash=env.state.state_hash(), elapsed_us=env.state.elapsed_us, own_elixir=elixir,
            executed_actions=np.zeros(5, np.int64), opponent_actions=np.zeros(5, np.int64),
            label_valid=bool(inputs["legal_wait"]), air_threat=air, xbow_present=siege))
        if len(records) < 17:
            env.step_v2_for_viewer((WaitAction(0), WaitAction(1)), viewer=0)
    arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    origin_hash = env.state.state_hash()
    class Policy:
        def __init__(self):
            self.calls = []
        def eval(self):
            return self
        def rollout_sample(self, *args, hidden, **kwargs):
            self.calls.append(float(hidden.item()))
            action = SimpleNamespace(mode=torch.zeros((1, 1), dtype=torch.int64),
                card_slot=torch.zeros((1, 1), dtype=torch.int64), placement=torch.zeros((1, 1, 2), dtype=torch.int64))
            return None, action, None, None, None, hidden + 1
    policy = Policy()
    monkeypatch.setattr("simulator.rl.v4.v4_ppo.runtime.configure_v4_runtime", lambda _: None)
    monkeypatch.setattr("simulator.rl.v4.frontier_eval.make_opponent", lambda *_args, **_kwargs:
        SimpleNamespace(choose_action=lambda *_args: WaitAction(1)))
    names = [_HAND[0] for row in arrays["hand_tokens"][-1]
             if (_HAND := sequences._HAND.get(int(round(float(row[0]) * 127))))]
    defender = next((name for name in names if name in ("musketeer", "fireball")), "musketeer")
    captured = sequences.capture_saving_sequence(env=env, hidden=torch.tensor([[[17.]]]), policy=policy,
        candidate=dict(t=16, state_hash=origin_hash, source_episode_sha256="source"), factual=BranchSpec("sim", 0),
        plan=sequences.SavingPlanSpec(defender, 16), opponent_spec={}, replica_seed=100, horizon=4,
        source_arrays=arrays, source_metadata=dict(job_id="train_origin", split="train", family="air"))
    assert policy.calls == [17., 18., 19.]
    assert env.state.state_hash() == origin_hash
    assert len(captured["arrays"]["state_hash"]) == 20
    assert not captured["arrays"]["active"][:16].any()
    for key in PUBLIC_KEYS:
        assert np.array_equal(captured["arrays"][key][:17], arrays[key])
    receipt = sequences.save_sequence_capture(tmp_path / "branch.npz", captured)
    loaded, metadata = sequences.load_sequence_capture(receipt["path"], expected_sha=receipt["sha256"])
    assert metadata["prefix_length"] == 16 and not metadata["plan_completed"]
    assert np.array_equal(loaded["executed_actions"], captured["arrays"]["executed_actions"])
    with pytest.raises(ValueError, match="file hash"):
        sequences.load_sequence_capture(receipt["path"], expected_sha="wrong")
    wrong = dict(arrays, event_history=np.array(arrays["event_history"], copy=True))
    wrong["event_history"][16, 0, 0] += 1
    with pytest.raises(ValueError, match="public origin"):
        sequences.capture_saving_sequence(env=env, hidden=torch.tensor([[[17.]]]), policy=policy,
            candidate=dict(t=16, state_hash=origin_hash), factual=BranchSpec("sim", 0),
            plan=sequences.SavingPlanSpec(defender, 16), opponent_spec={}, replica_seed=100, horizon=4,
            source_arrays=wrong, source_metadata=dict(job_id="train_origin", split="train", family="air"))
