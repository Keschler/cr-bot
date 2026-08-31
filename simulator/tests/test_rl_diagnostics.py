from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def test_teacher_disagreement_alone_is_not_classified_as_head_regression() -> None:
    from rl.diagnostics import classify_decision

    row = {
        "policy": {
            "executed_action": {"mode": "PLAY", "card_slot": 0, "world_cell": [3, 17]},
            "strategic_teacher_action": {"mode": "WAIT"},
            "actor_teacher_agreement": False,
        },
        "state_before": {"units": [], "players": [{"elixir_milli": 5000}]},
        "target_player": 0,
    }

    assert classify_decision(row) == ["teacher_disagreement"]


def test_action_equal_accepts_serialized_policy_descriptors() -> None:
    from rl.diagnostics import action_equal
    from cr_bot.domain.game_state import Action

    assert action_equal(
        {"mode": "PLAY", "card_slot": 2, "world_cell": [3, 17]},
        Action(kind="Play", card_idx=2, cell=(3, 17)),
    )
    assert action_equal({"mode": "WAIT"}, Action(kind="Wait"))


def test_simulator_play_action_is_not_serialized_as_wait() -> None:
    from actions import PlayCardAction
    from rl.diagnostics import _action_descriptor

    assert _action_descriptor(PlayCardAction(1, 2, (3, 12))) == {
        "mode": "PLAY",
        "card_slot": 2,
        "world_cell": [3, 12],
    }


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_teacher_agreement_compares_actor_not_teacher_executed_action() -> None:
    from rl import ActionBatch, ActionMasks, MaskedAutoregressivePolicy, ModelConfig
    from rl.diagnostics import build_policy_diagnostics

    config = ModelConfig(
        gru_hidden_dim=4,
        card_slots=2,
        placement_rows=2,
        placement_cols=2,
    )
    policy = SimpleNamespace(
        action_head=MaskedAutoregressivePolicy(config.gru_hidden_dim, config)
    )
    logits = policy.action_head(torch.zeros(1, 1, config.gru_hidden_dim))
    masks = ActionMasks(
        mode=torch.ones(1, 1, 2, dtype=torch.bool),
        card=torch.ones(1, 1, 2, dtype=torch.bool),
        placement=torch.ones(1, 1, 2, 2, 2, dtype=torch.bool),
    )
    executed = ActionBatch(
        mode=torch.ones(1, 1, dtype=torch.long),
        card_slot=torch.ones(1, 1, dtype=torch.long),
        placement=torch.zeros(1, 1, 2, dtype=torch.long),
    )
    actor = ActionBatch(
        mode=torch.zeros(1, 1, dtype=torch.long),
        card_slot=torch.zeros(1, 1, dtype=torch.long),
        placement=torch.zeros(1, 1, 2, dtype=torch.long),
    )

    row = build_policy_diagnostics(
        policy,
        SimpleNamespace(logits=logits),
        masks,
        executed,
        teacher_action={"mode": "PLAY", "card_slot": 1, "world_cell": [0, 0]},
        actor_actions=actor,
    )

    assert row["executed_action"] == {
        "mode": "PLAY",
        "card_slot": 1,
        "policy_cell": [0, 0],
        "world_cell": [0, 0],
    }
    assert row["actor_action"] == {"mode": "WAIT"}
    assert row["actor_teacher_agreement"] is False


def test_reference_action_difference_gets_timing_and_threat_context() -> None:
    from rl.diagnostics import classify_decision

    row = {
        "policy": {
            "executed_action": {"mode": "PLAY", "card_slot": 0, "world_cell": [3, 17]},
            "reference_action": {"mode": "WAIT"},
            "actor_teacher_agreement": True,
        },
        "state_before": {
            "units": [
                {
                    "owner": 1,
                    "card_id": "giant",
                    "y_mtile": 17000,
                }
            ],
            "players": [{"elixir_milli": 9000}],
        },
        "target_player": 0,
        "tower_damage_to_opponent": 0,
    }

    assert set(classify_decision(row)) == {
        "ground-threat-response",
        "mode-head-divergence",
        "potential-elixir-overcommitment",
        "threat-response",
        "wait-to-play-divergence",
    }


def test_exact_timing_labels_require_counterfactual_consequence() -> None:
    from rl.diagnose import _categories, _timing_consequence_categories

    good = {"mode": "WAIT"}
    candidate = {"mode": "PLAY", "card_slot": 0, "world_cell": [3, 17]}
    base = _categories(
        {"units": []},
        good_action=good,
        candidate_action=candidate,
        teacher_action=None,
        consequence={"major_consequence": True},
        causal_difference={
            "additional_tower_damage_to_self": 0,
            "additional_tower_damage_to_opponent": 0,
        },
    )

    assert "wait-to-play-divergence" in base
    assert "action-too-early" not in base
    unresolved = _timing_consequence_categories(
        target_player=0,
        good_action=good,
        candidate_action=candidate,
        immediate_difference={},
        good_follow_on=None,
        candidate_follow_on=None,
    )
    assert unresolved == ["timing-consequence-unresolved"]
    harmful = _timing_consequence_categories(
        target_player=0,
        good_action=good,
        candidate_action=candidate,
        immediate_difference={"additional_tower_damage_to_self": 100},
        good_follow_on=None,
        candidate_follow_on=None,
    )
    assert harmful == ["action-too-early", "bad-follow-on-consequence"]


def test_head_quality_requires_signed_counterfactual_tower_swing() -> None:
    from rl.diagnose import (
        _head_consequence_categories,
        _relative_state_difference,
    )

    good = {
        "tower_hp": {
            "player_0": {"left": {"hp": 1545}},
            "player_1": {"left": {"hp": 2982}},
        },
        "units": [],
    }
    candidate = {
        "tower_hp": {
            "player_0": {"left": {"hp": 2229}},
            "player_1": {"left": {"hp": 3017}},
        },
        "units": [],
    }
    difference = _relative_state_difference(good, candidate, target_player=0)

    assert difference["avoided_tower_damage_to_self"] == 684
    assert difference["foregone_tower_damage_to_opponent"] == 35
    assert difference["tower_swing_delta"] == 649
    assert _head_consequence_categories(
        good_action={"mode": "PLAY", "card_slot": 1, "world_cell": [14, 19]},
        candidate_action={"mode": "PLAY", "card_slot": 1, "world_cell": [3, 19]},
        immediate_difference={},
        candidate_follow_on={"state_difference_vs_good": difference},
    ) == ["placement-head-improvement"]
    assert _head_consequence_categories(
        good_action={"mode": "PLAY", "card_slot": 1, "world_cell": [3, 19]},
        candidate_action={"mode": "PLAY", "card_slot": 1, "world_cell": [14, 19]},
        immediate_difference=_relative_state_difference(
            candidate, good, target_player=0
        ),
        candidate_follow_on=None,
    ) == ["placement-head-regression"]


def test_checkpoint_diagnosis_preserves_evaluation_deck_shuffling(monkeypatch) -> None:
    from rl import diagnose
    from rl.evaluation_matrix import OpponentDeckSpec, OpponentStrategySpec

    deck = OpponentDeckSpec(
        deck_id="heldout-random",
        cards=(
            "hog-rider",
            "cannon",
            "musketeer",
            "skeletons",
            "ice-golem",
            "ice-spirit",
            "fireball",
            "log",
        ),
        metadata={"archetype": "random-legal"},
    )
    other_deck = OpponentDeckSpec(
        deck_id="heldout-aggressive",
        cards=deck.cards,
        metadata={"archetype": "aggressive-pressure"},
    )
    strategy = OpponentStrategySpec(
        strategy_id="deterministic-cycle",
        description="test controller",
    )
    config = SimpleNamespace(
        target_player=0,
        max_decisions=1200,
        shuffle_decks=True,
        player_deck=deck.cards,
        opponent_decks=(deck, other_deck),
        strategies=(strategy,),
        seeds=(10002,),
    )
    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        captured["build_kwargs"] = kwargs
        return config

    class FakeComparator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare(self, specs):
            captured["specs"] = specs
            return {"kind": "test"}

    monkeypatch.setattr(diagnose, "build_heldout_matrix_config", fake_build)
    monkeypatch.setattr(diagnose, "ExactStateComparator", FakeComparator)

    report = diagnose.compare_checkpoints(
        "good.pt",
        "candidate.pt",
        archetypes=("aggressive-pressure", "random-legal"),
        strategies=("deterministic-cycle",),
        seeds=(10002,),
        shuffle_decks=True,
        only_archetypes=("random-legal",),
    )

    assert captured["build_kwargs"]["shuffle_decks"] is True
    assert captured["specs"][0].shuffle_decks is True
    assert len(captured["specs"]) == 1
    assert captured["specs"][0].opponent_deck.deck_id == "heldout-random"
    assert report["matrix"]["shuffle_decks"] is True
    assert report["matrix"]["diagnosed_archetypes"] == ["random-legal"]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_ppo_ratio_diagnostics_reports_objective_clipping() -> None:
    from rl.diagnostics import ppo_ratio_diagnostics

    old = torch.zeros(1, 3)
    new = torch.tensor([[0.3, -0.3, 0.1]])
    advantages = torch.tensor([[1.0, -1.0, 1.0]])
    ratios, clipped = ppo_ratio_diagnostics(old, new, advantages, 0.2)

    torch.testing.assert_close(ratios, new.exp())
    assert clipped.tolist() == [[True, True, False]]


def test_action_distribution_reports_update_delta() -> None:
    from rl.prototype import _trace_action_distribution, _trace_action_distribution_delta

    first = [
        {"actor_action": {"mode": "WAIT"}, "hand_before": ["hog-rider"]},
        {
            "actor_action": {"mode": "PLAY", "card_slot": 0, "world_cell": [3, 17]},
            "hand_before": ["hog-rider"],
        },
    ]
    second = [
        {
            "actor_action": {"mode": "PLAY", "card_slot": 0, "world_cell": [3, 17]},
            "hand_before": ["hog-rider"],
        },
        {
            "actor_action": {"mode": "PLAY", "card_slot": 0, "world_cell": [3, 17]},
            "hand_before": ["hog-rider"],
        },
    ]
    first_distribution = _trace_action_distribution(first, "actor_action")
    second_distribution = _trace_action_distribution(second, "actor_action")

    assert first_distribution["mode_counts"] == {"PLAY": 1, "WAIT": 1}
    assert second_distribution["mode_counts"] == {"PLAY": 2}
    delta = _trace_action_distribution_delta(first_distribution, second_distribution)
    assert delta is not None
    assert delta["mode_probability_delta"]["PLAY"] == pytest.approx(0.5)
    assert delta["mode_probability_delta"]["WAIT"] == pytest.approx(-0.5)
