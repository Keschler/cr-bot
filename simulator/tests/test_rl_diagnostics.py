from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def test_teacher_disagreement_alone_is_not_classified_as_head_regression() -> None:
    from rl.foundation.diagnostics import classify_decision

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
    from rl.foundation.diagnostics import action_equal
    from cr_bot.domain.game_state import Action

    assert action_equal(
        {"mode": "PLAY", "card_slot": 2, "world_cell": [3, 17]},
        Action(kind="Play", card_idx=2, cell=(3, 17)),
    )
    assert action_equal({"mode": "WAIT"}, Action(kind="Wait"))


def test_simulator_play_action_is_not_serialized_as_wait() -> None:
    from actions import PlayCardAction
    from rl.foundation.diagnostics import _action_descriptor

    assert _action_descriptor(PlayCardAction(1, 2, (3, 12))) == {
        "mode": "PLAY",
        "card_slot": 2,
        "world_cell": [3, 12],
    }


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_teacher_agreement_compares_actor_not_teacher_executed_action() -> None:
    from rl import ActionBatch, ActionMasks, MaskedAutoregressivePolicy, ModelConfig
    from rl.foundation.diagnostics import build_policy_diagnostics

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
    from rl.foundation.diagnostics import classify_decision

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


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_ppo_ratio_diagnostics_reports_objective_clipping() -> None:
    from rl.foundation.diagnostics import ppo_ratio_diagnostics

    old = torch.zeros(1, 3)
    new = torch.tensor([[0.3, -0.3, 0.1]])
    advantages = torch.tensor([[1.0, -1.0, 1.0]])
    ratios, clipped = ppo_ratio_diagnostics(old, new, advantages, 0.2)

    torch.testing.assert_close(ratios, new.exp())
    assert clipped.tolist() == [[True, True, False]]
