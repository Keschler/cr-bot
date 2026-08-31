from __future__ import annotations

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
        "action-too-early",
        "ground-threat-response",
        "mode-head-regression",
        "potential-elixir-overcommitment",
        "threat-response",
    }


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
