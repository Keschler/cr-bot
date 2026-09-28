"""Focused contract tests for the visual V4 deployment adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from simulator.physical_lab.sensing.policy_bridge import (
    VisualPolicyObservation,
    observation_v2_from_game_state,
)
from simulator.physical_lab.workflow.v4_actor import V4Actor, V4ActorError
from simulator.rl.v4.frontier import frontier_model_config
from simulator.rl.v4.model_v4 import RecurrentV4Policy

from cr_bot.domain.game_state import (
    Detection,
    GameState,
    HudState,
    Match,
    PrincessTowerState,
)


def _checkpoint(path):
    config = frontier_model_config()
    policy = RecurrentV4Policy(config)
    with torch.no_grad():
        policy(
            torch.zeros(1, 1, 21, 32, 18),
            torch.zeros(1, 1, 768),
            torch.zeros(1, 1, 128, 32),
            torch.zeros(1, 1, 128, dtype=torch.bool),
            torch.zeros(1, 1, 4, 16),
            torch.full((1, 1, 128), 1.0 / 128),
            torch.zeros(1, 1, 128, dtype=torch.bool),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 16, 8),
            torch.ones(1, 1, dtype=torch.bool),
        )
    torch.save(policy.state_dict(), path)


def _visual_observation() -> VisualPolicyObservation:
    state = GameState(
        hud=HudState(
            time_left_s=120.0,
            overtime=False,
            elixir_self=10.0,
            hand_cards=["hog-rider", "cannon", "musketeer", "skeletons"],
            next_card="ice-golem",
            tower_hp_self=[3_050, 6_500, 3_050],
            tower_hp_enemy=[3_050, 6_500, 3_050],
            princess_towers=PrincessTowerState(
                own_left_alive=True,
                own_right_alive=True,
                enemy_left_alive=True,
                enemy_right_alive=True,
            ),
        ),
        total_remaining_s=120.0,
        own_units=[],
        enemy_units=[],
        seen_enemy_cards=[],
        elixir_enemy_est=5.0,
        own_king_active=False,
        enemy_king_active=False,
        started=True,
    )
    return observation_v2_from_game_state(
        state,
        arena_px=(0.0, 0.0, 1.0, 1.0),
    )


def _observation_with_enemy(*, match_time_s: float, track_id: int | None) -> VisualPolicyObservation:
    observation = _visual_observation()
    state = observation.game_state
    state.total_remaining_s = 300.0 - match_time_s
    state.enemy_units = [
        Match(
            troop=Detection(
                track_id=track_id,
                class_name="hog-rider",
                team="enemy",
                confidence=0.95,
                x1=0.45,
                y1=0.45,
                x2=0.55,
                y2=0.55,
                center_x=0.5,
                center_y=0.5,
                estimated_hp=1_000.0,
            ),
            bar=None,
        )
    ]
    return observation_v2_from_game_state(
        state,
        arena_px=(0.0, 0.0, 1.0, 1.0),
    )


def test_v4_actor_maps_public_visual_observation_and_scores_one_legal_action(tmp_path):
    path = tmp_path / "v4_actor.pt"
    _checkpoint(path)
    actor = V4Actor(path, device="cpu")

    observation = _visual_observation()
    assert observation.match_time_s == pytest.approx(180.0)
    action, suggestions, diagnostics = actor.decide_with_scores(observation)

    assert action.kind in {"Wait", "Play"}
    if action.kind == "Play":
        assert observation.legal_play[action.card_idx, action.cell[1], action.cell[0]]
    assert suggestions
    assert set(diagnostics) == {
        "mode_prob_wait",
        "mode_prob_play",
        "entropy",
        "top_log_prob",
    }
    assert all(0.0 <= diagnostics[name] <= 1.0 for name in ("mode_prob_wait", "mode_prob_play"))


def test_v4_actor_advances_public_memory_and_deduplicates_visual_tracks(tmp_path):
    path = tmp_path / "v4_actor.pt"
    _checkpoint(path)
    actor = V4Actor(path, device="cpu")

    actor.decide(_observation_with_enemy(match_time_s=10.0, track_id=7))
    first = actor._memory.snapshot()
    assert first.opp_elixir_hi == pytest.approx(6.0)

    # The detector reconstructs the Match wrapper on the next frame.  The
    # stable public track id must prevent a second spend, while the event-free
    # second of elixir regeneration still advances the interval.
    actor.decide(_observation_with_enemy(match_time_s=11.0, track_id=7))
    second = actor._memory.snapshot()
    regen = 1.0 / 2.8
    assert second.opp_elixir_lo == pytest.approx(regen)
    assert second.opp_elixir_hi == pytest.approx(6.0 + regen)


def test_v4_actor_skips_enemy_spend_without_a_stable_visual_track(tmp_path):
    path = tmp_path / "v4_actor.pt"
    _checkpoint(path)
    actor = V4Actor(path, device="cpu")

    actor.decide(_observation_with_enemy(match_time_s=10.0, track_id=None))
    first = actor._memory.snapshot()
    actor.decide(_observation_with_enemy(match_time_s=11.0, track_id=None))
    second = actor._memory.snapshot()

    # No stable identity means no spend inference; only time advancement is
    # allowed to affect the interval.
    regen_at_ten = 10.0 / 2.8
    assert first.opp_elixir_lo == pytest.approx(regen_at_ten)
    assert first.opp_elixir_hi == pytest.approx(10.0)
    assert second.opp_elixir_lo == pytest.approx(11.0 / 2.8)
    assert second.opp_elixir_hi == pytest.approx(10.0)


def test_v4_actor_rejects_missing_or_incompatible_checkpoint(tmp_path):
    with pytest.raises(V4ActorError, match="does not exist"):
        V4Actor(tmp_path / "missing.pt", device="cpu")

    path = tmp_path / "incompatible.pt"
    torch.save({"unexpected": torch.zeros(1)}, path)
    with pytest.raises(V4ActorError, match="could not load V4 checkpoint"):
        V4Actor(path, device="cpu")
