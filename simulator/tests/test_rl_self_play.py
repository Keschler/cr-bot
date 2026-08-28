from __future__ import annotations

import pytest

from cr_bot.domain.game_state import Action as PolicyAction
from rl.evaluation_matrix import EvaluationMatrixError, _controller_action
from rl.self_play import (
    SelfPlayConfigurationError,
    build_self_play_matrix_config,
    build_side_balanced_self_play_matrix_configs,
    checkpoint_strategy,
)


def test_self_play_matrix_uses_the_fixed_prototype_deck_and_frozen_strategies() -> None:
    config = build_self_play_matrix_config(
        "current.pt",
        ("old-a.pt", "old-b.pt"),
        seeds=(10, 11),
    )

    assert config.held_out is False
    assert config.match_count == 4
    assert config.opponent_decks[0].deck_id == "fixed-player-deck"
    assert [strategy.strategy_id for strategy in config.strategies] == [
        "checkpoint-0",
        "checkpoint-1",
    ]
    assert all("frozen public actor checkpoint" in strategy.description for strategy in config.strategies)
    assert config.strategies[0].as_dict()["metadata"]["checkpoint_path"] == "old-a.pt"
    assert config.strategies[1].as_dict()["metadata"]["checkpoint_path"] == "old-b.pt"


def test_self_play_matrix_accepts_a_configured_player_deck() -> None:
    from simulator.rl.opponent_pool import OpponentPool
    from simulator.ruleset import load_fixed_ruleset

    custom_deck = tuple(
        OpponentPool(load_fixed_ruleset(), seed=73)
        .sample_deck(73, archetype="beatdown")
        .cards
    )
    config = build_self_play_matrix_config(
        "current.pt",
        ("old.pt",),
        player_deck=custom_deck,
    )

    assert config.player_deck == custom_deck
    assert config.opponent_decks[0].cards == custom_deck


def test_self_play_matrix_rejects_a_symlinked_current_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_bytes(b"checkpoint")
    alias = tmp_path / "alias.pt"
    alias.symlink_to(checkpoint)

    with pytest.raises(SelfPlayConfigurationError, match="same artifact"):
        build_self_play_matrix_config(checkpoint, (alias,))


def test_side_balanced_self_play_configs_expose_both_target_players() -> None:
    configs = build_side_balanced_self_play_matrix_configs(
        "current.pt",
        ("old.pt",),
        seeds=(10, 11),
    )

    assert len(configs) == 2
    assert [config.target_player for config in configs] == [0, 1]
    assert [config.as_dict()["target_player"] for config in configs] == [0, 1]
    assert configs[0].seeds == configs[1].seeds == (10, 11)
    assert configs[0].player_deck == configs[1].player_deck
    assert configs[0].opponent_decks == configs[1].opponent_decks
    assert configs[0].match_count == configs[1].match_count == 2


def test_public_checkpoint_controller_seam_receives_public_observation() -> None:
    class PublicController:
        def choose_public_action(self, observation: object, *, player: int) -> PolicyAction:
            assert observation == "public-v2"
            assert player == 1
            return PolicyAction(kind="Wait")

    action = _controller_action(
        PublicController(),
        engine=object(),
        state=object(),
        player=1,
        public_observation="public-v2",
    )

    assert action.kind == "Wait"


def test_public_checkpoint_controller_requires_public_observation() -> None:
    class PublicController:
        def choose_public_action(self, observation: object, *, player: int) -> PolicyAction:
            return PolicyAction(kind="Wait")

    with pytest.raises(EvaluationMatrixError, match="V2 observation"):
        _controller_action(PublicController(), object(), object(), 1)


def test_checkpoint_strategy_rejects_empty_strategy_id() -> None:
    with pytest.raises(SelfPlayConfigurationError, match="strategy_id"):
        checkpoint_strategy("old.pt", strategy_id="")
