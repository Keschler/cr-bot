from __future__ import annotations

import pytest

from simulator.engine import BattleEngine
from simulator.env import SimulatorEnv
from simulator.rl.basic_scenarios import (
    BASIC_MECHANICS_SOURCES,
    BASIC_SCENARIO_REWARD_VERSION,
    BasicMechanicsScenarioEnv,
    BasicScenarioConfig,
    BasicScenarioError,
    basic_scenario_source,
)
from simulator.rl.opponent_pool import OpponentPool
from simulator.roster import PLAYER_DECK
from simulator.ruleset import load_fixed_ruleset


_ARCHETYPE = {
    "isolated-offense": "aggressive-pressure",
    "ground-defense": "beatdown",
    "air-defense": "air-beatdown",
    "spell-situations": "siege-bait",
    "kiting-cycling-elixir": "defensive-cycle",
}


def _environment(source: str, *, decisions: int = 4) -> BasicMechanicsScenarioEnv:
    ruleset = load_fixed_ruleset()
    opponent = OpponentPool(ruleset, seed=71).sample(
        0,
        archetype=_ARCHETYPE[source],
        strategy="deterministic-cycle",
    )
    base = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        decision_interval_us=250_000,
    )
    wrapped = BasicMechanicsScenarioEnv(
        base,
        BasicScenarioConfig(
            source=source,
            target_player=0,
            decision_limit=decisions,
        ),
    )
    wrapped.reset_v2(
        seed=123,
        decks=(tuple(PLAYER_DECK), opponent.deck.cards),
        shuffle_decks=True,
    )
    return wrapped


@pytest.mark.parametrize("source", BASIC_MECHANICS_SOURCES)
def test_basic_scenario_generates_valid_reproducible_state(source: str) -> None:
    first = _environment(source)
    second = _environment(source)

    assert first.state.state_hash() == second.state.state_hash()
    assert first.scenario_audit() == second.scenario_audit()
    audit = first.scenario_audit()
    latest = audit["latest"]
    assert latest["source"] == source
    assert latest["success_definition"] == "resulting-game-state"
    assert latest["reward_version"] == BASIC_SCENARIO_REWARD_VERSION
    assert latest["target_hand"] == list(first.state.players[0].hand)
    assert 3_000 <= latest["target_elixir_milli"] <= 10_000
    assert all(650 <= value <= 1_000 for value in latest["tower_hp_permille"].values())
    first.engine.validate_state(first.state)


def test_air_defense_contains_an_actual_air_threat() -> None:
    environment = _environment("air-defense")
    audit = environment.scenario_audit()["latest"]

    assert audit["setup_cards"]
    assert any(
        environment.engine.ruleset.card(card_id).mechanics.get("movement_layer") == "air"
        for card_id in audit["setup_cards"]
    )
    assert audit["threat_uids"]


def test_short_scenario_terminates_at_declared_horizon_without_truncation() -> None:
    environment = _environment("ground-defense", decisions=2)

    first = environment.step_v2((None, None))
    second = environment.step_v2((None, None))

    assert first.terminated is False
    assert second.terminated is True
    assert second.truncated is False
    assert second.info["terminal_reason"] == "basic_mechanics_scenario_horizon"
    assert second.info["episode_kind"] == "basic-mechanics-short-scenario"
    assert second.info["scenario_outcome"] in {"win", "draw", "loss"}


def test_reward_depends_on_resulting_state_not_a_card_label() -> None:
    environment = _environment("isolated-offense", decisions=4)
    opponent = 1
    tower = next(
        entity
        for entity in environment.state.entities.values()
        if entity.owner == opponent and entity.kind == "tower" and entity.role != "king"
    )
    tower.hp -= 100

    result = environment.step_v2((None, None))

    assert result.rewards[0] > 0.0
    assert result.rewards[1] == pytest.approx(-result.rewards[0])
    assert "teacher" not in result.info
    assert "correct_card" not in result.info


def test_all_wait_does_not_win_most_generated_setup_scenarios() -> None:
    outcomes = []
    for source in BASIC_MECHANICS_SOURCES:
        environment = _environment(source, decisions=64)
        result = None
        for _decision in range(64):
            result = environment.step_v2((None, None))
            if result.terminated or result.truncated:
                break
        assert result is not None
        outcomes.append(result.info["scenario_outcome"])

    assert outcomes.count("win") <= len(outcomes) // 2


def test_sampling_source_resolves_phase_one_rehearsal_only() -> None:
    assert basic_scenario_source("air-defense", episode_index=9) == "air-defense"
    assert basic_scenario_source("phase-1-rehearsal", episode_index=7) in BASIC_MECHANICS_SOURCES
    assert basic_scenario_source("uniform-archetypes", episode_index=7) is None


def test_invalid_basic_scenario_configuration_fails_closed() -> None:
    with pytest.raises(BasicScenarioError):
        BasicScenarioConfig(source="not-a-scenario", target_player=0)
    with pytest.raises(BasicScenarioError):
        BasicScenarioConfig(source="air-defense", target_player=2)
    with pytest.raises(BasicScenarioError):
        BasicScenarioConfig(source="air-defense", target_player=0, decision_limit=0)
