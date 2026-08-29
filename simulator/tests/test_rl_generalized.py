from __future__ import annotations

from collections import Counter
import hashlib
import json

import pytest

from simulator.engine import BattleEngine
from simulator.env import SimulatorEnv
from simulator.rl.generalized import (
    GeneralizedTrainingConfig,
    GeneralizedTrainingError,
    _curriculum_axes,
    _segment_config,
    _update_payoff_book_from_segment,
    build_heldout_matrix_config,
    make_scenario_opponent_action,
    sample_training_scenarios,
)
from simulator.rl.league import LeaguePayoffBook, LeaguePayoffStats
from simulator.rl.opponent_pool import OpponentPool
from simulator.rl.prototype import PrototypeConfig
from simulator.ruleset import load_fixed_ruleset
from simulator.roster import PLAYER_DECK


def test_training_schedule_keeps_regression_and_varies_scenarios() -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=44)
    first = sample_training_scenarios(
        pool,
        envs=4,
        segment_index=0,
        archetypes=("aggressive-pressure", "beatdown", "siege-bait"),
        strategies=("aggressive-pressure", "beatdown", "siege-bait"),
    )
    second = sample_training_scenarios(
        pool,
        envs=4,
        segment_index=1,
        archetypes=("aggressive-pressure", "beatdown", "siege-bait"),
        strategies=("aggressive-pressure", "beatdown", "siege-bait"),
    )

    assert len(first) == len(second) == 4
    assert first[0].deck.cards == tuple(PLAYER_DECK)
    assert first[0].strategy == "deterministic-cycle"
    assert {scenario.deck.archetype for scenario in first[1:]} <= {
        "aggressive-pressure",
        "beatdown",
        "siege-bait",
    }
    assert [scenario.as_dict() for scenario in first] != [
        scenario.as_dict() for scenario in second
    ]


def test_curriculum_axes_follow_cumulative_decision_budget() -> None:
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=2, horizon=8, updates=1),
        segments=1,
    )

    early = _curriculum_axes(config, 4, decision_count=0)
    scripted = _curriculum_axes(config, 4, decision_count=5_000_000)

    assert early[2] is not None
    assert early[2].stage_id == "mechanics-foundation"
    assert scripted[2] is not None
    assert scripted[2].stage_id == "scripted-threat-expansion"


def test_threat_stratified_schedule_reserves_air_and_ground_lanes() -> None:
    pool = OpponentPool(load_fixed_ruleset(), seed=45)

    for segment_index in (0, 1, 7):
        scenarios = sample_training_scenarios(
            pool,
            envs=4,
            segment_index=segment_index,
            archetypes=(
                "aggressive-pressure",
                "air-beatdown",
                "beatdown",
                "siege-bait",
            ),
            strategies=("aggressive-pressure", "beatdown", "siege-bait"),
            threat_stratified=True,
        )

        assert scenarios[0].deck.cards == tuple(PLAYER_DECK)
        assert [scenario.deck.archetype for scenario in scenarios[1:3]] == [
            "air-beatdown",
            "beatdown",
        ]
        assert scenarios[1].strategy in {"aggressive-pressure", "beatdown", "siege-bait"}
        assert scenarios[2].strategy in {"aggressive-pressure", "beatdown", "siege-bait"}


def test_phase_sampling_mix_is_interleaved_and_reproducible() -> None:
    from simulator.rl.curriculum import default_strategic_curriculum

    pool = OpponentPool(load_fixed_ruleset(), seed=46)
    mix = default_strategic_curriculum().stage_at(0).sampling_mix
    kwargs = {
        "envs": 21,
        "segment_index": 0,
        "archetypes": (
            "aggressive-pressure",
            "defensive-cycle",
            "beatdown",
            "air-beatdown",
            "siege-bait",
            "random-legal",
        ),
        "strategies": (
            "aggressive-pressure",
            "defensive-cycle",
            "beatdown",
            "siege-bait",
            "random-legal",
        ),
        "sampling_mix": mix,
    }
    first = sample_training_scenarios(pool, **kwargs)
    second = sample_training_scenarios(pool, **kwargs)

    assert [scenario.as_dict() for scenario in first] == [
        scenario.as_dict() for scenario in second
    ]
    assert first[0].sampling_source == "regression-anchor"
    assert Counter(s.sampling_source for s in first[1:]) == Counter(
        {
            "isolated-offense": 5,
            "ground-defense": 5,
            "air-defense": 4,
            "spell-situations": 3,
            "kiting-cycling-elixir": 3,
        }
    )
    assert first[1].deck.archetype == "aggressive-pressure"
    assert first[2].deck.archetype == "defensive-cycle"


def test_variant_phase_source_requests_variant_decks() -> None:
    pool = OpponentPool(load_fixed_ruleset(), seed=47)
    scenarios = sample_training_scenarios(
        pool,
        envs=2,
        segment_index=0,
        archetypes=("aggressive-pressure",),
        strategies=("aggressive-pressure",),
        include_regression=False,
        sampling_mix=(("randomized-variants", 1.0),),
    )

    assert scenarios[0].sampling_source == "randomized-variants"
    assert scenarios[0].deck.source == "curated-variant"


def test_generalized_config_records_threat_stratification() -> None:
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=4, horizon=2, updates=1),
        segments=1,
        threat_stratified=True,
    )
    assert config.threat_stratified is True


def test_temporary_potential_reward_anneals_by_global_segment() -> None:
    base = PrototypeConfig(
        horizon=8,
        potential_reward_weight=0.1,
    )

    assert _segment_config(
        base,
        segment_index=0,
        rollouts_per_scenario=1,
        potential_reward_anneal_segments=32,
    ).potential_reward_weight == pytest.approx(0.1)
    assert _segment_config(
        base,
        segment_index=16,
        rollouts_per_scenario=1,
        potential_reward_anneal_segments=32,
    ).potential_reward_weight == pytest.approx(0.05)
    assert _segment_config(
        base,
        segment_index=32,
        rollouts_per_scenario=1,
        potential_reward_anneal_segments=32,
    ).potential_reward_weight == pytest.approx(0.0)


def test_pfsp_updates_use_lane_indexed_terminal_outcomes() -> None:
    assignments = (None, "old-a", "old-b")
    report = {
        "update_rows": [
            {
                "rollout": {
                    "match_outcomes": [
                        {"lane": 0, "outcome": "win"},
                        {"lane": 1, "outcome": "loss"},
                    ]
                }
            },
            {
                "rollout": {
                    "match_outcomes": [
                        {"lane": 1, "outcome": "win"},
                        {"lane": 2, "outcome": "draw"},
                    ]
                }
            },
        ]
    }

    book, recorded = _update_payoff_book_from_segment(
        LeaguePayoffBook(),
        report,
        assignments,
        learner_agent_id="main",
    )

    assert recorded == 3
    assert book.stats("main", "old-a") == LeaguePayoffStats(wins=1, losses=1)
    assert book.stats("main", "old-b") == LeaguePayoffStats(draws=1)


def test_pfsp_payoff_paths_require_pfsp_mode() -> None:
    with pytest.raises(GeneralizedTrainingError, match="require pfsp"):
        GeneralizedTrainingConfig(pfsp_payoff_book="payoffs.json")


def test_scenario_callback_uses_authoritative_state_only_for_opponent_action() -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=7)
    scenarios = sample_training_scenarios(
        pool,
        envs=2,
        segment_index=0,
        archetypes=("aggressive-pressure",),
        strategies=("aggressive-pressure",),
    )
    environment = SimulatorEnv(
        engine=BattleEngine(ruleset),
        include_authoritative_state=False,
    )
    environment.reset_v2(
        seed=9,
        # Use the regression deck here because the legacy V2 hand projection
        # intentionally rejects simulator-only card forms such as Battle Ram.
        decks=(tuple(PLAYER_DECK), scenarios[0].deck.cards),
        shuffle_decks=False,
    )
    callback = make_scenario_opponent_action(scenarios)
    action = callback(environment, environment.observe_v2()[1], 1)
    assert environment.engine.validate_action(environment.state, action) is None


def test_v2_boundary_accepts_non_player_cards_in_opponent_deck() -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=19)
    deck = pool.sample_deck(0, archetype="aggressive-pressure")
    environment = SimulatorEnv(engine=BattleEngine(ruleset))

    observations = environment.reset_v2(
        seed=11,
        decks=(tuple(PLAYER_DECK), deck.cards),
        shuffle_decks=False,
    )

    assert len(observations) == 2
    assert observations[0].board.shape == (21, 32, 18)
    # Unrepresented opponent cards are masked as unavailable in the opponent
    # viewer's public hand; authoritative state retains the real deck.
    assert tuple(environment.state.players[1].deck) == deck.cards


def test_heldout_config_has_distinct_matrix_axes() -> None:
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=2, horizon=8, updates=1),
        segments=2,
    )
    matrix = build_heldout_matrix_config(
        "checkpoint.pt",
        seed=3,
        archetypes=("deterministic-cycle", "beatdown"),
        strategies=("deterministic-cycle", "beatdown"),
        seeds=(10, 11),
    )

    assert config.prototype_config.envs == 2
    assert matrix.match_count == 8
    assert matrix.held_out is False
    assert matrix.held_out_source is None
    assert matrix.excluded_deck_compositions == ()
    assert len({deck.deck_id for deck in matrix.opponent_decks}) == 2
    assert len({strategy.strategy_id for strategy in matrix.strategies}) == 2


def test_generalized_allows_factor_bc_with_regular_ppo() -> None:
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(
            envs=2,
            horizon=8,
            updates=1,
            imitation_only=False,
            expert_execution_probability=0.0,
            behavior_cloning_factor_coef=0.05,
        ),
        segments=1,
        expert_guidance=True,
    )

    assert config.prototype_config.behavior_cloning_factor_coef == 0.05


def test_generalized_training_can_keep_a_scenario_across_rollouts(monkeypatch, tmp_path) -> None:
    import simulator.rl.generalized as generalized

    calls: list[tuple[int, int, int]] = []

    def fake_train(config, *, checkpoint, checkpoint_out, **kwargs):
        calls.append((config.updates, config.envs, config.horizon))
        return {
            "final_update": sum(item[0] for item in calls),
            "outcomes": {
                "completed_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "truncated_matches": 0,
            },
        }

    monkeypatch.setattr(generalized, "train_prototype", fake_train)
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=2, horizon=8, updates=1),
        segments=2,
        rollouts_per_scenario=3,
        checkpoint_out=tmp_path / "generalized.pt",
    )

    report = generalized.train_generalized(config)

    assert calls == [(3, 2, 8), (3, 2, 8)]
    assert report["rollouts_per_scenario"] == 3
    assert report["transitions"] == 2 * 3 * 2 * 8
    assert report["decision_cursor_start"] == 0
    assert report["decision_cursor_end"] == report["transitions"]
    assert report["transitions_per_segment"] == 2 * 3 * 8
    assert report["curriculum_stage_basis"] == "cumulative-decisions"


def test_generalized_training_supports_player_one_side(monkeypatch, tmp_path) -> None:
    import simulator.rl.generalized as generalized

    observed: list[int] = []

    def fake_train(config, *, checkpoint, checkpoint_out, **kwargs):
        observed.append(config.target_player)
        return {
            "final_update": 1,
            "outcomes": {
                "completed_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "truncated_matches": 0,
            },
        }

    monkeypatch.setattr(generalized, "train_prototype", fake_train)
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(
            envs=1,
            horizon=8,
            updates=1,
            target_player=1,
        ),
        segments=1,
        checkpoint_out=tmp_path / "player-one.pt",
    )

    report = generalized.train_generalized(config)

    assert observed == [1]
    assert report["target_player"] == 1
    assert report["actor_player"] == 1
    assert report["opponent_player"] == 0


def test_generalized_resume_advances_schedule_from_sidecar(monkeypatch, tmp_path) -> None:
    import json
    import simulator.rl.generalized as generalized

    checkpoint = tmp_path / "generalized.pt"
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "kind": generalized.GENERALIZED_TRAINING_KIND,
                "segments": 2,
                "segment_indices": [0, 1],
                "decision_cursor_end": 987,
            }
        ),
        encoding="utf-8",
    )
    observed: list[int] = []

    def fake_train(config, *, checkpoint, checkpoint_out, opponent_decks, **kwargs):
        observed.append(config.seed)
        return {
            "final_update": 1,
            "outcomes": {
                "completed_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "truncated_matches": 0,
            },
        }

    monkeypatch.setattr(generalized, "train_prototype", fake_train)
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=2, horizon=8, updates=1),
        segments=2,
        checkpoint=checkpoint,
        checkpoint_out=tmp_path / "next.pt",
    )
    report = generalized.train_generalized(config)

    assert report["starting_segment"] == 2
    assert report["segment_offset_source"] == "generalized-sidecar"
    assert report["segment_indices"] == [2, 3]
    assert report["decision_cursor_start"] == 987
    assert report["decision_cursor_end"] == 987 + (2 * 8 * 2)
    assert observed[0] != observed[1]


def test_heldout_training_report_excludes_sampled_decks(tmp_path) -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=13)
    excluded = pool.sample_deck(100_000, archetype="beatdown")
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "scenario_schedule": [
                    [{"deck": {"cards": list(excluded.cards)}}]
                ]
            }
        ),
        encoding="utf-8",
    )

    matrix = build_heldout_matrix_config(
        "checkpoint.pt",
        seed=13,
        archetypes=("beatdown",),
        strategies=("beatdown",),
        training_report=report_path,
    )

    assert frozenset(matrix.opponent_decks[0].cards) != frozenset(excluded.cards)
    assert matrix.held_out_source == report_path
    assert matrix.held_out is False
    assert matrix.excluded_deck_compositions == (tuple(sorted(excluded.cards)),)


def test_heldout_report_with_complete_matching_provenance_is_verified(tmp_path) -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=14)
    excluded = pool.sample_deck(100_000, archetype="beatdown")
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"checkpoint-artifact")
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "kind": "recurrent_public_ppo_generalized_training",
                "schema_version": 1,
                "checkpoint": str(checkpoint),
                "checkpoint_fingerprint": {
                    "exists": True,
                    "sha256": hashlib.sha256(b"checkpoint-artifact").hexdigest(),
                },
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "player_deck": list(PLAYER_DECK),
                "scenario_schedule": [
                    [{"deck": {"cards": list(excluded.cards)}}]
                ],
            }
        ),
        encoding="utf-8",
    )

    matrix = build_heldout_matrix_config(
        checkpoint,
        seed=14,
        archetypes=("beatdown",),
        strategies=("beatdown",),
        training_report=report_path,
    )

    assert matrix.held_out is True
    assert matrix.held_out_source == report_path
    assert matrix.excluded_deck_compositions == (tuple(sorted(excluded.cards)),)


def test_heldout_missing_training_report_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"checkpoint-artifact")

    with pytest.raises(GeneralizedTrainingError, match="cannot read training report"):
        build_heldout_matrix_config(
            checkpoint,
            archetypes=("beatdown",),
            strategies=("beatdown",),
            training_report=tmp_path / "missing-training.json",
        )


def test_heldout_report_rejects_stale_checkpoint_fingerprint(tmp_path) -> None:
    ruleset = load_fixed_ruleset()
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"checkpoint-artifact")
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "kind": "recurrent_public_ppo_generalized_training",
                "schema_version": 1,
                "checkpoint": str(checkpoint),
                "checkpoint_fingerprint": {
                    "exists": True,
                    "sha256": "0" * 64,
                },
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "player_deck": list(PLAYER_DECK),
                "scenario_schedule": [
                    [{"deck": {"cards": list(PLAYER_DECK)}}]
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GeneralizedTrainingError, match="fingerprint does not match"):
        build_heldout_matrix_config(
            checkpoint,
            archetypes=("beatdown",),
            strategies=("beatdown",),
            training_report=report_path,
        )


def test_heldout_report_rejects_contradictory_identity_metadata(tmp_path) -> None:
    ruleset = load_fixed_ruleset()
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"checkpoint-artifact")
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "kind": "not-a-generalized-training-report",
                "schema_version": 1,
                "checkpoint": str(checkpoint),
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": "sha256:" + "0" * 64,
                "player_deck": list(PLAYER_DECK),
                "scenario_schedule": [
                    [{"deck": {"cards": list(PLAYER_DECK)}}]
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GeneralizedTrainingError, match="unsupported kind"):
        build_heldout_matrix_config(
            checkpoint,
            archetypes=("beatdown",),
            strategies=("beatdown",),
            training_report=report_path,
        )


def test_heldout_report_rejects_ruleset_and_checkpoint_mismatches(tmp_path) -> None:
    ruleset = load_fixed_ruleset()
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"checkpoint-artifact")
    report_checkpoint = tmp_path / "other-actor.pt"
    report_checkpoint.write_bytes(b"different-checkpoint")
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "kind": "recurrent_public_ppo_generalized_training",
                "schema_version": 1,
                "checkpoint": str(report_checkpoint),
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_hash": ruleset.content_hash,
                "player_deck": list(PLAYER_DECK),
                "scenario_schedule": [
                    [{"deck": {"cards": list(PLAYER_DECK)}}]
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GeneralizedTrainingError, match="checkpoint does not match"):
        build_heldout_matrix_config(
            checkpoint,
            archetypes=("beatdown",),
            strategies=("beatdown",),
            training_report=report_path,
        )

    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            ruleset.content_hash,
            "sha256:" + "0" * 64,
        ),
        encoding="utf-8",
    )
    with pytest.raises(GeneralizedTrainingError, match="ruleset_hash"):
        build_heldout_matrix_config(
            checkpoint,
            archetypes=("beatdown",),
            strategies=("beatdown",),
            training_report=report_path,
        )


def test_generalized_report_records_frozen_checkpoint_lane_assignments(
    monkeypatch,
    tmp_path,
) -> None:
    import simulator.rl.generalized as generalized

    calls: list[tuple[tuple[tuple[str, ...], ...], object]] = []

    def fake_train(config, *, checkpoint, checkpoint_out, opponent_decks, opponent_action, **kwargs):
        calls.append((tuple(tuple(deck) for deck in opponent_decks), opponent_action))
        return {
            "final_update": 1,
            "outcomes": {
                "completed_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "truncated_matches": 0,
            },
        }

    monkeypatch.setattr(generalized, "train_prototype", fake_train)
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=3, horizon=2, updates=1),
        segments=1,
        opponent_checkpoints=(tmp_path / "old-a.pt", tmp_path / "old-b.pt"),
        checkpoint_out=tmp_path / "next.pt",
    )

    report = generalized.train_generalized(config)

    assignments = report["opponent_assignments"][0]
    assert [row["source"] for row in assignments] == [
        "simulator-controller",
        "frozen-checkpoint",
        "frozen-checkpoint",
    ]
    assert [row["checkpoint"] for row in assignments] == [
        None,
        str(tmp_path / "old-a.pt"),
        str(tmp_path / "old-b.pt"),
    ]
    assert assignments[1]["checkpoint_fingerprint"]["exists"] is False
    assert assignments[2]["deck_cards"] == list(report["scenario_schedule"][0][2]["deck"]["cards"])
    assert len(calls) == 1


def test_generalized_player_deck_is_passed_to_training_and_reported(monkeypatch, tmp_path) -> None:
    import simulator.rl.generalized as generalized

    ruleset = load_fixed_ruleset()
    custom_deck = tuple(
        OpponentPool(ruleset, seed=91).sample_deck(91, archetype="beatdown").cards
    )
    captured: list[tuple[str, ...]] = []

    def fake_train(
        config,
        *,
        checkpoint,
        checkpoint_out,
        player_deck,
        opponent_decks,
        opponent_action,
        **kwargs,
    ):
        captured.append(tuple(player_deck))
        return {
            "final_update": 1,
            "player_deck": list(player_deck),
            "outcomes": {
                "completed_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "truncated_matches": 0,
            },
        }

    monkeypatch.setattr(generalized, "train_prototype", fake_train)
    config = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=2, horizon=2, updates=1),
        player_deck=list(custom_deck),
        segments=1,
        checkpoint_out=tmp_path / "custom-deck.pt",
    )

    report = generalized.train_generalized(config)

    assert config.player_deck == custom_deck
    assert captured == [custom_deck]
    assert report["player_deck"] == list(custom_deck)


def test_heldout_matrix_exposes_configured_player_deck_to_each_match() -> None:
    from simulator.rl.evaluation_matrix import run_evaluation_matrix

    ruleset = load_fixed_ruleset()
    custom_deck = tuple(
        OpponentPool(ruleset, seed=92).sample_deck(92, archetype="air-beatdown").cards
    )
    matrix = build_heldout_matrix_config(
        "checkpoint.pt",
        player_deck=custom_deck,
        archetypes=("beatdown",),
        strategies=("wait",),
        seeds=(7,),
    )
    seen: list[tuple[str, ...]] = []

    def fake_runner(spec):
        seen.append(tuple(spec.player_deck))
        return {"outcome": "truncated", "decisions": 1}

    report = run_evaluation_matrix(matrix, match_runner=fake_runner)

    assert matrix.player_deck == custom_deck
    assert seen == [custom_deck]
    assert report["player_deck"] == list(custom_deck)
    assert report["matches"][0]["player_deck"] == list(custom_deck)


def test_player_deck_defaults_remain_the_fixed_prototype_deck() -> None:
    from simulator.rl.evaluation_matrix import EvaluationMatrixConfig, OpponentDeckSpec

    generalized = GeneralizedTrainingConfig(
        prototype_config=PrototypeConfig(envs=1, horizon=2, updates=1),
        segments=1,
    )
    matrix = EvaluationMatrixConfig(
        checkpoint="checkpoint.pt",
        opponent_decks=(
            OpponentDeckSpec("opponent", tuple(PLAYER_DECK)),
        ),
        strategies=("wait",),
        seeds=(1,),
    )
    legacy_positional = GeneralizedTrainingConfig(
        PrototypeConfig(envs=1, horizon=2, updates=1),
        1,
        1,
    )

    assert generalized.player_deck == tuple(PLAYER_DECK)
    assert matrix.player_deck == tuple(PLAYER_DECK)
    assert legacy_positional.segments == 1
    assert legacy_positional.rollouts_per_scenario == 1
