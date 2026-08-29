from __future__ import annotations

from copy import deepcopy
import json

import pytest


def _deck(deck_id: str, first: str = "hog-rider"):
    from simulator.rl.evaluation_matrix import OpponentDeckSpec

    cards = (
        first,
        "cannon",
        "musketeer",
        "skeletons",
        "ice-golem",
        "ice-spirit",
        "fireball",
        "log",
    )
    return OpponentDeckSpec(deck_id, cards)


def _comparison_report(tmp_path, name, outcomes, *, metrics_by_cell=None):
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        run_evaluation_matrix,
    )

    decks = (_deck("cell-a"), _deck("cell-b", first="giant"))
    config = EvaluationMatrixConfig(
        checkpoint=tmp_path / name,
        opponent_decks=decks,
        strategies=("wait",),
        seeds=(1, 2),
        max_decisions=20,
        held_out=False,
        include_match_results=True,
    )

    def runner(spec):
        key = (spec.opponent_deck.deck_id, spec.seed)
        result = {
            "outcome": outcomes[key],
            "decisions": 4 + spec.seed,
            "return": 1.0 if outcomes[key] == "win" else -1.0,
        }
        if metrics_by_cell and key in metrics_by_cell:
            result["metrics"] = metrics_by_cell[key]
        return result

    return run_evaluation_matrix(config, match_runner=runner)


def test_matrix_config_normalizes_inputs_and_counts_cells() -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        OpponentDeckSpec,
        OpponentStrategySpec,
    )

    config = EvaluationMatrixConfig.from_mapping(
        {
            "checkpoint": "checkpoint.pt",
            "opponent_decks": [_deck("cycle").as_dict()],
            "strategies": ["WAIT"],
            "seeds": [3, 7],
            "max_decisions": 12,
            "domain_randomization": {
                "profile_id": "eval-interface",
                "decision_interval_jitter_ticks": 1,
                "action_latency_max_steps": 2,
                "entity_observation_noise_std": 0.01,
            },
        }
    )

    assert config.match_count == 2
    assert isinstance(config.opponent_decks[0], OpponentDeckSpec)
    assert isinstance(config.strategies[0], OpponentStrategySpec)
    assert config.strategies[0].strategy_id == "wait"
    assert config.as_dict()["seeds"] == [3, 7]
    assert config.domain_randomization is not None
    assert config.domain_randomization.profile_id == "eval-interface"
    assert config.as_dict()["domain_randomization"]["action_latency_max_steps"] == 2


def test_evaluation_batch_config_drops_incompatible_training_chunking() -> None:
    from simulator.rl.evaluation_matrix import _evaluation_batch_config
    from simulator.rl.prototype import PrototypeConfig

    stored = PrototypeConfig(
        envs=4,
        horizon=512,
        sequence_length=128,
    )

    full_match = _evaluation_batch_config(stored, envs=2, horizon=1200)
    compatible = _evaluation_batch_config(stored, envs=2, horizon=1024)

    assert full_match.horizon == 1200
    assert full_match.sequence_length is None
    assert compatible.sequence_length == 128


def test_evaluation_reports_terminal_crown_totals() -> None:
    from types import SimpleNamespace

    from simulator.rl.evaluation_matrix import (
        MatchResult,
        _CheckpointMatchRunner,
        _summary,
    )

    environment = SimpleNamespace(
        state=SimpleNamespace(
            players=(SimpleNamespace(crowns=2), SimpleNamespace(crowns=1))
        )
    )
    assert _CheckpointMatchRunner._crown_snapshot(environment) == {
        "player_0": 2,
        "player_1": 1,
    }

    summary = _summary(
        [
            MatchResult(
                "win",
                metrics={"crowns_end": {"player_0": 2, "player_1": 1}},
            ),
            MatchResult(
                "loss",
                metrics={"crowns_end": {"player_0": 0, "player_1": 3}},
            ),
        ]
    )
    assert summary["crowns_end"] == {"player_0": 2, "player_1": 4}
    assert summary["crowns_end_matches"] == 2


def test_matrix_aggregates_each_deck_strategy_seed_and_is_json_safe() -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        MatchResult,
        OpponentStrategySpec,
        run_evaluation_matrix,
    )

    decks = (_deck("cycle-a"), _deck("cycle-b", first="giant"))
    strategies = (
        OpponentStrategySpec("wait"),
        OpponentStrategySpec("custom", factory=lambda seed: object()),
    )
    config = EvaluationMatrixConfig(
        checkpoint="not-loaded-by-injected-runner.pt",
        opponent_decks=decks,
        strategies=strategies,
        seeds=(10, 11),
        include_match_results=True,
    )
    seen = []
    progress = []

    def runner(spec):
        seen.append((spec.opponent_deck.deck_id, spec.strategy.strategy_id, spec.seed))
        if spec.seed == 10 and spec.strategy.strategy_id == "wait":
            return MatchResult("win", decisions=4, return_value=1.0, winner=0)
        if spec.seed == 11 and spec.opponent_deck.deck_id == "cycle-b":
            return {"outcome": "truncated", "decisions": 12, "terminal_reason": "cap"}
        return {"winner": 1, "decisions": 5, "return": -1.0}

    report = run_evaluation_matrix(
        config,
        match_runner=runner,
        progress_callback=lambda complete, total, spec, result: progress.append(
            (complete, total, spec.seed, result.outcome)
        ),
    )

    assert len(seen) == 8
    assert seen[0] == ("cycle-a", "wait", 10)
    assert seen[-1] == ("cycle-b", "custom", 11)
    assert progress[-1][:2] == (8, 8)
    assert report["matrix_size"] == 8
    total = report["total"]
    assert total["matches"] == 8
    assert total["completed"] == 6
    assert total["wins"] == 2
    assert total["losses"] == 4
    assert total["draws"] == 0
    assert total["truncated"] == 2
    assert total["win_rate"] == pytest.approx(2 / 6)
    assert total["completion_rate"] == pytest.approx(6 / 8)
    assert total["truncation_rate"] == pytest.approx(2 / 8)
    assert total["all_wins"] is False
    assert total["all_completed_wins"] is False
    assert total["terminal_reasons"] == {"<unspecified>": 6, "cap": 2}
    assert total["opponent_rejected_actions"] == 0
    assert report["quality_gate"]["passed"] is False
    assert "truncated_matches" in report["quality_gate"]["failures"]
    assert total["decisions_total"] == 52
    assert total["decisions_min"] == 4
    assert total["decisions_max"] == 12
    assert report["schema_version"] == 2
    assert len(report["cell_ids"]) == 8
    assert len(set(report["cell_ids"])) == 8
    assert report["actor_player"] == 0
    assert report["opponent_player"] == 1
    assert report["provenance"]["schema_version"] == 1
    assert report["provenance"]["config_fingerprint"].startswith("sha256:")
    assert report["provenance"]["matrix_order"] == "opponent_decks,strategies,seeds"
    assert report["timing"]["execution_mode"] == "sequential"
    assert report["timing"]["batch_count"] == 8
    assert report["timing"]["wall_seconds"] >= 0.0
    assert report["timing"]["match_execution_seconds"] >= 0.0
    assert report["timing"]["started_at_utc"].endswith("Z")
    assert report["timing"]["finished_at_utc"].endswith("Z")
    assert report["held_out_audit"]["disjointness_verified"] is False
    assert len(report["matchups"]) == 4
    assert report["matchups"][0]["summary"]["wins"] == 1
    nested_match = report["matchups"][0]["matches"][0]
    assert nested_match["cell_id"] == "cycle-a::wait::seed-10"
    assert nested_match["opponent_deck"]["cards"] == list(decks[0].cards)
    assert nested_match["opponent_strategy"]["strategy_id"] == "wait"
    assert len(report["matches"]) == 8
    assert report["matches"][0]["cell_id"] == "cycle-a::wait::seed-10"
    assert report["matches"][0]["opponent_deck"]["cards"] == list(decks[0].cards)
    assert report["matches"][0]["opponent_strategy"]["strategy_id"] == "wait"
    json.dumps(report, allow_nan=False)


def test_matrix_report_keeps_actor_side_and_opponent_identity_for_player_one(
    tmp_path,
) -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        run_evaluation_matrix,
    )

    deck = _deck("left-side-opponent", first="giant")
    config = EvaluationMatrixConfig(
        checkpoint=tmp_path / "player-one.pt",
        opponent_decks=(deck,),
        strategies=("wait",),
        seeds=(73,),
        target_player=1,
        held_out=False,
    )

    report = run_evaluation_matrix(
        config,
        match_runner=lambda spec: {
            "winner": spec.target_player,
            "decisions": 2,
        },
    )

    assert report["target_player"] == 1
    assert report["actor_player"] == 1
    assert report["opponent_player"] == 0
    row = report["matches"][0]
    assert row["target_player"] == 1
    assert row["actor_player"] == 1
    assert row["opponent_player"] == 0
    assert row["opponent_deck"]["deck_id"] == deck.deck_id
    assert row["opponent_deck"]["cards"] == list(deck.cards)
    assert row["opponent_strategy"]["strategy_id"] == "wait"
    assert row["seed"] == 73
    assert report["total"]["wins"] == 1


def test_matrix_config_fingerprint_is_stable_for_identical_inputs(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        run_evaluation_matrix,
    )

    def make_report():
        config = EvaluationMatrixConfig(
            checkpoint=tmp_path / "same.pt",
            opponent_decks=(_deck("stable"),),
            strategies=("wait",),
            seeds=(9,),
            held_out=False,
        )
        return run_evaluation_matrix(
            config,
            match_runner=lambda _spec: {"outcome": "draw", "decisions": 1},
        )

    first = make_report()
    second = make_report()

    assert first["provenance"]["config_fingerprint"] == second["provenance"][
        "config_fingerprint"
    ]
    assert first["provenance"]["checkpoint_fingerprint"] == second["provenance"][
        "checkpoint_fingerprint"
    ]


def test_matrix_rows_preserve_checkpoint_provenance_and_heldout_exclusions(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        OpponentStrategySpec,
        run_evaluation_matrix,
    )

    checkpoint = tmp_path / "old-actor.pt"
    checkpoint.write_bytes(b"checkpoint-for-audit")
    excluded = tuple(sorted(_deck("excluded").cards))
    strategy = OpponentStrategySpec(
        "frozen-old-actor",
        factory=lambda _seed: object(),
        description="frozen actor",
        metadata={
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": "declared-by-caller",
        },
    )
    config = EvaluationMatrixConfig(
        checkpoint=checkpoint,
        opponent_decks=(_deck("heldout", first="giant"),),
        strategies=(strategy,),
        seeds=(17,),
        held_out=True,
        held_out_source=tmp_path / "training.json",
        excluded_deck_compositions=(excluded,),
    )

    report = run_evaluation_matrix(
        config,
        match_runner=lambda _spec: {
            "outcome": "truncated",
            "decisions": 3,
            "terminal_reason": "evaluation_cap",
            "metrics": {
                "target_plays_by_card": {"hog-rider": 1},
                "target_rejected_actions": 2,
                "target_play_trace": [{"card_id": "hog-rider"}],
            },
        },
    )

    row = report["matches"][0]
    assert row["cell_id"] == "heldout::frozen-old-actor::seed-17"
    assert row["opponent_strategy"]["metadata"]["checkpoint_path"] == str(checkpoint)
    assert row["opponent_deck"]["deck_id"] == "heldout"
    assert report["checkpoint_fingerprint"]["sha256"]
    assert report["held_out_audit"]["disjointness_verified"] is True
    assert report["held_out_audit"]["overlap"] == []
    assert report["matchups"][0]["summary"]["target_plays_by_card"] == {"hog-rider": 1}
    assert report["matchups"][0]["summary"]["target_rejected_actions"] == 2
    assert report["matchups"][0]["summary"]["opponent_rejected_actions"] == 0
    assert report["quality_gate"]["checks"]["no_rejected_actions"] is False


def test_matrix_quality_gate_separates_integrity_from_strength() -> None:
    from simulator.rl.evaluation_matrix import _evaluation_quality_gate

    report = {
        "policy_mode": "actor",
        "actor_controls_actions": True,
        "actor_privileged_inputs": False,
        "held_out": True,
        "held_out_audit": {
            "disjointness_verified": True,
            "overlap": [],
        },
        "total": {
            "matches": 2,
            "completed": 2,
            "wins": 0,
            "win_rate": 0.0,
            "truncated": 0,
            "target_rejected_actions": 0,
            "opponent_rejected_actions": 0,
        },
    }

    gate = _evaluation_quality_gate(report, {"status": "clean"})

    assert gate["passed"] is True
    assert gate["strength_evidence"]["used_as_gate"] is False

    report["total"]["opponent_rejected_actions"] = 1
    failed = _evaluation_quality_gate(report, {"status": "clean"})
    assert failed["passed"] is False
    assert failed["failures"] == ["rejected_actions"]


def test_matrix_rejects_duplicate_axes_and_nonfinite_results() -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        EvaluationMatrixError,
        MatchResult,
    )

    with pytest.raises(EvaluationMatrixError, match="unique deck IDs"):
        EvaluationMatrixConfig(
            checkpoint="checkpoint.pt",
            opponent_decks=(_deck("same"), _deck("same", first="giant")),
            strategies=("wait",),
            seeds=(1,),
        )
    with pytest.raises(EvaluationMatrixError, match="unique"):
        EvaluationMatrixConfig(
            checkpoint="checkpoint.pt",
            opponent_decks=(_deck("one"),),
            strategies=("wait",),
            seeds=(1, 1),
        )
    with pytest.raises(EvaluationMatrixError, match="finite"):
        MatchResult("win", return_value=float("nan"))


def test_builtin_strategy_builds_a_legal_deterministic_action() -> None:
    from simulator.engine import BattleEngine
    from simulator.rl.evaluation_matrix import OpponentStrategySpec
    from simulator.ruleset import load_ruleset

    ruleset = load_ruleset("v1")
    engine = BattleEngine(ruleset, validate_every_tick=False)
    state = engine.new_battle(
        decks=(
            _deck("player").cards,
            _deck("opponent").cards,
        ),
        seed=5,
        shuffle_decks=False,
    )
    controller = OpponentStrategySpec("deterministic-left").build(5)
    action = controller.choose_action(engine, state, 1)

    assert action.player == 1
    assert engine.validate_action(state, action) is None


def test_report_comparison_pairs_by_cell_and_summarizes_outcome_deltas(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import compare_evaluation_reports

    baseline = _comparison_report(
        tmp_path,
        "baseline.pt",
        {
            ("cell-a", 1): "loss",
            ("cell-a", 2): "draw",
            ("cell-b", 1): "win",
            ("cell-b", 2): "loss",
        },
    )
    candidate = _comparison_report(
        tmp_path,
        "candidate.pt",
        {
            ("cell-a", 1): "win",
            ("cell-a", 2): "draw",
            ("cell-b", 1): "loss",
            ("cell-b", 2): "loss",
        },
    )
    # List order is not part of cell identity.  The comparator must pair by
    # cell_id so a reordered report remains a valid paired evaluation.
    candidate["matches"] = list(reversed(candidate["matches"]))

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison["cell_alignment"]["identical"] is True
    assert comparison["cell_alignment"]["paired_count"] == 4
    assert comparison["provenance_match"] is True
    assert comparison["checkpoint_identities_equal"] is False
    assert comparison["quality_gate"]["passed"] is True
    assert comparison["summary"]["improved_cells"] == 1
    assert comparison["summary"]["regressed_cells"] == 1
    assert comparison["summary"]["unchanged_cells"] == 2
    assert comparison["summary"]["outcome_delta_total"] == 0

    deltas = {row["cell_id"]: row for row in comparison["per_cell"]}
    assert deltas["cell-a::wait::seed-1"]["outcome_delta"] == 2
    assert deltas["cell-a::wait::seed-2"]["outcome_delta"] == 0
    assert deltas["cell-b::wait::seed-1"]["outcome_delta"] == -2
    assert deltas["cell-b::wait::seed-2"]["outcome_transition"] == "loss->loss"


def test_report_comparison_allows_different_training_exclusion_lists(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import compare_evaluation_reports

    outcomes = {
        ("cell-a", 1): "loss",
        ("cell-a", 2): "win",
        ("cell-b", 1): "loss",
        ("cell-b", 2): "win",
    }
    baseline = _comparison_report(tmp_path, "baseline.pt", outcomes)
    candidate = _comparison_report(tmp_path, "candidate.pt", outcomes)
    baseline["held_out"] = True
    candidate["held_out"] = True
    baseline["held_out_audit"] = {
        "selected_deck_compositions": ["selected"],
        "overlap": [],
        "disjointness_verified": True,
        "excluded_deck_compositions": [["baseline-only"]],
    }
    candidate["held_out_audit"] = {
        "selected_deck_compositions": ["selected"],
        "overlap": [],
        "disjointness_verified": True,
        "excluded_deck_compositions": [["candidate-only"]],
    }

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison["provenance_match"] is True
    assert comparison["quality_gate"]["passed"] is True


def test_report_comparison_reports_provenance_and_cell_mismatches(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import compare_evaluation_reports

    outcomes = {
        ("cell-a", 1): "win",
        ("cell-a", 2): "win",
        ("cell-b", 1): "loss",
        ("cell-b", 2): "loss",
    }
    baseline = _comparison_report(tmp_path, "baseline.pt", outcomes)
    candidate = deepcopy(_comparison_report(tmp_path, "candidate.pt", outcomes))
    candidate["runner"] = {"runner": "injected", "ruleset_hash": "different"}
    candidate["matches"][0]["deck_cards"] = [
        "different-card",
        *candidate["matches"][0]["deck_cards"][1:],
    ]

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison["provenance_match"] is False
    assert any(
        item["field"] == "runner"
        for item in comparison["provenance_mismatches"]
    )
    assert comparison["cell_alignment"]["identical"] is False
    assert comparison["cell_alignment"]["cell_definition_mismatches"][0][
        "cell_id"
    ] == "cell-a::wait::seed-1"
    assert set(comparison["quality_gate"]["failures"]) >= {
        "cells_not_identical",
        "provenance_mismatch",
    }


def test_report_comparison_gates_truncations_and_rejected_actions(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import compare_evaluation_reports

    outcomes = {
        ("cell-a", 1): "win",
        ("cell-a", 2): "win",
        ("cell-b", 1): "win",
        ("cell-b", 2): "win",
    }
    baseline = _comparison_report(tmp_path, "baseline.pt", outcomes)
    candidate = _comparison_report(
        tmp_path,
        "candidate.pt",
        outcomes,
        metrics_by_cell={
            ("cell-a", 1): {
                "target_rejected_actions": 2,
                "opponent_rejected_actions": 1,
            }
        },
    )
    candidate["matches"][0]["outcome"] = "truncated"

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison["safety"]["truncations"]["either"] == 1
    assert comparison["safety"]["rejected_actions"]["candidate"] == {
        "target": 2,
        "opponent": 1,
        "total": 3,
        "cell_ids": ["cell-a::wait::seed-1"],
        "invalid_metrics": [],
    }
    assert comparison["quality_gate"]["passed"] is False
    assert set(comparison["quality_gate"]["failures"]) >= {
        "truncated_matches",
        "rejected_actions",
    }
    first = comparison["per_cell"][0]
    assert first["candidate_outcome"] == "truncated"
    assert first["outcome_delta"] is None


def test_report_comparison_is_bounded_and_requires_per_cell_rows(tmp_path) -> None:
    from simulator.rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        EvaluationMatrixError,
        compare_evaluation_reports,
        run_evaluation_matrix,
    )

    outcomes = {
        ("cell-a", 1): "win",
        ("cell-a", 2): "win",
        ("cell-b", 1): "win",
        ("cell-b", 2): "win",
    }
    baseline = _comparison_report(tmp_path, "baseline.pt", outcomes)
    with pytest.raises(EvaluationMatrixError, match="comparison limit"):
        compare_evaluation_reports(baseline, baseline, max_cells=3)

    summary_only = run_evaluation_matrix(
        EvaluationMatrixConfig(
            checkpoint=tmp_path / "summary-only.pt",
            opponent_decks=(_deck("cell-a"),),
            strategies=("wait",),
            seeds=(1,),
            held_out=False,
            include_match_results=False,
        ),
        match_runner=lambda _spec: {"outcome": "win"},
    )
    with pytest.raises(EvaluationMatrixError, match="per-cell"):
        compare_evaluation_reports(summary_only, summary_only)
