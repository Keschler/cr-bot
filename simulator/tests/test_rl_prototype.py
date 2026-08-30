from __future__ import annotations

import io
import json
from copy import deepcopy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def test_v2_environment_reset_and_step_stay_on_public_boundary() -> None:
    from simulator.env import EnvStepV2, SimulatorEnv
    from simulator.observation_v2 import PolicyObservationV2

    environment = SimulatorEnv(decision_interval_us=1_000_000)
    observations = environment.reset_v2(seed=7, shuffle_decks=False)
    assert all(isinstance(observation, PolicyObservationV2) for observation in observations)
    step = environment.step_v2((None, None))
    assert isinstance(step, EnvStepV2)
    assert all(isinstance(observation, PolicyObservationV2) for observation in step.observations)


def test_v2_single_view_matches_full_view() -> None:
    from simulator.env import SimulatorEnv

    environment = SimulatorEnv(decision_interval_us=1_000_000)
    full_before = environment.reset_v2(seed=7, shuffle_decks=False)
    for single, full in zip(
        (environment.observe_v2_for_viewer(0), environment.observe_v2_for_viewer(1)),
        full_before,
        strict=True,
    ):
        assert np.array_equal(single.board, full.board)
        assert np.array_equal(single.global_vector, full.global_vector)
        assert np.array_equal(single.entity_tokens, full.entity_tokens)
        assert np.array_equal(single.entity_mask, full.entity_mask)
        assert np.array_equal(single.legal_play, full.legal_play)
        assert single.legal_wait == full.legal_wait

    full = environment.step_v2((None, None))
    single_environment = SimulatorEnv(decision_interval_us=1_000_000)
    single_environment.reset_v2(seed=7, shuffle_decks=False)
    single = single_environment.step_v2_for_viewer((None, None), viewer=0)
    assert single.rewards == full.rewards
    assert single.terminated == full.terminated
    assert single.truncated == full.truncated
    assert single.info == full.info
    assert np.array_equal(single.observations[0].board, full.observations[0].board)
    assert np.array_equal(
        single.observations[0].global_vector,
        full.observations[0].global_vector,
    )
    assert np.array_equal(
        single.observations[0].entity_tokens,
        full.observations[0].entity_tokens,
    )
    assert np.array_equal(
        single.observations[0].entity_mask,
        full.observations[0].entity_mask,
    )
    assert np.array_equal(
        single.observations[0].legal_play,
        full.observations[0].legal_play,
    )
    assert single.observations[0].legal_wait == full.observations[0].legal_wait
    assert single.observations[1] is None


def test_trace_reconciles_card_placement_and_troop_positions() -> None:
    from cr_bot.domain.game_state import Action as PolicyAction
    from rl.prototype import _trace_decision
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.ruleset import load_ruleset

    ruleset = load_ruleset("v1")
    environment = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        expose_privileged_info=True,
        include_authoritative_state=False,
    )
    environment.reset_v2(seed=71, shuffle_decks=False)
    state_before = deepcopy(environment.state)
    assert state_before is not None
    card_id = state_before.players[0].hand[0]
    legal_cells = environment.engine.legal_cells(state_before, 0, card_id)
    assert legal_cells
    requested_cell = legal_cells[0]
    result = environment.step_v2(
        (
            PolicyAction(kind="Play", card_idx=0, cell=requested_cell),
            None,
        )
    )
    row = _trace_decision(
        SimpleNamespace(
            decision_index=0,
            target_action=PolicyAction(kind="Play", card_idx=0, cell=requested_cell),
            opponent_action=None,
            result=result,
            state_after=environment.state,
            physics_tick_before=state_before.tick,
            elapsed_us_before=state_before.elapsed_us,
            hand_before=tuple(state_before.players[0].hand),
            elixir_before=state_before.players[0].elixir_milli,
        ),
        target_player=0,
        state_before=state_before,
    )

    assert row["accepted"] is True
    assert row["action_status"] == "accepted"
    assert row["card_id"] == card_id
    assert row["played_card_id"] == card_id
    assert row["world_cell"] == list(requested_cell)
    assert row["played_world_cell"] == list(requested_cell)
    assert row["rejection_reason"] is None
    assert row["tower_hp_before"]
    assert row["tower_hp_after"]
    assert isinstance(row["troop_positions_before"], list)
    assert isinstance(row["troop_positions_after"], list)
    assert any(item["owner"] == 0 for item in row["troop_positions_after"])


def test_trace_infers_applied_card_when_vector_result_omits_events() -> None:
    from cr_bot.domain.game_state import Action as PolicyAction
    from rl.prototype import _trace_decision
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.ruleset import load_ruleset

    ruleset = load_ruleset("v1")
    environment = SimulatorEnv(
        engine=BattleEngine(ruleset, validate_every_tick=False),
        expose_privileged_info=False,
        include_authoritative_state=True,
    )
    environment.reset_v2(seed=71, shuffle_decks=False)
    state_before = deepcopy(environment.state)
    assert state_before is not None
    card_id = state_before.players[0].hand[0]
    requested_cell = environment.engine.legal_cells(state_before, 0, card_id)[0]
    action = PolicyAction(kind="Play", card_idx=0, cell=requested_cell)
    result = environment.step_v2((action, None))

    row = _trace_decision(
        SimpleNamespace(
            decision_index=0,
            target_action=action,
            opponent_action=None,
            result=result,
            state_after=environment.state,
            physics_tick_before=state_before.tick,
            elapsed_us_before=state_before.elapsed_us,
            hand_before=tuple(state_before.players[0].hand),
            elixir_before=state_before.players[0].elixir_milli,
        ),
        target_player=0,
        state_before=state_before,
    )

    assert row["action_events"] == []
    assert row["accepted"] is True
    assert row["application_evidence"] == "state_transition"
    assert row["played_card_id"] == card_id
    assert row["played_world_cell"] == list(requested_cell)


@requires_torch
def test_recurrent_prototype_train_resume_and_evaluate(tmp_path) -> None:
    from rl.prototype import (
        PrototypeConfig,
        PrototypeConfigurationError,
        evaluate_prototype,
        load_prototype_checkpoint,
        load_shadow_prototype_checkpoint,
        train_prototype,
    )

    config = PrototypeConfig(
        envs=1,
        horizon=2,
        updates=1,
        decision_interval_us=1_000_000,
        seed=41,
        shuffle_decks=False,
        update_epochs=1,
        sequence_minibatch_size=1,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        use_privileged_critic=False,
        collect_belief_targets=False,
        allow_provisional=True,
    )
    first_path = tmp_path / "first.pt"
    progress_updates: list[tuple[int, int]] = []
    first = train_prototype(
        config,
        checkpoint_out=first_path,
        progress_callback=lambda update, transitions: progress_updates.append(
            (update, transitions)
        ),
    )

    assert first["kind"] == "recurrent_public_ppo_prototype"
    assert first["actor_privileged_inputs"] is False
    assert first["final_update"] == 1
    assert progress_updates == [(1, 2)]
    assert first_path.exists()
    assert first["wall_seconds"] > 0.0
    assert first["decisions_per_second"] == pytest.approx(
        first["transitions"] / first["wall_seconds"],
        rel=0.0,
        abs=1e-12,
    )
    assert "checkpoint promotion" in first["throughput_scope"]
    assert "JSON report validation" in first["throughput_scope"]
    assert "--json-out" in first["throughput_exclusions"]

    trace_path = tmp_path / "evaluation-trace.json"
    evaluation = evaluate_prototype(
        first_path,
        episodes=1,
        max_decisions=2,
        trace_out=trace_path,
    )
    assert evaluation["actor_privileged_inputs"] is False
    assert evaluation["episodes"] == 1
    assert evaluation["completed"] + evaluation["truncated"] == 1
    assert evaluation["truncated"] == 1
    assert evaluation["completion_rate"] == 0.0
    assert evaluation["truncation_rate"] == 1.0
    assert evaluation["terminal_reasons"] == {"evaluation_cap": 1}
    assert len(evaluation["episode_results"]) == 1
    assert evaluation["episode_results"][0]["decisions"] == 2
    assert evaluation["episode_results"][0]["cap_reached"] is True
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["kind"] == "recurrent_public_ppo_prototype_evaluation_trace"
    assert trace["trace_schema_version"] == 2
    assert trace["position_schema"]["version"] == "authoritative-world-v2"
    assert set(trace["position_schema"]["snapshots"]) == {"before", "after"}
    assert trace["action_schema"]["version"] == "requested-vs-applied-v2"
    assert len(trace["episodes"]) == 1
    assert len(trace["episodes"][0]["trace"]) == 2
    assert trace["episodes"][0]["player_deck"] == trace["episodes"][0]["opponent_deck"]
    assert "tower_hp_end" in trace["episodes"][0]
    assert "mode" in trace["episodes"][0]["trace"][0]
    assert "card_id" in trace["episodes"][0]["trace"][0]
    assert "played_card_id" in trace["episodes"][0]["trace"][0]
    assert "policy_cell" in trace["episodes"][0]["trace"][0]
    assert "world_cell" in trace["episodes"][0]["trace"][0]
    assert "played_world_cell" in trace["episodes"][0]["trace"][0]
    assert "action_status" in trace["episodes"][0]["trace"][0]
    assert "rejection_reason" in trace["episodes"][0]["trace"][0]
    assert "troop_positions_before" in trace["episodes"][0]["trace"][0]
    assert "troop_positions_after" in trace["episodes"][0]["trace"][0]

    from rl.evaluation_matrix import (
        EvaluationMatrixConfig,
        OpponentDeckSpec,
        OpponentStrategySpec,
        run_evaluation_matrix,
    )
    from simulator.roster import PLAYER_DECK

    matrix = run_evaluation_matrix(
        EvaluationMatrixConfig(
            checkpoint=first_path,
            opponent_decks=(OpponentDeckSpec("cycle", tuple(PLAYER_DECK)),),
            strategies=(OpponentStrategySpec("wait"),),
            seeds=(73,),
            max_decisions=2,
            batch_size=2,
            held_out=False,
        )
    )
    matrix_result = matrix["matches"][0]
    assert matrix_result["cell_id"] == "cycle::wait::seed-73"
    assert "target_play_trace" in matrix_result["metrics"]
    assert "opponent_play_trace" in matrix_result["metrics"]
    assert set(matrix_result["metrics"]["troop_positions_end"]) == {
        "player_0",
        "player_1",
    }

    batched_evaluation = evaluate_prototype(
        first_path,
        episodes=2,
        max_decisions=2,
        batch_size=2,
    )
    assert batched_evaluation["episodes"] == 2
    assert batched_evaluation["completed"] + batched_evaluation["truncated"] == 2
    assert batched_evaluation["mean_decisions"] == 2.0
    assert [item["decisions"] for item in batched_evaluation["episode_results"]] == [2, 2]
    assert all(
        item["terminal_reason"] == "evaluation_cap"
        for item in batched_evaluation["episode_results"]
    )

    resumed_path = tmp_path / "resumed.pt"
    resumed = train_prototype(
        config,
        checkpoint=first_path,
        checkpoint_out=resumed_path,
    )
    assert resumed["starting_update"] == 1
    assert resumed["final_update"] == 2
    assert resumed_path.exists()

    payload = torch.load(resumed_path, map_location="cpu")
    metadata = payload["metadata"]
    assert metadata["checkpoint_format"] == "recurrent-public-ppo-prototype-v1"
    assert metadata["actor_observation"]["source"] == "SimulatorEnv.observe_v2"
    assert metadata["actor_observation"]["privileged_inputs"] is False

    tampered_path = tmp_path / "tampered.pt"
    metadata["actor_observation"]["privileged_inputs"] = True
    torch.save(payload, tampered_path)
    with pytest.raises(PrototypeConfigurationError, match="public-only"):
        load_prototype_checkpoint(tampered_path)

    stale_payload = torch.load(first_path, map_location="cpu")
    stale_metadata = dict(stale_payload["metadata"])
    stale_metadata["ruleset_hash"] = "sha256:" + "a" * 64
    stale_payload["metadata"] = stale_metadata
    stale_path = tmp_path / "stale.pt"
    torch.save(stale_payload, stale_path)
    with pytest.raises(PrototypeConfigurationError, match="checkpoint ruleset"):
        load_prototype_checkpoint(stale_path)

    _learner, _stale_config, loaded_metadata = load_shadow_prototype_checkpoint(
        stale_path,
        allow_stale_ruleset=True,
        device="cpu",
    )
    assert loaded_metadata["_checkpoint_ruleset_match"] is False
    assert loaded_metadata["_stale_ruleset_allowed"] is True
    assert loaded_metadata["_checkpoint_ruleset_hash"] == "sha256:" + "a" * 64
    assert loaded_metadata["_runtime_ruleset_hash"] != "sha256:" + "a" * 64

    wrong_id_payload = torch.load(first_path, map_location="cpu")
    wrong_id_metadata = dict(wrong_id_payload["metadata"])
    wrong_id_metadata["ruleset_id"] = "v2"
    wrong_id_payload["metadata"] = wrong_id_metadata
    wrong_id_path = tmp_path / "wrong-id.pt"
    torch.save(wrong_id_payload, wrong_id_path)
    with pytest.raises(PrototypeConfigurationError, match="checkpoint ruleset"):
        load_shadow_prototype_checkpoint(
            wrong_id_path,
            allow_stale_ruleset=True,
            device="cpu",
        )

    malformed_payload = torch.load(first_path, map_location="cpu")
    malformed_metadata = dict(malformed_payload["metadata"])
    malformed_metadata["ruleset_hash"] = "not-a-content-hash"
    malformed_payload["metadata"] = malformed_metadata
    malformed_path = tmp_path / "malformed-hash.pt"
    torch.save(malformed_payload, malformed_path)
    with pytest.raises(PrototypeConfigurationError, match="well-formed"):
        load_shadow_prototype_checkpoint(
            malformed_path,
            allow_stale_ruleset=True,
            device="cpu",
        )


@requires_torch
def test_training_diagnostics_capture_decisions_and_update_statistics(tmp_path) -> None:
    from rl.prototype import PrototypeConfig, train_prototype

    trace_path = tmp_path / "training-trace.json"
    config = PrototypeConfig(
        envs=1,
        horizon=2,
        updates=1,
        decision_interval_us=1_000_000,
        seed=47,
        shuffle_decks=False,
        update_epochs=1,
        sequence_minibatch_size=1,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        use_privileged_critic=False,
        collect_belief_targets=False,
        behavior_cloning_factor_coef=0.25,
        diagnostic_trace_out=trace_path,
        allow_provisional=True,
    )

    report = train_prototype(
        config,
        checkpoint_out=tmp_path / "diagnostic.pt",
        expert_guidance=True,
    )

    assert report["actor_controls_actions"] is True
    assert report["training_diagnostics"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["kind"] == "recurrent_public_ppo_prototype_training_trace"
    assert trace["actor_controls_actions"] is True
    assert len(trace["decisions"]) == 2
    decision = trace["decisions"][0]
    for field in (
        "elapsed_us_before",
        "hand_before",
        "elixir_milli_before",
        "tower_hp_before",
        "troop_positions_before",
        "legal_action_mask",
        "actor_action",
        "chosen_action_probability",
        "top_alternative_actions",
        "strategic_teacher_action",
        "actor_teacher_agreement",
        "critic_value_prediction",
        "return",
        "advantage",
        "ppo_probability_ratio",
        "ppo_clipping_occurred",
    ):
        assert field in decision
    update = trace["updates"][0]
    for field in (
        "advantage_distribution",
        "explained_variance",
        "factor_entropy",
        "action_distribution",
        "action_distribution_delta",
        "post_update_ratio_distribution",
        "per_head_gradient_norms",
    ):
        assert field in update
    assert update["metrics"]["factor_behavior_cloning_loss"] is not None
    assert update["metrics"]["effective_factor_behavior_cloning_coef"] == 0.0


@requires_torch
def test_flagged_training_candidate_is_quarantined_without_overwriting_destination(
    monkeypatch,
    tmp_path,
) -> None:
    from rl import exploit_audit, prototype

    destination = tmp_path / "candidate.pt"
    destination.write_bytes(b"last-clean-checkpoint")
    monkeypatch.setattr(
        exploit_audit,
        "audit_simulation_report",
        lambda report: {
            "kind": "simulator_exploit_audit",
            "schema_version": 1,
            "status": "flagged",
            "simulation_exploit": True,
            "quarantine_required": True,
            "flags": [{"code": "test-flag", "severity": "warning"}],
            "checked": [],
            "metrics": {},
        },
    )
    config = prototype.PrototypeConfig(
        envs=1,
        horizon=2,
        updates=1,
        decision_interval_us=1_000_000,
        seed=73,
        shuffle_decks=False,
        update_epochs=1,
        sequence_minibatch_size=1,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        use_privileged_critic=False,
        collect_belief_targets=False,
        allow_provisional=True,
    )

    report = prototype.train_prototype(config, checkpoint_out=destination)

    assert report["simulation_exploit_audit"]["status"] == "flagged"
    assert report["checkpoint_promotion"]["status"] == "quarantined"
    assert destination.read_bytes() == b"last-clean-checkpoint"
    quarantined = tmp_path / report["quarantined_checkpoint"]
    assert quarantined.exists()
    assert not list(tmp_path.glob("*.candidate"))


def test_training_progress_reports_counts_and_elapsed_time() -> None:
    from rl.prototype import _TrainingProgress

    stream = io.StringIO()
    clock_values = iter((10.0, 11.25, 12.5))
    progress = _TrainingProgress(
        total_updates=4,
        transitions_per_update=8,
        stream=stream,
        clock=lambda: next(clock_values),
    )

    progress.update(1, 8)
    progress.update(4, 32)
    progress.close()

    output = stream.getvalue()
    assert "1/4 updates" in output
    assert "8/32 transitions" in output
    assert "4/4 updates" in output
    assert "32/32 transitions" in output
    assert "elapsed 1.2s" in output
    assert output.endswith("\n")


def test_imitation_only_requires_expert_loss_and_guidance() -> None:
    from rl.prototype import PrototypeConfig, PrototypeConfigurationError

    with pytest.raises(PrototypeConfigurationError, match="imitation_only"):
        PrototypeConfig(imitation_only=True)

    config = PrototypeConfig(
        imitation_only=True,
        behavior_cloning_coef=1.0,
    )
    assert config.imitation_only is True


def test_prototype_defaults_to_teacher_free_ppo() -> None:
    from rl.prototype import PrototypeConfig, evaluate_prototype

    config = PrototypeConfig()

    assert config.expert_execution_probability == 0.0
    assert config.imitation_only is False
    assert config.behavior_cloning_coef == 0.0
    assert config.behavior_cloning_factor_coef == 0.0
    assert config.entropy_coef > 0.0
    assert inspect.signature(evaluate_prototype).parameters["policy_mode"].default == "actor"


def test_ppo_kl_guard_rolls_back_only_excessive_updates() -> None:
    from rl.prototype import _apply_update_approx_kl_guard

    class FakeLearner:
        update_count = 8

        def __init__(self) -> None:
            self.loaded_state = None

        def load_checkpoint_state(self, state) -> None:
            self.loaded_state = state
            self.update_count = state["update_count"]

    learner = FakeLearner()
    rolled_back = _apply_update_approx_kl_guard(
        learner,
        SimpleNamespace(approx_kl=0.009, update_index=9),
        max_update_approx_kl=0.008,
        state_before_update={"update_count": 8},
        starting_update=8,
    )

    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["accepted_update"] == 8
    assert learner.loaded_state == {"update_count": 8}

    accepted = _apply_update_approx_kl_guard(
        learner,
        SimpleNamespace(approx_kl=0.004, update_index=9),
        max_update_approx_kl=0.008,
        state_before_update={"update_count": 8},
        starting_update=8,
    )
    assert accepted["status"] == "accepted"


def test_sequence_length_is_optional_but_must_tile_the_horizon() -> None:
    from rl.prototype import PrototypeConfig, PrototypeConfigurationError

    assert PrototypeConfig(horizon=8, sequence_length=4).sequence_length == 4
    with pytest.raises(PrototypeConfigurationError, match="divide horizon"):
        PrototypeConfig(horizon=8, sequence_length=3)


@requires_torch
def test_teacher_forced_transitions_are_rejected_for_regular_ppo(tmp_path) -> None:
    from rl.prototype import (
        PrototypeConfig,
        PrototypeConfigurationError,
        train_prototype,
    )

    config = PrototypeConfig(
        envs=1,
        horizon=1,
        updates=1,
        use_privileged_critic=False,
        collect_belief_targets=False,
        behavior_cloning_coef=1.0,
        expert_execution_probability=1.0,
        allow_provisional=True,
    )
    with pytest.raises(PrototypeConfigurationError, match="teacher-executed"):
        train_prototype(
            config,
            expert_guidance=True,
            expert_action_callback=lambda *_args: None,
            checkpoint_out=tmp_path / "invalid.pt",
        )


def test_resume_controls_reset_optimizer_and_override_runtime_config() -> None:
    from rl import prototype

    config = prototype.PrototypeConfig()

    class FakeOptimizer:
        def __init__(self) -> None:
            self.state = {"old_parameter": {"step": 4}}
            self.param_groups = [{"lr": 3e-4}, {"lr": 3e-4}]

    class FakeLearner:
        def __init__(self) -> None:
            self.optimizer = FakeOptimizer()

    effective = prototype._resume_config(
        config,
        checkpoint="source.pt",
        resume_learning_rate=1e-5,
        resume_disable_belief_loss=True,
        resume_reset_optimizer=True,
    )
    assert effective.learning_rate == pytest.approx(1e-5)
    assert effective.belief_coef == 0.0
    assert effective.collect_belief_targets is False

    learner = FakeLearner()
    prototype._apply_resume_controls(
        learner,
        effective,
        resume_learning_rate=1e-5,
        resume_reset_optimizer=True,
    )
    assert learner.optimizer.state == {}
    assert [group["lr"] for group in learner.optimizer.param_groups] == [1e-5, 1e-5]
    with pytest.raises(prototype.PrototypeConfigurationError, match="require --checkpoint"):
        prototype._resume_config(
            config,
            checkpoint=None,
            resume_learning_rate=1e-5,
            resume_disable_belief_loss=False,
            resume_reset_optimizer=False,
        )


@requires_torch
def test_resume_retry_controls_are_serialized_and_applied(tmp_path) -> None:
    from rl.prototype import PrototypeConfig, train_prototype

    config = PrototypeConfig(
        envs=1,
        horizon=1,
        updates=1,
        decision_interval_us=1_000_000,
        seed=53,
        shuffle_decks=False,
        update_epochs=1,
        sequence_minibatch_size=1,
        model_dim=8,
        encoder_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=16,
        gru_hidden_dim=8,
        belief_coef=0.05,
        use_privileged_critic=False,
        collect_belief_targets=True,
        allow_provisional=True,
    )
    source_path = tmp_path / "source.pt"
    retry_path = tmp_path / "retry.pt"
    train_prototype(config, checkpoint_out=source_path)

    report = train_prototype(
        config,
        checkpoint=source_path,
        checkpoint_out=retry_path,
        resume_learning_rate=1e-5,
        resume_disable_belief_loss=True,
        resume_reset_optimizer=True,
    )
    assert report["resume_controls"] == {
        "learning_rate": 1e-5,
        "belief_loss_disabled": True,
        "optimizer_reset": True,
    }

    payload = torch.load(retry_path, map_location="cpu")
    metadata = payload["metadata"]
    assert metadata["config"]["learning_rate"] == pytest.approx(1e-5)
    assert metadata["config"]["belief_coef"] == 0.0
    assert metadata["config"]["collect_belief_targets"] is False
    assert payload["learner"]["learner_config"]["belief_coef"] == 0.0
    assert all(
        group["lr"] == pytest.approx(1e-5)
        for group in payload["learner"]["optimizer"]["param_groups"]
    )


def test_train_cli_keeps_json_on_stdout_and_progress_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    tmp_path,
) -> None:
    from rl import prototype

    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    progress_stream = TtyBuffer()
    report = {"kind": "test", "transitions": 6}

    def fake_train(
        config,
        *,
        checkpoint,
        checkpoint_out,
        progress_callback=None,
        progress_step_callback=None,
        resume_learning_rate=None,
        resume_disable_belief_loss=False,
        resume_reset_optimizer=False,
    ):
        assert checkpoint is None
        assert checkpoint_out == tmp_path / "checkpoint.pt"
        assert progress_callback is not None
        assert progress_step_callback is not None
        assert resume_learning_rate is None
        assert resume_disable_belief_loss is False
        assert resume_reset_optimizer is False
        progress_callback(1, config.envs * config.horizon)
        return report

    monkeypatch.setattr(prototype, "train_prototype", fake_train)
    monkeypatch.setattr(prototype.sys, "stderr", progress_stream)

    assert prototype.main(
        [
            "train",
            "--updates",
            "1",
            "--envs",
            "2",
            "--horizon",
            "3",
            "--checkpoint-out",
            str(tmp_path / "checkpoint.pt"),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == report
    output = progress_stream.getvalue()
    assert "1/1 updates" in output
    assert "6/6 transitions" in output
    assert output.startswith("\r")
    assert output.endswith("\n")


def test_train_cli_forwards_explicit_resume_retry_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    tmp_path,
) -> None:
    from rl import prototype

    received: dict[str, object] = {}

    def fake_train(
        config,
        *,
        checkpoint,
        checkpoint_out,
        progress_callback=None,
        progress_step_callback=None,
        resume_learning_rate=None,
        resume_disable_belief_loss=False,
        resume_reset_optimizer=False,
    ):
        received.update(
            {
                "config": config,
                "checkpoint": checkpoint,
                "checkpoint_out": checkpoint_out,
                "resume_learning_rate": resume_learning_rate,
                "resume_disable_belief_loss": resume_disable_belief_loss,
                "resume_reset_optimizer": resume_reset_optimizer,
            }
        )
        return {"kind": "test"}

    monkeypatch.setattr(prototype, "train_prototype", fake_train)
    assert prototype.main(
        [
            "train",
            "--checkpoint",
            str(tmp_path / "source.pt"),
            "--learning-rate",
            "1e-5",
            "--resume-no-belief-loss",
            "--resume-reset-optimizer",
            "--checkpoint-out",
            str(tmp_path / "retry.pt"),
        ]
    ) == 0

    config = received["config"]
    assert config.learning_rate == pytest.approx(1e-5)
    assert received["resume_learning_rate"] == pytest.approx(1e-5)
    assert received["resume_disable_belief_loss"] is True
    assert received["resume_reset_optimizer"] is True
    assert received["checkpoint"] == tmp_path / "source.pt"
    assert json.loads(capsys.readouterr().out) == {"kind": "test"}


def test_target_player_one_orders_learner_deck_on_world_player_one() -> None:
    from rl.prototype import _lane_deck_pairs

    learner = tuple(f"learner-{index}" for index in range(8))
    opponent = tuple(f"opponent-{index}" for index in range(8))

    assert _lane_deck_pairs(0, learner, (opponent,)) == ((learner, opponent),)
    assert _lane_deck_pairs(1, learner, (opponent,)) == ((opponent, learner),)
