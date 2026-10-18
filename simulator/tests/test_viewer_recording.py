"""Viewer instrumentation must preserve simulator state and policy sampling."""

import json
from itertools import count
from types import SimpleNamespace

import pytest

from simulator.actions import PlayCardAction, WaitAction
from simulator.engine import BattleEngine
from simulator.ruleset import load_fixed_ruleset
from simulator.state import AreaEffectState, ProjectileState, StatusState
from simulator.viewer.projection import project_policy, project_state
from simulator.viewer.recording import JsonlRecorder, TrainingRecorder, write_progress


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_projection_preserves_coordinates_actions_and_detaches_state():
    ruleset = load_fixed_ruleset()
    state = BattleEngine(ruleset).new_battle(seed=17)
    tower = next(entity for entity in state.entities.values() if entity.owner == 0)
    tower.statuses.append(StatusState("freeze", 200_000))
    state.projectiles[101] = ProjectileState(
        uid=101, source_uid=tower.uid, source_card_id="fireball", owner=0,
        x_mtile=1250, y_mtile=24000, target_x_mtile=3000, target_y_mtile=5000,
        damage=1, crown_damage=1, speed_mtile_per_s=1000, radius_mtile=250,
    )
    state.effects[102] = AreaEffectState(
        uid=102, source_uid=None, source_card_id="poison", owner=1,
        x_mtile=7500, y_mtile=25500, radius_mtile=3500, remaining_us=2_500_000,
        tick_interval_us=1_000_000,
    )
    before = state.to_primitive(include_events=True)
    frame = project_state(state, ruleset, index=4, actions=(PlayCardAction(0, 1, (2, 20)),))
    display_tower = next(entity for entity in frame["entities"] if entity["uid"] == tower.uid)
    assert display_tower["x"] == tower.x_mtile / 1000
    assert display_tower["y"] > 16  # Player zero remains at the bottom.
    assert display_tower["radius"] > 0
    assert display_tower["statuses"] == ["freeze"]
    assert frame["projectiles"][0]["x"] == 1.25
    assert frame["projectiles"][0]["target_y"] == 5
    assert frame["effects"][0]["remaining_s"] == 2.5
    assert frame["actions"] == [{
        "kind": "play", "player": 0, "slot": 1, "col": 2, "row": 20,
        "card_id": state.players[0].hand[1],
    }]
    frame["players"][0]["hand"].clear()
    assert state.to_primitive(include_events=True) == before
    json.dumps(frame, allow_nan=False)


def test_policy_projection_masks_unavailable_cards_and_preserves_rng():
    torch = pytest.importorskip("torch")
    logits = SimpleNamespace(
        mode=torch.zeros(1, 1, 2), card=torch.zeros(1, 1, 4),
        placement=torch.zeros(1, 1, 4, 32, 18),
    )
    masks = SimpleNamespace(
        mode=torch.ones(1, 1, 2, dtype=torch.bool),
        card=torch.tensor([[[True, False, False, False]]]),
        placement=torch.zeros(1, 1, 4, 32, 18, dtype=torch.bool),
    )
    masks.placement[0, 0, 0, 20, 2] = True
    actions = SimpleNamespace(
        mode=torch.tensor([[1]]), card_slot=torch.tensor([[0]]),
        placement=torch.tensor([[[20, 2]]]),
    )
    before = torch.get_rng_state().clone()
    projected = project_policy(logits, masks, actions)
    assert projected["distribution"] == "masked"
    assert torch.equal(torch.get_rng_state(), before)
    assert projected["slot_probabilities"] == [1, 0, 0, 0]
    assert len(projected["heatmap"]) == 576
    assert projected["heatmap"][20 * 18 + 2] == 1
    assert sum(projected["heatmap"]) == 1
    assert projected["action_probability"] == 0.5
    # No legal PLAY anywhere: conditional card/cell probability is zero.
    masks.card.zero_()
    masks.placement.zero_()
    masks.mode[0, 0, 1] = False
    actions.mode.zero_()
    projected = project_policy(logits, masks, actions)
    assert projected["slot_probabilities"] == [0, 0, 0, 0]
    assert sum(projected["heatmap"]) == 0
    assert projected["wait_probability"] == 1


def test_recorder_lines_are_visible_before_finish_and_cannot_be_overwritten(tmp_path):
    path = tmp_path / "match.jsonl"
    recorder = JsonlRecorder(path, match_id="match", source="training", metadata={"seed": 1})
    recorder.append({"index": 0, "tick": 0})
    assert [row["type"] for row in read_lines(path)] == ["header", "frame"]
    recorder.finish("rollout_boundary")
    recorder.finish("terminated", 0)
    assert read_lines(path)[-1] == {"type": "end", "reason": "rollout_boundary", "winner": None}
    with pytest.raises(ValueError, match="finished"):
        recorder.append({})
    with pytest.raises(FileExistsError):
        JsonlRecorder(path, match_id="match", source="training", metadata={})


def test_episode_sampling_spans_rollouts_and_disabled_recording_is_a_noop(tmp_path, monkeypatch):
    from simulator.viewer import recording

    monkeypatch.setattr(recording, "_TRAINING_EPISODES", count())
    monkeypatch.setenv("CR_SIM_VIEWER_EVERY", "2")
    monkeypatch.delenv("CR_SIM_VIEWER_DIR", raising=False)
    disabled = TrainingRecorder(seed=1, episode_cap=4)
    disabled.start_episode(None, {}, 1)  # Disabled mode never reads an environment.
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("CR_SIM_VIEWER_DIR", str(tmp_path))
    engine = BattleEngine(load_fixed_ruleset())
    env = SimpleNamespace(engine=engine, state=engine.new_battle())
    for seed in range(3):
        recorder = TrainingRecorder(seed=seed, episode_cap=4)
        recorder.start_episode(env, {}, 1)
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_optional_recording_io_failure_does_not_fail_training(tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("occupied")
    monkeypatch.setenv("CR_SIM_VIEWER_DIR", str(blocked))
    monkeypatch.setenv("CR_SIM_VIEWER_EVERY", "1")
    engine = BattleEngine(load_fixed_ruleset())
    env = SimpleNamespace(engine=engine, state=engine.new_battle())
    recorder = TrainingRecorder(seed=1, episode_cap=4)
    with pytest.warns(RuntimeWarning, match="recording disabled"):
        recorder.start_episode(env, {}, 1)
    assert recorder.recorder is None
    recorder.before_step(None, (), None, None, None)
    recorder.after_step(None, 0, reason="episode_cap")


@pytest.mark.parametrize("reason", ["terminated", "episode_cap", "rollout_boundary", "environment_truncated"])
def test_training_recorder_keeps_boundary_reason(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("CR_SIM_VIEWER_DIR", str(tmp_path))
    engine = BattleEngine(load_fixed_ruleset())
    state = engine.new_battle()
    env = SimpleNamespace(state=state, engine=engine)
    recorder = TrainingRecorder(seed=0, episode_cap=1)
    recorder.start_episode(env, {"entry_id": "opening"}, 1)
    # A pending pre-action frame is independent from the resulting state.
    recorder.pending = project_state(state, engine.ruleset, actions=(WaitAction(0),))
    if reason == "terminated":
        state.terminal, state.winner = True, 0
    recorder.after_step(env, 1.0, reason=reason)
    rows = read_lines(next(tmp_path.glob("*.jsonl")))
    assert rows[0]["metadata"]["ruleset_hash"] == engine.ruleset.content_hash
    assert rows[1]["terminal"] is False
    assert rows[1]["reward"] == 1.0
    assert rows[2]["terminal"] == state.terminal
    assert rows[-1] == {"type": "end", "reason": reason, "winner": state.winner}


def test_training_recording_matches_unrecorded_rollout(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.frontier_ppo import collection
    from simulator.rl.v4.frontier_ppo.environment import make_frontier_env
    from simulator.rl.v4.model_v4 import RecurrentV4Policy, V4ValueHead

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = frontier_model_config()
        policy = RecurrentV4Policy(config)
        critic = V4ValueHead(config.gru_hidden_dim)
        initial = make_frontier_env()
        initial.reset_v2(seed=15)
        entries = [{
            "entry_id": "opening", "initial_state": initial.save_state(),
            "opponent": {"strategy": "deterministic-cycle"},
        }]
        environments = []

        def tracked_environment():
            env = make_frontier_env()
            environments.append(env)
            return env

        monkeypatch.setattr(collection, "make_frontier_env", tracked_environment)
        monkeypatch.delenv("CR_SIM_VIEWER_DIR", raising=False)
        torch.manual_seed(123)
        plain, plain_info = collection.collect_frontier_rollout(
            policy, critic, None, entries, n_decisions=3, n_lanes=1,
            episode_cap=2, seed=7, device="cpu",
        )
        plain_rng = torch.get_rng_state().clone()
        plain_state = environments[-1].state.to_primitive(include_events=True)
        monkeypatch.setenv("CR_SIM_VIEWER_DIR", str(tmp_path))
        torch.manual_seed(123)
        recorded, recorded_info = collection.collect_frontier_rollout(
            policy, critic, None, entries, n_decisions=3, n_lanes=1,
            episode_cap=2, seed=7, device="cpu",
        )
        assert torch.equal(torch.get_rng_state(), plain_rng)
        assert environments[-1].state.to_primitive(include_events=True) == plain_state
        assert recorded_info == plain_info
        for field in ("mode", "card_slot", "placement"):
            assert torch.equal(getattr(recorded.trajectory.actions, field), getattr(plain.trajectory.actions, field))
        assert torch.equal(recorded.trajectory.rewards, plain.trajectory.rewards)
        recordings = [read_lines(path) for path in tmp_path.glob("*.jsonl")]
        assert {rows[-1]["reason"] for rows in recordings} == {"episode_cap", "rollout_boundary"}
        assert sum(len([row for row in rows if row["type"] == "frame"]) for rows in recordings) == 5
    finally:
        torch.set_num_threads(previous_threads)


def test_progress_is_atomic_and_curve_publishes_completed_updates(tmp_path, monkeypatch):
    from simulator.rl.v4.learning_curve import LearningCurveConfig, run_learning_curve, smoke_hooks

    telemetry = tmp_path / "telemetry"
    monkeypatch.setenv("CR_SIM_VIEWER_DIR", str(telemetry))
    report = run_learning_curve(LearningCurveConfig(
        output_dir=tmp_path / "curve", smoke=True, update_schedule=(0, 1),
    ), hooks=smoke_hooks())
    progress = json.loads((telemetry / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["current_update"] == 1
    assert list(progress["updates"]) == ["1"]
    assert len(progress["points"]) == len(report["points"]) == 2
    assert not list(telemetry.glob("*.tmp"))
    write_progress(telemetry, {"status": "running", "current_update": 2})
    assert json.loads((telemetry / "progress.json").read_text())["current_update"] == 2


def test_distillation_epoch_progress_preserves_training_and_rng(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from simulator.rl.v4.search_distillation import training
    from simulator.viewer import recording

    anchor = torch.nn.Linear(1, 1, bias=False)
    records = [SimpleNamespace(is_correction=True)]
    monkeypatch.setattr(training, "_stack_records", lambda *args, **kwargs: {})
    monkeypatch.setattr(training, "compute_batch_loss", lambda student, *args, **kwargs: (
        (student.weight - 1).square().sum(), {},
    ))
    monkeypatch.setattr(training, "evaluate_records", lambda *args, **kwargs: {"n": 0})
    monkeypatch.delenv("CR_SIM_VIEWER_DIR", raising=False)
    options = dict(method="hard-none", seed=4, context=0, epochs=2, batch_size=1)
    plain = training.train_one(anchor, records, [], checkpoint_path=tmp_path / "plain.pt", **options)
    plain_rng = torch.get_rng_state().clone()
    telemetry = tmp_path / "telemetry"
    monkeypatch.setenv("CR_SIM_VIEWER_DIR", str(telemetry))
    progress = []
    original_publish = recording.publish_progress

    def publish(payload):
        original_publish(payload)
        progress.append(json.loads((telemetry / "progress.json").read_text()))

    monkeypatch.setattr(recording, "publish_progress", publish)
    observed = training.train_one(anchor, records, [], checkpoint_path=tmp_path / "observed.pt", **options)
    assert torch.equal(torch.get_rng_state(), plain_rng)
    assert observed["history"] == plain["history"]
    plain_state = torch.load(tmp_path / "plain.pt", weights_only=False)["state_dict"]
    observed_state = torch.load(tmp_path / "observed.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(value, observed_state[key]) for key, value in plain_state.items())
    epochs = [row for row in progress if row["phase"] == "epoch"]
    assert [row["current_epoch"] for row in epochs] == [1, 2]
    assert epochs[0]["status"] == "running"
    assert progress[-1]["status"] == "complete"
    assert progress[-1]["source"] == "distillation"
