"""Small deterministic sequences for the standalone simulator viewer."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
import time

import pytest

from sim_viewer.matches import MatchBusyError, MatchManager


def _finished(manager: MatchManager, match_id: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = manager.status(match_id)
        if status["state"] != "running":
            thread = manager._matches[match_id].thread
            if thread is not None:
                thread.join(timeout=1)
            return manager.status(match_id)
        time.sleep(0.005)
    raise AssertionError(f"viewer match did not finish: {manager.status(match_id)}")


def _frontier(path: Path, *, decisions: int = 3) -> tuple[dict, dict]:
    from simulator.actions import PlayCardAction, WaitAction
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset

    env = SimulatorEnv(BattleEngine(load_fixed_ruleset(), validate_every_tick=False))
    env.reset_v2(seed=7, decks=(PLAYER_DECK, PLAYER_DECK), shuffle_decks=True)
    rows = []
    for index in range(decisions):
        if index == 0:
            card = env.state.players[0].hand[0]
            cell = env.engine.legal_cells(env.state, 0, card)[0]
            own = PlayCardAction(0, 0, cell)
            learner = {"mode": 1, "slot": 0, "col": cell[0], "row": cell[1]}
        else:
            own = WaitAction(0)
            learner = {"mode": 0}
        rows.append({
            "state_hash": env.state.state_hash(), "tick": env.state.tick,
            "learner": learner, "opponent": {"kind": "wait"},
            "mode_p": [0.25, 0.75], "card_p": [0.25] * 4,
            "entropy": {"joint": 1.2},
        })
        env._step_core((own, WaitAction(1)))
    raw = {"config": {"match_seed": 7, "learner_deck": list(PLAYER_DECK), "opponent_deck": list(PLAYER_DECK)}, "decisions": rows}
    path.write_text(json.dumps(raw), encoding="utf-8")
    return raw, env.state.to_primitive()


def test_seeded_frontier_replay_keeps_actions_and_final_state(tmp_path: Path) -> None:
    path = tmp_path / "source_match.json"
    _, expected = _frontier(path)
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_replay(path)
        status = _finished(manager, match["id"])
        assert status["state"] == "complete", status
        assert status["frame_count"] == 4
        frames = manager.frames(match["id"])
        assert frames[0]["actions"][0]["kind"] == "play"
        assert frames[0]["policy"]["play_probability"] == 0.75
        assert frames[0]["policy"]["heatmap_available"] is False
        assert frames[0]["policy"]["distribution"] == "unmasked"
        assert frames[-1]["tick"] == expected["tick"]
        assert frames[-1]["time_s"] == expected["elapsed_us"] / 1_000_000
        assert frames[-1]["actions"] == []
        assert frames[-1]["players"][0]["hand"] == expected["players"][0]["hand"]
        assert [frame["index"] for frame in manager.frames(match["id"], after=1, limit=1)] == [2]
        recording = Path(status["metadata"]["recording_path"])
        rows = [json.loads(line) for line in recording.read_text().splitlines()]
        assert rows[0]["type"] == "header"
        assert rows[-1]["type"] == "end"
        replay = manager.start_replay(recording)
        assert replay["state"] == "complete"
        assert manager.frames(replay["id"]) == frames
    finally:
        manager.close()


def test_frontier_replay_checks_intermediate_boundary_hashes(tmp_path: Path) -> None:
    path = tmp_path / "source_match.json"
    raw, _ = _frontier(path)
    raw["decisions"][1]["state_hash"] = "bad-intermediate-hash"
    path.write_text(json.dumps(raw))
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_replay(path)
        status = _finished(manager, match["id"])
        assert status["state"] == "error"
        assert "decision 1" in status["error"]
        assert "original engine/ruleset revision" in status["error"]
        assert status["frame_count"] == 1
        assert manager.frames(match["id"])[0]["index"] == 0
    finally:
        manager.close()


def test_replay_reward_belongs_to_displayed_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simulator.env import SimulatorEnv

    path = tmp_path / "source_match.json"
    _frontier(path)
    original_step = SimulatorEnv._step_core

    def step(self, actions):
        _, terminated, truncated, info = original_step(self, actions)
        return (float(self.state.tick), 0.0), terminated, truncated, info

    monkeypatch.setattr(SimulatorEnv, "_step_core", step)
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_replay(path)
        assert _finished(manager, match["id"])["state"] == "complete"
        frames = manager.frames(match["id"])
        assert [frame["reward"] for frame in frames] == [5.0, 10.0, 15.0, None]
        assert [frame["tick"] for frame in frames] == [0, 5, 10, 15]
    finally:
        manager.close()


def test_frontier_replay_rejects_explicit_engine_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "source_match.json"
    raw, _ = _frontier(path, decisions=1)
    raw["engine_version"] = "obsolete-engine"
    path.write_text(json.dumps(raw))
    manager = MatchManager(tmp_path / "recordings")
    try:
        status = _finished(manager, manager.start_replay(path)["id"])
        assert status["state"] == "error"
        assert "engine_version is incompatible" in status["error"]
    finally:
        manager.close()


def test_jsonl_tail_preserves_unbounded_history_and_incomplete_line(tmp_path: Path) -> None:
    from simulator.viewer.recording import JsonlRecorder

    path = tmp_path / "training_sample.jsonl"
    recorder = JsonlRecorder(path, match_id="training-1", source="ppo", metadata={"seed": 17})
    for index in range(620):
        recorder.append({"index": index, "tick": index * 5, "time_s": index / 4})
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_replay(path)
        assert match["state"] == "running"
        assert match["frame_count"] == 620
        assert match["metadata"]["seed"] == 17
        assert manager.frames(match["id"], limit=1)[0]["index"] == 0
        partial = json.dumps({"type": "frame", "index": 620, "tick": 3100})
        with path.open("a") as stream:
            stream.write(partial[:20])
        assert manager.status(match["id"])["frame_count"] == 620
        with path.open("a") as stream:
            stream.write(partial[20:] + "\n")
        assert manager.frames(match["id"], after=619)[0]["index"] == 620
        recorder.finish("match_complete", winner=1)
        status = manager.status(match["id"])
        assert status["state"] == "complete"
        assert status["metadata"]["result"]["winner"] == 1
    finally:
        manager.close()


def test_jsonl_tail_does_not_consume_worker_and_stop_blocks_further_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simulator.viewer.recording import JsonlRecorder

    recording = tmp_path / "training_sample.jsonl"
    writer = JsonlRecorder(recording, match_id="training", source="ppo", metadata={})
    source = tmp_path / "source_match.json"
    _frontier(source, decisions=1)
    entered, release = threading.Event(), threading.Event()

    def work(self, match, raw):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(MatchManager, "_replay_frontier", work)
    manager = MatchManager(tmp_path / "recordings")
    try:
        tail = manager.start_replay(recording)
        active = manager.start_replay(source)
        assert entered.wait(timeout=2)
        with pytest.raises(MatchBusyError):
            manager.start_replay(source)
        second_tail = manager.start_replay(recording)
        assert second_tail["state"] == "running"
        assert manager.stop(tail["id"])["state"] == "stopped"
        writer.append({"index": 0, "tick": 0})
        assert manager.status(tail["id"])["frame_count"] == 0
        assert manager.status(second_tail["id"])["frame_count"] == 1
        assert manager.stop(active["id"])["state"] == "stopped"
    finally:
        release.set()
        manager.close()


def test_jsonl_truncation_is_reported(tmp_path: Path) -> None:
    from simulator.viewer.recording import JsonlRecorder

    path = tmp_path / "sample.jsonl"
    recorder = JsonlRecorder(path, match_id="sample", source="ppo", metadata={})
    recorder.append({"index": 0})
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_replay(path)
        path.write_text("")
        status = manager.status(match["id"])
        assert status["state"] == "error"
        assert "truncated" in status["error"]
    finally:
        manager.close()


def test_stale_unfinished_recording_reports_waiting_writer_and_can_resume(tmp_path: Path) -> None:
    from simulator.viewer.recording import JsonlRecorder

    path = tmp_path / "sample.jsonl"
    recorder = JsonlRecorder(path, match_id="sample", source="training", metadata={})
    stale = time.time() - 60
    os.utime(path, (stale, stale))
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_replay(path)
        assert match["state"] == "running"
        assert match["metadata"]["recording_unfinished"]
        assert not match["metadata"]["writer_active"]
        recorder.append({"index": 0})
        assert manager.status(match["id"])["metadata"]["writer_active"]
        recorder.finish("done")
        final = manager.status(match["id"])
        assert final["state"] == "complete"
        assert not final["metadata"]["recording_unfinished"]
        assert not final["metadata"]["writer_active"]
    finally:
        manager.close()


@pytest.mark.parametrize("rows,message", [
    ([{"type": "frame", "index": 0}], "schema header"),
    ([{"type": "header", "schema_version": 2}], "schema_version"),
    ([{"type": "header", "schema_version": 1, "match_id": "one", "metadata": {}}], "source"),
    ([{"type": "header", "schema_version": 1, "match_id": "one", "source": "training", "metadata": {}}, {"type": "end"}, {"type": "frame", "index": 0}], "after its end"),
])
def test_jsonl_rejects_invalid_schema_and_records_after_end(tmp_path: Path, rows: list, message: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manager = MatchManager(tmp_path / "recordings")
    try:
        status = manager.start_replay(path)
        assert status["state"] == "error"
        assert message in status["error"]
    finally:
        manager.close()


@pytest.fixture
def compact_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        config = ModelConfigV4(model_dim=8, spatial_channels=4, fused_dim=8, gru_hidden_dim=8, transformer_heads=2, transformer_layers=1, transformer_ff_dim=16, spatial_head_dim=2)
        actor = RecurrentV4Policy(config)
        checkpoint = tmp_path / "compact" / "actor.pt"
        checkpoint.parent.mkdir()
        torch.save(actor.state_dict(), checkpoint)
        (checkpoint.parent / "manifest.json").write_text(json.dumps({"model_config": asdict(config)}))
    try:
        yield checkpoint
    finally:
        torch.set_num_threads(previous_threads)


def test_exhibition_is_seeded_and_preserves_rng_and_model_config(tmp_path: Path, compact_checkpoint: Path) -> None:
    import torch

    manager = MatchManager(tmp_path / "recordings")
    original_rng = torch.random.get_rng_state().clone()
    histories = []
    try:
        for _ in range(2):
            match = manager.start_exhibition({"checkpoint": str(compact_checkpoint), "seed": 19, "max_decisions": 3})
            status = _finished(manager, match["id"])
            assert status["state"] == "complete", status
            assert status["frame_count"] == 4
            assert status["metadata"]["model_config"]["transformer_heads"] == 2
            assert status["metadata"]["model_config_source"].endswith("manifest.json")
            assert len(status["metadata"]["checkpoint_sha256"]) == 64
            frames = manager.frames(match["id"])
            assert len(frames[0]["policy"]["heatmap"]) == 576
            assert frames[-1]["policy"] is None
            histories.append(frames)
        assert histories[0] == histories[1]
        assert torch.equal(torch.random.get_rng_state(), original_rng)
    finally:
        manager.close()


def test_checkpoint_opponent_uses_its_own_public_view(tmp_path: Path, compact_checkpoint: Path) -> None:
    manager = MatchManager(tmp_path / "recordings")
    try:
        match = manager.start_exhibition({"checkpoint": str(compact_checkpoint), "opponent": "checkpoint", "opponent_checkpoint": str(compact_checkpoint), "max_decisions": 4, "seed": 4})
        status = _finished(manager, match["id"])
        assert status["state"] == "complete", status
        assert status["metadata"]["opponent_checkpoint_sha256"] == status["metadata"]["checkpoint_sha256"]
        frames = manager.frames(match["id"])
        assert all([action["player"] for action in frame["actions"]] == [0, 1] for frame in frames[:-1])
        assert not any(event["kind"] == "action_rejected" for frame in frames for event in frame["events"])
    finally:
        manager.close()


def test_exhibition_rejects_unsupported_learner_deck_before_start(tmp_path: Path) -> None:
    manager = MatchManager(tmp_path / "recordings")
    try:
        with pytest.raises(ValueError, match="Hog-cycle hand vocabulary"):
            manager.start_exhibition({"checkpoint": "missing.pt", "learner_deck": ["hog-rider", "cannon", "musketeer", "skeletons", "ice-golem", "ice-spirit", "fireball", "knight"]})
    finally:
        manager.close()
