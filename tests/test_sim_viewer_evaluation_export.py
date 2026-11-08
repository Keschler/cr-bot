"""Exact native evaluation export, publication, provenance, and cleanup."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim_viewer import export_evaluations as export


@pytest.fixture
def native_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from simulator.actions import WaitAction
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.rl.v4 import policy_development as dev
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from simulator.viewer import projection

    source = {"sha256": "sealed-source"}
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"synthetic actor")
    arm = dict(name="control", kind="actor", checkpoint=str(checkpoint),
               sha256=export._sha(checkpoint), model_config={"fixed_wait": True})
    opponent = dict(deck=list(PLAYER_DECK), deck_id="synthetic-bait", strategy="siege-bait", controller_seed=53)
    job = dict(arm=arm, opponent=opponent, seed=7, cell_id="synthetic-bait__adaptive__7")

    def make_env():
        env = SimulatorEnv(BattleEngine(load_fixed_ruleset(), validate_every_tick=False))
        env.reset_v2(seed=job["seed"], decks=(PLAYER_DECK, PLAYER_DECK), shuffle_decks=True)
        return env

    expected_env = make_env()
    initial_hash = expected_env.state.state_hash()
    for _ in range(3):
        expected_env.step_v2((WaitAction(0), WaitAction(1)))
    native = dict(deck_id=opponent["deck_id"], archetype="siege-bait", opponent=opponent,
                  seed=7, result="win", policy_seed=991, controller_seed=53,
                  learner_deck=list(PLAYER_DECK), decision_interval_us=250_000,
                  deterministic_policy=False, engine_version=expected_env.state.engine_version,
                  ruleset_hash=expected_env.state.ruleset_hash, decisions=3, winner=0,
                  terminal_reason="test_finished", initial_state_hash=initial_hash,
                  final_state_hash=expected_env.state.state_hash())
    row = dict(job=job, native_record=native, diagnostics={"decisions": 3, "waits": 3})
    identity = dict(source=source, arms=[arm])
    row["run_hash"] = dev.digest(identity)
    screen = tmp_path / "screen_train"
    path = screen / "control" / (job["cell_id"] + ".json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row))
    (screen / "manifest.json").write_text(json.dumps(dict(identity=identity, run_hash=row["run_hash"])))
    (screen / "raw_digests.json").write_text(json.dumps({str(path.relative_to(screen)): export._sha(path)}))
    monkeypatch.setattr(dev, "runtime_source_fingerprint", lambda: deepcopy(source))
    monkeypatch.setattr(dev, "validate_match", lambda row, job: None)

    class Actor:
        samples = 0

        def rollout_sample(self, *args, **kwargs):
            self.samples += 1
            return (self.samples, "sampled_action")

    actor = Actor()

    def load_actor(path, sha, config_json):
        assert path == arm["checkpoint"] and sha == arm["sha256"]
        assert json.loads(config_json) == {"fixed_wait": True}
        return actor

    monkeypatch.setattr(dev, "_actor", load_actor)
    monkeypatch.setattr(projection, "project_policy", lambda logits, masks, actions: {
        "sample": logits, "masks": masks, "actions": actions,
    })
    native_step = dev._TraceEnvironment.step_v2

    def step_reward(trace, actions):
        result = native_step(trace, actions)
        return SimpleNamespace(rewards=(float(trace.state.tick), 0), terminated=result.terminated, truncated=result.truncated)

    monkeypatch.setattr(dev._TraceEnvironment, "step_v2", step_reward)
    executions = []

    def execute(job):
        executions.append(deepcopy(job))
        trace = dev._TraceEnvironment(make_env())
        for _ in range(3):
            actor.rollout_sample(*([None] * 9 + ["original_masks"]), hidden="native-carried-hidden")
            trace.step_v2((WaitAction(0), WaitAction(1)))
        return dict(job=job, native_record=deepcopy(native), diagnostics=deepcopy(row["diagnostics"]))

    monkeypatch.setattr(dev, "execute_match", execute)
    return SimpleNamespace(path=path, out=tmp_path / "viewer", actor=actor, executions=executions,
                           row=row, dev=dev, execute=execute, original_step=step_reward, source=source)


def test_export_preserves_native_calls_pre_action_state_reward_and_final_frame(native_job) -> None:
    job = native_job
    original_rollout = job.actor.rollout_sample
    receipt = export.export_record(job.path, job.out)
    assert receipt["exact_verified"] is True and receipt["reused"] is False
    assert job.executions == [job.row["job"]]
    assert job.actor.samples == 3
    assert job.actor.rollout_sample == original_rollout
    assert job.dev._TraceEnvironment.step_v2 is job.original_step
    recording = Path(receipt["recording_path"])
    rows = [json.loads(line) for line in recording.read_text().splitlines()]
    assert rows[0]["source"] == "evaluation"
    assert rows[0]["metadata"]["evaluation"]["exact_verified"] is True
    assert rows[0]["metadata"]["policy_seed"] == 991
    assert rows[0]["metadata"]["model_config"] == {"fixed_wait": True}
    assert rows[-1] == {"type": "end", "reason": "test_finished", "winner": 0}
    frames = rows[1:-1]
    assert [f["index"] for f in frames] == [0, 1, 2, 3]
    assert [f["tick"] for f in frames] == [0, 5, 10, 15]
    assert [f["reward"] for f in frames] == [5.0, 10.0, 15.0, None]
    assert [f["policy"]["sample"] for f in frames[:-1]] == [1, 2, 3]
    assert frames[-1]["actions"] == [] and frames[-1]["policy"] is None
    assert receipt["recording_sha256"] == export._sha(recording)
    assert "screen_train__control__synthetic-bait__adaptive__7__win" in recording.name
    reused = export.export_record(job.path, job.out)
    assert reused["reused"] is True
    assert len(job.executions) == 1 and job.actor.samples == 3


@pytest.mark.parametrize("section", ["native_record", "diagnostics"])
def test_native_mismatch_publishes_nothing_and_restores_instrumentation(native_job, monkeypatch, section) -> None:
    job = native_job
    original_rollout = job.actor.rollout_sample

    def mismatch(saved):
        row = job.execute(saved)
        row[section]["changed"] = True
        return row

    monkeypatch.setattr(job.dev, "execute_match", mismatch)
    with pytest.raises(ValueError, match=f"native replay differs.*{section}"):
        export.export_record(job.path, job.out)
    assert list(job.out.iterdir()) == []
    assert job.actor.rollout_sample == original_rollout
    assert job.dev._TraceEnvironment.step_v2 is job.original_step


def test_failed_execution_removes_staging_file_and_restores_hooks(native_job, monkeypatch) -> None:
    job = native_job
    original_rollout = job.actor.rollout_sample

    def interrupted(saved):
        job.execute(saved)
        raise RuntimeError("interrupted native replay")

    monkeypatch.setattr(job.dev, "execute_match", interrupted)
    with pytest.raises(RuntimeError, match="interrupted native replay"):
        export.export_record(job.path, job.out)
    assert list(job.out.iterdir()) == []
    assert job.actor.rollout_sample == original_rollout
    assert job.dev._TraceEnvironment.step_v2 is job.original_step


def test_cache_rejects_changed_recording_and_unverified_receipt(native_job) -> None:
    job = native_job
    receipt = export.export_record(job.path, job.out)
    recording = Path(receipt["recording_path"])
    recording.write_text(recording.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="provenance or recording hash differs"):
        export.export_record(job.path, job.out)
    assert len(job.executions) == 1


def test_source_mismatch_and_raw_tampering_fail_before_sampling(native_job, monkeypatch) -> None:
    job = native_job
    monkeypatch.setattr(job.dev, "runtime_source_fingerprint", lambda: {"sha256": "new-source"})
    with pytest.raises(ValueError, match="native source seal differs"):
        export.export_record(job.path, job.out)
    assert not job.executions and job.actor.samples == 0
    monkeypatch.setattr(job.dev, "runtime_source_fingerprint", lambda: job.source)
    job.path.write_text(job.path.read_text() + "\n")
    with pytest.raises(ValueError, match="raw match digest seal differs"):
        export.export_record(job.path, job.out)
    assert not job.executions


def test_receipt_publication_failure_removes_recording(native_job, monkeypatch) -> None:
    def fail(*args):
        raise OSError("receipt unavailable")

    monkeypatch.setattr(export, "_atomic_json", fail)
    with pytest.raises(OSError, match="receipt unavailable"):
        export.export_record(native_job.path, native_job.out)
    assert list(native_job.out.iterdir()) == []


def test_collect_combines_screens_and_individual_files_without_duplicates(native_job) -> None:
    assert export.collect_records([native_job.path], [native_job.path.parent]) == [native_job.path]
    with pytest.raises(ValueError, match="provide raw match JSON"):
        export.collect_records([], [])


def test_manifest_reports_only_verified_matches(native_job, monkeypatch) -> None:
    monkeypatch.setattr(export, "_worker_init", lambda: None)
    manifest = export.export_records([native_job.path], native_job.out, workers=1)
    assert manifest["status"] == "complete" and manifest["count"] == 1
    assert manifest["matches"][0]["exact_verified"] is True
    assert json.loads((native_job.out / "manifest.json").read_text()) == manifest


def test_failed_batch_manifest_does_not_claim_completion(native_job, monkeypatch) -> None:
    monkeypatch.setattr(export, "_worker_init", lambda: None)

    def mismatch(saved):
        row = native_job.execute(saved)
        row["native_record"]["changed"] = True
        return row

    monkeypatch.setattr(native_job.dev, "execute_match", mismatch)
    with pytest.raises(ValueError, match="native replay differs"):
        export.export_records([native_job.path], native_job.out, workers=1)
    manifest = json.loads((native_job.out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["verified_count"] == 0
    assert list(native_job.out.glob("*.jsonl")) == []
