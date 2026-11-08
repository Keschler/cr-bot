"""Collection integrity, public boundary, and actual-execution plan semantics."""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rl.v4 import tactical_data as data
from simulator.rl.v4.bootstrap import catalog_training_opponent_specs
from simulator.rl.v4.simulator_teacher import TeacherTarget


def test_default_grid_has_balanced_styles_drivers_and_disjoint_calibration():
    jobs = data.build_collection_jobs(catalog_training_opponent_specs(), 19)
    train = [job for job in jobs if job.split == "train"]
    calibration = [job for job in jobs if job.split == "calibration"]
    assert len(train) == 90 and len(calibration) == 30
    assert Counter(job.driver for job in train) == dict.fromkeys(data.DRIVERS, 30)
    assert Counter(job.driver for job in calibration) == dict.fromkeys(data.DRIVERS, 10)
    assert Counter(job.opponent["style"] for job in train) == {"tempo": 30, "patient": 30, "counterpush": 30}
    assert Counter(job.opponent["style"] for job in calibration) == {"tempo": 10, "patient": 10, "counterpush": 10}
    assert len({job.seed for job in jobs}) == 120
    for opponent in catalog_training_opponent_specs():
        rows = [job for job in train if job.opponent["deck_id"] == opponent.deck_id]
        assert {job.driver for job in rows} == set(data.DRIVERS)
        assert {job.opponent["style"] for job in rows} == {"tempo", "patient", "counterpush"}
    assert [asdict(job) for job in jobs] == [asdict(job) for job in data.build_collection_jobs(catalog_training_opponent_specs(), 19)]


def test_small_pilot_and_reserved_or_development_identity_are_explicit():
    jobs = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 2, train_matches=1, calibration_matches=1)
    assert len(jobs) == 2 and jobs[0].seed != jobs[1].seed
    for override in ({"style": "adaptive"}, {"split": "eval"}, {"deck_id": "not-a-training-deck"},
                     {"style_config": {"reserve_milli": 0}}, {"cards": ["hog-rider"] * 8}):
        with pytest.raises(ValueError):
            replace(jobs[0], opponent={**jobs[0].opponent, **override})
    with pytest.raises(ValueError):
        data.CollectionConfig(decision_interval_us=500_000)
    with pytest.raises(ValueError):
        data.CollectionConfig(max_decisions=1441)


def _public_inputs(step=0):
    values = {
        "raster": np.full((21, 32, 18), np.float32(.123456 + step / 100), np.float32),
        "global_features": np.zeros(768, np.float32), "entities": np.zeros((128, 32), np.float32),
        "entity_mask": np.zeros(128, bool), "hand_tokens": np.zeros((4, 16), np.float32),
        "opp_hand_probs": np.full(128, 1 / 128, np.float32), "opp_out_of_cycle": np.zeros(128, bool),
        "opp_elixir_interval": np.asarray((1, 9), np.float32), "event_history": np.zeros((16, 8), np.float32),
        "legal_play": np.ones((4, 32, 18), bool), "legal_wait": True,
    }
    values["global_features"][12:18] = (1 - step * .01, 1, 1, 1 - step * .02, 1, 1)
    return values


def test_public_measurements_are_weighted_public_hp_and_survival_is_unknown():
    values = _public_inputs()
    values["global_features"][12:18] = (1, .5, 1, 0, 1, 0)
    measured = data.public_measurement(values, 250_000, (3000, 5000, 3000))
    assert measured.match_seconds == .25
    assert measured.own_tower_health == pytest.approx(8500 / 11000)
    assert measured.enemy_tower_health == pytest.approx(5000 / 11000)
    assert measured.entities == () and measured.entities_complete is False


def test_active_mask_excludes_dead_and_friendly_xbow():
    from cr_bot.domain.card_metadata import CARD_METADATA
    values = _public_inputs()
    row = values["entities"][0]
    row[0] = CARD_METADATA["x-bow"]["id"] / max(value["id"] for value in CARD_METADATA.values())
    row[[1, 4, 6, 9, 10]] = 1
    row[3] = .2
    values["entity_mask"][0] = True
    assert data.active_public_mask(values) == (False, True)
    row[4] = 0
    assert data.active_public_mask(values) == (False, False)
    row[4], row[1] = 1, 0
    assert data.active_public_mask(values) == (False, False)
    row[1], row[5], row[6], row[3] = 1, 1, 0, .6
    assert data.active_public_mask(values)[0]


@pytest.fixture
def native_boundary(monkeypatch, tmp_path):
    """A three-decision real collection loop with a tiny environment boundary."""
    from simulator import actions, engine, env, public_state_estimator
    from simulator.rl.v4 import frontier_eval, simulator_teacher, tactical_plans
    from simulator.rl.v4.v4_ppo import runtime
    import torch

    class Event(dict):
        def __init__(self, kind, **kwargs):
            super().__init__(kwargs)
            self.kind = kind

    class TinyEnv:
        def __init__(self, bound_engine, *, decision_interval_us):
            assert decision_interval_us == 250_000
            self.engine = bound_engine
            self.state = SimpleNamespace(elapsed_us=0, events=[], terminal=False)
            self.state.state_hash = lambda: f"state-{self.state.elapsed_us}"

        def reset_v2(self, **kwargs):
            self.reset = kwargs

        def step_v2_for_viewer(self, requested, *, viewer):
            if isinstance(requested[0], actions.PlayCardAction):
                self.state.events.append(Event("card_played", player=0, card_id="hog-rider"))
            self.state.elapsed_us += 250_000
            terminal = self.state.elapsed_us == 750_000
            self.state.terminal = terminal
            values = _public_inputs(self.state.elapsed_us // 250_000)
            return SimpleNamespace(terminated=terminal, truncated=False,
                                   observations=(SimpleNamespace(global_vector=values["global_features"]), None),
                                   info={"winner": 0, "terminal_reason": "test-boundary"})

    instances = []

    class TinyPlans:
        def __init__(self):
            self.calls = []
            self.acknowledged = []
            self.last_decision = None
            self.last_source = "cycle"
            self.counters = tactical_plans.PlanCounters()
            instances.append(self)

        def label(self, *, decision_step, **kwargs):
            self.calls.append(decision_step)
            # A hypothetical second setup label differs; fresh instances must
            # never invent this followup on actor/cycle trajectories.
            return TeacherTarget(1, 0, 1 if len(self.calls) > 1 else 0, 0, 0)

        def acknowledge(self, executed_target, *, accepted=True):
            self.acknowledged.append((executed_target, accepted))

    class TinyActor:
        def __init__(self):
            self.config = None
            self.resets, self.hidden_inputs, self.samples = [], [], 0

        def initial_hidden(self, count, *, device):
            return torch.zeros(1, count, 1)

        def forward_step(self, inputs, *, hidden, reset_mask):
            self.hidden_inputs.append(float(hidden[0, 0, 0]))
            self.resets.append(bool(reset_mask[0, 0]))
            logits = SimpleNamespace(**{name: torch.zeros(1) for name in ("mode", "wait_duration", "card", "placement")})
            return logits, None, hidden + 1

        def sample_action(self, logits, masks):
            self.samples += 1
            return self.samples, torch.zeros(1), None

    actor = TinyActor()
    monkeypatch.setattr(engine, "BattleEngine", lambda ruleset, **kwargs: SimpleNamespace(ruleset=ruleset))
    monkeypatch.setattr(env, "SimulatorEnv", TinyEnv)
    monkeypatch.setattr(public_state_estimator, "PublicStateEstimator", lambda: object())
    style_specs = []

    def make_opponent(spec, seed, *, env):
        style_specs.append(spec)
        return SimpleNamespace(choose_action=lambda *args: actions.WaitAction(1), identity={"style_id": spec["style"]})

    monkeypatch.setattr(frontier_eval, "make_opponent", make_opponent)
    monkeypatch.setattr(simulator_teacher, "cycle_teacher_label", lambda **kwargs: TeacherTarget(0, 0, 0, 0, 0))
    monkeypatch.setattr(tactical_plans, "PlanTeacherLabeler", TinyPlans)
    monkeypatch.setattr(runtime, "observe_v4_inputs", lambda env, estimator: (None, _public_inputs(env.state.elapsed_us // 250_000), ["hog-rider"] * 4, 10.))
    monkeypatch.setattr(runtime, "load_frozen_anchor", lambda *args, **kwargs: actor)
    monkeypatch.setattr(runtime, "configure_v4_runtime", lambda policy: None)
    monkeypatch.setattr(runtime, "decode_v4_sim_actions", lambda sampled, **kwargs: [actions.PlayCardAction(0, 0, (0, 0))])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"pinned test actor")
    record = {"path": str(checkpoint), "sha256": data.sha256_file(checkpoint)}
    return instances, actor, record, style_specs


@pytest.mark.parametrize("driver", data.DRIVERS)
def test_collector_retains_native_chronology_and_actual_plan_progress(native_boundary, tmp_path, driver):
    instances, actor, checkpoint, style_specs = native_boundary
    jobs = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=(driver,), calibration_matches=0)
    identity = {"source_sha256": "unit-test", "model_config": {}}
    record = data.collect_episode(jobs[0], checkpoint, tmp_path / "episodes-output", identity)
    arrays, metadata = data.load_episode(tmp_path / "episodes-output" / record["path"], expected_sha=record["sha256"])
    assert len(arrays["raster"]) == 3
    assert arrays["raster"].dtype == np.float32
    assert arrays["raster"][0, 0, 0, 0] == _public_inputs()["raster"][0, 0, 0]
    assert arrays["elapsed_us"].tolist() == [0, 250_000, 500_000]
    assert arrays["reset_mask"].tolist() == [True, False, False]
    assert arrays["terminated_after"].tolist() == [False, False, True]
    assert arrays["measurement_seconds"].tolist() == [0., .25, .5, .75]
    assert arrays["outcome_mask"][:, :, :2].all()
    assert not arrays["outcome_mask"][:, :, 2:].any()
    assert not arrays["active"].any()  # All quiet rows were still retained.
    assert np.all(arrays["executed_actions"][:, 4] == 0)
    assert metadata["requested_learner_cards"] == metadata["accepted_learner_cards"]
    assert style_specs[0]["style"] == jobs[0].opponent["style"]
    if driver == "plans":
        assert len(instances) == 1
        assert instances[0].calls == [0, 1, 2]
        assert len(instances[0].acknowledged) == 3
        assert all(accepted for _, accepted in instances[0].acknowledged)
        assert arrays["targets_plans"][:, 2].tolist() == [0, 1, 1]
    else:
        assert len(instances) == 3
        assert all(len(instance.calls) == 1 and not instance.acknowledged for instance in instances)
        assert arrays["targets_plans"][:, 2].tolist() == [0, 0, 0]
    if driver == "actor":
        assert actor.hidden_inputs == [0., 1., 2.] and actor.resets == [True, False, False]
        assert actor.samples == 3


def test_censored_episode_leaves_no_accepted_artifact(native_boundary, tmp_path):
    _, _, checkpoint, _ = native_boundary
    job = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=("cycle",), calibration_matches=0)[0]
    output = tmp_path / "censored"
    with pytest.raises(RuntimeError, match="genuinely terminate"):
        data.collect_episode(job, checkpoint, output, {}, config=data.CollectionConfig(max_decisions=2))
    assert list((output / "episodes").iterdir()) == []


def test_resume_fails_closed_on_identity_checkpoint_or_durable_record_change(native_boundary, tmp_path):
    _, _, checkpoint, _ = native_boundary
    jobs = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=("cycle",), calibration_matches=0)
    source = tmp_path / "source.py"
    source.write_text("source = 1\n")
    identity = {"source_files": {str(source): data.sha256_file(source)}}
    output = tmp_path / "dataset"
    first = data.collect_dataset(jobs, checkpoint, output, identity)
    assert first["status"] == "complete"
    second = data.collect_dataset(jobs, checkpoint, output, identity, resume=True)
    assert first == second
    with pytest.raises(ValueError, match="identity/source/checkpoint/job grid"):
        data.collect_dataset(jobs, checkpoint, output, {**identity, "changed": True}, resume=True)
    source.write_text("source = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        data.collect_dataset(jobs, checkpoint, output, identity, resume=True)
    source.write_text("source = 1\n")
    path = output / first["completed"][jobs[0].job_id]["path"]
    record_path = path.with_suffix(".json")
    record = json.loads(record_path.read_text())
    record["driver"] = "actor"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="record and NPZ"):
        data.collect_dataset(jobs, checkpoint, output, identity, resume=True)
    checkpoint_path = checkpoint["path"]
    with open(checkpoint_path, "ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        data.collect_dataset(jobs, checkpoint, output, identity, resume=True)


def test_content_and_float_precision_changes_fail_validation(native_boundary, tmp_path):
    _, _, checkpoint, _ = native_boundary
    job = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=("cycle",), calibration_matches=0)[0]
    output = tmp_path / "dataset"
    record = data.collect_episode(job, checkpoint, output, {})
    arrays, metadata = data.load_episode(output / record["path"])
    modified = {**arrays, "elapsed_us": arrays["elapsed_us"][[0, 2, 1]]}
    with pytest.raises(ValueError, match="chronology"):
        data.validate_episode(modified, metadata)
    modified = {**arrays, "raster": arrays["raster"].astype(np.float16)}
    with pytest.raises(ValueError, match="float32"):
        data.validate_episode(modified, metadata)
    modified = {**arrays, "outcome_values": arrays["outcome_values"].copy()}
    modified["outcome_values"][0, 0, 0] = .9
    with pytest.raises(ValueError, match="consequence targets"):
        data.validate_episode(modified, metadata)


def test_legacy_teacher_wait_hint_is_normalized_to_native_one_step():
    assert data.target_array(TeacherTarget(0, 2, 8, 9, 3)).tolist() == [0, 0, 0, 0, 0]
    assert data.target_array(TeacherTarget(1, 2, 8, 9, 3)).tolist() == [1, 2, 8, 9, 0]


def test_validate_manifest_rejects_forged_origins_and_partial_coverage(native_boundary, tmp_path):
    from copy import deepcopy
    _, _, checkpoint, _ = native_boundary
    jobs = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 4, drivers=("cycle",), calibration_matches=0)
    manifest = data.collect_dataset(jobs, checkpoint, tmp_path / "manifest-check", {"source_sha256": "test"})
    data.validate_manifest(manifest, checkpoint)
    bad = deepcopy(manifest)
    bad["completed"] = {}
    with pytest.raises(ValueError, match="exact declared job grid"):
        data.validate_manifest(bad)
    bad = deepcopy(manifest)
    bad["completed"][jobs[0].job_id]["seed"] += 1
    with pytest.raises(ValueError, match="declared origin job"):
        data.validate_manifest(bad)
    bad = deepcopy(manifest)
    bad["identity"]["config"]["decision_interval_us"] = 500_000
    with pytest.raises(ValueError, match="identity hash"):
        data.validate_manifest(bad)
    with pytest.raises(ValueError, match="checkpoint/model config"):
        data.validate_manifest(manifest, {**checkpoint, "model_config": {"gru_hidden_dim": 7}})


def test_rejected_learner_play_never_becomes_a_dataset(native_boundary, monkeypatch, tmp_path):
    from simulator import env
    _, _, checkpoint, _ = native_boundary
    original = env.SimulatorEnv

    class RejectingEnv(original):
        def step_v2_for_viewer(self, actions, *, viewer):
            result = super().step_v2_for_viewer(actions, viewer=viewer)
            event = type(self.state.events[0])("action_rejected", player=0, reason="test rejection")
            self.state.events.append(event)
            return result

    monkeypatch.setattr(env, "SimulatorEnv", RejectingEnv)
    job = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=("plans",), calibration_matches=0)[0]
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="action rejected"):
        data.collect_episode(job, checkpoint, output, {})
    assert list((output / "episodes").iterdir()) == []


def test_unlabelled_prelude_is_retained_and_advances_actor_memory(native_boundary, monkeypatch, tmp_path):
    from simulator.rl.v4.v4_ppo import runtime
    _, actor, checkpoint, _ = native_boundary
    original = runtime.observe_v4_inputs

    def observe(env, estimator):
        observation, inputs, hand, elixir = original(env, estimator)
        if env.state.elapsed_us == 0:
            inputs["legal_play"][:] = False
            inputs["legal_wait"] = False
        return observation, inputs, hand, elixir

    monkeypatch.setattr(runtime, "observe_v4_inputs", observe)
    job = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=("actor",), calibration_matches=0)[0]
    output = tmp_path / "prelude"
    record = data.collect_episode(job, checkpoint, output, {})
    arrays, _ = data.load_episode(output / record["path"])
    assert arrays["label_valid"].tolist() == [False, True, True]
    assert np.all(arrays["targets_cycle"][0] == -1)
    assert np.all(arrays["targets_plans"][0] == -1)
    assert arrays["executed_actions"][0].tolist() == [0, 0, 0, 0, 0]
    assert actor.hidden_inputs == [0., 1., 2.] and actor.samples == 2


def test_worker_checks_wrapped_source_pins_before_publishing(native_boundary, monkeypatch, tmp_path):
    from simulator import env
    _, _, checkpoint, _ = native_boundary
    source = tmp_path / "pinned.py"
    source.write_text("original source\n")
    identity = {"source_files": {str(source): data.sha256_file(source)}}
    original = env.SimulatorEnv

    class MutatingEnv(original):
        def step_v2_for_viewer(self, actions, *, viewer):
            source.write_text("changed during collection\n")
            return super().step_v2_for_viewer(actions, viewer=viewer)

    monkeypatch.setattr(env, "SimulatorEnv", MutatingEnv)
    jobs = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 3, drivers=("cycle",), calibration_matches=0)
    output = tmp_path / "mutated-source"
    with pytest.raises(ValueError, match="source changed"):
        data.collect_dataset(jobs, checkpoint, output, identity)
    assert list((output / "episodes").iterdir()) == []
