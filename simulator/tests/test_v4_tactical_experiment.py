"""Experiment gates and protocol isolation, without running training games."""
from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from simulator.rl.v4 import tactical_experiment as experiment
from simulator.rl.v4.tactical_training import TacticalTrainingConfig


GATE = dict(minimum_net_wins=3, minimum_lavaloon_net_wins=1,
            reject_any_family_delta_at_or_below=-3, minimum_bait_net_wins=0,
            minimum_xbow_net_wins=0, minimum_dev_net_wins=0)


def comparison(net=5, **families):
    return dict(pairs=120, net_wins=net, favorable=8, unfavorable=3,
                family_net=dict(lavaloon=3, **families))


def test_overall_gain_cannot_override_bait_xbow_or_development_regression():
    assert experiment.qualification(comparison(), GATE, development=comparison(0))["eligible"]
    for family in ("bait", "xbow"):
        result = experiment.qualification(comparison(**{family: -1}), GATE)
        assert not result["eligible"]
        assert f"{family}_retention" in result["failures"]
    result = experiment.qualification(comparison(), GATE, development=comparison(-1))
    assert result["failures"] == ["held_out_style_regression"]
    result = experiment.qualification(comparison(golem=-3), GATE)
    assert "family_regression:golem" in result["failures"]


def test_other_family_gains_do_not_substitute_for_required_air_gain():
    row = comparison()
    row["family_net"]["lavaloon"] = 0
    assert "lavaloon_gain" in experiment.qualification(row, GATE)["failures"]


def test_combined_results_preserve_signed_family_deltas_and_counts():
    first, second = comparison(4, bait=-1), comparison(-2, bait=1)
    result = experiment._combine((first, second))
    assert result["pairs"] == 240
    assert result["net_wins"] == 2
    assert result["family_net"] == dict(lavaloon=6, bait=0)
    assert result["mean_score_delta"] == pytest.approx(2 / 240)


@pytest.fixture
def fake_preparation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"pinned actor")
    receipt = tmp_path / "actor.json"
    receipt.write_text(json.dumps(dict(checkpoint=dict(path=str(checkpoint),
        sha256=experiment.file_sha(checkpoint), model_config=dict(dropout=0.0)))))
    source = dict(files={"simulator/rl/v4/model_v4.py": "sourcehash"}, sha256="sourcehash")
    monkeypatch.setattr(experiment, "runtime_source_fingerprint", lambda: source)
    return tmp_path / "run", receipt, source


def test_prepare_seals_complete_shared_grid_and_disjoint_calibration(fake_preparation):
    out, receipt, source = fake_preparation
    plan = experiment.prepare_experiment(out=out, actor_json=receipt, training=TacticalTrainingConfig())
    assert len(plan["jobs"]) == 120
    train = [job for job in plan["jobs"] if job["split"] == "train"]
    calibration = [job for job in plan["jobs"] if job["split"] == "calibration"]
    assert len(train) == 90 and len(calibration) == 30
    assert len({job["seed"] for job in plan["jobs"]}) == 120
    assert {job["opponent"]["deck_id"] for job in train} == set(plan["train_decks"])
    assert plan["source"] == source
    assert experiment.load_plan(out) == plan
    assert experiment.prepare_experiment(out=out, actor_json=receipt, training=TacticalTrainingConfig()) == plan


def test_source_drift_and_mutated_plan_fail_closed(fake_preparation, monkeypatch):
    out, receipt, _ = fake_preparation
    experiment.prepare_experiment(out=out, actor_json=receipt, training=TacticalTrainingConfig())
    monkeypatch.setattr(experiment, "runtime_source_fingerprint", lambda: dict(sha256="changed"))
    with pytest.raises(ValueError, match="source changed"):
        experiment.load_plan(out)
    plan = json.loads((out / "plan.json").read_text())
    plan["training"]["epochs"] = 99
    (out / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="seal differs"):
        experiment.load_plan(out)


def test_smoke_subset_cannot_be_a_full_experiment(fake_preparation):
    out, receipt, _ = fake_preparation
    with pytest.raises(ValueError, match="subset"):
        experiment.prepare_experiment(out=out, actor_json=receipt, training=TacticalTrainingConfig(), deck_ids=["lavaloon-lightning"])
    with pytest.raises(ValueError, match="control"):
        experiment.prepare_experiment(out=out, actor_json=receipt, training=TacticalTrainingConfig(), arms=["outcomes"])
    plan = experiment.prepare_experiment(out=out, actor_json=receipt,
        training=TacticalTrainingConfig(epochs=1), smoke=True)
    assert plan["smoke"]
    assert plan["train_decks"] == ["lavaloon-lightning"]
    assert len(plan["jobs"]) == 4


def test_fresh_status_does_not_claim_completion(tmp_path):
    result = experiment.status(tmp_path)
    assert result["complete"] is False
    assert result["fits"] == {}


def test_expected_dose_uses_declared_jobs_not_completed_record_order():
    manifest = dict(jobs=[dict(job_id="a", split="train"), dict(job_id="b", split="train"), dict(job_id="c", split="calibration")],
        completed=dict(c=dict(split="calibration", decisions=100), b=dict(split="train", decisions=32), a=dict(split="train", decisions=65)))
    assert experiment.expected_dose(manifest, asdict(TacticalTrainingConfig())) == (388, 12)


def test_corpus_from_another_plan_is_rejected_even_if_self_consistent(tmp_path, monkeypatch):
    from simulator.rl.v4 import tactical_data
    monkeypatch.setattr(tactical_data, "validate_manifest", lambda *args, **kwargs: None)
    plan = dict(plan_sha256="expected", checkpoint={}, source=dict(sha256="source"), jobs=[])
    folder = tmp_path / "dataset"
    folder.mkdir()
    manifest = dict(identity=dict(caller=dict(plan_sha256="another", source_sha256="source"),
        config=asdict(tactical_data.CollectionConfig())), jobs=[], completed={})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different experiment"):
        experiment.validate_collection(tmp_path, plan)


@pytest.mark.parametrize("arguments", [["--epochs", "1"], ["--epoch", "1"], ["--smoke"], ["--arms", "control", "plans"]])
def test_explicit_resume_option_conflicts_are_rejected(tmp_path, monkeypatch, arguments):
    from scripts.run_v4_tactical_learning import main
    plan = dict(training=dict(epochs=4), arms=["control", "outcomes"], workers=3,
                fit_workers=3, smoke=False, train_decks=["lavaloon-lightning"])
    monkeypatch.setattr(experiment, "load_plan", lambda out: plan)
    with pytest.raises(SystemExit) as error:
        main(["--out", str(tmp_path), "--stage", "fit", "--resume", *arguments])
    assert error.value.code == 2


def test_fit_artifact_grid_and_epoch_are_verified_before_evaluation(tmp_path, monkeypatch):
    from simulator.rl.v4 import tactical_training
    config = asdict(TacticalTrainingConfig())
    plan = dict(plan_sha256="plan", checkpoint=dict(path="anchor", sha256="hash", model_config={}),
        source=dict(sha256="source"), arms=["control"], training=config, smoke=False)
    manifest = dict(jobs=[dict(job_id="a", split="train")],
                    completed=dict(a=dict(split="train", decisions=32)))
    monkeypatch.setattr(experiment, "validate_collection", lambda *args: manifest)
    monkeypatch.setattr(experiment, "teacher_screen", lambda *args: dict(qualification=dict(eligible=True)))
    (tmp_path / "dataset").mkdir()
    data_path = tmp_path / "dataset/manifest.json"
    data_path.write_text("{}")
    folder = tmp_path / "fit_control"
    folder.mkdir()
    actor_path = folder / "actor_epoch_004.pt"
    actor_path.write_bytes(b"actor")
    protocol = dict(version=tactical_training.TRAINING_VERSION, arm="control", config=config, checkpoint=plan["checkpoint"],
        dataset=str((tmp_path / "dataset").resolve()), dataset_manifest_sha256=experiment.file_sha(data_path),
        source=plan["source"], inference="native-stochastic-fixed250ms-carry", final_epoch=4,
        curriculum="train-only bounded public-scenario supervision weights; every episode retained")
    run_hash = experiment.digest(protocol)
    experiment.atomic_json(folder / "protocol.json", dict(protocol=protocol, run_hash=run_hash))
    result = dict(status="complete", run_hash=run_hash, arm="control", final_epoch=4,
        checkpoint=dict(path=str(actor_path.resolve()), sha256=experiment.file_sha(actor_path), model_config=dict(fixed_wait=True)),
        total_training_frame_visits=128, total_optimizer_steps=4)
    experiment.atomic_json(folder / "result.json", result)
    summary = dict(plan_sha256="plan", results=dict(control=result), skipped={},
        artifact_seals=dict(control=dict(protocol_sha256=experiment.file_sha(folder / "protocol.json"),
                                       result_sha256=experiment.file_sha(folder / "result.json"))))
    experiment.atomic_json(tmp_path / "fits.json", summary)
    assert experiment.validate_fits(tmp_path, plan)["results"]["control"] == result
    # A self-consistent report still cannot substitute a different epoch.
    result["final_epoch"] = 3
    experiment.atomic_json(folder / "result.json", result)
    summary["artifact_seals"]["control"]["result_sha256"] = experiment.file_sha(folder / "result.json")
    experiment.atomic_json(tmp_path / "fits.json", summary)
    with pytest.raises(ValueError, match="candidate protocol/actor/dose"):
        experiment.validate_fits(tmp_path, plan)
