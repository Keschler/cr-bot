"""Certified alternatives, complete recurrent chronology and actor retention."""
from dataclasses import asdict
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4 import correction_training as training
from simulator.rl.v4 import tactical_data as data
from simulator.rl.v4.bootstrap import catalog_training_opponent_specs
from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy
from simulator.rl.v4.tactical_outcomes import PublicTacticalMeasurement, build_tactical_outcome_targets
from simulator.rl.v4.tactical_training import chunk_batch, differentiable_unroll, tactical_batch_loss


def episode(length=5):
    specs = dict(raster=(21, 32, 18), global_features=(768,), entities=(128, 32),
                 entity_mask=(128,), hand_tokens=(4, 16), opp_hand_probs=(128,),
                 opp_out_of_cycle=(128,), opp_elixir_interval=(2,), event_history=(16, 8))
    arrays = {key: np.zeros((length, *shape), bool if key in data.BOOL_INPUT_KEYS else np.float32)
              for key, shape in specs.items()}
    arrays["raster"][:] = np.random.default_rng(8).uniform(0, 1, arrays["raster"].shape)
    arrays["hand_tokens"][..., 0] = np.array([49, 72, 96, 114]) / 127
    arrays["global_features"][:, 12:18] = 1
    arrays["legal_play"] = np.zeros((length, 4, 32, 18), bool)
    arrays["legal_play"][1:, 0, 22, 5] = True
    arrays["legal_wait"] = np.arange(length) > 0
    arrays["decision"] = np.arange(length)
    arrays["elapsed_us"] = np.arange(length) * 250_000
    arrays["reset_mask"] = np.arange(length) == 0
    arrays["label_valid"] = np.arange(length) > 0
    for name in ("active", "air_threat", "xbow_present", "truncated_after"):
        arrays[name] = np.zeros(length, bool)
    arrays["terminated_after"] = np.arange(length) == length - 1
    for name in ("targets_cycle", "targets_plans", "executed_actions", "opponent_actions"):
        arrays[name] = np.zeros((length, 5), np.int64)
    arrays["targets_cycle"][0] = arrays["targets_plans"][0] = -1
    arrays["requested_card"] = arrays["accepted_card"] = np.full(length, "", dtype="U1")
    arrays["state_hash"] = np.array([f"state-{index}" for index in range(length)])
    measurements = [PublicTacticalMeasurement(index * .25, 1., 1., terminated=index == length)
                    for index in range(length + 1)]
    outcomes = build_tactical_outcome_targets(measurements, list(range(length)))
    arrays.update(outcome_values=outcomes.values, outcome_mask=outcomes.mask,
                  measurement_seconds=np.arange(length + 1, dtype=np.float64) * .25,
                  measurement_tower_health=np.ones((length + 1, 2)),
                  measurement_terminated=np.arange(length + 1) == length,
                  measurement_truncated=np.zeros(length + 1, bool))
    return arrays


def compact_config():
    return ModelConfigV4(model_dim=8, spatial_channels=4, fused_dim=8, gru_hidden_dim=8,
        transformer_heads=2, transformer_layers=1, transformer_ff_dim=16, spatial_head_dim=4, fixed_wait=True)


def verdict(state_hash):
    result = dict(state_hash=state_hash, discovery_replica_indices=[0, 1], confirmation_replica_indices=[2, 3], branches=[
        dict(source="sim", mode=0, slot=0, row=0, col=0, discovery_values=[0., 0.], confirmation_values=[0., 0.]),
        dict(source="repair", mode=1, slot=0, row=22, col=5, discovery_values=[1., 1.], confirmation_values=[1., 1.]),
    ])
    for branch in result["branches"]:
        branch["replicas"] = [dict(replica_index=index, replica_seed=900 + index, value=value)
                              for index, value in zip((0, 1), branch["discovery_values"], strict=True)]
        branch["confirmation_replicas"] = [dict(replica_index=index, replica_seed=900 + index, value=value)
                                           for index, value in zip((2, 3), branch["confirmation_values"], strict=True)]
    return result


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = compact_config()
    actor = tmp_path / "baseline.pt"
    torch.save(RecurrentV4Policy(config).state_dict(), actor)
    checkpoint = dict(path=str(actor), sha256=training.file_sha(actor), model_config=asdict(config))
    source = dict(sha256="fixture-source", files={})
    monkeypatch.setattr(training, "runtime_source_fingerprint", lambda: source)
    dataset = tmp_path / "dataset"
    (dataset / "episodes").mkdir(parents=True)
    jobs = data.build_collection_jobs(catalog_training_opponent_specs()[:1], 529, drivers=("actor",))
    identity = dict(schema=data.SCHEMA, caller=dict(source_sha256=source["sha256"]), checkpoint=checkpoint,
                    config=asdict(data.CollectionConfig()), jobs=[asdict(job) for job in jobs])
    identity["identity_sha256"] = data._identity_digest(identity)
    collection = dict(schema=data.SCHEMA, status="complete", identity=identity, identity_sha256=identity["identity_sha256"],
                      jobs=[asdict(job) for job in jobs], completed={})
    for job in jobs:
        arrays = episode()
        metadata = dict(**asdict(job), schema=data.SCHEMA, identity_sha256=identity["identity_sha256"],
            collection_identity=identity, checkpoint=checkpoint, config=identity["config"],
            deck_id=job.opponent["deck_id"], family=job.opponent["archetype"], style=job.opponent["style"],
            decisions=5, terminated=True, truncated=False, invalid_actions=[0, 0], card_plays=[0, 0],
            requested_learner_cards={}, accepted_learner_cards={},
            input_digest=data.array_digest(arrays, (*data.PUBLIC_KEYS, "reset_mask", "elapsed_us")),
            label_digest=data.array_digest(arrays, data.LABEL_KEYS), arrays_digest=data.array_digest(arrays, tuple(sorted(arrays))))
        path = dataset / "episodes" / f"{job.job_id}.npz"
        np.savez_compressed(path, **arrays, metadata_json=np.asarray(data.canonical(metadata)))
        collection["completed"][job.job_id] = dict(**metadata, path=str(path.relative_to(dataset)), sha256=training.file_sha(path))
    training.atomic_json(dataset / "manifest.json", collection)
    job = jobs[0]
    corrections = dict(schema=training.CORRECTIONS_SCHEMA, checkpoint=copy.deepcopy(checkpoint),
        dataset_manifest_sha256=training.file_sha(dataset / "manifest.json"), source_sha256=source["sha256"], min_lcb=.05,
        records=[dict(job_id=job.job_id, t=3, state_hash="state-3", target=[1, 0, 22, 5, 0],
                      source_episode_sha256=collection["completed"][job.job_id]["sha256"], accepted=True, verdict=verdict("state-3"))])
    labels = tmp_path / "corrections.json"
    training.atomic_json(labels, corrections)
    return dict(dataset=dataset, checkpoint=checkpoint, corrections_manifest=labels), corrections, collection


def test_ground_correction_overrides_air_gate_and_keeps_all_other_valid_rows():
    original = episode()
    original["active"][2] = original["air_threat"][2] = True
    overlay = training.correction_episode(original, {3: [1, 0, 22, 5, 0]})
    assert original["active"][2] and not original["active"][3]
    assert overlay["raster"] is original["raster"]
    assert overlay["executed_actions"] is original["executed_actions"]
    assert np.array_equal(overlay["active"], [False, False, False, True, False])
    batch = chunk_batch([overlay, training.correction_episode(episode(3), {})],
                        [dict(family="bait"), dict(family="bait")], start=0, length=5, arm="control")
    assert int(batch["real"].sum()) == 8
    assert int(batch["active"].sum()) == 1
    assert int(batch["inactive"].sum()) == 5
    assert not batch["valid"][:, 0].any()
    assert not batch["valid"][1, 3:].any()
    assert torch.equal(batch["factual_actions"].mode, torch.zeros(2, 5, dtype=torch.int64))


def test_ground_correction_carries_countdown_history_and_trains_recurrent_weights():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    policy = RecurrentV4Policy(compact_config())
    anchor = copy.deepcopy(policy)
    data_episode = training.correction_episode(episode(), {3: [1, 0, 22, 5, 0]})
    metadata = [dict(family="bait")]
    full = chunk_batch([data_episode], metadata, start=0, length=5, arm="control")
    logits, recurrent, _ = differentiable_unroll(policy, full, None, full["reset_mask"])
    first = chunk_batch([data_episode], metadata, start=0, length=2, arm="control")
    _, _, hidden = differentiable_unroll(policy, first, None, first["reset_mask"])
    second = chunk_batch([data_episode], metadata, start=2, length=3, arm="control")
    carried, _, _ = differentiable_unroll(policy, second, hidden.detach(), second["reset_mask"])
    assert torch.allclose(carried.mode, logits.mode[:, 2:], atol=1e-6)
    with torch.no_grad():
        old, _, _ = differentiable_unroll(anchor, full, None, full["reset_mask"])
    loss, details, _ = tactical_batch_loss(policy, logits, old, recurrent, full,
                                          config=training.CorrectionTrainingConfig().tactical_config())
    loss.backward()
    assert details["active_rows"] == 1 and details["inactive_rows"] == 3
    assert sum(float(parameter.grad.square().sum()) for name, parameter in policy.recurrent.named_parameters()
               if "weight_hh" in name and parameter.grad is not None) > 0
    assert policy.heads.duration_head.weight.grad is None


def test_verified_manifest_and_zero_label_calibration_are_valid(corpus):
    arguments, _, collection = corpus
    loaded, _, labels = training.validate_corrections(**arguments)
    assert training.canonical(loaded) == training.canonical(collection)
    assert sum(map(len, labels.values())) == 1
    assert labels[collection["jobs"][1]["job_id"]] == {}


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.update(source_sha256="changed"), "binding"),
    (lambda value: value.update(dataset_manifest_sha256="changed"), "binding"),
    (lambda value: value["checkpoint"].update(model_config={}), "binding"),
    (lambda value: value["records"][0].update(source_episode_sha256="changed"), "identity"),
    (lambda value: value["records"][0].update(state_hash="changed"), "identity"),
    (lambda value: value["records"][0].update(target=[0, 0, 0, 0, 0]), "frozen validated"),
    (lambda value: value["records"][0]["verdict"].update(confirmation_replica_indices=[1, 2]), "independently"),
    (lambda value: value["records"][0]["verdict"].update(confirmation_replica_indices=[2, 2]), "independently"),
    (lambda value: value["records"][0]["verdict"]["branches"][0].pop("replicas"), "actual replica"),
    (lambda value: value["records"][0]["verdict"]["branches"][0]["replicas"][0].update(replica_seed=901), "actual replica"),
    (lambda value: value["records"][0]["verdict"]["branches"][0]["replicas"][0].update(replica_index=1), "actual replica"),
    (lambda value: value["records"][0]["verdict"]["branches"][0]["confirmation_replicas"][0].update(replica_seed=900), "paired actual seeds"),
    (lambda value: value["records"][0]["verdict"]["branches"][0].update(mode=1), "factual recorded"),
    (lambda value: value["records"].append(copy.deepcopy(value["records"][0])), "duplicate"),
    (lambda value: value["records"][0].update(accepted=False), "no independently"),
])
def test_uncertified_or_misassociated_labels_fail_closed(corpus, mutation, match):
    arguments, corrections, _ = corpus
    mutation(corrections)
    training.atomic_json(arguments["corrections_manifest"], corrections)
    with pytest.raises(ValueError, match=match):
        training.validate_corrections(**arguments)


def test_full_history_fit_exports_plain_actor_and_seals_exact_resume(corpus, tmp_path):
    arguments, _, _ = corpus
    out = tmp_path / "fit"
    config = training.CorrectionTrainingConfig(epochs=1, chunk_length=2, learning_rate=1e-4)
    baseline_bytes = training.file_sha(arguments["checkpoint"]["path"])
    result = training.train_corrections(**arguments, out=out, config=config)
    assert result["total_training_frame_visits"] == 5
    assert result["total_optimizer_steps"] == 3
    assert result["history"][0]["active_rows"] == 1
    assert result["history"][0]["inactive_rows"] == 3
    assert result["calibration"]["correction_nll"] is None
    assert result["calibration"]["correction_rows"] == 0
    assert result["calibration"]["real_rows"] == 5
    assert result["calibration"]["inactive_rows"] == 4
    assert result["calibration"]["used_for_selection"] is False
    assert result["recurrent_weight_change"] > 0
    assert result["selected_actor_changed"] is False
    assert result["actor_parameter_updates"] is True
    assert result["total_focused_target_presentations"] == 0
    assert result["total_correction_presentations"] == 1
    assert result["diagnostics_before"]["train"]["correction_rows"] == 1
    factor = result["diagnostics_after"]["train"]["factors"][0]
    assert factor["before"] == result["diagnostics_before"]["train"]["factors"][0]["after"]
    assert factor["target"] == [1, 0, 22, 5, 0]
    assert factor["after"]["card_probability_given_play"] == 1
    assert factor["after"]["nearby_cell_probability_given_card"] == 1
    assert result["learning_gate"]["held_out_label_uptake"]["status"] == "unmeasured"
    assert result["learning_gate"]["eligible_for_fresh_games"] is False
    assert training.file_sha(arguments["checkpoint"]["path"]) == baseline_bytes
    weights = torch.load(result["checkpoint"]["path"], weights_only=True)
    assert set(weights) == set(RecurrentV4Policy(compact_config()).state_dict())
    assert training.train_corrections(**arguments, out=out, config=config, resume=True) == result
    with pytest.raises(ValueError, match="identical explicit resume"):
        training.train_corrections(**arguments, out=out, config=training.CorrectionTrainingConfig(epochs=2), resume=True)


def test_held_out_correction_labels_receive_no_gradients(corpus, tmp_path):
    arguments, corrections, collection = corpus
    config = training.CorrectionTrainingConfig(epochs=1, chunk_length=2, correction_exposures=2)
    first = training.train_corrections(**arguments, out=tmp_path / "without_calibration_labels", config=config)
    held_out = collection["jobs"][1]["job_id"]
    record = copy.deepcopy(corrections["records"][0])
    record.update(job_id=held_out, source_episode_sha256=collection["completed"][held_out]["sha256"])
    corrections["records"].append(record)
    training.atomic_json(arguments["corrections_manifest"], corrections)
    second = training.train_corrections(**arguments, out=tmp_path / "with_calibration_labels", config=config)
    assert second["accepted_calibration_corrections"] == second["calibration"]["correction_rows"] == 1
    assert second["calibration"]["correction_nll"] is not None
    old = torch.load(first["checkpoint"]["path"], weights_only=True)
    new = torch.load(second["checkpoint"]["path"], weights_only=True)
    assert all(torch.equal(old[key], new[key]) for key in old)


def test_self_consistent_replacement_actor_is_not_the_final_trained_actor(corpus, tmp_path):
    arguments, _, _ = corpus
    out = tmp_path / "fit"
    config = training.CorrectionTrainingConfig(epochs=1, chunk_length=2)
    result = training.train_corrections(**arguments, out=out, config=config)
    replacement = RecurrentV4Policy(compact_config()).state_dict()
    torch.save(replacement, result["checkpoint"]["path"])
    result["checkpoint"]["sha256"] = training.file_sha(result["checkpoint"]["path"])
    training.atomic_json(out / "result.json", result)
    with pytest.raises(ValueError, match="differs from its final trained state"):
        training.train_corrections(**arguments, out=out, config=config, resume=True)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(status="partial"),
    lambda value: value.update(final_epoch=0),
    lambda value: value.update(total_training_frame_visits=4),
    lambda value: value.update(total_optimizer_steps=2),
    lambda value: value["history"][0].update(real_rows=4),
    lambda value: value["checkpoint"].update(model_config={}),
    lambda value: value.update(total_focused_target_presentations=1),
    lambda value: value.update(total_correction_presentations=2),
    lambda value: value["history"][0].update(focused_burnin_frame_visits=1),
    lambda value: value["learning_gate"].update(eligible_for_fresh_games=True),
    lambda value: value["diagnostics_after"]["train"]["factors"][0]["after"].update(nll=0.),
    lambda value: value["calibration"].update(retention_kl=100.),
])
def test_completed_resume_rejects_changed_actor_epoch_or_dose(corpus, tmp_path, mutation):
    arguments, _, _ = corpus
    out = tmp_path / "fit"
    config = training.CorrectionTrainingConfig(epochs=1, chunk_length=2)
    result = training.train_corrections(**arguments, out=out, config=config)
    mutation(result)
    training.atomic_json(out / "result.json", result)
    with pytest.raises(ValueError, match="completed correction"):
        training.train_corrections(**arguments, out=out, config=config, resume=True)


def test_accepted_calibration_alone_cannot_start_training(corpus):
    arguments, corrections, collection = corpus
    held_out = collection["jobs"][1]["job_id"]
    corrections["records"][0].update(job_id=held_out, source_episode_sha256=collection["completed"][held_out]["sha256"])
    training.atomic_json(arguments["corrections_manifest"], corrections)
    with pytest.raises(ValueError, match="no independently accepted training"):
        training.validate_corrections(**arguments)


@pytest.mark.parametrize("kwargs", [dict(epochs=0), dict(chunk_length=0), dict(learning_rate=float("nan")), dict(seed=-1),
    dict(correction_weight=0), dict(correction_weight=float("nan")), dict(correction_exposures=-1),
    dict(correction_exposures=1.5), dict(correction_exposures=True)])
def test_invalid_correction_training_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        training.CorrectionTrainingConfig(**kwargs)


def test_focused_exposures_and_weight_preserve_factual_retention_and_optimizer_dose(corpus, tmp_path, monkeypatch):
    arguments, _, _ = corpus
    original = training.tactical_batch_loss
    weights = []
    def capture(*args, **kwargs):
        batch = args[4]
        weights.extend(batch["tactical_weights"][batch["active"]].tolist())
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "tactical_batch_loss", capture)
    config = training.CorrectionTrainingConfig(epochs=2, chunk_length=2,
        correction_weight=3., correction_exposures=2)
    out = tmp_path / "focused"
    result = training.train_corrections(**arguments, out=out, config=config)
    assert weights == [3., 3.]
    assert result["total_training_frame_visits"] == 10
    assert result["total_optimizer_steps"] == 6
    assert result["total_correction_presentations"] == 6
    assert result["total_focused_target_presentations"] == result["total_focused_windows"] == 4
    assert result["total_focused_gradient_frame_visits"] == 8
    assert result["total_focused_burnin_frame_visits"] == 8
    assert result["total_correction_optimizer_steps"] == 4
    assert result["total_retention_only_optimizer_steps"] == 2
    assert all(row["active_rows"] == 1 and row["inactive_rows"] == 3 for row in result["history"])
    assert training.train_corrections(**arguments, out=out, config=config, resume=True) == result


def test_focused_window_rebuilds_the_current_student_entire_prefix(monkeypatch):
    torch.manual_seed(19)
    policy = RecurrentV4Policy(compact_config())
    arrays = training.correction_episode(episode(9), {2: [1, 0, 22, 5, 0], 7: [1, 0, 22, 5, 0]})
    meta = dict(family="bait")
    calls = []
    original = training.differentiable_unroll
    def capture(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), int(args[1]["real"].sum())))
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "differentiable_unroll", capture)
    for _ in range(2):
        with torch.no_grad():
            for value in policy.recurrent.parameters():
                value.add_(.05)
            batch = chunk_batch([arrays], [meta], start=0, length=8, arm="control")
            logits, _, _ = original(policy, batch, None, batch["reset_mask"])
            expected = -policy.log_prob(logits, batch["masks"], batch["actions"])[0, 7]
        calls.clear()
        actual = training._focused_loss(policy, arrays, meta, step=7,
            config=training.CorrectionTrainingConfig(chunk_length=2))
        assert calls == [(False, 2), (False, 2), (False, 2), (True, 2)]
        assert torch.allclose(actual, expected, atol=1e-6)
        policy.zero_grad(set_to_none=True)
        actual.backward()
        assert sum(float(value.grad.square().sum()) for name, value in policy.recurrent.named_parameters()
                   if "weight_hh" in name and value.grad is not None) > 0


@pytest.mark.parametrize("sequences", [[{}], {}])
def test_sequence_manifests_fail_before_creating_training_output(corpus, tmp_path, sequences):
    arguments, corrections, _ = corpus
    corrections["sequences"] = sequences
    training.atomic_json(arguments["corrections_manifest"], corrections)
    out = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="sequence"):
        training.train_corrections(**arguments, out=out)
    assert not out.exists()


def test_partial_resume_keeps_the_exact_exposure_schedule_and_final_actor(corpus, tmp_path, monkeypatch):
    arguments, _, _ = corpus
    config = training.CorrectionTrainingConfig(epochs=2, chunk_length=2, correction_exposures=2)
    complete = training.train_corrections(**arguments, out=tmp_path / "complete", config=config)
    original = training._atomic_save
    def interrupt(path, value):
        original(path, value)
        if path.name == "training_state.pt" and value["epoch"] == 1:
            raise RuntimeError("fixture interruption after sealed first epoch")
    monkeypatch.setattr(training, "_atomic_save", interrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        training.train_corrections(**arguments, out=out, config=config)
    monkeypatch.setattr(training, "_atomic_save", original)
    resumed = training.train_corrections(**arguments, out=out, config=config, resume=True)
    assert resumed["history"] == complete["history"]
    assert resumed["diagnostics_after"] == complete["diagnostics_after"]
    left = torch.load(complete["checkpoint"]["path"], weights_only=True)
    right = torch.load(resumed["checkpoint"]["path"], weights_only=True)
    assert all(torch.equal(left[key], right[key]) for key in left)
