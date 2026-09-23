"""Deployed checkpoint loading survives retirement of the training runners."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from simulator.engine import ENGINE_VERSION
from simulator.observation_v2 import OBSERVATION_V2_CONTRACT_HASH, OBSERVATION_V2_SCHEMA_VERSION
from simulator.rl.foundation.inference_checkpoint import (
    PROTOTYPE_CHECKPOINT_FORMAT,
    PrototypeConfig,
    PrototypeConfigurationError,
    _model_and_learner,
    load_shadow_prototype_checkpoint,
)
from simulator.ruleset import load_ruleset


@pytest.fixture
def checkpoint_payload():
    config = PrototypeConfig(device="cpu", allow_provisional=True)
    learner = _model_and_learner(config)
    ruleset = load_ruleset(config.ruleset_id)
    return {
        "metadata": {
            "checkpoint_format": PROTOTYPE_CHECKPOINT_FORMAT,
            "config": config.as_dict(),
            "ruleset_id": ruleset.ruleset_id,
            "ruleset_hash": ruleset.content_hash,
            "engine_version": ENGINE_VERSION,
            "actor_observation": {
                "privileged_inputs": False,
                "schema_version": OBSERVATION_V2_SCHEMA_VERSION,
                "contract_hash": OBSERVATION_V2_CONTRACT_HASH,
            },
        },
        "learner": learner.checkpoint_state(),
    }


def test_deployed_actor_loads_without_retired_training_package(tmp_path, checkpoint_payload):
    from simulator.physical_lab.workflow.prototype_controller import PrototypeActor
    path = tmp_path / "actor.pt"
    torch.save(checkpoint_payload, path)
    actor = PrototypeActor(path, device="cpu")
    assert actor.metadata["_checkpoint_ruleset_match"] is True
    expected = checkpoint_payload["learner"]
    actual = actor.learner.checkpoint_state()
    # Compare all actor parameters after loading, not only config/import success.
    for key, tensor in expected["policy"].items():
        assert torch.equal(actual["policy"][key], tensor), key


@pytest.mark.parametrize("field", ["public", "engine", "observation", "ruleset"])
def test_inference_reader_rejects_incompatible_artifacts(tmp_path, checkpoint_payload, field):
    payload = copy.deepcopy(checkpoint_payload)
    metadata = payload["metadata"]
    if field == "public":
        metadata["actor_observation"]["privileged_inputs"] = True
    elif field == "engine":
        metadata["engine_version"] = "incompatible"
    elif field == "observation":
        metadata["actor_observation"]["contract_hash"] = "incompatible"
    else:
        metadata["ruleset_hash"] = "sha256:" + "0" * 64
    path = tmp_path / "incompatible.pt"
    torch.save(payload, path)
    with pytest.raises(PrototypeConfigurationError):
        load_shadow_prototype_checkpoint(path, device="cpu")
