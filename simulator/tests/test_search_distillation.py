from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from simulator.rl.v4.search_distillation import (
    ActionTarget,
    FactorizedTarget,
    build_soft_search_target,
    hard_factorized_target,
    paired_advantage,
    parse_entry_id,
    compute_batch_loss,
    train_one,
)
from simulator.rl.v4.search_distillation.losses import _factorized_kl
from simulator.rl.v4.model_v4 import V4Logits, masks_from_legal_play


def _synthetic_observation() -> dict[str, np.ndarray]:
    """Build a small public-input fixture without the retired data generator."""

    from simulator.rl.v4.hand_features import build_hand_tokens

    return {
        "raster": np.zeros((21, 32, 18), dtype=np.float32),
        "global_features": np.zeros((768,), dtype=np.float32),
        "entities": np.zeros((128, 32), dtype=np.float32),
        "entity_mask": np.zeros((128,), dtype=bool),
        "hand_tokens": build_hand_tokens(
            ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
        ),
        "opp_hand_probs": np.full((128,), 1.0 / 128, dtype=np.float32),
        "opp_out_of_cycle": np.zeros((128,), dtype=bool),
        "opp_elixir_interval": np.asarray([0.0, 10.0], dtype=np.float32),
        "event_history": np.zeros((16, 8), dtype=np.float32),
    }


def test_cached_prefix_batch_respects_wait_legality() -> None:
    from simulator.rl.v4.search_distillation import CachedPrefix
    from simulator.rl.v4.search_distillation.batching import _stack_records

    inputs = _synthetic_observation()
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[0, 0, 0] = True
    record = CachedPrefix(
        key="test", match_id="test", offset=0, observations=[inputs],
        legal_play=[legal], legal_wait=[False], recorded_action=ActionTarget(1, 0, 0, 0),
    )
    batch = _stack_records([record], context=0, device=torch.device("cpu"))
    assert batch["masks"].mode[0, 0].tolist() == [False, True]


def test_parse_entry_id_and_paired_advantage_use_common_replicas() -> None:
    assert parse_entry_id("train:beatdown#4:t247") == ("train", "beatdown#4", 247)
    verdict = {
        "branches": [
            {"source": "sim", "mode": 1, "slot": 0, "row": 2, "col": 3,
             "mean": 0.10, "values": [0.0, 0.2, 0.1]},
            {"source": "random", "mode": 0, "mean": 0.30,
             "values": [0.2, 0.4, 0.3]},
        ]
    }
    stats = paired_advantage(verdict)
    assert np.isclose(stats["mean"], 0.2)
    assert np.isclose(stats["se"], 0.0)
    assert np.isclose(stats["lcb95"], 0.2)


def test_soft_target_retains_unsearched_prior_and_normalizes_factors() -> None:
    legal = np.ones((2, 2, 2), dtype=bool)
    old_mode = np.asarray([0.25, 0.75], dtype=np.float32)
    old_card = np.asarray([0.6, 0.4], dtype=np.float32)
    old_place = np.full((2, 2, 2), 0.25, dtype=np.float32)
    branches = [
        {"source": "sim", "mode": 1, "slot": 0, "row": 0, "col": 0,
         "mean": 0.0},
        {"source": "search", "mode": 1, "slot": 0, "row": 1, "col": 1,
         "mean": 0.4},
    ]
    target = build_soft_search_target(
        old_mode=old_mode, old_card=old_card, old_placement=old_place,
        legal_play=legal, branches=branches,
        baseline=ActionTarget(1, 0, 0, 0), temperature=0.25,
    )
    assert isinstance(target, FactorizedTarget)
    assert np.isclose(target.mode.sum(), 1.0)
    assert np.isclose(target.card.sum(), 1.0)
    assert np.isclose(target.placement[0].sum(), 1.0)
    assert target.mode[1] > old_mode[1] - 1e-6
    assert target.placement[0, 1, 1] > target.placement[0, 0, 0]
    # The unsearched slot remains a proper conditional distribution.
    assert np.isclose(target.placement[1].sum(), 1.0)


def test_hard_target_rejects_illegal_play() -> None:
    legal = np.zeros((2, 2, 2), dtype=bool)
    legal[0, 0, 0] = True
    target = hard_factorized_target(ActionTarget(1, 0, 0, 0), legal)
    assert target.mode.tolist() == [0.0, 1.0]
    assert target.card.tolist() == [1.0, 0.0]
    assert target.placement[0, 0, 0] == 1.0
    try:
        hard_factorized_target(ActionTarget(1, 1, 0, 0), legal)
    except ValueError as exc:
        assert "illegal" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("illegal target was accepted")


def test_factorized_kl_is_zero_for_identical_distributions() -> None:
    legal = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bool)
    legal[..., 0, :, :] = True
    masks = masks_from_legal_play(legal)
    mode = torch.tensor([[[0.4, 0.6]]])
    card = torch.tensor([[[1.0, 0.0]]])
    place = torch.zeros((1, 1, 2, 2, 2))
    place[..., 0, :, :] = 0.25
    assert torch.allclose(_factorized_kl((mode, card, place), (mode, card, place), masks), torch.zeros(()))


def test_soft_methods_train_interleaved_correction_and_replay_rows() -> None:
    """The original bug indexed a one-row correction subset with batch index 1."""
    legal = torch.ones((2, 1, 2, 2, 2), dtype=torch.bool)
    masks = masks_from_legal_play(legal)
    logits = V4Logits(
        mode=torch.zeros((2, 1, 2), requires_grad=True),
        wait_duration=torch.zeros((2, 1, 1)),
        card=torch.zeros((2, 1, 2), requires_grad=True),
        placement=torch.zeros((2, 1, 2, 2, 2), requires_grad=True),
    )
    replay = SimpleNamespace(is_correction=False, recorded_action=ActionTarget(0))
    correction = SimpleNamespace(
        is_correction=True, recorded_action=ActionTarget(1, 0, 0, 0),
        correction_action=ActionTarget(1, 0, 1, 1),
        branches=[
            {"mode": 1, "slot": 0, "row": 0, "col": 0, "mean": 0.0},
            {"mode": 1, "slot": 0, "row": 1, "col": 1, "mean": 0.4},
        ],
        legal_play=[np.ones((2, 2, 2), dtype=bool)],
    )
    with patch("simulator.rl.v4.search_distillation.losses._forward", return_value=(logits, masks)):
        for method in ("soft-replay", "soft-kl"):
            if logits.mode.grad is not None:
                logits.mode.grad.zero_()
            loss, _ = compute_batch_loss(None, None, {}, [replay, correction], method=method)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(logits.mode.grad).all()


def test_kl_retention_only_uses_final_decision() -> None:
    legal = torch.ones((1, 2, 1, 1, 1), dtype=torch.bool)
    masks = masks_from_legal_play(legal)
    old = (
        torch.tensor([[[0.9, 0.1], [0.5, 0.5]]]),
        torch.ones((1, 2, 1)), torch.ones((1, 2, 1, 1, 1)),
    )
    new = (
        torch.tensor([[[0.1, 0.9], [0.5, 0.5]]]),
        torch.ones((1, 2, 1)), torch.ones((1, 2, 1, 1, 1)),
    )
    assert torch.allclose(_factorized_kl(old, new, masks), torch.zeros(()))


def test_paired_confidence_uses_only_independent_confirmations() -> None:
    from simulator.rl.v4.search_distillation import validated_correction

    verdict = {"discovery_replica_indices": list(range(8)),
               "confirmation_replica_indices": list(range(8, 16)), "branches": [
        {"source": "sim", "mode": 0, "mean": 0.0, "values": [0.0] * 8,
         "confirmation_values": [0.0] * 8},
        {"source": "rule", "mode": 1, "slot": 0, "row": 0, "col": 0,
         "mean": 1.0, "values": [1.0] * 8, "confirmation_values": [-1.0] * 8},
    ]}
    accepted, _, stats = validated_correction(verdict)
    assert not accepted
    assert stats["mean"] == -1.0
    assert stats["confidence_source"] == "validation"


def test_single_confirmation_cannot_certify_correction() -> None:
    from simulator.rl.v4.search_distillation import validated_correction

    verdict = {"discovery_replica_indices": [0],
               "confirmation_replica_indices": [1], "branches": [
        {"source": "sim", "mode": 0, "values": [0.0],
         "confirmation_values": [0.0]},
        {"source": "rule", "mode": 1, "slot": 0, "row": 0, "col": 0,
         "values": [1.0], "confirmation_values": [1.0]},
    ]}
    accepted, action, stats = validated_correction(verdict)
    assert not accepted
    assert action is None
    assert stats["mean"] == 1.0
    assert stats["se"] == float("inf")
    assert stats["lcb95"] == -float("inf")


def test_deadline_inside_first_epoch_resumes_from_initial_boundary(tmp_path: Path) -> None:
    from simulator.rl.v4.search_distillation import DistillationDeadline

    anchor = torch.nn.Linear(1, 1, bias=False)
    records = [SimpleNamespace(is_correction=True) for _ in range(2)]
    checkpoint = tmp_path / "partial.pt"
    checks = 0

    def interrupt(_deadline):
        nonlocal checks
        checks += 1
        if checks == 4:  # after one optimizer step, before the next batch
            raise DistillationDeadline("test deadline")

    def loss(student, *_args, **_kwargs):
        return student.weight.sum(), {}

    with patch("simulator.rl.v4.search_distillation.training._stack_records", return_value={}), \
         patch("simulator.rl.v4.search_distillation.training.compute_batch_loss", side_effect=loss), \
         patch("simulator.rl.v4.search_distillation.training.evaluate_records", return_value={"n": 0}), \
         patch("simulator.rl.v4.search_distillation.training._check_deadline", side_effect=interrupt):
        partial = train_one(anchor, records, records, method="hard-none", seed=0,
                            context=0, epochs=1, batch_size=1, checkpoint_path=checkpoint,
                            deadline_epoch=1.0)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert partial["status"] == "partial"
    assert saved["epoch_completed"] == 0
    assert torch.equal(saved["state_dict"]["weight"], anchor.state_dict()["weight"])
    with patch("simulator.rl.v4.search_distillation.training._stack_records", return_value={}), \
         patch("simulator.rl.v4.search_distillation.training.compute_batch_loss", side_effect=loss), \
         patch("simulator.rl.v4.search_distillation.training.evaluate_records", return_value={"n": 0}):
        complete = train_one(anchor, records, records, method="hard-none", seed=0,
                             context=0, epochs=1, batch_size=1, checkpoint_path=checkpoint)
    assert complete["status"] == "complete"
    assert complete["optimizer_steps"] == 2


def test_prepare_records_keeps_corrections_unique_and_caps_retention(tmp_path: Path) -> None:
    """Exercise real verdict loading/gating; only simulator replay is stubbed."""
    import json
    from simulator.rl.v4.search_distillation import prepare_records

    match = {"config": {"archetype": "hog", "match_index": 0},
             "decisions": [{"t": t, "learner": {"mode": 0}} for t in range(8)]}
    (tmp_path / "train_hog_0_match.json").write_text(json.dumps(match))
    entries = []
    for offset, confirmed_gain in ((2, 0.5), (3, -0.5), (4, -0.5)):
        entries.append({"entry_id": f"train:hog#0:t{offset}"})
        verdict = {"discovery_replica_indices": [0, 1],
                   "confirmation_replica_indices": [2, 3], "branches": [
            {"source": "sim", "mode": 0, "mean": 0.0, "values": [0.0, 0.0],
             "confirmation_values": [0.0, 0.0]},
            {"source": "search", "mode": 1, "slot": 0, "row": 0, "col": 0,
             "mean": 0.5, "values": [0.5, 0.5],
             "confirmation_values": [confirmed_gain, confirmed_gain]},
        ]}
        (tmp_path / f"train_hog_0_t{offset}_verdict.json").write_text(json.dumps(verdict))

    def replay(_match, offsets, *, context, **_kwargs):
        return ({t: [{"time": np.asarray(t)}] * (context + 1) for t in offsets},
                {t: [np.ones((1, 1, 1), dtype=bool)] * (context + 1) for t in offsets},
                {t: [True] * (context + 1) for t in offsets})

    with patch("simulator.rl.v4.search_distillation.preparation._replay_selected_prefixes_with_legal", side_effect=replay):
        for count in (0, 1, 2):
            records = prepare_records(tmp_path, entries, context=1, retention_count=count)
            corrections = [r for r in records if r.is_correction]
            retention = [r for r in records if not r.is_correction]
            assert len(corrections) == 1
            assert corrections[0].key == "train:hog#0:t2"
            assert corrections[0].correction_action == ActionTarget(1, 0, 0, 0)
            assert len(retention) == count
            assert len({r.offset for r in records}) == count + 1


def test_prefix_cache_loads_original_public_class_paths(tmp_path: Path, monkeypatch) -> None:
    """Existing torch caches must load after the implementation becomes a package."""
    from simulator.rl.v4.search_distillation import CachedPrefix, CacheConfig

    record = CachedPrefix(
        key="train:hog#0:t1", match_id="hog#0", offset=1,
        observations=[{"raster": np.zeros((1, 2, 2), dtype=np.float32)}],
        legal_play=[np.ones((1, 2, 2), dtype=bool)], legal_wait=[True],
        recorded_action=ActionTarget(0), correction_action=ActionTarget(1, 0, 1, 1),
    )
    target = hard_factorized_target(ActionTarget(1, 0, 1, 1), record.legal_play[0])
    path = tmp_path / "old_prefix_cache.pt"
    # Serialize the exact historical module names without retaining binary fixtures.
    with monkeypatch.context() as patcher:
        for cls in (ActionTarget, CachedPrefix, CacheConfig, FactorizedTarget):
            patcher.setattr(cls, "__module__", "simulator.rl.v4.search_distillation")
        torch.save({"records": [record], "target": target, "config": CacheConfig()}, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(restored["records"][0], CachedPrefix)
    assert restored["records"][0].correction_action == ActionTarget(1, 0, 1, 1)
    np.testing.assert_array_equal(restored["records"][0].legal_play[0], record.legal_play[0])
    np.testing.assert_array_equal(restored["target"].placement, target.placement)
    assert isinstance(restored["config"], CacheConfig)


@pytest.mark.parametrize("method", ["hard-none", "hard-replay", "soft-replay", "soft-kl"])
def test_real_training_resumes_identically_at_epoch_boundary(tmp_path: Path, method: str) -> None:
    from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy
    from simulator.rl.v4.search_distillation import CachedPrefix, DistillationDeadline

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        anchor = RecurrentV4Policy(ModelConfigV4(
            model_dim=16, spatial_channels=8, fused_dim=24, gru_hidden_dim=24,
            transformer_heads=4, transformer_layers=1, transformer_ff_dim=32,
            spatial_head_dim=8,
        ))
        records = []
        for index in range(2):
            observation = _synthetic_observation()
            target = ActionTarget(1, 0, index, index)
            legal_play = np.ones((4, 32, 18), dtype=bool)
            records.append(CachedPrefix(
                key=f"train:synthetic#0:t{index}", match_id="synthetic#0", offset=index,
                observations=[observation, observation],
                legal_play=[legal_play, legal_play], legal_wait=[True, True],
                recorded_action=ActionTarget(0), correction_action=target,
                branches=[{"source": "sim", "mode": 0, "mean": 0.0},
                          {"source": "search", "mode": target.mode, "slot": target.slot,
                           "row": target.row, "col": target.col, "mean": 0.4}],
                is_correction=index == 0,
            ))
        common = dict(method=method, seed=3, context=1, epochs=2, batch_size=2)
        full_path = tmp_path / "full.pt"
        full = train_one(anchor, records, [], checkpoint_path=full_path, **common)
        resume_path = tmp_path / "resume.pt"
        checks = 0

        def interrupt(_deadline):
            nonlocal checks
            checks += 1
            if checks == 4:  # First epoch saved; second epoch has not started.
                raise DistillationDeadline("test epoch boundary")

        with patch("simulator.rl.v4.search_distillation.training._check_deadline", side_effect=interrupt):
            partial = train_one(anchor, records, [], checkpoint_path=resume_path, **common)
        assert partial["status"] == "partial"
        assert torch.load(resume_path, weights_only=False)["epoch_completed"] == 1
        resumed = train_one(anchor, records, [], checkpoint_path=resume_path, **common)
        assert resumed["status"] == "complete"
        assert resumed["resumed_from_epoch"] == 1
        assert resumed["history"] == full["history"]
        assert resumed["optimizer_steps"] == full["optimizer_steps"] == 2
        full_state = torch.load(full_path, weights_only=False)["state_dict"]
        resume_state = torch.load(resume_path, weights_only=False)["state_dict"]
        for name in full_state:
            torch.testing.assert_close(full_state[name], resume_state[name], rtol=0, atol=0)
        assert any(not torch.equal(full_state[name], value)
                   for name, value in anchor.state_dict().items())
    finally:
        torch.set_num_threads(previous_threads)
