from __future__ import annotations

import numpy as np
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
    _factorized_kl,
    compute_batch_loss,
    train_one,
)
from simulator.rl.v4.model_v4 import V4Logits, masks_from_legal_play


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
    with patch("simulator.rl.v4.search_distillation._forward", return_value=(logits, masks)):
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

    with patch("simulator.rl.v4.search_distillation._stack_records", return_value={}), \
         patch("simulator.rl.v4.search_distillation.compute_batch_loss", side_effect=loss), \
         patch("simulator.rl.v4.search_distillation.evaluate_records", return_value={"n": 0}), \
         patch("simulator.rl.v4.search_distillation._check_deadline", side_effect=interrupt):
        partial = train_one(anchor, records, records, method="hard-none", seed=0,
                            context=0, epochs=1, batch_size=1, checkpoint_path=checkpoint,
                            deadline_epoch=1.0)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert partial["status"] == "partial"
    assert saved["epoch_completed"] == 0
    assert torch.equal(saved["state_dict"]["weight"], anchor.state_dict()["weight"])
    with patch("simulator.rl.v4.search_distillation._stack_records", return_value={}), \
         patch("simulator.rl.v4.search_distillation.compute_batch_loss", side_effect=loss), \
         patch("simulator.rl.v4.search_distillation.evaluate_records", return_value={"n": 0}):
        complete = train_one(anchor, records, records, method="hard-none", seed=0,
                             context=0, epochs=1, batch_size=1, checkpoint_path=checkpoint)
    assert complete["status"] == "complete"
    assert complete["optimizer_steps"] == 2
