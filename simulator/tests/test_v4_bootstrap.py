"""Focused contracts for the retained from-scratch V4 bootstrap."""

from __future__ import annotations

from collections import Counter
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4.bootstrap import (  # noqa: E402
    BootstrapConfig,
    generate_bootstrap_dataset,
    to_torch_batch,
)
from simulator.rl.v4.hand_features import (  # noqa: E402
    HandEncodingError,
    build_hand_tokens,
)
from simulator.rl.v4.simulator_teacher import teacher_label  # noqa: E402
from simulator.rl.v4.supervised import V4SupervisedWeights, v4_supervised_loss


def test_hand_encoder_is_fixed_and_rejects_unknown_cards() -> None:
    first = build_hand_tokens(
        ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
    )
    second = build_hand_tokens(
        ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 16)
    with pytest.raises(HandEncodingError, match="outside the V4 learner deck"):
        build_hand_tokens(["unknown", "cannon", "musketeer", "skeletons"], 7.0)


def test_bootstrap_collection_is_deterministic_and_public() -> None:
    config = BootstrapConfig(n_states=2, seed=19, max_decisions=16)
    first = generate_bootstrap_dataset(config)
    second = generate_bootstrap_dataset(config)
    assert [sample.provenance["state_hash"] for sample in first] == [
        sample.provenance["state_hash"] for sample in second
    ]
    for sample in first:
        assert sample.raster.shape == (21, 32, 18)
        assert sample.global_features.shape == (768,)
        assert sample.entity_tokens.shape == (128, 32)
        assert sample.opp_hand_probs.shape == (128,)
        assert bool(np.isfinite(sample.raster).all())


def test_bootstrap_collection_balances_opponent_archetypes() -> None:
    config = BootstrapConfig(n_states=12, seed=7, max_decisions=32)
    samples = generate_bootstrap_dataset(config)
    assert Counter(sample.family for sample in samples) == {
        archetype: 2 for archetype in config.archetypes
    }


def test_bootstrap_batch_preserves_masks_and_action_targets() -> None:
    samples = generate_bootstrap_dataset(BootstrapConfig(n_states=3, seed=2, max_decisions=12))
    batch = to_torch_batch(samples)
    assert tuple(batch["raster"].shape) == (3, 1, 21, 32, 18)
    assert tuple(batch["actions"].mode.shape) == (3, 1)
    assert tuple(batch["masks"].mode.shape) == (3, 1, 2)
    assert torch.isfinite(batch["soft_placement"]).all()
    for index, sample in enumerate(samples):
        mode = int(batch["actions"].mode[index, 0])
        assert bool(batch["masks"].mode[index, 0, mode])
        if mode == 1:
            slot = int(batch["actions"].card_slot[index, 0])
            row, col = (int(value) for value in batch["actions"].placement[index, 0])
            assert bool(batch["masks"].placement[index, 0, slot, row, col])


def test_rules_teacher_returns_a_legal_public_target() -> None:
    hand = build_hand_tokens(
        ["hog-rider", "cannon", "musketeer", "skeletons"], 7.0
    )
    entities = np.zeros((128, 32), dtype=np.float32)
    entity_mask = np.zeros((128,), dtype=bool)
    entities[0, 1:7] = [1.0, 0.5, 0.75, 1.0, 0.0, 0.0]
    entity_mask[0] = True
    legal = np.zeros((4, 32, 18), dtype=bool)
    legal[1, 20:, :] = True
    target = teacher_label(
        hand_tokens=hand,
        entity_tokens=entities,
        entity_mask=entity_mask,
        legal_play=legal,
        legal_wait=True,
        own_elixir=7.0,
    )
    assert target.mode in (0, 1)
    if target.mode:
        assert bool(legal[target.card_slot, target.row, target.col])


def test_bootstrap_mode_loss_is_invariant_to_duplicate_wait_frames() -> None:
    from simulator.rl.foundation.trajectory import ActionMasks
    from simulator.rl.v4.model_v4 import V4ActionBatch, V4Logits

    def mode_loss(n_wait: int) -> float:
        modes = torch.tensor([0] * n_wait + [1], dtype=torch.long).reshape(-1, 1)
        n = n_wait + 1
        logits = V4Logits(
            mode=torch.tensor([[[2.0, -2.0]]] * n),
            wait_duration=torch.zeros(n, 1, 4),
            card=torch.zeros(n, 1, 1),
            placement=torch.zeros(n, 1, 1, 1, 1),
        )
        masks = ActionMasks(
            mode=torch.ones(n, 1, 2, dtype=torch.bool),
            card=torch.ones(n, 1, 1, dtype=torch.bool),
            placement=torch.ones(n, 1, 1, 1, 1, dtype=torch.bool),
        )
        actions = V4ActionBatch(
            mode=modes,
            card_slot=torch.zeros(n, 1, dtype=torch.long),
            placement=torch.zeros(n, 1, 2, dtype=torch.long),
            wait_duration=torch.zeros(n, 1, dtype=torch.long),
        )
        loss, _ = v4_supervised_loss(
            logits, masks, actions,
            weights=V4SupervisedWeights(
                w_wait_duration=0.0, w_card=0.0, w_placement=0.0
            ),
        )
        return float(loss)

    assert mode_loss(1) == pytest.approx(mode_loss(9), abs=1e-6)
