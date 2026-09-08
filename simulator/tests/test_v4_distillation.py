from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.distillation import (
    DistillationConfig,
    TACTICAL_FAMILIES,
    generate_dataset,
    generate_sample,
    to_torch_batch,
)
from simulator.rl.losses_v4 import V4SupervisedWeights, v4_supervised_loss
from simulator.rl.model_v4 import WAIT_DURATIONS, ModelConfigV4, RecurrentV4Policy
from simulator.rl.simulator_teacher import TeacherError, soft_placement_target, teacher_label


def _config(n: int = 64, seed: int = 5) -> DistillationConfig:
    return DistillationConfig(n_states=n, seed=seed)


def test_teacher_targets_are_always_legal() -> None:
    config = _config(n=8 * 12)
    for sample in generate_dataset(config):
        target = sample.target
        assert target.mode in (0, 1)
        assert 0 <= target.wait_duration_idx < len(WAIT_DURATIONS)
        if target.mode == 1:
            assert bool(sample.legal_play[target.card_slot, target.row, target.col])
            assert float(sample.soft_placement.sum()) == pytest.approx(1.0)
            assert bool(
                ((sample.soft_placement > 0.0) <= sample.legal_play[target.card_slot]).all()
            )
        else:
            assert sample.legal_wait


def test_teacher_covers_tactical_families() -> None:
    config = _config(n=400)
    families = {sample.family for sample in generate_dataset(config)}
    assert families == set(TACTICAL_FAMILIES)
    modes = {sample.target.mode for sample in generate_dataset(config)}
    assert modes == {0, 1}


def test_generator_is_deterministic_per_sample() -> None:
    config = _config()
    first = generate_sample(config, 17)
    second = generate_sample(config, 17)
    assert first.family == second.family
    assert np.array_equal(first.raster, second.raster)
    assert first.target == second.target
    assert first.provenance["generator"] == "synthetic-v4-distill-0"
    other = generate_sample(DistillationConfig(n_states=8, seed=6), 17)
    assert not np.array_equal(first.raster, other.raster)


def test_soft_placement_spreads_over_neighborhood() -> None:
    legal = np.zeros((6, 6), dtype=bool)
    legal[2:5, 2:5] = True
    probs = soft_placement_target(3, 3, legal)
    assert float(probs.sum()) == pytest.approx(1.0)
    assert int((probs > 0.0).sum()) == 9
    with pytest.raises(TeacherError):
        soft_placement_target(0, 0, np.zeros((6, 6), dtype=bool))


def _tiny_policy() -> RecurrentV4Policy:
    return RecurrentV4Policy(
        ModelConfigV4(
            model_dim=16,
            spatial_channels=8,
            fused_dim=24,
            gru_hidden_dim=24,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=32,
            spatial_head_dim=8,
        )
    )


def test_batch_actions_agree_with_masks() -> None:
    samples = generate_dataset(_config(n=16))
    batch = to_torch_batch(samples)
    masks = batch["masks"]
    actions = batch["actions"]
    assert batch["raster"].shape[0] == 16
    assert batch["reset_mask"].shape == (16, 1)
    for index in range(16):
        mode = int(actions.mode[index, 0])
        assert bool(masks.mode[index, 0, mode])
        if mode == 1:
            slot = int(actions.card_slot[index, 0])
            row = int(actions.placement[index, 0, 0])
            col = int(actions.placement[index, 0, 1])
            assert bool(masks.card[index, 0, slot])
            assert bool(masks.placement[index, 0, slot, row, col])


def test_supervised_loss_is_finite_and_learns() -> None:
    torch.manual_seed(1)
    samples = generate_dataset(_config(n=32, seed=9))
    batch = to_torch_batch(samples)
    policy = _tiny_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)

    def batch_loss() -> tuple[float, dict[str, float]]:
        logits, _, _ = policy(
            batch["raster"],
            batch["global_features"],
            batch["entities"],
            batch["entity_mask"],
            batch["hand_tokens"],
            batch["opp_hand_probs"],
            batch["opp_out_of_cycle"],
            batch["opp_elixir_interval"],
            batch["event_history"],
            batch["reset_mask"],
        )
        total, detail = v4_supervised_loss(
            logits, batch["masks"], batch["actions"],
            soft_placement=batch["soft_placement"],
        )
        return float(total.detach()), detail

    before, _ = batch_loss()
    assert np.isfinite(before)
    for _ in range(20):
        optimizer.zero_grad()
        logits, _, _ = policy(
            batch["raster"],
            batch["global_features"],
            batch["entities"],
            batch["entity_mask"],
            batch["hand_tokens"],
            batch["opp_hand_probs"],
            batch["opp_out_of_cycle"],
            batch["opp_elixir_interval"],
            batch["event_history"],
            batch["reset_mask"],
        )
        total, _ = v4_supervised_loss(
            logits, batch["masks"], batch["actions"],
            soft_placement=batch["soft_placement"],
        )
        total.backward()
        optimizer.step()
    after, detail = batch_loss()
    assert after < before
    assert detail["n_play"] + detail["n_wait"] == 32


def test_loss_ignores_unselected_factors() -> None:
    torch.manual_seed(2)
    samples = generate_dataset(_config(n=16, seed=11))
    waits = [s for s in samples if s.target.mode == 0]
    assert waits, "generator must produce WAIT states for this test"
    batch = to_torch_batch(waits[:8])
    policy = _tiny_policy()
    logits, _, _ = policy(
        batch["raster"],
        batch["global_features"],
        batch["entities"],
        batch["entity_mask"],
        batch["hand_tokens"],
        batch["opp_hand_probs"],
        batch["opp_out_of_cycle"],
        batch["opp_elixir_interval"],
        batch["event_history"],
        batch["reset_mask"],
    )
    _, detail = v4_supervised_loss(
        logits, batch["masks"], batch["actions"],
        soft_placement=batch["soft_placement"],
        weights=V4SupervisedWeights(),
    )
    assert detail["loss_card"] == 0.0
    assert detail["loss_placement"] == 0.0
    assert detail["n_play"] == 0.0


def test_teacher_label_rejects_fully_illegal_states() -> None:
    hand = np.zeros((4, 16), dtype=np.float32)
    entities = np.zeros((0, 32), dtype=np.float32)
    mask = np.zeros((0,), dtype=bool)
    legal = np.zeros((4, 8, 6), dtype=bool)
    with pytest.raises(TeacherError):
        teacher_label(
            hand_tokens=hand,
            entity_tokens=entities,
            entity_mask=mask,
            legal_play=legal,
            legal_wait=False,
            own_elixir=5.0,
        )
