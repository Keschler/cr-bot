"""Tests for the real-simulator Stage-1 distillation generator.

The sim generator must satisfy the same consumer contract as the synthetic
one (``DistillationSample`` + ``to_torch_batch``) while producing
observations from ``BasicMechanicsScenarioEnv`` instead of hand-made
features.  These tests pin determinism, legality, teacher compatibility,
and the provenance that distinguishes real-simulator states.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.distillation import (
    PLAYER_DECK,
    DistillationConfig,
    TACTICAL_FAMILIES,
    to_torch_batch,
)
from simulator.rl.model_v4 import WAIT_DURATIONS, ModelConfigV4, RecurrentV4Policy
from simulator.rl.losses_v4 import v4_supervised_loss
from simulator.rl.simulator_distillation import (
    FAMILY_TO_SOURCE,
    SIM_GENERATOR_VERSION,
    SOURCE_ARCHETYPE,
    build_hand_tokens,
    generate_sim_dataset,
    generate_sim_sample,
)
from simulator.rl.simulator_teacher import TeacherError


def _config(n: int = 32, seed: int = 5) -> DistillationConfig:
    return DistillationConfig(n_states=n, seed=seed)


def test_sim_families_map_to_known_sources() -> None:
    assert set(FAMILY_TO_SOURCE) == set(TACTICAL_FAMILIES)
    for source in FAMILY_TO_SOURCE.values():
        assert source in SOURCE_ARCHETYPE


def test_sim_sample_is_deterministic_per_index() -> None:
    config = _config()
    first = generate_sim_sample(config, 17)
    second = generate_sim_sample(config, 17)
    assert first.family == second.family
    assert np.array_equal(first.raster, second.raster)
    assert np.array_equal(first.global_features, second.global_features)
    assert np.array_equal(first.hand_tokens, second.hand_tokens)
    assert first.target == second.target
    assert first.provenance == second.provenance
    assert first.provenance["generator"] == SIM_GENERATOR_VERSION
    other = generate_sim_sample(DistillationConfig(n_states=8, seed=6), 17)
    assert not np.array_equal(first.raster, other.raster)


def test_sim_targets_are_always_legal() -> None:
    for sample in generate_sim_dataset(_config(n=48, seed=7)):
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


def test_sim_observations_are_real_simulator_tensors() -> None:
    sample = generate_sim_sample(_config(), 3)
    # Synthetic states carry splats on channels 0-2 only; real rasters use the
    # full vision channel set (towers, elixir, timers, ...).
    assert int((sample.raster != 0.0).sum()) > 0
    assert int((sample.global_features != 0.0).sum()) > 1
    assert sample.raster.shape == (21, 32, 18)
    assert sample.global_features.shape == (768,)
    assert sample.entity_tokens.shape == (128, 32)
    assert sample.hand_tokens.shape == (4, 16)
    provenance = sample.provenance
    assert provenance["source"] == FAMILY_TO_SOURCE[sample.family]
    assert provenance["state_hash"]
    assert provenance["setup_cards"] is not None
    assert len(provenance["target_hand"]) == 4


def test_sim_hand_tokens_match_observation_contract() -> None:
    sample = generate_sim_sample(_config(), 11)
    hand = sample.provenance["target_hand"]
    by_key = {row[0]: row for row in PLAYER_DECK}
    for slot, card in enumerate(hand):
        _, policy_id, cost, *_ = by_key[card]
        assert sample.hand_tokens[slot, 0] == pytest.approx(policy_id / 127.0)
        assert sample.hand_tokens[slot, 1] == pytest.approx(cost / 10.0)
        assert sample.hand_tokens[slot, 2] == pytest.approx(slot / 3.0)
        affordable = cost <= sample.own_elixir
        assert bool(sample.hand_tokens[slot, 3] > 0.5) is bool(affordable)


def test_sim_hand_builder_rejects_unknown_cards() -> None:
    with pytest.raises(TeacherError):
        build_hand_tokens(["hog-rider", "cannon", "musketeer", "not-a-card"], 10.0)


def test_sim_covers_tactical_families() -> None:
    families = {sample.family for sample in generate_sim_dataset(DistillationConfig(n_states=400, seed=0))}
    assert families == set(TACTICAL_FAMILIES)


def test_sim_air_defense_setup_contains_air() -> None:
    samples = generate_sim_dataset(DistillationConfig(n_states=200, seed=0))
    air = [sample for sample in samples if sample.family == "air-defense"]
    assert air, "stratified mix must include air-defense states"
    assert any(
        "baby-dragon" in sample.provenance["setup_cards"]
        or "balloon" in sample.provenance["setup_cards"]
        or "lava-hound" in sample.provenance["setup_cards"]
        for sample in air
    )


def test_sim_batch_actions_agree_with_masks() -> None:
    samples = generate_sim_dataset(_config(n=16, seed=9))
    batch = to_torch_batch(samples)
    masks = batch["masks"]
    actions = batch["actions"]
    assert batch["raster"].shape[0] == 16
    for index in range(16):
        mode = int(actions.mode[index, 0])
        assert bool(masks.mode[index, 0, mode])
        if mode == 1:
            slot = int(actions.card_slot[index, 0])
            row = int(actions.placement[index, 0, 0])
            col = int(actions.placement[index, 0, 1])
            assert bool(masks.card[index, 0, slot])
            assert bool(masks.placement[index, 0, slot, row, col])


def test_sim_supervised_loss_is_finite_and_learns() -> None:
    torch.manual_seed(1)
    samples = generate_sim_dataset(_config(n=32, seed=9))
    batch = to_torch_batch(samples)
    policy = RecurrentV4Policy(
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)

    def batch_loss() -> float:
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
        return float(total.detach())

    before = batch_loss()
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
    after = batch_loss()
    assert after < before
