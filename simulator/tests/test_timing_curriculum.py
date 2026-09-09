"""Tests for the V4 Stage-1 timing curriculum (real-simulator WAIT/PLAY timing).

Covers the acceptance criteria that can be checked without training:
deterministic generation, teacher legality, six-family coverage, duration
bands, WAIT-with-legal-PLAY quotas, temporal-sequence integrity, and the
elixir/mask consistency invariant (an elixir modification must never
disagree with the legal-action mask regenerated from simulator state).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.distillation import PLAYER_DECK, to_torch_batch
from simulator.rl.losses_v4 import v4_supervised_loss
from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy, WAIT_DURATIONS
from simulator.rl.timing_curriculum import (
    TIMING_FAMILIES,
    TIMING_GENERATOR_VERSION,
    TimingConfig,
    balance_timing_pool,
    generate_timing_dataset,
    generate_timing_natural,
    generate_timing_pool,
)
from simulator.rl.timing_teacher import TIMING_TEACHER_VERSION


def _small_config(**overrides) -> TimingConfig:
    params = {"n_states": 240, "seed": 3, "pool_factor": 2}
    params.update(overrides)
    return TimingConfig(**params)


def _pool_signature(pool: list[dict]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in pool:
        digest.update(np.ascontiguousarray(record["raster"]).tobytes())
        target = record["target"]
        digest.update(bytes((target.mode, target.card_slot, target.row, target.col)))
    return digest.hexdigest()


def test_timing_pool_is_deterministic() -> None:
    config = _small_config()
    first = generate_timing_pool(config)
    second = generate_timing_pool(config)
    assert len(first) == len(second) > 0
    assert _pool_signature(first) == _pool_signature(second)
    assert first[0]["provenance"] == second[0]["provenance"]


def test_timing_families_all_represented() -> None:
    pool = generate_timing_pool(_small_config())
    assert set(TIMING_FAMILIES) <= {record["family"] for record in pool}


def test_timing_teacher_actions_are_legal() -> None:
    pool = generate_timing_pool(_small_config())
    assert pool
    for record in pool:
        target = record["target"]
        assert target.mode in (0, 1)
        assert 0 <= target.wait_duration_idx < len(WAIT_DURATIONS)
        assert record["provenance"]["generator"] == TIMING_GENERATOR_VERSION
        assert record["provenance"]["teacher"] == TIMING_TEACHER_VERSION
        if target.mode == 1:
            assert bool(record["legal_play"][target.card_slot, target.row, target.col])
            assert float(record["soft_placement"].sum()) == pytest.approx(1.0)
        else:
            assert record["legal_wait"]


def test_timing_balance_quotas_hold() -> None:
    samples, stats = generate_timing_dataset(_small_config(n_states=240))
    assert len(samples) == 240
    assert abs(stats["wait_share"] - 0.5) <= 0.03
    for family in TIMING_FAMILIES:
        assert stats["family_share"][family] >= 0.10
    for duration in range(4):
        assert 0.15 <= stats["duration_share"][duration] <= 0.40
    assert stats["wait_with_play_share"] >= 0.70
    assert stats["wait_with_two_plus_share"] >= 0.40


def test_timing_provenance_bookkeeping_agrees_with_masks() -> None:
    pool = generate_timing_pool(_small_config())
    for record in pool:
        recomputed = int(np.asarray(record["legal_play"]).reshape(4, -1).any(axis=1).sum())
        assert recomputed == record["legal_cards"] == record["provenance"]["legal_cards"]
        assert record["provenance"]["play_legal_during_wait"] == bool(recomputed > 0)
        assert record["provenance"]["timing_family"] == record["family"]
        assert record["provenance"]["sequence_offset"] == record["offset"]


def test_timing_elixir_mask_consistency() -> None:
    """Elixir scalar, hand flags, global vector, and mask must agree.

    Every elixir modification in the curriculum happens in authoritative
    simulator state before observation, so an unaffordable card can never
    carry a legal cell.  This test would catch a feature-vector-only patch.
    """

    from cr_bot.features.global_features import GLOBAL_SCALAR_IDX

    pool = generate_timing_pool(_small_config())
    assert pool
    by_key = {row[0]: row for row in PLAYER_DECK}
    strategic_waits = 0
    for record in pool:
        sample_elixir = record["own_elixir"]
        assert sample_elixir * 1000.0 == pytest.approx(
            float(record["provenance"]["target_elixir_milli"])
        )
        hand = record["provenance"]["target_hand"]
        for slot, card in enumerate(hand):
            _, _, cost, *_ = by_key[card]
            affordable = cost <= sample_elixir
            assert bool(record["hand_tokens"][slot, 3] > 0.5) is bool(affordable)
            if not affordable:
                assert not bool(record["legal_play"][slot].any()), (
                    f"unaffordable {card} at {sample_elixir} elixir has legal cells"
                )
        elixir_slot = GLOBAL_SCALAR_IDX["elixir_self"]
        assert float(record["global_features"][elixir_slot]) == pytest.approx(
            min(sample_elixir / 10.0, 1.0)
        )
        if record["mode"] == 0 and record["play_legal"]:
            strategic_waits += 1
    # The invariant is vacuous without strategic WAITs; require their presence.
    assert strategic_waits > 0


def test_timing_sequences_have_integrity_and_switches() -> None:
    pool = generate_timing_pool(_small_config())
    by_seq: dict[str, list[dict]] = {}
    for record in pool:
        by_seq.setdefault(record["seq_id"], []).append(record)
    assert len(by_seq) > 10
    switches = 0
    for seq_id, records in by_seq.items():
        records.sort(key=lambda r: r["offset"])
        assert [r["offset"] for r in records] == list(range(len(records)))
        seeds = {r["provenance"]["env_seed"] for r in records}
        assert len(seeds) == 1
        modes = [r["mode"] for r in records]
        if any(a == 0 and b == 1 for a, b in zip(modes, modes[1:])):
            switches += 1
    assert switches > 0, "pool must contain WAIT->PLAY transition sequences"


def test_timing_batch_schema_matches_v4() -> None:
    samples, _ = generate_timing_dataset(_small_config(n_states=64))
    batch = to_torch_batch(samples)
    masks, actions = batch["masks"], batch["actions"]
    assert batch["raster"].shape[0] == len(samples)
    for index in range(len(samples)):
        mode = int(actions.mode[index, 0])
        assert bool(masks.mode[index, 0, mode])
        if mode == 1:
            slot = int(actions.card_slot[index, 0])
            row = int(actions.placement[index, 0, 0])
            col = int(actions.placement[index, 0, 1])
            assert bool(masks.placement[index, 0, slot, row, col])


def test_timing_supervision_trains_behavior() -> None:
    """Distillation on timing states must improve decisions, not just loss.

    Note: the reported card-loss *value* is NaN on strategic WAIT rows (a
    forward-only ``+inf * is_play=0`` artifact in the shared loss; gradients
    on those rows are exactly zero, so training is unaffected).  The loss
    module is frozen for this task, so this test asserts behavioral
    improvement (agreement) instead of loss decrease.
    """

    torch.manual_seed(7)
    samples, _ = generate_timing_dataset(_small_config(n_states=120, seed=11))
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

    def agreement() -> tuple[float, float]:
        policy.eval()
        with torch.no_grad():
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
            actions = batch["actions"]
            mode_acc = float(
                (logits.mode.argmax(-1) == actions.mode).float().mean()
            )
            is_play = actions.mode == 1
            card_acc = float(
                (logits.card.argmax(-1)[is_play] == actions.card_slot[is_play])
                .float()
                .mean()
            )
        policy.train()
        return mode_acc, card_acc

    mode_before, card_before = agreement()
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
    mode_after, card_after = agreement()
    assert card_after > card_before + 0.03
    assert mode_after >= mode_before
    assert all(
        bool(torch.isfinite(p).all()) for p in policy.parameters()
    ), "NaN parameters after distillation steps"


def test_timing_natural_set_builds_without_balancing() -> None:
    samples, stats = generate_timing_natural(_small_config(n_states=64, seed=9))
    assert len(samples) == 64
    assert stats["n"] == 64
    modes = Counter(s.target.mode for s in samples)
    assert modes[0] + modes[1] == 64


def test_timing_balance_rejects_empty_pool() -> None:
    with pytest.raises(Exception, match="empty"):
        balance_timing_pool([], _small_config())
