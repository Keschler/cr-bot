from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from simulator.rl.model_v4 import (
    WAIT_DURATIONS,
    ModelConfigV4,
    RecurrentV4Policy,
    V4ActionBatch,
    count_parameters,
)
from simulator.rl.trajectory import ActionMasks


def _small_config() -> ModelConfigV4:
    return ModelConfigV4(
        raster_channels=3,
        raster_height=8,
        raster_width=6,
        global_dim=16,
        entity_dim=5,
        max_entities=10,
        model_dim=16,
        spatial_channels=8,
        fused_dim=24,
        gru_hidden_dim=24,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=32,
        card_slots=3,
        placement_rows=8,
        placement_cols=6,
        spatial_head_dim=8,
    )


def _inputs(config: ModelConfigV4, batch: int = 2, time: int = 3):
    torch.manual_seed(0)
    raster = torch.randn(batch, time, 3, 8, 6)
    glob = torch.randn(batch, time, config.global_dim)
    entities = torch.randn(batch, time, 6, config.entity_dim)
    entity_mask = torch.ones(batch, time, 6, dtype=torch.bool)
    hand = torch.randn(batch, time, config.card_slots, 16)
    probs = torch.full((batch, time, 128), 1.0 / 128)
    out = torch.zeros(batch, time, 128, dtype=torch.bool)
    elixir = torch.tensor([[[2.0, 7.0]]], dtype=torch.float32).expand(batch, time, 2)
    events = torch.zeros(batch, time, 16, 8)
    reset = torch.zeros(batch, time, dtype=torch.bool)
    reset[:, 0] = True
    return raster, glob, entities, entity_mask, hand, probs, out, elixir, events, reset


def _masks(batch: int = 2, time: int = 3) -> ActionMasks:
    mode = torch.ones(batch, time, 2, dtype=torch.bool)
    card = torch.ones(batch, time, 3, dtype=torch.bool)
    placement = torch.ones(batch, time, 3, 8, 6, dtype=torch.bool)
    return ActionMasks(mode=mode, card=card, placement=placement)


def test_v4_forward_shapes_and_full_resolution() -> None:
    config = _small_config()
    policy = RecurrentV4Policy(config)
    logits, zt, ht = policy(*_inputs(config))
    assert logits.mode.shape == (2, 3, 2)
    assert logits.wait_duration.shape == (2, 3, len(WAIT_DURATIONS))
    assert logits.card.shape == (2, 3, 3)
    # Placement preserves the board-aligned spatial resolution (8x6 here).
    assert logits.placement.shape == (2, 3, 3, 8, 6)
    assert zt.shape == ht.shape == (2, 3, 24)


def test_v4_decode_is_deterministic_and_log_prob_finite() -> None:
    config = _small_config()
    policy = RecurrentV4Policy(config)
    masks = _masks()
    first = policy(*_inputs(config))
    second = policy(*_inputs(config))
    for a, b in zip(first[0].mode.flatten(), second[0].mode.flatten()):
        assert float(a) == float(b)
    actions = policy.act_deterministic(first[0], masks)
    log_prob = policy.log_prob(first[0], masks, actions)
    assert bool(torch.isfinite(log_prob).all())


def test_v4_decode_never_selects_illegal_actions() -> None:
    config = _small_config()
    policy = RecurrentV4Policy(config)
    logits, _, _ = policy(*_inputs(config))
    mode = torch.ones(2, 3, 2, dtype=torch.bool)
    mode[..., 1] = False  # PLAY illegal everywhere: forced WAIT.
    card = torch.ones(2, 3, 3, dtype=torch.bool)
    card[..., 1] = False  # slot 1 illegal: decode must honor the card mask.
    placement = torch.ones(2, 3, 3, 8, 6, dtype=torch.bool)
    masks = ActionMasks(mode=mode, card=card, placement=placement)
    actions = policy.act_deterministic(logits, masks)
    assert bool((actions.mode == 0).all())
    assert bool((actions.card_slot != 1).all())


def test_v4_rejects_empty_distributions() -> None:
    config = _small_config()
    policy = RecurrentV4Policy(config)
    logits, _, _ = policy(*_inputs(config))
    masks = _masks()
    bad_card = torch.zeros(2, 3, 3, dtype=torch.bool)
    bad = ActionMasks(mode=masks.mode, card=bad_card, placement=masks.placement)
    actions = V4ActionBatch(
        mode=torch.ones(2, 3, dtype=torch.long),
        card_slot=torch.zeros(2, 3, dtype=torch.long),
        placement=torch.zeros(2, 3, 2, dtype=torch.long),
        wait_duration=torch.zeros(2, 3, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="no legal action"):
        policy.log_prob(logits, bad, actions)


def test_v4_current_state_path_is_first_class() -> None:
    """Zeroing zt must move every head: memory alone cannot decide."""

    config = _small_config()
    policy = RecurrentV4Policy(config)
    inputs = _inputs(config)
    logits, zt, ht = policy(*inputs)
    _, _, entities, entity_mask, hand, _, _, _, _, _ = inputs
    raster = inputs[0]
    ablated = policy.heads(
        torch.zeros_like(zt),
        ht,
        policy.encoder.entities(entities, entity_mask),
        policy.encoder.hand_projection(hand),
        policy.encoder.spatial(raster)[1],
    )
    assert bool((ablated.mode - logits.mode).abs().sum() > 0)
    assert bool((ablated.card - logits.card).abs().sum() > 0)
    assert bool((ablated.placement - logits.placement).abs().sum() > 0)


def test_v4_production_config_runs_at_arena_resolution() -> None:
    config = ModelConfigV4()
    policy = RecurrentV4Policy(config)
    count = count_parameters(policy)
    assert count > 100_000
    batch, time, entities = 1, 1, 4
    raster = torch.randn(batch, time, 21, 32, 18)
    glob = torch.randn(batch, time, 768)
    ent = torch.randn(batch, time, entities, 32)
    ent_mask = torch.ones(batch, time, entities, dtype=torch.bool)
    hand = torch.randn(batch, time, 4, 16)
    probs = torch.full((batch, time, 128), 1.0 / 128)
    out = torch.zeros(batch, time, 128, dtype=torch.bool)
    elixir = torch.tensor([[[1.0, 9.0]]]).expand(batch, time, 2)
    events = torch.zeros(batch, time, 16, 8)
    reset = torch.ones(batch, time, dtype=torch.bool)
    logits, _, _ = policy(
        raster, glob, ent, ent_mask, hand, probs, out, elixir, events, reset
    )
    assert logits.placement.shape == (1, 1, 4, 32, 18)
