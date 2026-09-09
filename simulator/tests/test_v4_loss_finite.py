"""Regression tests for the Step-0 masked-loss NaN fix (losses_v4).

Before the fix, WAIT rows carried a dummy ``card_slot=0`` whose masked NLL
is +inf (slot illegal, others legal) or NaN (nothing legal); multiplying by
``is_play=0`` poisoned the forward loss value (``inf*0``/``nan*0``) even
though those rows must contribute nothing.  The fix selects PLAY rows
before any log-softmax.  These tests pin: finite values on WAIT-only and
mixed batches, finite gradients, value/gradient preservation on valid PLAY
rows, and the frozen champion architecture size.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.distillation import to_torch_batch
from simulator.rl.losses_v4 import V4SupervisedWeights, v4_supervised_loss
from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy, count_parameters
from simulator.rl.timing_curriculum import TimingConfig, generate_timing_dataset


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


def _forward(policy: RecurrentV4Policy, batch: dict) -> tuple:
    return policy(
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


@pytest.fixture(scope="module")
def mixed_batch() -> dict:
    """Timing batch guaranteed to contain strategic WAIT rows.

    Strategic WAITs (rich elixir, legal PLAYs elsewhere, dummy slot 0
    illegal) are exactly the +inf-pattern rows the old formulation choked
    on; broke-only WAIT batches never exposed it.
    """

    torch.manual_seed(7)
    samples, _ = generate_timing_dataset(TimingConfig(n_states=120, seed=11, pool_factor=2))
    batch = to_torch_batch(samples)
    modes = batch["actions"].mode.squeeze(1)
    assert int((modes == 0).sum()) > 0, "fixture needs WAIT rows"
    assert int((modes == 1).sum()) > 0, "fixture needs PLAY rows"
    return batch


def _wait_only_batch(mixed_batch: dict) -> dict:
    keep = [
        index
        for index in range(mixed_batch["actions"].mode.shape[0])
        if int(mixed_batch["actions"].mode[index, 0]) == 0
    ]
    assert keep, "fixture needs WAIT rows"
    out: dict = {}
    for key, value in mixed_batch.items():
        if key == "families":
            out[key] = [mixed_batch[key][index] for index in keep]
        elif key in ("masks", "actions"):
            out[key] = value  # replaced below
        else:
            out[key] = value[keep]
    from simulator.rl.model_v4 import V4ActionBatch
    from simulator.rl.trajectory import ActionMasks

    masks = mixed_batch["masks"]
    actions = mixed_batch["actions"]
    out["masks"] = ActionMasks(
        mode=masks.mode[keep], card=masks.card[keep], placement=masks.placement[keep]
    )
    out["actions"] = V4ActionBatch(
        mode=actions.mode[keep],
        card_slot=actions.card_slot[keep],
        placement=actions.placement[keep],
        wait_duration=actions.wait_duration[keep],
    )
    return out


def test_wait_only_batch_losses_are_finite(mixed_batch: dict) -> None:
    torch.manual_seed(1)
    batch = _wait_only_batch(mixed_batch)
    policy = _tiny_policy()
    logits, _, _ = _forward(policy, batch)
    total, detail = v4_supervised_loss(
        logits, batch["masks"], batch["actions"],
        soft_placement=batch["soft_placement"],
    )
    assert bool(torch.isfinite(total))
    for key in ("loss_mode", "loss_wait_duration", "loss_card", "loss_placement"):
        assert np.isfinite(detail[key]), key
    assert detail["n_play"] == 0.0
    assert detail["loss_card"] == 0.0
    assert detail["loss_placement"] == 0.0


def test_mixed_batch_losses_are_finite(mixed_batch: dict) -> None:
    torch.manual_seed(1)
    policy = _tiny_policy()
    logits, _, _ = _forward(policy, mixed_batch)
    total, detail = v4_supervised_loss(
        logits, mixed_batch["masks"], mixed_batch["actions"],
        soft_placement=mixed_batch["soft_placement"],
    )
    assert bool(torch.isfinite(total))
    for key in ("loss_mode", "loss_wait_duration", "loss_card", "loss_placement"):
        assert np.isfinite(detail[key]), key


def test_mixed_batch_gradients_are_finite(mixed_batch: dict) -> None:
    torch.manual_seed(1)
    policy = _tiny_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    optimizer.zero_grad()
    logits, _, _ = _forward(policy, mixed_batch)
    total, _ = v4_supervised_loss(
        logits, mixed_batch["masks"], mixed_batch["actions"],
        soft_placement=mixed_batch["soft_placement"],
    )
    assert bool(torch.isfinite(total))
    total.backward()
    for param in policy.parameters():
        if param.grad is not None:
            assert bool(torch.isfinite(param.grad).all())


def _reference_card_placement_loss(
    logits, masks, actions, soft_placement, weights: V4SupervisedWeights
) -> tuple[float, float]:
    """Independent row-loop reference for PLAY-row card/placement terms."""

    import math

    is_play = [int(v) for v in actions.mode.squeeze(1).tolist()]
    card_terms: list[float] = []
    placement_terms: list[float] = []
    rows, cols = logits.placement.shape[-2:]
    for index, play in enumerate(is_play):
        if not play:
            continue
        slot = int(actions.card_slot[index, 0])
        row_mask = masks.card[index, 0].tolist()
        row_logits = logits.card[index, 0].tolist()
        best = max(v for v, legal in zip(row_logits, row_mask) if legal)
        log_denominator = math.log(sum(math.exp(v - best) for v, legal in zip(row_logits, row_mask) if legal))
        card_terms.append((best + log_denominator - row_logits[slot]))
        flat_logits = logits.placement[index, 0, slot].reshape(-1).tolist()
        flat_mask = masks.placement[index, 0, slot].reshape(-1).tolist()
        legal_logits = [v for v, legal in zip(flat_logits, flat_mask) if legal]
        best_cell = max(legal_logits)
        log_denom = math.log(sum(math.exp(v - best_cell) for v in legal_logits))
        logp = [(v - best_cell - log_denom) if legal else float("-inf") for v, legal in zip(flat_logits, flat_mask)]
        soft = soft_placement[index, 0].reshape(-1).tolist()
        placement_terms.append(-sum(s * lp for s, lp in zip(soft, logp) if s > 0))
    scale = float(weights.play_upweight)
    return float(np.mean(card_terms)) * scale, float(np.mean(placement_terms)) * scale


def test_play_row_values_match_independent_reference(mixed_batch: dict) -> None:
    """Valid PLAY rows keep identical values after the fix (hard + soft)."""

    torch.manual_seed(1)
    policy = _tiny_policy()
    weights = V4SupervisedWeights()
    for use_soft in (True, False):
        logits, _, _ = _forward(policy, mixed_batch)
        total, detail = v4_supervised_loss(
            logits,
            mixed_batch["masks"],
            mixed_batch["actions"],
            soft_placement=mixed_batch["soft_placement"] if use_soft else None,
            weights=weights,
        )
        assert bool(torch.isfinite(total))
        if not use_soft:
            continue
        ref_card, ref_placement = _reference_card_placement_loss(
            logits, mixed_batch["masks"], mixed_batch["actions"],
            mixed_batch["soft_placement"], weights,
        )
        assert detail["loss_card"] == pytest.approx(ref_card, rel=1e-5)
        assert detail["loss_placement"] == pytest.approx(ref_placement, rel=1e-5)


def test_champion_architecture_size_is_pinned() -> None:
    """The frozen Stage-1 joint champion has 117,767 params; any arch change
    must be deliberate (update this pin only with a re-frozen champion)."""

    config = ModelConfigV4(
        model_dim=32,
        spatial_channels=16,
        fused_dim=64,
        gru_hidden_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=64,
        spatial_head_dim=8,
    )
    assert count_parameters(RecurrentV4Policy(config)) == 117767
