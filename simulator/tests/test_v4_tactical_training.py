"""Regression checks for full-history tactical fitting and exact retention."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy, V4Logits, masks_from_legal_play
from simulator.rl.v4.tactical_training import (
    INPUT_KEYS, TacticalTrainingConfig, adaptive_weights, chunk_batch,
    differentiable_unroll, full_action_kl, tactical_batch_loss, _rows,
)


def logits(seed: int, *, time: int = 3, rows: int = 2, cols: int = 2) -> V4Logits:
    generator = torch.Generator().manual_seed(seed)
    return V4Logits(
        mode=torch.randn(1, time, 2, generator=generator, requires_grad=True),
        wait_duration=torch.randn(1, time, 4, generator=generator, requires_grad=True),
        card=torch.randn(1, time, 4, generator=generator, requires_grad=True),
        placement=torch.randn(1, time, 4, rows, cols, generator=generator, requires_grad=True),
    )


def test_full_joint_kl_matches_explicit_enumeration_at_every_timestep():
    old, new = logits(1), logits(2)
    legal = torch.zeros(1, 3, 4, 2, 2, dtype=torch.bool)
    legal[0, 0, 0, 0, 0] = True
    legal[0, 0, 2, :, :] = True
    legal[0, 1, 3, 1, :] = True
    masks = masks_from_legal_play(legal, True)
    actual = full_action_kl(old, new, masks)
    expected = []
    for step in range(3):
        def joint(values):
            mode = values.mode[0, step].masked_fill(~masks.mode[0, step], -torch.inf).softmax(-1)
            probabilities = [mode[0]]
            if bool(masks.card[0, step].any()):
                cards = values.card[0, step].masked_fill(~masks.card[0, step], -torch.inf).softmax(-1)
                for slot in range(4):
                    if not bool(masks.card[0, step, slot]):
                        continue
                    cells = values.placement[0, step, slot].flatten().masked_fill(~legal[0, step, slot].flatten(), -torch.inf).softmax(-1)
                    probabilities.extend(mode[1] * cards[slot] * cells[legal[0, step, slot].flatten()])
            return torch.stack(probabilities)
        p, q = joint(old), joint(new)
        expected.append((p * (p.log() - q.log())).sum())
    assert actual.shape == (1, 3)
    assert torch.allclose(actual[0], torch.stack(expected), atol=2e-6)
    actual.sum().backward()
    for name in ("mode", "card", "placement"):
        assert bool(torch.isfinite(getattr(new, name).grad).all())
    assert new.wait_duration.grad is None


def test_identical_kl_and_wait_only_empty_conditionals_have_finite_gradients():
    values = logits(1)
    legal = torch.zeros(1, 3, 4, 2, 2, dtype=torch.bool)
    masks = masks_from_legal_play(legal, True)
    kl = full_action_kl(values, values, masks)
    assert torch.equal(kl, torch.zeros_like(kl))
    kl.sum().backward()
    for name in ("mode", "card", "placement"):
        assert bool(torch.isfinite(getattr(values, name).grad).all())


def episode(length: int = 5):
    rng = np.random.default_rng(5)
    specs = dict(raster=(21, 32, 18), global_features=(768,), entities=(128, 32),
                 entity_mask=(128,), hand_tokens=(4, 16), opp_hand_probs=(128,),
                 opp_out_of_cycle=(128,), opp_elixir_interval=(2,), event_history=(16, 8))
    arrays = {key: np.zeros((length, *shape), np.float32) for key, shape in specs.items()}
    for key in ("entity_mask", "opp_out_of_cycle"):
        arrays[key] = arrays[key].astype(bool)
    arrays["raster"][:] = rng.uniform(0, 1, arrays["raster"].shape)
    arrays["global_features"][:] = rng.uniform(0, 1, arrays["global_features"].shape)
    arrays["hand_tokens"][:, :, 0] = np.array((49, 72, 96, 114)) / 127
    arrays["opp_hand_probs"][:] = 1 / 128
    arrays["legal_play"] = np.zeros((length, 4, 32, 18), bool)
    arrays["legal_play"][1:, 1, 22, 5] = True
    arrays["legal_wait"] = np.ones(length, bool)
    arrays["legal_wait"][0] = False
    arrays["reset_mask"] = np.arange(length) == 0
    arrays["label_valid"] = np.arange(length) > 0
    arrays["active"] = np.arange(length) % 2 == 1
    arrays["air_threat"] = arrays["active"].copy()
    arrays["xbow_present"] = np.zeros(length, bool)
    targets = np.zeros((length, 5), np.int64)
    targets[0] = -1
    targets[1:] = (1, 1, 22, 5, 0)
    arrays["targets_cycle"] = targets
    arrays["targets_plans"] = targets.copy()
    arrays["executed_actions"] = targets.clip(0)
    arrays["outcome_values"] = np.full((length, 3, 4), .2, np.float32)
    arrays["outcome_mask"] = np.ones((length, 3, 4), bool)
    return arrays


def compact_policy():
    config = ModelConfigV4(model_dim=8, spatial_channels=4, fused_dim=8,
        gru_hidden_dim=8, transformer_heads=2, transformer_layers=1,
        transformer_ff_dim=16, spatial_head_dim=4, fixed_wait=True)
    return RecurrentV4Policy(config)


def test_chunk_padding_and_countdown_do_not_receive_any_supervision():
    episodes = [episode(5), episode(3)]
    metas = [dict(family="air"), dict(family="air")]
    batch = chunk_batch(episodes, metas, start=0, length=5, arm="control")
    assert int(batch["real"].sum()) == 8
    assert int(batch["valid"].sum()) == 6
    assert int(batch["active"].sum()) == 3
    assert int(batch["inactive"].sum()) == 3
    assert batch["masks"].mode[1, 4, 0]
    assert not batch["valid"][1, 4]
    assert not batch["valid"][:, 0].any()


def test_independent_chronological_carry_matches_full_sequence_and_trains_gru():
    torch.manual_seed(9)
    policy = compact_policy()
    data = episode()
    meta = dict(family="air")
    full = chunk_batch([data], [meta], start=0, length=5, arm="control")
    logits_full, ht_full, _ = differentiable_unroll(policy, full, None, full["reset_mask"])
    first = chunk_batch([data], [meta], start=0, length=2, arm="control")
    _, _, hidden = differentiable_unroll(policy, first, None, first["reset_mask"])
    second = chunk_batch([data], [meta], start=2, length=3, arm="control")
    small, ht_small, _ = differentiable_unroll(policy, second, hidden.detach(), second["reset_mask"])
    assert torch.allclose(ht_small, ht_full[:, 2:], atol=1e-6)
    assert torch.allclose(small.mode, logits_full.mode[:, 2:], atol=1e-6)
    old = V4Logits(*(getattr(logits_full, name).detach().clone() for name in ("mode", "wait_duration", "card", "placement")))
    loss, details, _ = tactical_batch_loss(policy, logits_full, old, ht_full, full, config=TacticalTrainingConfig())
    loss.backward()
    gradients = [parameter.grad for name, parameter in policy.recurrent.named_parameters() if "weight_hh" in name]
    assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert sum(float(gradient.square().sum()) for gradient in gradients) > 0
    assert details["active_rows"] == 2
    assert details["inactive_rows"] == 2
    assert policy.heads.duration_head.weight.grad is None


def test_auxiliary_head_uses_factual_action_and_masks_inactive_padding_rows():
    from simulator.rl.v4.tactical_outcomes import TacticalOutcomeConfig, build_tactical_outcome_head
    torch.manual_seed(11)
    policy = compact_policy()
    data = episode(3)
    # Teacher says PLAY; factual driver chose WAIT. The head must see WAIT.
    data["executed_actions"][:] = 0
    batch = chunk_batch([data], [dict(family="air")], start=0, length=3, arm="outcomes")
    new, ht, _ = differentiable_unroll(policy, batch, None, batch["reset_mask"])
    old = V4Logits(*(getattr(new, name).detach() for name in ("mode", "wait_duration", "card", "placement")))
    head = build_tactical_outcome_head(TacticalOutcomeConfig(recurrent_dim=8, hidden_dim=8))
    encodings = []
    head.register_forward_pre_hook(lambda module, args: encodings.append(args[1].detach().clone()))
    loss, details, _ = tactical_batch_loss(policy, new, old, ht, batch, config=TacticalTrainingConfig(), outcome_head=head)
    loss.backward()
    assert details["outcome_targets"] == 12
    assert torch.equal(encodings[0][..., 0], torch.ones(1, 3))
    assert torch.equal(encodings[0][..., 1:], torch.zeros(1, 3, 7))
    assert sum(float(parameter.grad.square().sum()) for parameter in head.parameters()) > 0


def test_curriculum_bounded_weights_keep_total_active_exposure_and_coverage():
    counts = dict(easy=100, learning=40, impossible=5, fresh=3)
    previous = dict(easy=dict(nll_sum=1., count=100), learning=dict(nll_sum=35., count=40),
                    impossible=dict(nll_sum=100., count=5))
    earlier = dict(learning=dict(nll_sum=55., count=40))
    weights = adaptive_weights(previous, earlier, counts)
    assert weights.keys() == counts.keys()
    assert all(value > 0 for value in weights.values())
    assert sum(weights[key] * count for key, count in counts.items()) == pytest.approx(sum(counts.values()))
    assert weights["learning"] > weights["impossible"]
    assert weights["learning"] > weights["easy"]


def test_episode_order_is_independent_of_worker_completion_order():
    jobs = [dict(job_id="a", split="train"), dict(job_id="b", split="train")]
    a, b = dict(job_id="a", split="train"), dict(job_id="b", split="train")
    assert _rows(dict(jobs=jobs, completed=dict(b=b, a=a)), "train") == [a, b]
    assert _rows(dict(jobs=jobs, completed=dict(a=a, b=b)), "train") == [a, b]


@pytest.mark.parametrize("kwargs", [dict(epochs=0), dict(chunk_length=0), dict(outcome_weight=0),
                                    dict(retention_weight=float("nan")), dict(seed=-1)])
def test_invalid_training_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        TacticalTrainingConfig(**kwargs)
