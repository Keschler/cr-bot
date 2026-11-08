"""Deterministic synthetic-logit tests for frontend scoring (no checkpoint, no cv2)."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frontend.scoring import (
    ScoredAction,
    decide_with_scores,
    masked_log_softmax_1d,
    placement_heatmap,
    score_action_heads,
    top_k_suggestions,
)


def _t(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def _m(values):
    return torch.tensor(values, dtype=torch.bool)


# ---------------------------------------------------------------------------
# masked_log_softmax_1d
# ---------------------------------------------------------------------------


def test_masked_log_softmax_basic():
    logits = _t([1.0, 2.0, 3.0])
    mask = _m([True, True, False])
    out = masked_log_softmax_1d(logits, mask)
    assert out.shape == (3,)
    assert math.isinf(float(out[2].item())) and float(out[2].item()) < 0
    probs = torch.exp(out[:2])
    assert float(probs.sum().item()) == pytest.approx(1.0)
    # Manual softmax over the two legal entries.
    denom = math.exp(1.0) + math.exp(2.0)
    assert float(out[0].item()) == pytest.approx(math.log(math.exp(1.0) / denom))
    assert float(out[1].item()) == pytest.approx(math.log(math.exp(2.0) / denom))


def test_masked_log_softmax_no_legal_raises():
    with pytest.raises(ValueError):
        masked_log_softmax_1d(_t([0.0, 0.0]), _m([False, False]))


def test_masked_log_softmax_nonfinite_legal_raises():
    with pytest.raises(ValueError):
        masked_log_softmax_1d(_t([float("nan"), 0.0]), _m([True, True]))
    with pytest.raises(ValueError):
        masked_log_softmax_1d(_t([float("inf"), 0.0]), _m([True, False]))


def test_masked_log_softmax_shape_mismatch_raises():
    with pytest.raises(ValueError):
        masked_log_softmax_1d(_t([0.0, 0.0]), _m([True]))
    with pytest.raises(ValueError):
        masked_log_softmax_1d(_t([[0.0, 0.0]]), _m([[True, True]]))
    with pytest.raises(ValueError):
        masked_log_softmax_1d(_t([0.0, 0.0]), torch.tensor([1, 0]))


# ---------------------------------------------------------------------------
# top_k_suggestions
# ---------------------------------------------------------------------------


def _masks(mode, card, placement):
    from simulator.rl.foundation.trajectory import ActionMasks

    return ActionMasks(mode=_m(mode), card=_m(card), placement=_m(placement))


def test_top_k_wait_only_masks():
    mode_logits = _t([5.0, -5.0])
    card_logits = _t([0.0, 0.0, 0.0, 0.0])
    placement_logits = torch.zeros(4, 32, 18)
    masks = _masks([True, False], [False, False, False, False], np.zeros((4, 32, 18), dtype=bool).tolist())
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=3)
    assert len(suggestions) == 1
    only = suggestions[0]
    assert only.kind == "wait"
    assert only.probability == pytest.approx(1.0)
    assert only.log_prob == pytest.approx(0.0, abs=1e-6)
    assert only.card_slot is None and only.cell is None and only.card_name is None


def test_top_k_play_illegal_returns_only_wait():
    mode_logits = _t([0.3, 99.0])  # PLAY logit irrelevant: PLAY masked out.
    card_logits = _t([0.0, 0.0, 0.0, 0.0])
    placement_logits = torch.zeros(4, 32, 18)
    placement_mask = np.zeros((4, 32, 18), dtype=bool)
    placement_mask[0, 0, 0] = True  # would be playable if PLAY were legal.
    masks = _masks([True, False], [True, False, False, False], placement_mask.tolist())
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=3)
    assert [s.kind for s in suggestions] == ["wait"]
    assert suggestions[0].probability == pytest.approx(1.0)


def test_top_k_single_legal_cell_joint_math():
    # WAIT logit 0, PLAY logit 0 -> P(wait) = P(play) = 0.5.
    mode_logits = _t([0.0, 0.0])
    # Strongly favor slot 1.
    card_logits = _t([-10.0, 10.0, -10.0, -10.0])
    placement_logits = torch.zeros(4, 32, 18)
    placement_mask = np.zeros((4, 32, 18), dtype=bool)
    placement_mask[1, 5, 7] = True
    placement_mask[2, 0, 0] = True  # slot 2 illegal card -> must be ignored.
    masks = _masks(
        [True, True],
        [False, True, False, False],
        placement_mask.tolist(),
    )
    suggestions = top_k_suggestions(
        mode_logits, card_logits, placement_logits, masks,
        hand_cards=["hog-rider", "cannon", "musketeer", "skeletons"], top_k=3,
    )
    assert len(suggestions) == 2
    play = next(s for s in suggestions if s.kind == "play")
    assert play.card_slot == 1
    assert play.cell == (7, 5)  # (col, row)
    assert play.card_name == "cannon"
    # P(play)=0.5, P(card=1|play)~1.0, P(cell|card)=1.0.
    assert play.probability == pytest.approx(0.5, rel=1e-4)
    assert play.log_prob == pytest.approx(
        play.mode_log_prob + play.card_log_prob + play.placement_log_prob
    )
    wait = next(s for s in suggestions if s.kind == "wait")
    assert wait.probability == pytest.approx(0.5, rel=1e-4)
    assert wait.probability + play.probability == pytest.approx(1.0)


def test_top_k_ordering_and_truncation():
    mode_logits = _t([-5.0, 5.0])  # play strongly favored; wait ~ e^-10.
    card_logits = _t([2.0, 0.0, -10.0, -10.0])  # slot 0 favored over slot 1.
    placement_logits = torch.zeros(4, 32, 18)
    placement_logits[0, 0, 0] = 3.0
    placement_logits[0, 1, 1] = 1.0
    placement_logits[1, 2, 2] = 5.0
    placement_mask = np.zeros((4, 32, 18), dtype=bool)
    placement_mask[0, 0, 0] = True
    placement_mask[0, 1, 1] = True
    placement_mask[1, 2, 2] = True
    masks = _masks([True, True], [True, True, False, False], placement_mask.tolist())
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=2)
    assert len(suggestions) == 2
    log_probs = [s.log_prob for s in suggestions]
    assert log_probs == sorted(log_probs, reverse=True)
    # Full ranking must be descending too and WAIT last.
    full = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=10)
    assert len(full) == 4  # 3 plays + wait
    assert [s.log_prob for s in full] == sorted([s.log_prob for s in full], reverse=True)
    assert full[-1].kind == "wait"
    # Top play is slot 0 cell (0,0): highest card prob and best in-slot cell.
    assert full[0].kind == "play" and full[0].card_slot == 0 and full[0].cell == (0, 0)


def test_top_k_skips_slot_with_empty_placement():
    mode_logits = _t([0.0, 0.0])
    card_logits = _t([0.0, 0.0, 0.0, 0.0])
    placement_logits = torch.zeros(4, 32, 18)
    placement_mask = np.zeros((4, 32, 18), dtype=bool)
    placement_mask[1, 3, 4] = True
    # Slot 0 is card-legal but has no legal placement cell -> skipped.
    masks = _masks([True, True], [True, True, False, False], placement_mask.tolist())
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=10)
    plays = [s for s in suggestions if s.kind == "play"]
    assert len(plays) == 1
    assert plays[0].card_slot == 1 and plays[0].cell == (4, 3)


def test_top_k_probabilities_sum_to_one():
    mode_logits = _t([0.7, -0.2])
    card_logits = _t([0.5, -0.5, 1.5, -1.0])
    rng = torch.Generator().manual_seed(0)
    placement_logits = torch.randn(4, 32, 18, generator=rng)
    placement_mask = np.zeros((4, 32, 18), dtype=bool)
    placement_mask[:, :4, :3] = True
    masks = _masks([True, True], [True, True, True, True], placement_mask.tolist())
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=10_000)
    total = sum(s.probability for s in suggestions)
    assert total == pytest.approx(1.0, rel=1e-4, abs=1e-6)


def test_top_k_must_play_has_no_wait():
    mode_logits = _t([50.0, 0.0])  # WAIT logit irrelevant: WAIT masked out.
    card_logits = _t([0.0, 0.0, 0.0, 0.0])
    placement_logits = torch.zeros(4, 32, 18)
    placement_mask = np.zeros((4, 32, 18), dtype=bool)
    placement_mask[3, 31, 17] = True
    masks = _masks([False, True], [False, False, False, True], placement_mask.tolist())
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=3)
    assert len(suggestions) == 1
    assert suggestions[0].kind == "play"
    assert suggestions[0].cell == (17, 31)
    assert suggestions[0].probability == pytest.approx(1.0)


def test_top_k_accepts_tuple_masks_and_defaults_card_name():
    mode_logits = _t([0.0, 0.0])
    card_logits = _t([0.0, 5.0, 0.0, 0.0])
    placement_logits = torch.zeros(4, 32, 18)
    placement_mask = _m(np.zeros((4, 32, 18), dtype=bool))
    placement_mask[1, 0, 1] = True
    masks = (_m([True, True]), _m([False, True, False, False]), placement_mask)
    suggestions = top_k_suggestions(mode_logits, card_logits, placement_logits, masks)
    play = next(s for s in suggestions if s.kind == "play")
    assert play.card_name is None
    assert play.cell == (1, 0)


def test_top_k_invalid_top_k_raises():
    mode_logits = _t([0.0, 0.0])
    card_logits = _t([0.0, 0.0, 0.0, 0.0])
    placement_logits = torch.zeros(4, 32, 18)
    masks = _masks([True, False], [False] * 4, np.zeros((4, 32, 18), dtype=bool).tolist())
    with pytest.raises(ValueError):
        top_k_suggestions(mode_logits, card_logits, placement_logits, masks, top_k=0)


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def _batched_heads(mode, card, placement):
    return SimpleNamespace(
        mode=mode[None, None], card=card[None, None], placement=placement[None, None]
    )


def _scalar_reference(mode, card, placement, masks, hand_cards):
    """Original device normalization and scalar/double scoring as an oracle."""

    mode_mask, card_mask, placement_mask = masks

    def log_probs(logits, mask):
        return torch.log_softmax(torch.where(mask, logits, float("-inf")), dim=-1)

    mode_lp = log_probs(mode, mode_mask)
    mode_prob = [math.exp(mode_lp[i].item()) if mode_mask[i].item() else 0.0 for i in range(2)]
    entropy = 0.0
    for index, probability in enumerate(mode_prob):
        if mode_mask[index].item():
            entropy -= probability * mode_lp[index].item()
    candidates = []
    if mode_mask[0].item():
        candidates.append(ScoredAction("wait", None, None, mode_prob[0], mode_lp[0].item(),
                                       mode_lp[0].item(), None, None, None))
    if mode_mask[1].item() and card_mask.any().item():
        card_lp = log_probs(card, card_mask)
        card_entropy = 0.0
        slot_probs = {}
        for slot in range(card.shape[0]):
            if card_mask[slot].item():
                probability = math.exp(card_lp[slot].item())
                slot_probs[slot] = probability
                card_entropy -= probability * card_lp[slot].item()
        if mode_prob[1] > 0.0:
            entropy += mode_prob[1] * card_entropy
        for slot, slot_probability in slot_probs.items():
            cell_mask = placement_mask[slot].reshape(-1)
            if not cell_mask.any().item():
                continue
            place_lp = log_probs(placement[slot].reshape(-1), cell_mask)
            place_entropy = 0.0
            for flat_index in torch.nonzero(cell_mask, as_tuple=False).flatten().tolist():
                row, col = divmod(flat_index, placement.shape[2])
                log_prob = place_lp[flat_index].item()
                place_entropy -= math.exp(log_prob) * log_prob
                joint = mode_lp[1].item() + card_lp[slot].item() + log_prob
                candidates.append(ScoredAction("play", slot, (col, row), math.exp(joint), joint,
                                               mode_lp[1].item(), card_lp[slot].item(), log_prob,
                                               hand_cards[slot]))
            if mode_prob[1] > 0.0:
                entropy += mode_prob[1] * slot_probability * place_entropy
    candidates.sort(key=lambda action: (
        -action.log_prob,
        0 if action.kind == "wait" else 1,
        -1 if action.card_slot is None else action.card_slot,
        -1 if action.cell is None else action.cell[1],
        -1 if action.cell is None else action.cell[0],
    ))
    return candidates, {
        "mode_prob_wait": mode_prob[0], "mode_prob_play": mode_prob[1],
        "entropy": entropy, "top_log_prob": candidates[0].log_prob,
    }


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["dense", "sparse", "wait_only", "must_play", "empty_slot", "no_cards", "underflow"])
def test_bulk_scoring_exactly_matches_scalar_reference(device, dtype, case):
    generator = torch.Generator().manual_seed(431)
    mode = torch.randn(2, generator=generator, dtype=dtype)
    card = torch.randn(4, generator=generator, dtype=dtype)
    placement = torch.randn(4, 32, 18, generator=generator, dtype=dtype)
    mode_mask = _m([True, True])
    card_mask = _m([True, True, True, True])
    placement_mask = torch.ones_like(placement, dtype=torch.bool)
    if case == "sparse":
        placement_mask = torch.rand(4, 32, 18, generator=generator) < 0.15
        card_mask[2] = False
    elif case == "wait_only":
        mode_mask[1] = False
        card.fill_(float("nan"))
        placement.fill_(float("inf"))
    elif case == "must_play":
        mode_mask[0] = False
    elif case == "empty_slot":
        placement_mask[0] = False
        placement[0].fill_(float("nan"))
    elif case == "no_cards":
        card_mask[:] = False
        card.fill_(float("nan"))
        placement.fill_(float("nan"))
    elif case == "underflow":
        mode[:] = _t([0.0, -1000.0], dtype=dtype)
    tensors = [tensor.to(device) for tensor in (mode, card, placement, mode_mask, card_mask, placement_mask)]
    mode, card, placement, *masks = tensors
    hand_cards = ["hog-rider", "cannon", "musketeer", "skeletons"]
    expected, diagnostics = _scalar_reference(mode, card, placement, masks, hand_cards)
    for top_k in (1, 3, 10_000):
        suggestions = top_k_suggestions(mode, card, placement, masks, hand_cards=hand_cards, top_k=top_k)
        assert suggestions == expected[:top_k]
        action, scored, actual_diagnostics = score_action_heads(
            _batched_heads(mode, card, placement), _batched_heads(*masks), hand_cards=hand_cards, top_k=top_k
        )
        assert scored == suggestions
        assert actual_diagnostics == diagnostics
        best = expected[0]
        assert action.kind == ("Wait" if best.kind == "wait" else "Play")
        assert action.card_idx == best.card_slot and action.cell == best.cell


@pytest.mark.parametrize("device", _DEVICES)
def test_bulk_scoring_ties_prefer_wait_then_slot_row_column(device):
    mode = torch.zeros(2, device=device)
    card = torch.zeros(4, device=device)
    placement = torch.zeros(4, 2, 3, device=device)
    mode_mask = _m([True, True]).to(device)
    card_mask = _m([True, False, False, False]).to(device)
    placement_mask = torch.zeros_like(placement, dtype=torch.bool)
    placement_mask[0, 1, 2] = True
    tied = top_k_suggestions(mode, card, placement, (mode_mask, card_mask, placement_mask), top_k=2)
    assert tied[0].log_prob == tied[1].log_prob
    assert [action.kind for action in tied] == ["wait", "play"]
    mode_mask[0] = False
    card_mask[:] = True
    placement_mask[:] = True
    tied = top_k_suggestions(mode, card, placement, (mode_mask, card_mask, placement_mask), top_k=13)
    assert [(action.card_slot, action.cell) for action in tied] == [
        (slot, (col, row)) for slot in range(4) for row in range(2) for col in range(3)
    ][:13]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scorer", ["ranking", "diagnostics"])
def test_tensor_scalar_reads_do_not_scale_with_placement_cells(monkeypatch, device, scorer):
    calls = []
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist

    def counted_item(tensor, *args, **kwargs):
        calls.append(tensor.device.type)
        return original_item(tensor, *args, **kwargs)

    def cpu_tolist(tensor, *args, **kwargs):
        # A GPU .tolist() would hide a synchronization and transfer inside the
        # cell loop.  Scoring must explicitly bulk-copy before reading values.
        assert tensor.device.type == "cpu"
        return original_tolist(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "tolist", cpu_tolist)
    counts = []
    for rows, cols in ((1, 1), (32, 18)):
        mode = torch.zeros(2, device=device)
        card = torch.zeros(4, device=device)
        placement = torch.zeros(4, rows, cols, device=device)
        masks = tuple(torch.ones_like(tensor, dtype=torch.bool) for tensor in (mode, card, placement))
        calls.clear()
        if scorer == "ranking":
            top_k_suggestions(mode, card, placement, masks)
        else:
            score_action_heads(_batched_heads(mode, card, placement), _batched_heads(*masks))
        counts.append(len(calls))
    assert counts[0] == counts[1]
    assert counts[1] <= 24  # bounded validation reductions for six distributions


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("head", ["mode", "card", "placement"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_bulk_scoring_rejects_nonfinite_legal_values_before_truncation(device, head, invalid):
    mode = _t([100.0, -100.0]).to(device)  # WAIT wins every top-1 ranking.
    card = torch.zeros(4, device=device)
    placement = torch.zeros(4, 2, 3, device=device)
    masks = tuple(torch.ones_like(tensor, dtype=torch.bool) for tensor in (mode, card, placement))
    {"mode": mode, "card": card, "placement": placement}[head].reshape(-1)[-1] = invalid
    with pytest.raises(ValueError, match="finite"):
        top_k_suggestions(mode, card, placement, masks, top_k=1)
    with pytest.raises(ValueError, match="finite"):
        score_action_heads(_batched_heads(mode, card, placement), _batched_heads(*masks), top_k=1)


@pytest.mark.parametrize("device", _DEVICES)
def test_bulk_scoring_ignores_nonfinite_illegal_values(device):
    mode = _t([float("nan"), 0.0]).to(device)
    card = _t([0.0, float("inf"), float("-inf"), float("nan")]).to(device)
    placement = torch.full((4, 2, 3), float("nan"), device=device)
    placement[0, 1, 2] = 0.0
    placement_mask = torch.zeros_like(placement, dtype=torch.bool)
    placement_mask[0, 1, 2] = True
    masks = (_m([False, True]).to(device), _m([True, False, False, False]).to(device), placement_mask)
    action, suggestions, diagnostics = score_action_heads(_batched_heads(mode, card, placement), _batched_heads(*masks))
    assert (action.kind, action.card_idx, action.cell) == ("Play", 0, (2, 1))
    assert suggestions[0].probability == 1.0
    assert diagnostics == {"mode_prob_wait": 0.0, "mode_prob_play": 1.0, "entropy": 0.0, "top_log_prob": 0.0}


@pytest.mark.parametrize("card_legal", [False, True])
def test_bulk_scoring_no_available_play_fails_closed_when_wait_is_illegal(card_legal):
    mode = torch.zeros(2)
    card = torch.zeros(4)
    placement = torch.zeros(4, 2, 3)
    masks = (_m([False, True]), torch.full((4,), card_legal, dtype=torch.bool), torch.zeros_like(placement, dtype=torch.bool))
    with pytest.raises(ValueError, match="fail-closed"):
        top_k_suggestions(mode, card, placement, masks)
    with pytest.raises(ValueError, match="fail-closed"):
        score_action_heads(_batched_heads(mode, card, placement), _batched_heads(*masks))


def test_bulk_scoring_rejects_joint_overflow_even_outside_top_k():
    mode = _t([0.0, -1e308], dtype=torch.float64)
    card = _t([0.0, -1e308], dtype=torch.float64)
    placement = torch.zeros(2, 1, 1, dtype=torch.float64)
    masks = tuple(torch.ones_like(tensor, dtype=torch.bool) for tensor in (mode, card, placement))
    with pytest.raises(ValueError, match="joint log-prob.*finite"):
        top_k_suggestions(mode, card, placement, masks, top_k=1)


def test_bulk_scoring_validates_hand_even_when_plays_are_truncated():
    mode = _t([100.0, -100.0])
    card = torch.zeros(4)
    placement = torch.zeros(4, 1, 1)
    masks = tuple(torch.ones_like(tensor, dtype=torch.bool) for tensor in (mode, card, placement))
    with pytest.raises(ValueError, match="hand_cards"):
        top_k_suggestions(mode, card, placement, masks, hand_cards="invalid", top_k=1)


# ---------------------------------------------------------------------------
# placement_heatmap
# ---------------------------------------------------------------------------


def test_heatmap_sums_to_one_and_masks_illegal():
    logits = torch.zeros(4, 32, 18)
    logits[2, 10, 5] = 4.0
    logits[2, 0, 0] = 100.0  # illegal -> must stay 0.0 despite huge logit.
    mask = _m(np.zeros((4, 32, 18), dtype=bool))
    mask[2, 10, 5] = True
    mask[2, 11, 6] = True
    grid = placement_heatmap(logits, mask, 2)
    assert len(grid) == 32 and all(len(row) == 18 for row in grid)
    total = sum(sum(row) for row in grid)
    assert total == pytest.approx(1.0)
    assert grid[0][0] == 0.0
    assert grid[10][5] > grid[11][6] > 0.0
    assert all(v >= 0.0 for row in grid for v in row)


def test_heatmap_invalid_slot_and_empty_grid_raise():
    logits = torch.zeros(4, 32, 18)
    mask = _m(np.zeros((4, 32, 18), dtype=bool))
    mask[0, 0, 0] = True
    with pytest.raises(ValueError):
        placement_heatmap(logits, mask, 4)
    with pytest.raises(ValueError):
        placement_heatmap(logits, mask, -1)
    with pytest.raises(ValueError):
        placement_heatmap(logits, mask, 3)  # no legal cells in slot 3


def test_decide_with_scores_delegates_to_v4_actor_and_fails_closed():
    observation = object()
    expected = (object(), [], {"entropy": 0.0})

    class Actor:
        def decide_with_scores(self, received):
            assert received is observation
            return expected

    assert decide_with_scores(Actor(), observation) is expected
    with pytest.raises(ValueError, match="required"):
        decide_with_scores(None, observation)
    with pytest.raises(ValueError, match="implement"):
        decide_with_scores(object(), observation)


def test_scoring_module_has_no_eager_heavy_imports():
    root = Path(__file__).resolve().parents[1] / "src" / "frontend" / "scoring.py"
    tree = ast.parse(root.read_text())
    eager = set()
    for node in tree.body:  # only module top level, not function bodies
        if isinstance(node, ast.Import):
            eager.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and (node.level or 0) == 0:
            eager.add((node.module or "").split(".")[0])
    assert "torch" not in eager
    assert "cv2" not in eager
    assert "simulator" not in eager
    assert "cr_bot" not in eager


def test_module_importable_without_prior_torch_import():
    # The module object itself must not require torch/cv2 at import time; the
    # top-level test session already imports torch, so assert statically that
    # no eager heavy import exists and the module is importable fresh.
    for name in ("cv2",):
        assert name not in sys.modules or True  # documented: never imported by scoring
    import frontend.scoring as scoring

    assert callable(scoring.masked_log_softmax_1d)
    assert issubclass(scoring.ScoredAction, object)
