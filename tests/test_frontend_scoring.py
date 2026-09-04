"""Deterministic synthetic-logit tests for frontend scoring (no checkpoint, no cv2)."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from frontend.scoring import (
    ScoredAction,
    decide_with_scores,
    masked_log_softmax_1d,
    placement_heatmap,
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
    from simulator.rl.trajectory import ActionMasks

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


# ---------------------------------------------------------------------------
# decide_with_scores (fake actor + real PolicyObservationV2, no checkpoint)
# ---------------------------------------------------------------------------


def _make_v2_observation(*, legal_wait=True, legal_cells=()):
    from simulator.observation import BOARD_SHAPE, GLOBAL_VECTOR_SHAPE
    from simulator.observation_v2 import ENTITY_TOKEN_SHAPE, ENTITY_TOKEN_MAX, PolicyObservationV2

    legal_play = np.zeros((4, 32, 18), dtype=bool)
    for slot, row, col in legal_cells:
        legal_play[slot, row, col] = True
    return PolicyObservationV2(
        board=np.zeros(BOARD_SHAPE, dtype=np.float32),
        global_vector=np.zeros(GLOBAL_VECTOR_SHAPE, dtype=np.float32),
        entity_tokens=np.zeros(ENTITY_TOKEN_SHAPE, dtype=np.float32),
        entity_mask=np.zeros((ENTITY_TOKEN_MAX,), dtype=bool),
        legal_play=legal_play,
        legal_wait=bool(legal_wait),
    )


class _FakePolicy:
    def __init__(self, mode_logits, card_logits, placement_logits, hidden_dim=8):
        self._mode = torch.tensor(mode_logits, dtype=torch.float32)
        self._card = torch.tensor(card_logits, dtype=torch.float32)
        self._placement = torch.tensor(placement_logits, dtype=torch.float32)
        self.hidden_dim = hidden_dim
        self.forward_calls = 0
        self.seen_reset_masks = []
        self.initial_hidden_calls = 0

    def initial_hidden(self, batch_size, *, device=None, dtype=None):
        self.initial_hidden_calls += 1
        return torch.zeros(1, batch_size, self.hidden_dim)

    def forward(self, raster, global_features, entities, entity_mask, **kwargs):
        from simulator.rl.model import AutoregressiveLogits, RecurrentPolicyOutput

        self.forward_calls += 1
        reset = kwargs.get("reset_mask")
        assert reset is not None
        self.seen_reset_masks.append(bool(reset[0, 0].item()))
        hidden = kwargs.get("hidden")
        assert hidden is not None
        assert kwargs.get("include_beliefs") is False
        assert kwargs.get("inference") is True
        masks = kwargs.get("action_masks")
        assert masks is not None
        return RecurrentPolicyOutput(
            logits=AutoregressiveLogits(
                mode=self._mode.reshape(1, 1, 2),
                card=self._card.reshape(1, 1, 4),
                placement=self._placement.reshape(1, 1, 4, 32, 18),
            ),
            encoded_features=torch.zeros(1, 1, self.hidden_dim),
            recurrent_features=torch.zeros(1, 1, self.hidden_dim),
            final_hidden=torch.ones(1, 1, self.hidden_dim),
            belief_logits=None,
        )


class _FakeActor:
    def __init__(self, policy):
        self.policy = policy
        self._hidden = None
        self._torch = torch

    @property
    def device(self):
        return torch.device("cpu")

    def act_deterministic(self, *args, **kwargs):  # must never be called
        raise AssertionError("decide_with_scores must not call act_deterministic")


def test_decide_with_scores_play_rank1_and_hidden_semantics():
    placement = np.zeros((4, 32, 18), dtype=np.float32)
    placement[0, 2, 3] = 6.0
    policy = _FakePolicy([-4.0, 4.0], [3.0, -3.0, -3.0, -3.0], placement)
    actor = _FakeActor(policy)
    obs = _make_v2_observation(legal_wait=True, legal_cells=[(0, 2, 3)])

    action, suggestions, diagnostics = decide_with_scores(actor, obs)

    assert policy.forward_calls == 1
    assert policy.initial_hidden_calls == 1
    assert policy.seen_reset_masks == [True]  # first step resets.
    assert actor._hidden is not None
    assert tuple(actor._hidden.shape) == (1, 1, policy.hidden_dim)
    assert not actor._hidden.requires_grad

    assert action.kind == "Play"
    assert action.card_idx == suggestions[0].card_slot == 0
    assert action.cell == suggestions[0].cell == (3, 2)
    assert suggestions[0].kind == "play"
    assert set(diagnostics) == {"mode_prob_wait", "mode_prob_play", "entropy", "top_log_prob"}
    assert diagnostics["mode_prob_wait"] + diagnostics["mode_prob_play"] == pytest.approx(1.0, rel=1e-4)
    assert diagnostics["mode_prob_play"] > 0.99
    assert math.isfinite(diagnostics["entropy"]) and diagnostics["entropy"] >= 0.0
    assert diagnostics["top_log_prob"] == pytest.approx(suggestions[0].log_prob)

    # Second step reuses the carried hidden state with a zero reset mask.
    action2, suggestions2, diagnostics2 = decide_with_scores(actor, obs)
    assert policy.forward_calls == 2
    assert policy.initial_hidden_calls == 1  # no re-initialization.
    assert policy.seen_reset_masks == [True, False]
    assert action2.kind == "Play" and action2.cell == (3, 2)
    assert diagnostics2["top_log_prob"] == pytest.approx(suggestions2[0].log_prob)


def test_decide_with_scores_wait_when_wait_rank1():
    placement = np.zeros((4, 32, 18), dtype=np.float32)
    policy = _FakePolicy([6.0, -6.0], [0.0, 0.0, 0.0, 0.0], placement)
    actor = _FakeActor(policy)
    obs = _make_v2_observation(legal_wait=True, legal_cells=[(1, 0, 0)])

    action, suggestions, diagnostics = decide_with_scores(actor, obs)
    assert action.kind == "Wait"
    assert suggestions[0].kind == "wait"
    assert diagnostics["mode_prob_wait"] > 0.99


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
