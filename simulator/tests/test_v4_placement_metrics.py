from __future__ import annotations

import numpy as np
import pytest

from simulator.evaluation import (
    per_sample_placement_stats,
    placement_error_stats,
    soft_region_mass,
    top1_cells,
)


def _maps() -> np.ndarray:
    """Two samples, two cards, 4x5 grids with known argmax cells."""

    maps = np.zeros((2, 2, 4, 5), dtype=np.float64)
    maps[0, 0, 1, 2] = 0.9  # sample 0, slot 0 -> (row 1, col 2)
    maps[0, 0, 1, 3] = 0.1
    maps[1, 1, 3, 4] = 1.0  # sample 1, slot 1 -> (row 3, col 4)
    return maps


def _norm_maps() -> np.ndarray:
    """Probability maps with uniform fill on unused card slots."""

    maps = _maps()
    maps[0, 1] = 1.0
    maps[1, 0] = 1.0
    return maps / maps.sum(axis=(2, 3), keepdims=True)


def test_top1_cells_selects_card_map() -> None:
    rows, cols = top1_cells(_maps(), np.array([0, 1]))
    assert rows.tolist() == [1, 3]
    assert cols.tolist() == [2, 4]
    with pytest.raises(ValueError, match="out-of-range"):
        top1_cells(_maps(), np.array([0, 2]))
    with pytest.raises(ValueError, match="must not be empty"):
        top1_cells(np.zeros((0, 2, 4, 5)), np.zeros((0,), dtype=int))


def test_error_stats_match_hand_computation() -> None:
    # Teacher targets: sample 0 exact at (1, 2); sample 1 at (1, 2) while
    # prediction is (3, 4): dr=2, dc=2, euclid=sqrt(8), manhattan=4.
    stats = placement_error_stats(
        np.array([1, 3]),
        np.array([2, 4]),
        np.array([1, 1]),
        np.array([2, 2]),
    )
    assert stats["n"] == 2.0
    assert stats["exact_acc"] == pytest.approx(0.5)
    assert stats["within_1_acc"] == pytest.approx(0.5)
    assert stats["within_2_acc"] == pytest.approx(1.0)
    assert stats["mean_euclidean"] == pytest.approx(float(np.sqrt(8)) / 2.0)
    assert stats["median_euclidean"] == pytest.approx(float(np.sqrt(8)) / 2.0)
    assert stats["mean_manhattan"] == pytest.approx(2.0)
    assert stats["median_manhattan"] == pytest.approx(2.0)


def test_adjacent_diagonal_counts_as_within_1_not_exact() -> None:
    stats = placement_error_stats(
        np.array([2]), np.array([3]), np.array([1]), np.array([2])
    )
    assert stats["exact_acc"] == pytest.approx(0.0)
    assert stats["within_1_acc"] == pytest.approx(1.0)
    assert stats["mean_euclidean"] == pytest.approx(float(np.sqrt(2)))
    assert stats["mean_manhattan"] == pytest.approx(2.0)


def test_error_stats_reject_mismatched_or_empty() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        placement_error_stats(np.array([1, 2]), np.array([1]), np.array([1, 2]), np.array([1, 2]))
    with pytest.raises(ValueError, match="must not be empty"):
        placement_error_stats(np.array([]), np.array([]), np.array([]), np.array([]))


def test_soft_region_mass_measures_teacher_approved_mass() -> None:
    maps = _norm_maps()
    soft = np.zeros((2, 4, 5), dtype=np.float64)
    soft[0, 0:3, 1:4] = 1.0 / 9.0  # 3x3 region around (1, 2)
    soft[1, 0:2, 0:2] = 1.0 / 4.0  # region far from prediction (3, 4)
    result = soft_region_mass(maps, np.array([0, 1]), soft)
    assert result["n"] == 2.0
    # Sample 0 holds all mass inside its region; sample 1 holds none.
    assert result["mean_mass"] == pytest.approx(0.5)
    assert result["majority_fraction"] == pytest.approx(0.5)


def test_soft_region_mass_rejects_empty_support() -> None:
    maps = _norm_maps()
    with pytest.raises(ValueError, match="at least one cell"):
        soft_region_mass(maps, np.array([0, 1]), np.zeros((2, 4, 5)))
    with pytest.raises(ValueError, match="shape"):
        soft_region_mass(maps, np.array([0, 1]), np.zeros((2, 4)))


def test_per_sample_stats_agree_with_aggregates() -> None:
    maps = _norm_maps()
    soft = np.zeros((2, 4, 5), dtype=np.float64)
    soft[0, 0:3, 1:4] = 1.0 / 9.0
    soft[1, 2:4, 3:5] = 1.0 / 4.0
    per = per_sample_placement_stats(
        maps, np.array([0, 1]), soft, np.array([1, 3]), np.array([2, 4])
    )
    assert per["pred_row"].tolist() == [1.0, 3.0]
    assert per["pred_col"].tolist() == [2.0, 4.0]
    assert per["euclidean"].tolist() == pytest.approx([0.0, 0.0])
    assert per["mass"].tolist() == pytest.approx([1.0, 1.0])
    assert set(per) == {
        "pred_row", "pred_col", "euclidean", "manhattan", "chebyshev", "mass",
    }
