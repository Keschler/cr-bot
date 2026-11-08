"""Declared extra presentations and the distinction between fit and strength."""
from collections import Counter
from copy import deepcopy

import pytest

from simulator.rl.v4.correction_exposure import (
    CorrectionGateConfig, correction_learning_gate, exposure_schedule,
)


def test_schedule_preserves_update_count_and_exact_per_target_presentations():
    factual = {"factual-b": [9], "factual-a": [3, 5]}
    branches = {"branch": [10, 11]}
    schedule = exposure_schedule(factual, branches, exposures=3, optimizer_steps=4, seed=42)
    counts = Counter(target for update in schedule for target in update)
    assert len(schedule) == 4
    assert counts == {("factual-a", 3): 3, ("factual-a", 5): 3,
                      ("factual-b", 9): 3, ("branch", 10): 4, ("branch", 11): 4}
    assert schedule == exposure_schedule(dict(reversed(list(factual.items()))), branches,
                                         exposures=3, optimizer_steps=4, seed=42)
    assert exposure_schedule(factual, {}, exposures=0, optimizer_steps=4, seed=42) == [[], [], [], []]


def diagnostic():
    return dict(inactive_rows=3, retention_kl=.001, retention_kl_max=.002, factors=[dict(
        target=[1, 0, 20, 6, 0], before=dict(nll=5.),
        after=dict(nll=3., target_mode_probability=.9,
                   card_probability_given_play=.9, nearby_cell_probability_given_card=.8))])


def test_gate_distinguishes_train_memorization_holdout_uptake_and_retention():
    train, held_out = diagnostic(), diagnostic()
    passed = correction_learning_gate(train, held_out)
    assert passed["eligible_for_fresh_games"]
    assert passed["selected_actor_changed"] is False
    assert passed["used_for_epoch_selection"] is False
    held_out["factors"] = []
    unmeasured = correction_learning_gate(train, held_out)
    assert unmeasured["train_memorization"]["passed"]
    assert unmeasured["held_out_label_uptake"]["status"] == "unmeasured"
    assert not unmeasured["eligible_for_fresh_games"]


@pytest.mark.parametrize("mutation", [
    lambda row: row["factors"][0]["after"].update(nll=6.),
    lambda row: row["factors"][0]["after"].update(nll=float("nan")),
    lambda row: row["factors"][0]["after"].update(target_mode_probability=.01),
    lambda row: row["factors"][0]["after"].update(card_probability_given_play=.01),
    lambda row: row["factors"][0]["after"].update(nearby_cell_probability_given_card=.01),
    lambda row: row.update(retention_kl=.1),
    lambda row: row.update(retention_kl_max=1.),
    lambda row: row.update(inactive_rows=0),
])
def test_gate_fails_closed_when_label_factors_or_retention_fail(mutation):
    train, held_out = diagnostic(), deepcopy(diagnostic())
    mutation(held_out)
    assert not correction_learning_gate(train, held_out)["eligible_for_fresh_games"]


@pytest.mark.parametrize("kwargs", [dict(min_nll_drop=-1), dict(max_mean_retention_kl=float("nan")),
                                    dict(min_card_probability=1.1)])
def test_gate_rejects_invalid_declared_thresholds(kwargs):
    with pytest.raises(ValueError):
        CorrectionGateConfig(**kwargs)
