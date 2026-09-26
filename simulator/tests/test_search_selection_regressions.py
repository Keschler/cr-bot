"""Joint dataset quotas and consistent fallback branch continuations."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from simulator.rl.v4.search_dataset import targets
from simulator.rl.v4.search_dataset.config import SearchDatasetConfig
from simulator.rl.v4.search_dataset.selection import _select_candidates
from simulator.rl.v4.simulator_teacher import TeacherError


def candidate(index, mask):
    # Put arrays first so equality-based list removal reproduces the original bug.
    return {
        "array": np.array([index, index + 1]),
        "record": {"family": "test", "provenance": {"state_hash": str(index)}},
        **dict(zip(("adopted", "spell", "transition", "sim"), mask)),
        "search": {"actor_score": None}, "regret_rule": 1.0,
    }


def config(n, **kwargs):
    return SearchDatasetConfig(n_states=n, min_adopted_share=0, min_spell_share=0,
                               min_transition_share=0, min_sim_share=0, **kwargs)


def test_selection_repairs_joint_quotas_without_array_equality(monkeypatch):
    monkeypatch.setattr("simulator.rl.v4.search_dataset.selection._hash_order",
                        lambda seed, value: int(value))
    records = [candidate(0, (False, True, False, False)),
               candidate(1, (False, False, False, False)),
               candidate(2, (False, False, True, False))]
    cfg = SearchDatasetConfig(n_states=2, min_adopted_share=0, min_sim_share=0,
                              min_spell_share=.5, min_transition_share=.5)
    selected = _select_candidates(records, cfg)
    assert [r["record"]["provenance"]["state_hash"] for r in selected] == ["0", "2"]


def test_adoption_quota_survives_other_quota_repairs():
    records = [candidate(0, (True, False, False, False)),
               candidate(1, (False, False, False, False)),
               candidate(2, (False, True, True, True))]
    cfg = SearchDatasetConfig(n_states=2, min_adopted_share=.5, min_spell_share=.5,
                              min_transition_share=.5, min_sim_share=.5)
    selected = _select_candidates(records, cfg)
    assert {id(r) for r in selected} == {id(records[0]), id(records[2])}


def test_joint_selection_matches_exhaustive_feasibility():
    rng = np.random.default_rng(719)
    fields = ("adopted", "spell", "transition", "sim")
    for _ in range(100):
        records = [candidate(i, rng.integers(0, 2, 4)) for i in range(7)]
        needed = rng.integers(0, 4, 4)
        cfg = SearchDatasetConfig(n_states=3, **{
            f"min_{field}_share": int(n) / 3 for field, n in zip(fields, needed)})
        feasible = any(all(sum(bool(r[field]) for r in subset) >= n
                           for field, n in zip(fields, needed))
                       for subset in combinations(records, 3))
        if not feasible:
            with pytest.raises(TeacherError, match="unsatisfiable"):
                _select_candidates(records, cfg)
            continue
        selected = _select_candidates(records, cfg)
        assert len(selected) == len({id(r) for r in selected}) == 3
        assert all(sum(bool(r[field]) for r in selected) >= n
                   for field, n in zip(fields, needed))
        assert [id(r) for r in selected] == [id(r) for r in _select_candidates(records, cfg)]


def test_actor_fallback_uses_same_policy_controller(monkeypatch):
    controller = object()
    calls = []
    result = SimpleNamespace(state_hash="state", branches=[])
    def search(*args, **kwargs):
        calls.append(kwargs)
        return result
    def score(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(score=1.25)
    monkeypatch.setattr(targets, "search_state", search)
    monkeypatch.setattr(targets, "score_branch", score)
    record = {"legal_play": np.ones((1, 2, 2), dtype=bool), "legal_wait": True,
              "target": SimpleNamespace(mode=0), "hand_keys": ["knight"],
              "family": "test", "provenance": {"state_hash": "state"}}
    scored = targets._search_live(object(), record,
                                  {"mode": 1, "card": 0, "row": 1, "col": 1},
                                  horizon=28, max_branches=1, seed=12,
                                  continuation_mode="active-policy",
                                  learner_controller=controller)
    assert scored["actor_score"] == 1.25
    assert len(calls) == 2
    assert all(call["learner_controller"] is controller for call in calls)
    assert all(call["continuation_mode"] == "active-policy" for call in calls)
