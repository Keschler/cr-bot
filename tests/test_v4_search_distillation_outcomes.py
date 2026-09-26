"""Fresh search regret must compare actions under one evaluation protocol."""
import json

import pytest
import torch

from simulator.rl.v4 import frontier_eval
from simulator.rl.v4.search_distillation.evaluation import _paired_outcome_evaluation
from simulator.rl.v4.search_distillation.records import ActionTarget, CachedPrefix


def test_search_regret_replays_reference_with_same_seeds_and_horizon(tmp_path, monkeypatch):
    path = tmp_path / "match.json"
    path.write_text(json.dumps({"config": {}, "decisions": [{"state_hash": "s"}]}))
    record = CachedPrefix(
        key="train:hog#0:t0", match_id="hog#0", offset=0,
        observations=[], legal_play=[], legal_wait=[], recorded_action=ActionTarget(0),
        correction_action=ActionTarget(1, 1), is_correction=True,
        match_path=str(path), outcome_horizon=56, outcome_replicas=3,
        branches=[{"source": "sim", "mode": 0, "values": [0.0]},
                  {"mode": 1, "slot": 1, "values": [999.0]}],
    )
    calls = []
    anchor = object()
    monkeypatch.setattr(frontier_eval, "replay_to_state", lambda *args: (object(), None))

    def branch(**kwargs):
        calls.append(kwargs)
        # The discovered best is worse on independent replicas. Historical
        # value 999 must never enter this fresh regret calculation.
        value = {"recorded": 1.0, "selected": 3.0, "search_best": 2.0}[kwargs["branch"].source]
        return {"value": value}

    monkeypatch.setattr(frontier_eval, "run_branch", branch)
    outcome = _paired_outcome_evaluation(record, ActionTarget(1, 2), anchor, device=torch.device("cpu"))
    assert outcome["evaluated"]
    assert outcome["mean"] == pytest.approx(2.0)
    assert outcome["search_regret"] == pytest.approx(-1.0)
    assert outcome["search_regret_values"] == [-1.0] * 3
    assert len(calls) == 9
    for start in range(0, 9, 3):
        paired = calls[start:start + 3]
        assert len({call["replica_seed"] for call in paired}) == 1
        assert all(call["horizon"] == 56 and call["policy"] is anchor for call in paired)


def test_singleton_paired_outcome_reports_unresolved_se(tmp_path, monkeypatch):
    path = tmp_path / "match.json"
    path.write_text(json.dumps({"config": {}, "decisions": [{"state_hash": "s"}]}))
    record = CachedPrefix(
        key="train:hog#0:t0", match_id="hog#0", offset=0,
        observations=[], legal_play=[], legal_wait=[], recorded_action=ActionTarget(0),
        match_path=str(path), outcome_horizon=56, outcome_replicas=1,
    )
    anchor = object()
    monkeypatch.setattr(frontier_eval, "replay_to_state", lambda *args: (object(), None))
    monkeypatch.setattr(
        frontier_eval,
        "run_branch",
        lambda **kwargs: {"value": 3.0 if kwargs["branch"].source == "selected" else 1.0},
    )

    outcome = _paired_outcome_evaluation(
        record, ActionTarget(1, 2), anchor, device=torch.device("cpu"),
    )

    assert outcome["evaluated"]
    assert outcome["n"] == 1
    assert outcome["mean"] == pytest.approx(2.0)
    assert outcome["se"] == float("inf")
