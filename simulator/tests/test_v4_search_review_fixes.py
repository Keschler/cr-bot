from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rl.v4.frontier_eval import (
    fixed_action_horizon_gate,
    horizon_stability,
    merge_independent_confirmation,
)


def _branch(source, *, slot=0, row=0, col=0, mean=0.0, values=(0.0,)):
    return {
        "source": source,
        "mode": 1,
        "slot": slot,
        "row": row,
        "col": col,
        "mean": mean,
        "values": list(values),
        "replica_indices": list(range(len(values))),
        "replicas": [
            {"replica_index": i, "replica_seed": 100 + i, "value": value}
            for i, value in enumerate(values)
        ],
    }


def _verdict(state, *, best, branches):
    return {
        "state_hash": state,
        "best_source": best["source"],
        "best_action": dict(best),
        "regret": float(best.get("mean", 0.0)),
        "branches": branches,
    }


def test_horizon_stability_compares_full_action_identity_for_duplicate_sources():
    short = _verdict(
        "state",
        best={"source": "policy-top", "mode": 1, "slot": 0, "row": 1, "col": 2, "mean": 2.0},
        branches=[_branch("policy-top", row=1, col=2, mean=2.0)],
    )
    reference = _verdict(
        "state",
        best={"source": "policy-top", "mode": 1, "slot": 0, "row": 1, "col": 3, "mean": 2.0},
        branches=[_branch("policy-top", row=1, col=3, mean=2.0)],
    )
    summary = horizon_stability({64: [short], 256: [reference]})
    assert summary["horizons"][64]["top1_agree"] == 0.0


def test_horizon_gate_counts_missing_fixed_action_comparisons():
    action = {"source": "rule", "mode": 1, "slot": 0, "row": 1, "col": 1}
    reference = _verdict(
        "state",
        best=dict(action, mean=1.0),
        branches=[
            _branch("sim", mean=0.0, values=(0.0, 0.0)),
            _branch("rule", row=1, col=1, mean=1.0, values=(1.0, 1.0)),
        ],
    )
    short = _verdict(
        "state",
        best=dict(action, mean=1.0),
        branches=[
            _branch("sim", mean=0.0, values=(0.0, 0.0)),
            _branch("rule", row=1, col=1, mean=1.0, values=(1.0, 1.0)),
        ],
    )
    gate = fixed_action_horizon_gate(
        {56: [short], 112: [reference]}, reference=112, required_horizons=(28, 56)
    )
    assert gate["pass"] is False
    assert gate["comparisons"]["28"]["unavailable"] == 1
    assert gate["comparisons"]["56"]["pass"] is True


def test_merge_confirmation_keeps_discovery_values_and_replica_provenance():
    action = {"source": "rule", "mode": 1, "slot": 0, "row": 1, "col": 1}
    discovery = _verdict(
        "state",
        best=dict(action, mean=1.0),
        branches=[_branch("sim", mean=0.0), _branch("rule", row=1, col=1, mean=1.0, values=(1.0, 1.0))],
    )
    discovery["replica_indices"] = [0, 1]
    confirmation = _verdict(
        "state",
        best=dict(action, mean=1.5),
        branches=[_branch("sim", mean=0.0, values=(0.0,)), _branch("rule", row=1, col=1, mean=1.5, values=(1.5,))],
    )
    confirmation["replica_indices"] = [2]
    for branch in confirmation["branches"]:
        branch["replicas"] = [{"replica_index": 2, "replica_seed": 102, "value": branch["values"][0]}]
        branch["replica_indices"] = [2]
    merged = merge_independent_confirmation(discovery, confirmation, frozen_action=action)
    rule = next(branch for branch in merged["branches"] if branch["source"] == "rule")
    assert rule["values"] == [1.0, 1.0]
    assert rule["all_values"] == [1.0, 1.0, 1.5]
    assert {r["replica_index"] for r in rule["replicas"]} == {0, 1, 2}
    assert merged["frozen_confirmation"]["replica_indices"] == [2]


def test_source_best_action_selects_highest_mean_duplicate_source():
    from scripts.run_v4_teacher_audit import _source_best_action

    source = {
        "best_source": "policy-top",
        "branches": [
            _branch("policy-top", row=1, col=1, mean=0.5),
            _branch("policy-top", row=1, col=2, mean=0.9),
        ],
    }
    assert _source_best_action(source)["col"] == 2
