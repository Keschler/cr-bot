import pytest

from scripts.run_v4_teacher_audit import (
    atomic_json,
    expand_sources,
    paired_action_metrics,
    parse_source_token,
    select_audit_records,
    select_high_rep_records,
)


def _record(i: int, *, split: str = "train", archetype: str | None = None) -> dict:
    archetype = archetype or ("beatdown" if i % 2 else "air-beatdown")
    match = f"{archetype}#{i % 8}"
    return {
        "audit_id": f"source-{split}:{split}:{match}:t{i}",
        "source_id": f"source-{split}",
        "source_root": "/tmp/source",
        "split": split,
        "match_id": match,
        "archetype": archetype,
        "t": i,
        "tick": i * 5,
        "state_hash": f"state-{split}-{i}",
        "match_path": "/tmp/match.json",
        "source_verdict": {
            "class": "frontier",
            "regret": 0.08 + (i % 20) * 0.02,
            "stability": 1.0,
            "best_source": "rule",
            "best_mean": 0.2,
            "sim_mean": 0.0,
            "filter": {"score": float(i % 11)},
            "tags": {},
            "branches": [
                {
                    "source": "sim",
                    "mode": 1,
                    "slot": 0,
                    "row": 1,
                    "col": 1,
                    "values": [0.0] * 16,
                },
                {
                    "source": "rule",
                    "mode": 1,
                    "slot": 0,
                    "row": 1,
                    "col": 2,
                    "values": [0.2] * 16,
                },
            ],
        },
    }


def test_source_selector_defaults_to_train_and_supports_explicit_sealed(tmp_path):
    root, split = parse_source_token(str(tmp_path))
    assert root == tmp_path.resolve()
    assert split == "train"
    root, split = parse_source_token(f"{tmp_path}:sealed")
    assert root == tmp_path.resolve()
    assert split == "sealed"
    assert expand_sources([f"{tmp_path}:all"]) == [
        expand_sources([f"{tmp_path}:train"])[0],
        expand_sources([f"{tmp_path}:sealed"])[0],
    ]


def test_selection_is_deterministic_stratified_and_capped():
    records = [
        _record(i, split="train" if i < 80 else "sealed")
        for i in range(160)
    ]
    first = select_audit_records(records, count=80, seed=123, per_match_cap=10)
    second = select_audit_records(records, count=80, seed=123, per_match_cap=10)
    assert [r["audit_id"] for r in first] == [r["audit_id"] for r in second]
    counts = {}
    for record in first:
        key = (record["source_id"], record["split"], record["match_id"])
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 10
    assert {r["split"] for r in first} == {"train", "sealed"}
    assert {r["regret_bin"] for r in first} == {"low", "mid", "high"}


def test_high_rep_selection_uses_only_pre_audit_fields():
    records = [_record(i, split="train") for i in range(100)]
    selected = select_high_rep_records(records, count=30, seed=456, per_match_cap=4)
    assert len(selected) == 30
    assert all("regret_bin" in r for r in selected)
    counts = {}
    for record in selected:
        key = (record["source_id"], record["split"], record["match_id"])
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 4


def test_paired_metrics_use_selected_replica_indices():
    verdict = {
        "branches": [
            {
                "source": "sim",
                "mode": 1,
                "slot": 0,
                "row": 1,
                "col": 1,
                "values": [0.0] * 16,
            },
            {
                "source": "rule",
                "mode": 1,
                "slot": 0,
                "row": 1,
                "col": 2,
                "values": [0.2] * 8 + [-0.1] * 8,
            },
        ]
    }
    frozen = {"source": "rule", "mode": 1, "slot": 0, "row": 1, "col": 2}
    held_out = paired_action_metrics(verdict, frozen, indices=range(8, 16))
    assert held_out["available"] is True
    assert held_out["n"] == 8
    assert held_out["mean_delta"] == pytest.approx(-0.1)
    assert held_out["classification"] == "negative"
    training = paired_action_metrics(verdict, frozen, indices=range(0, 8))
    assert training["mean_delta"] == pytest.approx(0.2)


def test_atomic_json_replaces_partial_file(tmp_path):
    path = tmp_path / "nested" / "result.json"
    atomic_json(path, {"status": "complete", "value": 3})
    assert path.read_text().endswith("\n")
    assert '"value":3' in path.read_text()
    atomic_json(path, {"status": "complete", "value": 4})
    assert '"value":4' in path.read_text()
    assert not list(path.parent.glob("*.tmp"))
