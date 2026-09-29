"""Focused source-state integrity checks for search distillation replay."""

import json
from pathlib import Path

import numpy as np
import pytest

from simulator.rl.v4.search_distillation import preparation


class _State:
    def __init__(self, digest: str) -> None:
        self.digest = digest

    def state_hash(self) -> str:
        return self.digest


class _Env:
    def __init__(self, digest: str) -> None:
        self.state = _State(digest)

    def step_v2(self, _actions) -> None:
        return None


def _inputs() -> dict[str, np.ndarray]:
    return {
        name: np.zeros((1,), dtype=np.float32)
        for name in (
            "raster", "global_features", "entities", "entity_mask",
            "hand_tokens", "opp_hand_probs", "opp_out_of_cycle",
            "opp_elixir_interval", "event_history",
        )
    }


def test_selected_prefix_replay_rejects_recorded_state_hash_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(preparation, "_fresh_env", lambda _match: _Env("actual"))
    monkeypatch.setattr(
        preparation,
        "_observe_inputs",
        lambda *_args: (_inputs(), np.ones((1, 1, 1), dtype=bool), [], 0.0),
    )
    match = {
        "decisions": [{
            "state_hash": "stale",
            "learner": {"mode": 0},
            "opponent": {"kind": "wait"},
        }],
    }

    with pytest.raises(ValueError, match="prefix replay diverged"):
        preparation.replay_selected_prefixes(match, [0], context=0)


def test_prepare_records_replays_rejected_entries_for_hash_validation(tmp_path: Path, monkeypatch) -> None:
    match = {
        "config": {"archetype": "hog", "match_index": 0},
        "decisions": [{
            "t": 0,
            "state_hash": "stale",
            "learner": {"mode": 0},
            "opponent": {"kind": "wait"},
        }],
    }
    (tmp_path / "train_hog_0_match.json").write_text(json.dumps(match))
    (tmp_path / "train_hog_0_t0_verdict.json").write_text("{}")
    calls: list[tuple[int, ...]] = []

    def replay(_match, offsets, *, context, **_kwargs):
        calls.append(tuple(offsets))
        raise ValueError("stale source state")

    monkeypatch.setattr(
        preparation, "_replay_selected_prefixes_with_legal", replay
    )
    entries = [{"entry_id": "train:hog#0:t0"}]

    with pytest.raises(ValueError, match="stale source state"):
        preparation.prepare_records(tmp_path, entries, retention_count=0)
    assert calls == [(0,)]
