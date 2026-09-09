"""Tests for the Stage-2 search dataset builder.

Covers end-to-end determinism, adopted-target legality, adoption semantics
(timing-holds never overturned, material gaps adopted, ties kept), sim
rebuild determinism, full search provenance, and the champion loader's
lazy-parameter handling.  Pipeline tests use a fixed-seed random policy
as the champion stand-in: determinism and legality do not need a good
actor.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.distillation import to_torch_batch
from simulator.rl.model_v4 import ModelConfigV4, RecurrentV4Policy
from simulator.rl.search_dataset import (
    ADOPT_MARGIN_ACTION,
    SearchDatasetConfig,
    _adopt_target,
    _new_sim_env,
    generate_search_dataset,
    load_champion_actor,
)
from simulator.rl.search_teacher import BranchAction, BranchScore, SearchResult
from simulator.rl.simulator_teacher import TeacherTarget
from simulator.ruleset import load_fixed_ruleset
from simulator.rl.opponent_pool import OpponentPool


def _random_champion() -> RecurrentV4Policy:
    torch.manual_seed(1234)
    return RecurrentV4Policy(
        ModelConfigV4(
            model_dim=16,
            spatial_channels=8,
            fused_dim=24,
            gru_hidden_dim=24,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=32,
            spatial_head_dim=8,
        )
    )


def _tiny_config(**overrides) -> SearchDatasetConfig:
    params = {
        "n_states": 48,
        "seed": 5,
        "timing_pool_states": 120,
        "sim_candidates": 48,
        "replay_sequences": 10,
        "min_adopted_share": 0.0,
        "min_spell_share": 0.0,
        "min_transition_share": 0.0,
        "min_sim_share": 0.0,
        "family_min_share": 0.0,
    }
    params.update(overrides)
    return SearchDatasetConfig(**params)


def _signature(samples) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample in samples:
        digest.update(np.ascontiguousarray(sample.raster).tobytes())
        target = sample.target
        digest.update(bytes((target.mode, target.card_slot, target.row, target.col)))
    return digest.hexdigest()


def test_search_dataset_is_deterministic() -> None:
    champion = _random_champion()
    first, stats_first = generate_search_dataset(_tiny_config(), champion)
    second, stats_second = generate_search_dataset(_tiny_config(), champion)
    assert len(first) == len(second) > 0
    assert _signature(first) == _signature(second)
    assert first[0].provenance == second[0].provenance
    assert stats_first == stats_second


def test_search_dataset_targets_legal_and_provenanced() -> None:
    champion = _random_champion()
    samples, stats = generate_search_dataset(_tiny_config(), champion)
    assert stats["n"] == len(samples)
    for sample in samples:
        target = sample.target
        assert target.mode in (0, 1)
        if target.mode == 1:
            assert bool(sample.legal_play[target.card_slot, target.row, target.col])
            assert float(sample.soft_placement.sum()) == pytest.approx(1.0)
        else:
            assert sample.legal_wait
        prov = sample.provenance
        for key in (
            "search_best_kind",
            "search_best_score",
            "search_branch_count",
            "search_branches",
            "regret_rule",
            "regret_actor",
            "adopted_search",
            "timing_hold",
        ):
            assert key in prov, key
        assert len(prov["search_branches"]) == prov["search_branch_count"] >= 2
        assert prov["state_hash"]
    batch = to_torch_batch(samples)
    assert batch["raster"].shape[0] == len(samples)


def _fake_result(best_cell: tuple[int, int], best_score: float, rule_score: float) -> SearchResult:
    def branch(kind, slot, row, col, score):
        action = BranchAction(kind=kind, slot=slot, row=row, col=col, card_key="cannon")
        terms = {"tower_dealt": 0.0, "tower_taken": 0.0, "threat_removed": 0.0,
                 "deployed_survived": 0.0, "elixir_spent": 0.0}
        return BranchScore(action=action, score=score, terms=terms, steps_run=8)

    branches = [
        branch("wait", 0, 0, 0, 0.0),
        branch("play", 1, 5, 5, rule_score),  # the rule's own cell
        branch("play", 1, best_cell[0], best_cell[1], best_score),
    ]
    best_index = 2
    return SearchResult(
        state_hash="abc", seed=0, source="unit", family="unit",
        horizon=8, branches=branches, best_index=best_index,
        second_gap=abs(best_score - rule_score),
    )


def _fake_record(rule_mode: int, base_mode: int) -> dict:
    return {
        "family": "unit",
        "target": TeacherTarget(mode=rule_mode, card_slot=1, row=5, col=5, wait_duration_idx=0),
        "legal_play": np.zeros((4, 32, 18), dtype=bool),
        "legal_wait": True,
        "own_elixir": 5.0,
        "hand_keys": ["hog-rider", "cannon", "musketeer", "skeletons"],
        "provenance": {"target_hand": ["hog-rider", "cannon", "musketeer", "skeletons"]},
    }


def test_adoption_never_overturns_timing_holds() -> None:
    record = _fake_record(rule_mode=0, base_mode=1)
    base = TeacherTarget(mode=1, card_slot=0, row=1, col=1, wait_duration_idx=0)
    result = _fake_result((7, 7), 5.0, 0.0)
    target, adopted, _, _ = _adopt_target(record, result, base)
    assert adopted is False
    assert target.mode == 0


def test_adoption_fires_on_material_gap_and_keeps_ties() -> None:
    record = _fake_record(rule_mode=1, base_mode=1)
    base = TeacherTarget(mode=1, card_slot=0, row=1, col=1, wait_duration_idx=0)
    record["legal_play"][1, 5, 5] = True
    record["legal_play"][1, 7, 7] = True
    big = _fake_result((7, 7), ADOPT_MARGIN_ACTION + 1.0, 0.0)
    adopted_target, adopted, regret, _ = _adopt_target(record, big, base)
    assert adopted is True
    assert regret == pytest.approx(ADOPT_MARGIN_ACTION + 1.0)
    assert (adopted_target.row, adopted_target.col) == (7, 7)
    small = _fake_result((7, 7), 0.01, 0.0)
    kept, adopted, _, _ = _adopt_target(record, small, base)
    assert adopted is False
    assert kept.mode == 1 and kept.card_slot == 1


def test_sim_rebuild_is_deterministic() -> None:
    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=9)
    first, _, _ = _new_sim_env(ruleset, pool, 9, 3, 9)
    second, _, _ = _new_sim_env(ruleset, pool, 9, 3, 9)
    assert first.state.state_hash() == second.state.state_hash()


def test_champion_loader_materializes_lazy_params(tmp_path) -> None:
    """Reproduce the production failure: a trained champion state dict holds
    the lazily-created placement cell bias, which a fresh policy lacks."""

    torch.manual_seed(0)
    tiny = ModelConfigV4(model_dim=8, spatial_channels=4, fused_dim=12, gru_hidden_dim=12)
    policy = RecurrentV4Policy(tiny)
    dummy = {
        "raster": torch.zeros(1, 1, 21, 32, 18),
        "global_features": torch.zeros(1, 1, 768),
        "entities": torch.zeros(1, 1, 128, 32),
        "entity_mask": torch.zeros(1, 1, 128, dtype=torch.bool),
        "hand_tokens": torch.zeros(1, 1, 4, 16),
        "opp_hand_probs": torch.full((1, 1, 128), 1.0 / 128),
        "opp_out_of_cycle": torch.zeros(1, 1, 128, dtype=torch.bool),
        "opp_elixir_interval": torch.zeros(1, 1, 2),
        "event_history": torch.zeros(1, 1, 16, 8),
        "reset_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    policy.eval()
    with torch.no_grad():
        policy(*dummy.values())
    path = tmp_path / "champ.pt"
    torch.save(policy.state_dict(), path)
    assert "heads.cell_bias" in torch.load(path, map_location="cpu", weights_only=True)
    loaded = load_champion_actor(str(path), tiny)
    for first, second in zip(policy.parameters(), loaded.parameters()):
        assert torch.equal(first, second)
