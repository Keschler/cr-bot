from __future__ import annotations

import numpy as np
import pytest

from simulator.evaluation import (
    ChampionEntry,
    ChampionRegistry,
    PromotionThresholds,
    evaluate_promotion,
    holm_reject,
    macro_match_score,
    macro_tactical_score,
    paired_differences,
    required_sample_size,
    retire_manifest,
    sample_sealed_manifest,
    seal_current_contracts,
    side_swap_blocks,
    stratified_paired_bootstrap_lcb,
)


def _pools() -> dict[str, dict[str, list[str]]]:
    return {
        "deck_ids": {"scripted": ["d1", "d2"], "meta": ["d3"]},
        "strategy_ids": {"scripted": ["s1"], "meta": ["s2", "s3"]},
    }


def test_sealed_contracts_pin_live_revision() -> None:
    first = seal_current_contracts(code_revision="rev-a")
    second = seal_current_contracts(code_revision="rev-a")
    assert first.contract_hash == second.contract_hash
    assert first.engine_version == "reference-0.37.0"
    assert first.ruleset_id == "v1"
    assert first.observation_schema_version == "public-v3-1"
    other = seal_current_contracts(code_revision="rev-b")
    assert other.contract_hash != first.contract_hash


def test_manifest_samples_side_swaps_after_freeze() -> None:
    pools = _pools()
    manifest = sample_sealed_manifest(
        challenger_hash="challenger-1",
        contract_hash="contract-1",
        strata_counts={"scripted": 3, "meta": 2},
        deck_ids=pools["deck_ids"],
        strategy_ids=pools["strategy_ids"],
        manifest_seed=11,
    )
    assert len(manifest.cells) == 10
    pairs: dict[str, list[int]] = {}
    for cell in manifest.cells:
        pairs.setdefault(cell.pair_id, []).append(cell.side)
    assert pairs and all(sorted(sides) == [0, 1] for sides in pairs.values())

    resealed = sample_sealed_manifest(
        challenger_hash="challenger-2",
        contract_hash="contract-1",
        strata_counts={"scripted": 3, "meta": 2},
        deck_ids=pools["deck_ids"],
        strategy_ids=pools["strategy_ids"],
        manifest_seed=11,
    )
    assert resealed.manifest_hash != manifest.manifest_hash


def test_retired_manifest_cannot_promote_again() -> None:
    pools = _pools()
    manifest = sample_sealed_manifest(
        challenger_hash="challenger-9",
        contract_hash="contract-1",
        strata_counts={"scripted": 1},
        deck_ids=pools["deck_ids"],
        strategy_ids=pools["strategy_ids"],
        manifest_seed=99,
    )
    retire_manifest(manifest.manifest_hash)
    with pytest.raises(ValueError, match="retired manifest"):
        sample_sealed_manifest(
            challenger_hash="challenger-9",
            contract_hash="contract-1",
            strata_counts={"scripted": 1},
            deck_ids=pools["deck_ids"],
            strategy_ids=pools["strategy_ids"],
            manifest_seed=99,
        )


def test_paired_bootstrap_lcb_is_deterministic() -> None:
    blocks = [[0.1, 0.0], [0.2], [-0.05, 0.05], [0.3]]
    first = stratified_paired_bootstrap_lcb(blocks, n_bootstrap=2000, seed=3)
    second = stratified_paired_bootstrap_lcb(blocks, n_bootstrap=2000, seed=3)
    assert first == second
    assert first["lcb"] <= first["mean"]
    assert first["confidence"] == 0.95


def test_paired_helpers_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="must match"):
        paired_differences([1.0, 0.0], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        macro_match_score({})
    with pytest.raises(ValueError, match="must match"):
        side_swap_blocks(["a"], [1.0, 2.0])
    assert required_sample_size(0.5, 0.1) > 0
    assert holm_reject([0.001, 0.4], alpha=0.05) == [True, False]


def test_macro_scores_weight_families_equally() -> None:
    assert macro_match_score({"a": [1.0, 1.0], "b": [0.0]}) == pytest.approx(0.5)
    assert macro_tactical_score({"offense": [2.0], "ground-defense": [0.0]}) == pytest.approx(1.0)


def test_promotion_requires_every_gate() -> None:
    passing = evaluate_promotion(
        champion_hash="B",
        challenger_hash="C",
        contract_hash="K",
        manifest_hash="M",
        thresholds=PromotionThresholds(),
        validity_clean=True,
        match_lcb=0.05,
        tact_lcb=0.01,
        stratum_lcbs={"scripted": 0.0, "meta": 0.01},
        cat_ucb_diff=0.0,
        head_to_head_diff=0.1,
        integrity_ok=True,
    )
    assert passing.promoted

    failing = evaluate_promotion(
        champion_hash="B",
        challenger_hash="C",
        contract_hash="K",
        manifest_hash="M",
        thresholds=PromotionThresholds(),
        validity_clean=True,
        match_lcb=0.005,
        tact_lcb=0.01,
        stratum_lcbs={"scripted": 0.0},
        cat_ucb_diff=0.0,
        head_to_head_diff=0.1,
        integrity_ok=True,
    )
    assert not failing.promoted
    assert any(gate.name == "G1-primary-match" and not gate.passed for gate in failing.gates)


def test_champion_registry_is_monotonic() -> None:
    registry = ChampionRegistry()
    with pytest.raises(ValueError, match="empty"):
        registry.current()
    entry = ChampionEntry(
        name="v3-anchor",
        checkpoint_path="outputs/v3.pt",
        checkpoint_sha256="sha256:abc",
        code_revision="rev-a",
        contract_hash="contract-1",
        manifest_hash="manifest-1",
        summary="16/24 paired",
    )
    before = registry.chain_hash()
    registry.promote(entry)
    assert registry.current().name == "v3-anchor"
    assert registry.chain_hash() != before
    restored = ChampionRegistry.from_json(registry.to_json())
    assert restored.current().as_dict() == entry.as_dict()
    assert np.isfinite(float(len(restored)))
