"""Focused provenance checks for the V4 data runners."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


frontier = _load_script(
    "v4_frontier_generation_integrity",
    REPO / "scripts/run_v4_frontier_generation.py",
)
distillation = _load_script(
    "v4_search_distillation_integrity",
    REPO / "scripts/run_v4_search_distillation.py",
)


def test_source_manifests_exclude_generated_simulator_outputs() -> None:
    frontier_paths = frontier._source_hashes()
    distillation_paths = distillation._implementation_identity()

    assert frontier_paths
    assert distillation_paths
    assert all("/simulator/outputs/" not in path for path in frontier_paths)
    assert all("simulator/outputs/" not in path for path in distillation_paths)
    assert "simulator/rl/v4/search_distillation/preparation.py" in distillation_paths


def test_frontier_runner_accepts_train_only_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_v4_frontier_generation.py", "--checkpoint", "actor.pt",
        "--out", "frontier", "--sealed-matches-per-archetype", "0",
    ])
    assert frontier.parser().parse_args().sealed_matches_per_archetype == 0
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", [
            "run_v4_frontier_generation.py", "--checkpoint", "actor.pt",
            "--out", "frontier", "--sealed-matches-per-archetype", "-1",
        ])
        frontier.parser().parse_args()


def test_reuse_manifest_requires_matching_sources_and_collection_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"actor")
    checkpoint_hash = frontier.hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_hashes = {"simulator/example.py": "a" * 64}
    config = {
        "train_profile": "pinned",
        "train_pool_seed": 1,
        "sealed_pool_seed": 2,
        "max_decisions": 1200,
        "real_deck_catalog_sha256": None,
    }
    manifest = {
        "checkpoint_sha256": checkpoint_hash,
        "source_hashes": source_hashes,
        "config": config,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    assert frontier._validate_reuse_source_manifest(
        path,
        checkpoint_sha256=checkpoint_hash,
        source_hashes=source_hashes,
        config=config,
    ) == manifest

    changed = dict(manifest)
    changed["source_hashes"] = {"simulator/example.py": "b" * 64}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="simulator/source files"):
        frontier._validate_reuse_source_manifest(
            path,
            checkpoint_sha256=checkpoint_hash,
            source_hashes=source_hashes,
            config=config,
        )

    changed = dict(manifest)
    changed["config"] = {**config, "max_decisions": 64}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="collection config"):
        frontier._validate_reuse_source_manifest(
            path,
            checkpoint_sha256=checkpoint_hash,
            source_hashes=source_hashes,
            config=config,
        )
