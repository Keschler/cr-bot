"""The 40-deck protocol must load five actors and execute exactly 2,000 cells."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest
import torch

from simulator.rl.v4 import catalog_evaluation as catalog
from simulator.rl.v4.frontier import frontier_model_config
from simulator.rl.v4.frontier_ppo import evaluation
from simulator.rl.v4.model_v4 import RecurrentV4Policy
from simulator.rl.v4.real_decks import catalog_sha256
from simulator.roster import PLAYER_DECK


@pytest.fixture
def manifest(tmp_path):
    torch.set_num_threads(1)
    specs = []
    with torch.random.fork_rng(devices=[]):
        for index, name in enumerate(catalog.POLICY_NAMES):
            torch.manual_seed(100 + index)
            actor = RecurrentV4Policy(frontier_model_config())
            path = tmp_path / f"actor{index}.pt"
            torch.save(actor.state_dict(), path)
            specs.append(dict(name=name, checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(schema=catalog.COMPARISON_SCHEMA, policies=specs, seeds=list(range(10)),
                model_config=asdict(frontier_model_config()), catalog_sha256=catalog_sha256(),
                max_decisions=1440, deterministic_policy=False)


def test_preflight_loads_distinct_frozen_actors_without_playing(manifest, monkeypatch):
    monkeypatch.setattr(evaluation, "full_match_record", lambda **kwargs: pytest.fail("preflight played a game"))
    before = torch.get_rng_state().clone()
    report = catalog.preflight_comparison(manifest)
    assert torch.equal(before, torch.get_rng_state())
    assert report["games_played"] == 0 and report["expected_total_games"] == 2000
    assert report["games_per_policy"] == {"train": 300, "eval": 100}
    assert report["device"] == "cpu" and report["learner_deck"] == list(PLAYER_DECK)
    assert len({actor["parameter_sha256"] for actor in report["actors"]}) == 5
    assert len(set(report["deck_ids"]["train"]) | set(report["deck_ids"]["eval"])) == 40
    assert not set(report["deck_ids"]["train"]) & set(report["deck_ids"]["eval"])


@pytest.mark.parametrize("field,value,error", [
    ("seeds", [1] * 10, "ten unique"),
    ("max_decisions", 1200, "1,440"),
    ("deterministic_policy", True, "stochastic"),
    ("catalog_sha256", "0" * 64, "catalog hash"),
])
def test_preflight_rejects_changed_protocol(manifest, field, value, error):
    manifest[field] = value
    with pytest.raises(ValueError, match=error):
        catalog.preflight_comparison(manifest)


def test_preflight_rejects_wrong_weights_and_duplicate_named_weights(manifest):
    manifest["policies"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint hash"):
        catalog.preflight_comparison(manifest)
    second = manifest["policies"][1]
    manifest["policies"][0].update(checkpoint=second["checkpoint"], sha256=second["sha256"])
    with pytest.raises(ValueError, match="more than once"):
        catalog.preflight_comparison(manifest)


def test_preflight_rejects_duplicate_weights_saved_at_different_paths(manifest):
    first, second = manifest["policies"][:2]
    Path(first["checkpoint"]).write_bytes(Path(second["checkpoint"]).read_bytes())
    first["sha256"] = second["sha256"]
    with pytest.raises(ValueError, match="identical actor weights"):
        catalog.preflight_comparison(manifest)


def _records(kwargs, *, result="win"):
    return [dict(
        archetype=spec["archetype"], deck_id=spec["deck_id"], seed=seed,
        result=result, winner=0 if result == "win" else None,
        terminated=result != "truncated", truncated=result == "truncated", decisions=1440,
        real_deck_split=spec["real_deck_split"], controller_seed=spec["controller_seed"],
        strategy=spec["strategy"], learner_deck=list(PLAYER_DECK), learner_player=0,
        deterministic_policy=False, max_decisions=1440, invalid_actions=[0, 0],
        opponent={"deck": list(spec["deck"]), "controller_seed": spec["controller_seed"]},
    ) for spec in kwargs["opponent_specs"] for seed in kwargs["seeds"]]


def test_frozen_comparison_executes_exact_grid_once_and_retains_both_splits(manifest, tmp_path, monkeypatch):
    calls = []

    def fake_matches(**kwargs):
        policy = kwargs["policy"]
        assert not policy.training and all(not p.requires_grad for p in policy.parameters())
        assert policy.config.fixed_wait and not kwargs["deterministic"]
        assert kwargs["max_decisions"] == 1440
        calls.append((catalog._parameter_digest(policy), len(kwargs["opponent_specs"])))
        return list(reversed(_records(kwargs)))

    monkeypatch.setattr(evaluation, "full_match_record", fake_matches)
    report = catalog.run_catalog_comparison(manifest, output_dir=tmp_path / "out")
    assert report["games_played"] == 2000 and report["technical_validity"]
    assert len(calls) == 10 and len({digest for digest, _ in calls}) == 5
    assert [count for _, count in calls] == [30, 10] * 5
    assert all(actor["outcomes"]["train"]["games"] == 300 for actor in report["actors"])
    assert all(actor["outcomes"]["eval"]["games"] == 100 for actor in report["actors"])
    assert len(report["pairwise"]["train"]) == len(report["pairwise"]["eval"]) == 10
    raw = list((tmp_path / "out").glob("*.jsonl"))
    assert len(raw) == 10 and sum(len(path.read_text().splitlines()) for path in raw) == 2000


def test_fully_censored_comparison_keeps_raw_records_and_suppresses_inference(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "full_match_record", lambda **kwargs: _records(kwargs, result="truncated"))
    report = catalog.run_catalog_comparison(manifest, output_dir=tmp_path / "out")
    assert report["games_played"] == 2000
    assert all(actor["outcomes"]["eval"]["truncated"] == 100 for actor in report["actors"])
    assert all(not pair["statistics"]["paired_inference_valid"] for split in report["pairwise"].values() for pair in split)


def test_comparison_rejects_controller_mismatch_even_when_fully_censored(manifest, tmp_path, monkeypatch):
    def changed(**kwargs):
        rows = _records(kwargs, result="truncated")
        rows[0]["controller_seed"] += 1
        return rows

    monkeypatch.setattr(evaluation, "full_match_record", changed)
    with pytest.raises(ValueError, match="declared comparison conditions"):
        catalog.run_catalog_comparison(manifest, output_dir=tmp_path / "out")
    assert (tmp_path / "out" / "actor_0_train.jsonl").is_file()
    assert not (tmp_path / "out" / "comparison.json").exists()
