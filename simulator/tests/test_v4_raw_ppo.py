"""Raw PPO starts must be search-free, legal, and usable by the shared collector."""

import pytest
import torch
from pathlib import Path
import subprocess
import sys

from simulator.rl.v4.learning_curve import LearningCurveConfig, _load_runtime_entries
from simulator.rl.v4.raw_ppo import RAW_PPO_STARTS_VERSION, make_match_start_entries


def test_match_start_entries_cover_pool_without_search_labels():
    entries = make_match_start_entries(pool_seed=7, matches_per_archetype=1)
    assert len(entries) == 6
    assert len({entry["provenance"]["state_hash"] for entry in entries}) == 6
    assert all(entry["provenance"]["generator"] == RAW_PPO_STARTS_VERSION for entry in entries)
    assert all(entry["initial_state"]["tick"] == 0 for entry in entries)
    assert all("verdict" not in entry and "hidden0" not in entry for entry in entries)


def test_mixed_starts_add_deck_and_behavior_diversity():
    from simulator.rl.v4.frontier_ppo.environment import heldout_opponent_decks

    entries = make_match_start_entries(pool_seed=7, matches_per_archetype=2, profile="mixed")
    assert len(entries) == 14
    assert {entry["opponent"]["strategy"] for entry in entries} >= {
        "random-legal", "beatdown", "siege-bait"
    }
    assert all(entry["provenance"]["profile"] == "mixed" for entry in entries)
    heldout = heldout_opponent_decks()
    assert len(heldout) == 6
    assert any(spec["strategy"] == "random-legal" for spec in heldout)
    assert not (
        {frozenset(entry["opponent_deck"]) for entry in entries}
        & {frozenset(spec["deck"]) for spec in heldout}
    )


def test_real_deck_catalog_is_supported_and_disjoint():
    from simulator.rl.v4.real_decks import (
        REAL_DECK_ARCHETYPES, _source_cards, load_real_decks,
    )
    from simulator.rl.v4.frontier_ppo.environment import real_eval_opponent_decks

    catalog = load_real_decks()
    assert len(catalog) == 40
    assert {row.archetype for row in catalog} == set(REAL_DECK_ARCHETYPES)
    assert all(
        sum(row.split == split and row.archetype == archetype for row in catalog)
        == (6 if split == "train" else 2)
        for split in ("train", "eval")
        for archetype in REAL_DECK_ARCHETYPES
    )
    assert all(set(row.cards) == set(_source_cards(row.source_url)) for row in catalog)
    training = {frozenset(row.cards) for row in catalog if row.split == "train"}
    heldout = {frozenset(row.cards) for row in catalog if row.split == "eval"}
    assert len(training) == 30 and len(heldout) == 10 and not training & heldout
    assert {frozenset(spec["deck"]) for spec in real_eval_opponent_decks()} == heldout
    with pytest.raises(ValueError, match="unsupported card forms"):
        _source_cards("https://royaleapi.com/decks/stats/cannon%2Cfireball%2Chog-rider%2Cice-golem%2Cice-spirit%2Cmusketeer-ev1%2Cskeletons%2Czap")


def test_full_match_pairs_by_deck_identity_within_archetype():
    from simulator.rl.v4.learning_curve import paired_match_statistics

    baseline = [
        {"archetype": "beatdown", "deck_id": "deck-a", "seed": 1, "result": "win"},
        {"archetype": "beatdown", "deck_id": "deck-b", "seed": 1, "result": "loss"},
    ]
    candidate = [
        {"archetype": "beatdown", "deck_id": "deck-b", "seed": 1, "result": "loss"},
        {"archetype": "beatdown", "deck_id": "deck-a", "seed": 1, "result": "win"},
    ]
    result = paired_match_statistics(baseline, candidate, max_decisions=1200)
    assert result["n_paired"] == 2
    assert result["paired_mean_score_delta"] == 0


def test_real_ppo_openings_match_frontier_and_reject_eval_deck(tmp_path):
    from simulator.rl.v4.frontier import PoolMatchConfig, make_pool_match_start
    from simulator.rl.v4.real_decks import catalog_sha256

    entries = make_match_start_entries(pool_seed=23, matches_per_archetype=2, profile="real")
    assert len(entries) == 10
    for entry in entries:
        archetype = entry["opponent"]["archetype"]
        index = entry["provenance"]["match_index"]
        env, scenario, match_seed, _ = make_pool_match_start(PoolMatchConfig(
            archetype=archetype, match_index=index, pool_seed=23,
            deck_profile="real", real_deck_split="train",
        ))
        assert entry["opponent_deck"] == list(scenario.deck.cards)
        assert entry["provenance"]["match_seed"] == match_seed
        assert entry["provenance"]["state_hash"] == env.state.state_hash()
        assert entry["provenance"]["real_deck_catalog_sha256"] == catalog_sha256()

    train = tmp_path / "train.pt"
    sealed = tmp_path / "sealed.pt"
    torch.save(entries, train)
    torch.save([], sealed)
    config = LearningCurveConfig(
        training_source="match-start", bc_coef=0, train_entries=train,
        sealed_entries=sealed, output_dir=tmp_path / "curve",
    )
    loaded, _, _, _ = _load_runtime_entries(config)
    assert len(loaded) == 10
    from simulator.rl.v4.real_decks import real_decks_for
    entries[0]["opponent_deck"] = list(real_decks_for("eval", "aggressive-pressure")[0].cards)
    torch.save(entries, train)
    with pytest.raises(ValueError, match="train catalog"):
        _load_runtime_entries(config)


def test_real_frontier_match_records_catalog_deck():
    from simulator.rl.v4.frontier import (
        PoolMatchConfig, collect_pool_match, frontier_model_config,
    )
    from simulator.rl.v4.model_v4 import RecurrentV4Policy
    from simulator.rl.v4.real_decks import real_decks_for

    result = collect_pool_match(
        policy=RecurrentV4Policy(frontier_model_config()),
        cfg=PoolMatchConfig(
            archetype="air-beatdown", match_index=0, pool_seed=23,
            max_decisions=2, deck_profile="real", real_deck_split="eval",
        ),
        device="cpu",
    )
    assert result["config"]["opponent_deck_id"] == real_decks_for(
        "eval", "air-beatdown"
    )[0].deck_id
    assert result["config"]["real_deck_split"] == "eval"
    assert len(result["decisions"]) == 2


def test_policy_opponent_uses_rotated_legal_actions(tmp_path):
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.model_v4 import RecurrentV4Policy
    from simulator.rl.v4.frontier_eval import load_frontier_state, make_opponent
    from simulator.rl.v4.frontier_ppo.environment import make_frontier_env
    from simulator.actions import PlayCardAction

    checkpoint = tmp_path / "actor.pt"
    torch.save(RecurrentV4Policy(frontier_model_config()).state_dict(), checkpoint)
    entries = make_match_start_entries(
        pool_seed=7, matches_per_archetype=1, policy_opponents=[checkpoint]
    )
    entry = entries[-1]
    env = make_frontier_env()
    load_frontier_state(env, entry["initial_state"])
    opponent = make_opponent(entry["opponent"], 0, env=env)
    action = opponent.choose_action(env.engine, env.state, 1)
    assert env.engine.validate_action(env.state, action) is None
    if isinstance(action, PlayCardAction):
        viewer = env.observe_v2_for_viewer(1)
        col, row = action.cell
        assert viewer.legal_play[action.card_slot, 31 - row, 17 - col]

    from simulator.rl.v4.frontier_ppo import collect_frontier_rollout
    from simulator.rl.v4.model_v4 import V4ValueHead

    policy = RecurrentV4Policy(frontier_model_config())
    critic = V4ValueHead(frontier_model_config().gru_hidden_dim)
    batch, info = collect_frontier_rollout(
        policy, critic, None, [entry], n_decisions=2, n_lanes=1,
        episode_cap=2, device="cpu", seed=1, step_tower_reward=True,
    )
    assert tuple(batch.trajectory.rewards.shape) == (1, 2)
    assert info["illegal_attempts"] == 0

    from simulator.rl.v4.frontier_ppo.evaluation import full_match_record

    spec = dict(entry["opponent"], deck=entry["opponent_deck"])
    matches = full_match_record(
        policy=policy, opponent_specs=[spec], seeds=[41], max_decisions=2,
        device="cpu",
    )
    assert len(matches) == 1
    assert matches[0]["result"] == "truncated"


def test_raw_ppo_source_rejects_bc_and_search_entries(tmp_path):
    with pytest.raises(ValueError, match="bc_coef=0"):
        LearningCurveConfig(training_source="match-start")
    train = tmp_path / "train.pt"
    sealed = tmp_path / "sealed.pt"
    torch.save([{"entry_id": "bad", "provenance": {"generator": "frontier"}}], train)
    torch.save([], sealed)
    config = LearningCurveConfig(
        training_source="match-start", bc_coef=0, train_entries=train,
        sealed_entries=sealed, output_dir=tmp_path / "curve",
    )
    with pytest.raises(ValueError, match="search-free"):
        _load_runtime_entries(config)
    torch.save(make_match_start_entries(pool_seed=7, matches_per_archetype=1), train)
    loaded, _, _, _ = _load_runtime_entries(config)
    assert len(loaded) == 6


def test_raw_match_start_runs_through_ppo_collector():
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.frontier_ppo import collect_frontier_rollout
    from simulator.rl.v4.model_v4 import RecurrentV4Policy, V4ValueHead

    model_config = frontier_model_config()
    policy = RecurrentV4Policy(model_config)
    critic = V4ValueHead(model_config.gru_hidden_dim)
    entry = make_match_start_entries(pool_seed=9, matches_per_archetype=1)[0]
    batch, info = collect_frontier_rollout(
        policy, critic, None, [entry], n_decisions=2, n_lanes=1,
        episode_cap=2, device="cpu", seed=1, step_tower_reward=True,
    )
    assert tuple(batch.trajectory.rewards.shape) == (1, 2)
    assert torch.isfinite(batch.trajectory.rewards).all()
    assert info["illegal_attempts"] == 0


def test_random_actor_is_a_loadable_plain_checkpoint(tmp_path):
    from simulator.rl.v4.frontier import frontier_model_config
    from simulator.rl.v4.v4_ppo import load_frozen_anchor

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "random_actor.pt"
    subprocess.run(
        [sys.executable, str(root / "scripts/run_v4_random_actor.py"),
         "--seed", "7", "--out", str(output)],
        check=True, capture_output=True, text=True,
    )
    actor = load_frozen_anchor(str(output), frontier_model_config(), device="cpu")
    assert actor.anchor_sha256
