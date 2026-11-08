"""Diverse starts use the native public runtime and retain benchmark isolation."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from simulator.actions import WaitAction
from simulator.rl.foundation.opponent_styles import (
    DEV_STYLE_IDS,
    TRAIN_STYLE_IDS,
    resolve_style_config,
    style_identity,
)
from simulator.rl.v4.frontier import (
    PoolMatchConfig,
    frontier_model_config,
    make_pool_match_start,
)
from simulator.rl.v4.frontier_eval import load_frontier_state, make_opponent
from simulator.rl.v4.frontier_ppo import collect_frontier_rollout
from simulator.rl.v4.frontier_ppo.environment import make_frontier_env
from simulator.rl.v4.frontier_ppo.evaluation import full_match_record
from simulator.rl.v4.learning_curve import LearningCurveConfig, _load_runtime_entries
from simulator.rl.v4.model_v4 import RecurrentV4Policy, V4ValueHead
from simulator.rl.v4.raw_ppo import (
    DIVERSE_PPO_STARTS_VERSION,
    RAW_PPO_STARTS_VERSION,
    make_diverse_match_start_entries,
    make_match_start_entries,
)
from simulator.rl.v4.real_decks import catalog_sha256, load_real_decks
from simulator.roster import PLAYER_DECK


ROOT = Path(__file__).resolve().parents[2]
STARTS_SCRIPT = ROOT / "scripts/run_v4_raw_ppo_starts.py"


@pytest.fixture(scope="module")
def diverse_entries():
    return make_diverse_match_start_entries(pool_seed=71)


def test_diverse_starts_cover_every_training_deck_and_only_training_styles(diverse_entries):
    catalog = load_real_decks()
    training = {deck.deck_id: deck for deck in catalog if deck.split == "train"}
    reserved = {deck.deck_id for deck in catalog if deck.split == "eval"}
    counts = Counter((entry["opponent"]["deck_id"], entry["opponent"]["style"]) for entry in diverse_entries)
    assert len(training) == 30
    assert len(diverse_entries) == 90
    assert counts == Counter({(deck_id, style): 1 for deck_id in training for style in TRAIN_STYLE_IDS})
    assert not ({entry["opponent"]["deck_id"] for entry in diverse_entries} & reserved)
    assert not ({entry["opponent"]["style"] for entry in diverse_entries} & set(DEV_STYLE_IDS))
    for entry in diverse_entries:
        spec, provenance = entry["opponent"], entry["provenance"]
        deck = training[spec["deck_id"]]
        assert entry["split"] == spec["real_deck_split"] == "train"
        assert entry["learner_deck"] == list(PLAYER_DECK)
        assert tuple(entry["opponent_deck"]) == deck.cards
        assert spec["archetype"] == deck.archetype
        assert provenance["generator"] == RAW_PPO_STARTS_VERSION
        assert provenance["diversity_generator"] == DIVERSE_PPO_STARTS_VERSION
        assert provenance["real_deck_catalog_sha256"] == catalog_sha256()
        assert provenance["real_deck_id"] == deck.deck_id
        assert entry["initial_state"]["tick"] == 0
        assert "verdict" not in entry and "hidden0" not in entry
        assert spec["style_identity"] == style_identity(
            deck.archetype, spec["style"], seed=spec["controller_seed"]
        )


def test_diverse_starts_reproduce_openings_and_controller_identity(diverse_entries):
    assert diverse_entries == make_diverse_match_start_entries(pool_seed=71)
    assert len({entry["entry_id"] for entry in diverse_entries}) == 90
    assert len({entry["provenance"]["match_seed"] for entry in diverse_entries}) == 90
    assert len({entry["opponent"]["controller_seed"] for entry in diverse_entries}) == 90
    assert len({entry["provenance"]["state_hash"] for entry in diverse_entries}) == 90
    openings = make_diverse_match_start_entries(pool_seed=72, matches_per_deck=2, styles=("tempo",))
    assert len(openings) == 60
    for deck_id in {entry["opponent"]["deck_id"] for entry in openings}:
        pair = [entry for entry in openings if entry["opponent"]["deck_id"] == deck_id]
        assert {entry["provenance"]["match_index"] for entry in pair} == {0, 1}
        assert pair[0]["provenance"]["match_seed"] != pair[1]["provenance"]["match_seed"]
        assert pair[0]["provenance"]["state_hash"] != pair[1]["provenance"]["state_hash"]
        assert pair[0]["opponent"] == pair[1]["opponent"]


@pytest.mark.parametrize("styles", ((), ("tempo", "tempo"), ("adaptive",), ("split-pressure",), ("unknown",)))
def test_diverse_builder_rejects_empty_duplicate_unknown_or_development_styles(styles):
    with pytest.raises(ValueError, match="training styles"):
        make_diverse_match_start_entries(pool_seed=71, styles=styles)


@pytest.mark.parametrize("kwargs", (
    {"pool_seed": -1}, {"pool_seed": True},
    {"matches_per_deck": 0}, {"matches_per_deck": True},
))
def test_diverse_builder_rejects_invalid_workload_values(kwargs):
    with pytest.raises(ValueError):
        make_diverse_match_start_entries(**dict({"pool_seed": 71}, **kwargs))


def test_all_diverse_openings_load_and_step_through_public_environment(diverse_entries):
    for entry in diverse_entries:
        env = make_frontier_env()
        # The native V4 loader reconstructs public memory without asking the
        # legacy V1 contract to represent the broader enemy catalog.
        load_frontier_state(env, entry["initial_state"])
        assert env.state.state_hash() == entry["provenance"]["state_hash"]
        public = env.observe_v2_for_viewer(0)
        assert public.legal_wait
        assert public.legal_play.shape == (4, 32, 18)
        opponent = make_opponent(entry["opponent"], 123_456, env=env)
        assert opponent.identity == entry["opponent"]["style_identity"]
        action = opponent.choose_action(env.engine, env.state, 1)
        assert env.engine.validate_action(env.state, action) is None
        result = env.step_v2((WaitAction(0), action))
        assert not result.terminated and not result.truncated
        assert env.state.tick == env.decision_interval_ticks
        assert not any(event.kind == "action_rejected" for event in env.state.events)


def test_diverse_entries_load_in_native_reward_only_ppo_and_collect(diverse_entries, tmp_path):
    train, sealed = tmp_path / "train.pt", tmp_path / "sealed.pt"
    torch.save(diverse_entries, train)
    torch.save([], sealed)
    loaded, heldout, _, _ = _load_runtime_entries(LearningCurveConfig(
        training_source="match-start", bc_coef=0,
        train_entries=train, sealed_entries=sealed, output_dir=tmp_path / "curve",
    ))
    assert loaded == diverse_entries and heldout == []
    model_config = frontier_model_config()
    with torch.random.fork_rng():
        torch.manual_seed(8)
        actor = RecurrentV4Policy(model_config)
        critic = V4ValueHead(model_config.gru_hidden_dim)
    batch, info = collect_frontier_rollout(
        actor, critic, None, loaded, n_decisions=2, n_lanes=1,
        episode_cap=1440, device="cpu", seed=19, step_tower_reward=True,
    )
    assert tuple(batch.trajectory.rewards.shape) == (1, 2)
    assert torch.isfinite(batch.trajectory.rewards).all()
    assert info["illegal_attempts"] == 0
    # Artifact coverage does not imply a short rollout visited or completed
    # every available matchup. Unfinished tails are neither draws nor cuts.
    assert info["entry_coverage"] == {"available": 90, "visited": 1, "completed": 0}
    assert set(info["entry_outcomes"]) <= {entry["entry_id"] for entry in loaded}
    assert sum(counts["visits"] for counts in info["entry_outcomes"].values()) == 1
    assert all(counts[field] == 0 for counts in info["entry_outcomes"].values()
               for field in ("wins", "losses", "draws", "truncated"))


@pytest.mark.parametrize("profile", ("pinned", "mixed", "real"))
def test_legacy_starts_keep_their_historical_openings_and_no_styles(profile):
    entries = make_match_start_entries(pool_seed=73, matches_per_archetype=1, profile=profile)
    for entry in entries:
        assert "style" not in entry["opponent"]
        assert "style_identity" not in entry["opponent"]
        assert "diversity_generator" not in entry["provenance"]
        archetype = entry["opponent"]["archetype"]
        env, scenario, match_seed, _ = make_pool_match_start(PoolMatchConfig(
            archetype=archetype, pool_seed=73, match_index=0,
            allow_variants=profile == "mixed",
            deck_profile="real" if profile == "real" else "scripted",
        ))
        assert entry["initial_state"] == env.save_state()
        assert entry["opponent_deck"] == list(scenario.deck.cards)
        assert entry["opponent"]["strategy"] == scenario.strategy
        assert entry["provenance"]["match_seed"] == match_seed
    assert make_match_start_entries(pool_seed=73, matches_per_archetype=1) == make_match_start_entries(
        pool_seed=73, matches_per_archetype=1, profile="pinned"
    )


def test_diverse_real_cli_writes_native_entries_and_pinned_manifest(tmp_path):
    output = tmp_path / "diverse.pt"
    result = subprocess.run([
        sys.executable, str(STARTS_SCRIPT), "--profile", "diverse-real",
        "--pool-seed", "74", "--matches-per-deck", "2", "--styles", "tempo",
        "--out", str(output),
    ], check=True, capture_output=True, text=True)
    entries = torch.load(output, map_location="cpu", weights_only=False)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert entries == make_diverse_match_start_entries(pool_seed=74, matches_per_deck=2, styles=("tempo",))
    assert manifest["entry_count"] == len(entries) == 60
    assert manifest["profile"] == "diverse-real"
    assert manifest["matches_per_deck"] == 2
    assert manifest["styles"] == ["tempo"]
    assert manifest["real_deck_catalog_sha256"] == catalog_sha256()
    assert manifest["entry_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(result.stdout)["entry_sha256"] == manifest["entry_sha256"]


def test_cli_default_retains_pinned_openings(tmp_path):
    output = tmp_path / "pinned.pt"
    subprocess.run([
        sys.executable, str(STARTS_SCRIPT), "--pool-seed", "76",
        "--matches-per-archetype", "1", "--out", str(output),
    ], check=True, capture_output=True, text=True)
    entries = torch.load(output, map_location="cpu", weights_only=False)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert entries == make_match_start_entries(pool_seed=76, matches_per_archetype=1)
    assert manifest["profile"] == "pinned" and manifest["styles"] == []
    assert manifest["matches_per_deck"] is None


@pytest.mark.parametrize("args,expected", (
    (("--profile", "diverse-real", "--styles", "adaptive"), "held-out styles"),
    (("--profile", "pinned", "--styles", "tempo"), "requires --profile diverse-real"),
))
def test_cli_rejects_reserved_styles_and_ignored_style_options(tmp_path, args, expected):
    output = tmp_path / "invalid.pt"
    result = subprocess.run([
        sys.executable, str(STARTS_SCRIPT), *args, "--out", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 2 and expected in result.stderr
    assert not output.exists() and not output.with_suffix(".json").exists()


def test_native_match_identity_pins_named_and_config_only_custom_styles(diverse_entries):
    entry = diverse_entries[0]
    spec = dict(entry["opponent"], deck=entry["opponent_deck"])
    spec["style_config"] = {"reserve_milli": 1_000, "cycle_floor_milli": 4_000}
    spec["style_identity"] = style_identity(
        spec["strategy"], spec["style"], seed=spec["controller_seed"], config=spec["style_config"],
    )
    config_only = copy.deepcopy(spec)
    config_only["style_config"] = resolve_style_config(spec["style"], spec["style_config"]).as_dict()
    del config_only["style"]
    with torch.random.fork_rng():
        torch.manual_seed(9)
        actor = RecurrentV4Policy(frontier_model_config())
    matches = full_match_record(
        policy=actor, opponent_specs=[spec, config_only], seeds=[75],
        max_decisions=1, device="cpu",
    )
    assert len(matches) == 2
    assert matches[0]["opponent"]["style"] == spec["style_identity"]
    assert matches[1]["opponent"]["style"] == spec["style_identity"]
    assert matches[0]["policy_seed"] == matches[1]["policy_seed"]
    assert matches[0]["action_trace_sha256"] == matches[1]["action_trace_sha256"]
    assert matches[0]["final_state_hash"] == matches[1]["final_state_hash"]
    assert all(match["invalid_actions"] == [0, 0] for match in matches)
