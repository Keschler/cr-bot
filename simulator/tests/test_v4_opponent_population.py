"""Population identity, coverage and training-only feedback regressions."""

from dataclasses import asdict, replace
import hashlib
import json
import random
from pathlib import Path
import runpy

import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy
from simulator.rl.v4.opponent_population import (
    LearnedOpponentMember, OpponentPopulation, TrainMatchupResult,
    make_population_match_start_entries,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def members(tmp_path):
    config = ModelConfigV4(
        model_dim=8, spatial_channels=4, fused_dim=16, gru_hidden_dim=16,
        transformer_heads=2, transformer_layers=1, transformer_ff_dim=16,
        spatial_head_dim=4,
    )
    result = []
    with torch.random.fork_rng(devices=[]):
        for index in range(3):
            torch.manual_seed(index)
            path = tmp_path / f"snapshot-{index}.pt"
            torch.save(RecurrentV4Policy(config).state_dict(), path)
            result.append(LearnedOpponentMember(
                member_id=f"snapshot-{index}", checkpoint=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_config=config,
                split="dev" if index == 2 else "train", sampling_seed=index,
                generation=index,
            ))
    return tuple(result)


def test_manifest_roundtrip_pins_model_deck_hash_and_inference(members, tmp_path):
    population = OpponentPopulation("league-1", members)
    path = tmp_path / "population.json"
    path.write_text(json.dumps(population.to_dict()), encoding="utf-8")
    restored = OpponentPopulation.from_json(path)
    assert restored == population
    spec = restored.members[0].to_opponent_spec(episode_seed=17)
    assert spec["model_config"] == asdict(members[0].model_config)
    assert spec["sha256"] == members[0].sha256
    assert spec["deck"] == list(members[0].deck)
    assert spec["inference"] == "stochastic"
    assert spec["population_split"] == "train"
    assert spec == members[0].to_opponent_spec(episode_seed=17)
    assert spec["sampling_seed"] != members[0].to_opponent_spec(episode_seed=18)["sampling_seed"]


def test_historical_snapshots_are_retained_with_finite_coverage(members):
    original = OpponentPopulation("league", (members[0],))
    expanded = original.with_member(members[1])
    assert original.members == (members[0],)
    assert expanded.members == members[:2]
    schedule = expanded.schedule(episodes=12, seed=55)
    assert set(member.member_id for member in schedule[:2]) == {"snapshot-0", "snapshot-1"}
    assert schedule == expanded.schedule(episodes=12, seed=55)
    with pytest.raises(ValueError, match="cover every member"):
        expanded.schedule(episodes=1, seed=55)
    with pytest.raises(ValueError, match="unique"):
        expanded.with_member(members[0])


def test_train_only_difficulty_prioritization_has_coverage_floor(members):
    population = OpponentPopulation("league", members)
    results = tuple(
        TrainMatchupResult(f"{member_id}:{index}", member_id, score, "train", True, False)
        for member_id, score in (("snapshot-0", 0), ("snapshot-1", 1))
        for index in range(8)
    )
    weights = population.sampling_weights(train_results=results, coverage_fraction=0.4)
    assert weights["snapshot-0"] > weights["snapshot-1"] >= 0.2
    assert sum(weights.values()) == pytest.approx(1)
    assert population.sampling_weights() == {"snapshot-0": 0.5, "snapshot-1": 0.5}
    assert population.sampling_weights(split="dev") == {"snapshot-2": 1.0}
    with pytest.raises(ValueError, match="development.*feedback"):
        population.sample(seed=1, episode_index=0, split="dev", train_results=results)
    with pytest.raises(ValueError, match="non-training or unknown"):
        population.sampling_weights(train_results=(
            TrainMatchupResult("dev-match", "snapshot-2", 0, "train", True, False),
        ))
    with pytest.raises(ValueError, match="duplicate"):
        population.sampling_weights(train_results=(results[0], results[0]))


@pytest.mark.parametrize("split, completed, truncated, score", [
    ("dev", True, False, 0), ("eval", True, False, 0),
    ("reserved", True, False, 0), ("train", False, False, 0),
    ("train", True, True, 0), ("train", True, False, float("nan")),
    ("train", True, False, True), ("train", True, False, 0.7),
])
def test_censored_or_nontrain_feedback_cannot_enter_population(split, completed, truncated, score):
    with pytest.raises(ValueError):
        TrainMatchupResult("match", "member", score, split, completed, truncated)


def test_population_sampling_is_stateless_and_does_not_touch_global_rng(members):
    population = OpponentPopulation("league", members)
    state = random.getstate()
    first = tuple(population.sample(seed=11, episode_index=index).member_id for index in range(40))
    assert random.getstate() == state
    random.seed(789)
    try:
        second = tuple(population.sample(seed=11, episode_index=index).member_id for index in range(40))
        reordered = OpponentPopulation("league", tuple(reversed(members)))
        assert second == first
        assert tuple(reordered.sample(seed=11, episode_index=index).member_id for index in range(40)) == first
    finally:
        random.setstate(state)


def test_hash_deck_config_and_split_mismatches_fail_early(members):
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(members[0], sha256="0" * 64)
    with pytest.raises(ValueError, match="canonical Hog"):
        replace(members[0], deck=("x-bow", *members[0].deck[1:]))
    with pytest.raises(ValueError, match="every ModelConfigV4 field"):
        replace(members[0], model_config={"model_dim": 8})
    with pytest.raises(ValueError, match="public input contract"):
        replace(members[0], model_config=replace(members[0].model_config, hand_token_dim=32))
    with pytest.raises(ValueError, match="both train and dev"):
        OpponentPopulation("league", (members[0], replace(members[0], member_id="dev-copy", split="dev")))
    manifest = OpponentPopulation("league", members).to_dict()
    manifest["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        OpponentPopulation.from_dict(manifest)


def test_train_population_raw_openings_are_reproducible_public_and_collector_usable(members, tmp_path):
    from simulator.rl.v4.frontier_eval import load_frontier_state, make_opponent
    from simulator.rl.v4.frontier_ppo import collect_frontier_rollout
    from simulator.rl.v4.frontier_ppo.environment import make_frontier_env
    from simulator.rl.v4.learning_curve import LearningCurveConfig, _load_runtime_entries
    from simulator.rl.v4.model_v4 import V4ValueHead
    from simulator.rl.v4.raw_ppo import RAW_PPO_STARTS_VERSION

    population = OpponentPopulation("league", members)
    entries = make_population_match_start_entries(population, pool_seed=13, episodes=4)
    assert entries == make_population_match_start_entries(population, pool_seed=13, episodes=4)
    assert {entry["opponent"]["member_id"] for entry in entries} == {"snapshot-0", "snapshot-1"}
    assert all(entry["initial_state"]["tick"] == 0 and entry["split"] == "train" for entry in entries)
    assert all("verdict" not in entry and "hidden0" not in entry for entry in entries)
    assert len({entry["provenance"]["match_hash"] for entry in entries}) == 4
    for entry in entries:
        assert entry["provenance"]["generator"] == RAW_PPO_STARTS_VERSION
        assert entry["provenance"]["profile"] == "learned-population"
        assert entry["provenance"]["population_sha256"] == population.fingerprint
        assert entry["opponent"]["population_schema_version"] == 1
        env = make_frontier_env()
        load_frontier_state(env, entry["initial_state"])
        assert env.state.state_hash() == entry["provenance"]["state_hash"]
        opponent = make_opponent(entry["opponent"], 0, env=env)
        assert opponent.seed == entry["opponent"]["sampling_seed"]
        assert env.engine.validate_action(env.state, opponent.choose_action(env.engine, env.state, 1)) is None

    train = tmp_path / "train.pt"
    torch.save(entries, train)
    config = LearningCurveConfig(
        training_source="match-start", bc_coef=0, train_entries=train,
        output_dir=tmp_path / "curve",
    )
    loaded, _, _, _ = _load_runtime_entries(config)
    assert len(loaded) == 4
    policy = RecurrentV4Policy(members[0].model_config)
    critic = V4ValueHead(members[0].model_config.gru_hidden_dim)
    batch, info = collect_frontier_rollout(
        policy, critic, None, loaded, n_decisions=2, n_lanes=1,
        episode_cap=2, device="cpu", seed=7, step_tower_reward=True,
    )
    assert tuple(batch.trajectory.rewards.shape) == (1, 2)
    assert info["illegal_attempts"] == 0


def test_population_openings_reject_wrong_architecture_and_dev_only_population(members):
    invalid = replace(members[0], model_config=replace(members[0].model_config, model_dim=16))
    with pytest.raises(RuntimeError, match="size mismatch"):
        make_population_match_start_entries(OpponentPopulation("wrong", (invalid,)), pool_seed=1, episodes=1)
    with pytest.raises(ValueError, match="no train members"):
        make_population_match_start_entries(OpponentPopulation("dev-only", (members[2],)), pool_seed=1, episodes=1)


def test_population_start_cli_writes_pinned_manifest_without_overwriting(members, tmp_path, capsys):
    root = Path(__file__).resolve().parents[2]
    main = runpy.run_path(str(root / "scripts" / "run_v4_opponent_population.py"))["main"]
    population = OpponentPopulation("league", members)
    manifest = tmp_path / "population.json"
    manifest.write_text(json.dumps(population.to_dict()), encoding="utf-8")
    feedback = tmp_path / "feedback.json"
    feedback.write_text(json.dumps([asdict(TrainMatchupResult(
        "train-match-1", "snapshot-0", 0, "train", True, False,
    ))]), encoding="utf-8")
    output = tmp_path / "openings.pt"
    unrelated_temp = output.with_suffix(".pt.tmp")
    unrelated_temp.write_text("preserve", encoding="utf-8")
    args = [
        "--population", str(manifest), "--pool-seed", "9", "--episodes", "4",
        "--train-feedback", str(feedback), "--out", str(output),
    ]
    assert main(args) == 0
    saved = torch.load(output, weights_only=False)
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert len(saved) == receipt["entry_count"] == 4
    assert receipt["split"] == "train"
    assert receipt["population_sha256"] == population.fingerprint
    assert receipt["population_file_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["entry_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["training_feedback_count"] == 1
    assert receipt["sampling_weights"]["snapshot-0"] > receipt["sampling_weights"]["snapshot-1"]
    assert set(receipt["member_episode_counts"]) == {"snapshot-0", "snapshot-1"}
    assert unrelated_temp.read_text() == "preserve"
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert output.read_bytes() == original
    capsys.readouterr()


def test_population_start_cli_rejects_reserved_feedback_before_writing(members, tmp_path, capsys):
    root = Path(__file__).resolve().parents[2]
    main = runpy.run_path(str(root / "scripts" / "run_v4_opponent_population.py"))["main"]
    manifest = tmp_path / "population.json"
    manifest.write_text(json.dumps(OpponentPopulation("league", members).to_dict()))
    feedback = tmp_path / "feedback.json"
    feedback.write_text(json.dumps([{
        "match_id": "heldout-1", "member_id": "snapshot-0", "learner_score": 0,
        "split": "reserved", "completed": True, "truncated": False,
    }]))
    output = tmp_path / "must-not-exist.pt"
    with pytest.raises(SystemExit) as error:
        main([
            "--population", str(manifest), "--episodes", "2", "--train-feedback", str(feedback),
            "--out", str(output),
        ])
    assert error.value.code == 2
    assert not output.exists() and not output.with_suffix(".json").exists()
    assert "train results only" in capsys.readouterr().err
