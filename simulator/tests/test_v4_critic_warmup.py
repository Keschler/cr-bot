"""Completed-episode targets, held-out value fitting and actor-update gates."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4.critic_warmup import (
    completed_return_targets, critic_warmup_gate, warmup_critic,
)
from simulator.rl.v4.learning_curve import (
    LearningCurveConfig, LearningCurveHooks, TrainingState, run_learning_curve,
)
from simulator.rl.v4.model_v4 import V4ValueHead


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _batch(features, rewards, terminated, reset_mask, *, truncated=None, valid_mask=None):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    return SimpleNamespace(
        recurrent=features,
        trajectory=SimpleNamespace(
            rewards=rewards, terminated=torch.as_tensor(terminated, dtype=torch.bool),
            truncated=(torch.zeros_like(rewards, dtype=torch.bool) if truncated is None
                       else torch.as_tensor(truncated, dtype=torch.bool)),
            sequence=SimpleNamespace(reset_mask=torch.as_tensor(reset_mask, dtype=torch.bool)),
            valid_mask=(None if valid_mask is None else torch.as_tensor(valid_mask, dtype=torch.bool)),
        ),
    )


def test_returns_require_real_reset_and_terminal_and_exclude_cuts_prefixes_and_tails():
    rewards = np.array([
        [0, 0, 1, 0, 0, -1],
        [0, -1, 0, 0, 0, 0],
        [9, 9, 0, 0, 1, 0],
    ], dtype=float)
    terminated = np.zeros_like(rewards, dtype=bool)
    terminated[0, [2, 5]] = True
    terminated[1, 1] = True  # Unknown prefix: no reset in this batch.
    terminated[2, 4] = True
    truncated = np.zeros_like(terminated)
    truncated[1, 3] = True
    reset = np.zeros_like(terminated)
    reset[0, [0, 3]] = True
    reset[1, [2, 4]] = True  # A cut and an unfinished tail.
    reset[2, [0, 2]] = True  # A later reset discards an unfinished prefix.
    targets, selected, episodes = completed_return_targets(
        rewards, terminated, truncated, reset, gamma=0.5,
    )
    np.testing.assert_array_equal(targets[0], [.25, .5, 1, -.25, -.5, -1])
    np.testing.assert_array_equal(targets[1], np.zeros(6))
    np.testing.assert_array_equal(targets[2], [0, 0, .25, .5, 1, 0])
    np.testing.assert_array_equal(selected, [
        [True] * 6, [False] * 6, [False, False, True, True, True, False],
    ])
    assert targets.dtype == np.float32
    assert episodes == 3


def test_padding_breaks_episode_until_a_new_genuine_reset():
    rewards = np.array([[0, 0, 1, 0, -1]], dtype=float)
    term = np.array([[False, False, True, False, True]])
    reset = np.array([[True, False, False, True, False]])
    valid = np.array([[True, False, True, True, True]])
    targets, selected, episodes = completed_return_targets(
        rewards, term, np.zeros_like(term), reset, gamma=1, valid_mask=valid,
    )
    np.testing.assert_array_equal(selected, [[False, False, False, True, True]])
    np.testing.assert_array_equal(targets, [[0, 0, 0, -1, -1]])
    assert episodes == 1


@pytest.mark.parametrize("gamma", [0, -1, 1.01, float("nan"), float("inf")])
def test_return_targets_reject_invalid_discount(gamma):
    flags = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="gamma"):
        completed_return_targets([[1]], flags, ~flags, flags, gamma=gamma)


def test_return_targets_validate_shapes_flags_finite_rewards_and_float32_range():
    flags = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="boolean boundaries"):
        completed_return_targets([[1]], [[1]], ~flags, flags, gamma=1)
    with pytest.raises(ValueError, match="boolean boundaries"):
        completed_return_targets([1], flags, ~flags, flags, gamma=1)
    with pytest.raises(ValueError, match="ambiguous"):
        completed_return_targets([[1]], flags, flags, flags, gamma=1)
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonfinite reward"):
            completed_return_targets([[value]], flags, ~flags, flags, gamma=1)
    with pytest.raises(ValueError, match="valid mask"):
        completed_return_targets([[1]], flags, ~flags, flags, gamma=1, valid_mask=[[1]])
    with pytest.raises(ValueError, match="float32 range"):
        completed_return_targets(
            [[2e38, 2e38]], [[False, True]], [[False, False]], [[True, False]], gamma=1,
        )
    empty = np.zeros((2, 0), dtype=bool)
    targets, selected, episodes = completed_return_targets(np.zeros((2, 0)), empty, empty, empty, gamma=1)
    assert targets.shape == selected.shape == (2, 0) and episodes == 0


def _synthetic_batch(*, shift=0.0):
    outcomes = torch.tensor([1., -1.] * 4)
    features = torch.zeros(8, 4, 4)
    features[:, :, 0] = outcomes[:, None]
    features[:, :, 1] = torch.linspace(0, .8, 4)[None, :] + shift
    features[:, :, 2] = 1
    features[:, :, 3] = torch.linspace(-.2, .2, 4)[None, :] - shift
    features.requires_grad_(True)
    rewards = torch.zeros(8, 4)
    rewards[:, -1] = outcomes
    terminated = torch.zeros(8, 4, dtype=torch.bool)
    terminated[:, -1] = True
    reset = torch.zeros_like(terminated)
    reset[:, 0] = True
    return _batch(features, rewards, terminated, reset)


def test_actual_v4_value_head_learns_heldout_targets_without_actor_or_feature_gradients():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        actor = torch.nn.Linear(4, 4)
        critic = V4ValueHead(4)
    original_actor = {name: value.clone() for name, value in actor.state_dict().items()}
    original_critic = {name: value.clone() for name, value in critic.state_dict().items()}
    shapes = []
    hook = critic.register_forward_pre_hook(lambda _module, args: shapes.append(tuple(args[0].shape)))
    train, validation = _synthetic_batch(), _synthetic_batch(shift=.07)
    try:
        report = warmup_critic(
            policy=actor, critic=critic, optimizer=torch.optim.Adam(critic.parameters(), lr=.03),
            batch=train, validation_batch=validation, epochs=100, batch_size=16, gamma=1, seed=43,
        )
    finally:
        hook.remove()
    assert report["complete_train_episodes"] == report["complete_validation_episodes"] == 8
    assert report["train_outcomes"] == report["validation_outcomes"] == {"wins": 4, "losses": 4, "draws": 0}
    assert report["initial"]["rows"] == report["validation"]["rows"] == 32
    assert report["validation"]["mse"] < report["initial_validation"]["mse"] * .1
    assert report["validation"]["explained_variance"] > .9
    assert report["optimizer_steps"] == 200
    assert critic_warmup_gate(report)["eligible"]
    assert all(len(shape) == 3 and shape[1:] == (1, 4) for shape in shapes)
    assert all(torch.equal(actor.state_dict()[name], value) for name, value in original_actor.items())
    assert any(not torch.equal(critic.state_dict()[name], value) for name, value in original_critic.items())
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert train.recurrent.grad is validation.recurrent.grad is None


def test_warmup_uses_valid_mask_and_rejects_missing_cut_or_nonfinite_features():
    actor, critic = torch.nn.Linear(4, 4), V4ValueHead(4)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.01)
    batch = _synthetic_batch()
    batch.trajectory.valid_mask = torch.ones(8, 4, dtype=torch.bool)
    batch.trajectory.valid_mask[0, 1] = False
    report = warmup_critic(policy=actor, critic=critic, optimizer=optimizer, batch=batch, epochs=1)
    assert report["complete_train_episodes"] == 7
    assert report["train_outcomes"] == {"wins": 3, "losses": 4, "draws": 0}
    batch.recurrent = None
    with pytest.raises(ValueError, match="stored recurrent"):
        warmup_critic(policy=actor, critic=critic, optimizer=optimizer, batch=batch)
    batch = _synthetic_batch()
    batch.trajectory.terminated.zero_()
    with pytest.raises(ValueError, match="no complete"):
        warmup_critic(policy=actor, critic=critic, optimizer=optimizer, batch=batch)
    batch = _synthetic_batch()
    batch.recurrent = batch.recurrent.detach().clone()
    batch.recurrent[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="features must be finite"):
        warmup_critic(policy=actor, critic=critic, optimizer=optimizer, batch=batch)
    with pytest.raises(ValueError, match="share actor"):
        warmup_critic(policy=critic, critic=critic, optimizer=optimizer, batch=_synthetic_batch())


def _good_report():
    return {
        "complete_train_episodes": 8, "complete_validation_episodes": 8,
        "train_outcomes": {"wins": 4, "losses": 4, "draws": 0},
        "validation_outcomes": {"wins": 4, "losses": 4, "draws": 0},
        "initial_validation": {"mse": 1.0, "explained_variance": 0.0},
        "validation": {"mse": .25, "explained_variance": .75},
        "actor_unchanged": True,
    }


@pytest.mark.parametrize("field, value, reason", [
    ("complete_train_episodes", 3, "too_few_complete_train_episodes"),
    ("complete_validation_episodes", 3, "too_few_complete_validation_episodes"),
    ("train_outcomes", {"wins": 8}, "train_lacks_win_loss_coverage"),
    ("validation_outcomes", {"losses": 8}, "validation_lacks_win_loss_coverage"),
    ("actor_unchanged", False, "actor_weights_changed"),
    ("validation", {"mse": 1.1, "explained_variance": .2}, "validation_mse_did_not_improve"),
    ("validation", {"mse": .1, "explained_variance": .01}, "validation_explained_variance_below_gate"),
    ("validation", {"mse": .1, "explained_variance": float("nan")}, "validation_explained_variance_below_gate"),
    ("validation", {"mse": .1, "explained_variance": float("inf")}, "validation_explained_variance_below_gate"),
    ("validation", {"mse": float("nan"), "explained_variance": .8}, "validation_mse_did_not_improve"),
    ("initial_validation", {"mse": float("inf")}, "validation_mse_did_not_improve"),
])
def test_warmup_gate_rejects_insufficient_coverage_or_failed_value_prediction(field, value, reason):
    report = _good_report()
    report[field] = value
    gate = critic_warmup_gate(report)
    assert not gate["eligible"] and reason in gate["reasons"]


def test_warmup_gate_validates_thresholds_and_accepts_declared_success():
    assert critic_warmup_gate(_good_report())["eligible"]
    for minimum in (0, -1, True):
        with pytest.raises(ValueError, match="minimum_episodes"):
            critic_warmup_gate(_good_report(), minimum_episodes=minimum)
    for minimum in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="minimum_validation_ev"):
            critic_warmup_gate(_good_report(), minimum_validation_ev=minimum)


def test_anchored_match_start_is_explicit_and_reward_only_defaults_keep_old_contract():
    default = LearningCurveConfig()
    assert default.critic_warmup_epochs == 0 and default.gamma == .99 and default.gae_lambda == .95
    reward_only = LearningCurveConfig(training_source="match-start", bc_coef=0)
    assert reward_only.bc_coef == 0 and reward_only.critic_warmup_epochs == 0
    with pytest.raises(ValueError, match="bc_coef=0"):
        LearningCurveConfig(training_source="match-start", bc_coef=.05)
    anchored = LearningCurveConfig(training_source="anchored-match-start", bc_coef=.05)
    assert anchored.bc_coef == .05
    with pytest.raises(ValueError, match="positive frozen-policy retention"):
        replace(anchored, bc_coef=0)
    warm = replace(anchored, reward_mode="terminal", episode_cap=1440, critic_warmup_epochs=1)
    for change in (
        {"training_source": "frontier"}, {"reward_mode": "tower-delta"},
        {"smoke": True}, {"evaluation_only": True}, {"critic_warmup_decisions": 1439},
    ):
        with pytest.raises(ValueError, match="critic warmup"):
            replace(warm, **change)


@pytest.mark.parametrize("eligible", [False, True])
def test_learning_curve_gate_stops_before_actor_updates_and_preserves_zero_checkpoint(eligible, tmp_path):
    from simulator.rl.v4.raw_ppo import RAW_PPO_STARTS_VERSION
    from simulator.rl.v4.real_decks import catalog_sha256, real_decks_for

    deck = real_decks_for("train", "beatdown")[0]
    actor = torch.nn.Linear(1, 1)
    original = copy.deepcopy(actor.state_dict())
    checkpoint = tmp_path / "actor.pt"
    torch.save(original, checkpoint)
    train_path = tmp_path / "train.pt"
    torch.save([{
        "entry_id": f"train-{index}", "split": "train", "initial_state": {"tick": 0},
        "opponent_deck_id": deck.deck_id, "opponent_deck": list(deck.cards),
        "real_deck_split": "train", "opponent": {"archetype": deck.archetype},
        "provenance": {"generator": RAW_PPO_STARTS_VERSION, "profile": "real",
                       "real_deck_id": deck.deck_id, "real_deck_split": "train",
                       "real_deck_catalog_sha256": catalog_sha256(),
                       "state_hash": f"state-{index}", "match_seed": index + 11},
    } for index in range(2)], train_path)
    events = []

    def warmup(**kwargs):
        events.append("warmup")
        assert all(torch.equal(kwargs["policy"].state_dict()[name], value) for name, value in original.items())
        return {"actor_unchanged": True, "gate": {
            "eligible": eligible, "reasons": [] if eligible else ["validation_mse_did_not_improve"],
        }}

    def collect(**_kwargs):
        assert eligible
        events.append("collect")
        return {}, {}

    def update(**kwargs):
        events.append("update")
        with torch.no_grad():
            kwargs["policy"].weight.add_(.1)
        return {}

    def matches(**kwargs):
        events.append("evaluate")
        return [{"archetype": deck.archetype, "deck_id": deck.deck_id, "seed": seed,
                 "result": "win", "terminated": True, "decisions": 1}
                for seed in kwargs["seeds"]]

    hooks = LearningCurveHooks(
        load_policy=lambda **_kwargs: copy.deepcopy(actor), clone_policy=copy.deepcopy,
        make_training_state=lambda **_kwargs: TrainingState(), collect_rollout=collect,
        update=update, evaluate_matches=matches, warmup_critic=warmup,
        save_checkpoint=lambda **kwargs: torch.save(kwargs["policy"].state_dict(), kwargs["path"]),
    )
    config = LearningCurveConfig(
        checkpoint=checkpoint, train_entries=train_path, output_dir=tmp_path / "curve",
        training_source="anchored-match-start", bc_coef=.05, critic_warmup_epochs=1,
        reward_mode="terminal", episode_cap=1440, update_schedule=(0, 1),
        opponent_specs=({"archetype": deck.archetype, "deck_id": deck.deck_id},), eval_seeds=(3, 4),
    )
    report = run_learning_curve(config, hooks=hooks)
    assert events[0] == "warmup"
    assert [point["update"] for point in report["points"]] == ([0, 1] if eligible else [0])
    zero = torch.load(report["points"][0]["checkpoint"]["path"], weights_only=True)
    assert all(torch.equal(zero[name], value) for name, value in original.items())
    assert report["protocol"]["critic_warmup"]["gate"]["eligible"] is eligible
    if eligible:
        assert events.count("collect") == events.count("update") == 1
        one = torch.load(report["points"][1]["checkpoint"]["path"], weights_only=True)
        assert not torch.equal(one["weight"], zero["weight"])
        assert report["protocol"]["quarantine_stop"] is None
    else:
        assert "collect" not in events and "update" not in events
        assert report["protocol"]["quarantine_stop"]["stopped_before_actor_updates"]
        assert report["checkpoint_selection"]["selected_update"] == 0
    assert json.loads((config.output_dir / "critic_warmup.json").read_text())["gate"]["eligible"] is eligible


def test_learning_curve_cli_exposes_opt_in_warmup_and_retention_flags():
    root = Path(__file__).resolve().parents[2]
    parser = runpy.run_path(str(root / "scripts" / "run_v4_learning_curve.py"))["build_parser"]()
    defaults = parser.parse_args([])
    assert defaults.critic_warmup_epochs == 0
    chosen = parser.parse_args([
        "--training-source", "anchored-match-start", "--bc-coef", ".05",
        "--critic-warmup-epochs", "8", "--critic-warmup-decisions", "1440",
        "--critic-warmup-lanes", "4", "--critic-warmup-minimum-episodes", "6",
        "--critic-warmup-minimum-ev", ".2",
    ])
    assert chosen.training_source == "anchored-match-start" and chosen.bc_coef == .05
    assert chosen.critic_warmup_epochs == 8 and chosen.critic_warmup_minimum_episodes == 6
    assert chosen.critic_warmup_minimum_ev == .2

