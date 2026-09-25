"""Small simulator sequences reproducing V4 training/runtime wiring errors."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from simulator.rl.v4.frontier import frontier_model_config
from simulator.rl.v4.frontier_ppo import (
    action_logit_margins, collect_frontier_rollout, full_match_record,
    make_frontier_env, pinned_opponent_decks,
)
from simulator.rl.v4.model_v4 import RecurrentV4Policy, V4ActionBatch, V4ValueHead
from simulator.rl.foundation.objectives import compute_gae
from simulator.rl.v4.v4_ppo import (
    collect_v4_rollout, configure_v4_runtime, decode_v4_sim_actions,
    return_distribution, rollout_next_values, update_v4_ppo,
)
from simulator.roster import PLAYER_DECK
from simulator.tests.test_v4_ppo_adapter import _inputs, _lane, _masks


def policy():
    torch.manual_seed(7)
    return RecurrentV4Policy(frontier_model_config())


def force_mode(net, mode):
    with torch.no_grad():
        net.heads.mode_head.weight.zero_()
        net.heads.mode_head.bias[:] = torch.tensor([80., -80.] if mode == 0 else [-80., 80.])


def test_fixed_wait_has_no_duration_probability_entropy_or_gradient():
    net = policy()
    configure_v4_runtime(net)
    data = _inputs(batch=2)
    masks = _masks(batch=2, wait_only_rows=(0, 1))
    logits, _, _ = net(**data)
    altered = replace(logits, wait_duration=logits.wait_duration * -500 + 400)
    torch.manual_seed(2)
    first, lp, entropy = net.sample_action(logits, masks)
    torch.manual_seed(2)
    second, other_lp, other_entropy = net.sample_action(altered, masks)
    assert torch.equal(first.wait_duration, torch.zeros_like(first.wait_duration))
    assert torch.equal(first.mode, second.mode)
    assert torch.equal(lp, other_lp)
    assert torch.count_nonzero(entropy["duration"]) == 0
    assert torch.equal(entropy["joint"], other_entropy["joint"])
    (-lp.mean() - entropy["joint"].mean()).backward()
    assert net.heads.duration_head.weight.grad is None


def test_decoder_rejects_unsupported_duration_instead_of_silently_ignoring_it():
    action = V4ActionBatch(torch.tensor([0]), torch.tensor([0]), torch.zeros(1, 2, dtype=torch.long), torch.tensor([3]))
    with pytest.raises(ValueError, match="duration index 0"):
        decode_v4_sim_actions(action)


def test_internal_cut_uses_successor_not_unrelated_reset_value():
    values = torch.tensor([[1., 2., 1000., 4.]])
    terminated = torch.tensor([[False, False, False, False]])
    truncated = torch.tensor([[False, True, False, False]])
    saved = torch.tensor([[float('nan'), 3., float('nan'), 5.]])
    nxt = rollout_next_values(values, terminated, truncated, saved)
    assert nxt.tolist() == [[2., 3., 4., 5.]]
    advantages, _ = compute_gae(torch.zeros_like(values), values, nxt, terminated, truncated, gamma=.99, gae_lambda=.95)
    assert advantages[0, 1].item() == pytest.approx(.99 * 3 - 2)
    # Terminal transitions never bootstrap, even if a stale successor was provided.
    terminated[0, 1] = True
    assert rollout_next_values(values, terminated, truncated, saved)[0, 1] == 0
    with pytest.raises(ValueError, match="missing post-step"):
        rollout_next_values(values, terminated, truncated, torch.full_like(saved, float('nan')))


def test_anchor_targets_come_from_anchor_weights():
    learner, anchor = policy(), policy()
    force_mode(learner, 1)
    force_mode(anchor, 0)
    critic = V4ValueHead(64)
    lane = _lane("ground-defense", "beatdown", 5)
    batch, _ = collect_v4_rollout(learner, critic, anchor, [lane], n_decisions=3, deterministic_policy=True)
    assert batch.trajectory.actions.mode[0, 0].item() == 1
    assert torch.count_nonzero(batch.anchor_actions.mode) == 0
    assert batch.next_values is not None
    assert torch.isfinite(batch.next_values).all()


def entries(count=3):
    spec = pinned_opponent_decks()[0]
    result = []
    for i in range(count):
        env = make_frontier_env()
        env.reset_v2(seed=100 + i, decks=(tuple(PLAYER_DECK), tuple(spec['deck'])), shuffle_decks=True)
        result.append(dict(entry_id=f"train:{i}", split='train', initial_state=env.save_state(),
                           hidden0=torch.zeros(1, 1, 64), opponent=spec,
                           learner_deck=PLAYER_DECK, opponent_deck=spec['deck']))
    return result


def test_frontier_rotates_entries_and_values_before_reset(monkeypatch):
    import simulator.rl.v4.frontier_ppo.collection as frontier
    import simulator.rl.v4.v4_ppo as runtime
    data = entries()
    seen, ticks = [], []
    original = frontier.load_frontier_state
    def load(env, state):
        seen.append(state)
        original(env, state)
    def successor(net, critic, env, estimator, hidden, device):
        ticks.append(env.state.tick)
        return 123.
    monkeypatch.setattr(frontier, 'load_frontier_state', load)
    monkeypatch.setattr(runtime, 'post_step_value', successor)
    net = policy()
    force_mode(net, 0)
    batch, info = collect_frontier_rollout(net, V4ValueHead(64), None, data,
        n_lanes=1, n_decisions=3, episode_cap=1, device='cpu', deterministic_policy=True)
    assert [id(x) for x in seen] == [id(x['initial_state']) for x in data]
    assert all(t > 0 for t in ticks)
    assert batch.next_values.tolist() == [[123., 123., 123.]]
    assert info['episodes'] == 3


def test_full_match_deterministic_keeps_memory_and_reports_censoring(monkeypatch):
    net = policy()
    force_mode(net, 0)
    original = net.forward_step
    history = []
    def forward(inputs, *, hidden=None, reset_mask=None):
        history.append(hidden.detach().clone())
        return original(inputs, hidden=hidden, reset_mask=reset_mask)
    monkeypatch.setattr(net, 'forward_step', forward)
    result = full_match_record(policy=net, opponent_specs=pinned_opponent_decks()[:1], seeds=[71],
        max_decisions=3, device='cpu', deterministic=True)
    assert len(history) == 3
    assert torch.count_nonzero(history[0]) == 0
    assert torch.count_nonzero(history[1]) > 0
    assert result[0]['result'] == 'truncated'
    assert not result[0]['terminated']


def test_placement_bias_is_registered_before_first_forward():
    net = policy()
    optimizer = torch.optim.Adam(net.parameters())
    assert any(p is net.heads.cell_bias for group in optimizer.param_groups for p in group['params'])


def test_successor_value_uses_next_observation_without_consuming_rng():
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.rl.v4.v4_ppo import post_step_value
    from simulator.rl.v4.frontier_eval import _observe_inputs, _step_inputs
    from simulator.actions import WaitAction
    net = policy()
    configure_v4_runtime(net)
    net.eval()
    critic = V4ValueHead(64)
    game = make_frontier_env()
    game.reset_v2(seed=44, decks=(tuple(PLAYER_DECK), tuple(PLAYER_DECK)), shuffle_decks=False)
    inputs, _, _, _ = _observe_inputs(game, 'cpu')
    with torch.no_grad():
        _, current, hidden = net.forward_step(_step_inputs(inputs, 'cpu'))
    game.step_v2((WaitAction(0), WaitAction(1)))
    future, _, _, _ = _observe_inputs(game, 'cpu')
    with torch.no_grad():
        _, expected_features, _ = net.forward_step(_step_inputs(future, 'cpu'), hidden=hidden)
        expected = critic(expected_features).item()
    before_rng = torch.get_rng_state().clone()
    actual = post_step_value(net, critic, game, PublicStateEstimator(), hidden, 'cpu')
    assert torch.equal(before_rng, torch.get_rng_state())
    assert actual == pytest.approx(expected)
    assert not torch.equal(current, expected_features)


def test_entropy_quarantine_uses_post_update_logits(monkeypatch):
    """A collapsing post-update policy must trip the entropy quarantine."""

    net = policy()
    critic = V4ValueHead(64)
    batch, _ = collect_v4_rollout(
        net, critic, None, [_lane("ground-defense", "beatdown", 5)],
        n_decisions=2, deterministic_policy=True,
    )
    # Make the comparison baseline deterministic and non-zero.  The wrapper
    # below returns zero entropy only for a fresh post-step logits object; the
    # stale pre-step object returns one, reproducing the old false negative.
    batch.per_head_entropy = {
        key: torch.ones_like(value)
        for key, value in batch.per_head_entropy.items()
    }
    original = net.action_entropy
    first_logits = None
    seen = []

    def collapse_post(logits, masks):
        nonlocal first_logits
        seen.append(logits)
        if first_logits is None:
            first_logits = logits
            return original(logits, masks)
        base = original(logits, masks)
        if logits is first_logits:
            return {key: torch.ones_like(value) for key, value in base.items()}
        return {key: torch.zeros_like(value) for key, value in base.items()}

    monkeypatch.setattr(net, "action_entropy", collapse_post)
    report = update_v4_ppo(
        net, critic, torch.optim.Adam(net.parameters(), lr=1e-5),
        torch.optim.Adam(critic.parameters(), lr=1e-4), batch,
    )
    assert len(seen) == 2
    assert seen[1] is not seen[0]
    assert report.per_head_entropy["joint_collapse"] == pytest.approx(1.0)
    assert report.quarantined
    assert any("joint entropy collapse" in reason for reason in report.quarantine_reasons)


def test_wait_only_margin_report_has_explicit_empty_placement_summary():
    net = policy()
    force_mode(net, 0)
    report = action_logit_margins(net, entries(1), device="cpu")
    assert report["n_play"] == 0
    assert report["n_wait"] == 1
    assert report["placement"] == {
        "n": 0,
        "mean": None,
        "p10": None,
        "median": None,
        "p90": None,
        "frac_close_lt_010": None,
        "frac_close_lt_025": None,
    }


def test_return_distribution_quantile_levels_follow_input_device(monkeypatch):
    seen = {}
    original_quantile = torch.quantile

    def record_quantile(values, q, *args, **kwargs):
        seen["values"] = values.device
        seen["q"] = q.device
        return original_quantile(values, q, *args, **kwargs)

    # The CPU assertion covers the call contract on every runner; the CUDA
    # variant below exercises the device mismatch that motivated this check.
    monkeypatch.setattr(torch, "quantile", record_quantile)
    return_distribution(torch.arange(8, dtype=torch.float32))
    assert seen["q"] == seen["values"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_return_distribution_runs_on_cuda():
    result = return_distribution(torch.arange(8, dtype=torch.float32, device="cuda"))
    assert result["n"] == 8


@pytest.mark.parametrize("epochs", [1, 2])
def test_movement_quarantine_measures_updated_candidate(epochs):
    from simulator.rl.v4.v4_ppo import _forward_sequence
    from simulator.rl.foundation.objectives import PPOObjectiveConfig

    net = policy()
    critic = V4ValueHead(64)
    batch, _ = collect_v4_rollout(
        net, critic, None, [_lane("ground-defense", "beatdown", 5)],
        n_decisions=4, deterministic_policy=True,
    )
    config = PPOObjectiveConfig(clip_epsilon=0.1, bc_coef=0.0)
    report = update_v4_ppo(
        net, critic, torch.optim.Adam(net.parameters(), lr=0.01),
        torch.optim.Adam(critic.parameters(), lr=1e-4), batch,
        ppo_config=config, ppo_epochs=epochs,
    )
    trajectory = batch.trajectory
    with torch.no_grad():
        logits, _, _ = _forward_sequence(net, trajectory.sequence, "cpu")
        log_probs = net.log_prob(logits, trajectory.action_masks, trajectory.actions)
        difference = log_probs - trajectory.old_log_probs
    assert report.approx_kl == pytest.approx((-difference).mean().item())
    assert report.mean_abs_log_ratio == pytest.approx(difference.abs().mean().item())
    assert report.clip_fraction == pytest.approx(
        ((difference.exp() - 1).abs() > config.clip_epsilon).float().mean().item()
    )
    assert report.mean_abs_log_ratio > 0.05
    assert report.quarantined
    assert any("mean-abs-log-ratio" in reason for reason in report.quarantine_reasons)


def test_monte_carlo_terminal_reward_reaches_prior_actions_without_crossing_resets():
    from simulator.rl.v4.v4_ppo import monte_carlo_returns

    rewards = torch.tensor([[0., 2., 0., 4.]])
    terminated = torch.tensor([[False, True, False, True]])
    returns, stats = monte_carlo_returns(rewards, terminated, gamma=0.5)
    assert returns.tolist() == [[1., 2., 2., 4.]]
    assert stats["truncated_tail_fraction"] == 0.0
