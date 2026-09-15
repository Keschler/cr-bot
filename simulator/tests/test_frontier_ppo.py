"""Tests for frontier-PPO collection and controlled evaluation."""

import os

import pytest

from simulator.rl.frontier_ppo import (
    FRONTIER_PPO_EPISODE_CAP,
    build_comparison_summary,
    classify_paired_sign,
    full_match_record,
    paired_delta_stats,
    pinned_opponent_decks,
    sealed_choice_regret,
    summarize_choice_regret,
)

CKPT = "outputs/v4/stage2_12k_v2_e12_s12.pt"
TRAIN_PT = "outputs/v4/frontier/frontier_train.pt"
SEALED_PT = "outputs/v4/frontier/frontier_sealed.pt"


def test_pinned_opponent_decks_stable():
    first = pinned_opponent_decks(pool_seed=0, episode_index=500)
    second = pinned_opponent_decks(pool_seed=0, episode_index=500)
    assert len(first) == 6
    assert first == second
    assert {s["archetype"] for s in first} == {
        "deterministic-cycle", "aggressive-pressure", "defensive-cycle",
        "beatdown", "air-beatdown", "siege-bait",
    }


@pytest.mark.skipif(not os.path.exists(CKPT), reason="champion checkpoint missing")
def test_full_match_record_deterministic_micro():
    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    specs = pinned_opponent_decks()[:1]
    one = full_match_record(policy=policy, opponent_specs=specs, seeds=[7],
                            device="cpu", deterministic=True, max_decisions=8)
    two = full_match_record(policy=policy, opponent_specs=specs, seeds=[7],
                            device="cpu", deterministic=True, max_decisions=8)
    assert one == two
    assert one[0]["decisions"] == 8 and one[0]["result"] in ("win", "loss", "draw")


@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(TRAIN_PT)),
    reason="champion checkpoint or train set missing",
)
def test_collect_frontier_rollout_shapes_and_rotation():
    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.frontier_ppo import collect_frontier_rollout
    from simulator.rl.model_v4 import V4ValueHead
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    critic = V4ValueHead(frontier_model_config().gru_hidden_dim)
    entries = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)[:4]
    batch, info = collect_frontier_rollout(
        policy, critic, None, entries,
        n_decisions=8, n_lanes=2, episode_cap=4, device="cpu", seed=0,
    )
    traj = batch.trajectory
    assert tuple(traj.rewards.shape) == (2, 8)
    assert tuple(traj.actions.mode.shape) == (2, 8)
    assert tuple(batch.recurrent.shape) == (2, 8, 64)
    # episode_cap=4 over 8 decisions forces rotation: truncations present,
    # multiple episodes visited, finite sparse rewards.
    assert bool(traj.truncated.any().item())
    assert info["episodes"] >= 4
    assert torch.isfinite(traj.rewards).all()
    assert set(torch.unique(traj.rewards).tolist()) <= {-1.0, 0.0, 1.0}
    assert info["episodes"] >= 4 and info["mean_episode_len"] > 0
    assert batch.anchor_actions is None


@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(TRAIN_PT)),
    reason="champion checkpoint or train set missing",
)
def test_truncation_reward_bonus_and_flag_off_parity():
    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.frontier_ppo import collect_frontier_rollout
    from simulator.rl.model_v4 import V4ValueHead
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    critic = V4ValueHead(frontier_model_config().gru_hidden_dim)
    entries = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)[:4]
    kwargs = dict(n_decisions=8, n_lanes=2, episode_cap=4, device="cpu", seed=0)
    torch.manual_seed(0)
    batch_on, info_on = collect_frontier_rollout(
        policy, critic, None, entries, truncation_reward=True, **kwargs)
    assert info_on["truncation_reward"] is True
    assert len(info_on["truncation_bonuses"]) > 0
    assert all(isinstance(b, float) for b in info_on["truncation_bonuses"])
    torch.manual_seed(0)
    batch_off, info_off = collect_frontier_rollout(
        policy, critic, None, entries, truncation_reward=False, **kwargs)
    assert info_off["truncation_bonuses"] == []
    assert set(torch.unique(batch_off.trajectory.rewards).tolist()) <= {-1.0, 0.0, 1.0}
    # Flag-off is the exp1 codepath: identical states/actions/terminations.
    assert torch.equal(batch_on.trajectory.actions.mode, batch_off.trajectory.actions.mode)
    assert torch.equal(batch_on.trajectory.terminated, batch_off.trajectory.terminated)
    assert torch.equal(batch_on.trajectory.truncated, batch_off.trajectory.truncated)
@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(TRAIN_PT)),
    reason="champion checkpoint or train set missing",
)
def test_step_tower_reward_telescopes_to_lump():
    """Per-step tower deltas sum to the truncation lump (same totals, new timing)."""

    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.frontier_ppo import collect_frontier_rollout
    from simulator.rl.model_v4 import V4ValueHead
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    critic = V4ValueHead(frontier_model_config().gru_hidden_dim)
    entries = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)[:4]
    kwargs = dict(n_decisions=8, n_lanes=2, episode_cap=4, device="cpu", seed=0)
    torch.manual_seed(0)
    batch_lump, _ = collect_frontier_rollout(
        policy, critic, None, entries, truncation_reward=True, **kwargs)
    torch.manual_seed(0)
    batch_step, info_step = collect_frontier_rollout(
        policy, critic, None, entries, step_tower_reward=True, **kwargs)
    assert info_step["step_tower_reward"] is True
    # RNG parity: identical trajectories, boundaries, and actions.
    assert torch.equal(batch_step.trajectory.actions.mode, batch_lump.trajectory.actions.mode)
    assert torch.equal(batch_step.trajectory.terminated, batch_lump.trajectory.terminated)
    assert torch.equal(batch_step.trajectory.truncated, batch_lump.trajectory.truncated)
    # Truncated episodes: per-step deltas telescope exactly to the lump.
    for lane in range(2):
        start = 0
        for t in range(8):
            boundary = bool(batch_lump.trajectory.terminated[lane, t]) or bool(
                batch_lump.trajectory.truncated[lane, t])
            if not boundary and t < 7:
                continue
            if bool(batch_lump.trajectory.truncated[lane, t]):
                lump_total = float(batch_lump.trajectory.rewards[lane, start : t + 1].sum())
                step_total = float(batch_step.trajectory.rewards[lane, start : t + 1].sum())
                assert lump_total == pytest.approx(step_total, abs=1e-4)
            start = t + 1


@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(SEALED_PT)),
    reason="champion checkpoint or sealed set missing",
)
def test_sealed_choice_regret_self_consistent_micro():
    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    entries = torch.load(SEALED_PT, map_location="cpu", weights_only=False)[:1]
    out = sealed_choice_regret(
        policy=policy, champion=policy, entries=entries,
        n_replicas=1, horizon=8, device="cpu",
    )
    assert len(out) == 1
    assert out[0]["same_action"] is True
    assert out[0]["delta"] == pytest.approx(0.0)


@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(SEALED_PT)),
    reason="champion checkpoint or sealed set missing",
)
def test_sealed_choice_reports_per_replica_values():
    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    entries = torch.load(SEALED_PT, map_location="cpu", weights_only=False)[:1]
    out = sealed_choice_regret(
        policy=policy, champion=policy, entries=entries,
        n_replicas=2, horizon=8, device="cpu",
    )
    assert len(out[0]["champ_values"]) == 2
    assert len(out[0]["policy_values"]) == 2
    assert out[0]["champ_mean"] == pytest.approx(sum(out[0]["champ_values"]) / 2)
    assert out[0]["policy_mean"] == pytest.approx(sum(out[0]["policy_values"]) / 2)
    assert out[0]["delta"] == pytest.approx(
        out[0]["policy_mean"] - out[0]["champ_mean"], abs=1e-4)


def test_paired_delta_stats_known_values():
    import math
    import statistics

    stats = paired_delta_stats([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(statistics.stdev([1.0, 2.0, 3.0]))
    assert stats["se"] == pytest.approx(statistics.stdev([1.0, 2.0, 3.0]) / math.sqrt(3))


def test_paired_delta_stats_degenerate_cases():
    assert paired_delta_stats([]) == {"mean": 0.0, "std": 0.0, "se": 0.0, "n": 0}
    single = paired_delta_stats([0.5])
    assert single["mean"] == pytest.approx(0.5)
    assert single["std"] == 0.0 and single["se"] == 0.0
    zeros = paired_delta_stats([0.0, 0.0, 0.0, 0.0])
    assert zeros["mean"] == 0.0 and zeros["std"] == 0.0 and zeros["se"] == 0.0


def test_classify_paired_sign_thresholds():
    assert classify_paired_sign(0.0, 0.0, same_action=True) == "same"
    assert classify_paired_sign(1.0, 0.1, same_action=False) == "improved"
    assert classify_paired_sign(-1.0, 0.1, same_action=False) == "regressed"
    # Interval covers zero -> unresolved even with nonzero mean.
    assert classify_paired_sign(0.05, 0.10, same_action=False) == "unresolved"
    # Exact boundary (|mean| == k*se) is not confident.
    assert classify_paired_sign(0.2, 0.1, same_action=False) == "unresolved"


def _synthetic_choice_row(entry_id, delta, sign, same=False):
    return {
        "entry_id": entry_id,
        "delta": delta,
        "paired_mean": delta,
        "paired_se": 0.01 if sign in ("improved", "regressed") else 10.0,
        "same_action": same,
        "sign": sign,
    }


def test_summarize_choice_regret_aggregates():
    rows = [
        _synthetic_choice_row("a", 0.4, "improved"),
        _synthetic_choice_row("b", 0.2, "unresolved"),
        _synthetic_choice_row("c", -0.1, "unresolved"),
        _synthetic_choice_row("d", -0.9, "regressed"),
        _synthetic_choice_row("e", 0.0, "same", same=True),
        _synthetic_choice_row("f", 0.1, "unresolved"),
    ]
    summary = summarize_choice_regret(rows)
    assert summary["n"] == 6
    assert summary["n_same"] == 1
    assert summary["n_changed"] == 5
    assert summary["mean_delta"] == pytest.approx(-0.05)
    assert summary["median_delta"] == pytest.approx(0.05)
    # n=6 trims one entry per tail: drops -0.9 and 0.4.
    assert summary["trimmed_mean"] == pytest.approx(0.05)
    assert summary["n_improved"] == 1
    assert summary["n_regressed"] == 1
    assert summary["n_unresolved"] == 3
    assert summary["n_catastrophic"] == 1
    assert summary["catastrophic_ids"] == ["d"]
    assert summary["top_mover_id"] == "d"
    assert summary["mean_without_top"] == pytest.approx(0.12)


def test_summarize_flags_single_state_dominance():
    rows = [_synthetic_choice_row(f"s{i}", 0.01, "unresolved") for i in range(11)]
    rows.append(_synthetic_choice_row("outlier", -1.65, "regressed"))
    summary = summarize_choice_regret(rows)
    assert summary["mean_delta"] < 0
    assert summary["mean_without_top"] > 0
    assert summary["single_state_dominated"] is True
    assert summary["median_delta"] == pytest.approx(0.01)
    # Median/trimmed resist the outlier while the mean does not.
    assert abs(summary["median_delta"] - summary["mean_delta"]) > 0.1


def test_build_comparison_summary_eval_only_wiring():
    pre = {
        "choice_mean_delta": 0.0, "choice_changed": 0,
        "choice_summary": {"n": 1, "mean_delta": 0.0},
        "train_choice_mean_delta": 0.0, "train_choice_changed": 0,
        "train_choice_summary": {"n": 2, "mean_delta": 0.0},
        "rollout_mean_return": 0.08, "match_wld": [2, 22, 0],
    }
    post = {
        "choice_mean_delta": -0.02, "choice_changed": 3,
        "choice_summary": {"n": 1, "mean_delta": -0.02},
        "train_choice_mean_delta": 0.01, "train_choice_changed": 4,
        "train_choice_summary": {"n": 2, "mean_delta": 0.01},
        "rollout_mean_return": 0.06, "match_wld": [5, 19, 0],
    }
    summary = build_comparison_summary(
        pre, post, {"accepted": True}, {"accepted": False},
        eval_only_checkpoint="outputs/v4/some.pt",
    )
    # Reference pre is preserved verbatim; fresh post is compared against it.
    assert summary["choice_delta_pre"] == 0.0
    assert summary["choice_delta_post"] == -0.02
    assert summary["match_wld_pre"] == [2, 22, 0]
    assert summary["match_wld_post"] == [5, 19, 0]
    assert summary["gates_green_pre"] is True
    assert summary["gates_green_post"] is False
    assert summary["eval_only_checkpoint"] == "outputs/v4/some.pt"
    assert summary["choice_summary_post"] == {"n": 1, "mean_delta": -0.02}
    assert summary["train_choice_summary_pre"] == {"n": 2, "mean_delta": 0.0}


def test_frontier_ppo_script_eval_wiring_flags():
    import subprocess

    proc = subprocess.run(
        ["outputs/venv/bin/python", "scripts/run_v4_frontier_ppo.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    for flag in ("--choice-replicas", "--adaptive-choice-replicas",
                 "--choice-max-replicas", "--choice-se-k",
                 "--eval-only-checkpoint", "--reference-report"):
        assert flag in proc.stdout, flag


@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(SEALED_PT)),
    reason="champion checkpoint or sealed set missing",
)
def test_sealed_choice_reports_paired_stats():
    import torch

    from simulator.rl.frontier import frontier_model_config
    from simulator.rl.v4_ppo import load_frozen_anchor

    policy = load_frozen_anchor(CKPT, frontier_model_config(), device="cpu")
    entries = torch.load(SEALED_PT, map_location="cpu", weights_only=False)[:1]
    out = sealed_choice_regret(
        policy=policy, champion=policy, entries=entries,
        n_replicas=4, horizon=8, device="cpu",
    )
    row = out[0]
    assert row["n_replicas"] == 4
    assert row["replicas_extended"] is False
    assert len(row["paired_deltas"]) == 4
    assert row["paired_mean"] == pytest.approx(row["delta"])
    expected = [p - c for p, c in zip(row["policy_values"], row["champ_values"])]
    assert row["paired_deltas"] == pytest.approx(expected, abs=1e-4)
    stats = paired_delta_stats(expected)
    assert row["paired_std"] == pytest.approx(stats["std"], abs=1e-4)
    assert row["paired_se"] == pytest.approx(stats["se"], abs=1e-4)
    assert row["sign"] == "same"


def test_sealed_choice_adaptive_replica_handling(monkeypatch):
    import simulator.rl.frontier_ppo as fppo

    calls: list[str] = []
    pattern = [0.6, -0.5] * 16  # small edge buried in alternating noise
    state = {"n": 0}

    def fake_run_branch(*, env, hidden, branch, opponent_spec,
                        replica_seed, horizon, policy, device=None):
        calls.append(branch.source)
        if branch.source == "c":
            return {"value": 0.0}
        value = pattern[state["n"]]
        state["n"] += 1
        return {"value": value}

    monkeypatch.setattr(fppo, "run_branch", fake_run_branch)
    monkeypatch.setattr(fppo, "make_frontier_env", lambda: object())
    monkeypatch.setattr(fppo, "load_frontier_state", lambda env, state: None)

    import torch

    hidden = torch.zeros(4)

    class _Branch:
        def __init__(self, mode=1, slot=0, row=0, col=0):
            self.mode, self.slot, self.row, self.col = mode, slot, row, col

    # Policy and champion disagree so the state counts as changed; the
    # alternating fake values keep the paired sign unresolved.
    champ_branch = _Branch(mode=1, slot=0, row=1, col=1)
    pol_branch = _Branch(mode=1, slot=0, row=2, col=2)
    seq = [champ_branch, pol_branch]
    monkeypatch.setattr(fppo, "_policy_argmax_at_entry",
                        lambda policy, entry, device: seq.pop(0))
    entry = {"entry_id": "volatile#0", "initial_state": {},
             "hidden0": hidden, "opponent": {}}
    out = fppo.sealed_choice_regret(
        policy=object(), champion=object(), entries=[entry],
        n_replicas=2, horizon=4, device="cpu",
        adaptive_replicas=True, max_replicas=8, se_k=2.0,
    )
    row = out[0]
    assert row["same_action"] is False
    assert row["n_replicas"] == 8
    assert row["replicas_extended"] is True
    assert len(row["paired_deltas"]) == 8
    assert row["paired_deltas"] == pytest.approx(pattern[:8])
    # 2 champ + 2 policy calls per doubling stage: 2 -> 4 -> 8 replicas.
    assert calls.count("c") == 8 and calls.count("p") == 8
    assert row["sign"] == "unresolved"


def test_sealed_choice_replica_count_validation():
    import torch

    entry = {"entry_id": "x", "initial_state": {},
             "hidden0": torch.zeros(4), "opponent": {}}
    with pytest.raises(ValueError):
        sealed_choice_regret(policy=object(), champion=object(),
                             entries=[entry], n_replicas=0)
    with pytest.raises(ValueError):
        sealed_choice_regret(policy=object(), champion=object(),
                             entries=[entry], n_replicas=8, max_replicas=4)
