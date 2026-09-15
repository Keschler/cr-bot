"""Frontier PPO: train on frontier entries + controlled pre/post evaluation.

Training episodes start from materialized frontier entries (naturally
reached, demonstrably mistaken states) on bare simulator envs with the sparse
terminal-outcome reward.  Entry loads use zero-restart recurrent semantics
(``reset=True``), identical to the PPO smoke setup, so the update unroll
from ``initial_hidden=zeros`` stays exact.

Evaluation (identical protocol pre/post, paired seeds throughout):

* sealed choice-regret: each sealed entry's frozen-sim action vs the
  policy-under-test's action, both scored by frozen-champion continuation
  with matched replica seeds;
* sealed rollout return: stochastic rollouts from sealed entries with common
  random numbers across the two policies;
* full-match W/L/D: fresh deterministic matches vs the pinned 6-archetype
  pool, paired by (archetype, seed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    from .frontier import _stable_seed
    from .frontier_eval import (
        BranchSpec,
        load_frontier_state,
        make_opponent,
        run_branch,
        tower_fracs,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.rl.frontier import _stable_seed
    from simulator.rl.frontier_eval import (
        BranchSpec,
        load_frontier_state,
        make_opponent,
        run_branch,
        tower_fracs,
    )


FRONTIER_PPO_VERSION: str = "frontier-ppo-v4-0"
FRONTIER_PPO_EPISODE_CAP: int = 128

# Paired-uncertainty reporting for sealed choice-regret.
# Deltas are policy-minus-champion continuation values under common random
# numbers (matched replica seeds), so per-replica paired differences are the
# correct unit for standard errors -- not the unpaired group means.
PAIRED_SIGN_K: float = 2.0
ADAPTIVE_REPLICA_CAP: int = 32
CATASTROPHIC_DELTA: float = -0.5
TRIM_FRACTION: float = 0.1


def paired_delta_stats(diffs: Sequence[float]) -> dict[str, float]:
    """Mean / sample-std / standard-error of paired replica deltas.

    Uses the sample standard deviation (ddof=1); ``n < 2`` yields std=se=0.
    """
    import math
    import statistics

    vals = [float(v) for v in diffs]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "se": 0.0, "n": 0}
    mean = sum(vals) / n
    if n < 2:
        return {"mean": mean, "std": 0.0, "se": 0.0, "n": n}
    std = statistics.stdev(vals)
    return {"mean": mean, "std": std, "se": std / math.sqrt(n), "n": n}


def classify_paired_sign(
    mean: float, se: float, *, same_action: bool, k: float = PAIRED_SIGN_K
) -> str:
    """Confidence label for one state's paired delta.

    ``same``: identical first action (paired deltas identically zero).
    Otherwise the ``k``-SE interval around the mean decides: fully above
    zero is ``improved``, fully below is ``regressed``, anything covering
    zero is ``unresolved``.
    """
    if same_action:
        return "same"
    if mean - k * se > 0:
        return "improved"
    if mean + k * se < 0:
        return "regressed"
    return "unresolved"


def _sign_of_entry(entry: dict[str, Any], k: float = PAIRED_SIGN_K) -> str:
    """Read a stored entry's sign, recomputing from paired stats if needed."""
    sign = entry.get("sign")
    if sign in ("same", "improved", "regressed", "unresolved"):
        return sign
    diffs = entry.get("paired_deltas")
    if diffs is not None:
        stats = paired_delta_stats([float(v) for v in diffs])
        return classify_paired_sign(
            stats["mean"], stats["se"],
            same_action=bool(entry.get("same_action")), k=k,
        )
    if entry.get("same_action"):
        return "same"
    return "unresolved"


def summarize_choice_regret(
    choice: Sequence[dict[str, Any]],
    *,
    se_k: float = PAIRED_SIGN_K,
    trim_fraction: float = TRIM_FRACTION,
) -> dict[str, Any]:
    """Aggregate sealed statistics that survive single-state dominance.

    Reports mean delta with its across-entry SE, median, trimmed mean
    (``trim_fraction`` cut from each tail; at least one entry per tail when
    ``n >= 6``), confident improved/regressed/unresolved counts from the
    per-entry paired intervals, catastrophic regressions (confident +
    ``paired_mean <= CATASTROPHIC_DELTA``), and a leave-top-mover-out
    dominance check so aggregate means dominated by one volatile state are
    flagged rather than claimed as improvement.
    """
    import math
    import statistics

    rows = list(choice)
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "n_same": 0, "n_changed": 0,
            "mean_delta": 0.0, "mean_se": 0.0,
            "median_delta": 0.0, "trimmed_mean": 0.0,
            "n_improved": 0, "n_regressed": 0, "n_unresolved": 0,
            "n_catastrophic": 0, "catastrophic_ids": [],
            "regressed_ids": [], "improved_ids": [],
            "top_mover_id": None, "top_mover_delta": 0.0,
            "mean_without_top": 0.0, "single_state_dominated": False,
        }
    deltas = [float(r["delta"]) for r in rows]
    mean = sum(deltas) / n
    mean_se = (statistics.stdev(deltas) / math.sqrt(n)) if n >= 2 else 0.0
    median = statistics.median(deltas)
    cut = int(n * trim_fraction)
    if n >= 6:
        cut = max(1, cut)
    else:
        cut = 0
    trimmed = sorted(deltas)[cut : n - cut] if cut else sorted(deltas)
    trimmed_mean = sum(trimmed) / len(trimmed)
    improved_ids, regressed_ids, unresolved_ids = [], [], []
    catastrophic_ids = []
    for r in rows:
        sign = _sign_of_entry(r, k=se_k)
        if sign == "improved":
            improved_ids.append(r["entry_id"])
        elif sign == "regressed":
            regressed_ids.append(r["entry_id"])
            paired_mean = r.get("paired_mean", r["delta"])
            if float(paired_mean) <= CATASTROPHIC_DELTA:
                catastrophic_ids.append(r["entry_id"])
        elif sign == "unresolved":
            unresolved_ids.append(r["entry_id"])
    n_same = sum(1 for r in rows if r.get("same_action"))
    top_idx = max(range(n), key=lambda i: abs(deltas[i]))
    top_id = rows[top_idx]["entry_id"]
    top_delta = deltas[top_idx]
    rest = deltas[:top_idx] + deltas[top_idx + 1 :]
    mean_without_top = sum(rest) / len(rest) if rest else 0.0
    dominated = (
        (mean > 0) != (mean_without_top > 0)
        or abs(mean - mean_without_top) > mean_se
    ) if n >= 2 else False
    return {
        "n": n,
        "n_same": n_same,
        "n_changed": n - n_same,
        "mean_delta": round(mean, 5),
        "mean_se": round(mean_se, 5),
        "median_delta": round(median, 5),
        "trimmed_mean": round(trimmed_mean, 5),
        "n_improved": len(improved_ids),
        "n_regressed": len(regressed_ids),
        "n_unresolved": len(unresolved_ids),
        "n_catastrophic": len(catastrophic_ids),
        "catastrophic_ids": catastrophic_ids,
        "regressed_ids": regressed_ids,
        "improved_ids": improved_ids,
        "top_mover_id": top_id,
        "top_mover_delta": round(top_delta, 5),
        "mean_without_top": round(mean_without_top, 5),
        "single_state_dominated": bool(dominated),
    }


def build_comparison_summary(
    pre: dict[str, Any],
    post: dict[str, Any],
    gates_before: dict[str, Any],
    gates_after: dict[str, Any],
    eval_only_checkpoint: Any = None,
) -> dict[str, Any]:
    """Shared pre/post summary for training and eval-only report paths.

    Pure function (no I/O, no models) so eval-only checkpoint wiring --
    reference ``pre`` preserved, fresh ``post`` compared against it -- is
    unit-testable without running the simulator.
    """
    return {
        "choice_delta_pre": pre["choice_mean_delta"],
        "choice_delta_post": post["choice_mean_delta"],
        "train_choice_delta_pre": pre.get("train_choice_mean_delta"),
        "train_choice_delta_post": post.get("train_choice_mean_delta"),
        "rollout_return_pre": pre["rollout_mean_return"],
        "rollout_return_post": post["rollout_mean_return"],
        "match_wld_pre": pre["match_wld"],
        "match_wld_post": post["match_wld"],
        "gates_green_pre": bool(gates_before.get("accepted")),
        "gates_green_post": bool(gates_after.get("accepted")),
        "choice_summary_pre": pre.get("choice_summary"),
        "choice_summary_post": post.get("choice_summary"),
        "train_choice_summary_pre": pre.get("train_choice_summary"),
        "train_choice_summary_post": post.get("train_choice_summary"),
        "eval_only_checkpoint": eval_only_checkpoint,
    }


def make_frontier_env() -> Any:
    from simulator.engine.core import BattleEngine
    from simulator.env import RewardConfig, SimulatorEnv
    from simulator.ruleset import load_fixed_ruleset

    return SimulatorEnv(
        BattleEngine(load_fixed_ruleset(), validate_every_tick=False),
        reward=RewardConfig.terminal_outcome(),
    )


def collect_frontier_rollout(
    policy: Any,
    critic: Any,
    anchor: Any,
    entries: Sequence[dict[str, Any]],
    *,
    n_decisions: int,
    n_lanes: int = 8,
    episode_cap: int = FRONTIER_PPO_EPISODE_CAP,
    device: Any = None,
    deterministic_policy: bool = False,
    seed: int = 0,
    truncation_reward: bool = False,
    step_tower_reward: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Fixed-length rollout across frontier-entry episodes (JSON-free lanes).

    Reward modes (exactly one should hold; both off reproduces exp1):

    * neither flag (exp1): every non-terminal step pays 0; terminal pays
      the env sparse outcome only.
    * ``truncation_reward`` (exp2/exp3): as above, plus an episode that
      reaches ``episode_cap`` without terminating pays, on its final step,
      the tower-differential improvement over the episode.
    * ``step_tower_reward`` (exp4): the same tower-differential objective
      densified per step — each decision pays the differential change
      since the previous decision (potential shaping with Phi = tower
      differential; episode totals telescope to the truncation-lump
      equivalent).  No truncation lump is added in this mode.  Terminal
      episodes keep the env sparse outcome plus the final delta.

    No other shaping exists in any mode.
    """

    import torch

    try:
        from .model_v4 import V4ActionBatch, masks_from_legal_play
        from .v4_ppo import (
            _stack_rollout,
            _v4_step_inputs,
            decode_v4_sim_actions,
        )
    except ImportError:  # pragma: no cover
        from simulator.rl.model_v4 import V4ActionBatch, masks_from_legal_play
        from simulator.rl.v4_ppo import (
            _stack_rollout,
            _v4_step_inputs,
            decode_v4_sim_actions,
        )
    from simulator.public_state_estimator import PublicStateEstimator

    if not entries:
        raise ValueError("collect_frontier_rollout requires at least one entry")
    if n_lanes < 1 or type(n_decisions) is not int or n_decisions < 1:
        raise ValueError("n_lanes >= 1 and integer n_decisions >= 1 required")

    policy.eval()
    critic.eval()
    if anchor is not None:
        anchor.eval()
    envs = [make_frontier_env() for _ in range(n_lanes)]
    hidden = policy.initial_hidden(n_lanes, device=device)
    cursor = int(seed) % len(entries)
    visits = [0] * n_lanes
    lane_entry = [None] * n_lanes
    lane_steps = [0] * n_lanes
    lane_opponent = [None] * n_lanes
    estimators = [PublicStateEstimator() for _ in range(n_lanes)]
    needs_entry = [True] * n_lanes
    lane_towers_start = [(0.0, 0.0)] * n_lanes  # (enemy_frac, own_frac) at episode start
    lane_towers_prev = [(0.0, 0.0)] * n_lanes  # running differential anchor (exp4)
    truncation_bonuses: list[float] = []

    steps: list[dict[str, torch.Tensor]] = []
    wins = losses = draws = 0
    wait_count = 0
    card_counts: dict[int, int] = {}
    illegal_attempts = 0
    episodes = 0
    episode_lens: list[int] = []
    with torch.no_grad():
        for _ in range(n_decisions):
            for i in range(n_lanes):
                if needs_entry[i]:
                    entry = entries[(cursor + i) % len(entries)]
                    load_frontier_state(envs[i], entry["initial_state"])
                    opp = entry["opponent"]
                    lane_opponent[i] = make_opponent(
                        opp,
                        _stable_seed("frontier-ppo", entry["entry_id"], visits[i]) & 0xFFFFFFFF,
                    )
                    estimators[i] = PublicStateEstimator()
                    hidden[:, i : i + 1, :] = 0.0
                    lane_entry[i] = entry
                    lane_steps[i] = 0
                    if truncation_reward or step_tower_reward:
                        towers = tower_fracs(envs[i].state)
                        start = (
                            float(sum(towers.get("enemy", []))),
                            float(sum(towers.get("own", []))),
                        )
                        lane_towers_start[i] = start
                        lane_towers_prev[i] = start
                    visits[i] += 1
                    episodes += 1
                    needs_entry[i] = False
            obs_list, hands, elixirs, snaps, legals = [], [], [], [], []
            for i in range(n_lanes):
                obs = envs[i].observe_v2_for_viewer(0)
                state = envs[i].state
                hand = list(state.players[0].hand[:4])
                elixir = float(state.players[0].elixir_milli) / 1000.0
                obs_list.append(_v4_step_inputs(obs, hand, elixir, estimators[i].snapshot()))
                hands.append(hand)
                elixirs.append(elixir)
                legals.append(np.array(obs.legal_play, dtype=bool, copy=True))
            F = torch.float32
            BOOL_KEYS = ("entity_mask", "opp_out_of_cycle")
            plain = [
                {k: v for k, v in ob.items() if k not in ("legal_play", "legal_wait")}
                for ob in obs_list
            ]
            array_inputs = {
                key: torch.stack(
                    [
                        torch.as_tensor(
                            np.asarray(d[key], dtype=bool if key in BOOL_KEYS else np.float32)
                        )
                        for d in plain
                    ]
                ).to(device)
                for key in plain[0]
            }
            legal = torch.stack(
                [torch.as_tensor(lg, dtype=torch.bool) for lg in legals]
            ).to(device).unsqueeze(1)
            masks = masks_from_legal_play(legal)
            # Every lane just (re)started an entry when lane_steps == 0:
            # zero-restart recurrent semantics (update-exact, smoke-identical).
            reset_rows = [lane_steps[i] == 0 for i in range(n_lanes)]
            reset_mask = torch.tensor(reset_rows, dtype=torch.bool, device=device).reshape(-1, 1)
            logits, actions, log_probs, entropy, ht, next_hidden = policy.rollout_sample(
                array_inputs["raster"].unsqueeze(1),
                array_inputs["global_features"].unsqueeze(1),
                array_inputs["entities"].unsqueeze(1),
                array_inputs["entity_mask"].unsqueeze(1),
                array_inputs["hand_tokens"].unsqueeze(1),
                array_inputs["opp_hand_probs"].unsqueeze(1),
                array_inputs["opp_out_of_cycle"].unsqueeze(1),
                array_inputs["opp_elixir_interval"].unsqueeze(1),
                array_inputs["event_history"].unsqueeze(1),
                masks,
                reset_mask=reset_mask,
                hidden=hidden,
            )
            if deterministic_policy:
                actions = policy.act_deterministic(logits, masks)
                log_probs = policy.log_prob(logits, masks, actions)
                entropy = policy.action_entropy(logits, masks)
            values = critic(ht)
            if anchor is not None:
                anchor_actions = anchor.act_deterministic(logits, masks)
            else:
                anchor_actions = None
            flat_actions = V4ActionBatch(
                mode=actions.mode[:, 0],
                card_slot=actions.card_slot[:, 0],
                placement=actions.placement[:, 0],
                wait_duration=actions.wait_duration[:, 0],
            )
            decoded = decode_v4_sim_actions(flat_actions, player=0)
            rewards, terminated, truncated = [], [], []
            for i in range(n_lanes):
                opp_action = lane_opponent[i].choose_action(envs[i].engine, envs[i].state, 1)
                result = envs[i].step_v2((decoded[i], opp_action))
                step_reward = float(result.rewards[0])
                lane_steps[i] += 1
                done = bool(result.terminated)
                cut = (not done) and lane_steps[i] >= episode_cap
                terminated.append(done)
                truncated.append(cut)
                if truncation_reward and cut:
                    towers = tower_fracs(envs[i].state)
                    start_enemy, start_own = lane_towers_start[i]
                    bonus = (start_enemy - float(sum(towers.get("enemy", [])))) - (
                        start_own - float(sum(towers.get("own", [])))
                    )
                    step_reward += bonus
                    truncation_bonuses.append(round(bonus, 5))
                if step_tower_reward:
                    towers = tower_fracs(envs[i].state)
                    prev_enemy, prev_own = lane_towers_prev[i]
                    now = (
                        float(sum(towers.get("enemy", []))),
                        float(sum(towers.get("own", []))),
                    )
                    step_reward += (prev_enemy - now[0]) - (prev_own - now[1])
                    lane_towers_prev[i] = now
                rewards.append(step_reward)
                if done:
                    winner = result.info.get("winner")
                    if winner == 0:
                        wins += 1
                    elif winner == 1:
                        losses += 1
                    else:
                        draws += 1
                    episode_lens.append(lane_steps[i])
                    needs_entry[i] = True
                elif cut:
                    episode_lens.append(lane_steps[i])
                    needs_entry[i] = True
                mode = int(actions.mode[i, 0])
                if mode == 0:
                    wait_count += 1
                else:
                    slot = int(actions.card_slot[i, 0])
                    card_counts[slot] = card_counts.get(slot, 0) + 1
                    row = int(actions.placement[i, 0, 0])
                    col = int(actions.placement[i, 0, 1])
                    if not bool(legals[i][slot, row, col]):
                        illegal_attempts += 1
            hidden = next_hidden.detach()
            anchor_present = anchor_actions is not None
            if anchor_actions is not None:
                anchor_flat = V4ActionBatch(
                    mode=anchor_actions.mode[:, 0],
                    card_slot=anchor_actions.card_slot[:, 0],
                    placement=anchor_actions.placement[:, 0],
                    wait_duration=anchor_actions.wait_duration[:, 0],
                )
            else:
                anchor_flat = None
            steps.append(
                {
                    "inputs": {k: v.detach().cpu() for k, v in array_inputs.items()},
                    "recurrent": ht.detach().cpu(),
                    "logits_mode": logits.mode.detach().cpu(),
                    "logits_duration": logits.wait_duration.detach().cpu(),
                    "logits_card": logits.card.detach().cpu(),
                    "logits_placement": logits.placement.detach().cpu(),
                    "masks_mode": masks.mode.detach().cpu(),
                    "masks_card": masks.card.detach().cpu(),
                    "masks_placement": masks.placement.detach().cpu(),
                    "actions_mode": actions.mode.detach().cpu(),
                    "actions_card": actions.card_slot.detach().cpu(),
                    "actions_placement": actions.placement.detach().cpu(),
                    "actions_duration": actions.wait_duration.detach().cpu(),
                    "anchor_mode": anchor_flat.mode.detach().cpu()
                    if anchor_flat is not None
                    else torch.zeros_like(actions.mode.detach().cpu()),
                    "anchor_card": anchor_flat.card_slot.detach().cpu()
                    if anchor_flat is not None
                    else torch.zeros_like(actions.card_slot.detach().cpu()),
                    "anchor_placement": anchor_flat.placement.detach().cpu()
                    if anchor_flat is not None
                    else torch.zeros_like(actions.placement.detach().cpu()),
                    "anchor_duration": anchor_flat.wait_duration.detach().cpu()
                    if anchor_flat is not None
                    else torch.zeros_like(actions.wait_duration.detach().cpu()),
                    "anchor_present": torch.ones(n_lanes, 1, dtype=torch.bool)
                    if anchor_present
                    else torch.zeros(n_lanes, 1, dtype=torch.bool),
                    "log_probs": log_probs.detach().cpu(),
                    "entropy": {k: v.detach().cpu() for k, v in entropy.items()},
                    "values": values.detach().cpu(),
                    "rewards": torch.tensor(rewards, dtype=torch.float32),
                    "terminated": torch.tensor(terminated, dtype=torch.bool),
                    "truncated": torch.tensor(truncated, dtype=torch.bool),
                    "reset": reset_mask.detach().cpu(),
                }
            )
    first_hidden = policy.initial_hidden(n_lanes, device=torch.device("cpu"))
    batch = _stack_rollout(steps, first_hidden, n_lanes, n_decisions, device)
    total = n_lanes * n_decisions
    info = {
        "wait_fraction": wait_count / max(1, total),
        "card_histogram": {s: c / max(1, total) for s, c in card_counts.items()},
        "illegal_attempts": illegal_attempts,
        "episode_wins": wins,
        "episode_losses": losses,
        "episode_draws": draws,
        "episode_win_rate": wins / max(1, wins + losses + draws),
        "episodes": episodes,
        "mean_episode_len": float(sum(episode_lens) / max(1, len(episode_lens))),
        "truncation_reward": bool(truncation_reward),
        "step_tower_reward": bool(step_tower_reward),
        "truncation_bonuses": truncation_bonuses,
    }
    return batch, info


def _policy_argmax_at_entry(
    policy: Any, entry: dict[str, Any], device: Any = None
) -> BranchSpec:
    """Deterministic action of a policy at a sealed entry state."""

    import torch

    try:
        from .frontier_eval import _observe_inputs
        from .model_v4 import masks_from_legal_play
    except ImportError:  # pragma: no cover
        from simulator.rl.frontier_eval import _observe_inputs
        from simulator.rl.model_v4 import masks_from_legal_play

    env = make_frontier_env()
    load_frontier_state(env, entry["initial_state"])
    inputs, legal, _, _ = _observe_inputs(env, device)
    import numpy as _np

    tensors = {
        k: torch.as_tensor(_np.asarray(v, dtype=_np.float32), device=device).unsqueeze(0).unsqueeze(0)
        for k, v in inputs.items()
        if k not in ("legal_play", "legal_wait", "entity_mask", "opp_out_of_cycle")
    }
    tensors["entity_mask"] = torch.as_tensor(
        _np.asarray(inputs["entity_mask"], dtype=bool), device=device
    ).unsqueeze(0).unsqueeze(0)
    tensors["opp_out_of_cycle"] = torch.as_tensor(
        _np.asarray(inputs["opp_out_of_cycle"], dtype=bool), device=device
    ).unsqueeze(0).unsqueeze(0)
    legal_t = torch.as_tensor(legal, device=device).unsqueeze(0).unsqueeze(0)
    masks = masks_from_legal_play(legal_t)
    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, _, _ = policy.forward(
            tensors["raster"], tensors["global_features"], tensors["entities"],
            tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
            tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
            tensors["event_history"], reset,
        )
        acts = policy.act_deterministic(logits, masks)
    mode = int(acts.mode[0, 0])
    return BranchSpec(
        source="policy", mode=mode,
        slot=int(acts.card_slot[0, 0]),
        row=int(acts.placement[0, 0, 0]),
        col=int(acts.placement[0, 0, 1]),
    )


def sealed_choice_regret(
    *,
    policy: Any,
    champion: Any,
    entries: Sequence[dict[str, Any]],
    n_replicas: int = 3,
    horizon: int = 128,
    device: Any = None,
    adaptive_replicas: bool = False,
    max_replicas: int = ADAPTIVE_REPLICA_CAP,
    se_k: float = PAIRED_SIGN_K,
) -> list[dict[str, Any]]:
    """Metric A: policy action vs champion action under champion continuation.

    Both branches share each replica's seed (common random numbers), so the
    per-replica paired differences ``policy_value - champ_value`` are the
    unit for uncertainty: each changed state reports paired mean/std/SE plus
    a confidence sign (improved/regressed/unresolved at ``se_k`` SEs).

    With ``adaptive_replicas``, volatile changed states whose paired
    interval still covers zero are extended by doubling (8 -> 16 -> 32 up to
    ``max_replicas``); same-action states never extend since their paired
    deltas are identically zero. Replica ``r`` always uses the same stable
    seed, so extension only appends new replicas deterministically.
    """

    if n_replicas < 1:
        raise ValueError("n_replicas >= 1 required")
    if max_replicas < n_replicas:
        raise ValueError("max_replicas must be >= n_replicas")

    out = []
    import torch

    for entry in entries:
        env = make_frontier_env()
        load_frontier_state(env, entry["initial_state"])
        hidden0 = torch.as_tensor(entry["hidden0"], device=device).contiguous()
        a_champ = _policy_argmax_at_entry(champion, entry, device)
        a_pol = _policy_argmax_at_entry(policy, entry, device)
        same = (a_champ.mode, a_champ.slot, a_champ.row, a_champ.col) == (
            a_pol.mode, a_pol.slot, a_pol.row, a_pol.col)
        opp_spec = entry["opponent"]
        vals_champ, vals_pol = [], []

        def _run_replica(r: int) -> None:
            replica_seed = _stable_seed("sealed-choice", entry["entry_id"], r) & 0xFFFFFFFF
            oc = run_branch(env=env, hidden=hidden0,
                            branch=BranchSpec(source="c", mode=a_champ.mode, slot=a_champ.slot, row=a_champ.row, col=a_champ.col),
                            opponent_spec=opp_spec, replica_seed=replica_seed,
                            horizon=horizon, policy=champion, device=device)
            op = run_branch(env=env, hidden=hidden0,
                            branch=BranchSpec(source="p", mode=a_pol.mode, slot=a_pol.slot, row=a_pol.row, col=a_pol.col),
                            opponent_spec=opp_spec, replica_seed=replica_seed,
                            horizon=horizon, policy=champion, device=device)
            vals_champ.append(oc["value"])
            vals_pol.append(op["value"])

        for r in range(n_replicas):
            _run_replica(r)
        extended = False
        while adaptive_replicas and not same and len(vals_champ) < max_replicas:
            diffs = [p - c for p, c in zip(vals_pol, vals_champ)]
            stats = paired_delta_stats(diffs)
            if classify_paired_sign(stats["mean"], stats["se"], same_action=False, k=se_k) != "unresolved":
                break
            target = min(max_replicas, len(vals_champ) * 2)
            for r in range(len(vals_champ), target):
                _run_replica(r)
            extended = True
        diffs = [p - c for p, c in zip(vals_pol, vals_champ)]
        stats = paired_delta_stats(diffs)
        paired_mean = stats["mean"]
        sign = classify_paired_sign(
            stats["mean"], stats["se"], same_action=same, k=se_k)
        mean_c = sum(vals_champ) / len(vals_champ)
        mean_p = sum(vals_pol) / len(vals_pol)
        out.append(
            {
                "entry_id": entry["entry_id"],
                "same_action": same,
                "champ_action": [a_champ.mode, a_champ.slot, a_champ.row, a_champ.col],
                "policy_action": [a_pol.mode, a_pol.slot, a_pol.row, a_pol.col],
                "champ_mean": round(mean_c, 5),
                "policy_mean": round(mean_p, 5),
                "delta": round(paired_mean, 5),
                # Per-replica values for uncertainty reporting (8+ replica
                # standard); means above must equal these lists' means.
                "champ_values": [round(v, 5) for v in vals_champ],
                "policy_values": [round(v, 5) for v in vals_pol],
                # Paired deltas under CRN + their uncertainty. paired_mean
                # equals delta; sign is the confidence label at se_k SEs.
                "paired_deltas": [round(v, 5) for v in diffs],
                "paired_mean": round(paired_mean, 5),
                "paired_std": round(stats["std"], 5),
                "paired_se": round(stats["se"], 5),
                "n_replicas": len(vals_champ),
                "replicas_extended": extended,
                "sign": sign,
            }
        )
    return out


def sealed_rollout_returns(
    *,
    policy: Any,
    entries: Sequence[dict[str, Any]],
    n_replicas: int = 3,
    horizon: int = 128,
    device: Any = None,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Metric B: stochastic policy rollouts from sealed entries (CRN-ready seeds)."""

    import torch

    try:
        from .model_v4 import masks_from_legal_play
        from .v4_ppo import _v4_step_inputs, decode_v4_sim_actions
    except ImportError:  # pragma: no cover
        from simulator.rl.model_v4 import masks_from_legal_play
        from simulator.rl.v4_ppo import _v4_step_inputs, decode_v4_sim_actions
    from simulator.public_state_estimator import PublicStateEstimator

    out = []
    for entry in entries:
        rep_returns, rep_winners, rep_lens = [], [], []
        for r in range(n_replicas):
            replica_seed = (_stable_seed("sealed-rollout", base_seed, entry["entry_id"], r) & 0xFFFFFFFF)
            torch.manual_seed(replica_seed)
            try:
                torch.cuda.manual_seed_all(replica_seed)
            except Exception:
                pass
            env = make_frontier_env()
            load_frontier_state(env, entry["initial_state"])
            hidden = torch.zeros_like(torch.as_tensor(entry["hidden0"], device=device))
            opponent = make_opponent(entry["opponent"], replica_seed)
            estimator = PublicStateEstimator()
            total = 0.0
            winner: Any = None
            ran = 0
            with torch.no_grad():
                for _ in range(horizon):
                    if env.state is None or env.state.terminal:
                        break
                    obs = env.observe_v2_for_viewer(0)
                    st = env.state
                    hand = list(st.players[0].hand[:4])
                    elixir = float(st.players[0].elixir_milli) / 1000.0
                    tensors = _v4_step_inputs(obs, hand, elixir, estimator.snapshot())
                    F = torch.float32
                    args = {
                        k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
                        for k, v in tensors.items()
                        if k not in ("legal_play", "legal_wait", "entity_mask", "opp_out_of_cycle")
                    }
                    args["entity_mask"] = torch.as_tensor(
                        np.asarray(tensors["entity_mask"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
                    args["opp_out_of_cycle"] = torch.as_tensor(
                        np.asarray(tensors["opp_out_of_cycle"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
                    legal = torch.as_tensor(
                        np.array(obs.legal_play, dtype=bool, copy=True), device=device).unsqueeze(0).unsqueeze(0)
                    masks = masks_from_legal_play(legal)
                    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
                    _, acts, _, _, _, nxt = policy.rollout_sample(
                        args["raster"], args["global_features"], args["entities"],
                        args["entity_mask"], args["hand_tokens"], args["opp_hand_probs"],
                        args["opp_out_of_cycle"], args["opp_elixir_interval"],
                        args["event_history"], masks, reset_mask=reset, hidden=hidden,
                    )
                    hidden = nxt.detach()
                    flat = acts
                    from simulator.rl.model_v4 import V4ActionBatch as _AB

                    mine = decode_v4_sim_actions(
                        _AB(mode=flat.mode[:, 0], card_slot=flat.card_slot[:, 0],
                            placement=flat.placement[:, 0], wait_duration=flat.wait_duration[:, 0]),
                        player=0,
                    )[0]
                    opp = opponent.choose_action(env.engine, env.state, 1)
                    result = env.step_v2((mine, opp))
                    total += result.rewards[0]
                    ran += 1
                    if result.terminated or result.truncated:
                        winner = result.info.get("winner")
                        break
            rep_returns.append(round(total, 4))
            rep_winners.append(winner)
            rep_lens.append(ran)
        out.append(
            {
                "entry_id": entry["entry_id"],
                "mean_return": round(sum(rep_returns) / len(rep_returns), 4),
                "returns": rep_returns,
                "winners": rep_winners,
                "mean_len": round(sum(rep_lens) / len(rep_lens), 1),
            }
        )
    return out


def pinned_opponent_decks(
    *, pool_seed: int = 0, episode_index: int = 500
) -> list[dict[str, Any]]:
    """Pinned-deck specs for the 6 discovery archetypes (eval pairing)."""

    from simulator.ruleset import load_fixed_ruleset

    try:
        from .opponent_pool import OpponentPool
    except ImportError:  # pragma: no cover
        from simulator.rl.opponent_pool import OpponentPool

    from simulator.rl.frontier import DISCOVERY_ARCHETYPES

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=pool_seed)
    specs = []
    for arch in DISCOVERY_ARCHETYPES:
        sampled = pool.sample(episode_index, archetype=arch, allow_variants=False)
        specs.append(
            {
                "archetype": arch,
                "strategy": sampled.strategy,
                "deck": list(sampled.deck.cards),
                "controller_seed": int(sampled.controller_seed),
            }
        )
    return specs


def full_match_record(
    *,
    policy: Any,
    opponent_specs: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    device: Any = None,
    deterministic: bool = True,
    max_decisions: int = 1200,
) -> list[dict[str, Any]]:
    """Metric C: paired deterministic full matches vs the pinned pool."""

    import torch

    from simulator.roster import PLAYER_DECK

    try:
        from .model_v4 import masks_from_legal_play
        from .v4_ppo import _v4_step_inputs, decode_v4_sim_actions
    except ImportError:  # pragma: no cover
        from simulator.rl.model_v4 import masks_from_legal_play
        from simulator.rl.v4_ppo import _v4_step_inputs, decode_v4_sim_actions
    from simulator.public_state_estimator import PublicStateEstimator

    out = []
    for spec in opponent_specs:
        for match_seed in seeds:
            env = make_frontier_env()
            env.reset_v2(
                seed=int(match_seed),
                decks=(tuple(PLAYER_DECK), tuple(spec["deck"])),
                shuffle_decks=True,
            )
            opponent = make_opponent(
                {"strategy": spec["strategy"], "controller_seed": spec["controller_seed"]},
                int(match_seed),
            )
            hidden = policy.initial_hidden(1, device=device)
            estimator = PublicStateEstimator()
            winner: Any = None
            ran = 0
            with torch.no_grad():
                for _ in range(max_decisions):
                    if env.state is None or env.state.terminal:
                        break
                    obs = env.observe_v2_for_viewer(0)
                    st = env.state
                    hand = list(st.players[0].hand[:4])
                    elixir = float(st.players[0].elixir_milli) / 1000.0
                    tensors = _v4_step_inputs(obs, hand, elixir, estimator.snapshot())
                    F = torch.float32
                    args = {
                        k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
                        for k, v in tensors.items()
                        if k not in ("legal_play", "legal_wait", "entity_mask", "opp_out_of_cycle")
                    }
                    args["entity_mask"] = torch.as_tensor(
                        np.asarray(tensors["entity_mask"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
                    args["opp_out_of_cycle"] = torch.as_tensor(
                        np.asarray(tensors["opp_out_of_cycle"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
                    legal = torch.as_tensor(
                        np.array(obs.legal_play, dtype=bool, copy=True), device=device).unsqueeze(0).unsqueeze(0)
                    masks = masks_from_legal_play(legal)
                    reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
                    if deterministic:
                        logits, _, _ = policy.forward(
                            args["raster"], args["global_features"], args["entities"],
                            args["entity_mask"], args["hand_tokens"], args["opp_hand_probs"],
                            args["opp_out_of_cycle"], args["opp_elixir_interval"],
                            args["event_history"], reset,
                        )
                        acts = policy.act_deterministic(logits, masks)
                    else:
                        _, acts, _, _, _, nxt = policy.rollout_sample(
                            args["raster"], args["global_features"], args["entities"],
                            args["entity_mask"], args["hand_tokens"], args["opp_hand_probs"],
                            args["opp_out_of_cycle"], args["opp_elixir_interval"],
                            args["event_history"], masks, reset_mask=reset, hidden=hidden,
                        )
                        hidden = nxt.detach()
                    from simulator.rl.model_v4 import V4ActionBatch as _AB

                    mine = decode_v4_sim_actions(
                        _AB(mode=acts.mode[:, 0], card_slot=acts.card_slot[:, 0],
                            placement=acts.placement[:, 0], wait_duration=acts.wait_duration[:, 0]),
                        player=0,
                    )[0]
                    opp = opponent.choose_action(env.engine, env.state, 1)
                    result = env.step_v2((mine, opp))
                    ran += 1
                    if result.terminated or result.truncated:
                        winner = result.info.get("winner")
                        break
            out.append(
                {
                    "archetype": spec["archetype"],
                    "seed": int(match_seed),
                    "winner": winner,
                    "decisions": ran,
                    "result": "win" if winner == 0 else ("loss" if winner == 1 else "draw"),
                }
            )
    return out


def action_logit_margins(
    policy: Any, entries: Sequence[dict[str, Any]], device: Any = None
) -> dict[str, Any]:
    """Per-entry chosen-vs-runner-up margins (direct confidence readout).

    Same zero-start forward convention as :func:`_policy_argmax_at_entry`
    so margins describe the exact logits the policy acts on.  Margins are
    probability gaps after legality masking: mode (2-way), card (legal
    slots), placement (legal cells of the chosen slot).  Duration excluded
    (WAIT-only head, rarely decisive).
    """

    import torch

    try:
        from .frontier_eval import _observe_inputs
        from .model_v4 import masks_from_legal_play
    except ImportError:  # pragma: no cover
        from simulator.rl.frontier_eval import _observe_inputs
        from simulator.rl.model_v4 import masks_from_legal_play

    per_entry = []
    for entry in entries:
        env = make_frontier_env()
        load_frontier_state(env, entry["initial_state"])
        inputs, legal, _, _ = _observe_inputs(env, device)
        tensors = {
            k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
            for k, v in inputs.items()
            if k not in ("legal_play", "legal_wait", "entity_mask", "opp_out_of_cycle")
        }
        tensors["entity_mask"] = torch.as_tensor(
            np.asarray(inputs["entity_mask"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
        tensors["opp_out_of_cycle"] = torch.as_tensor(
            np.asarray(inputs["opp_out_of_cycle"], dtype=bool), device=device).unsqueeze(0).unsqueeze(0)
        legal_t = torch.as_tensor(legal, device=device).unsqueeze(0).unsqueeze(0)
        masks = masks_from_legal_play(legal_t)
        reset = torch.zeros((1, 1), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _, _ = policy.forward(
                tensors["raster"], tensors["global_features"], tensors["entities"],
                tensors["entity_mask"], tensors["hand_tokens"], tensors["opp_hand_probs"],
                tensors["opp_out_of_cycle"], tensors["opp_elixir_interval"],
                tensors["event_history"], reset,
            )
            mode_logp = torch.where(
                masks.mode, logits.mode[0, 0].float(),
                torch.full_like(logits.mode[0, 0], float("-inf")),
            )
            mode_p = torch.softmax(mode_logp, dim=-1).cpu().numpy().reshape(-1)
            card_logp = torch.where(
                masks.card, logits.card[0, 0].float(),
                torch.full_like(logits.card[0, 0], float("-inf")),
            )
            card_p = torch.softmax(card_logp, dim=-1).cpu().numpy().reshape(-1)
            mode_sorted = sorted(float(v) for v in mode_p)
            mode_margin = mode_sorted[-1] - (mode_sorted[-2] if len(mode_sorted) > 1 else 0.0)
            card_sorted = sorted(float(v) for v in card_p)
            card_margin = card_sorted[-1] - (card_sorted[-2] if len(card_sorted) > 1 else 0.0)
            chosen_mode = int(mode_p.argmax())
            placement_margin = None
            if chosen_mode == 1:
                slot = int(card_p.argmax())
                cell_logp = torch.where(
                    masks.placement[0, 0, slot],
                    logits.placement[0, 0, slot].float(),
                    torch.full_like(logits.placement[0, 0, slot], float("-inf")),
                )
                cell_p = torch.softmax(cell_logp.reshape(-1), dim=-1).cpu().numpy()
                cell_sorted = sorted(float(v) for v in cell_p)
                placement_margin = cell_sorted[-1] - (
                    cell_sorted[-2] if len(cell_sorted) > 1 else 0.0)
        per_entry.append(
            {
                "entry_id": entry["entry_id"],
                "chosen_mode": chosen_mode,
                "mode_margin": round(float(mode_margin), 4),
                "card_margin": round(float(card_margin), 4),
                "placement_margin": round(float(placement_margin), 4) if placement_margin is not None else None,
            }
        )

    def _summary(vals: list[float]) -> dict[str, float]:
        s = sorted(vals)
        n = len(s)
        return {
            "n": n,
            "mean": round(sum(s) / n, 4),
            "p10": round(s[max(0, n // 10)], 4),
            "median": round(s[n // 2], 4),
            "p90": round(s[min(n - 1, 9 * n // 10)], 4),
            "frac_close_lt_010": round(sum(1 for v in s if v < 0.10) / n, 4),
            "frac_close_lt_025": round(sum(1 for v in s if v < 0.25) / n, 4),
        }

    return {
        "version": FRONTIER_PPO_VERSION,
        "per_entry": per_entry,
        "mode": _summary([e["mode_margin"] for e in per_entry]),
        "card": _summary([e["card_margin"] for e in per_entry]),
        "placement": _summary([e["placement_margin"] for e in per_entry if e["placement_margin"] is not None]),
        "n_play": sum(1 for e in per_entry if e["chosen_mode"] == 1),
        "n_wait": sum(1 for e in per_entry if e["chosen_mode"] == 0),
    }


__all__ = [
    "ADAPTIVE_REPLICA_CAP",
    "CATASTROPHIC_DELTA",
    "FRONTIER_PPO_EPISODE_CAP",
    "FRONTIER_PPO_VERSION",
    "PAIRED_SIGN_K",
    "TRIM_FRACTION",
    "action_logit_margins",
    "build_comparison_summary",
    "classify_paired_sign",
    "collect_frontier_rollout",
    "full_match_record",
    "make_frontier_env",
    "paired_delta_stats",
    "pinned_opponent_decks",
    "sealed_choice_regret",
    "sealed_rollout_returns",
    "summarize_choice_regret",
]
