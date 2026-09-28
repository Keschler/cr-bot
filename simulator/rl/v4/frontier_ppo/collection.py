"""Frontier episode collection with explicit termination and bootstrap boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..frontier import _stable_seed
from ..frontier_eval import load_frontier_state, make_opponent, tower_fracs
from .constants import FRONTIER_PPO_EPISODE_CAP
from .environment import make_frontier_env


@dataclass
class _FrontierLane:
    """Mutable episode state for one environment; recurrent tensors stay batched."""

    env: Any
    estimator: Any
    opponent: Any = None
    visits: int = 0
    steps: int = 0
    needs_entry: bool = True
    towers_start: tuple[float, float] = (0.0, 0.0)
    towers_prev: tuple[float, float] = (0.0, 0.0)


def _start_entry(lane: _FrontierLane, entry: dict[str, Any], seed: int, tower_reward: bool) -> None:
    """Reset one episode, including estimator history and tower reward anchors."""
    from simulator.public_state_estimator import PublicStateEstimator

    load_frontier_state(lane.env, entry["initial_state"])
    opp = entry["opponent"]
    lane.opponent = make_opponent(
        opp,
        _stable_seed("frontier-ppo", seed, entry["entry_id"], lane.visits) & 0xFFFFFFFF,
    )
    lane.estimator = PublicStateEstimator()
    lane.steps = 0
    if tower_reward:
        towers = tower_fracs(lane.env.state)
        start = (
            float(sum(towers.get("enemy", []))),
            float(sum(towers.get("own", []))),
        )
        lane.towers_start = start
        lane.towers_prev = start
    lane.visits += 1
    lane.needs_entry = False


def _observe_lanes(lanes: Sequence[_FrontierLane], device: Any):
    """Batch current observations without advancing any environment or memory."""
    import torch
    from ..model_v4 import masks_from_legal_play
    from ..v4_ppo.runtime import observe_v4_inputs

    obs_list, legals = [], []
    for lane in lanes:
        obs, packed, _, _ = observe_v4_inputs(lane.env, lane.estimator)
        obs_list.append(packed)
        legals.append(np.array(obs.legal_play, dtype=bool, copy=True))
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
    legal_wait = torch.tensor(
        [bool(ob["legal_wait"]) for ob in obs_list], dtype=torch.bool, device=device
    ).unsqueeze(1)
    masks = masks_from_legal_play(legal, legal_wait)
    return array_inputs, legals, masks


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
      since the previous decision. Undiscounted totals telescope; discounted
      returns generally differ from the truncation-lump objective.  No truncation lump is added in this mode.  Terminal
      episodes keep the env sparse outcome plus the final delta.

    No other shaping exists in any mode.
    """

    import torch

    from ..model_v4 import V4ActionBatch
    from ..v4_ppo import decode_v4_sim_actions, configure_v4_runtime
    from ..v4_ppo.rollout_utils import _stack_rollout, post_step_value
    from simulator.public_state_estimator import PublicStateEstimator

    if episode_cap < 1 or type(episode_cap) is not int:
        raise ValueError("episode_cap must be a positive integer")
    if truncation_reward and step_tower_reward:
        raise ValueError("choose exactly one tower reward mode")
    if not entries:
        raise ValueError("collect_frontier_rollout requires at least one entry")
    if n_lanes < 1 or type(n_decisions) is not int or n_decisions < 1:
        raise ValueError("n_lanes >= 1 and integer n_decisions >= 1 required")

    configure_v4_runtime(policy)
    policy.eval()
    critic.eval()
    if anchor is not None:
        configure_v4_runtime(anchor)
        anchor.eval()
    lanes = [_FrontierLane(make_frontier_env(), PublicStateEstimator()) for _ in range(n_lanes)]
    hidden = policy.initial_hidden(n_lanes, device=device)
    anchor_hidden = anchor.initial_hidden(n_lanes, device=device) if anchor is not None else None
    cursor = int(seed) % len(entries)
    truncation_bonuses: list[float] = []

    steps: list[dict[str, torch.Tensor]] = []
    wins = losses = draws = 0
    wait_count = 0
    card_counts: dict[int, int] = {}
    illegal_attempts = 0
    episodes = 0
    episode_lens: list[int] = []
    with torch.no_grad():
        for decision_index in range(n_decisions):
            for i in range(n_lanes):
                if lanes[i].needs_entry:
                    entry = entries[(cursor + i + lanes[i].visits * n_lanes) % len(entries)]
                    _start_entry(lanes[i], entry, seed, truncation_reward or step_tower_reward)
                    hidden[:, i : i + 1, :] = 0.0
                    episodes += 1
            array_inputs, legals, masks = _observe_lanes(lanes, device)
            # Every lane just (re)started an entry when its step count is zero:
            # zero-restart recurrent semantics (update-exact, smoke-identical).
            reset_rows = [lane.steps == 0 for lane in lanes]
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
                anchor_logits, _, anchor_hidden = anchor.forward_step(
                    {k: v.unsqueeze(1) for k, v in array_inputs.items()},
                    hidden=anchor_hidden, reset_mask=reset_mask,
                )
                anchor_actions = anchor.act_deterministic(anchor_logits, masks)
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
            boundary_values = [float("nan")] * n_lanes
            for i in range(n_lanes):
                opp_action = lanes[i].opponent.choose_action(lanes[i].env.engine, lanes[i].env.state, 1)
                result = lanes[i].env.step_v2((decoded[i], opp_action))
                step_reward = float(result.rewards[0])
                lanes[i].steps += 1
                done = bool(result.terminated) and not bool(result.truncated)
                cut = (not done) and (result.truncated or lanes[i].steps >= episode_cap)
                terminated.append(done)
                truncated.append(cut)
                if truncation_reward and cut:
                    towers = tower_fracs(lanes[i].env.state)
                    start_enemy, start_own = lanes[i].towers_start
                    bonus = (start_enemy - float(sum(towers.get("enemy", [])))) - (
                        start_own - float(sum(towers.get("own", [])))
                    )
                    step_reward += bonus
                    truncation_bonuses.append(round(bonus, 5))
                if step_tower_reward:
                    towers = tower_fracs(lanes[i].env.state)
                    prev_enemy, prev_own = lanes[i].towers_prev
                    now = (
                        float(sum(towers.get("enemy", []))),
                        float(sum(towers.get("own", []))),
                    )
                    step_reward += (prev_enemy - now[0]) - (prev_own - now[1])
                    lanes[i].towers_prev = now
                rewards.append(step_reward)
                if not done and (cut or decision_index == n_decisions - 1):
                    boundary_values[i] = post_step_value(
                        policy, critic, lanes[i].env, lanes[i].estimator, next_hidden[:, i:i + 1], device,
                    )
                if done:
                    winner = result.info.get("winner")
                    if winner == 0:
                        wins += 1
                    elif winner == 1:
                        losses += 1
                    else:
                        draws += 1
                    episode_lens.append(lanes[i].steps)
                    lanes[i].needs_entry = True
                elif cut:
                    episode_lens.append(lanes[i].steps)
                    lanes[i].needs_entry = True
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
                    "boundary_values": torch.tensor(boundary_values, dtype=torch.float32),
                    "reset": reset_mask.detach().cpu(),
                }
            )
    first_hidden = policy.initial_hidden(n_lanes, device=torch.device("cpu"))
    batch = _stack_rollout(steps, first_hidden, device)
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
