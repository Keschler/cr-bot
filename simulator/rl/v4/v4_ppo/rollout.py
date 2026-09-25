"""Collect and assemble simulator rollouts with episode-safe bootstraps."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import torch
from .runtime import configure_v4_runtime, decode_v4_sim_actions, _v4_step_inputs
from ..model_v4 import RecurrentV4Policy, V4ValueHead, V4ActionBatch, masks_from_legal_play
from ...foundation.trajectory import ActionMasks, RecurrentSequence, TrajectoryBatch
from .records import V4RolloutBatch
from .config import V4_RUNTIME_VERSION


def post_step_value(policy, critic, env, estimator, hidden, device):
    """Value of the real successor observation, before any environment reset."""
    if env.state.terminal and env.state.terminal_reason != "runner_tick_limit":
        return 0.0
    estimator.sync_from_env(env, viewer=0)
    obs = env.observe_v2_for_viewer(0)
    own = env.state.players[0]
    packed = _v4_step_inputs(obs, list(own.hand), own.elixir_milli / 1000.0, estimator.snapshot())
    inputs = {key: torch.as_tensor(value, device=device).unsqueeze(0).unsqueeze(0)
              for key, value in packed.items() if key not in ("legal_play", "legal_wait")}
    with torch.no_grad():
        _, ht, _ = policy.forward_step(inputs, hidden=hidden)
        return float(critic(ht).reshape(-1)[0].item())


def rollout_next_values(values, terminated, truncated, boundary_values):
    """Shift within episodes only; cuts and rollout tails use saved successors."""
    result = torch.zeros_like(values)
    result[:, :-1] = values[:, 1:]
    needs_successor = truncated.clone()
    needs_successor[:, -1] = True
    needs_successor &= ~terminated
    if not torch.isfinite(boundary_values[needs_successor]).all():
        raise ValueError("missing post-step bootstrap value at a truncation or rollout tail")
    result[needs_successor] = boundary_values[needs_successor]
    result[terminated] = 0.0
    return result


def _observe_lanes(
    lanes: Sequence[dict[str, Any]], device: Any,
) -> tuple[dict[str, torch.Tensor], ActionMasks, torch.Tensor]:
    """Synchronize public histories and pack one decision across all lanes."""
    batch_inputs: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "raster", "global_features", "entities", "entity_mask",
            "hand_tokens", "opp_hand_probs", "opp_out_of_cycle",
            "opp_elixir_interval", "event_history", "legal_play",
        )
    }
    reset_rows: list[bool] = []
    legal_list: list[np.ndarray] = []
    for lane in lanes:
        env = lane["env"]
        state = env.state
        if state is None:
            raise RuntimeError("rollout lane lost its authoritative state")
        observation = env.observe_v2_for_viewer(0)
        player_state = state.players[0]
        hand = list(player_state.hand)
        elixir = float(player_state.elixir_milli) / 1000.0
        lane["estimator"].sync_from_env(env, viewer=0)
        snapshot = lane["estimator"].snapshot()
        packed = _v4_step_inputs(observation, hand, elixir, snapshot)
        for key, value in packed.items():
            if key == "legal_wait":
                continue
            batch_inputs[key].append(value)
        legal_list.append(packed["legal_play"])
        reset_rows.append(bool(lane.get("needs_reset", True)))
        lane["needs_reset"] = False
    array_inputs = {
        key: torch.as_tensor(np.stack(values)).unsqueeze(1).to(device)
        for key, values in batch_inputs.items()
        if key != "legal_play"
    }
    # Boolean arrays need explicit dtype routing.
    array_inputs["entity_mask"] = array_inputs["entity_mask"].to(torch.bool)
    array_inputs["opp_out_of_cycle"] = array_inputs["opp_out_of_cycle"].to(torch.bool)
    legal = torch.as_tensor(np.stack(legal_list)).unsqueeze(1).to(device)
    masks = masks_from_legal_play(legal)
    reset_mask = torch.tensor(reset_rows, dtype=torch.bool, device=device).reshape(-1, 1)
    return array_inputs, masks, reset_mask


def collect_v4_rollout(
    policy: RecurrentV4Policy,
    critic: V4ValueHead,
    anchor: RecurrentV4Policy | None,
    lanes: Sequence[dict[str, Any]],
    *,
    n_decisions: int,
    device: Any = None,
    deterministic_anchor: bool = True,
    deterministic_policy: bool = False,
) -> tuple[V4RolloutBatch, dict[str, Any]]:
    """Collect one fixed-length V4 rollout across scenario lanes.

    Each lane dict carries ``env`` (a reset ``BasicMechanicsScenarioEnv``),
    ``opponent`` (a controller with ``choose_action(engine, state, player)``),
    and ``estimator`` (a fresh ``PublicStateEstimator``).  Envs that
    terminate mid-rollout reset in place with the lane's next seed and a
    ``reset_mask`` break, so every lane contributes exactly ``n_decisions``.
    """

    device = device or next(policy.parameters()).device
    n_lanes = len(lanes)
    if n_lanes < 1:
        raise ValueError("at least one rollout lane is required")
    if type(n_decisions) is not int or n_decisions < 1:
        raise ValueError("n_decisions must be a positive integer")
    if type(deterministic_anchor) is not bool:
        raise TypeError("deterministic_anchor must be boolean")
    if type(deterministic_policy) is not bool:
        raise TypeError("deterministic_policy must be boolean")
    configure_v4_runtime(policy)
    policy.eval()
    critic.eval()
    if anchor is not None:
        configure_v4_runtime(anchor)
        anchor.eval()
    anchor_hidden = anchor.initial_hidden(n_lanes, device=device) if anchor is not None else None
    hidden = policy.initial_hidden(n_lanes, device=device)
    steps: list[dict[str, torch.Tensor]] = []
    illegal_attempts = 0
    wait_count = 0
    episode_wins = 0
    episode_losses = 0
    episode_draws = 0
    all_rewards: list[float] = []
    lane_outcomes: list[list[int]] = [[0, 0, 0] for _ in range(n_lanes)]
    card_counts: dict[int, int] = {}
    with torch.no_grad():
        for decision_index in range(n_decisions):
            array_inputs, masks, reset_mask = _observe_lanes(lanes, device)
            logits, actions, log_probs, entropy, ht, next_hidden = policy.rollout_sample(
                array_inputs["raster"],
                array_inputs["global_features"],
                array_inputs["entities"],
                array_inputs["entity_mask"],
                array_inputs["hand_tokens"],
                array_inputs["opp_hand_probs"],
                array_inputs["opp_out_of_cycle"],
                array_inputs["opp_elixir_interval"],
                array_inputs["event_history"],
                masks,
                reset_mask=reset_mask,
                hidden=hidden,
            )
            if deterministic_policy:
                # Eval-fair comparison: argmax actions scored under the same
                # masks (always finite: argmax selections are legal).
                actions = policy.act_deterministic(logits, masks)
                log_probs = policy.log_prob(logits, masks, actions)
                entropy = policy.action_entropy(logits, masks)
            values = critic(ht)
            anchor_actions: V4ActionBatch | None = None
            if anchor is not None:
                anchor_logits, _, anchor_hidden = anchor.forward_step(
                    array_inputs, hidden=anchor_hidden, reset_mask=reset_mask
                )
                if deterministic_anchor:
                    anchor_actions = anchor.act_deterministic(anchor_logits, masks)
                    # Anchor decode runs under no-grad; the learner scores
                    # these actions differentiably at update time.
                else:
                    sampled, _, _ = anchor.sample_action(anchor_logits, masks)
                    anchor_actions = sampled
            sim_actions = decode_v4_sim_actions(
                V4ActionBatch(
                    mode=actions.mode[:, 0].reshape(-1),
                    card_slot=actions.card_slot[:, 0].reshape(-1),
                    placement=actions.placement[:, 0].reshape(-1, 2),
                    wait_duration=actions.wait_duration[:, 0].reshape(-1),
                ),
                player=0,
            )
            rewards: list[float] = []
            terminated: list[bool] = []
            truncated: list[bool] = []
            boundary_values = [float("nan")] * n_lanes
            for lane_index, lane in enumerate(lanes):
                env = lane["env"]
                opponent_action = lane["opponent"].choose_action(env.engine, env.state, 1)
                result = env.step_v2((sim_actions[lane_index], opponent_action))
                rewards.append(float(result.rewards[0]))
                all_rewards.append(float(result.rewards[0]))
                done = bool(result.terminated) and not bool(result.truncated)
                terminated.append(done)
                truncated.append(bool(result.truncated))
                if done:
                    winner = result.info.get("winner", None) if isinstance(result.info, dict) else None
                    if winner == 0:
                        episode_wins += 1
                        lane_outcomes[lane_index][0] += 1
                    elif winner == 1:
                        episode_losses += 1
                        lane_outcomes[lane_index][1] += 1
                    else:
                        episode_draws += 1
                        lane_outcomes[lane_index][2] += 1
                if not done and (result.truncated or decision_index == n_decisions - 1):
                    boundary_values[lane_index] = post_step_value(
                        policy, critic, env, lane["estimator"],
                        next_hidden[:, lane_index:lane_index + 1], device,
                    )
                if bool(result.terminated) or bool(result.truncated):
                    seed = int(lane.get("next_seed", 0))
                    lane["env"].reset_v2(seed=seed, decks=lane["decks"], shuffle_decks=True)
                    lane["estimator"].reset()
                    lane["needs_reset"] = True
                    lane["next_seed"] = seed + 1
                if int(actions.mode[lane_index, 0]) == 0:
                    wait_count += 1
                else:
                    card_counts[int(actions.card_slot[lane_index, 0])] = (
                        card_counts.get(int(actions.card_slot[lane_index, 0]), 0) + 1
                    )
                # Defense in depth: decoded actions must honor legality even
                # though sampling already guarantees it.
                played = int(actions.mode[lane_index, 0]) == 1
                if played:
                    slot = int(actions.card_slot[lane_index, 0])
                    row = int(actions.placement[lane_index, 0, 0])
                    col = int(actions.placement[lane_index, 0, 1])
                    if not bool(masks.placement[lane_index, 0, slot, row, col]):
                        illegal_attempts += 1
            hidden = next_hidden.detach()
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
                    "anchor_mode": anchor_actions.mode.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.mode.detach().cpu()),
                    "anchor_card": anchor_actions.card_slot.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.card_slot.detach().cpu()),
                    "anchor_placement": anchor_actions.placement.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.placement.detach().cpu()),
                    "anchor_duration": anchor_actions.wait_duration.detach().cpu()
                    if anchor_actions is not None
                    else torch.zeros_like(actions.wait_duration.detach().cpu()),
                    "anchor_present": torch.ones(n_lanes, 1, dtype=torch.bool)
                    if anchor_actions is not None
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
    batch = _stack_rollout(steps, first_hidden, n_lanes, n_decisions, device)
    total_actions = n_lanes * n_decisions
    histogram = (
        {str(slot): count / total_actions for slot, count in sorted(card_counts.items())}
        if total_actions
        else {}
    )
    n_episodes = episode_wins + episode_losses + episode_draws
    return batch, {
        "wait_fraction": wait_count / total_actions if total_actions else 0.0,
        "card_histogram": histogram,
        "illegal_attempts": illegal_attempts,
        "median_reward": float(np.median(all_rewards)) if all_rewards else 0.0,
        "episode_wins": episode_wins,
        "episode_losses": episode_losses,
        "episode_draws": episode_draws,
        "episode_win_rate": episode_wins / n_episodes if n_episodes else 0.0,
        "lane_outcomes": lane_outcomes,
        "runtime_version": V4_RUNTIME_VERSION,
    }


def _stack_rollout(
    steps: list[dict[str, torch.Tensor]],
    first_hidden: torch.Tensor,
    n_lanes: int,
    n_decisions: int,
    device: Any,
) -> V4RolloutBatch:
    """Stack per-decision rows into trajectory + anchor containers.

    Per-step rows carry the singleton time axis used by the forward call;
    stacking squeezes it first so the batch is exactly
    ``[lanes, decisions, ...]``.  Rewards/flags are stored 1-D per step.
    """

    def stack(key: str) -> torch.Tensor:
        rows = [
            row[key].squeeze(1) if row[key].ndim > 1 else row[key]
            for row in steps
        ]
        return torch.stack(rows, dim=1).to(device)

    def stacked_input(key: str) -> torch.Tensor:
        return torch.stack(
            [row["inputs"][key].squeeze(1) for row in steps], dim=1
        ).to(device)

    sequence = RecurrentSequence(
        raster=stacked_input("raster"),
        global_features=stacked_input("global_features"),
        entities=stacked_input("entities"),
        entity_mask=stacked_input("entity_mask").to(torch.bool),
        reset_mask=torch.stack(
            [row["reset"].squeeze(1) for row in steps], dim=1
        ).to(device, dtype=torch.bool),
        # Smoke episodes reset explicitly and re-evaluate from the stored
        # initial state; per-step hidden snapshots and burn-in windows are
        # the documented follow-up, not smoke requirements.
        initial_hidden=first_hidden.to(device),
        hand_tokens=stacked_input("hand_tokens"),
        opp_hand_probs=stacked_input("opp_hand_probs"),
        opp_out_of_cycle=stacked_input("opp_out_of_cycle").to(torch.bool),
        opp_elixir_interval=stacked_input("opp_elixir_interval"),
        event_history=stacked_input("event_history"),
    )
    masks = ActionMasks(
        mode=stack("masks_mode"),
        card=stack("masks_card"),
        placement=stack("masks_placement"),
    )
    actions = V4ActionBatch(
        mode=stack("actions_mode"),
        card_slot=stack("actions_card"),
        placement=stack("actions_placement"),
        wait_duration=stack("actions_duration"),
    )
    anchor_present = stack("anchor_present")
    anchor_actions = None
    if bool(anchor_present.any().item()):
        anchor_actions = V4ActionBatch(
            mode=stack("anchor_mode"),
            card_slot=stack("anchor_card"),
            placement=stack("anchor_placement"),
            wait_duration=stack("anchor_duration"),
        )
    trajectory = TrajectoryBatch(
        sequence=sequence,
        action_masks=masks,
        actions=actions,
        rewards=stack("rewards"),
        terminated=stack("terminated").to(torch.bool),
        truncated=stack("truncated").to(torch.bool),
        old_log_probs=stack("log_probs"),
        values=stack("values"),
    )
    entropies = {
        key: torch.stack(
            [row["entropy"][key].squeeze(1) for row in steps], dim=1
        ).to(device)
        for key in ("joint", "mode", "duration", "card", "placement")
    }
    old_head_logits = {
        key: torch.stack(
            [row[f"logits_{key}"].squeeze(1) for row in steps], dim=1
        ).to(device)
        for key in ("mode", "duration", "card", "placement")
    }
    return V4RolloutBatch(
        trajectory=trajectory,
        anchor_actions=anchor_actions,
        recurrent=torch.stack(
            [row["recurrent"].squeeze(1) for row in steps], dim=1
        ).to(device),
        old_head_logits=old_head_logits,
        per_head_entropy=entropies,
        next_values=rollout_next_values(
            trajectory.values, trajectory.terminated, trajectory.truncated, stack("boundary_values")
        ),
    )

