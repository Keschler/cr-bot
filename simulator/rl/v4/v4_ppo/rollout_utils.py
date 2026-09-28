"""Shared tensor assembly and successor bootstrapping for V4 rollouts."""

from __future__ import annotations

from typing import Any

import torch

from .records import V4RolloutBatch
from .runtime import _v4_step_inputs, batch_v4_inputs
from ..model_v4 import V4ActionBatch
from ...foundation.trajectory import ActionMasks, RecurrentSequence, TrajectoryBatch


def post_step_value(policy, critic, env, estimator, hidden, device):
    """Evaluate the real successor observation before any environment reset."""
    if env.state.terminal and env.state.terminal_reason != "runner_tick_limit":
        return 0.0
    estimator.sync_from_env(env, viewer=0)
    obs = env.observe_v2_for_viewer(0)
    own = env.state.players[0]
    packed = _v4_step_inputs(
        obs,
        list(own.hand),
        own.elixir_milli / 1000.0,
        estimator.snapshot(),
    )
    inputs = batch_v4_inputs(packed, device)
    with torch.no_grad():
        _, hidden_out, _ = policy.forward_step(inputs, hidden=hidden)
        return float(critic(hidden_out).reshape(-1)[0].item())


def rollout_next_values(values, terminated, truncated, boundary_values):
    """Shift values within episodes and use saved successors at boundaries."""
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


def _stack_rollout(
    steps: list[dict[str, torch.Tensor]],
    first_hidden: torch.Tensor,
    device: Any,
) -> V4RolloutBatch:
    """Assemble frontier collector rows into one recurrent PPO batch.

    Per-step rows carry the singleton time axis used by the forward call;
    stacking squeezes it first so the batch is exactly ``[lanes, decisions,
    ...]``. Rewards and flags are stored one-dimensional per step.
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
            trajectory.values,
            trajectory.terminated,
            trajectory.truncated,
            stack("boundary_values"),
        ),
    )
