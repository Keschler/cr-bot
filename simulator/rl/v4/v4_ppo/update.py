"""Optimize the actor and detached critic, then measure the candidate."""

from __future__ import annotations

from typing import Any
import torch
from .runtime import configure_v4_runtime, _forward_sequence
from ..model_v4 import RecurrentV4Policy, V4ValueHead
from ...foundation.objectives import PPOObjectiveConfig, behavior_cloning_loss, compute_gae, ppo_objective
from .records import V4RolloutBatch, V4UpdateReport
from .config import V4_RUNTIME_VERSION
from .diagnostics import _head_grad_norms, _term_grad_norm, build_update_report


def update_v4_ppo(
    policy: RecurrentV4Policy,
    critic: V4ValueHead,
    actor_optimizer: Any,
    critic_optimizer: Any,
    batch: V4RolloutBatch,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ppo_config: PPOObjectiveConfig | None = None,
    bc_coef: float = 0.05,
    ppo_epochs: int = 1,
) -> V4UpdateReport:
    """Run PPO epochs over one collected rollout.

    Advantages come from :func:`compute_gae` with critic bootstrapping on
    nonterminal truncations.  The critic trains on detached actor features,
    so value updates never move actor parameters (isolation is asserted by
    tests).  Entropy uses the current policy (not stored values).  Returns
    the final epoch's diagnostics plus quarantine flags; the caller decides
    whether to keep the updated weights.
    """

    objective_config = ppo_config or PPOObjectiveConfig(bc_coef=bc_coef)
    if type(ppo_epochs) is not int or ppo_epochs < 1:
        raise ValueError("ppo_epochs must be a positive integer")
    trajectory = batch.trajectory
    sequence = trajectory.sequence
    device = sequence.raster.device
    # Keep dropout identical to collection while retaining autograd.
    policy.eval()
    critic.train()
    if batch.recurrent is None:
        raise ValueError("rollout batch lacks stored recurrent features")
    if batch.runtime_version != V4_RUNTIME_VERSION:
        raise ValueError("rollout runtime version mismatch; recollect with the current runtime")
    configure_v4_runtime(policy)
    if batch.next_values is None:
        raise ValueError("rollout lacks real successor values; recollect instead of shifting across resets")
    next_values = batch.next_values.to(device)
    advantages, returns = compute_gae(
        trajectory.rewards.to(device),
        trajectory.values.to(device),
        next_values,
        trajectory.terminated.to(device),
        trajectory.truncated.to(device),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    report: V4UpdateReport | None = None
    with torch.no_grad():
        pre_logits, _, _ = _forward_sequence(policy, sequence, device)
        pre_decoded = policy.act_deterministic(pre_logits, trajectory.action_masks)
    for _ in range(ppo_epochs):
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        logits, ht, _ = _forward_sequence(policy, sequence, device)
        new_log_probs = policy.log_prob(logits, trajectory.action_masks, trajectory.actions)
        values = critic(ht.detach())
        anchor_log_probs: torch.Tensor | None = None
        if batch.anchor_actions is not None and float(objective_config.bc_coef) > 0.0:
            anchor_log_probs = policy.log_prob(
                logits, trajectory.action_masks, batch.anchor_actions
            )
        new_entropy = policy.action_entropy(logits, trajectory.action_masks)["joint"]
        result = ppo_objective(
            old_log_probs=trajectory.old_log_probs.to(device),
            new_log_probs=new_log_probs,
            advantages=advantages,
            values=values,
            returns=returns,
            entropy=new_entropy,
            old_values=trajectory.values.to(device),
            behavior_cloning_log_probs=anchor_log_probs,
            config=objective_config,
        )
        result.total_loss.backward(retain_graph=True)
        grad_norms = _head_grad_norms(policy, critic)
        # Per-term actor gradient norms answer "PPO vs BC: who is actually
        # moving the policy?"  Separate autograd passes (graph retained
        # above); the stepping backward below reuses it a final time.
        actor_params = [p for p in policy.parameters() if p.requires_grad]
        policy_grad_norm = _term_grad_norm(result.policy_loss, actor_params)
        if anchor_log_probs is not None:
            bc_measured = behavior_cloning_loss(
                policy.log_prob(logits, trajectory.action_masks, batch.anchor_actions),
                torch.ones_like(anchor_log_probs),
            )
            bc_grad_norm = _term_grad_norm(bc_measured, actor_params)
        else:
            bc_grad_norm = 0.0
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        result.total_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        report = build_update_report(
            policy, batch, pre_decoded, result, values, returns, advantages,
            objective_config, grad_norms, policy_grad_norm, bc_grad_norm,
        )
    assert report is not None
    return report

