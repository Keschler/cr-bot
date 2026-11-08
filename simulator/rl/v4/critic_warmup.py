"""Fit only the value head on genuinely completed frozen-actor episodes."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

CRITIC_WARMUP_VERSION = "completed-episode-value-warmup-1"


def completed_return_targets(rewards: Any, terminated: Any, truncated: Any,
                             reset_mask: Any, *, gamma: float, valid_mask: Any = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Discount terminal outcomes backward without labeling censored prefixes.

    A supervised segment must begin at a genuine reset and finish with native
    termination in this batch. Truncation and unfinished tails receive no target.
    """
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be finite and in (0, 1]")
    rewards = np.asarray(rewards, dtype=np.float64)
    flags = [np.asarray(value) for value in (terminated, truncated, reset_mask)]
    if rewards.ndim != 2 or any(value.shape != rewards.shape or value.dtype != np.bool_ for value in flags):
        raise ValueError("rewards and boolean boundaries must share [lane,time] shape")
    if not np.isfinite(rewards).all() or np.any(flags[0] & flags[1]):
        raise ValueError("nonfinite reward or ambiguous termination/truncation")
    terminated, truncated, reset_mask = flags
    valid = np.ones(rewards.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != rewards.shape or valid.dtype != np.bool_:
        raise ValueError("valid mask must be boolean with the reward sequence shape")
    targets, selected = np.zeros_like(rewards), np.zeros(rewards.shape, dtype=bool)
    episodes = 0
    for lane in range(len(rewards)):
        start = None
        for step in range(rewards.shape[1]):
            if not valid[lane, step]:
                start = None
                continue
            if reset_mask[lane, step]:
                start = step
            if truncated[lane, step]:
                start = None
            elif terminated[lane, step]:
                if start is not None:
                    value = 0.
                    for position in range(step, start - 1, -1):
                        value = float(rewards[lane, position]) + gamma * value
                        if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
                            raise ValueError("discounted return is nonfinite or outside float32 range")
                        targets[lane, position] = value
                        selected[lane, position] = True
                    episodes += 1
                start = None
    return targets.astype(np.float32), selected, episodes


def _policy_digest(policy: Any) -> str:
    result = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        result.update(name.encode() + b"\0")
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _dataset(batch: Any, gamma: float):
    import torch
    if batch.recurrent is None:
        raise ValueError("critic warmup requires stored recurrent features")
    trajectory = batch.trajectory
    targets, mask, episodes = completed_return_targets(
        trajectory.rewards.detach().cpu().numpy(), trajectory.terminated.detach().cpu().numpy(),
        trajectory.truncated.detach().cpu().numpy(), trajectory.sequence.reset_mask.detach().cpu().numpy(),
        gamma=gamma, valid_mask=(None if trajectory.valid_mask is None
                                else trajectory.valid_mask.detach().cpu().numpy()))
    features = batch.recurrent.detach().cpu()
    if tuple(features.shape[:2]) != tuple(mask.shape):
        raise ValueError("critic features and terminal target sequence differ")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("critic features must be finite")
    terminal_rewards = trajectory.rewards.detach().cpu().numpy()[mask & trajectory.terminated.detach().cpu().numpy()]
    outcomes = dict(wins=int((terminal_rewards > 0).sum()), losses=int((terminal_rewards < 0).sum()),
                    draws=int((terminal_rewards == 0).sum()))
    return features[torch.as_tensor(mask)].unsqueeze(1), torch.as_tensor(targets[mask]), episodes, outcomes


def _metrics(critic: Any, features: Any, targets: Any) -> dict[str, Any]:
    import torch
    device = next(critic.parameters()).device
    with torch.no_grad():
        predictions = critic(features.to(device)).reshape(-1).cpu()
    residual = targets - predictions
    variance = float(targets.var(unbiased=False))
    return dict(rows=len(targets), mse=float(residual.square().mean()),
                explained_variance=None if variance < 1e-12 else 1 - float(residual.var(unbiased=False)) / variance,
                target_variance=variance, target_mean=float(targets.mean()))


def warmup_critic(*, policy: Any, critic: Any, optimizer: Any, batch: Any,
                  validation_batch: Any | None = None, epochs: int = 8,
                  batch_size: int = 256, gamma: float = .999, seed: int = 0) -> dict[str, Any]:
    """Fixed-dose value fitting; never changes the actor or treats cuts as draws."""
    import torch
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    before = _policy_digest(policy)
    if {id(value) for value in policy.parameters()} & {id(value) for value in critic.parameters()}:
        raise ValueError("critic must not share actor parameters")
    features, targets, episodes, outcomes = _dataset(batch, gamma)
    if not len(targets):
        raise ValueError("no complete reset-to-terminal episode is available for critic warmup")
    initial = _metrics(critic, features, targets)
    validation_data = None if validation_batch is None else _dataset(validation_batch, gamma)
    initial_validation = (_metrics(critic, *validation_data[:2])
                          if validation_data is not None and len(validation_data[1]) else None)
    critic.train()
    generator = torch.Generator().manual_seed(seed)
    device = next(critic.parameters()).device
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            values = critic(features[indices].to(device)).reshape(-1)
            loss = (values - targets[indices].to(device)).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("critic warmup loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            optimizer.step()
            steps += 1
    critic.eval()
    final = _metrics(critic, features, targets)
    validation = None
    validation_episodes = 0
    validation_outcomes = None
    if validation_data is not None:
        x, y, validation_episodes, validation_outcomes = validation_data
        if len(y):
            validation = _metrics(critic, x, y)
    if _policy_digest(policy) != before:
        raise RuntimeError("critic warmup changed actor weights")
    return dict(version=CRITIC_WARMUP_VERSION, complete_train_episodes=episodes,
                complete_validation_episodes=validation_episodes, epochs=epochs,
                optimizer_steps=steps, gamma=gamma, initial=initial, final=final,
                initial_validation=initial_validation, validation=validation,
                train_outcomes=outcomes, validation_outcomes=validation_outcomes, actor_unchanged=True,
                limitation="Value fit on correlated frozen-policy states; held-out value prediction and fresh match outcomes still required.")


def critic_warmup_gate(report: dict[str, Any], *, minimum_episodes: int = 4,
                       minimum_validation_ev: float = .05) -> dict[str, Any]:
    """A declared engineering gate, not a claim of statistically proven skill."""
    if type(minimum_episodes) is not int or minimum_episodes < 1:
        raise ValueError("minimum_episodes must be a positive integer")
    if not math.isfinite(minimum_validation_ev) or not 0 <= minimum_validation_ev <= 1:
        raise ValueError("minimum_validation_ev must be finite and in [0, 1]")
    reasons = []
    if report.get("actor_unchanged") is not True:
        reasons.append("actor_weights_changed")
    for split in ("train", "validation"):
        if report[f"complete_{split}_episodes"] < minimum_episodes:
            reasons.append(f"too_few_complete_{split}_episodes")
        outcomes = report.get(f"{split}_outcomes") or {}
        if not outcomes.get("wins") or not outcomes.get("losses"):
            reasons.append(f"{split}_lacks_win_loss_coverage")
    initial, final = report.get("initial_validation"), report.get("validation")
    if (final is None or final.get("explained_variance") is None
            or not math.isfinite(final["explained_variance"])
            or final["explained_variance"] < minimum_validation_ev):
        reasons.append("validation_explained_variance_below_gate")
    if (initial is None or final is None or not math.isfinite(initial["mse"])
            or not math.isfinite(final["mse"]) or not final["mse"] < initial["mse"]):
        reasons.append("validation_mse_did_not_improve")
    return dict(eligible=not reasons, reasons=reasons, minimum_episodes=minimum_episodes,
                minimum_validation_ev=minimum_validation_ev)
