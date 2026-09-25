"""Return summaries, potential annealing, and isolated critic fitting."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from ..model_v4 import V4ValueHead
from .diagnostics import _explained_variance


def potential_anneal_weight(
    initial_weight: float, update: int, n_updates: int, *, anneal: bool = True
) -> float:
    """Potential-shaping weight for one smoke update (sparse backbone).

    Annealed linearly from ``initial_weight`` to exactly 0.0 at the final
    update so training ends on the sparse terminal objective.  With
    ``anneal=False`` the weight stays constant (ablation only).
    """

    if not np.isfinite(float(initial_weight)) or float(initial_weight) < 0.0:
        raise ValueError("initial_weight must be a finite non-negative number")
    if type(update) is not int or update < 0:
        raise ValueError("update must be a non-negative integer")
    if type(n_updates) is not int or n_updates <= 0:
        raise ValueError("n_updates must be a positive integer")
    if type(anneal) is not bool:
        raise TypeError("anneal must be boolean")
    if not anneal:
        return float(initial_weight)
    return max(0.0, float(initial_weight) * (1.0 - update / max(1, n_updates - 1)))


def monte_carlo_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rewards-to-go with zero bootstrap at rollout cuts.

    Terminated episodes yield exact returns.  Trailing nonterminal tails
    bootstrap zero, which biases them toward zero; the returned
    ``truncated_tail_fraction`` quantifies how much of the data is
    affected.  If it dominates, the honest fix is longer horizons, not a
    bigger model.
    """

    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    rewards = rewards.detach().float()
    terminated = terminated.detach().to(torch.bool)
    batch, time = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    for step in range(time - 1, -1, -1):
        running = rewards[:, step] + float(gamma) * running * (~terminated[:, step]).to(
            rewards.dtype
        )
        # Keep this reward for earlier actions in the same episode.  The
        # terminal mask above already prevents bootstrapping across resets.
        out[:, step] = running
    # Steps biased by the zero bootstrap: nonterminal-final lanes, steps
    # strictly before the cut edge and after the lane's last termination.
    biased = 0
    for lane in range(batch):
        terms = torch.nonzero(terminated[lane], as_tuple=False).reshape(-1)
        last_term = int(terms.max().item()) if int(terms.numel()) > 0 else -1
        if not bool(terminated[lane, -1].item()):
            biased += max(0, (time - 1) - max(last_term + 1, 0))
    stats = {
        "truncated_tail_fraction": biased / max(1, batch * time),
    }
    return out, stats


def return_distribution(values: torch.Tensor) -> dict[str, float]:
    """Quantile/sparsity summary of a return (or reward) tensor."""

    flat = values.detach().float().reshape(-1)
    if int(flat.numel()) == 0:
        raise ValueError("empty tensor has no distribution")
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            dtype=flat.dtype,
            device=flat.device,
        ),
    ).tolist()
    near_zero = float((flat.abs() < 1e-6).float().mean().item())
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "near_zero_fraction": near_zero,
        "positive_fraction": float((flat > 1e-6).float().mean().item()),
        "negative_fraction": float((flat < -1e-6).float().mean().item()),
    }


def fit_value_head(
    critic: V4ValueHead,
    recurrent: torch.Tensor,
    targets: torch.Tensor,
    *,
    heldout_recurrent: torch.Tensor | None = None,
    heldout_targets: torch.Tensor | None = None,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Supervised critic warmup on fixed Monte-Carlo targets (actor frozen).

    Only the value head trains, on caller-provided recurrent features, so
    this cannot move the actor by construction.  Tracks train and held-out
    loss plus explained variance every epoch and returns the best-held-out
    state (in-memory copy) for the caller to persist.
    """

    generator = torch.Generator().manual_seed(int(seed))
    try:
        device = next(critic.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    recurrent = recurrent.detach().float().to(device)
    targets = targets.detach().float().to(device)
    if recurrent.shape[:2] != targets.shape[:2]:
        raise ValueError("recurrent features and targets must share batch/time")
    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(lr))
    flat_x = recurrent.reshape(-1, recurrent.shape[-1])
    flat_y = targets.reshape(-1)
    if heldout_recurrent is not None and heldout_targets is not None:
        held_x = heldout_recurrent.detach().float().to(device).reshape(-1, recurrent.shape[-1])
        held_y = heldout_targets.detach().float().to(device).reshape(-1)
    else:
        held_x, held_y = None, None
    best = {"heldout_loss": float("inf"), "epoch": -1, "state": None}
    history: list[dict[str, float]] = []
    order_base = torch.randperm(flat_x.shape[0], generator=generator)
    for epoch in range(int(epochs)):
        order = order_base[torch.randperm(flat_x.shape[0], generator=generator)]
        for start in range(0, int(flat_x.shape[0]), 256):
            chunk = order[start : start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss = (critic(flat_x[chunk].unsqueeze(1)).squeeze(1) - flat_y[chunk]).square().mean()
            loss.backward()
            optimizer.step()
        critic.eval()
        with torch.no_grad():
            train_pred = critic(flat_x.unsqueeze(1)).squeeze(1)
            train_loss = float(((train_pred - flat_y).square()).mean().item())
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_ev": _explained_variance(flat_y, train_pred.reshape_as(flat_y)),
            }
            if held_x is not None:
                held_pred = critic(held_x.unsqueeze(1)).squeeze(1)
                held_loss = float(((held_pred - held_y).square()).mean().item())
                row["heldout_loss"] = held_loss
                row["heldout_ev"] = _explained_variance(held_y, held_pred.reshape_as(held_y))
                if held_loss < best["heldout_loss"]:
                    best = {
                        "heldout_loss": held_loss,
                        "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
                    }
        critic.train()
        history.append(row)
    if best["state"] is not None:
        critic.load_state_dict(best["state"])
    return {"history": history, "best": best}

