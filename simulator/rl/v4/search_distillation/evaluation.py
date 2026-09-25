"""Evaluate selected actions with independent paired simulator outcomes."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import json
import math

import numpy as np
import torch

from ..frontier_eval import _stable_seed
from ..model_v4 import RecurrentV4Policy
from .artifacts import DistillationDeadline, _check_deadline
from .batching import _forward, _stack_records
from .losses import _factorized_kl
from .records import ActionTarget, CachedPrefix
from .targets import _branch_mean, _old_action_from_distributions, factorized_distributions


def _paired_outcome_evaluation(
    record: CachedPrefix,
    selected_action: ActionTarget,
    policy: RecurrentV4Policy,
    *,
    device: torch.device,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    """Evaluate a selected action and the recorded action on paired replicas.

    Discovery branches are incomplete by design.  A policy can select an
    action absent from those branches, so using the branch table would turn a
    real outcome into a missing value.  This helper replays the exact source
    state and evaluates both actions with the same replica seeds.  It is kept
    best effort for legacy caches without ``match_path``; the report then
    exposes incomplete coverage instead of dropping the row silently.
    """

    if not record.match_path:
        return {
            "evaluated": False, "coverage": 0.0, "reason": "source_match_unavailable",
            "n": 0, "mean": None, "se": None, "values": [],
        }
    try:
        match_path = Path(record.match_path)
        match = json.loads(match_path.read_text(encoding="utf-8"))
        from ..frontier_eval import BranchSpec, replay_to_state, run_branch

        _check_deadline(deadline_epoch)
        env, hidden = replay_to_state(match, int(record.offset), policy, device)
        config = match.get("config", {})
        opponent_spec = dict(record.opponent_spec or {
            "strategy": str(config.get("strategy", "wait")),
            "controller_seed": int(config.get("controller_seed", 0)),
        })
        replicas = max(1, int(record.outcome_replicas))
        horizon = max(1, int(record.outcome_horizon))
        seed_base = int(record.outcome_replica_seed_base)
        selected_branch = BranchSpec(
            source="selected", mode=selected_action.mode, slot=selected_action.slot,
            row=selected_action.row, col=selected_action.col,
        )
        recorded = record.recorded_action
        recorded_branch = BranchSpec(
            source="recorded", mode=recorded.mode, slot=recorded.slot,
            row=recorded.row, col=recorded.col,
        )
        values: list[float] = []
        selected_values: list[float] = []
        recorded_values: list[float] = []
        state_hash = str(match.get("decisions", [])[int(record.offset)].get("state_hash", record.key))
        for replica in range(replicas):
            _check_deadline(deadline_epoch)
            replica_seed = _stable_seed(
                "search-distillation-outcome", seed_base, state_hash, replica,
            ) & 0xFFFFFFFF
            selected_out = run_branch(
                env=env, hidden=hidden, branch=selected_branch,
                opponent_spec=opponent_spec, replica_seed=replica_seed,
                horizon=horizon, policy=policy, device=device,
            )
            _check_deadline(deadline_epoch)
            recorded_out = run_branch(
                env=env, hidden=hidden, branch=recorded_branch,
                opponent_spec=opponent_spec, replica_seed=replica_seed,
                horizon=horizon, policy=policy, device=device,
            )
            selected_value = float(selected_out["value"])
            recorded_value = float(recorded_out["value"])
            selected_values.append(selected_value)
            recorded_values.append(recorded_value)
            values.append(selected_value - recorded_value)
        array = np.asarray(values, dtype=np.float64)
        return {
            "evaluated": True, "coverage": 1.0, "reason": "paired_research",
            "n": int(len(values)), "mean": float(array.mean()),
            "se": float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
            "values": values, "selected_values": selected_values,
            "recorded_values": recorded_values,
        }
    except DistillationDeadline:
        raise
    except Exception as exc:  # pragma: no cover - simulator/model dependent
        return {
            "evaluated": False, "coverage": 0.0,
            "reason": f"evaluation_error:{type(exc).__name__}",
            "n": 0, "mean": None, "se": None, "values": [],
        }


def evaluate_records(
    policy: RecurrentV4Policy,
    anchor: RecurrentV4Policy,
    records: Sequence[CachedPrefix],
    *,
    context: int,
    batch_size: int,
    device: torch.device,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    """Evaluate action agreement, KL drift, and paired outcome coverage."""

    policy.eval()
    anchor.eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        _check_deadline(deadline_epoch)
        group = list(records[start:start + batch_size])
        batch = _stack_records(group, context=context, device=device)
        with torch.no_grad():
            new_logits, masks = _forward(policy, batch)
            old_logits, _ = _forward(anchor, batch)
            new_dist = factorized_distributions(new_logits, masks)
            old_dist = factorized_distributions(old_logits, masks)
        old_actions = _old_action_from_distributions(old_dist)
        new_actions = _old_action_from_distributions(new_dist)
        kl = _factorized_kl(old_dist, new_dist, masks).detach().item()
        for record, old_action, new_action in zip(group, old_actions, new_actions):
            _check_deadline(deadline_epoch)
            row = {
                "entry_id": record.key,
                "old_action": list(old_action.key()),
                "new_action": list(new_action.key()),
                "same_action": bool(old_action.key() == new_action.key()),
                "is_correction": bool(record.is_correction),
                "paired_mean": float(record.paired_mean),
                "correction_lcb": float(record.correction_lcb),
                "kl_batch": float(kl),
            }
            outcome = _paired_outcome_evaluation(
                # Keep continuation policy fixed to the frozen anchor.  The
                # measured delta then answers whether the selected first
                # action itself is better, rather than mixing in a different
                # continuation policy for every ablation.
                record, new_action, anchor, device=device, deadline_epoch=deadline_epoch,
            )
            row["outcome_evaluated"] = bool(outcome["evaluated"])
            row["outcome_coverage"] = float(outcome["coverage"])
            row["outcome_reason"] = str(outcome["reason"])
            row["paired_outcome_delta"] = outcome["mean"]
            row["paired_outcome_se"] = outcome["se"]
            if record.is_correction:
                # The source best action was selected on discovery replicas.
                # If the student's selected action was absent from that branch
                # set, infer its regret from its independently re-searched
                # paired delta against the recorded action.  This keeps every
                # selected action in the metric denominator.
                best = max(record.branches, key=lambda b: _branch_mean(b)) if record.branches else None
                sim = next((b for b in record.branches if b.get("source") == "sim"), None)
                if outcome["evaluated"] and best is not None and sim is not None:
                    selected_mean = _branch_mean(sim) + float(outcome["mean"])
                    row["search_regret"] = float(_branch_mean(best) - selected_mean)
                else:
                    row["search_regret"] = None
                row["correction_target_match"] = bool(record.correction_action and new_action.key() == record.correction_action.key())
            rows.append(row)
    policy.train()
    corrections = [r for r in rows if r["is_correction"]]
    retention = [r for r in rows if not r["is_correction"]]
    regrets = [float(r["search_regret"]) for r in corrections if r["search_regret"] is not None]
    outcome_rows = [r for r in rows if bool(r.get("outcome_evaluated"))]
    outcome_deltas = [float(r["paired_outcome_delta"]) for r in outcome_rows if r.get("paired_outcome_delta") is not None]
    correction_outcome_deltas = [
        float(r["paired_outcome_delta"]) for r in corrections
        if bool(r.get("outcome_evaluated")) and r.get("paired_outcome_delta") is not None
    ]
    retention_outcome_deltas = [
        float(r["paired_outcome_delta"]) for r in retention
        if bool(r.get("outcome_evaluated")) and r.get("paired_outcome_delta") is not None
    ]
    return {
        "n": len(rows),
        "n_corrections": len(corrections),
        "n_retention": len(retention),
        "correction_target_match": sum(bool(r.get("correction_target_match")) for r in corrections) / max(1, len(corrections)),
        "correction_action_regret_mean": float(np.mean(regrets)) if regrets else None,
        "correction_action_regret_median": float(np.median(regrets)) if regrets else None,
        "correction_regret_coverage": len(regrets) / max(1, len(corrections)),
        "retention_action_agreement": sum(bool(r["same_action"]) for r in retention) / max(1, len(retention)),
        "retention_outcome_delta_mean": float(np.mean(retention_outcome_deltas)) if retention_outcome_deltas else None,
        "retention_outcome_coverage": len(retention_outcome_deltas) / max(1, len(retention)),
        "paired_outcome_delta_mean": float(np.mean(outcome_deltas)) if outcome_deltas else None,
        "paired_outcome_coverage": len(outcome_rows) / max(1, len(rows)),
        "paired_outcome_coverage_complete": len(outcome_rows) == len(rows),
        "action_change_count": sum(not bool(r["same_action"]) for r in rows),
        "kl_mean": float(np.mean([float(r["kl_batch"]) for r in rows])) if rows else 0.0,
        "rows": rows,
    }

