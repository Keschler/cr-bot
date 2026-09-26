"""Confidence gates and legal action targets derived from search."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import math

import numpy as np
import torch

from ..model_v4 import V4Logits
from .records import (
    ActionTarget,
    CachedPrefix,
    DEFAULT_MIN_LCB,
    DEFAULT_TEMPERATURE,
    FactorizedTarget,
)


def action_from_record(record: Mapping[str, Any]) -> ActionTarget:
    """Convert a recorded learner action into the rectangular target form."""

    return ActionTarget(
        mode=int(record.get("mode", 0)),
        slot=int(record.get("slot", 0)),
        row=int(record.get("row", 0)),
        col=int(record.get("col", 0)),
    )


def _action_from_branch(branch: Mapping[str, Any]) -> ActionTarget:
    return ActionTarget(
        mode=int(branch.get("mode", 0)),
        slot=int(branch.get("slot", 0)),
        row=int(branch.get("row", 0)),
        col=int(branch.get("col", 0)),
    )


def _branch_key(branch: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return _action_from_branch(branch).key()


def _validation_branches(verdict: Mapping[str, Any]) -> dict[tuple[int, int, int, int], Mapping[str, Any]]:
    """Index an optional independent validation branch collection.

    The frontier evaluator has used both per-branch ``validation_values`` and
    a separate ``validation_branches`` collection while this experiment was
    developed.  Accepting both makes the confidence API explicit without
    silently treating discovery samples as confirmation samples.
    """

    raw = verdict.get("validation_branches", ())
    if isinstance(raw, Mapping):
        raw = raw.values()
    result: dict[tuple[int, int, int, int], Mapping[str, Any]] = {}
    for branch in raw or ():
        if isinstance(branch, Mapping):
            result[_branch_key(branch)] = branch
    return result


def _branch_values(
    branch: Mapping[str, Any], *, independent: bool,
    validation: Mapping[tuple[int, int, int, int], Mapping[str, Any]] | None = None,
) -> list[float]:
    """Read replica values for discovery or independent validation."""

    if independent:
        for key in (
            "confirmation_values", "validation_values", "holdout_values",
            "independent_values", "confirm_values",
        ):
            values = branch.get(key)
            if values:
                return [float(value) for value in values]
        if validation is not None:
            other = validation.get(_branch_key(branch))
            if other is not None:
                for key in (
                    "confirmation_values", "values", "validation_values",
                    "holdout_values", "independent_values",
                ):
                    values = other.get(key)
                    if values:
                        return [float(value) for value in values]
        return []
    for key in ("discovery_values", "values"):
        values = branch.get(key)
        if values:
            return [float(value) for value in values]
    return []


def _branch_mean(
    branch: Mapping[str, Any], *, independent: bool = False,
    validation: Mapping[tuple[int, int, int, int], Mapping[str, Any]] | None = None,
) -> float:
    values = _branch_values(branch, independent=independent, validation=validation)
    if values:
        return float(np.mean(values))
    return float(branch.get("mean", 0.0))


def paired_advantage(verdict: Mapping[str, Any], *, require_independent: bool = False) -> dict[str, Any]:
    """Return paired best-vs-recorded statistics from one verdict.

    The best action is selected from discovery means.  If independent
    validation replicas are present, the frozen action is then scored only on
    those replicas.  This prevents the common ``argmax``-then-confidence
    mistake in which the same samples both choose and certify a correction.
    ``require_independent`` is useful to make data preparation fail closed.
    """

    branches = list(verdict.get("branches", ()))
    sim = next((b for b in branches if b.get("source") == "sim"), None)
    if sim is None:
        raise ValueError("verdict has no sim branch")
    validation = _validation_branches(verdict)
    # Selection is intentionally always made on discovery values/means.
    best = max(branches, key=lambda b: _branch_mean(b))
    best_values = _branch_values(best, independent=True, validation=validation)
    sim_values = _branch_values(sim, independent=True, validation=validation)
    independent = bool(best_values and sim_values and len(best_values) == len(sim_values))
    # Confirmation replicas must be disjoint from discovery replicas.  When
    # the producer records indices, also require the same paired index set for
    # best and sim so the subtraction remains common-random-number paired.
    discovery_indices = set(int(v) for v in verdict.get("discovery_replica_indices", ()))
    discovery_indices.update(int(v) for v in best.get("discovery_replica_indices", ()))
    discovery_indices.update(int(v) for v in sim.get("discovery_replica_indices", ()))
    confirmation_indices_best = tuple(int(v) for v in best.get(
        "confirmation_replica_indices", best.get("validation_replica_indices", verdict.get("confirmation_replica_indices", ()))))
    confirmation_indices_sim = tuple(int(v) for v in sim.get(
        "confirmation_replica_indices", sim.get("validation_replica_indices", verdict.get("confirmation_replica_indices", ()))))
    if require_independent:
        independent = independent and bool(discovery_indices) and bool(confirmation_indices_best)
    if confirmation_indices_best or confirmation_indices_sim:
        independent = independent and bool(confirmation_indices_best)
        independent = independent and confirmation_indices_best == confirmation_indices_sim
        independent = independent and len(confirmation_indices_best) == len(best_values)
        independent = independent and not discovery_indices.intersection(confirmation_indices_best)
    if require_independent and not independent:
        raise ValueError("verdict lacks independent validation replicas for confidence")
    if not independent:
        best_values = _branch_values(best, independent=False)
        sim_values = _branch_values(sim, independent=False)
    if best_values and len(best_values) == len(sim_values):
        diffs = np.asarray(best_values, dtype=np.float64) - np.asarray(sim_values, dtype=np.float64)
        mean = float(diffs.mean())
        # A single paired replica estimates the mean but provides no sample
        # variance. Treat its lower confidence bound as unresolved rather
        # than certifying a correction from one favorable outcome.
        se = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("inf")
    else:
        mean = _branch_mean(best) - _branch_mean(sim)
        se = float(best.get("se", 0.0)) + float(sim.get("se", 0.0))
    return {
        "mean": mean,
        "se": se,
        "lcb95": mean - 1.96 * se,
        "best_mean": float(np.mean(best_values)) if best_values else _branch_mean(best),
        "sim_mean": float(np.mean(sim_values)) if sim_values else _branch_mean(sim),
        "same_action": float(_branch_key(best) == _branch_key(sim)),
        "best_index": float(branches.index(best)),
        "confidence_independent": float(independent),
        "confidence_source": "validation" if independent else "discovery",
        "validation_replicas": float(len(best_values)) if independent else 0.0,
    }


def validated_correction(
    verdict: Mapping[str, Any], *, min_lcb: float = DEFAULT_MIN_LCB,
    require_independent: bool = True,
) -> tuple[bool, ActionTarget | None, dict[str, Any]]:
    """Apply the confidence gate used by the training data builder.

    Discovery-only verdicts are rejected by default.  Callers that knowingly
    run a legacy exploratory ablation may set ``require_independent=False``;
    the returned stats still identify the resulting confidence source.
    """

    try:
        stats = paired_advantage(verdict, require_independent=require_independent)
    except ValueError:
        if require_independent:
            return False, None, {
                "mean": 0.0, "se": float("inf"), "lcb95": -float("inf"),
                "best_mean": 0.0, "sim_mean": 0.0, "same_action": 1.0,
                "best_index": -1.0, "confidence_independent": 0.0,
                "confidence_source": "missing", "validation_replicas": 0.0,
            }
        raise
    branches = list(verdict.get("branches", ()))
    best = max(branches, key=lambda b: _branch_mean(b))
    sim = next(b for b in branches if b.get("source") == "sim")
    different = _branch_key(best) != _branch_key(sim)
    accepted = (
        different
        and (not require_independent or bool(stats.get("confidence_independent")))
        and stats["lcb95"] > float(min_lcb)
    )
    return accepted, (_action_from_branch(best) if accepted else None), stats


def _masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked softmax that returns zeros for rows with no legal entries."""

    masked = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    result = torch.softmax(masked, dim=-1)
    return torch.where(mask.any(dim=-1, keepdim=True), result, torch.zeros_like(result))


def factorized_distributions(
    logits: V4Logits, masks: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return legal mode/card/placement probabilities from V4 logits."""

    mode = _masked_distribution(logits.mode.float(), masks.mode)
    card = _masked_distribution(logits.card.float(), masks.card)
    flat = logits.placement.float().reshape(*logits.placement.shape[:-2], -1)
    flat_mask = masks.placement.reshape(*masks.placement.shape[:-2], -1)
    place = _masked_distribution(flat, flat_mask).reshape_as(logits.placement)
    return mode, card, place


def _normalise(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        if fallback is None:
            fallback = np.ones_like(values)
        values = np.asarray(fallback, dtype=np.float64)
        total = float(values.sum())
    return (values / max(total, np.finfo(np.float64).tiny)).astype(np.float32)


def build_soft_search_target(
    *,
    old_mode: np.ndarray,
    old_card: np.ndarray,
    old_placement: np.ndarray,
    legal_play: np.ndarray,
    branches: Sequence[Mapping[str, Any]],
    baseline: ActionTarget | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    advantage_clip: float = 0.5,
) -> FactorizedTarget:
    """Build a dense ``pi_old * exp(A / temperature)`` target.

    Only tested branches receive a search update.  Untested legal actions
    retain their old prior, preventing a seven-branch search from pretending
    that its incomplete branch set is exhaustive.
    """

    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be positive and finite")
    legal = np.asarray(legal_play, dtype=bool)
    if legal.ndim != 3:
        raise ValueError("legal_play must have shape [slots, rows, cols]")
    mode = _normalise(np.asarray(old_mode, dtype=np.float64))
    card = np.asarray(old_card, dtype=np.float64)
    place = np.asarray(old_placement, dtype=np.float64)
    slots, rows, cols = legal.shape
    if card.shape != (slots,) or place.shape != (slots, rows, cols):
        raise ValueError("old factorized distributions do not match legality")
    card = _normalise(card, legal.reshape(slots, -1).any(axis=1).astype(np.float32))
    for slot in range(slots):
        place[slot] = _normalise(place[slot], legal[slot].astype(np.float32).reshape(-1)).reshape(rows, cols)
        place[slot] *= legal[slot]
        place[slot] = _normalise(place[slot], legal[slot].astype(np.float32)).reshape(rows, cols)

    joint = np.zeros((2, slots, rows, cols), dtype=np.float64)
    joint[0, 0, 0, 0] = float(mode[0])
    for slot in range(slots):
        joint[1, slot] = float(mode[1]) * float(card[slot]) * place[slot]
    q_base = None
    if baseline is not None:
        for branch in branches:
            if _branch_key(branch) == baseline.key():
                q_base = float(branch.get("mean", 0.0))
                break
    if q_base is None:
        q_base = float(next((b.get("mean", 0.0) for b in branches if b.get("source") == "sim"), 0.0))
    for branch in branches:
        action = _action_from_branch(branch)
        if action.mode == 0:
            if joint[0, 0, 0, 0] <= 0.0:
                continue
            index = (0, 0, 0, 0)
        elif 0 <= action.slot < slots and 0 <= action.row < rows and 0 <= action.col < cols and legal[action.slot, action.row, action.col]:
            index = (1, action.slot, action.row, action.col)
        else:
            continue
        q = float(branch.get("mean", q_base))
        multiplier = math.exp(float(np.clip(q - q_base, -advantage_clip, advantage_clip)) / float(temperature))
        joint[index] *= multiplier
    total = float(joint.sum())
    if total <= 0.0:
        raise ValueError("search target has no legal probability mass")
    joint /= total
    target_mode = _normalise(joint.sum(axis=(1, 2, 3)))
    play_mass = float(target_mode[1])
    target_card = _normalise(joint[1].sum(axis=(1, 2)), card)
    target_place = np.zeros((slots, rows, cols), dtype=np.float32)
    for slot in range(slots):
        mass = float(joint[1, slot].sum())
        if mass > 0.0:
            target_place[slot] = (joint[1, slot] / mass).astype(np.float32)
        elif card[slot] > 0.0 and legal[slot].any():
            target_place[slot] = _normalise(legal[slot].astype(np.float32), legal[slot].astype(np.float32)).reshape(rows, cols)
    # ``play_mass`` is intentionally read above: it documents that card and
    # placement remain conditionals.  The target object stores conditionals.
    _ = play_mass
    return FactorizedTarget(target_mode, target_card, target_place)


def hard_factorized_target(action: ActionTarget, legal_play: np.ndarray) -> FactorizedTarget:
    """Create a legal one-hot factorised target."""

    legal = np.asarray(legal_play, dtype=bool)
    slots, rows, cols = legal.shape
    mode = np.zeros(2, dtype=np.float32)
    mode[int(action.mode)] = 1.0
    card = np.zeros(slots, dtype=np.float32)
    place = np.zeros((slots, rows, cols), dtype=np.float32)
    if action.mode == 0:
        card[:] = _normalise(np.zeros(slots, dtype=np.float32), legal.reshape(slots, -1).any(axis=1).astype(np.float32))
        for slot in range(slots):
            if legal[slot].any():
                place[slot] = _normalise(legal[slot].astype(np.float32), legal[slot].astype(np.float32)).reshape(rows, cols)
        return FactorizedTarget(mode, card, place)
    if not (0 <= action.slot < slots and 0 <= action.row < rows and 0 <= action.col < cols and legal[action.slot, action.row, action.col]):
        raise ValueError("hard target action is illegal")
    card[action.slot] = 1.0
    place[action.slot, action.row, action.col] = 1.0
    for slot in range(slots):
        if slot != action.slot and legal[slot].any():
            place[slot] = _normalise(legal[slot].astype(np.float32), legal[slot].astype(np.float32)).reshape(rows, cols)
    return FactorizedTarget(mode, card, place)


def _old_action_from_distributions(dist: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[ActionTarget]:
    mode, card, place = dist
    mode = mode[:, -1].detach().cpu().numpy()
    card = card[:, -1].detach().cpu().numpy()
    place = place[:, -1].detach().cpu().numpy()
    result = []
    for i in range(len(mode)):
        m = int(np.argmax(mode[i]))
        if m == 0:
            result.append(ActionTarget(0))
            continue
        slot = int(np.argmax(card[i]))
        cell = int(np.argmax(place[i, slot]))
        result.append(ActionTarget(m, slot, cell // place.shape[-1], cell % place.shape[-1]))
    return result


def _record_soft_target(record: CachedPrefix, old: tuple[torch.Tensor, torch.Tensor, torch.Tensor], index: int, temperature: float) -> FactorizedTarget:
    old_mode = old[0][index, -1].detach().cpu().numpy()
    old_card = old[1][index, -1].detach().cpu().numpy()
    old_place = old[2][index, -1].detach().cpu().numpy()
    baseline = record.recorded_action
    return build_soft_search_target(
        old_mode=old_mode, old_card=old_card, old_placement=old_place,
        legal_play=record.legal_play[-1], branches=record.branches,
        baseline=baseline, temperature=temperature,
    )
