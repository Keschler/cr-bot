"""Search-policy distillation experiments for the V4 actor.

This module is deliberately separate from :mod:`search_dataset` and the PPO
runner.  Frontier entries contain simulator snapshots and verdicts, rather
than materialised actor tensors.  The helpers here replay each source match
once, cache public observations for the requested offsets, and train the
actor on either the target observation alone or a fixed recurrent burn-in
prefix.

The experiment has four intentionally small ablations:

``hard-none``
    Hard search-best correction targets only.
``hard-replay``
    Hard corrections plus replay of the recorded actor action.
``soft-replay``
    Search-value-weighted factorised targets plus recorded-action replay.
``soft-kl``
    Search-value-weighted targets plus KL retention to the frozen anchor.

The module does not change the public V4 model or loss APIs.  It is an
experiment harness, so all target distributions are explicit and saved in
the run manifest/checkpoint metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .frontier import frontier_model_config
from .frontier_eval import (
    _fresh_env,
    _observe_inputs,
    _recorded_action,
    _stable_seed,
    _step_inputs,
    load_frontier_state,
)
from .frontier_ppo import make_frontier_env
from .model_v4 import RecurrentV4Policy, V4Logits, masks_from_legal_play


SEARCH_DISTILLATION_VERSION = "search-distillation-v2"
DEFAULT_CONTEXT = 32
DEFAULT_TEMPERATURE = 0.25
DEFAULT_MIN_LCB = 0.05
_ENTRY_RE = re.compile(r"^(?P<split>[^:]+):(?P<match>.+):t(?P<t>[0-9]+)$")


class DistillationDeadline(RuntimeError):
    """Raised internally when a run reaches its externally supplied deadline."""


def _deadline_reached(deadline_epoch: float | None) -> bool:
    return deadline_epoch is not None and time.time() >= float(deadline_epoch)


def _check_deadline(deadline_epoch: float | None) -> None:
    if _deadline_reached(deadline_epoch):
        raise DistillationDeadline(f"distillation deadline reached at {time.time():.3f}")


@dataclass(frozen=True, slots=True)
class ActionTarget:
    """A rectangular V4 action used by the experiment harness."""

    mode: int
    slot: int = 0
    row: int = 0
    col: int = 0

    def key(self) -> tuple[int, int, int, int]:
        return (int(self.mode), int(self.slot), int(self.row), int(self.col))


@dataclass(frozen=True, slots=True)
class FactorizedTarget:
    """A legal factorised action distribution.

    ``card`` is conditional on PLAY and ``placement`` is conditional on each
    card slot.  WAIT has no meaningful card or placement distribution.
    """

    mode: np.ndarray
    card: np.ndarray
    placement: np.ndarray

    def __post_init__(self) -> None:
        mode = np.asarray(self.mode, dtype=np.float32)
        card = np.asarray(self.card, dtype=np.float32)
        placement = np.asarray(self.placement, dtype=np.float32)
        if mode.shape != (2,):
            raise ValueError("mode target must have shape [2]")
        if card.ndim != 1:
            raise ValueError("card target must be one-dimensional")
        if placement.shape[:1] != card.shape or placement.ndim != 3:
            raise ValueError("placement target must have shape [slots, rows, cols]")
        if not np.isfinite(mode).all() or not np.isfinite(card).all() or not np.isfinite(placement).all():
            raise ValueError("target distributions must be finite")
        if np.any(mode < -1e-6) or np.any(card < -1e-6) or np.any(placement < -1e-6):
            raise ValueError("target distributions must be non-negative")
        if not np.isclose(float(mode.sum()), 1.0, atol=2e-4):
            raise ValueError("mode target must sum to one")
        if mode[1] > 1e-6 and not np.isclose(float(card.sum()), 1.0, atol=2e-4):
            raise ValueError("PLAY card target must sum to one")
        for slot in range(card.shape[0]):
            if card[slot] > 1e-6 and not np.isclose(float(placement[slot].sum()), 1.0, atol=2e-4):
                raise ValueError("each active placement target must sum to one")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "card", card)
        object.__setattr__(self, "placement", placement)


@dataclass(slots=True)
class CachedPrefix:
    """Public observations around one source-match decision."""

    key: str
    match_id: str
    offset: int
    observations: list[dict[str, np.ndarray]]
    legal_play: list[np.ndarray]
    legal_wait: list[bool]
    recorded_action: ActionTarget
    correction_action: ActionTarget | None = None
    branches: list[dict[str, Any]] = field(default_factory=list)
    paired_mean: float = 0.0
    paired_se: float = 0.0
    correction_lcb: float = 0.0
    is_correction: bool = False
    # Optional source metadata used by independent paired outcome evaluation.
    # Keeping paths and small config values instead of whole matches makes the
    # prefix cache remain compact while allowing validation to revisit an
    # action that was absent from the discovery branch set.
    match_path: str | None = None
    opponent_spec: dict[str, Any] | None = None
    outcome_horizon: int = 128
    outcome_replicas: int = 4
    outcome_replica_seed_base: int = 0

    @property
    def current(self) -> dict[str, np.ndarray]:
        return self.observations[-1]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    context: int = DEFAULT_CONTEXT

    def __post_init__(self) -> None:
        if type(self.context) is not int or self.context < 0:
            raise ValueError("context must be a non-negative integer")


def parse_entry_id(entry_id: str) -> tuple[str, str, int]:
    """Parse ``train:archetype#0:t247`` without relying on filenames."""

    match = _ENTRY_RE.match(str(entry_id))
    if match is None:
        raise ValueError(f"invalid frontier entry id: {entry_id!r}")
    return match.group("split"), match.group("match"), int(match.group("t"))


def match_json_path(root: Path, split: str, match_id: str) -> Path:
    """Resolve a materialized match JSON path."""

    path = root / f"{split}_{match_id.replace('#', '_')}_match.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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
        se = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
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


def _copy_input(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def _required_observation_inputs(inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = (
        "raster", "global_features", "entities", "entity_mask", "hand_tokens",
        "opp_hand_probs", "opp_out_of_cycle", "opp_elixir_interval", "event_history",
    )
    out: dict[str, np.ndarray] = {}
    for name in required:
        value = np.asarray(inputs[name])
        out[name] = np.ascontiguousarray(value.copy())
    return out


def replay_selected_prefixes(
    match: Mapping[str, Any],
    offsets: Sequence[int],
    *,
    context: int = DEFAULT_CONTEXT,
) -> dict[int, list[dict[str, np.ndarray]]]:
    """Replay one source match and return fixed-length public input prefixes."""

    if context < 0:
        raise ValueError("context must be non-negative")
    result, _, _ = _replay_selected_prefixes_with_legal(
        match, offsets, context=context
    )
    return result


def _sample_offsets_for_match(
    match: Mapping[str, Any],
    *,
    excluded: set[int],
    count: int,
    context: int,
    seed: int,
) -> list[int]:
    candidates = [
        int(record["t"])
        for record in match["decisions"]
        if int(record["t"]) >= context and int(record["t"]) not in excluded
    ]
    rng = np.random.default_rng(seed)
    if len(candidates) <= count:
        return candidates
    chosen = rng.choice(np.asarray(candidates), size=count, replace=False)
    return sorted(int(v) for v in chosen)


def _match_id_from_json(path: Path, match: Mapping[str, Any]) -> tuple[str, str] | None:
    """Recover the split/match key used by frontier entry ids.

    Materialised match files carry the archetype and match index in their
    config, so this remains correct even when an archetype itself contains an
    underscore.  The filename split is used only to keep train and sealed
    pools separate.
    """

    name = path.name
    suffix = "_match.json"
    if not name.endswith(suffix):
        return None
    split = name[:-len(suffix)].split("_", 1)[0]
    config = match.get("config", {})
    archetype = config.get("archetype")
    index = config.get("match_index")
    if split and archetype is not None and index is not None:
        return str(split), f"{archetype}#{index}"
    return None


def _load_verdict(root: Path, split: str, match_id: str, offset: int) -> dict[str, Any]:
    path = root / f"{split}_{match_id.replace('#', '_')}_t{offset}_verdict.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_frontier_entries(path: Path) -> list[dict[str, Any]]:
    """Load a frontier entry checkpoint and normalize it to plain dicts."""

    entries = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(entries, list):
        raise ValueError(f"frontier entries must be a list: {path}")
    return [dict(entry) for entry in entries]


def prepare_records(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    context: int = DEFAULT_CONTEXT,
    min_lcb: float = DEFAULT_MIN_LCB,
    retention_count: int = 0,
    seed: int = 0,
) -> list[CachedPrefix]:
    """Build correction and ordinary replay records from one split."""

    if context < 0:
        raise ValueError("context must be non-negative")
    by_match: dict[str, dict[str, Any]] = {}
    selected_offsets: dict[str, set[int]] = {}
    corrections: list[tuple[dict[str, Any], str, int, dict[str, Any], ActionTarget, dict[str, float]]] = []
    for raw in entries:
        entry = dict(raw)
        split, match_id, offset = parse_entry_id(str(entry["entry_id"]))
        match = by_match.setdefault(match_id, json.loads(match_json_path(root, split, match_id).read_text(encoding="utf-8")))
        verdict = _load_verdict(root, split, match_id, offset)
        accepted, target, stats = validated_correction(verdict, min_lcb=min_lcb)
        selected_offsets.setdefault(match_id, set()).add(offset)
        if accepted and target is not None:
            corrections.append((entry, match_id, offset, verdict, target, stats))
    if retention_count > 0:
        per_match = max(1, int(math.ceil(retention_count / max(1, len(by_match)))))
        for match_id, match in sorted(by_match.items()):
            excluded = selected_offsets.get(match_id, set())
            offsets = _sample_offsets_for_match(
                match, excluded=excluded, count=per_match, context=context,
                seed=hash((seed, match_id)) & 0xFFFFFFFF,
            )
            selected_offsets.setdefault(match_id, set()).update(offsets)
    prefixes: dict[str, dict[int, list[dict[str, np.ndarray]]]] = {}
    for match_id, offsets in selected_offsets.items():
        prefixes[match_id] = replay_selected_prefixes(by_match[match_id], sorted(offsets), context=context)
    records: list[CachedPrefix] = []
    for entry, match_id, offset, verdict, target, stats in corrections:
        match = by_match[match_id]
        action = action_from_record(match["decisions"][offset]["learner"])
        records.append(CachedPrefix(
            key=str(entry["entry_id"]), match_id=match_id, offset=offset,
            observations=prefixes[match_id][offset],
            legal_play=[np.asarray(match["decisions"][offset].get("legal", {}).get("play", []))] if False else [
                np.asarray(
                    # use the replayed legal mask for each context step below
                    _legal_from_observation(match, offset, prefixes[match_id][offset], index),
                    dtype=bool,
                )
                for index in range(len(prefixes[match_id][offset]))
            ],
            legal_wait=[True] * len(prefixes[match_id][offset]),
            recorded_action=action, correction_action=target,
            branches=list(verdict["branches"]), paired_mean=float(stats["mean"]),
            paired_se=float(stats["se"]), correction_lcb=float(stats["lcb95"]),
            is_correction=True,
        ))
    # Rebuild retention records from the offsets selected above.  Corrected
    # offsets are excluded, so a source state cannot be both targets.
    corrected_keys = {(r.match_id, r.offset) for r in records}
    for match_id, offsets in selected_offsets.items():
        match = by_match[match_id]
        for offset in sorted(offsets):
            if (match_id, offset) in corrected_keys:
                continue
            if offset >= len(match["decisions"]):
                continue
            action = action_from_record(match["decisions"][offset]["learner"])
            sequence = prefixes[match_id][offset]
            records.append(CachedPrefix(
                key=f"retention:{match_id}:t{offset}", match_id=match_id,
                offset=offset, observations=sequence,
                legal_play=[_legal_from_observation(match, offset, sequence, index) for index in range(len(sequence))],
                legal_wait=[True] * len(sequence), recorded_action=action,
                is_correction=False,
            ))
    # Keep the requested retention cardinality as an upper bound while always
    # retaining every accepted correction.
    corrections_only = [r for r in records if r.is_correction]
    retention = [r for r in records if not r.is_correction]
    if retention_count > 0:
        retention = retention[:retention_count]
    return corrections_only + retention


def _legal_from_observation(
    match: Mapping[str, Any], offset: int, sequence: Sequence[Mapping[str, np.ndarray]], index: int
) -> np.ndarray:
    """Read legality from the cached public input when present.

    ``replay_selected_prefixes`` currently stores legality separately only in
    the replay result; the observation itself has no legality key.  This
    helper is replaced by ``_attach_legal`` during normal preparation and is
    retained as a fail-closed compatibility path.
    """

    # The fallback is intentionally permissive for tests using synthetic
    # cached inputs.  Real records are populated by ``_attach_legal`` below.
    value = sequence[index].get("legal_play")
    if value is not None:
        return np.asarray(value, dtype=bool)
    record = match["decisions"][min(offset, len(match["decisions"]) - 1)]
    legal = record.get("legal_play")
    if legal is not None:
        return np.asarray(legal, dtype=bool)
    # Simulator inputs always have the V4 4x32x18 grid.  An all-true fallback
    # is only for legacy records; it is never used for target validation.
    return np.ones((4, 32, 18), dtype=bool)


def _attach_legal_to_prefix_cache(
    cache: dict[int, list[dict[str, np.ndarray]]],
    legal_cache: dict[int, list[np.ndarray]],
) -> None:
    for offset, sequence in cache.items():
        for inputs, legal in zip(sequence, legal_cache[offset]):
            inputs["legal_play"] = np.asarray(legal, dtype=bool)


def _replay_selected_prefixes_with_legal(
    match: Mapping[str, Any], offsets: Sequence[int], *, context: int,
    deadline_epoch: float | None = None,
) -> tuple[dict[int, list[dict[str, np.ndarray]]], dict[int, list[np.ndarray]], dict[int, list[bool]]]:
    """Replay helper retaining legality and WAIT availability."""

    requested = sorted({int(v) for v in offsets})
    if not requested:
        return {}, {}, {}
    decisions = match["decisions"]
    env = _fresh_env(dict(match))
    from simulator.public_state_estimator import PublicStateEstimator

    estimator = PublicStateEstimator()
    all_inputs: dict[int, dict[str, np.ndarray]] = {}
    all_legal: dict[int, np.ndarray] = {}
    all_wait: dict[int, bool] = {}
    maximum = requested[-1]
    for index in range(maximum + 1):
        _check_deadline(deadline_epoch)
        inputs, legal, _, _ = _observe_inputs(env, "cpu", estimator)
        all_inputs[index] = _required_observation_inputs(inputs)
        all_inputs[index]["legal_play"] = np.asarray(legal, dtype=bool).copy()
        all_wait[index] = bool(inputs.get("legal_wait", True))
        all_legal[index] = np.asarray(legal, dtype=bool).copy()
        if index < maximum:
            record = decisions[index]
            env.step_v2((_recorded_action(record["learner"], 0), _recorded_action(record["opponent"], 1)))
    outputs: dict[int, list[dict[str, np.ndarray]]] = {}
    legal_outputs: dict[int, list[np.ndarray]] = {}
    wait_outputs: dict[int, list[bool]] = {}
    for offset in requested:
        first = max(0, offset - context)
        sequence = [all_inputs[index] for index in range(first, offset + 1)]
        legal_sequence = [all_legal[index] for index in range(first, offset + 1)]
        wait_sequence = [all_wait[index] for index in range(first, offset + 1)]
        while len(sequence) < context + 1:
            sequence.insert(0, {name: _copy_input(value) for name, value in sequence[0].items()})
            legal_sequence.insert(0, legal_sequence[0].copy())
            wait_sequence.insert(0, wait_sequence[0])
        outputs[offset] = sequence
        legal_outputs[offset] = legal_sequence
        wait_outputs[offset] = wait_sequence
    return outputs, legal_outputs, wait_outputs


def _replace_prepare_replay() -> None:
    """Marker for source compatibility; preparation uses the legal variant."""


def _build_records_impl(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    context: int = DEFAULT_CONTEXT,
    min_lcb: float = DEFAULT_MIN_LCB,
    retention_count: int = 0,
    seed: int = 0,
    require_independent: bool = True,
    deadline_epoch: float | None = None,
) -> list[CachedPrefix]:
    if retention_count < 0:
        raise ValueError("retention_count must be non-negative")
    by_match: dict[tuple[str, str], dict[str, Any]] = {}
    match_paths: dict[tuple[str, str], Path] = {}
    selected: dict[tuple[str, str], set[int]] = {}
    corrections: list[tuple[dict[str, Any], tuple[str, str], int, dict[str, Any], ActionTarget, dict[str, float]]] = []
    requested_splits: set[str] = set()
    for raw in entries:
        _check_deadline(deadline_epoch)
        entry = dict(raw)
        split, match_id, offset = parse_entry_id(str(entry["entry_id"]))
        requested_splits.add(split)
        key = (split, match_id)
        if key not in by_match:
            path = match_json_path(root, split, match_id)
            by_match[key] = json.loads(path.read_text(encoding="utf-8"))
            match_paths[key] = path
        verdict = _load_verdict(root, split, match_id, offset)
        accepted, target, stats = validated_correction(
            verdict, min_lcb=min_lcb, require_independent=require_independent,
        )
        selected.setdefault(key, set()).add(offset)
        if accepted and target is not None:
            corrections.append((entry, key, offset, verdict, target, stats))

    # Frontier entries are intentionally sparse.  Ordinary replay must draw
    # from the complete train rollout pool, including matches that contributed
    # no accepted frontier entry.  The validation root is a separate call, so
    # restricting to the entry splits here avoids accidentally mixing pools.
    for path in sorted(root.glob("*_match.json")):
        _check_deadline(deadline_epoch)
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        key = _match_id_from_json(path, match)
        if key is None or (requested_splits and key[0] not in requested_splits):
            continue
        if key not in by_match:
            by_match[key] = match
            match_paths[key] = path
    if retention_count > 0:
        # One offset is drawn from each match in each round.  This gives every
        # eligible match an independent replay opportunity before any match
        # receives a second state, avoiding the old correction-bearing bias and
        # the old sorted-key truncation bias.
        candidates: dict[tuple[str, str], list[int]] = {}
        for key, match in sorted(by_match.items()):
            excluded = selected.get(key, set())
            values = [
                int(record["t"])
                for record in match.get("decisions", ())
                if int(record["t"]) >= context and int(record["t"]) not in excluded
            ]
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16)
            )
            rng.shuffle(values)
            candidates[key] = values
        chosen = 0
        while chosen < retention_count:
            progressed = False
            for key in sorted(candidates):
                if chosen >= retention_count:
                    break
                values = candidates[key]
                if not values:
                    continue
                selected.setdefault(key, set()).add(int(values.pop()))
                chosen += 1
                progressed = True
            if not progressed:
                break
    prefixes: dict[tuple[str, str], tuple[dict[int, list[dict[str, np.ndarray]]], dict[int, list[np.ndarray]], dict[int, list[bool]]]] = {}
    for key, offsets in selected.items():
        _check_deadline(deadline_epoch)
        prefixes[key] = _replay_selected_prefixes_with_legal(
            by_match[key], sorted(offsets), context=context, deadline_epoch=deadline_epoch,
        )
    records: list[CachedPrefix] = []
    for entry, key, offset, verdict, target, stats in corrections:
        split, match_id = key
        match = by_match[key]
        seqs, legal, waits = prefixes[key]
        records.append(CachedPrefix(
            key=str(entry["entry_id"]), match_id=match_id, offset=offset,
            observations=seqs[offset], legal_play=legal[offset], legal_wait=waits[offset],
            recorded_action=action_from_record(match["decisions"][offset]["learner"]),
            correction_action=target, branches=list(verdict["branches"]),
            paired_mean=float(stats["mean"]), paired_se=float(stats["se"]),
            correction_lcb=float(stats["lcb95"]), is_correction=True,
            match_path=str(match_paths[key]),
            opponent_spec={
                "strategy": str(match.get("config", {}).get("strategy", "wait")),
                "controller_seed": int(match.get("config", {}).get("controller_seed", 0)),
            },
            outcome_horizon=int(verdict.get("horizon", 128)),
            outcome_replicas=int(verdict.get("n_replicas", 4)),
            outcome_replica_seed_base=int(verdict.get("replica_seed_base", 0)),
        ))
    # ``CachedPrefix`` historically stored only ``match_id``.  Within one
    # split call this is sufficient and avoids changing the serialized public
    # key format.
    corrected = {(split, r.match_id, r.offset) for (split, _), r in zip(
        (key for key, *_rest in corrections), records
    )}
    for key, offsets in selected.items():
        split, match_id = key
        match = by_match[key]
        seqs, legal, waits = prefixes[key]
        for offset in sorted(offsets):
            if (split, match_id, offset) in corrected:
                continue
            records.append(CachedPrefix(
                key=f"retention:{split}:{match_id}:t{offset}", match_id=match_id,
                offset=offset, observations=seqs[offset], legal_play=legal[offset],
                legal_wait=waits[offset], recorded_action=action_from_record(match["decisions"][offset]["learner"]),
                match_path=str(match_paths[key]),
                opponent_spec={
                    "strategy": str(match.get("config", {}).get("strategy", "wait")),
                    "controller_seed": int(match.get("config", {}).get("controller_seed", 0)),
                },
            ))
    corrections_out = [r for r in records if r.is_correction]
    retention_out = [r for r in records if not r.is_correction]
    return corrections_out + (retention_out if retention_count > 0 else [])


# Replace the first implementation with the legality-preserving implementation
# while keeping one public, documented entry point.
prepare_records = _build_records_impl


def _stack_records(records: Sequence[CachedPrefix], *, context: int, device: torch.device) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot stack an empty record list")
    expected = context + 1
    def selected(record: CachedPrefix) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
        if len(record.observations) < expected:
            raise ValueError("record does not contain the requested context")
        return record.observations[-expected:], record.legal_play[-expected:]
    names = (
        "raster", "global_features", "entities", "entity_mask", "hand_tokens",
        "opp_hand_probs", "opp_out_of_cycle", "opp_elixir_interval", "event_history",
    )
    output: dict[str, Any] = {}
    for name in names:
        dtype = torch.bool if name == "entity_mask" or name == "opp_out_of_cycle" else torch.float32
        output[name] = torch.stack(
            [torch.stack([torch.as_tensor(obs[name], dtype=dtype) for obs in selected(r)[0]]) for r in records]
        ).to(device)
    legal = torch.stack(
        [torch.stack([torch.as_tensor(value, dtype=torch.bool) for value in selected(r)[1]]) for r in records]
    ).to(device)
    output["legal_play"] = legal
    output["masks"] = masks_from_legal_play(legal)
    output["reset_mask"] = torch.zeros(len(records), expected, dtype=torch.bool, device=device)
    output["reset_mask"][:, 0] = True
    return output


def _forward(policy: RecurrentV4Policy, batch: Mapping[str, Any]) -> tuple[V4Logits, Any]:
    logits, _, _ = policy(
        batch["raster"], batch["global_features"], batch["entities"],
        batch["entity_mask"], batch["hand_tokens"], batch["opp_hand_probs"],
        batch["opp_out_of_cycle"], batch["opp_elixir_interval"],
        batch["event_history"], batch["reset_mask"],
    )
    return logits, batch["masks"]


def _factorized_kl(
    old: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    new: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    masks: Any,
) -> torch.Tensor:
    """Mean factorized KL at the decision step.

    Prefixes are fed to the GRU to establish recurrent state, but retention
    supervision belongs to the selected decision.  Reducing over every
    prefix step accidentally turns burn-in into a second, implicit training
    objective and makes ``context=32`` differ from ``context=0`` even when the
    final distributions match.
    """

    old_mode, old_card, old_place = old
    new_mode, new_card, new_place = new
    if old_mode.ndim >= 3:
        old_mode, new_mode = old_mode[:, -1, ...], new_mode[:, -1, ...]
        old_card, new_card = old_card[:, -1, ...], new_card[:, -1, ...]
        old_place, new_place = old_place[:, -1, ...], new_place[:, -1, ...]
    eps = torch.finfo(old_mode.dtype).tiny
    mode = (old_mode * (old_mode.clamp_min(eps).log() - new_mode.clamp_min(eps).log())).sum(-1)
    play = old_mode[..., 1]
    card = (old_card * (old_card.clamp_min(eps).log() - new_card.clamp_min(eps).log())).sum(-1)
    place = (old_place * (old_place.clamp_min(eps).log() - new_place.clamp_min(eps).log())).sum((-1, -2))
    place = (place * old_card).sum(-1)
    card_mask = masks.card[:, -1, ...] if masks.card.ndim >= 3 else masks.card
    valid_card = card_mask.any(-1).to(mode.dtype)
    return (mode + play * (card + place) * valid_card).mean()


def _factorized_hard_nll(
    logits: V4Logits, masks: Any, actions: Sequence[ActionTarget]
) -> torch.Tensor:
    mode, card, place = factorized_distributions(logits, masks)
    eps = torch.finfo(mode.dtype).tiny
    mode_t = torch.tensor([a.mode for a in actions], device=mode.device, dtype=torch.long)
    card_t = torch.tensor([a.slot for a in actions], device=mode.device, dtype=torch.long)
    row_t = torch.tensor([a.row for a in actions], device=mode.device, dtype=torch.long)
    col_t = torch.tensor([a.col for a in actions], device=mode.device, dtype=torch.long)
    # The caller supplies final-step logits [B, 1, ...].
    mode = mode[:, -1]
    card = card[:, -1]
    place = place[:, -1]
    loss = -mode[torch.arange(len(actions), device=mode.device), mode_t].clamp_min(eps).log()
    play = mode_t == 1
    if bool(play.any()):
        idx = torch.where(play)[0]
        loss[idx] += -card[idx, card_t[idx]].clamp_min(eps).log()
        loss[idx] += -place[idx, card_t[idx], row_t[idx], col_t[idx]].clamp_min(eps).log()
    return loss.mean()


def _target_tensor(target: FactorizedTarget, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for value in (target.mode, target.card, target.placement))  # type: ignore[return-value]


def _factorized_soft_kl(
    old: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    new: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: FactorizedTarget,
) -> torch.Tensor:
    target_mode, target_card, target_place = _target_tensor(target, new[0].device)
    eps = torch.finfo(new[0].dtype).tiny
    mode = (target_mode * (target_mode.clamp_min(eps).log() - new[0][:, -1].clamp_min(eps).log())).sum(-1)
    play = target_mode[..., 1]
    card = (target_card * (target_card.clamp_min(eps).log() - new[1][:, -1].clamp_min(eps).log())).sum(-1)
    place = (target_place * (target_place.clamp_min(eps).log() - new[2][:, -1].clamp_min(eps).log())).sum((-1, -2))
    place = (place * target_card).sum(-1)
    return (mode + play * (card + place)).mean()


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
        from .frontier_eval import BranchSpec, replay_to_state, run_branch

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


def compute_batch_loss(
    policy: RecurrentV4Policy,
    anchor: RecurrentV4Policy,
    batch: Mapping[str, Any],
    records: Sequence[CachedPrefix],
    *,
    method: str,
    temperature: float = DEFAULT_TEMPERATURE,
    retention_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one ablation loss on final target steps."""

    if method not in {"hard-none", "hard-replay", "soft-replay", "soft-kl"}:
        raise ValueError(f"unknown method {method!r}")
    logits, masks = _forward(policy, batch)
    with torch.no_grad():
        anchor_logits, _ = _forward(anchor, batch)
        old = factorized_distributions(anchor_logits, masks)
    corrections = [r for r in records if r.is_correction]
    retention = [r for r in records if not r.is_correction]
    # All batch records share one final-step index.  Build losses by subsets so
    # correction/retention cardinality cannot silently change weighting.
    losses: list[torch.Tensor] = []
    details: dict[str, float] = {"n_correction": float(len(corrections)), "n_retention": float(len(retention))}
    if corrections:
        indices = [i for i, r in enumerate(records) if r.is_correction]
        sub_logits = V4Logits(
            mode=logits.mode[indices], wait_duration=logits.wait_duration[indices],
            card=logits.card[indices], placement=logits.placement[indices],
        )
        sub_masks = type(masks)(mode=masks.mode[indices], card=masks.card[indices], placement=masks.placement[indices])
        if method.startswith("hard"):
            corr_loss = _factorized_hard_nll(sub_logits, sub_masks, [r.correction_action for r in corrections if r.correction_action is not None])
        else:
            corr_values = []
            sub_old = tuple(v[indices] for v in old)
            sub_new = factorized_distributions(sub_logits, sub_masks)
            for j, record in enumerate(corrections):
                # ``sub_old`` is already selected.  Indexing it with the
                # original batch index crashes as soon as retention and
                # correction records are interleaved (and silently selects
                # the wrong row when it happens to fit).
                corr_values.append(_factorized_soft_kl(
                    tuple(v[j:j + 1] for v in sub_old),
                    tuple(v[j:j + 1] for v in sub_new),
                    _record_soft_target(record, sub_old, j, temperature),
                ))
            corr_loss = torch.stack(corr_values).mean()
        losses.append(corr_loss)
        details["correction_loss"] = float(corr_loss.detach())
    if retention and method != "hard-none":
        indices = [i for i, r in enumerate(records) if not r.is_correction]
        sub_logits = V4Logits(
            mode=logits.mode[indices], wait_duration=logits.wait_duration[indices],
            card=logits.card[indices], placement=logits.placement[indices],
        )
        sub_masks = type(masks)(mode=masks.mode[indices], card=masks.card[indices], placement=masks.placement[indices])
        sub_new = factorized_distributions(sub_logits, sub_masks)
        sub_old = tuple(v[indices] for v in old)
        if method == "hard-replay":
            replay_loss = _factorized_hard_nll(sub_logits, sub_masks, [r.recorded_action for r in retention])
        elif method == "soft-replay":
            # Hard and soft replay compare the same recorded source action.
            # Using the anchor argmax here changes labels and confounds the
            # target softness ablation with a replay-target change.
            replay_loss = _factorized_hard_nll(sub_logits, sub_masks, [r.recorded_action for r in retention])
        else:
            replay_loss = _factorized_kl(sub_old, sub_new, sub_masks)
        losses.append(float(retention_weight) * replay_loss)
        details["retention_loss"] = float(replay_loss.detach())
    if not losses:
        raise ValueError("batch has no active loss terms")
    total = torch.stack(losses).sum()
    details["loss"] = float(total.detach())
    return total, details


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


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so a deadline cannot leave a bad file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_report(
    *, method: str, seed: int, context: int, epochs: int, batch_size: int,
    learning_rate: float, history: Sequence[Mapping[str, Any]],
    optimizer_steps: int, updates_per_epoch: int,
    correction_examples: int, retention_examples: int,
    status: str, metrics: Mapping[str, Any] | None = None,
    resumed_from_epoch: int = 0,
) -> dict[str, Any]:
    total_examples = correction_examples + retention_examples
    report: dict[str, Any] = {
        "status": status,
        "method": method, "seed": int(seed), "context": int(context),
        "epochs": int(epochs), "batch_size": int(batch_size),
        "learning_rate": float(learning_rate), "history": list(history),
        "optimizer_steps": int(optimizer_steps),
        "updates_per_epoch": int(updates_per_epoch),
        "correction_examples_seen": int(correction_examples),
        "retention_examples_seen": int(retention_examples),
        "correction_exposure": float(correction_examples / max(1, total_examples)),
        "resumed_from_epoch": int(resumed_from_epoch),
    }
    if metrics is not None:
        report["validation"] = dict(metrics)
    return report


def _save_training_state(
    *, checkpoint_path: Path, student: RecurrentV4Policy, optimizer: torch.optim.Optimizer,
    report: Mapping[str, Any], epoch_completed: int,
) -> None:
    _atomic_torch_save({
        "state_dict": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "report": dict(report),
        "epoch_completed": int(epoch_completed),
    }, checkpoint_path)


def train_one(
    anchor: RecurrentV4Policy,
    train_records: Sequence[CachedPrefix],
    validation_records: Sequence[CachedPrefix],
    *,
    method: str,
    seed: int,
    context: int,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    retention_weight: float = 1.0,
    temperature: float = DEFAULT_TEMPERATURE,
    device: torch.device | str = "cpu",
    checkpoint_path: Path | None = None,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    """Train one deterministic method/seed and return report metadata.

    Every method receives the same number of optimizer steps per epoch.  The
    hard-none ablation cycles correction records with replacement when its
    pool is smaller; the report records the resulting correction exposure so
    retention gains are not interpreted without accounting for extra updates.
    A partial checkpoint is committed at each completed epoch and is resumed
    automatically when the same run is invoked again.
    """

    device = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    student = copy.deepcopy(anchor).to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.train()
    anchor_device = copy.deepcopy(anchor).to(device)
    anchor_device.eval()
    for parameter in anchor_device.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(learning_rate))
    active = list(train_records) if method != "hard-none" else [r for r in train_records if r.is_correction]
    if not active:
        raise ValueError(f"no active training records for {method}")
    if int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError("epochs and batch_size must be positive")

    # Equal optimizer budget: all methods use the full train-record cardinality
    # to determine steps, while method-specific pools determine labels.
    updates_per_epoch = max(1, int(math.ceil(len(train_records) / int(batch_size))))
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    correction_examples = 0
    retention_examples = 0
    start_epoch = 0
    resumed_from_epoch = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"cannot read existing distillation checkpoint {checkpoint_path}") from exc
        if not isinstance(saved, Mapping) or not isinstance(saved.get("report"), Mapping):
            raise RuntimeError(f"invalid distillation checkpoint schema: {checkpoint_path}")
        saved_report = saved["report"]
        saved_status = saved_report.get("status")
        if saved_status == "complete":
            raise RuntimeError(f"completed checkpoint already exists; use a new run or resume manifest: {checkpoint_path}")
        if saved_status != "partial":
            raise RuntimeError(f"checkpoint is neither complete nor resumable: {checkpoint_path}")
        if (
            saved_report.get("method") != method
            or int(saved_report.get("seed", -1)) != int(seed)
            or int(saved_report.get("context", -1)) != int(context)
        ):
            raise RuntimeError(f"checkpoint metadata does not match requested run: {checkpoint_path}")
        try:
            student.load_state_dict(saved["state_dict"])
            if saved.get("optimizer_state"):
                optimizer.load_state_dict(saved["optimizer_state"])
            history = [dict(row) for row in saved_report.get("history", [])]
            optimizer_steps = int(saved_report.get("optimizer_steps", 0))
            correction_examples = int(saved_report.get("correction_examples_seen", 0))
            retention_examples = int(saved_report.get("retention_examples_seen", 0))
            start_epoch = int(saved.get("epoch_completed", len(history)))
        except Exception as exc:
            raise RuntimeError(f"invalid resumable distillation checkpoint: {checkpoint_path}") from exc
        resumed_from_epoch = start_epoch
        print(f"[search-distill] resumed method={method} seed={seed} context={context} epoch={start_epoch}", flush=True)

    def make_report(status: str, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = _training_report(
            method=method, seed=seed, context=context, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, history=history,
            optimizer_steps=optimizer_steps, updates_per_epoch=updates_per_epoch,
            correction_examples=correction_examples, retention_examples=retention_examples,
            status=status, metrics=metrics, resumed_from_epoch=resumed_from_epoch,
        )
        payload["parameters"] = int(sum(p.numel() for p in student.parameters() if p.requires_grad))
        if checkpoint_path is not None:
            payload["checkpoint"] = str(checkpoint_path)
        return payload

    # A deadline inside epoch zero must resume from the untouched initial
    # state. Save that boundary before the first optimizer step.
    if checkpoint_path is not None and not checkpoint_path.is_file():
        _save_training_state(
            checkpoint_path=checkpoint_path, student=student, optimizer=optimizer,
            report=make_report("partial"), epoch_completed=0,
        )

    try:
        for epoch in range(start_epoch, int(epochs)):
            _check_deadline(deadline_epoch)
            rng = np.random.default_rng(int(seed) + epoch * 1009)
            order: list[int] = []
            while len(order) < updates_per_epoch * int(batch_size):
                order.extend(int(i) for i in rng.permutation(len(active)).tolist())
            order = order[:updates_per_epoch * int(batch_size)]
            totals: list[float] = []
            epoch_corrections = 0
            epoch_retention = 0
            for start in range(0, len(order), int(batch_size)):
                _check_deadline(deadline_epoch)
                group = [active[int(i)] for i in order[start:start + int(batch_size)]]
                epoch_corrections += sum(int(record.is_correction) for record in group)
                epoch_retention += len(group) - sum(int(record.is_correction) for record in group)
                batch = _stack_records(group, context=context, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss, details = compute_batch_loss(
                    student, anchor_device, batch, group, method=method,
                    temperature=temperature, retention_weight=retention_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {method} loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                _check_deadline(deadline_epoch)
                optimizer.step()
                optimizer_steps += 1
                totals.append(float(loss.detach().cpu()))
            correction_examples += epoch_corrections
            retention_examples += epoch_retention
            history.append({
                "epoch": float(epoch), "loss": float(np.mean(totals)),
                "optimizer_steps": float(updates_per_epoch),
                "correction_examples": float(epoch_corrections),
                "retention_examples": float(epoch_retention),
            })
            if checkpoint_path is not None:
                partial = make_report("partial")
                _save_training_state(
                    checkpoint_path=checkpoint_path, student=student, optimizer=optimizer,
                    report=partial, epoch_completed=epoch + 1,
                )
            print(f"[search-distill] method={method} seed={seed} epoch={epoch + 1}/{epochs} loss={history[-1]['loss']:.6f} steps={updates_per_epoch}", flush=True)
        metrics = evaluate_records(
            student, anchor_device, validation_records, context=context,
            batch_size=batch_size, device=device, deadline_epoch=deadline_epoch,
        )
    except DistillationDeadline as exc:
        partial = make_report("partial")
        partial["deadline_epoch"] = float(deadline_epoch) if deadline_epoch is not None else None
        partial["deadline_reason"] = str(exc)
        # The latest completed epoch, or the untouched initial state, remains
        # on disk. Discard any partial-epoch updates when resuming.
        return partial

    payload = make_report("complete", metrics)
    if checkpoint_path is not None:
        _save_training_state(
            checkpoint_path=checkpoint_path, student=student, optimizer=optimizer,
            report=payload, epoch_completed=int(epochs),
        )
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__all__ = [
    "ActionTarget", "CachedPrefix", "CacheConfig", "FactorizedTarget",
    "SEARCH_DISTILLATION_VERSION", "build_soft_search_target", "compute_batch_loss",
    "evaluate_records", "factorized_distributions", "file_sha256", "git_revision",
    "hard_factorized_target", "load_frontier_entries", "match_json_path", "paired_advantage",
    "parse_entry_id", "prepare_records", "replay_selected_prefixes", "train_one",
    "validated_correction",
]
