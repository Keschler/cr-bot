"""Opt-in, factual action-conditioned tactical consequence prediction.

The target builder consumes one chronological *public* episode, with a
pre-action measurement at every anchor and later measurements after actual
actions. It does not rank actions or invent counterfactual labels. Survival
cohorts are fixed at the anchor: later spawns cannot replace a dead defender.
The actor's recurrent features and its executed action are the head's only
inputs. Nothing in this module reads a simulator engine or opponent hand.

Measurement and target utilities do not import Torch. Construct the optional
head with :func:`build_tactical_outcome_head`; save its ``state_dict`` together
with ``dataclasses.asdict(head.config)`` to retain the horizon contract.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Literal, Sequence

import numpy as np


DEFAULT_HORIZONS_SECONDS = (2.0, 6.0, 12.0)
OUTCOME_NAMES = (
    "own_tower_damage", "enemy_tower_damage",
    "air_threat_survival", "own_defender_survival",
)


def _horizons(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (not result or any(not math.isfinite(value) or value <= 0 for value in result)
            or any(left >= right for left, right in zip(result, result[1:]))):
        raise ValueError("horizons_seconds must be finite, positive, and strictly increasing")
    return result


@dataclass(frozen=True, slots=True)
class TacticalOutcomeConfig:
    recurrent_dim: int = 256
    hidden_dim: int = 128
    card_slots: int = 4
    placement_rows: int = 32
    placement_cols: int = 18
    horizons_seconds: tuple[float, ...] = DEFAULT_HORIZONS_SECONDS

    def __post_init__(self) -> None:
        for name in ("recurrent_dim", "hidden_dim", "card_slots", "placement_rows", "placement_cols"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "horizons_seconds", _horizons(self.horizons_seconds))

    @property
    def action_dim(self) -> int:
        return 2 + self.card_slots + 2


@dataclass(frozen=True, slots=True)
class PublicTrackedEntity:
    """Stable public track identity and liveness, never an engine entity.

    ``alive=False`` requires public evidence of death. A disappeared detection
    alone is insufficient. Mark own troops/buildings belonging to the desired
    defender cohort with ``is_defender=True``; spells and towers are excluded
    by the public producer. IDs must remain stable and never be reused within
    an episode. HP, target IDs, cooldowns, and spawn provenance are unnecessary.
    """

    track_id: int
    side: Literal["own", "enemy"]
    is_air: bool = False
    is_defender: bool = False
    alive: bool = True

    def __post_init__(self) -> None:
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if self.side not in ("own", "enemy"):
            raise ValueError("side must be own or enemy")
        for name in ("is_air", "is_defender", "alive"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class PublicTacticalMeasurement:
    """A public measurement at explicit elapsed match time in seconds.

    Tower health is ``sum(current tower HP) / sum(maximum tower HP)`` using
    the same fixed denominator throughout the episode. ``None`` masks a side.
    ``entities_complete`` means absent tracks are *known no longer alive*,
    which is stronger than all visible detections having been enumerated.
    Leave it false for occlusion, concealment, tracking loss, or token caps.
    A genuine terminal measurement becomes absorbing; a truncated one does not.
    """

    match_seconds: float
    own_tower_health: float | None
    enemy_tower_health: float | None
    entities: tuple[PublicTrackedEntity, ...] = ()
    entities_complete: bool = False
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.match_seconds) or self.match_seconds < 0:
            raise ValueError("match_seconds must be finite and non-negative")
        for name in ("own_tower_health", "enemy_tower_health"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be None or a finite fraction in [0, 1]")
        for name in ("entities_complete", "terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.terminated and self.truncated:
            raise ValueError("termination and truncation are mutually exclusive")
        entities = tuple(self.entities)
        if any(not isinstance(entity, PublicTrackedEntity) for entity in entities):
            raise TypeError("entities must contain PublicTrackedEntity values")
        if len({entity.track_id for entity in entities}) != len(entities):
            raise ValueError("public track IDs must be unique within each measurement")
        object.__setattr__(self, "entities", entities)


def normalized_tower_health(tower_hp: Sequence[float] | None, *, maximum_total_hp: float) -> float | None:
    """Normalize measured public HP by an explicit, fixed maximum total HP."""

    if not math.isfinite(maximum_total_hp) or maximum_total_hp <= 0:
        raise ValueError("maximum_total_hp must be finite and positive")
    if tower_hp is None:
        return None
    values = tuple(float(value) for value in tower_hp)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("tower_hp must contain finite, non-negative measurements")
    health = math.fsum(values) / maximum_total_hp
    if health > 1 + 1e-7:
        raise ValueError("measured tower HP exceeds maximum_total_hp")
    return min(1.0, health)


@dataclass(frozen=True, slots=True)
class TacticalOutcomeTargets:
    """Readonly ``[anchor, horizon, outcome]`` targets and independent masks.

    ``future_indices`` is -1 when the time horizon is censored or unsampled;
    it otherwise records the actual measurement index, including absorption.
    ``measured_seconds`` exposes timing tolerance and terminal shortening.
    """

    values: np.ndarray
    mask: np.ndarray
    future_indices: np.ndarray
    measured_seconds: np.ndarray
    anchor_indices: tuple[int, ...]
    horizons_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        horizons = _horizons(self.horizons_seconds)
        anchors = tuple(self.anchor_indices)
        shape = (len(anchors), len(horizons), len(OUTCOME_NAMES))
        for name, dtype, expected in (
            ("values", np.float32, shape), ("mask", np.bool_, shape),
            ("future_indices", np.int64, shape[:2]),
            ("measured_seconds", np.float64, shape[:2]),
        ):
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        selected = self.values[self.mask]
        if not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
            raise ValueError("valid outcome targets must be finite fractions in [0, 1]")
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "horizons_seconds", horizons)


def _survival(
    cohort: tuple[int, ...],
    future: PublicTacticalMeasurement,
    known_dead: frozenset[int],
) -> float | None:
    if not cohort:
        return None
    known = {entity.track_id: entity.alive for entity in future.entities}
    # Public death evidence remains valid after a track's tombstone disappears.
    # An earlier live detection cannot establish liveness at the deadline.
    known.update((track_id, False) for track_id in known_dead)
    if not future.entities_complete and any(track_id not in known for track_id in cohort):
        return None
    return sum(bool(known.get(track_id, False)) for track_id in cohort) / len(cohort)


def build_tactical_outcome_targets(
    measurements: Sequence[PublicTacticalMeasurement],
    anchor_indices: Sequence[int],
    *,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
    max_lag_seconds: float = 0.25,
) -> TacticalOutcomeTargets:
    """Label executed actions using later public measurements in one episode.

    Anchor indices identify measurements taken immediately before the executed
    action. Select the first measurement at/after each deadline, only within
    ``max_lag_seconds``; never use a measurement from before the deadline.
    An earlier true terminal snapshot is used for later horizons, describing
    the absorbing end state. Truncations and unfinished tails stay masked.
    An empty survival cohort is masked rather than labeled as zero survival.
    """

    horizons = _horizons(horizons_seconds)
    if not math.isfinite(max_lag_seconds) or max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be finite and non-negative")
    samples, anchors = tuple(measurements), tuple(anchor_indices)
    if any(not isinstance(value, PublicTacticalMeasurement) for value in samples):
        raise TypeError("measurements must contain PublicTacticalMeasurement values")
    times = [float(sample.match_seconds) for sample in samples]
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("measurements must have strictly increasing match_seconds")
    if any(sample.terminated or sample.truncated for sample in samples[:-1]):
        raise ValueError("episode boundaries must be final; split episodes before building targets")
    if any(type(index) is not int or index < 0 or index >= len(samples) for index in anchors):
        raise ValueError("anchor_indices must identify existing measurements")
    if any(left >= right for left, right in zip(anchors, anchors[1:])):
        raise ValueError("anchor_indices must be strictly increasing")
    if any(samples[index].terminated or samples[index].truncated for index in anchors):
        raise ValueError("an executed action cannot be anchored at an episode boundary")
    dead_at_measurement: list[frozenset[int]] = []
    seen, dead = set(), set()
    for sample in samples:
        present = {entity.track_id for entity in sample.entities}
        if any(entity.alive and entity.track_id in dead for entity in sample.entities):
            raise ValueError("public track IDs cannot become alive after known death or be reused")
        if sample.entities_complete:
            dead.update(seen - present)
        dead.update(entity.track_id for entity in sample.entities if not entity.alive)
        seen.update(present)
        dead_at_measurement.append(frozenset(dead))
    shape = (len(anchors), len(horizons), len(OUTCOME_NAMES))
    values, mask = np.zeros(shape, np.float32), np.zeros(shape, bool)
    future_indices = np.full(shape[:2], -1, np.int64)
    measured_seconds = np.full(shape[:2], np.nan, np.float64)
    for row, anchor_index in enumerate(anchors):
        anchor = samples[anchor_index]
        threats = tuple(entity.track_id for entity in anchor.entities
                        if entity.alive and entity.side == "enemy" and entity.is_air)
        defenders = tuple(entity.track_id for entity in anchor.entities
                          if entity.alive and entity.side == "own" and entity.is_defender)
        for column, horizon in enumerate(horizons):
            deadline = times[anchor_index] + horizon
            if samples[-1].terminated and times[-1] <= deadline:
                future_index = len(samples) - 1
            else:
                future_index = bisect_left(times, deadline, lo=anchor_index + 1)
                if (future_index == len(samples)
                        or times[future_index] - deadline > max_lag_seconds + 1e-9):
                    continue
            future = samples[future_index]
            future_indices[row, column] = future_index
            measured_seconds[row, column] = times[future_index]
            outcomes: list[float | None] = []
            for name in ("own_tower_health", "enemy_tower_health"):
                start, end = getattr(anchor, name), getattr(future, name)
                outcomes.append(None if start is None or end is None else max(0.0, start - end))
            outcomes.extend((
                _survival(threats, future, dead_at_measurement[future_index]),
                _survival(defenders, future, dead_at_measurement[future_index]),
            ))
            for outcome, value in enumerate(outcomes):
                if value is not None:
                    values[row, column, outcome] = value
                    mask[row, column, outcome] = True
    return TacticalOutcomeTargets(values, mask, future_indices, measured_seconds, anchors, horizons)


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        from ..foundation._compat import TorchUnavailableError
        raise TorchUnavailableError("Tactical outcome heads require the optional PyTorch dependency.") from exc
    return torch


def encode_tactical_actions(actions: Any, *, config: TacticalOutcomeConfig | None = None) -> Any:
    """Encode V4 ``mode/card_slot/placement`` tensors without privileged labels.

    Layout: WAIT/PLAY one-hot, PLAY-gated card-slot one-hot, PLAY-gated
    row/column cell centers in [0, 1]. Placement follows V4 ``(row, col)``.
    WAIT card/placement placeholders are ignored. This encoder supports the
    fixed-WAIT runtime; an explicit V4 WAIT duration must be its zero index.
    """

    torch, config = _torch(), config or TacticalOutcomeConfig()
    mode, slot, placement = actions.mode, actions.card_slot, actions.placement
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    for name, tensor in (("mode", mode), ("card_slot", slot), ("placement", placement)):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in integer_types:
            raise TypeError(f"{name} must be an integer torch.Tensor")
    if slot.shape != mode.shape or placement.shape != mode.shape + (2,):
        raise ValueError("action factors must share a prefix; placement ends in (row, col)")
    if slot.device != mode.device or placement.device != mode.device:
        raise ValueError("action factors must share a device")
    if bool(((mode != 0) & (mode != 1)).any()):
        raise ValueError("mode must be WAIT=0 or PLAY=1")
    play = mode == 1
    duration = getattr(actions, "wait_duration", None)
    if duration is not None:
        if not isinstance(duration, torch.Tensor) or duration.dtype not in integer_types:
            raise TypeError("wait_duration must be an integer torch.Tensor")
        if duration.shape != mode.shape or duration.device != mode.device:
            raise ValueError("wait_duration must share the mode shape and device")
        if bool(((~play) & (duration != 0)).any()):
            raise ValueError("tactical action encoding requires fixed WAIT duration index zero")
    invalid = ((slot < 0) | (slot >= config.card_slots)
               | (placement[..., 0] < 0) | (placement[..., 0] >= config.placement_rows)
               | (placement[..., 1] < 0) | (placement[..., 1] >= config.placement_cols))
    if bool((play & invalid).any()):
        raise ValueError("PLAY card slot and placement must be in range")
    safe_slot = torch.where(play, slot, 0).long()
    safe_cell = torch.where(play.unsqueeze(-1), placement, 0).to(torch.float32)
    dimensions = torch.tensor((config.placement_rows, config.placement_cols), device=mode.device)
    return torch.cat((
        torch.nn.functional.one_hot(mode.long(), 2).to(torch.float32),
        torch.nn.functional.one_hot(safe_slot, config.card_slots).to(torch.float32) * play.unsqueeze(-1),
        ((safe_cell + 0.5) / dimensions) * play.unsqueeze(-1),
    ), dim=-1)


@lru_cache(maxsize=1)
def _head_type() -> Any:
    torch = _torch()

    class TacticalOutcomeHead(torch.nn.Module):
        def __init__(self, config: TacticalOutcomeConfig) -> None:
            super().__init__()
            self.config = config
            self.network = torch.nn.Sequential(
                torch.nn.Linear(config.recurrent_dim + config.action_dim, config.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(config.hidden_dim, len(config.horizons_seconds) * len(OUTCOME_NAMES)),
            )

        def forward(self, recurrent_features: Any, action_encoding: Any) -> Any:
            if (recurrent_features.ndim < 1 or recurrent_features.shape[-1] != self.config.recurrent_dim
                    or action_encoding.shape != recurrent_features.shape[:-1] + (self.config.action_dim,)):
                raise ValueError("recurrent features and action encoding must have aligned prefixes and configured dimensions")
            if not torch.is_floating_point(recurrent_features):
                raise TypeError("recurrent_features must be floating point")
            action_encoding = action_encoding.to(device=recurrent_features.device, dtype=recurrent_features.dtype)
            logits = self.network(torch.cat((recurrent_features, action_encoding), dim=-1))
            return logits.reshape(recurrent_features.shape[:-1] + (len(self.config.horizons_seconds), len(OUTCOME_NAMES)))

    return TacticalOutcomeHead


def build_tactical_outcome_head(config: TacticalOutcomeConfig | None = None) -> Any:
    """Lazily create a small Torch module returning raw ``[..., H, 4]`` logits."""

    return _head_type()(config or TacticalOutcomeConfig())


def tactical_outcome_loss(logits: Any, targets: Any, mask: Any = None, *, beta: float = 0.1) -> tuple[Any, dict[str, float | int]]:
    """Masked Smooth-L1 over sigmoid predictions, averaged over valid elements.

    Pass a :class:`TacticalOutcomeTargets` or raw values plus a boolean mask.
    Censored targets never dilute the denominator or contribute gradients.
    Features are not detached: callers may jointly train the actor recurrent
    representation or explicitly detach it for a frozen-representation probe.
    """

    torch = _torch()
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not isinstance(logits, torch.Tensor) or not torch.is_floating_point(logits):
        raise TypeError("logits must be a floating point torch.Tensor")
    if isinstance(targets, TacticalOutcomeTargets):
        if mask is not None:
            raise ValueError("typed targets already contain their mask")
        targets, mask = targets.values.copy(), targets.mask.copy()
    if mask is None:
        raise ValueError("an explicit target mask is required")
    values = torch.as_tensor(targets, device=logits.device, dtype=logits.dtype)
    selected = torch.as_tensor(mask, device=logits.device)
    if (logits.ndim < 2 or logits.shape[-1] != len(OUTCOME_NAMES)
            or values.shape != logits.shape or selected.shape != logits.shape):
        raise ValueError("logits, values, and mask must share [..., horizon, 4] shape")
    if selected.dtype != torch.bool:
        raise TypeError("target mask must be boolean")
    if (not bool(torch.isfinite(values[selected]).all())
            or bool(((values[selected] < 0) | (values[selected] > 1)).any())):
        raise ValueError("valid targets must be finite fractions in [0, 1]")
    if not bool(torch.isfinite(logits[selected]).all()):
        raise ValueError("valid prediction logits must be finite")
    count = int(selected.sum().item())
    loss = (torch.nn.functional.smooth_l1_loss(logits[selected].sigmoid(), values[selected], beta=beta)
            if count else logits.reshape(-1)[:0].sum())
    metrics: dict[str, float | int] = {"valid_targets": count, "loss": float(loss.detach().item())}
    for index, name in enumerate(OUTCOME_NAMES):
        active = selected[..., index]
        metrics[f"{name}_count"] = int(active.sum().item())
        if bool(active.any()):
            error = logits[..., index][active].detach().sigmoid() - values[..., index][active]
            metrics[f"{name}_mae"] = float(error.abs().mean().item())
    return loss, metrics
