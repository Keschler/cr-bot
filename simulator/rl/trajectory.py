"""Tensor containers for recurrent policy sequences and PPO rollouts.

These dataclasses deliberately contain storage and validation only.  They do
not implement batching, replay, or PPO optimization.  The reset mask follows
the convention used by :class:`rl.model.GRURecurrentCore`: ``True`` at time
``t`` clears the hidden state immediately before observation ``t`` is
processed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._compat import TorchUnavailableError

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "rl.trajectory requires PyTorch. Install torch to use recurrent "
            "trajectory containers."
        ) from exc
    raise


def _require_tensor(name: str, value: object, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    return value


def _require_bool(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")


def _require_long(name: str, value: torch.Tensor) -> None:
    if value.dtype not in {torch.int64, torch.int32, torch.int16, torch.int8}:
        raise TypeError(f"{name} must have an integer dtype")


@dataclass(frozen=True, slots=True)
class ActionMasks:
    """Legality masks for the autoregressive action factorization.

    Shapes are ``[..., 2]`` for ``mode`` (WAIT, PLAY), ``[..., K]`` for
    ``card`` and ``[..., K, rows, columns]`` for ``placement``.  The leading
    dimensions normally are ``[batch, time]``.
    """

    mode: torch.Tensor
    card: torch.Tensor
    placement: torch.Tensor

    def __post_init__(self) -> None:
        mode = _require_tensor("mode mask", self.mode)
        card = _require_tensor("card mask", self.card)
        placement = _require_tensor("placement mask", self.placement)
        _require_bool("mode mask", mode)
        _require_bool("card mask", card)
        _require_bool("placement mask", placement)
        if mode.ndim < 1 or mode.shape[-1] != 2:
            raise ValueError("mode mask must have final dimension 2")
        if card.ndim != mode.ndim or card.shape[:-1] != mode.shape[:-1]:
            raise ValueError("card mask leading dimensions must match mode mask")
        if card.shape[-1] < 1:
            raise ValueError("card mask must contain at least one card slot")
        if placement.ndim != mode.ndim + 2:
            raise ValueError("placement mask must have card, row, and column dimensions")
        if placement.shape[:-3] != mode.shape[:-1]:
            raise ValueError("placement mask leading dimensions must match mode mask")
        if placement.shape[-3] != card.shape[-1]:
            raise ValueError("placement mask card dimension must match card mask")
        if placement.shape[-2] < 1 or placement.shape[-1] < 1:
            raise ValueError("placement mask must have a non-empty grid")

    @property
    def prefix_shape(self) -> tuple[int, ...]:
        return tuple(self.mode.shape[:-1])

    @property
    def card_slots(self) -> int:
        return int(self.card.shape[-1])

    @property
    def placement_shape(self) -> tuple[int, int]:
        return int(self.placement.shape[-2]), int(self.placement.shape[-1])


@dataclass(frozen=True, slots=True)
class ActionBatch:
    """Autoregressive actions stored with a recurrent rollout.

    ``mode`` is encoded as ``0`` for WAIT and ``1`` for PLAY.  ``placement``
    stores ``(row, column)``; values for WAIT entries are ignored but are kept
    in the tensor so a complete ``[batch, time]`` rollout remains rectangular.
    """

    mode: torch.Tensor
    card_slot: torch.Tensor
    placement: torch.Tensor

    def __post_init__(self) -> None:
        mode = _require_tensor("mode action", self.mode)
        card_slot = _require_tensor("card-slot action", self.card_slot)
        placement = _require_tensor("placement action", self.placement)
        _require_long("mode action", mode)
        _require_long("card-slot action", card_slot)
        _require_long("placement action", placement)
        if mode.ndim < 1:
            raise ValueError("action tensors must have at least one dimension")
        if card_slot.shape != mode.shape:
            raise ValueError("card-slot action shape must match mode action shape")
        if placement.shape[:-1] != mode.shape or placement.shape[-1] != 2:
            raise ValueError("placement action must have shape mode.shape + (2,)")

    @property
    def prefix_shape(self) -> tuple[int, ...]:
        return tuple(self.mode.shape)


@dataclass(frozen=True, slots=True)
class RecurrentSequence:
    """A contiguous public-observation sequence for recurrent training.

    ``hidden_states``, when present, stores the hidden state *before* each
    observation with shape ``[batch, time, layers, hidden]``.  ``initial_hidden``
    uses PyTorch GRU layout ``[layers, batch, hidden]`` and is the state before
    the first observation.  Either can be retained by a rollout collector to
    support burn-in and sequence minibatches later.
    """

    raster: torch.Tensor
    global_features: torch.Tensor
    entities: torch.Tensor
    entity_mask: torch.Tensor
    reset_mask: torch.Tensor
    hidden_states: torch.Tensor | None = None
    initial_hidden: torch.Tensor | None = None

    def __post_init__(self) -> None:
        raster = _require_tensor("raster", self.raster, ndim=5)
        global_features = _require_tensor("global_features", self.global_features, ndim=3)
        entities = _require_tensor("entities", self.entities, ndim=4)
        entity_mask = _require_tensor("entity_mask", self.entity_mask, ndim=3)
        reset_mask = _require_tensor("reset_mask", self.reset_mask, ndim=2)
        _require_bool("entity_mask", entity_mask)
        _require_bool("reset_mask", reset_mask)
        batch, time = raster.shape[:2]
        if batch < 1 or time < 1:
            raise ValueError("recurrent sequences must have positive batch and time dimensions")
        if global_features.shape[:2] != (batch, time):
            raise ValueError("global_features must share raster batch and time dimensions")
        if entities.shape[:2] != (batch, time):
            raise ValueError("entities must share raster batch and time dimensions")
        if entity_mask.shape != entities.shape[:3]:
            raise ValueError("entity_mask must match entities batch, time, and entity dimensions")
        if reset_mask.shape != (batch, time):
            raise ValueError("reset_mask must match raster batch and time dimensions")
        if self.hidden_states is not None:
            hidden_states = _require_tensor("hidden_states", self.hidden_states, ndim=4)
            if hidden_states.shape[:2] != (batch, time):
                raise ValueError("hidden_states must have shape [batch, time, layers, hidden]")
        if self.initial_hidden is not None:
            initial_hidden = _require_tensor("initial_hidden", self.initial_hidden, ndim=3)
            if initial_hidden.shape[1] != batch:
                raise ValueError("initial_hidden must have shape [layers, batch, hidden]")

    @property
    def batch_size(self) -> int:
        return int(self.raster.shape[0])

    @property
    def time_steps(self) -> int:
        return int(self.raster.shape[1])


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Recurrent PPO storage without any optimization behavior."""

    sequence: RecurrentSequence
    action_masks: ActionMasks
    actions: ActionBatch
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def __post_init__(self) -> None:
        prefix = self.sequence.raster.shape[:2]
        if self.action_masks.prefix_shape != tuple(prefix):
            raise ValueError("action masks must match sequence batch and time dimensions")
        if self.actions.prefix_shape != tuple(prefix):
            raise ValueError("actions must match sequence batch and time dimensions")
        for name, value in (
            ("rewards", self.rewards),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
            ("old_log_probs", self.old_log_probs),
            ("values", self.values),
            ("advantages", self.advantages),
            ("returns", self.returns),
        ):
            if value is None:
                continue
            tensor = _require_tensor(name, value, ndim=2)
            if tensor.shape != prefix:
                raise ValueError(f"{name} must have shape [batch, time]")
        _require_bool("terminated", self.terminated)
        _require_bool("truncated", self.truncated)

    @property
    def batch_size(self) -> int:
        return self.sequence.batch_size

    @property
    def time_steps(self) -> int:
        return self.sequence.time_steps
