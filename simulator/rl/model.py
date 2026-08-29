"""Small PyTorch production-model foundation for recurrent Clash Royale RL.

This module defines the model boundary only.  It contains no optimizer, PPO
loss, rollout collection, or checkpoint format.  Inputs are sequences with
shape ``[batch, time, ...]``:

* ``raster``: ``[B, T, raster_channels, height, width]``;
* ``global_features``: ``[B, T, global_dim]``;
* ``entities``: ``[B, T, entities, entity_feature_dim]``;
* ``entity_mask``: ``[B, T, entities]`` (``True`` means a public token exists);
* ``reset_mask``: ``[B, T]`` (``True`` resets GRU state before that step).

The action head factorizes decisions as WAIT/PLAY, card slot, then a
card-conditioned placement cell.  Legality is applied to each categorical
distribution before log probabilities are evaluated.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from threading import Lock

from ._compat import TorchUnavailableError

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "rl.model requires PyTorch. Install torch to use the production "
            "hybrid recurrent policy modules."
        ) from exc
    raise

from .trajectory import ActionBatch, ActionMasks


_INFERENCE_MHA_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Dimensions for the hybrid encoder, recurrent core, and action head."""

    raster_channels: int = 21
    raster_height: int = 32
    raster_width: int = 18
    global_dim: int = 768
    # Matches ``PolicyObservationV2``'s public entity-token contract.
    entity_dim: int = 32
    max_entities: int = 128
    model_dim: int = 128
    encoder_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 256
    gru_hidden_dim: int = 256
    gru_layers: int = 1
    card_slots: int = 4
    belief_card_count: int = 128
    placement_rows: int = 32
    placement_cols: int = 18
    dropout: float = 0.0
    # ``-1``/``0`` keeps the legacy compressed-global path. New production
    # runs can opt into an explicit per-slot public hand embedding; keeping
    # this switch at the model boundary lets older checkpoints remain usable
    # as frozen opponents.
    hand_feature_offset: int = -1
    hand_card_count: int = 0
    # Optional direct public context for the WAIT/PLAY gate. This remains
    # separate from the recurrent encoder so a new actor can learn sharp
    # elixir/hand boundaries without changing the legacy checkpoint ABI when
    # the option is disabled.
    direct_public_action_features: bool = False
    # Optional direct public context for card-slot selection.  This is kept as
    # a separate switch because older experiments enabled only the mode gate;
    # keeping its default disabled preserves strict loading of those
    # checkpoints while allowing new runs to learn the hand/cost decision
    # without waiting for a recurrent representation to discover it.
    direct_public_card_features: bool = False
    # Optional recurrent/entity context for the direct card head.  The plain
    # direct card head sees hand/elixir scalars but cannot distinguish a safe
    # cycle from a defensive state.  Keeping this as a separate switch avoids
    # changing the shape of existing direct-card checkpoints.
    contextual_public_card_features: bool = False
    # Optional legality context for the mode gate.  The masks are public
    # inputs and already encode the exact affordability/placement boundary;
    # exposing them to a small context head avoids forcing a recurrent core to
    # rediscover a conjunction such as "Hog is held but not affordable".
    direct_public_mask_features: bool = False
    # Full public state plus legality context for a nonlinear mode gate.  This
    # can represent the interaction between a hand identity, its affordability
    # mask, and current elixir; the older mask-only switch remains available
    # for checkpoint compatibility.
    direct_public_context_features: bool = False
    # Optional slot-wise card identity scorer.  Unlike the legacy direct card
    # head, which emits all four slot logits from one wide global vector, this
    # path scores each public hand segment independently.  It is useful for
    # fresh checkpoints that must distinguish a defensive card from Hog when
    # the same slot changes across the deck cycle.
    direct_public_slot_card_features: bool = False
    # Preserve a board-aligned feature map for placement. The pooled raster
    # representation remains the fallback when this option is disabled.
    spatial_placement_features: bool = False
    spatial_placement_dim: int = 32

    def __post_init__(self) -> None:
        positive_fields = (
            "raster_channels",
            "raster_height",
            "raster_width",
            "global_dim",
            "entity_dim",
            "max_entities",
            "model_dim",
            "encoder_dim",
            "transformer_heads",
            "transformer_layers",
            "transformer_ff_dim",
            "gru_hidden_dim",
            "gru_layers",
            "card_slots",
            "belief_card_count",
            "placement_rows",
            "placement_cols",
            "spatial_placement_dim",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.model_dim % self.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.hand_feature_offset == -1:
            if self.hand_card_count != 0:
                raise ValueError(
                    "hand_card_count must be zero when hand_feature_offset is disabled"
                )
        else:
            if type(self.hand_feature_offset) is not int or self.hand_feature_offset < 0:
                raise ValueError("hand_feature_offset must be -1 or non-negative")
            if type(self.hand_card_count) is not int or self.hand_card_count <= 0:
                raise ValueError(
                    "hand_card_count must be positive when hand features are enabled"
                )
            hand_end = self.hand_feature_offset + self.card_slots * self.hand_card_count
            if hand_end > self.global_dim:
                raise ValueError("explicit hand features must fit inside global_dim")
        if type(self.direct_public_action_features) is not bool:
            raise ValueError("direct_public_action_features must be boolean")
        if type(self.direct_public_card_features) is not bool:
            raise ValueError("direct_public_card_features must be boolean")
        if type(self.contextual_public_card_features) is not bool:
            raise ValueError("contextual_public_card_features must be boolean")
        if type(self.direct_public_mask_features) is not bool:
            raise ValueError("direct_public_mask_features must be boolean")
        if type(self.direct_public_context_features) is not bool:
            raise ValueError("direct_public_context_features must be boolean")
        if type(self.direct_public_slot_card_features) is not bool:
            raise ValueError("direct_public_slot_card_features must be boolean")
        if type(self.spatial_placement_features) is not bool:
            raise ValueError("spatial_placement_features must be boolean")
        if self.direct_public_slot_card_features and self.hand_feature_offset < 0:
            raise ValueError(
                "direct_public_slot_card_features require explicit hand features"
            )

    @property
    def global_features(self) -> int:
        """Alias matching the observation field name."""

        return self.global_dim

    @property
    def entity_feature_dim(self) -> int:
        """Alias matching the observation field name."""

        return self.entity_dim


@dataclass(frozen=True, slots=True)
class AutoregressiveLogits:
    """Unnormalized logits emitted by the three action factors."""

    mode: torch.Tensor
    card: torch.Tensor
    placement: torch.Tensor


@dataclass(frozen=True, slots=True)
class OpponentBeliefLogits:
    """Training-only predictions of hidden opponent state."""

    enemy_elixir: torch.Tensor
    enemy_hand: torch.Tensor
    enemy_next_card: torch.Tensor


@dataclass(frozen=True, slots=True)
class RecurrentPolicyOutput:
    """Intermediate tensors useful to rollout collectors and diagnostics."""

    logits: AutoregressiveLogits
    encoded_features: torch.Tensor
    recurrent_features: torch.Tensor
    final_hidden: torch.Tensor
    belief_logits: OpponentBeliefLogits | None = None
    # Optional projected public one-hot hand features used by the action heads.
    hand_features: torch.Tensor | None = None
    # Optional board-aligned features used by the spatial placement head.
    spatial_features: torch.Tensor | None = None

    @property
    def mode_logits(self) -> torch.Tensor:
        return self.logits.mode

    @property
    def card_logits(self) -> torch.Tensor:
        return self.logits.card

    @property
    def placement_logits(self) -> torch.Tensor:
        return self.logits.placement


class OpponentBeliefHeads(nn.Module):
    """Auxiliary heads that train GRU memory without exposing private state."""

    def __init__(self, hidden_dim: int, card_count: int = 128) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if type(card_count) is not int or card_count <= 0:
            raise ValueError("card_count must be a positive integer")
        self.hidden_dim = hidden_dim
        self.card_count = card_count
        self.enemy_elixir = nn.Linear(hidden_dim, 1)
        self.enemy_hand = nn.Linear(hidden_dim, card_count)
        self.enemy_next_card = nn.Linear(hidden_dim, card_count)

    def forward(self, recurrent_features: torch.Tensor) -> OpponentBeliefLogits:
        _require_floating("recurrent features", recurrent_features, ndim=3)
        if recurrent_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"recurrent features final dimension must be {self.hidden_dim}")
        return OpponentBeliefLogits(
            enemy_elixir=self.enemy_elixir(recurrent_features).squeeze(-1),
            enemy_hand=self.enemy_hand(recurrent_features),
            enemy_next_card=self.enemy_next_card(recurrent_features),
        )


class PrivilegedCritic(nn.Module):
    """Asymmetric value head using exact simulator state only during training.

    ``recurrent_features`` are detached by default.  This makes it explicit
    that privileged value supervision cannot backpropagate private-state
    information into the actor's observation encoder.  A separate critic
    encoder can still be trained on the exact state representation supplied by
    the caller.
    """

    def __init__(
        self,
        recurrent_dim: int,
        privileged_dim: int,
        *,
        hidden_dim: int | None = None,
        detach_actor_features: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("recurrent_dim", recurrent_dim),
            ("privileged_dim", privileged_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim is None:
            hidden_dim = max(64, min(512, recurrent_dim))
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.recurrent_dim = recurrent_dim
        self.privileged_dim = privileged_dim
        self.detach_actor_features = bool(detach_actor_features)
        self.privileged_encoder = nn.Sequential(
            nn.Linear(privileged_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(recurrent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        recurrent_features: torch.Tensor,
        privileged_features: torch.Tensor,
    ) -> torch.Tensor:
        _require_floating("recurrent features", recurrent_features, ndim=3)
        _require_floating("privileged features", privileged_features, ndim=3)
        if recurrent_features.shape[:2] != privileged_features.shape[:2]:
            raise ValueError("recurrent and privileged features must share batch/time dimensions")
        if recurrent_features.shape[-1] != self.recurrent_dim:
            raise ValueError(f"recurrent features final dimension must be {self.recurrent_dim}")
        if privileged_features.shape[-1] != self.privileged_dim:
            raise ValueError(f"privileged features final dimension must be {self.privileged_dim}")
        actor_features = (
            recurrent_features.detach()
            if self.detach_actor_features
            else recurrent_features
        )
        privileged = self.privileged_encoder(privileged_features)
        return self.value_head(torch.cat((actor_features, privileged), dim=-1)).squeeze(-1)


class HybridEncoder(nn.Module):
    """Encode raster, public entity tokens, and global features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        raster_hidden = max(8, config.model_dim // 2)
        self.raster_encoder = nn.Sequential(
            nn.Conv2d(config.raster_channels, raster_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(raster_hidden, config.model_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.entity_projection = nn.Sequential(
            nn.Linear(config.entity_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.entity_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.transformer_layers,
        )
        self.null_entity = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        nn.init.normal_(self.null_entity, mean=0.0, std=0.02)
        self.global_projection = nn.Sequential(
            nn.Linear(config.global_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.hand_projection: nn.Module | None = None
        if config.hand_feature_offset >= 0:
            self.hand_projection = nn.Sequential(
                nn.Linear(config.hand_card_count, config.model_dim),
                nn.GELU(),
                nn.LayerNorm(config.model_dim),
            )
        fusion_input_dim = 3 * config.model_dim
        if config.hand_feature_offset >= 0:
            fusion_input_dim += config.card_slots * config.model_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.encoder_dim),
            nn.GELU(),
            nn.LayerNorm(config.encoder_dim),
        )

    def _raw_hand_features(self, global_features: torch.Tensor) -> torch.Tensor:
        """Return the public one-hot hand table as ``[B, T, slots, cards]``."""

        _require_floating("global_features", global_features, ndim=3)
        batch, time, global_dim = global_features.shape
        if global_dim != self.config.global_dim:
            raise ValueError("global_features final dimension does not match ModelConfig")
        config = self.config
        hand_end = config.hand_feature_offset + config.card_slots * config.hand_card_count
        flat = global_features.reshape(batch * time, global_dim)
        hand = flat[:, config.hand_feature_offset:hand_end].reshape(
            batch * time,
            config.card_slots,
            config.hand_card_count,
        )
        return hand.reshape(batch, time, config.card_slots, config.hand_card_count)

    def public_hand_features(
        self,
        global_features: torch.Tensor,
    ) -> torch.Tensor | None:
        """Project each public one-hot hand slot for the action heads."""

        if self.config.hand_feature_offset < 0:
            return None
        if self.hand_projection is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("hand projection is not initialized")
        hand = self._raw_hand_features(global_features)
        return self.hand_projection(hand)

    def forward(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations while preserving the two-value return contract."""

        encoded, _spatial = self.forward_with_spatial(
            raster,
            global_features,
            entities,
            entity_mask,
        )
        return encoded

    def forward_with_spatial(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded, spatial, _hand_features = self.forward_with_aux(
            raster,
            global_features,
            entities,
            entity_mask,
        )
        return encoded, spatial

    def forward_with_aux(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        inference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode observations and return reusable projected hand features.

        ``forward_with_spatial`` retains its two-value public contract. This
        auxiliary form lets the recurrent actor reuse the hand features
        that were already computed for encoder fusion instead of projecting
        them a second time for the action head.
        """
        if type(inference) is not bool:
            raise TypeError("inference must be boolean")
        _require_floating("raster", raster, ndim=5)
        _require_floating("global_features", global_features, ndim=3)
        _require_floating("entities", entities, ndim=4)
        _require_bool("entity_mask", entity_mask, ndim=3)
        batch, time, channels, height, width = raster.shape
        config = self.config
        if (channels, height, width) != (
            config.raster_channels,
            config.raster_height,
            config.raster_width,
        ):
            raise ValueError(
                "raster shape does not match ModelConfig: "
                f"expected {(config.raster_channels, config.raster_height, config.raster_width)}, "
                f"got {(channels, height, width)}"
            )
        if global_features.shape[:2] != (batch, time) or global_features.shape[-1] != config.global_dim:
            raise ValueError("global_features must have shape [batch, time, global_dim]")
        if entities.shape[:2] != (batch, time) or entities.shape[-1] != config.entity_dim:
            raise ValueError("entities must have shape [batch, time, entity_count, entity_dim]")
        if entity_mask.shape != entities.shape[:3]:
            raise ValueError("entity_mask must match entities batch, time, and entity dimensions")
        entity_count = int(entities.shape[2])
        if entity_count < 1 or entity_count > config.max_entities:
            raise ValueError(f"entity_count must be in [1, {config.max_entities}]")

        flat_batch_time = batch * time
        raster_input = raster.reshape(flat_batch_time, channels, height, width)
        spatial_features: torch.Tensor | None = None
        if config.spatial_placement_features:
            # Reuse the legacy convolution weights and expose the feature map
            # before global pooling. This adds no duplicated convolution
            # parameters and keeps the old model path unchanged when the
            # feature is disabled.
            raster_hidden_features = self.raster_encoder[0](raster_input)
            raster_hidden_features = self.raster_encoder[1](raster_hidden_features)
            raster_hidden_features = self.raster_encoder[2](raster_hidden_features)
            raster_hidden_features = self.raster_encoder[3](raster_hidden_features)
            raster_features = self.raster_encoder[4](raster_hidden_features)
            raster_features = self.raster_encoder[5](raster_features)
            spatial_features = raster_hidden_features.reshape(
                batch,
                time,
                config.model_dim,
                height,
                width,
            )
        else:
            raster_features = self.raster_encoder(raster_input)
        global_features_flat = global_features.reshape(flat_batch_time, config.global_dim)
        global_features_encoded = self.global_projection(global_features_flat)

        hand_features = self.public_hand_features(global_features)
        raw_entity_features = entities.reshape(flat_batch_time, entity_count, config.entity_dim)
        entity_mask_flat = entity_mask.reshape(flat_batch_time, entity_count)
        null_entity = self.null_entity.expand(flat_batch_time, -1, -1)
        # Hand cards are public one-hot table features projected per action
        # slot. The Transformer remains an entity-only contextualizer.
        #
        # In padded inference lanes, compact before the entity projection as
        # well as before attention. The projection is row-wise, so applying
        # it only to active rows is numerically equivalent while avoiding
        # work for masked entity padding. Training and dense inference retain
        # the declared tensor layout.
        if inference and not bool(entity_mask_flat.all().item()):
            pooled_entities = _inference_compact_entity_pool(
                self.entity_transformer,
                raw_entity_features,
                entity_mask_flat,
                null_entity,
                entity_projection=self.entity_projection,
            )
            transformed = None
        else:
            entity_features = self.entity_projection(raw_entity_features)
            null_mask = torch.zeros(
                (flat_batch_time, 1), dtype=torch.bool, device=entity_mask.device
            )
            transformer_input = torch.cat((entity_features, null_entity), dim=1)
            key_padding_mask = torch.cat((~entity_mask_flat, null_mask), dim=1)
            # A full public token set is common during batched inference.
            # Passing an all-false padding mask makes PyTorch build a nested
            # tensor even though there is nothing to pad. Keep the masked
            # path for genuine padding, but use the dense Transformer kernel
            # when every token is present; the attention computation is
            # otherwise identical.
            if bool(key_padding_mask.any().item()):
                transformed = self.entity_transformer(
                    transformer_input,
                    src_key_padding_mask=key_padding_mask,
                )
            elif inference:
                transformed = _inference_dense_entity_transformer(
                    self.entity_transformer,
                    transformer_input,
                    disable_mha_fastpath=True,
                )
            else:
                transformed = self.entity_transformer(transformer_input)
        if transformed is not None:
            transformed_entities = transformed[:, :entity_count]
            transformed_null = transformed[:, entity_count]
            present = entity_mask_flat.unsqueeze(-1)
            present_count = present.sum(dim=1)
            pooled_entities = (transformed_entities * present).sum(dim=1)
            pooled_entities = pooled_entities / present_count.clamp_min(1)
            pooled_entities = torch.where(
                present_count > 0,
                pooled_entities,
                transformed_null,
            )

        fusion_features = [raster_features, pooled_entities, global_features_encoded]
        if hand_features is not None:
            fusion_features.append(hand_features.reshape(
                flat_batch_time,
                config.card_slots * config.model_dim,
            ))
        fused = self.fusion(torch.cat(fusion_features, dim=-1))
        return (
            fused.reshape(batch, time, config.encoder_dim),
            spatial_features,
            hand_features,
        )


def _require_floating(name: str, value: torch.Tensor, *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _require_bool(name: str, value: torch.Tensor, *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use dtype torch.bool")


def _inference_dense_entity_transformer(
    transformer: nn.TransformerEncoder,
    inputs: torch.Tensor,
    *,
    disable_mha_fastpath: bool = False,
) -> torch.Tensor:
    """Run a dense CPU entity batch in sequence-first layout.

    ``TransformerEncoderLayer`` retains the same parameters and operation
    ordering in either layout.  On the deployment CPU workload, the regular
    sequence-first kernel avoids the batch-first fast-path dispatch overhead.
    Keep the established batch-first path for training, accelerators, and
    genuinely padded batches so sparse-observation behavior remains unchanged.

    PyTorch stores ``batch_first`` on each attention module rather than on the
    encoder call.  The flag is changed only for this synchronous inference
    call and restored in ``finally``; no parameters or checkpoint keys are
    modified.

    On the benchmark CPU, this sequence-first path is faster when PyTorch's
    native MHA fast path is disabled.  That backend switch is global, so the
    optional toggle is protected by a process-local lock and restored in
    ``finally``.  Compact sparse batches retain PyTorch's default dispatch.
    """

    if inputs.device.type != "cpu" or transformer.training:
        return transformer(inputs)
    layers = tuple(transformer.layers)
    if not layers:
        return transformer(inputs)
    attention_modules = tuple(layer.self_attn for layer in layers)
    previous_batch_first = tuple(
        bool(attention.batch_first) for attention in attention_modules
    )
    if not all(previous_batch_first):
        return transformer(inputs)
    mha_backend = getattr(torch.backends, "mha", None)
    can_toggle_mha = bool(
        disable_mha_fastpath
        and mha_backend is not None
        and hasattr(mha_backend, "get_fastpath_enabled")
        and hasattr(mha_backend, "set_fastpath_enabled")
    )
    lock_context = _INFERENCE_MHA_LOCK if can_toggle_mha else nullcontext()
    with lock_context:
        for attention in attention_modules:
            attention.batch_first = False
        previous_mha_fastpath: bool | None = None
        try:
            if can_toggle_mha:
                previous_mha_fastpath = bool(mha_backend.get_fastpath_enabled())
                mha_backend.set_fastpath_enabled(False)
            return transformer(inputs.transpose(0, 1)).transpose(0, 1)
        finally:
            if previous_mha_fastpath is not None:
                mha_backend.set_fastpath_enabled(previous_mha_fastpath)
            for attention, batch_first in zip(
                attention_modules,
                previous_batch_first,
                strict=True,
            ):
                attention.batch_first = batch_first


def _inference_compact_entity_pool(
    transformer: nn.TransformerEncoder,
    entity_features: torch.Tensor,
    entity_mask: torch.Tensor,
    null_entity: torch.Tensor,
    *,
    entity_projection: nn.Module | None = None,
) -> torch.Tensor:
    """Pool masked entity rows after compacting each inference lane.

    V2 padding is lane-local, but a regular batched Transformer must use the
    largest lane width.  Since the entity encoder has no positional encoding,
    masked rows can be removed before attention without changing the public
    entity set or its pooled representation.  When supplied, the projection
    is also applied after compaction so padded raw rows do not incur projection
    work. Equal active counts are grouped into one call to avoid a Transformer
    invocation per lane.
    """

    if entity_features.ndim != 3 or entity_mask.ndim != 2:
        raise ValueError("compact entity inputs must be [batch, entities, features] and [batch, entities]")
    if entity_features.shape[:2] != entity_mask.shape:
        raise ValueError("compact entity features and mask must share batch/entity dimensions")
    if null_entity.ndim != 3 or null_entity.shape[:2] != (entity_features.shape[0], 1):
        raise ValueError("null_entity must have shape [batch, 1, feature_dim]")

    batch, _entity_count, feature_dim = entity_features.shape
    output_dim = null_entity.shape[-1]
    if entity_projection is None and output_dim != feature_dim:
        raise ValueError("null_entity feature dimension must match entity features")
    active_counts = entity_mask.sum(dim=1)
    if not bool(active_counts.any().item()):
        if transformer.training:
            # ``act_deterministic`` is an evaluation API, but preserve the
            # reference semantics if a caller invokes it before ``eval()``.
            projected_entities = (
                entity_projection(entity_features)
                if entity_projection is not None
                else entity_features
            )
            inputs = torch.cat((projected_entities, null_entity), dim=1)
            padding = torch.cat(
                (
                    ~entity_mask,
                    torch.zeros(
                        (batch, 1),
                        dtype=torch.bool,
                        device=entity_mask.device,
                    ),
                ),
                dim=1,
            )
            transformed = transformer(inputs, src_key_padding_mask=padding)
            return transformed[:, -1]
        return _inference_cached_null_pool(transformer, null_entity)

    pooled = torch.empty(
        (batch, output_dim),
        dtype=entity_features.dtype,
        device=entity_features.device,
    )
    for count_tensor in torch.unique(active_counts, sorted=True):
        count = int(count_tensor.item())
        rows = torch.nonzero(active_counts == count_tensor, as_tuple=False).flatten()
        selected_mask = entity_mask.index_select(0, rows)
        selected_entities = entity_features.index_select(0, rows)
        if count:
            selected_entities = selected_entities[selected_mask].reshape(
                rows.shape[0],
                count,
                feature_dim,
            )
            if entity_projection is not None:
                selected_entities = entity_projection(selected_entities)
        else:
            selected_entities = entity_features.new_empty(
                (rows.shape[0], 0, output_dim)
            )
        selected_input = torch.cat(
            (selected_entities, null_entity.index_select(0, rows)),
            dim=1,
        )
        transformed = _inference_dense_entity_transformer(
            transformer,
            selected_input,
        )
        selected_pool = (
            transformed[:, :count].mean(dim=1)
            if count
            else transformed[:, 0]
        )
        pooled.index_copy_(0, rows, selected_pool)
    return pooled


def _inference_cached_null_pool(
    transformer: nn.TransformerEncoder,
    null_entity: torch.Tensor,
) -> torch.Tensor:
    """Return the transformed null token, caching unchanged inference weights."""

    parameters = tuple(transformer.parameters())
    cache_key = (
        str(null_entity.device),
        str(null_entity.dtype),
        int(null_entity._version),
        tuple(int(parameter._version) for parameter in parameters),
    )
    cache = getattr(transformer, "_inference_null_pool_cache", None)
    if cache is None or cache[0] != cache_key:
        transformed = _inference_dense_entity_transformer(
            transformer,
            null_entity[:1],
        )
        cache = (cache_key, transformed[:, 0].detach())
        # Keep this deployment-only tensor out of Module parameters, buffers,
        # and checkpoint state. The version key above invalidates it after an
        # optimizer update or checkpoint mutation.
        transformer._inference_null_pool_cache = cache
    return cache[1].expand(null_entity.shape[0], -1)


def _inference_raster_layout(raster: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    """Use the faster CPU convolution layout for deployment-only inference.

    The public raster contract remains ``[B, T, C, H, W]``.  This only changes
    the storage layout of the flattened frames consumed by the convolution;
    it does not alter values, model parameters, or the actor's computation.
    Training and the reference forward path deliberately retain their normal
    layout so this optimization cannot change PPO numerics.
    """

    if not isinstance(raster, torch.Tensor) or raster.ndim != 5:
        return raster
    if raster.device.type != "cpu":
        return raster
    batch, time, channels, height, width = raster.shape
    if (channels, height, width) != (
        config.raster_channels,
        config.raster_height,
        config.raster_width,
    ):
        return raster
    flat = raster.reshape(batch * time, channels, height, width)
    if flat.is_contiguous(memory_format=torch.channels_last):
        return raster
    return flat.contiguous(memory_format=torch.channels_last).reshape(
        batch,
        time,
        channels,
        height,
        width,
    )


def _inference_entity_tokens(
    entities: torch.Tensor,
    entity_mask: torch.Tensor,
    config: ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop only globally padded tail rows for deployment inference.

    V2 observations append zero-padded, masked rows.  A masked Transformer
    key cannot affect any valid query, so removing those tail rows is
    mathematically equivalent while reducing attention work for sparse game
    states.  The reference/training path keeps the declared tensor width.
    """

    if (
        not isinstance(entities, torch.Tensor)
        or not isinstance(entity_mask, torch.Tensor)
        or entities.device.type != "cpu"
        or (
            entities.ndim != 4
            or entity_mask.ndim != 3
            or entities.shape[:3] != entity_mask.shape
            or entities.shape[2] < 1
            or entities.shape[2] > config.max_entities
        )
    ):
        return entities, entity_mask
    present_columns = entity_mask.any(dim=(0, 1))
    if bool(present_columns[-1].item()):
        return entities, entity_mask
    if bool(present_columns.any().item()):
        active_count = int(present_columns.nonzero()[-1].item()) + 1
    else:
        # The model contract requires at least one entity row. Keep one
        # masked placeholder so the null token has the same role as before.
        active_count = 1
    if active_count >= entities.shape[2]:
        return entities, entity_mask
    return entities[..., :active_count, :], entity_mask[..., :active_count]


class GRURecurrentCore(nn.Module):
    """GRU core with explicit per-timestep episode reset semantics."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 1) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if type(layers) is not int or layers <= 0:
            raise ValueError("layers must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        return torch.zeros(
            self.layers,
            batch_size,
            self.hidden_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_floating("recurrent features", features, ndim=3)
        batch, time, input_dim = features.shape
        if input_dim != self.input_dim:
            raise ValueError(f"recurrent features final dimension must be {self.input_dim}")
        if time < 1:
            raise ValueError("recurrent sequence must contain at least one timestep")
        if hidden is None:
            hidden = self.initial_state(batch, device=features.device, dtype=features.dtype)
        else:
            _require_floating("hidden", hidden, ndim=3)
            if hidden.shape != (self.layers, batch, self.hidden_dim):
                raise ValueError(
                    "hidden must have shape [layers, batch, hidden_dim] "
                    f"= {(self.layers, batch, self.hidden_dim)}"
                )
        if reset_mask is None:
            reset_mask = torch.zeros((batch, time), dtype=torch.bool, device=features.device)
        else:
            _require_bool("reset_mask", reset_mask, ndim=2)
            if reset_mask.shape != (batch, time):
                raise ValueError("reset_mask must have shape [batch, time]")

        if time == 1:
            # Rollout inference always supplies one timestep. Calling the GRU
            # directly avoids the Python list/slice/concatenate loop used for
            # multi-step training sequences while preserving reset semantics.
            hidden = hidden.masked_fill(reset_mask[:, 0].reshape(1, batch, 1), 0.0)
            return self.gru(features, hidden)

        outputs: list[torch.Tensor] = []
        for timestep in range(time):
            reset = reset_mask[:, timestep].reshape(1, batch, 1)
            hidden = hidden.masked_fill(reset, 0.0)
            output, hidden = self.gru(features[:, timestep : timestep + 1], hidden)
            outputs.append(output)
        return torch.cat(outputs, dim=1), hidden


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"{name} logits and mask must have the same shape")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} mask must use dtype torch.bool")
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError(f"{name} contains a timestep with no legal action")
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    return F.log_softmax(masked_logits, dim=-1)


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    """Return a legal argmax without normalizing an unused distribution.

    Deterministic deployment only needs the ordering of legal logits.  The
    PPO path still uses :func:`_masked_log_softmax`; keeping this helper
    separate makes the inference shortcut explicit and preserves the same
    fail-closed validation for empty legality rows.
    """

    if logits.shape != mask.shape:
        raise ValueError(f"{name} logits and mask must have the same shape")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} mask must use dtype torch.bool")
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError(f"{name} contains a timestep with no legal action")
    return torch.where(mask, logits, torch.full_like(logits, -torch.inf)).argmax(dim=-1)


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """Make masked diagnostics finite for categories that are not selected.

    A PLAY mask can legitimately have no legal card/placement when WAIT is the
    only legal mode.  Those unselected distributions are never used by
    ``log_prob``; this helper gives callers finite diagnostic tensors without
    weakening legality checks for selected actions.
    """

    has_legal = mask.any(dim=-1, keepdim=True)
    return torch.where(has_legal, mask, torch.ones_like(mask))


class MaskedAutoregressivePolicy(nn.Module):
    """WAIT/PLAY -> card slot -> card-conditioned placement policy head."""

    WAIT = 0
    PLAY = 1

    def __init__(self, hidden_dim: int, config: ModelConfig) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.config = config
        self.card_slots = config.card_slots
        self.placement_rows = config.placement_rows
        self.placement_cols = config.placement_cols
        self.mode_head = nn.Linear(hidden_dim, 2)
        self.public_mode_head: nn.Module | None = None
        if config.direct_public_action_features:
            self.public_mode_head = nn.Linear(config.global_dim, 2)
        self.public_card_head: nn.Module | None = None
        self.contextual_public_card_features = bool(
            config.contextual_public_card_features
        )
        if config.direct_public_card_features:
            card_input_dim = config.global_dim + (
                hidden_dim if self.contextual_public_card_features else 0
            )
            self.public_card_head = nn.Sequential(
                nn.Linear(card_input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, config.card_slots),
            )
        self.public_slot_card_head: nn.Module | None = None
        if config.direct_public_slot_card_features:
            self.public_slot_card_head = nn.Sequential(
                nn.Linear(config.hand_card_count, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )
        self.public_mask_head: nn.Module | None = None
        if config.direct_public_mask_features:
            self.public_mask_head = nn.Sequential(
                nn.Linear(config.card_slots + 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2),
            )
        self.public_context_head: nn.Module | None = None
        if config.direct_public_context_features:
            self.public_context_head = nn.Sequential(
                nn.Linear(config.global_dim + config.card_slots + 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2),
            )
        self.card_head = nn.Linear(hidden_dim, config.card_slots)
        # This is positional context for the four action slots. Card identity
        # comes from the projected one-hot card-table features in
        # ``hand_features``.
        self.card_embedding = nn.Embedding(config.card_slots, hidden_dim)
        self.hand_card_score: nn.Module | None = None
        if config.hand_feature_offset >= 0:
            self.hand_card_score = nn.Linear(hidden_dim, 1)
        self.placement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.placement_rows * config.placement_cols),
        )
        self.spatial_placement_key: nn.Module | None = None
        self.spatial_placement_query: nn.Module | None = None
        if config.spatial_placement_features:
            self.spatial_placement_key = nn.Conv2d(
                config.model_dim,
                config.spatial_placement_dim,
                kernel_size=1,
            )
            self.spatial_placement_query = nn.Linear(
                hidden_dim,
                config.spatial_placement_dim,
            )

    def _prepare_action_prefix(
        self,
        recurrent_features: torch.Tensor,
        hand_features: torch.Tensor | None,
        public_features: torch.Tensor | None,
        public_action_masks: ActionMasks | None,
        spatial_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mode/card logits and card-conditioned contexts once."""

        mode_logits = self._prepare_mode_logits(
            recurrent_features,
            public_features,
            public_action_masks,
        )
        card_logits, card_context = self._prepare_card_prefix(
            recurrent_features,
            hand_features,
            public_features,
        )
        if self.spatial_placement_key is not None:
            self._validate_spatial_features(recurrent_features, spatial_features)
        return mode_logits, card_logits, card_context

    def _prepare_mode_logits(
        self,
        recurrent_features: torch.Tensor,
        public_features: torch.Tensor | None,
        public_action_masks: ActionMasks | None,
    ) -> torch.Tensor:
        """Build only the mode logits needed before a PLAY branch is known."""

        _require_floating("recurrent features", recurrent_features, ndim=3)
        if recurrent_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"recurrent features final dimension must be {self.hidden_dim}")
        mode_logits = self.mode_head(recurrent_features)
        if self.public_mode_head is not None:
            if public_features is None:
                raise ValueError(
                    "direct public action features require global observation features"
                )
            _require_floating("public action features", public_features, ndim=3)
            if public_features.shape[:2] != recurrent_features.shape[:2]:
                raise ValueError(
                    "public action features must share recurrent batch/time dimensions"
                )
            if public_features.shape[-1] != self.public_mode_head.in_features:
                raise ValueError(
                    "public action feature width does not match the configured model"
                )
            # In the direct-public variant the public stream is the primary
            # source for the mode decision.  Adding it to a recurrent mode
            # head lets the placement/card gradients perturb the shared GRU
            # until a WAIT-heavy target collapses into PLAY (or vice versa).
            # Keep the legacy recurrent path unchanged when the option is off;
            # fresh direct actors get a stable, independently trainable gate.
            mode_logits = self.public_mode_head(public_features)
        elif public_features is not None and public_features.shape[:2] != recurrent_features.shape[:2]:
            raise ValueError(
                "public action features must share recurrent batch/time dimensions"
            )

        public_mask_features: torch.Tensor | None = None
        if self.public_mask_head is not None or self.public_context_head is not None:
            if public_action_masks is None:
                raise ValueError(
                    "direct public legality features require public action masks"
                )
            if not isinstance(public_action_masks, ActionMasks):
                raise TypeError("public_action_masks must be an ActionMasks instance")
            if public_action_masks.prefix_shape != tuple(recurrent_features.shape[:2]):
                raise ValueError(
                    "public action masks must share recurrent batch/time dimensions"
                )
            if public_action_masks.card_slots != self.card_slots:
                raise ValueError("public action masks have the wrong card-slot count")
            public_mask_features = torch.cat(
                (
                    public_action_masks.mode.to(
                        device=recurrent_features.device,
                        dtype=recurrent_features.dtype,
                    ),
                    public_action_masks.card.to(
                        device=recurrent_features.device,
                        dtype=recurrent_features.dtype,
                    ),
                ),
                dim=-1,
            )
            if self.public_mask_head is not None:
                mode_logits = mode_logits + self.public_mask_head(public_mask_features)
        if self.public_context_head is not None:
            if public_features is None or public_mask_features is None:
                raise ValueError(
                    "direct public context features require public features and masks"
                )
            mode_logits = self.public_context_head(
                torch.cat((public_features, public_mask_features), dim=-1)
            )
        return mode_logits

    def _validate_spatial_features(
        self,
        recurrent_features: torch.Tensor,
        spatial_features: torch.Tensor | None,
    ) -> None:
        if spatial_features is None:
            raise ValueError(
                "spatial placement features are required by this model variant"
            )
        _require_floating("spatial placement features", spatial_features, ndim=5)
        if spatial_features.shape[:2] != recurrent_features.shape[:2]:
            raise ValueError(
                "spatial placement features must share recurrent batch/time dimensions"
            )
        if spatial_features.shape[2] != self.config.model_dim:
            raise ValueError(
                "spatial placement feature channels must match ModelConfig.model_dim"
            )

    def _prepare_card_prefix(
        self,
        recurrent_features: torch.Tensor,
        hand_features: torch.Tensor | None,
        public_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build card logits and contexts after the mode selects PLAY."""

        _require_floating("recurrent features", recurrent_features, ndim=3)
        if recurrent_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"recurrent features final dimension must be {self.hidden_dim}")
        if hand_features is not None:
            if self.hand_card_score is None:
                raise ValueError("hand features require an explicit-hand model variant")
            if hand_features.shape != (
                *recurrent_features.shape[:2],
                self.card_slots,
                self.hidden_dim,
            ):
                raise ValueError(
                    "hand_features must have shape [batch, time, card_slots, hidden_dim]"
                )
        card_logits = self.card_head(recurrent_features)
        if self.public_card_head is not None:
            if public_features is None:
                raise ValueError(
                    "direct public card features require global observation features"
                )
            _require_floating("public card features", public_features, ndim=3)
            if public_features.shape[:2] != recurrent_features.shape[:2]:
                raise ValueError(
                    "public card features must share recurrent batch/time dimensions"
                )
            card_features = public_features
            if self.contextual_public_card_features:
                card_features = torch.cat((public_features, recurrent_features), dim=-1)
            if card_features.shape[-1] != self.public_card_head[0].in_features:
                raise ValueError(
                    "public card feature width does not match the configured model"
                )
            # The same separation applies to card identity.  Placement still
            # uses the recurrent/entity stream below, while the explicit
            # public hand/cost stream owns the card-slot decision.
            card_logits = self.public_card_head(card_features)
        if self.public_slot_card_head is not None:
            if public_features is None:
                raise ValueError(
                    "direct public slot-card features require global observation features"
                )
            config = self.config
            hand_end = config.hand_feature_offset + (
                config.card_slots * config.hand_card_count
            )
            flat_public = public_features.reshape(-1, public_features.shape[-1])
            raw_hand = flat_public[:, config.hand_feature_offset:hand_end].reshape(
                *public_features.shape[:2],
                config.card_slots,
                config.hand_card_count,
            )
            slot_scores = self.public_slot_card_head(raw_hand).squeeze(-1)
            card_logits = card_logits + slot_scores
        card_context = recurrent_features.unsqueeze(-2) + self.card_embedding.weight.view(
            1, 1, self.card_slots, self.hidden_dim
        )
        if hand_features is not None:
            card_logits = card_logits + self.hand_card_score(hand_features).squeeze(-1)
            card_context = card_context + hand_features

        return card_logits, card_context

    def _placement_logits(
        self,
        card_context: torch.Tensor,
        spatial_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode placement logits for the card contexts supplied by a caller."""

        # ``card_context`` is normally [B, T, K, H]. The selected-card fast
        # path supplies K=1; the card dimension is already present in the
        # linear output and must not be appended a second time.
        placement_logits = self.placement_head(card_context).reshape(
            *card_context.shape[:3],
            self.placement_rows,
            self.placement_cols,
        )
        if self.spatial_placement_key is not None:
            if self.spatial_placement_query is None:  # pragma: no cover - invariant
                raise RuntimeError("spatial placement query is not initialized")
            if spatial_features is None:  # pragma: no cover - validated by prefix
                raise RuntimeError("spatial placement features are not initialized")
            batch, time = card_context.shape[:2]
            spatial = spatial_features.reshape(
                batch * time,
                self.config.model_dim,
                spatial_features.shape[-2],
                spatial_features.shape[-1],
            )
            if spatial.shape[-2:] != (self.placement_rows, self.placement_cols):
                spatial = F.interpolate(
                    spatial,
                    size=(self.placement_rows, self.placement_cols),
                    mode="bilinear",
                    align_corners=False,
                )
            keys = self.spatial_placement_key(spatial).reshape(
                batch,
                time,
                self.config.spatial_placement_dim,
                self.placement_rows,
                self.placement_cols,
            )
            queries = self.spatial_placement_query(card_context)
            placement_logits = placement_logits + torch.einsum(
                "btsd,btdrc->btsrc",
                queries,
                keys,
            ) / math.sqrt(float(self.config.spatial_placement_dim))
        return placement_logits

    def forward(
        self,
        recurrent_features: torch.Tensor,
        hand_features: torch.Tensor | None = None,
        public_features: torch.Tensor | None = None,
        public_action_masks: ActionMasks | None = None,
        spatial_features: torch.Tensor | None = None,
    ) -> AutoregressiveLogits:
        mode_logits, card_logits, card_context = self._prepare_action_prefix(
            recurrent_features,
            hand_features,
            public_features,
            public_action_masks,
            spatial_features,
        )
        placement_logits = self._placement_logits(card_context, spatial_features)
        return AutoregressiveLogits(mode_logits, card_logits, placement_logits)

    def sample_fast(
        self,
        recurrent_features: torch.Tensor,
        hand_features: torch.Tensor | None,
        public_features: torch.Tensor | None,
        public_action_masks: ActionMasks,
        spatial_features: torch.Tensor | None,
    ) -> tuple[ActionBatch, torch.Tensor, torch.Tensor, AutoregressiveLogits]:
        """Sample a rollout action without decoding unused placements.

        The hierarchical sampling order is deliberately identical to
        :meth:`sample`: mode first, then card for PLAY rows, then placement for
        the selected card.  The full PPO forward path remains the reference
        implementation; this method only avoids evaluating the three unused
        card-placement branches during actor-controlled collection.
        """

        _require_floating("recurrent features", recurrent_features, ndim=3)
        if public_action_masks.prefix_shape != tuple(recurrent_features.shape[:2]):
            raise ValueError("action masks must share recurrent batch/time dimensions")
        if public_action_masks.card_slots != self.card_slots:
            raise ValueError("action masks have the wrong card-slot count")

        mode_logits = self._prepare_mode_logits(
            recurrent_features,
            public_features,
            public_action_masks,
        )
        mode_log_probs = _masked_log_softmax(
            mode_logits,
            public_action_masks.mode,
            "mode",
        )
        mode_distribution = torch.distributions.Categorical(logits=mode_log_probs)
        mode = mode_distribution.sample()
        card_slot = torch.zeros_like(mode, dtype=torch.long)
        placement = torch.zeros(
            (*mode.shape, 2),
            dtype=torch.long,
            device=mode.device,
        )
        joint_log_probs = mode_distribution.log_prob(mode)
        entropy = mode_distribution.entropy()

        card_logits = torch.zeros(
            (*mode.shape, self.card_slots),
            dtype=recurrent_features.dtype,
            device=recurrent_features.device,
        )
        placement_logits = torch.zeros(
            (
                *mode.shape,
                self.card_slots,
                self.placement_rows,
                self.placement_cols,
            ),
            dtype=recurrent_features.dtype,
            device=recurrent_features.device,
        )

        play = mode == self.PLAY
        if bool(play.any().item()):
            # Card and placement heads cannot affect WAIT rows. Restrict both
            # computations to PLAY rows, then restrict placement to the card
            # sampled from each of those rows.
            play_recurrent = recurrent_features[play].unsqueeze(1)
            play_hand = (
                None if hand_features is None else hand_features[play].unsqueeze(1)
            )
            play_public = (
                None if public_features is None else public_features[play].unsqueeze(1)
            )
            play_card_logits, play_card_context = self._prepare_card_prefix(
                play_recurrent,
                play_hand,
                play_public,
            )
            play_card_logits = play_card_logits[:, 0]
            play_card_context = play_card_context[:, 0]
            play_card_masks = public_action_masks.card[play]
            play_card_log_probs = _masked_log_softmax(
                play_card_logits,
                play_card_masks,
                "card",
            )
            card_distribution = torch.distributions.Categorical(
                logits=play_card_log_probs,
            )
            selected_cards = card_distribution.sample()
            card_slot[play] = selected_cards
            joint_log_probs = joint_log_probs.clone()
            joint_log_probs[play] += card_distribution.log_prob(selected_cards)
            entropy = entropy.clone()
            entropy[play] += card_distribution.entropy()
            card_logits[play] = play_card_logits

            selected_context = play_card_context.gather(
                1,
                selected_cards.reshape(-1, 1, 1).expand(
                    -1,
                    1,
                    self.hidden_dim,
                ),
            ).unsqueeze(1)
            play_spatial = (
                None
                if spatial_features is None
                else spatial_features[play].unsqueeze(1)
            )
            if self.spatial_placement_key is not None:
                self._validate_spatial_features(play_recurrent, play_spatial)
            selected_placement_logits = self._placement_logits(
                selected_context,
                play_spatial,
            ).squeeze(2).squeeze(1)
            selected_masks = public_action_masks.placement[play].gather(
                1,
                selected_cards.reshape(-1, 1, 1, 1).expand(
                    -1,
                    1,
                    self.placement_rows,
                    self.placement_cols,
                ),
            ).squeeze(1)
            placement_log_probs = _masked_log_softmax(
                selected_placement_logits.reshape(selected_cards.shape[0], -1),
                selected_masks.reshape(selected_cards.shape[0], -1),
                "placement",
            )
            placement_distribution = torch.distributions.Categorical(
                logits=placement_log_probs,
            )
            selected_cells = placement_distribution.sample()
            selected_placements = placement[play]
            selected_placements[:, 0] = torch.div(
                selected_cells,
                self.placement_cols,
                rounding_mode="floor",
            )
            selected_placements[:, 1] = selected_cells.remainder(self.placement_cols)
            placement[play] = selected_placements
            joint_log_probs[play] += placement_distribution.log_prob(selected_cells)
            entropy[play] += placement_distribution.entropy()

            selected_rows = placement_logits[play]
            selected_rows.scatter_(
                1,
                selected_cards.reshape(-1, 1, 1, 1).expand(
                    -1,
                    1,
                    self.placement_rows,
                    self.placement_cols,
                ),
                selected_placement_logits.unsqueeze(1),
            )
            placement_logits[play] = selected_rows

        return (
            ActionBatch(mode=mode, card_slot=card_slot, placement=placement),
            joint_log_probs,
            entropy,
            AutoregressiveLogits(mode_logits, card_logits, placement_logits),
        )

    def deterministic_action_fast(
        self,
        recurrent_features: torch.Tensor,
        hand_features: torch.Tensor | None,
        public_features: torch.Tensor | None,
        public_action_masks: ActionMasks,
        spatial_features: torch.Tensor | None,
    ) -> ActionBatch:
        """Select a deterministic legal action without decoding unused cards.

        The full :meth:`forward` path remains the reference/training ABI. This
        deployment path computes the same masked mode and card argmaxes, then
        decodes placement only for rows that selected PLAY and only for their
        selected card slot. It deliberately returns actions only because
        deployment callers do not need PPO logits, entropy, or log-probability
        diagnostics.
        """

        _require_floating("recurrent features", recurrent_features, ndim=3)
        if public_action_masks.prefix_shape != tuple(recurrent_features.shape[:2]):
            raise ValueError("action masks must share recurrent batch/time dimensions")
        if public_action_masks.card_slots != self.card_slots:
            raise ValueError("action masks have the wrong card-slot count")

        mode_logits = self._prepare_mode_logits(
            recurrent_features,
            public_features,
            public_action_masks,
        )
        mode = _masked_argmax(
            mode_logits,
            public_action_masks.mode,
            "mode",
        )
        card_slot = torch.zeros_like(mode, dtype=torch.long)
        placement = torch.zeros(
            (*mode.shape, 2),
            dtype=torch.long,
            device=mode.device,
        )
        play = mode == self.PLAY
        if bool(play.any().item()):
            # Card and placement heads cannot affect a WAIT action. Restrict
            # them to PLAY rows after the mode decision so wait-heavy batches
            # avoid the unused card projections and context construction.
            play_recurrent = recurrent_features[play].unsqueeze(1)
            play_hand = (
                None if hand_features is None else hand_features[play].unsqueeze(1)
            )
            play_public = (
                None if public_features is None else public_features[play].unsqueeze(1)
            )
            play_card_logits, play_card_context = self._prepare_card_prefix(
                play_recurrent,
                play_hand,
                play_public,
            )
            play_spatial = (
                None
                if spatial_features is None
                else spatial_features[play].unsqueeze(1)
            )
            if self.spatial_placement_key is not None:
                self._validate_spatial_features(play_recurrent, play_spatial)
            play_card_logits = play_card_logits[:, 0]
            play_card_context = play_card_context[:, 0]
            play_card_masks = public_action_masks.card[play]
            selected_cards = _masked_argmax(
                play_card_logits,
                play_card_masks,
                "card",
            )
            card_slot[play] = selected_cards

            selected_context = play_card_context.gather(
                1,
                selected_cards.reshape(-1, 1, 1).expand(
                    -1,
                    1,
                    self.hidden_dim,
                ),
            ).unsqueeze(1)
            selected_placement = self._placement_logits(
                selected_context,
                play_spatial,
            ).squeeze(2)
            selected_masks = public_action_masks.placement[play].gather(
                1,
                selected_cards.reshape(-1, 1, 1, 1).expand(
                    -1,
                    1,
                    self.placement_rows,
                    self.placement_cols,
                ),
            ).squeeze(1)
            cells = _masked_argmax(
                selected_placement.reshape(selected_cards.shape[0], -1),
                selected_masks.reshape(selected_cards.shape[0], -1),
                "placement",
            )
            selected_placements = placement[play]
            selected_placements[:, 0] = torch.div(
                cells,
                self.placement_cols,
                rounding_mode="floor",
            )
            selected_placements[:, 1] = cells.remainder(self.placement_cols)
            placement[play] = selected_placements
        return ActionBatch(mode=mode, card_slot=card_slot, placement=placement)

    def sample(
        self,
        logits: AutoregressiveLogits,
        masks: ActionMasks,
    ) -> tuple[ActionBatch, torch.Tensor, torch.Tensor]:
        """Sample legal hierarchical actions and return log-probability/entropy."""

        mode_log_probs = _masked_log_softmax(logits.mode, masks.mode, "mode")
        mode_distribution = torch.distributions.Categorical(logits=mode_log_probs)
        mode = mode_distribution.sample()
        card_slot = torch.zeros_like(mode, dtype=torch.long)
        placement = torch.zeros((*mode.shape, 2), dtype=torch.long, device=mode.device)
        joint_log_probs = mode_distribution.log_prob(mode)
        entropy = mode_distribution.entropy()

        play = mode == self.PLAY
        if bool(play.any().item()):
            play_card_logits = logits.card[play]
            play_card_masks = masks.card[play]
            play_card_log_probs = _masked_log_softmax(
                play_card_logits,
                play_card_masks,
                "card",
            )
            card_distribution = torch.distributions.Categorical(logits=play_card_log_probs)
            selected_cards = card_distribution.sample()
            card_slot[play] = selected_cards
            joint_log_probs = joint_log_probs.clone()
            joint_log_probs[play] += card_distribution.log_prob(selected_cards)
            entropy = entropy.clone()
            entropy[play] += card_distribution.entropy()

            play_placement_logits = logits.placement[play]
            play_placement_masks = masks.placement[play]
            sample_number = torch.arange(
                selected_cards.shape[0],
                device=selected_cards.device,
            )
            selected_logits = play_placement_logits[sample_number, selected_cards]
            selected_masks = play_placement_masks[sample_number, selected_cards]
            placement_log_probs = _masked_log_softmax(
                selected_logits.reshape(selected_cards.shape[0], -1),
                selected_masks.reshape(selected_cards.shape[0], -1),
                "placement",
            )
            placement_distribution = torch.distributions.Categorical(logits=placement_log_probs)
            selected_cells = placement_distribution.sample()
            selected_placements = placement[play]
            selected_placements[:, 0] = torch.div(
                selected_cells,
                self.placement_cols,
                rounding_mode="floor",
            )
            selected_placements[:, 1] = selected_cells.remainder(self.placement_cols)
            placement[play] = selected_placements
            joint_log_probs[play] += placement_distribution.log_prob(selected_cells)
            entropy[play] += placement_distribution.entropy()

        actions = ActionBatch(mode=mode, card_slot=card_slot, placement=placement)
        # Re-evaluate through the validation path so the stored old log-prob
        # has exactly the same factorization used by PPO updates.
        return actions, self.log_prob(logits, actions, masks), entropy

    def masked_log_probs(
        self,
        logits: AutoregressiveLogits,
        masks: ActionMasks,
    ) -> AutoregressiveLogits:
        """Return per-factor masked log probabilities for diagnostics."""

        mode = _masked_log_softmax(logits.mode, masks.mode, "mode")
        card = _masked_log_softmax(logits.card, _safe_mask(masks.card), "card")
        placement_shape = logits.placement.shape
        placement_flat = logits.placement.reshape(*placement_shape[:-2], -1)
        placement_mask = masks.placement.reshape(*masks.placement.shape[:-2], -1)
        placement = _masked_log_softmax(
            placement_flat,
            _safe_mask(placement_mask),
            "placement",
        )
        return AutoregressiveLogits(
            mode,
            card,
            placement.reshape(placement_shape),
        )

    def log_prob(
        self,
        logits: AutoregressiveLogits,
        actions: ActionBatch,
        masks: ActionMasks,
    ) -> torch.Tensor:
        """Compose the joint log probability and reject illegal actions."""

        if logits.mode.shape[:-1] != actions.mode.shape:
            raise ValueError("action and logits batch/time dimensions do not match")
        if masks.prefix_shape != tuple(actions.mode.shape):
            raise ValueError("action masks and actions batch/time dimensions do not match")
        mode = actions.mode.to(dtype=torch.long)
        if bool(((mode < 0) | (mode > 1)).any().item()):
            raise ValueError("mode action must be WAIT=0 or PLAY=1")
        mode_log_probs = _masked_log_softmax(logits.mode, masks.mode, "mode")
        selected_mode_legal = masks.mode.gather(-1, mode.unsqueeze(-1)).squeeze(-1)
        if not bool(selected_mode_legal.all().item()):
            raise ValueError("action selected an illegal WAIT/PLAY mode")
        result = mode_log_probs.gather(-1, mode.unsqueeze(-1)).squeeze(-1)

        play = mode == self.PLAY
        if not bool(play.any().item()):
            return result

        play_card_logits = logits.card[play]
        play_card_masks = masks.card[play]
        play_card_log_probs = _masked_log_softmax(play_card_logits, play_card_masks, "card")
        card_indices = actions.card_slot[play].to(dtype=torch.long)
        if bool(((card_indices < 0) | (card_indices >= self.card_slots)).any().item()):
            raise ValueError("card-slot action is outside the configured card vocabulary")
        selected_card_legal = play_card_masks.gather(-1, card_indices.unsqueeze(-1)).squeeze(-1)
        if not bool(selected_card_legal.all().item()):
            raise ValueError("action selected an illegal card slot")
        selected_card_log_probs = play_card_log_probs.gather(
            -1,
            card_indices.unsqueeze(-1),
        ).squeeze(-1)

        play_placement_logits = logits.placement[play]
        play_placement_masks = masks.placement[play]
        row_col = actions.placement[play].to(dtype=torch.long)
        rows = row_col[:, 0]
        columns = row_col[:, 1]
        if bool(
            ((rows < 0) | (rows >= self.placement_rows) | (columns < 0) | (columns >= self.placement_cols))
            .any()
            .item()
        ):
            raise ValueError("placement action is outside the configured grid")
        sample_indices = rows * self.placement_cols + columns
        sample_number = torch.arange(
            card_indices.shape[0],
            device=card_indices.device,
        )
        selected_placement_logits = play_placement_logits[sample_number, card_indices]
        selected_placement_masks = play_placement_masks[sample_number, card_indices]
        selected_placement_log_probs = _masked_log_softmax(
            selected_placement_logits.reshape(card_indices.shape[0], -1),
            selected_placement_masks.reshape(card_indices.shape[0], -1),
            "placement",
        )
        selected_placement_legal = selected_placement_masks.reshape(card_indices.shape[0], -1).gather(
            -1,
            sample_indices.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(selected_placement_legal.all().item()):
            raise ValueError("action selected an illegal placement cell")
        selected_placement_log_probs = selected_placement_log_probs.gather(
            -1,
            sample_indices.unsqueeze(-1),
        ).squeeze(-1)
        result = result.clone()
        result[play] = result[play] + selected_card_log_probs + selected_placement_log_probs
        return result


class RecurrentHybridPolicy(nn.Module):
    """Hybrid observation encoder, GRU memory, and masked action head."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = HybridEncoder(config)
        self.core = GRURecurrentCore(
            input_dim=config.encoder_dim,
            hidden_dim=config.gru_hidden_dim,
            layers=config.gru_layers,
        )
        self.hand_action_projection: nn.Module | None = None
        if config.hand_feature_offset >= 0:
            self.hand_action_projection = nn.Sequential(
                nn.Linear(config.model_dim, config.gru_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.gru_hidden_dim),
            )
        self.action_head = MaskedAutoregressivePolicy(config.gru_hidden_dim, config)
        self.belief_heads = OpponentBeliefHeads(
            config.gru_hidden_dim,
            card_count=config.belief_card_count,
        )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.core.initial_state(batch_size, device=device, dtype=dtype)

    def _encode_recurrent_features(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None,
        hidden: torch.Tensor | None,
        inference: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Encode public inputs once for full or rollout-only action paths."""

        encoded_features, spatial_features, hand_features = self.encoder.forward_with_aux(
            raster,
            global_features,
            entities,
            entity_mask,
            inference=inference,
        )
        recurrent_features, final_hidden = self.core(
            encoded_features,
            hidden=hidden,
            reset_mask=reset_mask,
        )
        if hand_features is not None:
            if self.hand_action_projection is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("explicit hand features have no action projection")
            hand_features = self.hand_action_projection(hand_features)
        return (
            encoded_features,
            spatial_features,
            recurrent_features,
            final_hidden,
            hand_features,
        )

    def forward(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        action_masks: ActionMasks | None = None,
        include_beliefs: bool = True,
        inference: bool = False,
    ) -> RecurrentPolicyOutput:
        if type(include_beliefs) is not bool:
            raise TypeError("include_beliefs must be boolean")
        if type(inference) is not bool:
            raise TypeError("inference must be boolean")
        (
            encoded_features,
            spatial_features,
            recurrent_features,
            final_hidden,
            hand_features,
        ) = self._encode_recurrent_features(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            hidden=hidden,
            inference=inference,
        )
        logits = self.action_head(
            recurrent_features,
            hand_features=hand_features,
            public_features=global_features,
            public_action_masks=action_masks,
            spatial_features=spatial_features,
        )
        belief_logits = self.belief_heads(recurrent_features) if include_beliefs else None
        return RecurrentPolicyOutput(
            logits=logits,
            encoded_features=encoded_features,
            recurrent_features=recurrent_features,
            final_hidden=final_hidden,
            belief_logits=belief_logits,
            hand_features=hand_features,
            spatial_features=spatial_features,
        )

    def rollout_sample(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        action_masks: ActionMasks,
        *,
        reset_mask: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        include_beliefs: bool = False,
        inference: bool = False,
    ) -> tuple[RecurrentPolicyOutput, ActionBatch, torch.Tensor, torch.Tensor]:
        """Run the actor-controlled stochastic rollout path.

        This keeps the encoder, recurrent core, legality masks, and sampled
        distributions identical to :meth:`forward`/``sample``. Only the
        unselected card-placement branches are omitted. The returned logits
        contain exact mode/card logits and the selected placement logits; the
        method is intended for rollout collection, not PPO re-evaluation.
        """

        if type(include_beliefs) is not bool:
            raise TypeError("include_beliefs must be boolean")
        if type(inference) is not bool:
            raise TypeError("inference must be boolean")
        (
            encoded_features,
            spatial_features,
            recurrent_features,
            final_hidden,
            hand_features,
        ) = self._encode_recurrent_features(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            hidden=hidden,
            inference=inference,
        )
        actions, log_probs, entropy, logits = self.action_head.sample_fast(
            recurrent_features,
            hand_features,
            global_features,
            action_masks,
            spatial_features,
        )
        belief_logits = self.belief_heads(recurrent_features) if include_beliefs else None
        output = RecurrentPolicyOutput(
            logits=logits,
            encoded_features=encoded_features,
            recurrent_features=recurrent_features,
            final_hidden=final_hidden,
            belief_logits=belief_logits,
            hand_features=hand_features,
            spatial_features=spatial_features,
        )
        return output, actions, log_probs, entropy

    def act_deterministic(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        action_masks: ActionMasks,
        *,
        reset_mask: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[ActionBatch, torch.Tensor]:
        """Run the deployment actor without unused training outputs.

        This keeps the same encoder, recurrent core, action parameters, and
        legality masks as :meth:`forward`. It omits belief logits and uses the
        action head's selected-card placement path, so it is suitable for
        deterministic evaluation and live shadow/self-play inference.
        """

        inference_raster = _inference_raster_layout(raster, self.config)
        inference_entities, inference_entity_mask = _inference_entity_tokens(
            entities,
            entity_mask,
            self.config,
        )
        encoded_features, spatial_features, hand_features = self.encoder.forward_with_aux(
            inference_raster,
            global_features,
            inference_entities,
            inference_entity_mask,
            inference=True,
        )
        recurrent_features, final_hidden = self.core(
            encoded_features,
            hidden=hidden,
            reset_mask=reset_mask,
        )
        if hand_features is not None:
            if self.hand_action_projection is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("explicit hand features have no action projection")
            hand_features = self.hand_action_projection(hand_features)
        actions = self.action_head.deterministic_action_fast(
            recurrent_features,
            hand_features,
            global_features,
            action_masks,
            spatial_features,
        )
        return actions, final_hidden

    def log_prob(
        self,
        output: RecurrentPolicyOutput,
        actions: ActionBatch,
        masks: ActionMasks,
    ) -> torch.Tensor:
        """Evaluate a rollout action against the output of :meth:`forward`."""

        return self.action_head.log_prob(output.logits, actions, masks)


__all__ = [
    "ActionBatch",
    "ActionMasks",
    "AutoregressiveLogits",
    "GRURecurrentCore",
    "HybridEncoder",
    "MaskedAutoregressivePolicy",
    "ModelConfig",
    "OpponentBeliefHeads",
    "OpponentBeliefLogits",
    "PrivilegedCritic",
    "RecurrentHybridPolicy",
    "RecurrentPolicyOutput",
]
