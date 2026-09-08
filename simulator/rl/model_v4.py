"""V4 hierarchical recurrent policy: current-state-first actor.

Differences from the V3 actor (``rl.model``) that this module implements:

1. The spatial CNN keeps a board-aligned feature map at full ``32 x 18``
   resolution all the way to placement scoring; a pooled summary feeds
   fusion, but placement never passes through the global-pool bottleneck.
2. Every action head receives the fused *current* public state ``zt`` plus
   the recurrent memory ``ht`` (``[zt, ht]``), so immediate card, elixir,
   threat, and geometry facts cannot be lost to stale recurrent preference.
3. The four hand cards are card-conditioned query tokens over battlefield
   context, not a four-logit readout from one undifferentiated vector.
4. Placement is a card-conditioned dot-product decoder over the spatial map
   (one selected-card map only, masked softmax over legal cells).
5. WAIT is a semi-Markov action with a learned duration head over
   ``WAIT_DURATIONS = (1, 2, 4, 8)`` base decision ticks.

Inputs are ``[batch, time, ...]`` sequences of ``PolicyObservationV3``
tensors plus legality masks from :mod:`rl.trajectory`.  No hidden
simulator state enters this module; the privileged critic stays separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._compat import TorchUnavailableError

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise TorchUnavailableError(
            "rl.model_v4 requires PyTorch. Install torch to use the V4 "
            "hierarchical recurrent policy modules."
        ) from exc
    raise

from .trajectory import ActionBatch, ActionMasks


WAIT_DURATIONS: tuple[int, ...] = (1, 2, 4, 8)
"""Learned WAIT durations in base decision ticks (paper section 3.7)."""


@dataclass(frozen=True, slots=True)
class ModelConfigV4:
    """Dimensions for the V4 encoder, recurrent core, and action heads."""

    raster_channels: int = 21
    raster_height: int = 32
    raster_width: int = 18
    global_dim: int = 768
    entity_dim: int = 32
    max_entities: int = 128
    hand_token_dim: int = 16
    belief_cards: int = 128
    event_history_len: int = 16
    event_dim: int = 8
    model_dim: int = 128
    spatial_channels: int = 64
    fused_dim: int = 256
    gru_hidden_dim: int = 256
    gru_layers: int = 1
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 256
    card_slots: int = 4
    placement_rows: int = 32
    placement_cols: int = 18
    spatial_head_dim: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "raster_channels",
            "raster_height",
            "raster_width",
            "global_dim",
            "entity_dim",
            "max_entities",
            "hand_token_dim",
            "belief_cards",
            "event_history_len",
            "event_dim",
            "model_dim",
            "spatial_channels",
            "fused_dim",
            "gru_hidden_dim",
            "gru_layers",
            "transformer_heads",
            "transformer_layers",
            "transformer_ff_dim",
            "card_slots",
            "placement_rows",
            "placement_cols",
            "spatial_head_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_dim % self.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class V4ActionBatch:
    """V4 actions: V3 factorization plus the learned WAIT duration index."""

    mode: torch.Tensor
    card_slot: torch.Tensor
    placement: torch.Tensor
    wait_duration: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("mode", "card_slot", "placement", "wait_duration"):
            value = object.__getattribute__(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.card_slot.shape != self.mode.shape:
            raise ValueError("card_slot shape must match mode shape")
        if self.wait_duration.shape != self.mode.shape:
            raise ValueError("wait_duration shape must match mode shape")
        if self.placement.shape[:-1] != self.mode.shape or self.placement.shape[-1] != 2:
            raise ValueError("placement must have shape mode.shape + (2,)")


@dataclass(frozen=True, slots=True)
class V4Logits:
    """Unnormalized logits for every V4 action factor."""

    mode: torch.Tensor
    wait_duration: torch.Tensor
    card: torch.Tensor
    placement: torch.Tensor


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError("logits and mask shapes must match")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError("mask contains a distribution with no legal action")
    masked = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
    return F.log_softmax(masked, dim=-1)


class SpatialEncoder(nn.Module):
    """Board-aligned CNN; the unpooled map survives to placement decoding."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        self.body = nn.Sequential(
            nn.Conv2d(config.raster_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, config.spatial_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pooled [B, T, C], spatial [B, T, C, H, W])``."""

        batch, time, _, height, width = raster.shape
        flat = raster.reshape(batch * time, -1, height, width)
        spatial = self.body(flat)
        pooled = spatial.mean(dim=(2, 3)).reshape(batch, time, -1)
        return pooled, spatial.reshape(
            batch, time, self.config.spatial_channels, height, width
        )


class EntityContext(nn.Module):
    """Compact Transformer over public entity tokens (V3 representation)."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.entity_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.null_entity = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        nn.init.normal_(self.null_entity, mean=0.0, std=0.02)

    def forward(
        self, entities: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return pooled entity context ``[B, T, model_dim]``."""

        batch, time, count, _ = entities.shape
        flat_batch = batch * time
        features = self.projection(entities.reshape(flat_batch, count, -1))
        mask_flat = entity_mask.reshape(flat_batch, count)
        null = self.null_entity.expand(flat_batch, -1, -1)
        stacked = torch.cat((features, null), dim=1)
        padding = torch.cat(
            (
                ~mask_flat,
                torch.zeros((flat_batch, 1), dtype=torch.bool, device=entities.device),
            ),
            dim=1,
        )
        transformed = self.transformer(stacked, src_key_padding_mask=padding)
        entity_part = transformed[:, :count]
        null_part = transformed[:, count]
        present = mask_flat.unsqueeze(-1)
        pooled = (entity_part * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
        pooled = torch.where(present.sum(dim=1) > 0, pooled, null_part)
        return pooled.reshape(batch, time, -1)


class V4Encoder(nn.Module):
    """Fuse spatial, entity, hand, global, and belief inputs into ``zt``."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        self.spatial = SpatialEncoder(config)
        self.entities = EntityContext(config)
        self.hand_projection = nn.Sequential(
            nn.Linear(config.hand_token_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(config.global_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        belief_in = 2 * config.belief_cards + 2 + config.event_history_len * config.event_dim
        self.belief_projection = nn.Sequential(
            nn.Linear(belief_in, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                config.spatial_channels + (3 + config.card_slots) * config.model_dim,
                config.fused_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.fused_dim),
        )

    def forward(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        hand_tokens: torch.Tensor,
        opp_hand_probs: torch.Tensor,
        opp_out_of_cycle: torch.Tensor,
        opp_elixir_interval: torch.Tensor,
        event_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(zt, spatial_map, entity_context, hand_queries)``."""

        config = self.config
        batch, time = raster.shape[:2]
        raster_pooled, spatial_map = self.spatial(raster)
        entity_context = self.entities(entities, entity_mask)
        hand_queries = self.hand_projection(hand_tokens)
        global_encoded = self.global_projection(
            global_features.reshape(batch * time, -1)
        ).reshape(batch, time, -1)
        belief_flat = torch.cat(
            (
                opp_hand_probs.reshape(batch, time, -1),
                opp_out_of_cycle.to(global_features.dtype).reshape(batch, time, -1),
                opp_elixir_interval.reshape(batch, time, -1),
                event_history.reshape(batch, time, -1),
            ),
            dim=-1,
        )
        belief_encoded = self.belief_projection(belief_flat.reshape(batch * time, -1)).reshape(
            batch, time, -1
        )
        hand_flat = hand_queries.reshape(batch, time, -1)
        expected_hand = config.card_slots * config.model_dim
        if hand_flat.shape[-1] != expected_hand:
            raise ValueError("hand token layout does not match ModelConfigV4")
        fused = self.fusion(
            torch.cat(
                (
                    raster_pooled.reshape(batch, time, -1),
                    entity_context,
                    global_encoded,
                    hand_flat,
                    belief_encoded,
                ),
                dim=-1,
            )
        )
        return fused, spatial_map, entity_context, hand_queries


class GRUCore(nn.Module):
    """Single-layer GRU with per-step reset semantics (matches V3)."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.gru = nn.GRU(
            config.fused_dim, config.gru_hidden_dim, config.gru_layers, batch_first=True
        )

    def forward(self, fused: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
        batch, time, _ = fused.shape
        hidden = torch.zeros(
            self.gru.num_layers, batch, self.gru.hidden_size, device=fused.device, dtype=fused.dtype
        )
        outputs: list[torch.Tensor] = []
        for step in range(time):
            reset = reset_mask[:, step]
            if bool(reset.any().item()):
                cleared = hidden.clone()
                cleared[:, reset] = 0.0
                hidden = cleared
            out, hidden = self.gru(fused[:, step : step + 1], hidden)
            outputs.append(out[:, 0])
        return torch.stack(outputs, dim=1)


class V4ActionHead(nn.Module):
    """Hierarchical heads; every head sees ``[zt, ht]`` directly."""

    WAIT = 0
    PLAY = 1

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        joint = config.fused_dim + config.gru_hidden_dim
        self.mode_head = nn.Linear(joint, 2)
        self.duration_head = nn.Linear(joint, len(WAIT_DURATIONS))
        # Card-query head: each hand token queries battlefield context.
        self.card_query = nn.Linear(config.model_dim + joint, config.model_dim)
        self.card_key = nn.Linear(config.model_dim, config.model_dim)
        self.card_score = nn.Linear(config.model_dim, 1)
        # Placement decoder: projected dot product over the spatial map.
        self.spatial_key = nn.Conv2d(
            config.spatial_channels, config.spatial_head_dim, kernel_size=1
        )
        self.placement_query = nn.Linear(config.model_dim + joint, config.spatial_head_dim)
        self.cell_bias: nn.Parameter | None = None

    def _joint(self, zt: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        if zt.shape != ht.shape or zt.shape[-1] != self.config.fused_dim:
            raise ValueError("zt and ht must share [B, T, fused_dim]")
        if self.config.fused_dim != self.config.gru_hidden_dim:
            raise ValueError("V4 heads require fused_dim == gru_hidden_dim")
        return torch.cat((zt, ht), dim=-1)

    def _cell_bias(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        if self.cell_bias is None or tuple(self.cell_bias.shape) != (rows, cols):
            bias = torch.zeros(rows, cols, device=device, dtype=dtype)
            self.cell_bias = nn.Parameter(bias)
        return self.cell_bias

    def forward(
        self,
        zt: torch.Tensor,
        ht: torch.Tensor,
        entity_context: torch.Tensor,
        hand_queries: torch.Tensor,
        spatial_map: torch.Tensor,
    ) -> V4Logits:
        batch, time, _ = zt.shape
        rows, cols = spatial_map.shape[-2:]
        joint = self._joint(zt, ht)
        mode_logits = self.mode_head(joint)
        duration_logits = self.duration_head(joint)
        # Card scores: token-conditioned query dotted with battlefield key.
        flat_joint = joint.reshape(batch * time, -1)
        flat_hand = hand_queries.reshape(batch * time, self.config.card_slots, -1)
        joint_broadcast = flat_joint.unsqueeze(1).expand(-1, self.config.card_slots, -1)
        queries = torch.tanh(self.card_query(torch.cat((flat_hand, joint_broadcast), dim=-1)))
        keys = self.card_key(entity_context.reshape(batch * time, -1)).unsqueeze(1)
        card_logits = (
            self.card_score(queries * keys).squeeze(-1)
            / math.sqrt(self.config.model_dim)
        ).reshape(batch, time, -1)
        # Placement: every card gets a full-resolution map (loss needs all).
        spatial_keys = self.spatial_key(
            spatial_map.reshape(batch * time, spatial_map.shape[2], rows, cols)
        )
        spatial_keys = spatial_keys.reshape(
            batch, time, self.config.spatial_head_dim, rows, cols
        )
        placement_queries = self.placement_query(
            torch.cat((flat_hand, joint_broadcast), dim=-1)
        ).reshape(batch, time, self.config.card_slots, -1)
        bias = self._cell_bias(rows, cols, spatial_map.device, spatial_map.dtype)
        maps = torch.einsum("btkd,btdhw->btkhw", placement_queries, spatial_keys) + bias
        return V4Logits(
            mode=mode_logits,
            wait_duration=duration_logits,
            card=card_logits,
            placement=maps,
        )


class RecurrentV4Policy(nn.Module):
    """Encoder -> GRU -> hierarchical V4 action heads."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        self.encoder = V4Encoder(config)
        self.recurrent = GRUCore(config)
        self.heads = V4ActionHead(config)

    def forward(
        self,
        raster: torch.Tensor,
        global_features: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        hand_tokens: torch.Tensor,
        opp_hand_probs: torch.Tensor,
        opp_out_of_cycle: torch.Tensor,
        opp_elixir_interval: torch.Tensor,
        event_history: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> tuple[V4Logits, torch.Tensor, torch.Tensor]:
        """Return ``(logits, zt, ht)`` for a ``[B, T]`` sequence."""

        zt, spatial_map, entity_context, hand_queries = self.encoder(
            raster,
            global_features,
            entities,
            entity_mask,
            hand_tokens,
            opp_hand_probs,
            opp_out_of_cycle,
            opp_elixir_interval,
            event_history,
        )
        ht = self.recurrent(zt, reset_mask)
        logits = self.heads(zt, ht, entity_context, hand_queries, spatial_map)
        return logits, zt, ht

    def log_prob(
        self,
        logits: V4Logits,
        masks: ActionMasks,
        actions: V4ActionBatch,
    ) -> torch.Tensor:
        """Joint log probability with legality rejection (fail-closed)."""

        mode_logp = _masked_log_softmax(logits.mode, masks.mode)
        mode_selected = actions.mode.clamp(0, 1).unsqueeze(-1)
        joint = mode_logp.gather(-1, mode_selected).squeeze(-1)
        is_play = actions.mode == V4ActionHead.PLAY
        is_wait = actions.mode == V4ActionHead.WAIT
        if bool(is_wait.any().item()):
            duration_logp = F.log_softmax(logits.wait_duration, dim=-1)
            duration_selected = actions.wait_duration.clamp(0, len(WAIT_DURATIONS) - 1).unsqueeze(-1)
            joint = joint + (duration_logp.gather(-1, duration_selected).squeeze(-1) * is_wait)
        if bool(is_play.any().item()):
            card_logp = _masked_log_softmax(logits.card, masks.card)
            card_selected = actions.card_slot.clamp(0, masks.card.shape[-1] - 1).unsqueeze(-1)
            joint = joint + (card_logp.gather(-1, card_selected).squeeze(-1) * is_play)
            rows, cols = logits.placement.shape[-2:]
            if tuple(masks.placement.shape[-2:]) != (rows, cols):
                raise ValueError("placement mask grid must match the spatial map resolution")
            cells = rows * cols
            prefix = tuple(logits.placement.shape[:-3])
            flat_logits = logits.placement.reshape(prefix + (masks.card.shape[-1], cells))
            flat_mask = masks.placement.reshape(prefix + (masks.card.shape[-1], cells))
            placement_logp = _masked_log_softmax(flat_logits, flat_mask)
            slot = actions.card_slot.clamp(0, masks.card.shape[-1] - 1)
            slot_index = slot.reshape(slot.shape + (1, 1)).expand(
                slot.shape + (1, cells)
            )
            selected = placement_logp.gather(-2, slot_index).squeeze(-2)
            cell_index = (
                actions.placement[..., 0].clamp(0, rows - 1) * cols
                + actions.placement[..., 1].clamp(0, cols - 1)
            ).clamp(0, cells - 1).unsqueeze(-1)
            joint = joint + (selected.gather(-1, cell_index).squeeze(-1) * is_play)
        return joint

    @torch.no_grad()
    def act_deterministic(
        self,
        logits: V4Logits,
        masks: ActionMasks,
    ) -> V4ActionBatch:
        """Argmax decoding honoring legality (evaluation behavior)."""

        mode = torch.where(
            masks.mode, logits.mode, torch.full_like(logits.mode, float("-inf"))
        ).argmax(dim=-1)
        duration = logits.wait_duration.argmax(dim=-1)
        card = torch.where(
            masks.card, logits.card, torch.full_like(logits.card, float("-inf"))
        ).argmax(dim=-1)
        batch, time = mode.shape
        rows, cols = logits.placement.shape[-2:]
        if tuple(masks.placement.shape[-2:]) != (rows, cols):
            raise ValueError(
                "placement mask grid must match the spatial map resolution "
                f"(logits {(rows, cols)} vs mask {tuple(masks.placement.shape[-2:])})"
            )
        flat = logits.placement.reshape(batch, time, -1, rows * cols)
        flat_mask = masks.placement.reshape(batch, time, -1, rows * cols)
        safe = torch.where(
            flat_mask, flat, torch.full_like(flat, float("-inf"))
        ).argmax(dim=-1)
        is_play = mode == V4ActionHead.PLAY
        chosen = safe.gather(-1, card.clamp(0, safe.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
        rows_idx = (chosen // cols).to(torch.long)
        cols_idx = (chosen % cols).to(torch.long)
        placement = torch.zeros(batch, time, 2, dtype=torch.long, device=mode.device)
        placement[..., 0] = torch.where(is_play, rows_idx, placement[..., 0])
        placement[..., 1] = torch.where(is_play, cols_idx, placement[..., 1])
        return V4ActionBatch(
            mode=mode.to(torch.long),
            card_slot=card.to(torch.long),
            placement=placement,
            wait_duration=duration.to(torch.long),
        )


def count_parameters(policy: RecurrentV4Policy) -> int:
    return sum(int(p.numel()) for p in policy.parameters() if p.requires_grad)


def to_action_batch(v4_actions: V4ActionBatch) -> ActionBatch:
    """Project V4 actions onto the shared V3 trajectory container."""

    return ActionBatch(
        mode=v4_actions.mode,
        card_slot=v4_actions.card_slot,
        placement=v4_actions.placement,
    )


__all__ = [
    "WAIT_DURATIONS",
    "GRUCore",
    "ModelConfigV4",
    "RecurrentV4Policy",
    "SpatialEncoder",
    "EntityContext",
    "V4ActionBatch",
    "V4ActionHead",
    "V4Encoder",
    "V4Logits",
    "count_parameters",
    "to_action_batch",
]
