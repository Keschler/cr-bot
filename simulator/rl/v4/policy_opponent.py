"""Frozen V4 policy acting as player one through the public observation API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from functools import lru_cache
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulator.actions import PlayCardAction, WaitAction
from simulator.public_state_estimator import PublicStateEstimator
from simulator.roster import PLAYER_DECK

from .frontier import frontier_model_config
from .model_v4 import ModelConfigV4, RecurrentV4Policy, V4ActionBatch, masks_from_legal_play
from .v4_ppo.runtime import (
    _v4_step_inputs,
    batch_v4_inputs,
    configure_v4_runtime,
    load_frozen_anchor,
)


def exact_model_config(value: ModelConfigV4 | Mapping[str, Any]) -> ModelConfigV4:
    """Validate an explicit checkpoint configuration and its public input contract.

    Partial dictionaries are deliberately rejected: architecture defaults can
    change, whereas a population manifest must describe an exact checkpoint.
    This still supports different network widths/depths, not different decks or
    observation schemas.
    """

    if isinstance(value, ModelConfigV4):
        config = value
    elif isinstance(value, Mapping):
        names = {item.name for item in fields(ModelConfigV4)}
        if set(value) != names:
            raise ValueError("model_config must contain every ModelConfigV4 field exactly")
        config = ModelConfigV4(**dict(value))
    else:
        raise TypeError("model_config must be ModelConfigV4 or its complete mapping")
    contract = ModelConfigV4()
    for name in (
        "raster_channels", "raster_height", "raster_width", "global_dim",
        "entity_dim", "max_entities", "hand_token_dim", "belief_cards",
        "event_history_len", "event_dim", "card_slots", "placement_rows",
        "placement_cols",
    ):
        if getattr(config, name) != getattr(contract, name):
            raise ValueError(f"model_config {name} differs from the V4 public input contract")
    if config.fused_dim != config.gru_hidden_dim:
        raise ValueError("V4 actor requires equal fused_dim and gru_hidden_dim")
    return config


def supported_opponent_deck(deck: Sequence[str]) -> tuple[str, ...]:
    """Fail before inference if a Hog-only checkpoint is assigned another deck."""

    result = tuple(deck)
    if result != tuple(PLAYER_DECK):
        raise ValueError("V4 learned opponents currently support the canonical Hog 2.6 deck only")
    return result


@lru_cache(maxsize=8)
def _frozen_actor(path: str, sha256: str) -> Any:
    # Model construction initializes parameters before loading the checkpoint.
    # Keep that one-time initialization inside an isolated RNG scope so a cold
    # cache does not consume the caller's Torch stream while a warm cache does
    # not.  The loaded weights, rather than the initialization values, define
    # the actor used below.
    with torch.random.fork_rng(devices=[]):
        actor = load_frozen_anchor(
            path, frontier_model_config(), expected_sha256=sha256, device="cpu"
        )
    configure_v4_runtime(actor)
    return actor


@lru_cache(maxsize=8)
def _configured_actor(path: str, sha256: str, config: ModelConfigV4) -> Any:
    """Load opt-in explicit metadata, including versioned checkpoint envelopes.

    Historical raw state dictionaries remain supported. Envelopes use exactly
    ``model_config`` and ``state_dict``; their metadata must agree with the
    pinned population member. Loading the bytes already hashed avoids a
    second-read race between the digest check and deserialization.
    """

    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("policy opponent checkpoint hash mismatch")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("policy opponent checkpoint must be a state dictionary or metadata envelope")
    if "model_config" in state or "state_dict" in state:
        if set(state) != {"model_config", "state_dict"}:
            raise ValueError("checkpoint envelope needs exactly model_config and state_dict")
        if exact_model_config(state["model_config"]) != config:
            raise ValueError("checkpoint model_config does not match pinned population metadata")
        state = state["state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint state_dict must be a mapping")
    with torch.random.fork_rng(devices=[]):
        actor = RecurrentV4Policy(config)
        actor.eval()
        # Materialize the lazy placement bias using the fixed public schema.
        with torch.no_grad():
            actor(
                torch.zeros(1, 1, 21, 32, 18), torch.zeros(1, 1, 768),
                torch.zeros(1, 1, 128, 32), torch.zeros(1, 1, 128, dtype=torch.bool),
                torch.zeros(1, 1, 4, 16), torch.full((1, 1, 128), 1.0 / 128),
                torch.zeros(1, 1, 128, dtype=torch.bool), torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 16, 8), torch.ones(1, 1, dtype=torch.bool),
            )
        result = actor.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if key != "heads.cell_bias"]
        if result.unexpected_keys or missing:
            raise ValueError(
                f"policy opponent checkpoint incompatible: unexpected={result.unexpected_keys} missing={missing}"
            )
    configure_v4_runtime(actor)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.anchor_sha256 = sha256
    return actor


def _sample_action(logits: Any, masks: Any, generator: torch.Generator) -> V4ActionBatch:
    """Sample the native fixed-WAIT factorization with an episode-local RNG."""

    def sample(values: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        if not bool(legal.any(dim=-1).all()):
            raise ValueError("cannot sample an action factor without a legal choice")
        masked = torch.where(legal, values, torch.full_like(values, float("-inf")))
        probabilities = torch.softmax(masked, dim=-1)
        shape = probabilities.shape[:-1]
        return torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), 1,
            replacement=True, generator=generator,
        ).reshape(shape)

    mode = sample(logits.mode, masks.mode)
    play = mode == 1
    card = torch.zeros_like(mode)
    placement = torch.zeros(mode.shape + (2,), dtype=torch.long)
    if bool(play.any()):
        card[play] = sample(logits.card[play], masks.card[play])
        rows, cols = logits.placement.shape[-2:]
        slot_index = card[play].reshape(-1, 1, 1).expand(-1, 1, rows * cols)
        values = logits.placement.reshape(*mode.shape, -1, rows * cols)[play]
        legal = masks.placement.reshape(*mode.shape, -1, rows * cols)[play]
        cell = sample(
            values.gather(1, slot_index).squeeze(1),
            legal.gather(1, slot_index).squeeze(1),
        )
        placement[play] = torch.stack((cell // cols, cell % cols), dim=-1)
    return V4ActionBatch(mode, card, placement, torch.zeros_like(mode))


class V4PolicyOpponent:
    """One episode's recurrent state for a frozen, pinned actor checkpoint."""

    def __init__(
        self, env: Any, checkpoint: str, sha256: str, *,
        model_config: ModelConfigV4 | Mapping[str, Any] | None = None,
        inference: str = "deterministic", seed: int = 0,
        deck: Sequence[str] | None = None,
    ) -> None:
        if not checkpoint or not sha256:
            raise ValueError("policy opponent needs a checkpoint path and sha256")
        if inference not in ("deterministic", "stochastic"):
            raise ValueError("policy opponent inference must be deterministic or stochastic")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("policy opponent seed must be an unsigned 64-bit integer")
        self.deck = supported_opponent_deck(PLAYER_DECK if deck is None else deck)
        if env.state is None or tuple(env.state.players[1].deck) != self.deck:
            raise ValueError("policy opponent environment deck does not match its pinned Hog deck")
        self.env = env
        path = str(Path(checkpoint).expanduser().resolve(strict=True))
        self.model_config = None if model_config is None else exact_model_config(model_config)
        self.actor = (
            _frozen_actor(path, sha256) if self.model_config is None
            else _configured_actor(path, sha256, self.model_config)
        )
        self.inference = inference
        self.seed = seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.hidden = self.actor.initial_hidden(1, device="cpu")
        self.estimator = PublicStateEstimator()

    def choose_action(self, engine: Any, state: Any, player: int) -> Any:
        if player != 1 or engine is not self.env.engine or state is not self.env.state:
            raise ValueError("V4 policy opponent must act as player one in its bound environment")
        observation = self.env.observe_v2_for_viewer(1)
        own = state.players[1]
        self.estimator.sync_from_env(self.env, viewer=1)
        packed = _v4_step_inputs(
            observation, list(own.hand[:4]), own.elixir_milli / 1000.0,
            self.estimator.snapshot(),
        )
        inputs = batch_v4_inputs(packed, "cpu")
        legal = torch.as_tensor(np.array(observation.legal_play, copy=True)).unsqueeze(0).unsqueeze(0)
        masks = masks_from_legal_play(legal, bool(observation.legal_wait))
        with torch.no_grad():
            logits, _, self.hidden = self.actor.forward_step(
                inputs, hidden=self.hidden,
                reset_mask=torch.zeros((1, 1), dtype=torch.bool),
            )
            action = (
                self.actor.act_deterministic(logits, masks)
                if self.inference == "deterministic"
                else _sample_action(logits, masks, self.generator)
            )
        if int(action.mode.reshape(-1)[0]) == 0:
            return WaitAction(1)
        row, col = (int(value) for value in action.placement.reshape(-1, 2)[0])
        # The player-one observation and legal mask are rotated 180 degrees.
        return PlayCardAction(
            1, int(action.card_slot.reshape(-1)[0]), (17 - col, 31 - row)
        )
