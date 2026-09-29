"""Frozen V4 policy acting as player one through the public observation API."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch

from simulator.actions import PlayCardAction, WaitAction
from simulator.public_state_estimator import PublicStateEstimator

from .frontier import frontier_model_config
from .model_v4 import masks_from_legal_play
from .v4_ppo.runtime import (
    _v4_step_inputs,
    batch_v4_inputs,
    configure_v4_runtime,
    load_frozen_anchor,
)


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


class V4PolicyOpponent:
    """One episode's recurrent state for a frozen, pinned actor checkpoint."""

    def __init__(self, env: Any, checkpoint: str, sha256: str) -> None:
        if not checkpoint or not sha256:
            raise ValueError("policy opponent needs a checkpoint path and sha256")
        self.env = env
        self.actor = _frozen_actor(checkpoint, sha256)
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
            action = self.actor.act_deterministic(logits, masks)
        if int(action.mode.reshape(-1)[0]) == 0:
            return WaitAction(1)
        row, col = (int(value) for value in action.placement.reshape(-1, 2)[0])
        # The player-one observation and legal mask are rotated 180 degrees.
        return PlayCardAction(
            1, int(action.card_slot.reshape(-1)[0]), (17 - col, 31 - row)
        )
