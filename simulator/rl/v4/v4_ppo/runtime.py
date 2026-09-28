"""Frozen checkpoints, fixed-step action decoding, and policy inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence
import numpy as np
import torch
from ..model_v4 import ModelConfigV4, RecurrentV4Policy, V4ActionBatch
from ..hand_features import build_hand_tokens
from ...foundation.trajectory import RecurrentSequence


def load_frozen_anchor(
    path: str,
    config: ModelConfigV4 | None = None,
    *,
    expected_sha256: str | None = None,
    device: Any = None,
) -> RecurrentV4Policy:
    """Load simx12 (or any V4 checkpoint) as a frozen BC anchor.

    The anchor is set to eval mode with every parameter frozen; an optional
    sha256 over the checkpoint file pins exactly which weights train
    against.  Run provenance should record the returned digest.
    """

    import hashlib

    with open(path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"anchor checkpoint hash mismatch: expected {expected_sha256}, got {digest}"
        )
    policy = RecurrentV4Policy(config or ModelConfigV4())
    policy.eval()
    with torch.no_grad():
        policy(
            torch.zeros(1, 1, 21, 32, 18),
            torch.zeros(1, 1, 768),
            torch.zeros(1, 1, 128, 32),
            torch.zeros(1, 1, 128, dtype=torch.bool),
            torch.zeros(1, 1, 4, 16),
            torch.full((1, 1, 128), 1.0 / 128),
            torch.zeros(1, 1, 128, dtype=torch.bool),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 16, 8),
            torch.ones(1, 1, dtype=torch.bool),
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    # The placement cell-bias materializes lazily on first forward, so a
    # checkpoint saved before any forward legitimately lacks it.  Tolerate
    # exactly that key and nothing else (fail-closed on any other drift).
    result = policy.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys]
    missing = [k for k in result.missing_keys if k != "heads.cell_bias"]
    if unexpected or missing:
        raise RuntimeError(
            f"anchor checkpoint incompatible: unexpected={unexpected} missing={missing}"
        )
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    if device is not None:
        policy = policy.to(device)
    policy.anchor_sha256 = digest  # type: ignore[attr-defined]
    return policy


def configure_v4_runtime(policy: RecurrentV4Policy) -> None:
    """Use the action distribution the fixed-decision simulator executes.

    Supervised duration weights remain checkpoint-compatible, but duration
    is not an action factor, an entropy bonus, or a BC penalty in this runtime.
    """
    policy.config = replace(policy.config, fixed_wait=True)


def decode_v4_sim_actions(
    actions: V4ActionBatch, *, player: int = 0
) -> list[Any]:
    """Decode one batch element of V4 actions to simulator actions."""

    try:
        from ....actions import PlayCardAction, WaitAction
    except ImportError:  # pragma: no cover - top-level ``rl`` imports
        from simulator.actions import PlayCardAction, WaitAction
    if type(player) is not int or player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    decoded: list[Any] = []
    mode = actions.mode.reshape(-1)
    slot = actions.card_slot.reshape(-1)
    place = actions.placement.reshape(-1, 2)
    for index in range(int(mode.numel())):
        if int(mode[index]) == 0:
            if int(actions.wait_duration.reshape(-1)[index]) != 0:
                raise ValueError("fixed-step runtime requires WAIT duration index 0")
            decoded.append(WaitAction(player))
        else:
            decoded.append(
                PlayCardAction(
                    player,
                    int(slot[index]),
                    (int(place[index][1]), int(place[index][0])),
                )
            )
    return decoded


def _hand_tokens_partial(hand: Sequence[str], elixir: float) -> np.ndarray:
    """Encode a live hand that may transiently hold fewer than four cards.

    Fresh-reset distillation always sees full hands, but mid-rollout the
    refill cooldown leaves three-card hands.  Present cards keep their true
    slot encoding; missing slots stay zero (their mask rows are empty, so
    the sampler can never select them).
    """

    tokens = np.zeros((4, 16), dtype=np.float32)
    for slot, card in enumerate(list(hand)[:4]):
        row = np.array(build_hand_tokens([card] * 4, float(elixir))[0], copy=True)
        row[2] = slot / 3.0
        tokens[slot] = row
    return tokens


def _v4_step_inputs(
    observation: Any,
    hand: Sequence[str],
    elixir: float,
    snapshot: Any,
) -> dict[str, np.ndarray]:
    """Map one public observation + estimator snapshot to V4 inputs."""

    return {
        "raster": np.array(observation.board, dtype=np.float32),
        "global_features": np.array(observation.global_vector, dtype=np.float32),
        "entities": np.array(observation.entity_tokens, dtype=np.float32),
        "entity_mask": np.array(observation.entity_mask, dtype=bool),
        "hand_tokens": _hand_tokens_partial(hand, float(elixir)),
        "opp_hand_probs": np.array(snapshot.opp_hand_probs, dtype=np.float32),
        "opp_out_of_cycle": np.array(snapshot.opp_out_of_cycle, dtype=bool),
        "opp_elixir_interval": np.asarray(
            [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], dtype=np.float32
        ),
        "event_history": np.array(snapshot.event_history, dtype=np.float32),
        "legal_play": np.array(observation.legal_play, dtype=bool),
        "legal_wait": bool(observation.legal_wait),
    }


def observe_v4_inputs(env: Any, estimator: Any) -> tuple[Any, dict[str, np.ndarray], list[str], float]:
    """Observe player zero and update public memory for one V4 decision."""

    observation = env.observe_v2_for_viewer(0)
    player = env.state.players[0]
    hand = list(player.hand[:4])
    elixir = float(player.elixir_milli) / 1000.0
    estimator.sync_from_env(env, viewer=0)
    return observation, _v4_step_inputs(observation, hand, elixir, estimator.snapshot()), hand, elixir


def batch_v4_inputs(inputs: dict[str, Any], device: Any) -> dict[str, torch.Tensor]:
    """Add one batch and time dimension to public V4 model features."""

    boolean_keys = {"entity_mask", "opp_out_of_cycle"}
    return {
        key: torch.as_tensor(
            np.asarray(value, dtype=bool if key in boolean_keys else np.float32),
            device=device,
        ).unsqueeze(0).unsqueeze(0)
        for key, value in inputs.items()
        if key not in ("legal_play", "legal_wait")
    }


def _forward_sequence(
    policy: RecurrentV4Policy, sequence: RecurrentSequence, device: Any
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Re-evaluate stored V4 inputs (requires the retained V4 fields)."""

    for name in (
        "hand_tokens", "opp_hand_probs", "opp_out_of_cycle",
        "opp_elixir_interval", "event_history",
    ):
        if getattr(sequence, name) is None:
            raise ValueError(f"sequence lacks retained V4 field {name!r}")
    zt, spatial_map, entity_context, hand_queries = policy.encoder(
        sequence.raster.to(device),
        sequence.global_features.to(device),
        sequence.entities.to(device),
        sequence.entity_mask.to(device),
        sequence.hand_tokens.to(device),
        sequence.opp_hand_probs.to(device),
        sequence.opp_out_of_cycle.to(device),
        sequence.opp_elixir_interval.to(device),
        sequence.event_history.to(device),
    )
    hidden = sequence.initial_hidden
    if hidden is not None:
        hidden = hidden.to(device)
    ht, next_hidden = policy.recurrent.forward_with_hidden(
        zt, sequence.reset_mask.to(device), hidden=hidden
    )
    logits = policy.heads(zt, ht, entity_context, hand_queries, spatial_map)
    return logits, ht, next_hidden
