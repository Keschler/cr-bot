"""Deployment adapter for the retained V4 recurrent policy.

The simulator and the visual frontend meet at the public V2 observation
boundary.  This module adds the few live values V4 needs (hand tokens and
public memory), loads a plain V4 state-dict checkpoint, and decodes one
fixed-step action.  No learner, optimizer, privileged state, or legacy
checkpoint reader is imported here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class V4ActorError(RuntimeError):
    """Raised when a visual observation cannot be used by the V4 actor."""


@dataclass(slots=True)
class _StepOutput:
    logits: Any
    masks: Any
    hidden: Any
    hand_cards: tuple[str | None, ...]


def _canonical_card(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from ...observation import policy_card_name
        from ..sensing.policy_bridge import _visual_feature_card_name

        canonical = policy_card_name(value) or _visual_feature_card_name(value)
    except (ImportError, TypeError, ValueError):
        canonical = None
    return canonical or value.strip().casefold().replace("_", "-")


def _card_cost(card: str) -> float | None:
    try:
        from cr_bot.domain.card_metadata import CARD_METADATA

        raw = CARD_METADATA.get(card, {}).get("elixir_cost")
        cost = float(raw)
    except (ImportError, TypeError, ValueError):
        return None
    return cost if np.isfinite(cost) and 0.0 <= cost <= 10.0 else None


def _as_batch(value: Any, *, device: Any, boolean: bool = False) -> Any:
    import torch

    array = np.array(value, dtype=bool if boolean else np.float32, copy=True)
    return torch.as_tensor(array, device=device).unsqueeze(0).unsqueeze(0)


def _decode_action(actions: Any) -> Any:
    """Decode the singleton V4 action using the public cell convention."""

    try:
        from cr_bot.domain.game_state import Action
    except ImportError as error:  # pragma: no cover - deployment environment
        raise V4ActorError("cr_bot is required to decode V4 actions") from error
    mode = getattr(actions, "mode", None)
    slot = getattr(actions, "card_slot", None)
    placement = getattr(actions, "placement", None)
    if mode is None or slot is None or placement is None:
        raise V4ActorError("V4 action batch is missing mode, card slot, or placement")
    if int(mode.numel()) != 1 or int(slot.numel()) != 1:
        raise V4ActorError("V4 actor must decode exactly one action")
    mode_value = int(mode.reshape(-1)[0].item())
    if mode_value == 0:
        return Action(kind="Wait")
    if mode_value != 1:
        raise V4ActorError(f"V4 actor returned unsupported mode {mode_value}")
    card_slot = int(slot.reshape(-1)[0].item())
    cell = placement.reshape(-1, 2)
    if cell.shape[0] != 1:
        raise V4ActorError("V4 actor placement batch must contain one action")
    row = int(cell[0, 0].item())
    col = int(cell[0, 1].item())
    if not 0 <= card_slot < 4 or not 0 <= row < 32 or not 0 <= col < 18:
        raise V4ActorError(
            f"V4 actor returned an out-of-range play: slot={card_slot}, cell={(col, row)}"
        )
    return Action(kind="Play", card_idx=card_slot, cell=(col, row))


class V4Actor:
    """Frozen V4 actor for live and recorded visual inference."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        try:
            import torch
            from ...rl.v4.frontier import frontier_model_config
            from ...rl.v4.v4_ppo import configure_v4_runtime, load_frozen_anchor
        except ImportError as error:  # pragma: no cover - environment-specific
            raise V4ActorError("V4 inference requires the simulator V4 runtime and PyTorch") from error
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise V4ActorError(f"V4 checkpoint does not exist: {path}")
        try:
            policy = load_frozen_anchor(
                str(path),
                config=frontier_model_config(),
                device=device,
            )
            configure_v4_runtime(policy)
        except Exception as error:
            raise V4ActorError(f"could not load V4 checkpoint {path}: {error}") from error
        self.policy = policy
        self.config = policy.config
        self.metadata = {
            "checkpoint_format": "v4-state-dict",
            "checkpoint": str(path),
            "sha256": getattr(policy, "anchor_sha256", None),
        }
        self._torch = torch
        self._device = torch.device(device)
        self._hidden: Any | None = None
        self._memory: Any = None
        self._seen_enemy_units: set[tuple[Any, str]] = set()
        self._last_tick = 0.0
        self._pending_own_play: tuple[str, float] | None = None
        self._last_step: _StepOutput | None = None

    @property
    def device(self) -> Any:
        return self._device

    def reset(self) -> None:
        """Reset recurrent and public memory at a match boundary."""

        self._hidden = None
        self._memory = None
        self._seen_enemy_units.clear()
        self._last_tick = 0.0
        self._pending_own_play = None
        self._last_step = None

    def snapshot_state(self) -> dict[str, Any]:
        """Capture recurrent and public memory for a side-effect-free replay."""

        return {
            "hidden": self._hidden,
            "memory": copy.deepcopy(self._memory),
            "seen_enemy_units": set(self._seen_enemy_units),
            "last_tick": self._last_tick,
            "pending_own_play": self._pending_own_play,
            "last_step": self._last_step,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`snapshot_state`."""

        if not isinstance(state, dict):
            raise V4ActorError("actor state snapshot must be a dictionary")
        self._hidden = state.get("hidden")
        self._memory = state.get("memory")
        self._seen_enemy_units = set(state.get("seen_enemy_units", ()))
        self._last_tick = float(state.get("last_tick", 0.0))
        self._pending_own_play = state.get("pending_own_play")
        self._last_step = state.get("last_step")

    def _ensure_memory(self) -> Any:
        if self._memory is None:
            try:
                from ...public_state_estimator import PublicStateEstimator
            except ImportError as error:  # pragma: no cover
                raise V4ActorError("public state estimator is unavailable") from error
            self._memory = PublicStateEstimator()
        return self._memory

    def _sync_public_memory(self, observation: Any) -> Any:
        """Record visual events without reading simulator-private state."""

        game_state = getattr(observation, "game_state", None)
        tick = float(getattr(observation, "match_time_s", 0.0))
        if not np.isfinite(tick) or tick < self._last_tick:
            raise V4ActorError("visual match clock moved backwards")
        memory = self._ensure_memory()
        if self._pending_own_play is not None:
            from ...public_state_estimator import PublicEvent

            _card, cost = self._pending_own_play
            memory.observe(PublicEvent("own_deploy", tick, cost))
            memory.observe(PublicEvent("action_taken", tick, cost))
            self._pending_own_play = None

        enemy_units = getattr(game_state, "enemy_units", ())
        if isinstance(enemy_units, (list, tuple)):
            try:
                from cr_bot.domain.card_metadata import CARD_METADATA
            except ImportError as error:  # pragma: no cover
                raise V4ActorError("card metadata is unavailable") from error
            for match in enemy_units:
                troop = getattr(match, "troop", None)
                card = _canonical_card(getattr(troop, "class_name", None))
                cost = _card_cost(card) if card is not None else None
                if card is None or cost is None or card not in CARD_METADATA:
                    continue
                raw_track = getattr(troop, "track_id", None)
                # A match object is rebuilt on every visual frame.  Its object
                # id therefore cannot identify one public unit across frames;
                # without a stable detector track, skip the spend rather than
                # repeatedly charging the opponent's elixir interval.
                if type(raw_track) not in (int, str) or (
                    isinstance(raw_track, str) and not raw_track.strip()
                ):
                    continue
                key = (raw_track, card)
                if key in self._seen_enemy_units:
                    continue
                self._seen_enemy_units.add(key)
                memory.observe_opponent_spend(
                    tick_seconds=tick,
                    card_key=card,
                    elixir_cost=cost,
                )
        memory.advance_to(tick)
        self._last_tick = tick
        return memory.snapshot()

    def _forward(self, observation: Any) -> _StepOutput:
        hand_raw = getattr(observation, "hand_cards", None)
        if not isinstance(hand_raw, (tuple, list)) or len(hand_raw) != 4:
            raise V4ActorError(
                "V4 inference requires four visible hand cards at the public boundary"
            )
        hand = tuple(_canonical_card(card) for card in hand_raw)
        if any(card is None for card in hand):
            raise V4ActorError("V4 inference requires four identified hand cards")
        own_elixir = getattr(observation, "own_elixir", None)
        if own_elixir is None or not np.isfinite(float(own_elixir)):
            raise V4ActorError("V4 inference requires a finite own-elixir reading")
        base = getattr(observation, "base", observation)
        for field in ("board", "global_vector", "entity_tokens", "entity_mask", "legal_play"):
            if not hasattr(base, field):
                raise V4ActorError(f"V4 observation is missing {field!r}")
        snapshot = self._sync_public_memory(observation)
        try:
            from ...rl.v4.hand_features import build_hand_tokens
            from ...rl.v4.model_v4 import masks_from_legal_play
        except ImportError as error:  # pragma: no cover
            raise V4ActorError("V4 hand and action adapters are unavailable") from error
        try:
            import torch

            inputs = {
                "raster": _as_batch(base.board, device=self._device),
                "global_features": _as_batch(base.global_vector, device=self._device),
                "entities": _as_batch(base.entity_tokens, device=self._device),
                "entity_mask": _as_batch(base.entity_mask, device=self._device, boolean=True),
                "hand_tokens": _as_batch(
                    build_hand_tokens(hand, float(own_elixir)), device=self._device
                ),
                "opp_hand_probs": _as_batch(snapshot.opp_hand_probs, device=self._device),
                "opp_out_of_cycle": _as_batch(
                    snapshot.opp_out_of_cycle, device=self._device, boolean=True
                ),
                "opp_elixir_interval": _as_batch(
                    [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], device=self._device
                ),
                "event_history": _as_batch(snapshot.event_history, device=self._device),
            }
            legal_play = _as_batch(base.legal_play, device=self._device, boolean=True)
            masks = masks_from_legal_play(legal_play, bool(base.legal_wait))
            reset = torch.ones((1, 1), dtype=torch.bool, device=self._device)
            hidden = None
            if self._hidden is not None:
                reset.zero_()
                hidden = self._hidden
            with torch.inference_mode():
                logits, _recurrent, next_hidden = self.policy.forward_step(
                    inputs, hidden=hidden, reset_mask=reset
                )
            self._hidden = next_hidden.detach()
        except V4ActorError:
            raise
        except Exception as error:
            raise V4ActorError(f"V4 inference failed closed: {error}") from error
        output = _StepOutput(logits, masks, self._hidden, hand)
        self._last_step = output
        return output

    def _remember_action(self, action: Any) -> None:
        if str(getattr(action, "kind", "")).casefold() != "play":
            return
        slot = getattr(action, "card_idx", None)
        step = self._last_step
        if step is None or type(slot) is not int or not 0 <= slot < len(step.hand_cards):
            return
        card = step.hand_cards[slot]
        cost = _card_cost(card) if card is not None else None
        if card is not None and cost is not None:
            self._pending_own_play = (card, cost)

    def decide(self, observation: Any) -> Any:
        step = self._forward(observation)
        try:
            actions = self.policy.act_deterministic(step.logits, step.masks)
            action = _decode_action(actions)
        except Exception as error:
            raise V4ActorError(f"could not decode V4 action: {error}") from error
        self._remember_action(action)
        return action

    def decide_with_scores(self, observation: Any) -> tuple[Any, list[Any], dict[str, float]]:
        """Score and decode one V4 step from the same forward pass."""

        step = self._forward(observation)
        try:
            try:
                from src.frontend.scoring import score_action_heads
            except ImportError:
                from frontend.scoring import score_action_heads  # type: ignore
            action, suggestions, diagnostics = score_action_heads(
                step.logits,
                step.masks,
                hand_cards=step.hand_cards,
            )
        except Exception as error:
            raise V4ActorError(f"could not score V4 action: {error}") from error
        self._remember_action(action)
        return action, suggestions, diagnostics


__all__ = ["V4Actor", "V4ActorError"]
