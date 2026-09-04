"""Frontend policy scoring over recurrent actor logits.

This module contains pure scoring helpers that turn one step of
autoregressive policy logits (WAIT/PLAY mode, card slot, card-conditioned
placement) into ranked, human-readable action suggestions.

Design notes:

* No third-party imports happen at module import time.  In particular this
  module must import without ``torch``, ``cv2``, YOLO/OCR runtimes, or ADB
  transport code so unit tests can import it with lightweight mocks.
  ``torch`` (and the simulator/cr_bot boundaries) are imported lazily inside
  the functions that need them.
* All helpers fail closed with :class:`ValueError` on illegal/empty masks,
  shape mismatches, or non-finite floats.
* Placement grids use ``[slot, row, column]`` layout with ``row`` in
  ``[0, 32)`` and ``column`` in ``[0, 18)``.  Public ``cell`` coordinates are
  ``(col, row)`` tuples, matching
  ``simulator.physical_lab.prototype_controller.policy_action_from_batch``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time torch must stay lazy
    import torch


__all__ = [
    "ScoredAction",
    "masked_log_softmax_1d",
    "top_k_suggestions",
    "placement_heatmap",
    "decide_with_scores",
]


_WAIT_INDEX = 0
_PLAY_INDEX = 1


@dataclass(frozen=True, slots=True)
class ScoredAction:
    """One ranked action hypothesis with its factorized log-probability."""

    kind: str  # "wait" or "play"
    card_slot: int | None
    cell: tuple[int, int] | None  # (col, row)
    probability: float
    log_prob: float
    mode_log_prob: float
    card_log_prob: float | None
    placement_log_prob: float | None
    card_name: str | None

    def __post_init__(self) -> None:
        if self.kind not in ("wait", "play"):
            raise ValueError(f"ScoredAction kind must be 'wait' or 'play', got {self.kind!r}")
        probability = float(self.probability)
        if not math.isfinite(probability) or not -1e-6 <= probability <= 1.0 + 1e-6:
            raise ValueError(f"ScoredAction probability must be finite in [0, 1], got {self.probability!r}")
        for name in ("log_prob", "mode_log_prob"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"ScoredAction {name} must be finite, got {getattr(self, name)!r}")
        for name in ("card_log_prob", "placement_log_prob"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"ScoredAction {name} must be finite or None, got {value!r}")
        if self.kind == "wait":
            if self.card_slot is not None:
                raise ValueError("WAIT ScoredAction must have card_slot=None")
            if self.cell is not None:
                raise ValueError("WAIT ScoredAction must have cell=None")
            if self.card_log_prob is not None or self.placement_log_prob is not None:
                raise ValueError("WAIT ScoredAction must have card/placement log-probs of None")
            if self.card_name is not None:
                raise ValueError("WAIT ScoredAction must have card_name=None")
        else:
            if type(self.card_slot) is not int or self.card_slot < 0:
                raise ValueError(f"PLAY ScoredAction needs a non-negative card_slot, got {self.card_slot!r}")
            cell = self.cell
            if (
                not isinstance(cell, tuple)
                or len(cell) != 2
                or type(cell[0]) is not int
                or type(cell[1]) is not int
                or cell[0] < 0
                or cell[1] < 0
            ):
                raise ValueError(f"PLAY ScoredAction needs cell=(col, row) of non-negative ints, got {cell!r}")
            if self.card_log_prob is None or self.placement_log_prob is None:
                raise ValueError("PLAY ScoredAction needs card and placement log-probs")
            if self.card_name is not None and not isinstance(self.card_name, str):
                raise ValueError(f"PLAY ScoredAction card_name must be str or None, got {self.card_name!r}")


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def masked_log_softmax_1d(logits_1d: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
    """Return masked log-softmax over one flat categorical distribution.

    Illegal entries receive ``-inf``.  Fail-closed: raises :class:`ValueError`
    when the inputs are not matching 1D float/bool tensors, when no entry is
    legal, or when any legal logit (or resulting legal log-probability) is
    non-finite.
    """

    import torch

    if not isinstance(logits_1d, torch.Tensor):
        raise ValueError(f"logits_1d must be a torch.Tensor, got {type(logits_1d).__name__}")
    if not isinstance(mask_1d, torch.Tensor):
        raise ValueError(f"mask_1d must be a torch.Tensor, got {type(mask_1d).__name__}")
    if logits_1d.ndim != 1:
        raise ValueError(f"logits_1d must be 1D, got shape {tuple(logits_1d.shape)}")
    if mask_1d.ndim != 1:
        raise ValueError(f"mask_1d must be 1D, got shape {tuple(mask_1d.shape)}")
    if logits_1d.shape != mask_1d.shape:
        raise ValueError(
            "logits_1d and mask_1d must have the same shape, "
            f"got {tuple(logits_1d.shape)} vs {tuple(mask_1d.shape)}"
        )
    if mask_1d.dtype != torch.bool:
        raise ValueError(f"mask_1d must use dtype torch.bool, got {mask_1d.dtype}")
    if not logits_1d.is_floating_point():
        raise ValueError(f"logits_1d must use a floating-point dtype, got {logits_1d.dtype}")
    if logits_1d.numel() == 0:
        raise ValueError("masked_log_softmax_1d: empty distribution (fail-closed, no legal entries)")
    if not bool(mask_1d.any().item()):
        raise ValueError("masked_log_softmax_1d: no legal entries (fail-closed)")
    legal_logits = logits_1d[mask_1d]
    if not bool(torch.isfinite(legal_logits).all().item()):
        raise ValueError("masked_log_softmax_1d: legal logits must all be finite (fail-closed)")
    masked = torch.where(mask_1d, logits_1d, torch.full_like(logits_1d, float("-inf")))
    result = torch.log_softmax(masked, dim=-1)
    if not bool(torch.isfinite(result[mask_1d]).all().item()):
        raise ValueError("masked_log_softmax_1d: legal log-probabilities must all be finite (fail-closed)")
    return result


def _unpack_masks(masks: Any) -> tuple[Any, Any, Any]:
    """Accept ActionMasks-like objects, 3-tuples, or dicts of torch masks."""

    if hasattr(masks, "mode") and hasattr(masks, "card") and hasattr(masks, "placement"):
        return masks.mode, masks.card, masks.placement
    if isinstance(masks, dict):
        try:
            return masks["mode"], masks["card"], masks["placement"]
        except KeyError as error:
            raise ValueError(
                "masks dict must contain 'mode', 'card', and 'placement' entries"
            ) from error
    if isinstance(masks, (list, tuple)) and len(masks) == 3:
        return masks[0], masks[1], masks[2]
    raise ValueError(
        "masks must be an ActionMasks (mode/card/placement attributes), "
        "a (mode_mask, card_mask, placement_mask) triple, or a dict with "
        "'mode'/'card'/'placement' entries"
    )


def _card_name_at(hand_cards: Sequence | None, slot: int) -> str | None:
    if hand_cards is None:
        return None
    if isinstance(hand_cards, (str, bytes)) or not isinstance(hand_cards, Sequence):
        raise ValueError("hand_cards must be a sequence of card names or None")
    try:
        length = len(hand_cards)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("hand_cards must be a sized sequence or None") from error
    if slot < 0 or slot >= length:
        return None
    try:
        raw = hand_cards[slot]
    except (IndexError, TypeError, KeyError):
        return None
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    return raw


def top_k_suggestions(
    mode_logits: torch.Tensor,
    card_logits: torch.Tensor,
    placement_logits: torch.Tensor,
    masks: Any,
    *,
    hand_cards: Sequence | None = None,
    top_k: int = 3,
) -> list[ScoredAction]:
    """Rank WAIT plus every legal (card slot, cell) play by joint probability.

    ``mode_logits``/``card_logits`` are 1D slices for ``B=0, T=0`` with shapes
    ``[2]``/``[K]`` and ``placement_logits`` is ``[K, rows, cols]``; ``masks``
    carries the same-shaped bool legality tensors (as an ``ActionMasks``, a
    triple, or a dict).

    The joint of a play is
    ``P(mode=PLAY) * P(card=s | PLAY) * P(cell | card=s)``.  Candidates are
    sorted by descending log-probability and truncated to ``top_k``.  When
    PLAY is illegal only WAIT is returned; a legal card slot whose placement
    grid has no legal cell is skipped.
    """

    import torch

    if type(top_k) is not int or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    for name, tensor in (
        ("mode_logits", mode_logits),
        ("card_logits", card_logits),
        ("placement_logits", placement_logits),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype, got {tensor.dtype}")
    mode_mask, card_mask, placement_mask = _unpack_masks(masks)
    for name, tensor in (
        ("mode mask", mode_mask),
        ("card mask", card_mask),
        ("placement mask", placement_mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dtype != torch.bool:
            raise ValueError(f"{name} must use dtype torch.bool, got {tensor.dtype}")
    if mode_logits.ndim != 1 or mode_logits.shape[-1] != 2:
        raise ValueError(f"mode_logits must have shape [2], got {tuple(mode_logits.shape)}")
    if mode_mask.ndim != 1 or mode_mask.shape != mode_logits.shape:
        raise ValueError(
            f"mode mask must match mode_logits shape {tuple(mode_logits.shape)}, "
            f"got {tuple(mode_mask.shape)}"
        )
    if card_logits.ndim != 1 or card_logits.shape != card_mask.shape:
        raise ValueError(
            "card_logits and card mask must share one shape, "
            f"got {tuple(card_logits.shape)} vs {tuple(card_mask.shape)}"
        )
    card_slots = int(card_logits.shape[0])
    if card_slots < 1:
        raise ValueError("card_logits must contain at least one card slot")
    if placement_logits.ndim != 3:
        raise ValueError(f"placement_logits must have shape [slots, rows, cols], got {tuple(placement_logits.shape)}")
    if placement_mask.shape != placement_logits.shape:
        raise ValueError(
            "placement mask must match placement_logits shape "
            f"{tuple(placement_logits.shape)}, got {tuple(placement_mask.shape)}"
        )
    if int(placement_logits.shape[0]) != card_slots:
        raise ValueError(
            "placement_logits card dimension must match card_logits slots, "
            f"got {tuple(placement_logits.shape)} vs {tuple(card_logits.shape)}"
        )
    rows = int(placement_logits.shape[1])
    cols = int(placement_logits.shape[2])
    if rows < 1 or cols < 1:
        raise ValueError(f"placement grid must be non-empty, got {(rows, cols)}")

    mode_log_probs = masked_log_softmax_1d(mode_logits, mode_mask)
    wait_legal = bool(mode_mask[_WAIT_INDEX].item())
    play_legal = bool(mode_mask[_PLAY_INDEX].item())
    mode_lp_wait = _finite_float("mode log-prob(wait)", mode_log_probs[_WAIT_INDEX].item()) if wait_legal else None
    mode_lp_play = _finite_float("mode log-prob(play)", mode_log_probs[_PLAY_INDEX].item()) if play_legal else None

    candidates: list[ScoredAction] = []
    if wait_legal:
        assert mode_lp_wait is not None
        probability = _finite_float("P(wait)", math.exp(mode_lp_wait))
        if not 0.0 - 1e-6 <= probability <= 1.0 + 1e-6:
            raise ValueError(f"P(wait) out of range: {probability!r}")
        candidates.append(
            ScoredAction(
                kind="wait",
                card_slot=None,
                cell=None,
                probability=probability,
                log_prob=mode_lp_wait,
                mode_log_prob=mode_lp_wait,
                card_log_prob=None,
                placement_log_prob=None,
                card_name=None,
            )
        )

    if play_legal:
        assert mode_lp_play is not None
        if not bool(card_mask.any().item()):
            # PLAY is nominally legal but no card slot is.  Fail closed when
            # WAIT cannot cover the decision; otherwise surface WAIT alone.
            if not wait_legal:
                raise ValueError("PLAY is legal but the card mask has no legal slots (fail-closed)")
        else:
            card_log_probs = masked_log_softmax_1d(card_logits, card_mask)
            for slot in range(card_slots):
                if not bool(card_mask[slot].item()):
                    continue
                cell_mask = placement_mask[slot]
                if not bool(cell_mask.any().item()):
                    # Legal card with no legal placement: skip this slot.
                    continue
                card_lp = _finite_float(f"card log-prob(slot {slot})", card_log_probs[slot].item())
                flat_logits = placement_logits[slot].reshape(-1)
                flat_mask = cell_mask.reshape(-1)
                place_log_probs = masked_log_softmax_1d(flat_logits, flat_mask)
                legal_indices = torch.nonzero(flat_mask, as_tuple=False).flatten().tolist()
                for flat_index in legal_indices:
                    flat_index = int(flat_index)
                    row, col = divmod(flat_index, cols)
                    place_lp = _finite_float(
                        f"placement log-prob(slot {slot} cell {(col, row)})",
                        place_log_probs[flat_index].item(),
                    )
                    joint_log_prob = _finite_float(
                        f"joint log-prob(slot {slot} cell {(col, row)})",
                        mode_lp_play + card_lp + place_lp,
                    )
                    probability = _finite_float(
                        f"joint P(slot {slot} cell {(col, row)})",
                        math.exp(joint_log_prob),
                    )
                    if not 0.0 - 1e-6 <= probability <= 1.0 + 1e-6:
                        raise ValueError(f"joint play probability out of range: {probability!r}")
                    candidates.append(
                        ScoredAction(
                            kind="play",
                            card_slot=slot,
                            cell=(col, row),
                            probability=probability,
                            log_prob=joint_log_prob,
                            mode_log_prob=mode_lp_play,
                            card_log_prob=card_lp,
                            placement_log_prob=place_lp,
                            card_name=_card_name_at(hand_cards, slot),
                        )
                    )

    if not candidates:
        raise ValueError("top_k_suggestions: no scorable actions (fail-closed)")

    def _rank_key(action: ScoredAction) -> tuple[float, int, int, int, int]:
        if action.kind == "wait":
            return (-action.log_prob, 0, -1, -1, -1)
        assert action.card_slot is not None and action.cell is not None
        return (-action.log_prob, 1, action.card_slot, action.cell[1], action.cell[0])

    candidates.sort(key=_rank_key)
    return candidates[:top_k]


def placement_heatmap(
    placement_logits: torch.Tensor,
    placement_masks: torch.Tensor,
    card_slot: int,
) -> list[list[float]]:
    """Return the ``[rows, cols]`` placement probability grid for one slot.

    Softmax is taken over the legal cells of ``card_slot``; illegal cells are
    ``0.0``.  Fail-closed with :class:`ValueError` on bad shapes, an
    out-of-range slot, non-finite legal logits, or a slot with no legal cell.
    """

    import torch

    if not isinstance(placement_logits, torch.Tensor):
        raise ValueError(f"placement_logits must be a torch.Tensor, got {type(placement_logits).__name__}")
    if not isinstance(placement_masks, torch.Tensor):
        raise ValueError(f"placement_masks must be a torch.Tensor, got {type(placement_masks).__name__}")
    if placement_logits.ndim != 3:
        raise ValueError(
            f"placement_logits must have shape [slots, rows, cols], got {tuple(placement_logits.shape)}"
        )
    if placement_masks.shape != placement_logits.shape:
        raise ValueError(
            "placement_masks must match placement_logits shape "
            f"{tuple(placement_logits.shape)}, got {tuple(placement_masks.shape)}"
        )
    if placement_masks.dtype != torch.bool:
        raise ValueError(f"placement_masks must use dtype torch.bool, got {placement_masks.dtype}")
    if not placement_logits.is_floating_point():
        raise ValueError(f"placement_logits must use a floating-point dtype, got {placement_logits.dtype}")
    card_slots = int(placement_logits.shape[0])
    if type(card_slot) is not int or not 0 <= card_slot < card_slots:
        raise ValueError(f"card_slot must be an integer in [0, {card_slots}), got {card_slot!r}")
    logits_2d = placement_logits[card_slot]
    mask_2d = placement_masks[card_slot]
    if not bool(mask_2d.any().item()):
        raise ValueError(f"placement_heatmap: card slot {card_slot} has no legal cells (fail-closed)")
    log_probs = masked_log_softmax_1d(logits_2d.reshape(-1), mask_2d.reshape(-1))
    probs = torch.where(mask_2d.reshape(-1), torch.exp(log_probs), torch.zeros_like(log_probs))
    grid = probs.reshape(int(logits_2d.shape[0]), int(logits_2d.shape[1]))
    total = _finite_float("placement heatmap total", float(grid.sum().item()))
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"placement heatmap must sum to 1.0, got {total!r}")
    nested = grid.detach().to("cpu").tolist()
    for row in nested:
        for value in row:
            _finite_float("placement heatmap cell", value)
    return nested


def _extract_hand_cards(observation: Any) -> list[str] | None:
    """Best-effort public hand-name lookup; ``None`` when unavailable."""

    for attribute in ("hand_cards", "hand"):
        value = getattr(observation, attribute, None)
        if isinstance(value, (list, tuple)) and value:
            return [str(item) if not isinstance(item, str) else item for item in value]
    for wrapper in ("game_state", "state"):
        inner = getattr(observation, wrapper, None)
        if inner is not None:
            for attribute in ("hand_cards", "hand"):
                value = getattr(inner, attribute, None)
                if isinstance(value, (list, tuple)) and value:
                    return [str(item) if not isinstance(item, str) else item for item in value]
            hud = getattr(inner, "hud", None)
            value = getattr(hud, "hand_cards", None)
            if isinstance(value, (list, tuple)) and value:
                return [str(item) if not isinstance(item, str) else item for item in value]
    hud = getattr(observation, "hud", None)
    value = getattr(hud, "hand_cards", None)
    if isinstance(value, (list, tuple)) and value:
        return [str(item) if not isinstance(item, str) else item for item in value]
    return None


def decide_with_scores(actor: Any, observation: Any) -> tuple[Any, list[ScoredAction], dict[str, float]]:
    """Single-forward-pass replacement for ``PrototypeActor.decide``.

    Builds model inputs via ``observation_to_model_inputs``, manages
    ``actor._hidden`` plus the reset mask exactly like
    ``PrototypeActor.decide`` (fresh initial hidden plus all-ones reset on the
    first step, carried hidden plus all-zeros reset afterwards), runs one
    ``actor.policy.forward(..., include_beliefs=False, inference=True)`` pass
    under inference mode, ranks suggestions with :func:`top_k_suggestions`,
    derives the deterministic action as the rank-1 suggestion, stores
    ``actor._hidden = final_hidden.detach()``, and returns
    ``(Action, suggestions, diagnostics)`` where diagnostics carries
    ``mode_prob_wait``, ``mode_prob_play``, ``entropy``, and ``top_log_prob``.

    Never calls ``act_deterministic`` (which would advance the hidden state a
    second time).  ``torch`` and the simulator/cr_bot boundaries are imported
    lazily so this module stays importable without ``torch``.
    """

    import math

    import torch
    from simulator.physical_lab.prototype_controller import observation_to_model_inputs

    from cr_bot.domain.game_state import Action

    if actor is None:
        raise ValueError("decide_with_scores: actor must not be None (fail-closed)")
    if observation is None:
        raise ValueError("decide_with_scores: observation must not be None (fail-closed)")
    policy = getattr(actor, "policy", None)
    if policy is None:
        raise ValueError("decide_with_scores: actor has no .policy recurrent policy (fail-closed)")
    device = getattr(actor, "device", None)
    if device is None:
        raise ValueError("decide_with_scores: actor has no .device (fail-closed)")
    if not callable(getattr(policy, "forward", None)):
        raise ValueError("decide_with_scores: actor.policy has no forward method (fail-closed)")
    if not callable(getattr(policy, "initial_hidden", None)):
        raise ValueError("decide_with_scores: actor.policy has no initial_hidden method (fail-closed)")

    model_inputs = observation_to_model_inputs(observation, device=device)
    try:
        board, global_vector, entity_tokens, entity_mask, masks = model_inputs
    except (TypeError, ValueError) as error:
        raise ValueError(
            "decide_with_scores: observation_to_model_inputs must return "
            "(board, global_vector, entity_tokens, entity_mask, masks) (fail-closed)"
        ) from error

    torch_mod = getattr(actor, "_torch", None) or torch
    use_initial_hidden = getattr(actor, "_hidden", None) is None
    if use_initial_hidden:
        hidden = policy.initial_hidden(1, device=device)
        reset_mask = torch_mod.ones((1, 1), dtype=torch_mod.bool, device=device)
    else:
        hidden = actor._hidden
        reset_mask = torch_mod.zeros((1, 1), dtype=torch_mod.bool, device=device)
    if (
        not isinstance(reset_mask, torch.Tensor)
        or tuple(reset_mask.shape) != (1, 1)
        or reset_mask.dtype != torch.bool
    ):
        raise ValueError("decide_with_scores: reset_mask must be a bool tensor with shape [1, 1] (fail-closed)")

    inference_mode = getattr(torch_mod, "inference_mode", None) or torch.inference_mode
    with inference_mode():
        output = policy.forward(
            board,
            global_vector,
            entity_tokens,
            entity_mask,
            action_masks=masks,
            reset_mask=reset_mask,
            hidden=hidden,
            include_beliefs=False,
            inference=True,
        )

    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("decide_with_scores: policy.forward returned no action logits (fail-closed)")
    final_hidden = getattr(output, "final_hidden", None)
    if final_hidden is None:
        raise ValueError("decide_with_scores: policy.forward returned no final_hidden (fail-closed)")
    try:
        mode_logits = logits.mode[0, 0]
        card_logits = logits.card[0, 0]
        placement_logits = logits.placement[0, 0]
        mode_mask = masks.mode[0, 0]
        card_mask = masks.card[0, 0]
        placement_mask = masks.placement[0, 0]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError(
            "decide_with_scores: policy logits/masks must expose [B=1, T=1, ...] "
            "mode/card/placement tensors (fail-closed)"
        ) from error

    suggestions = top_k_suggestions(
        mode_logits,
        card_logits,
        placement_logits,
        (mode_mask, card_mask, placement_mask),
        hand_cards=_extract_hand_cards(observation),
        top_k=3,
    )
    if not suggestions:
        raise ValueError("decide_with_scores: no scorable actions (fail-closed)")
    best = suggestions[0]
    if best.kind == "wait":
        action = Action(kind="Wait")
    else:
        if best.card_slot is None or best.cell is None:
            raise ValueError("decide_with_scores: rank-1 play is missing slot/cell (fail-closed)")
        action = Action(kind="Play", card_idx=int(best.card_slot), cell=(int(best.cell[0]), int(best.cell[1])))

    try:
        actor._hidden = final_hidden.detach()
    except (AttributeError, RuntimeError) as error:
        raise ValueError(f"decide_with_scores: could not detach final hidden state: {error}") from error

    mode_log_probs = masked_log_softmax_1d(mode_logits, mode_mask)
    wait_legal = bool(mode_mask[_WAIT_INDEX].item())
    play_legal = bool(mode_mask[_PLAY_INDEX].item())
    mode_prob_wait = (
        _finite_float("mode_prob_wait", math.exp(float(mode_log_probs[_WAIT_INDEX].item())))
        if wait_legal
        else 0.0
    )
    mode_prob_play = (
        _finite_float("mode_prob_play", math.exp(float(mode_log_probs[_PLAY_INDEX].item())))
        if play_legal
        else 0.0
    )
    for name, value in (("mode_prob_wait", mode_prob_wait), ("mode_prob_play", mode_prob_play)):
        if not 0.0 - 1e-6 <= value <= 1.0 + 1e-6:
            raise ValueError(f"decide_with_scores: {name} out of range: {value!r}")

    # Exact autoregressive joint entropy: H(mode) + P(PLAY) * H(card | PLAY)
    # + P(PLAY) * sum_s P(card=s | PLAY) * H(placement | s).  Card slots that
    # are legal but have no legal placement cell contribute nothing (they are
    # already excluded from the ranked suggestions).
    entropy = 0.0
    for index in range(2):
        if bool(mode_mask[index].item()):
            log_prob = _finite_float(
                f"mode entropy log-prob({index})", float(mode_log_probs[index].item())
            )
            entropy -= mode_prob_wait * log_prob if index == _WAIT_INDEX else mode_prob_play * log_prob
    if play_legal and mode_prob_play > 0.0 and bool(card_mask.any().item()):
        card_log_probs = masked_log_softmax_1d(card_logits, card_mask)
        card_entropy = 0.0
        slot_prob: dict[int, float] = {}
        for slot in range(int(card_logits.shape[0])):
            if not bool(card_mask[slot].item()):
                continue
            card_lp = _finite_float(
                f"card entropy log-prob({slot})", float(card_log_probs[slot].item())
            )
            prob = math.exp(card_lp)
            slot_prob[slot] = prob
            card_entropy -= prob * card_lp
        entropy += mode_prob_play * card_entropy
        for slot, prob in slot_prob.items():
            cell_mask = placement_mask[slot]
            if not bool(cell_mask.any().item()):
                continue
            place_log_probs = masked_log_softmax_1d(
                placement_logits[slot].reshape(-1), cell_mask.reshape(-1)
            )
            place_entropy = 0.0
            for flat_index in torch.nonzero(cell_mask.reshape(-1), as_tuple=False).flatten().tolist():
                place_lp = _finite_float(
                    f"placement entropy log-prob({int(flat_index)})",
                    float(place_log_probs[int(flat_index)].item()),
                )
                place_entropy -= math.exp(place_lp) * place_lp
            entropy += mode_prob_play * prob * place_entropy
    entropy = _finite_float("entropy", entropy)
    if entropy < -1e-6:
        raise ValueError(f"decide_with_scores: entropy must be non-negative, got {entropy!r}")

    diagnostics = {
        "mode_prob_wait": mode_prob_wait,
        "mode_prob_play": mode_prob_play,
        "entropy": entropy,
        "top_log_prob": _finite_float("top_log_prob", best.log_prob),
    }
    return action, suggestions, diagnostics
