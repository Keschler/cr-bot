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
  the deployed V4 actor action adapter.
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
    "score_action_heads",
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


@dataclass(frozen=True, slots=True)
class _ScoringHeads:
    """Validated head values copied to Python once per distribution.

    Normalization stays on the input device/dtype; Python floats retain the
    original double-precision joint and entropy arithmetic.  No tensor scalar
    reads are needed while visiting placement cells.
    """

    mode_log_probs: list[float]
    mode_legal: list[bool]
    card_log_probs: list[float] | None
    card_legal: list[bool]
    placements: list[list[tuple[int, float]]]
    cols: int


def _prepare_scoring_heads(
    mode_logits: torch.Tensor,
    card_logits: torch.Tensor,
    placement_logits: torch.Tensor,
    masks: Any,
) -> _ScoringHeads:
    """Validate and normalize once, with bulk transfers bounded by slots."""

    import torch

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

    mode_log_probs = masked_log_softmax_1d(mode_logits, mode_mask).detach().cpu().tolist()
    mode_legal = mode_mask.detach().cpu().tolist()
    card_legal = card_mask.detach().cpu().tolist()
    placements: list[list[tuple[int, float]]] = [[] for _ in range(card_slots)]
    card_log_probs = None
    if mode_legal[_PLAY_INDEX]:
        if not any(card_legal):
            # WAIT may cover a nominally legal PLAY with no legal card slots.
            if not mode_legal[_WAIT_INDEX]:
                raise ValueError("PLAY is legal but the card mask has no legal slots (fail-closed)")
        else:
            card_log_probs = masked_log_softmax_1d(card_logits, card_mask).detach().cpu().tolist()
            placement_legal = placement_mask.reshape(card_slots, -1).detach().cpu().tolist()
            for slot, slot_legal in enumerate(card_legal):
                if not slot_legal or not any(placement_legal[slot]):
                    continue
                log_probs = masked_log_softmax_1d(
                    placement_logits[slot].reshape(-1), placement_mask[slot].reshape(-1)
                ).detach().cpu().tolist()
                placements[slot] = [
                    (index, log_prob)
                    for index, (legal, log_prob) in enumerate(zip(placement_legal[slot], log_probs))
                    if legal
                ]
    return _ScoringHeads(mode_log_probs, mode_legal, card_log_probs, card_legal, placements, cols)


def _top_k_from_heads(
    heads: _ScoringHeads,
    *,
    hand_cards: Sequence | None,
    top_k: int,
) -> list[ScoredAction]:
    # Tuple layout: joint log-prob, slot (-1 for WAIT), flat cell, probability,
    # card log-prob, placement log-prob, card name.  Only the selected entries
    # become ScoredActions, but validate every candidate before truncating.
    candidates: list[tuple[float, int, int, float, float | None, float | None, str | None]] = []
    if heads.mode_legal[_WAIT_INDEX]:
        mode_lp_wait = _finite_float("mode log-prob(wait)", heads.mode_log_probs[_WAIT_INDEX])
        probability = _finite_float("P(wait)", math.exp(mode_lp_wait))
        if not -1e-6 <= probability <= 1.0 + 1e-6:
            raise ValueError(f"P(wait) out of range: {probability!r}")
        candidates.append((mode_lp_wait, -1, -1, probability, None, None, None))

    if heads.mode_legal[_PLAY_INDEX] and heads.card_log_probs is not None:
        mode_lp_play = _finite_float("mode log-prob(play)", heads.mode_log_probs[_PLAY_INDEX])
        for slot, cells in enumerate(heads.placements):
            if not cells:
                continue
            card_lp = _finite_float(f"card log-prob(slot {slot})", heads.card_log_probs[slot])
            card_name = _card_name_at(hand_cards, slot)
            for flat_index, place_lp in cells:
                row, col = divmod(flat_index, heads.cols)
                joint_log_prob = _finite_float(
                    f"joint log-prob(slot {slot} cell {(col, row)})",
                    mode_lp_play + card_lp + place_lp,
                )
                probability = _finite_float(
                    f"joint P(slot {slot} cell {(col, row)})", math.exp(joint_log_prob)
                )
                if not -1e-6 <= probability <= 1.0 + 1e-6:
                    raise ValueError(f"joint play probability out of range: {probability!r}")
                candidates.append((joint_log_prob, slot, flat_index, probability, card_lp, place_lp, card_name))

    if not candidates:
        raise ValueError("top_k_suggestions: no scorable actions (fail-closed)")
    # Flat cells are row-major, so this exactly preserves WAIT, slot, row, col
    # ordering on ties without device-dependent topk tie selection.
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    suggestions = []
    for log_prob, slot, flat_index, probability, card_lp, place_lp, card_name in candidates[:top_k]:
        is_wait = slot == -1
        row, col = divmod(flat_index, heads.cols)
        suggestions.append(
            ScoredAction(
                kind="wait" if is_wait else "play",
                card_slot=None if is_wait else slot,
                cell=None if is_wait else (col, row),
                probability=probability,
                log_prob=log_prob,
                mode_log_prob=heads.mode_log_probs[_WAIT_INDEX if is_wait else _PLAY_INDEX],
                card_log_prob=card_lp,
                placement_log_prob=place_lp,
                card_name=card_name,
            )
        )
    return suggestions


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
    sorted by descending log-probability and truncated to ``top_k``.  Ties
    prefer WAIT, then slot, row, and column in ascending order.  When PLAY is
    illegal only WAIT is returned; a legal card slot whose placement grid has
    no legal cell is skipped.
    """

    if type(top_k) is not int or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    heads = _prepare_scoring_heads(mode_logits, card_logits, placement_logits, masks)
    return _top_k_from_heads(heads, hand_cards=hand_cards, top_k=top_k)


def score_action_heads(
    logits: Any,
    masks: Any,
    *,
    hand_cards: Sequence | None = None,
    top_k: int = 3,
) -> tuple[Any, list[ScoredAction], dict[str, float]]:
    """Rank one V4 policy output without advancing recurrent state.

    V4 and the former actor share the same mode/card/placement factorization,
    while their recurrent forward APIs differ.  Keeping this pure head
    scoring helper separate lets deployment adapters own model state and lets
    the frontend use one probability calculation for both paths.
    """

    try:
        mode = logits.mode[0, 0]
        card = logits.card[0, 0]
        placement = logits.placement[0, 0]
        mode_mask = masks.mode[0, 0]
        card_mask = masks.card[0, 0]
        placement_mask = masks.placement[0, 0]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("policy heads must expose singleton V4 logits and masks") from error
    if type(top_k) is not int or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    heads = _prepare_scoring_heads(
        mode,
        card,
        placement,
        (mode_mask, card_mask, placement_mask),
    )
    suggestions = _top_k_from_heads(
        heads,
        hand_cards=hand_cards,
        top_k=top_k,
    )
    best = suggestions[0]
    try:
        from cr_bot.domain.game_state import Action
    except ImportError as error:  # pragma: no cover - deployment environment
        raise ValueError("cr_bot is required to decode scored actions") from error
    if best.kind == "wait":
        action = Action(kind="Wait")
    else:
        if best.card_slot is None or best.cell is None:
            raise ValueError("rank-1 play is missing slot/cell")
        action = Action(
            kind="Play",
            card_idx=int(best.card_slot),
            cell=(int(best.cell[0]), int(best.cell[1])),
        )

    mode_log_probs = heads.mode_log_probs
    wait_legal = heads.mode_legal[_WAIT_INDEX]
    play_legal = heads.mode_legal[_PLAY_INDEX]
    mode_prob_wait = math.exp(mode_log_probs[_WAIT_INDEX]) if wait_legal else 0.0
    mode_prob_play = math.exp(mode_log_probs[_PLAY_INDEX]) if play_legal else 0.0

    entropy = 0.0
    for index, probability in (
        (_WAIT_INDEX, mode_prob_wait),
        (_PLAY_INDEX, mode_prob_play),
    ):
        if heads.mode_legal[index]:
            entropy -= probability * mode_log_probs[index]
    if play_legal and mode_prob_play > 0.0 and heads.card_log_probs is not None:
        card_log_probs = heads.card_log_probs
        card_entropy = 0.0
        slot_prob: dict[int, float] = {}
        for slot, legal in enumerate(heads.card_legal):
            if not legal:
                continue
            log_prob = card_log_probs[slot]
            probability = math.exp(log_prob)
            slot_prob[slot] = probability
            card_entropy -= probability * log_prob
        entropy += mode_prob_play * card_entropy
        for slot, probability in slot_prob.items():
            place_entropy = 0.0
            for _, log_prob in heads.placements[slot]:
                place_entropy -= math.exp(log_prob) * log_prob
            entropy += mode_prob_play * probability * place_entropy
    entropy = _finite_float("entropy", entropy)
    if entropy < -1e-6:
        raise ValueError(f"entropy must be non-negative, got {entropy!r}")
    diagnostics = {
        "mode_prob_wait": _finite_float("mode_prob_wait", mode_prob_wait),
        "mode_prob_play": _finite_float("mode_prob_play", mode_prob_play),
        "entropy": entropy,
        "top_log_prob": _finite_float("top_log_prob", best.log_prob),
    }
    return action, suggestions, diagnostics


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


def decide_with_scores(actor: Any, observation: Any) -> tuple[Any, list[ScoredAction], dict[str, float]]:
    """Score one V4 actor decision through its deployment adapter."""

    if actor is None or observation is None:
        raise ValueError("actor and public observation are required")
    decide = getattr(actor, "decide_with_scores", None)
    if not callable(decide):
        raise ValueError("actor must implement decide_with_scores")
    return decide(observation)
