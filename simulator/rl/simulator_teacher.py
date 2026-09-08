"""Simulator-grounded supervised teacher for V4 distillation (paper Stage 1).

The teacher maps a public observation to a supervised target
``(mode, card, placement, WAIT duration)`` using validated tactical rules
plus simulator legality.  It is deliberately *not* a strategy oracle: each
rule keys off immediate public facts (threat position, cluster density,
elixir, affordability) and every proposal is clamped to the legal mask.
Over time, counterfactual search (Stage 2) replaces this teacher as the
authority; the teacher's job now is to bootstrap basic tactics from
causally clean simulator states instead of unverified expert video.

Entity feature order follows ``observation_v2.ENTITY_TOKEN_FEATURES``;
hand-token order follows ``observation_v3.HAND_TOKEN_FEATURES``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class TeacherError(ValueError):
    """Raised when no legal supervised target exists for a state."""


# observation_v2 entity columns.
_COL_SIDE = 1
_COL_X = 2
_COL_Y = 3
_COL_HP = 4
_COL_IS_AIR = 5
_COL_IS_BUILDING = 6

# observation_v3 hand-token columns.
_HAND_COST = 1
_HAND_AFFORDABLE = 3
_HAND_PLAYABLE = 4
_HAND_WIN_CONDITION = 5
_HAND_SPELL = 6
_HAND_BUILDING = 7
_HAND_AIR_ATTACKER = 9


@dataclass(frozen=True, slots=True)
class TeacherTarget:
    """One supervised decision target (all indices are integer labels)."""

    mode: int  # 0 = WAIT, 1 = PLAY
    card_slot: int
    row: int
    col: int
    wait_duration_idx: int  # index into model_v4.WAIT_DURATIONS


def _playable(hand_tokens: np.ndarray, card_mask: np.ndarray) -> np.ndarray:
    affordable = hand_tokens[:, _HAND_AFFORDABLE] > 0.5
    playable = hand_tokens[:, _HAND_PLAYABLE] > 0.5
    return affordable & playable & card_mask


def nearest_legal_cell(
    row: int, col: int, legal: np.ndarray, *, max_radius: int = 4
) -> tuple[int, int] | None:
    """Spiral-search the nearest legal cell (Chebyshev rings, row-major)."""

    rows, cols = legal.shape
    for radius in range(max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                candidate = (row + dr, col + dc)
                if 0 <= candidate[0] < rows and 0 <= candidate[1] < cols:
                    if bool(legal[candidate]):
                        return candidate
    return None


def soft_placement_target(
    row: int, col: int, legal: np.ndarray
) -> np.ndarray:
    """Spread mass uniformly over legal cells within Chebyshev distance 1.

    Near-equivalent placements share supervision instead of forcing one
    arbitrary cell; cells outside the neighborhood get zero mass.
    """

    rows, cols = legal.shape
    probs = np.zeros((rows, cols), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            candidate = (row + dr, col + dc)
            if 0 <= candidate[0] < rows and 0 <= candidate[1] < cols:
                if bool(legal[candidate]):
                    probs[candidate] = 1.0
    total = float(probs.sum())
    if total <= 0.0:
        raise TeacherError("no legal cell in the placement neighborhood")
    return probs / np.float32(total)


def _to_cell(y_norm: float, x_norm: float, rows: int, cols: int) -> tuple[int, int]:
    row = int(min(max(float(y_norm), 0.0), 1.0) * rows)
    col = int(min(max(float(x_norm), 0.0), 1.0) * cols)
    return (min(row, rows - 1), min(col, cols - 1))


def wait_duration_for_elixir(elixir: float) -> int:
    """Map elixir to a WAIT-duration index (longer waits when broke)."""

    if elixir < 3.0:
        return 3
    if elixir < 6.0:
        return 2
    if elixir < 8.0:
        return 1
    return 0


def teacher_label(
    *,
    hand_tokens: np.ndarray,
    entity_tokens: np.ndarray,
    entity_mask: np.ndarray,
    legal_play: np.ndarray,
    legal_wait: bool,
    own_elixir: float,
) -> TeacherTarget:
    """Produce the supervised target for one public state.

    Rule priority: air-aware defense of the most advanced threat, spell
    value on clusters, win-condition pressure when safe and rich, cheap
    cycling when floating elixir, otherwise WAIT with an elixir-aware
    duration.  Every PLAY proposal is resolved through the legal mask;
    when the preferred rule has no legal cell the teacher falls through
    to WAIT (or raises if even WAIT is illegal).
    """

    rows, cols = legal_play.shape[1], legal_play.shape[2]
    card_mask = legal_play.reshape(4, -1).any(axis=1)
    playable = _playable(hand_tokens, card_mask)
    enemies = entity_tokens[entity_mask & (entity_tokens[:, _COL_SIDE] > 0.5)]
    threats = enemies[enemies[:, _COL_Y] > 0.45] if enemies.size else enemies[:0]

    def play(slot: int, row: int, col: int) -> TeacherTarget | None:
        cell = nearest_legal_cell(row, col, np.asarray(legal_play[int(slot)]))
        if cell is None:
            return None
        return TeacherTarget(
            mode=1,
            card_slot=int(slot),
            row=int(cell[0]),
            col=int(cell[1]),
            wait_duration_idx=0,
        )

    # 1. Defense: answer the most advanced threat, preferring an
    #    air-attacker against air and a building/swarm otherwise.
    if len(threats) and bool(playable.any()):
        order = np.argsort(-threats[:, _COL_Y], kind="stable")
        target = threats[order[0]]
        need_air = bool(target[_COL_IS_AIR] > 0.5)
        candidates = np.flatnonzero(playable)
        if need_air:
            air_ok = candidates[hand_tokens[candidates, _HAND_AIR_ATTACKER] > 0.5]
            if len(air_ok):
                candidates = air_ok
        else:
            ground = candidates[
                (hand_tokens[candidates, _HAND_BUILDING] > 0.5)
                | (hand_tokens[candidates, _HAND_COST] <= 0.3)
            ]
            if len(ground):
                candidates = ground
        threat_cell = _to_cell(float(target[_COL_Y]), float(target[_COL_X]), rows, cols)
        # Kite/pull toward our side: two rows below the threat.
        anchor = (min(threat_cell[0] + 2, rows - 1), threat_cell[1])
        for slot in candidates:
            if hand_tokens[int(slot), _HAND_SPELL] > 0.5:
                continue
            decided = play(int(slot), *anchor)
            if decided is not None:
                return decided

    # 2. Spell value: a cluster of two or more enemies within a small box.
    if len(enemies) >= 2 and bool(playable.any()):
        positions = enemies[:, (_COL_Y, _COL_X)]
        spread = np.abs(positions[:, None, :] - positions[None, :, :]).max(axis=-1)
        clustered = bool((spread < 0.12).sum(axis=1).max() >= 2)
        if clustered:
            centroid = positions.mean(axis=0)
            anchor = _to_cell(float(centroid[0]), float(centroid[1]), rows, cols)
            spells = np.flatnonzero(
                playable & (hand_tokens[:, _HAND_SPELL] > 0.5)
            )
            for slot in spells:
                decided = play(int(slot), *anchor)
                if decided is not None:
                    return decided

    # 3. Win-condition pressure when safe and rich.
    if not len(threats) and own_elixir >= 6.0 and bool(playable.any()):
        win = np.flatnonzero(
            playable & (hand_tokens[:, _HAND_WIN_CONDITION] > 0.5)
        )
        if len(win):
            left = int((enemies[:, _COL_X] < 0.5).sum()) if len(enemies) else 0
            right = int((enemies[:, _COL_X] >= 0.5).sum()) if len(enemies) else 0
            lane_col = 4 if left <= right else cols - 5
            for slot in win:
                decided = play(int(slot), rows // 2, lane_col)
                if decided is not None:
                    return decided

    # 4. Cheap cycle when floating elixir with nothing to answer.
    if not len(threats) and own_elixir >= 8.0 and bool(playable.any()):
        cheap = np.flatnonzero(
            playable & (hand_tokens[:, _HAND_COST] <= 0.2)
        )
        for slot in cheap:
            decided = play(int(slot), rows - 2, 2)
            if decided is not None:
                return decided

    # 5. WAIT with an elixir-aware duration.
    if not bool(legal_wait):
        raise TeacherError("teacher found no legal WAIT or PLAY action")
    return TeacherTarget(
        mode=0,
        card_slot=0,
        row=0,
        col=0,
        wait_duration_idx=wait_duration_for_elixir(float(own_elixir)),
    )


__all__ = [
    "TeacherError",
    "TeacherTarget",
    "nearest_legal_cell",
    "soft_placement_target",
    "teacher_label",
    "wait_duration_for_elixir",
]
