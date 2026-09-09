"""Timing-aware teacher for the V4 Stage-1 timing curriculum.

The base rules teacher (:mod:`rl.simulator_teacher`) almost never emits WAIT
when a PLAY is legal, so a dataset built only from its labels cannot teach
WAIT/PLAY timing (the real-state pool is ~96.6% PLAY).  This module adds five
narrow timing rules in front of the base teacher and delegates everything
else to it, so card/placement supervision stays identical:

* T1 opening-scout: match just started, no threats, rich elixir.  Hold a
  short WAIT to see the opponent's first move instead of committing blind.
* T5 preserve-defense: ground threat approaching, Cannon in hand but not yet
  affordable while a cheap card is playable.  WAIT for Cannon instead of
  spending the cheap card and delaying the hard counter.
* T2 defensive-timing: threat still far and the only affordable answers are
  defensive buildings (which tick down from placement).  WAIT until the
  threat enters the placement window.  Troop answers are played immediately
  (they engage on contact), which preserves kiting behavior.
* T3 counterpush: no threats, a surviving own troop, win condition
  affordable.  PLAY the win condition behind the survivor.  This only fills
  the gap where the base teacher WAITs (elixir below its pressure
  threshold); it never overrides a base PLAY.
* T4 avoid-overcommit: own attacker alive (committed push), elixir below the
  cycle threshold.  WAIT instead of adding support/cycle that should be held
  for defense.  Only overrides base win/cycle PLAYs, never defense/spell.

Every rule keys off public observation features (entity positions, hand
affordability, elixir, match clock) plus the same legal mask the policy
sees.  WAIT durations always come from the shared elixir-aware duration
mapping, so duration supervision stays "natural".
"""

from __future__ import annotations

import numpy as np

try:
    from ..observation_v2 import ENTITY_TOKEN_FEATURES
    from ..observation_v3 import HAND_TOKEN_FEATURES
    from .distillation import PLAYER_DECK as FIXED_DECK_TABLE
    from .simulator_teacher import (
        TeacherError,
        TeacherTarget,
        nearest_legal_cell,
        teacher_label,
        wait_duration_for_elixir,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.observation_v2 import ENTITY_TOKEN_FEATURES
    from simulator.observation_v3 import HAND_TOKEN_FEATURES
    from simulator.rl.distillation import PLAYER_DECK as FIXED_DECK_TABLE
    from simulator.rl.simulator_teacher import (
        TeacherError,
        TeacherTarget,
        nearest_legal_cell,
        teacher_label,
        wait_duration_for_elixir,
    )


TIMING_TEACHER_VERSION: str = "timing-rules-v4-0"

_E_SIDE = ENTITY_TOKEN_FEATURES.index("side")
_E_X = ENTITY_TOKEN_FEATURES.index("x")
_E_Y = ENTITY_TOKEN_FEATURES.index("y")
_E_IS_BUILDING = ENTITY_TOKEN_FEATURES.index("is_building")

_H_COST = HAND_TOKEN_FEATURES.index("elixir_cost_norm")
_H_AFFORDABLE = HAND_TOKEN_FEATURES.index("affordable")
_H_WIN = HAND_TOKEN_FEATURES.index("is_win_condition")
_H_SPELL = HAND_TOKEN_FEATURES.index("is_spell")
_H_BUILDING = HAND_TOKEN_FEATURES.index("is_building")

_POLICY_ID = {row[0]: row[1] for row in FIXED_DECK_TABLE}
_POLICY_NORM = {key: pid / 127.0 for key, pid in _POLICY_ID.items()}
_COST = {row[0]: row[2] for row in FIXED_DECK_TABLE}

_THREAT_Y = 0.45
"""Enemy units past this normalized depth count as threats (base-teacher rule)."""


def _slot_with_policy(hand_tokens: np.ndarray, policy_id: int) -> int | None:
    """Return the hand slot holding a card policy id, if any."""

    target = policy_id / 127.0
    for slot in range(hand_tokens.shape[0]):
        if abs(float(hand_tokens[slot, 0]) - target) < 1e-4:
            return slot
    return None


def _threats(entity_tokens: np.ndarray, entity_mask: np.ndarray) -> np.ndarray:
    enemies = entity_tokens[entity_mask & (entity_tokens[:, _E_SIDE] > 0.5)]
    if not len(enemies):
        return enemies[:0]
    return enemies[enemies[:, _E_Y] > _THREAT_Y]


def _own_troops(entity_tokens: np.ndarray, entity_mask: np.ndarray) -> np.ndarray:
    own = entity_tokens[entity_mask & (entity_tokens[:, _E_SIDE] < 0.5)]
    if not len(own):
        return own[:0]
    return own[own[:, _E_IS_BUILDING] < 0.5]


def _wait(own_elixir: float, legal_wait: bool) -> TeacherTarget:
    if not bool(legal_wait):
        raise TeacherError("timing teacher found no legal WAIT or PLAY action")
    return TeacherTarget(
        mode=0,
        card_slot=0,
        row=0,
        col=0,
        wait_duration_idx=wait_duration_for_elixir(float(own_elixir)),
    )


def timing_teacher_label(
    *,
    hand_tokens: np.ndarray,
    entity_tokens: np.ndarray,
    entity_mask: np.ndarray,
    legal_play: np.ndarray,
    legal_wait: bool,
    own_elixir: float,
    tick_seconds: float,
    y_hold: float = 0.60,
    scout_seconds: float = 0.75,
) -> TeacherTarget:
    """Label one public state with the timing-aware teacher.

    ``tick_seconds`` is the public match clock.  ``y_hold`` is the far edge
    of the defensive placement window: threats below it are held against
    when only buildings can answer.  ``scout_seconds`` bounds the opening
    scouting hold.  All other behavior (including every PLAY placement)
    delegates to the base rules teacher.
    """

    rows, cols = legal_play.shape[1], legal_play.shape[2]
    card_mask = legal_play.reshape(4, -1).any(axis=1)
    threats = _threats(entity_tokens, entity_mask)
    elixir = float(own_elixir)

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

    # T1 opening-scout: rich, quiet, brand-new match.  Hold briefly to see
    # the opponent's first move rather than committing blind.
    if (
        float(tick_seconds) < float(scout_seconds)
        and not len(threats)
        and elixir >= 9.0
        and bool(card_mask.any())
    ):
        return _wait(elixir, legal_wait)

    if len(threats):
        need_air = bool((threats[:, ENTITY_TOKEN_FEATURES.index("is_air")] > 0.5).any())
        if not need_air:
            cannon_slot = _slot_with_policy(hand_tokens, _POLICY_ID["cannon"])
            # T5 preserve-defense: Cannon in hand but unaffordable while a
            # cheap card is playable.  Hold for the hard counter instead of
            # spending now and delaying it past the threat's arrival.
            if (
                cannon_slot is not None
                and _COST["cannon"] > elixir
                and elixir >= 1.0
                and bool(card_mask.any())
            ):
                return _wait(elixir, legal_wait)
            # T2 defensive-timing: threat still outside the placement window
            # and the only affordable answers are buildings, which lose
            # lifetime from the moment they are placed.  Troop answers
            # engage on contact and are played immediately (base rule).
            order = np.argsort(-threats[:, _E_Y], kind="stable")
            most_advanced_y = float(threats[order[0]][_E_Y])
            playable = np.flatnonzero(
                (hand_tokens[:, _H_AFFORDABLE] > 0.5) & card_mask
            )
            ground_answers = [
                int(slot)
                for slot in playable
                if hand_tokens[int(slot), _H_SPELL] < 0.5
                and float(hand_tokens[int(slot), _H_COST]) <= 0.3
            ]
            building_only = bool(ground_answers) and all(
                hand_tokens[slot, _H_BUILDING] > 0.5 for slot in ground_answers
            )
            if most_advanced_y < float(y_hold) and building_only:
                return _wait(elixir, legal_wait)

    base = teacher_label(
        hand_tokens=hand_tokens,
        entity_tokens=entity_tokens,
        entity_mask=entity_mask,
        legal_play=legal_play,
        legal_wait=legal_wait,
        own_elixir=elixir,
    )
    survivors = _own_troops(entity_tokens, entity_mask)
    hog_slot = _slot_with_policy(hand_tokens, _POLICY_ID["hog-rider"])

    # T3 counterpush: survivor on board, quiet arena, Hog affordable.  The
    # base teacher WAITs below its pressure threshold; play the counterpush
    # behind the survivor instead.  Never overrides a base PLAY.
    if (
        base.mode == 0
        and not len(threats)
        and len(survivors)
        and hog_slot is not None
        and bool(card_mask[hog_slot])
    ):
        lane_col = 4 if float(survivors[:, _E_X].mean()) < 0.5 else cols - 5
        decided = play(hog_slot, rows // 2, lane_col)
        if decided is not None:
            return decided

    # T4 avoid-overcommit: push already committed and elixir below the cycle
    # threshold.  Hold for defense instead of adding support.  With no live
    # enemies the base teacher can only want win/cycle PLAYs, so overriding
    # its PLAY here never touches defense or spell decisions.
    if (
        base.mode == 1
        and not len(threats)
        and len(survivors)
        and elixir < 8.0
    ):
        return _wait(elixir, legal_wait)

    return base


__all__ = [
    "TIMING_TEACHER_VERSION",
    "timing_teacher_label",
]
