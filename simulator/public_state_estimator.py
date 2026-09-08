"""Deterministic public state estimator for the V4 actor (paper Stage 1).

The estimator sits between simulator public events and the policy.  It
performs public bookkeeping that the GRU should not have to rediscover:
observed card cycle, opponent elixir bounds, entity/event recency.  The
actor never receives hidden simulator state through this path.

API boundary (enforced by design, checked in tests):

* inputs: public events (deployments, tower damage, deaths, action timing),
  observed elixir spending, and the current public tick;
* outputs: state summaries, belief distributions, and event history only.

The estimator must never output an action recommendation.  There is no
``recommend_*``, ``suggest_*``, or ``should_play_*`` method; strategic
choice stays inside the learned policy.  Deterministic game bookkeeping
such as cycle arithmetic and elixir bounds is appropriate here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Final
import numpy as np

from .observation_v3 import (
    BELIEF_CARD_COUNT,
    EVENT_DIM,
    EVENT_HISTORY_LEN,
    EVENT_TYPES,
    PolicyObservationV3,
    uniform_hand_belief,
)


MAX_ELIXIR: Final = 10.0
ELIXIR_REGEN_PER_SECOND: Final = 1.0 / 2.8
EVENT_TYPE_INDEX: Final = {name: index for index, name in enumerate(EVENT_TYPES)}


def belief_index(card_key: str) -> int:
    """Map a public card key to a deterministic belief-vocabulary index."""

    if not isinstance(card_key, str) or not card_key:
        raise ValueError("card_key must be a non-empty string")
    digest = hashlib.sha256(card_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % BELIEF_CARD_COUNT


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """One public game event observed by the estimator."""

    event_type: str
    tick_seconds: float
    magnitude: float = 0.0

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPE_INDEX:
            raise ValueError(f"unknown public event type: {self.event_type!r}")
        if not np.isfinite(float(self.tick_seconds)) or float(self.tick_seconds) < 0.0:
            raise ValueError("tick_seconds must be a finite non-negative number")
        if not np.isfinite(float(self.magnitude)):
            raise ValueError("magnitude must be finite")


@dataclass(frozen=True, slots=True)
class EstimatorSnapshot:
    """Immutable public summary; contains no action recommendation."""

    tick_seconds: float
    opp_elixir_lo: float
    opp_elixir_hi: float
    opp_hand_probs: np.ndarray
    opp_out_of_cycle: np.ndarray
    event_history: np.ndarray


class PublicStateEstimator:
    """Deterministic public bookkeeping between events and the V4 actor."""

    def __init__(
        self,
        *,
        elixir_regen_per_second: float = ELIXIR_REGEN_PER_SECOND,
        deck_size: int = 8,
    ) -> None:
        if not np.isfinite(elixir_regen_per_second) or elixir_regen_per_second <= 0.0:
            raise ValueError("elixir_regen_per_second must be finite and positive")
        if type(deck_size) is not int or deck_size <= 0:
            raise ValueError("deck_size must be a positive integer")
        self._regen = float(elixir_regen_per_second)
        self._deck_size = deck_size
        self._events: list[PublicEvent] = []
        self._seen: dict[int, int] = {}
        self._elixir_lo = 0.0
        self._elixir_hi = MAX_ELIXIR
        self._tick = 0.0

    def reset(self) -> None:
        self._events.clear()
        self._seen.clear()
        self._elixir_lo = 0.0
        self._elixir_hi = MAX_ELIXIR
        self._tick = 0.0

    @property
    def tick_seconds(self) -> float:
        return self._tick

    def observe(self, event: PublicEvent) -> None:
        """Record one public event and advance deterministic bookkeeping."""

        if not isinstance(event, PublicEvent):
            raise TypeError("event must be a PublicEvent")
        if event.tick_seconds < self._tick:
            raise ValueError("events must arrive in non-decreasing tick order")
        self._advance_to(event.tick_seconds)
        self._events.append(event)

    def observe_opponent_spend(
        self, *, tick_seconds: float, card_key: str, elixir_cost: float
    ) -> None:
        """Record an observed opponent deployment and tighten elixir belief."""

        if not np.isfinite(float(elixir_cost)) or not 0.0 <= float(elixir_cost) <= MAX_ELIXIR:
            raise ValueError("elixir_cost must be in [0, 10]")
        if float(tick_seconds) < self._tick:
            raise ValueError("spend must not go backwards in time")
        self._advance_to(float(tick_seconds))
        index = belief_index(card_key)
        self._seen[index] = self._seen.get(index, 0) + 1
        cost = float(elixir_cost)
        self._elixir_hi = max(0.0, self._elixir_hi - cost)
        self._elixir_lo = max(0.0, min(self._elixir_lo, self._elixir_hi))
        self._events.append(
            PublicEvent(event_type="enemy_deploy", tick_seconds=float(tick_seconds), magnitude=cost)
        )

    def _advance_to(self, tick_seconds: float) -> None:
        forward = float(tick_seconds) - self._tick
        if forward > 0.0:
            gain = forward * self._regen
            self._elixir_lo = min(MAX_ELIXIR, self._elixir_lo + gain)
            self._elixir_hi = min(MAX_ELIXIR, self._elixir_hi + gain)
            self._tick = float(tick_seconds)

    def out_of_cycle(self) -> np.ndarray:
        """Cards definitely outside the opponent deck (deterministic rule).

        Once ``deck_size`` distinct opponent cards have been observed, every
        other vocabulary entry is out of cycle.  Before that, nothing is
        ruled out: absence of evidence is not evidence of absence.
        """

        mask = np.zeros((BELIEF_CARD_COUNT,), dtype=bool)
        if len(self._seen) >= self._deck_size:
            mask[:] = True
            for index in self._seen:
                mask[index] = False
        mask.setflags(write=False)
        return mask

    def hand_belief(self) -> np.ndarray:
        """Possible-hand distribution: uniform over in-cycle entries."""

        out = self.out_of_cycle()
        possible = ~out
        probs = np.zeros((BELIEF_CARD_COUNT,), dtype=np.float32)
        count = int(possible.sum())
        if count:
            probs[possible] = np.float32(1.0 / count)
        probs.setflags(write=False)
        return probs

    def elixir_interval(self) -> tuple[float, float]:
        return (float(self._elixir_lo), float(self._elixir_hi))

    def event_history(self) -> np.ndarray:
        """Encode the trailing ring buffer as an ``(16, 8)`` float table."""

        table = np.zeros((EVENT_HISTORY_LEN, EVENT_DIM), dtype=np.float32)
        recent = self._events[-EVENT_HISTORY_LEN:]
        start = EVENT_HISTORY_LEN - len(recent)
        for row, event in enumerate(recent, start=start):
            table[row, EVENT_TYPE_INDEX[event.event_type]] = 1.0
            table[row, 6] = np.float32(min(event.tick_seconds / 600.0, 1.0))
            table[row, 7] = np.float32(max(-1.0, min(event.magnitude / 10.0, 1.0)))
        table.setflags(write=False)
        return table

    def snapshot(self) -> EstimatorSnapshot:
        probs = self.hand_belief()
        out = self.out_of_cycle()
        history = self.event_history()
        # Copies stay mutable for the caller; the stored buffers are immutable.
        return EstimatorSnapshot(
            tick_seconds=self._tick,
            opp_elixir_lo=float(self._elixir_lo),
            opp_elixir_hi=float(self._elixir_hi),
            opp_hand_probs=np.array(probs, copy=True),
            opp_out_of_cycle=np.array(out, copy=True),
            event_history=np.array(history, copy=True),
        )

    def build_observation(
        self,
        observation_v1,
        public_entity_rows: np.ndarray | None = None,
        *,
        hand_tokens: np.ndarray | None = None,
    ) -> PolicyObservationV3:
        """Construct the actor input from the V1 boundary plus bookkeeping."""

        snapshot = self.snapshot()
        return PolicyObservationV3.from_v2(
            observation_v1,
            public_entity_rows,
            hand_tokens=hand_tokens,
            opp_hand_probs=snapshot.opp_hand_probs,
            opp_out_of_cycle=snapshot.opp_out_of_cycle,
            opp_elixir_interval=(snapshot.opp_elixir_lo, snapshot.opp_elixir_hi),
            event_history=snapshot.event_history,
        )


__all__ = [
    "ELIXIR_REGEN_PER_SECOND",
    "MAX_ELIXIR",
    "EstimatorSnapshot",
    "PublicEvent",
    "PublicStateEstimator",
    "belief_index",
]
