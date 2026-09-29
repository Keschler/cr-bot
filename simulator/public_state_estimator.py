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

from dataclasses import dataclass
from collections import deque
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
from .soa import is_visibility_concealed
from .soa import is_visibility_concealed


PUBLIC_ESTIMATOR_VERSION: Final = "public-cycle-history-2"
MAX_ELIXIR: Final = 10.0
ELIXIR_REGEN_PER_SECOND: Final = 1.0 / 2.8
EVENT_TYPE_INDEX: Final = {name: index for index, name in enumerate(EVENT_TYPES)}


def belief_index(card_key: str) -> int:
    """Map a public card key to a deterministic belief-vocabulary index."""

    if not isinstance(card_key, str) or not card_key:
        raise ValueError("card_key must be a non-empty string")
    from cr_bot.domain.card_metadata import CARD_METADATA
    from .observation import policy_card_name
    canonical = policy_card_name(card_key) or card_key
    metadata = CARD_METADATA.get(canonical)
    if metadata is None or not 0 <= int(metadata["id"]) < BELIEF_CARD_COUNT:
        raise ValueError(f"card is outside the public belief vocabulary: {card_key!r}")
    return int(metadata["id"])



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
        self._events: deque[PublicEvent] = deque(maxlen=EVENT_HISTORY_LEN)
        self._seen: dict[int, int] = {}
        self._elixir_lo = 0.0
        self._elixir_hi = MAX_ELIXIR
        self._tick = 0.0
        self._last_play: dict[int, int] = {}
        self._plays = 0
        self._memory = None
        self._event_source = None
        self._event_count = 0
        self._mutation_revision = None
        self._last_event = None
        self._synced_tick = 0.0
        self._cycle_supported = True

    def reset(self) -> None:
        self._events.clear()
        self._seen.clear()
        self._elixir_lo = 0.0
        self._elixir_hi = MAX_ELIXIR
        self._tick = 0.0
        self._last_play.clear()
        self._plays = 0
        self._memory = None
        self._event_source = None
        self._event_count = 0
        self._mutation_revision = None
        self._last_event = None
        self._synced_tick = 0.0
        self._cycle_supported = True

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
        self._plays += 1
        self._last_play[index] = self._plays
        cost = float(elixir_cost)
        self._elixir_hi = max(0.0, self._elixir_hi - cost)
        self._elixir_lo = max(0.0, min(self._elixir_lo - cost, self._elixir_hi))
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

    def advance_to(self, tick_seconds: float) -> None:
        """Advance public time bookkeeping when no event was observed.

        Visual inference often receives several frames between card plays.
        Keeping this operation public lets those callers update the elixir
        interval without reaching into the estimator's private state.
        """

        try:
            tick = float(tick_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("tick_seconds must be a finite non-negative number") from error
        if not np.isfinite(tick) or tick < 0.0:
            raise ValueError("tick_seconds must be a finite non-negative number")
        if tick < self._tick:
            raise ValueError("time must not move backwards")
        self._advance_to(tick)

    def out_of_cycle(self) -> np.ndarray:
        """Known unavailable cards under the standard four-other-play rotation.

        Unseen cards remain possible until the whole deck is publicly known.
        Champion deployments disable the four-play inference conservatively:
        this adapter does not yet model reduced champion cycles.
        """
        mask = np.zeros((BELIEF_CARD_COUNT,), dtype=bool)
        if len(self._seen) == self._deck_size:
            mask[:] = True
            for index in self._seen:
                mask[index] = False
        if self._cycle_supported:
            for index, last in self._last_play.items():
                if self._plays - last < 4:
                    mask[index] = True
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
        recent = list(self._events)
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

    def sync_from_env(self, env, *, viewer: int = 0) -> EstimatorSnapshot:
        """Consume public events once, rebuilding on reset or restored history.

        No opponent hand, draw pile, or authoritative elixir is read. Existing
        ObservationMemory reconstructs phase-aware elixir from public rules,
        elapsed time and observed spending, including Mirror's effective cost.
        Tower owner/type are public board facts used to classify damage events.
        """
        from .observation import ObservationMemory
        if type(viewer) is not int or viewer not in (0, 1):
            raise ValueError("viewer must be 0 or 1")
        state, rules = env.state, env.engine.ruleset
        if state is None:
            raise ValueError("cannot synchronize an environment without state")
        events = state.events
        elapsed = state.elapsed_us / 1_000_000.0
        rewrite = (
            self._event_source is not events or len(events) < self._event_count
            or elapsed < self._synced_tick
            or (self._event_count > 0 and events[self._event_count - 1] != self._last_event)
            or (len(events) == self._event_count and self._mutation_revision is not None
                and getattr(events, "mutation_revision", None) != self._mutation_revision)
            or (self._memory is not None and self._memory.viewer != viewer)
        )
        if rewrite:
            self.reset()
        if self._memory is None:
            self._memory = ObservationMemory(viewer)
        self._memory.update(state, rules)
        # A complete event-inclusive frontier snapshot contains the prefix;
        # processing from cursor zero reconstructs exactly the same history.
        towers = {uid: entity.owner for uid, entity in state.entities.items() if entity.kind == "tower"}
        for event in events[self._event_count:]:
            time = max(self._tick, min(elapsed, (event.tick + 1) * rules.tick_us / 1_000_000.0))
            owner = event.get("player", event.get("owner"))
            if event.kind == "card_played" and owner in (0, 1):
                key = event.get("card_id", event.get("card"))
                card = rules.card(key)
                cost = event.get("cost_milli", card.elixir_milli) / 1000.0
                if owner == 1 - viewer:
                    # Champion cycling needs additional lifecycle inference;
                    # do not manufacture false four-play certainty for it.
                    if key in {"archer-queen", "golden-knight", "skeleton-king", "mighty-miner", "monk", "little-prince", "goblinstein", "boss-bandit"}:
                        self._cycle_supported = False
                    self.observe_opponent_spend(tick_seconds=time, card_key=key, elixir_cost=cost)
                else:
                    self.observe(PublicEvent("own_deploy", time, cost))
                    self.observe(PublicEvent("action_taken", time, cost))
            elif event.kind == "damage_applied" and event.get("target_uid") in towers:
                owner = towers[event.get("target_uid")]
                name = "own_tower_damaged" if owner == viewer else "enemy_tower_damaged"
                self.observe(PublicEvent(name, time, float(event.get("damage", 0)) / 1000.0))
            elif event.kind == "entity_died":
                # A body that was hidden when it died was never observed, so
                # its death is not part of the public stream. Emitting it told
                # the actor that something died at a time when the board showed
                # nothing, which is a hidden-information leak even though the
                # event carries no card identity. A concealed body is
                # re-checked here rather than at the time of the event because
                # the retention rules mean the flags can still be read.
                died = state.entities.get(event.get("uid"))
                if died is not None and not is_visibility_concealed(died):
                    self.observe(PublicEvent("unit_died", time, 1.0))
        self._tick = elapsed
        estimate = self._memory.opponent_elixir_est
        self._elixir_lo = self._elixir_hi = estimate
        self._event_source = events
        self._event_count = len(events)
        self._mutation_revision = getattr(events, "mutation_revision", None)
        self._last_event = events[-1] if events else None
        self._synced_tick = elapsed
        return self.snapshot()

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
    "PUBLIC_ESTIMATOR_VERSION",
    "ELIXIR_REGEN_PER_SECOND",
    "MAX_ELIXIR",
    "EstimatorSnapshot",
    "PublicEvent",
    "PublicStateEstimator",
    "belief_index",
]
