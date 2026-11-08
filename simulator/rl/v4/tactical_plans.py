"""Opt-in, public-only short defense plans at the native 250 ms cadence.

This is a candidate action source, not a replacement for the policy or the
cycle teacher. ``choose`` starts a small deterministic plan; a controller
retains its intent across observations and releases control on completion or
interruption. A caller must execute each emitted target and acknowledge it
before the controller can advance. Terminal decisions have ``target=None``
and should return control to the caller's ordinary action source.

Only hand availability, legal masks and visible entity geometry/metadata are
used. No engine, enemy hand, draw queue, target lock or combat timer enters
this module. Public entity tokens have no stable tracking ID, so association
is deliberately conservative and ambiguous matches interrupt the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Mapping

import numpy as np

from .hand_features import V4_LEARNER_DECK
from .simulator_teacher import TeacherTarget, cycle_teacher_label, nearest_legal_cell


DECISION_INTERVAL_US = 250_000
TACTICAL_PLANS_VERSION = "public-defense-plans-v1"
_ROWS, _COLS = 32, 18
_HAND = {row[1]: (row[0], row[2]) for row in V4_LEARNER_DECK}
_URGENT_Y = float(np.float32(.70))


def _snapshot(value: np.ndarray, shape: tuple[int, ...], name: str, *, boolean: bool = False) -> np.ndarray:
    array = np.array(value, dtype=bool if boolean else np.float32, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not boolean and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PlanPublicState:
    """One copied public observation, indexed in 250 ms decisions.

    ``decision_step`` is a caller-owned observation counter, not a physics
    tick or a match-clock value. It must advance by one during a live plan.
    The fixed grid and entity/hand namespaces are the existing V4 contracts.
    """

    decision_step: int
    hand_tokens: np.ndarray
    entity_tokens: np.ndarray
    entity_mask: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    own_elixir: float
    decision_interval_us: int = DECISION_INTERVAL_US

    def __post_init__(self) -> None:
        if type(self.decision_step) is not int or self.decision_step < 0:
            raise ValueError("decision_step must be a nonnegative integer")
        if self.decision_interval_us != DECISION_INTERVAL_US:
            raise ValueError("short defense plans require native 250000us observations")
        if not np.isfinite(self.own_elixir) or not 0 <= self.own_elixir <= 10:
            raise ValueError("own_elixir must be finite and between zero and ten")
        for name, shape, boolean in (
            ("hand_tokens", (4, 16), False),
            ("entity_tokens", (128, 32), False),
            ("entity_mask", (128,), True),
            ("legal_play", (4, _ROWS, _COLS), True),
        ):
            object.__setattr__(self, name, _snapshot(getattr(self, name), shape, name, boolean=boolean))
        object.__setattr__(self, "legal_wait", bool(self.legal_wait))

    @classmethod
    def from_inputs(
        cls, inputs: Mapping[str, object], step: int, own_elixir: float,
        *, decision_interval_us: int = DECISION_INTERVAL_US,
    ) -> PlanPublicState:
        """Select only the public arrays already returned by V4 observation."""

        return cls(
            decision_step=step,
            hand_tokens=inputs["hand_tokens"],
            entity_tokens=inputs["entities"],
            entity_mask=inputs["entity_mask"],
            legal_play=inputs["legal_play"],
            legal_wait=inputs["legal_wait"],
            own_elixir=own_elixir,
            decision_interval_us=decision_interval_us,
        )


class PlanRole(str, Enum):
    PULL = "pull"
    INTERCEPT = "intercept"
    SUPPORT = "support"
    STALL = "stall"
    CYCLE = "cycle"


class PlanStatus(str, Enum):
    PLAY = "play"
    WAIT = "wait"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class PublicThreat:
    card_id: int
    name: str
    x: float
    y: float
    hp: float
    is_air: bool
    targets_buildings_only: bool
    cost: int


@dataclass(frozen=True, slots=True)
class DefensePlanStep:
    role: PlanRole
    cards: tuple[str, ...]
    min_delay_steps: int = 0
    max_delay_steps: int = 0
    reserve_elixir: float = 0.0


@dataclass(frozen=True, slots=True)
class DefensePlanProposal:
    plan_id: str
    created_step: int
    threat: PublicThreat
    steps: tuple[DefensePlanStep, ...]
    max_decisions: int = 17


@dataclass(frozen=True, slots=True)
class PlanDecision:
    target: TeacherTarget | None
    status: PlanStatus
    reason: str
    plan_id: str
    step_index: int
    decision_step: int

    @property
    def active(self) -> bool:
        return self.status in (PlanStatus.PLAY, PlanStatus.WAIT)


@dataclass(frozen=True, slots=True)
class PlanCounters:
    accepted_plays: int = 0
    waits: int = 0
    completions: int = 0
    interruptions: int = 0
    urgency_interruptions: int = 0
    rejected_plays: int = 0


@lru_cache(maxsize=1)
def _cards() -> tuple[int, dict[int, tuple[str, Mapping[str, object]]]]:
    from cr_bot.domain.card_metadata import CARD_METADATA

    return max(int(row["id"]) for row in CARD_METADATA.values()), {
        int(row["id"]): (name, row) for name, row in CARD_METADATA.items()
    }


def _bodies(state: PlanPublicState, side: int) -> tuple[PublicThreat, ...]:
    maximum, cards = _cards()
    bodies = []
    for row, masked in zip(state.entity_tokens, state.entity_mask, strict=True):
        if not masked or int(row[1] > .5) != side:
            continue
        if row[4] <= 0 or row[7] > .5 or row[8] > .5 or row[9] < .5 or row[10] < .5:
            continue
        if not (0 <= row[2] <= 1 and 0 <= row[3] <= 1):
            continue
        card_id = int(round(float(row[0]) * maximum))
        card = cards.get(card_id)
        if card is None:
            continue
        name, metadata = card
        # Card IDs and flags use the public catalog ABI, which is separate
        # from the learner-hand vocabulary. Conflicting/invalid flags are
        # not evidence for an expensive defense.
        if metadata.get("kind") not in ("troop", "building"):
            continue
        if bool(row[5] > .5) != bool(metadata.get("is_air")):
            continue
        if bool(row[6] > .5) != (metadata.get("kind") == "building"):
            continue
        bodies.append(PublicThreat(
            card_id, name, float(row[2]), float(row[3]), float(row[4]),
            bool(row[5] > .5), bool(metadata.get("targets_buildings_only")),
            int(metadata.get("elixir_cost") or 0),
        ))
    return tuple(bodies)


def _distance(a: PublicThreat, b: PublicThreat) -> float:
    return float(np.hypot((a.x - b.x) * _COLS, (a.y - b.y) * _ROWS))


def _friendly(state: PlanPublicState, threat: PublicThreat, name: str, *, reach: float = 6.0) -> bool:
    return any(
        body.name == name and body.hp >= float(np.float32(.35))
        and abs(body.x - threat.x) * _COLS <= 5
        and body.y >= threat.y - 1 / _ROWS
        and _distance(body, threat) <= reach
        for body in _bodies(state, 0)
    )


def _anchor(role: PlanRole, threat: PublicThreat) -> tuple[int, int]:
    row = min(_ROWS - 1, int(threat.y * _ROWS))
    col = min(_COLS - 1, int(threat.x * _COLS))
    if role == PlanRole.PULL:
        return (20, 7 if col < 9 else 10) if row < 21 else (min(row + 1, 23), col + (1 if col < 9 else -1))
    if role in (PlanRole.INTERCEPT, PlanRole.SUPPORT):
        # Separate the ranged unit from the central pull and from the tower.
        return min(max(row + 3, 18), 26), min(col + 3, 7) if col < 9 else max(col - 3, 10)
    if role == PlanRole.STALL:
        return min(max(row + 1, 17), 26), col
    return 29, 4 if col >= 9 else 13


def _hand_names(state: PlanPublicState) -> set[str]:
    return {
        identity[0] for row in state.hand_tokens
        if (identity := _HAND.get(int(round(float(row[0]) * 127)))) is not None
    }


def _defense_zone(body: PublicThreat) -> bool:
    return body.y >= (13 / _ROWS if body.is_air else float(np.float32(.45)))


def _needs_support(threat: PublicThreat, enemies: tuple[PublicThreat, ...]) -> bool:
    return threat.is_air or threat.cost >= 5 or sum(_distance(body, threat) <= 5 for body in enemies) >= 2


def _play(state: PlanPublicState, step: DefensePlanStep, threat: PublicThreat) -> TeacherTarget | None:
    for name in step.cards:
        for slot, row in enumerate(state.hand_tokens):
            identity = _HAND.get(int(round(float(row[0]) * 127)))
            if identity is None or identity[0] != name:
                continue
            # Check actual public elixir as well as both hand availability
            # flags: a stale affordable bit must never authorize spending.
            if row[3] <= .5 or row[4] <= .5 or state.own_elixir + 1e-6 < identity[1] + step.reserve_elixir:
                continue
            anchor = _anchor(step.role, threat)
            if name == "fireball":
                anchor = min(31, int(threat.y * 32)), min(17, int(threat.x * 18))
            cell = nearest_legal_cell(*anchor, state.legal_play[slot], max_radius=2)
            if cell is not None:
                return TeacherTarget(1, slot, *cell, 0)
    return None


def propose_defense_plans(state: PlanPublicState) -> tuple[DefensePlanProposal, ...]:
    """Return a bounded public portfolio; proposal order is deterministic.

    One action per observation remains mandatory. Optional followups are
    checked again against a fresh public hand and elixir; no draw is assumed.
    """

    enemies = _bodies(state, 1)
    if any(body.y >= _URGENT_Y for body in enemies):
        return ()
    candidates = sorted(
        (body for body in enemies if _defense_zone(body)),
        key=lambda body: (-body.y, -int(body.is_air), body.card_id, body.x),
    )
    proposals = []
    for threat in candidates:
        # Tokens cannot distinguish two coincident members of one card.
        if sum(body.card_id == threat.card_id and _distance(body, threat) <= 2.5 for body in enemies) != 1:
            continue
        if _friendly(state, threat, "musketeer"):
            continue
        support_needed = _needs_support(threat, enemies)
        # A pull is useful alone against an isolated Hog/Ram. An expensive
        # or aerial attacker needs a known ranged followup in the hand.
        if threat.targets_buildings_only and threat.y >= float(np.float32(.48)) and not _friendly(state, threat, "cannon") and (not support_needed or "musketeer" in _hand_names(state)):
            steps = (
                DefensePlanStep(PlanRole.PULL, ("cannon",), reserve_elixir=4 if support_needed else 0),
                DefensePlanStep(PlanRole.SUPPORT, ("musketeer",), 4, 12),
            )
            if _play(state, steps[0], threat) is not None and (not support_needed or _play(state, steps[1], threat) is not None):
                proposals.append(DefensePlanProposal("air-pull-intercept" if threat.is_air else "cannon-musketeer", state.decision_step, threat, steps))
        if threat.is_air:
            steps = (
                DefensePlanStep(PlanRole.INTERCEPT, ("musketeer",)),
                DefensePlanStep(PlanRole.STALL, ("ice-spirit",), 4, 10),
            )
            if _play(state, steps[0], threat) is not None:
                proposals.append(DefensePlanProposal("air-intercept-stall", state.decision_step, threat, steps))
        if proposals:
            break
    return tuple(proposals)


def choose(state: PlanPublicState) -> DefensePlanProposal | None:
    """Choose the first deterministic proposal, or defer to the caller."""

    return next(iter(propose_defense_plans(state)), None)


class DefensePlanController:
    """Retain one plan with explicit execution acknowledgement and expiry."""

    def __init__(self, proposal: DefensePlanProposal) -> None:
        if (not proposal.steps or type(proposal.max_decisions) is not int or proposal.max_decisions <= 0
                or type(proposal.created_step) is not int or proposal.created_step < 0):
            raise ValueError("a plan must have steps and a positive lifetime")
        for step in proposal.steps:
            if (not step.cards or type(step.min_delay_steps) is not int or type(step.max_delay_steps) is not int
                    or not 0 <= step.min_delay_steps <= step.max_delay_steps
                    or not np.isfinite(step.reserve_elixir) or not 0 <= step.reserve_elixir <= 10):
                raise ValueError("invalid plan step")
        self.proposal = proposal
        self._threat = proposal.threat
        self._index = 0
        self._last_play_step = proposal.created_step
        self._last_state_step: int | None = None
        self._last_decision: PlanDecision | None = None
        self._pending: PlanDecision | None = None
        self._terminal: PlanDecision | None = None
        self._previous_enemies: tuple[PublicThreat, ...] | None = None
        self._counts = {name: 0 for name in PlanCounters.__dataclass_fields__}

    @property
    def counters(self) -> PlanCounters:
        return PlanCounters(**self._counts)

    @property
    def active(self) -> bool:
        return self._terminal is None

    def _decision(self, state: PlanPublicState, status: PlanStatus, reason: str, target: TeacherTarget | None = None) -> PlanDecision:
        decision = PlanDecision(target, status, reason, self.proposal.plan_id, self._index, state.decision_step)
        self._last_decision = decision
        if status in (PlanStatus.COMPLETE, PlanStatus.INTERRUPTED):
            self._terminal = decision
            self._counts["completions" if status == PlanStatus.COMPLETE else "interruptions"] += 1
            if reason.startswith("urgent"):
                self._counts["urgency_interruptions"] += 1
        elif status == PlanStatus.WAIT:
            self._counts["waits"] += 1
        else:
            self._pending = decision
        return decision

    def _wait(self, state: PlanPublicState, reason: str) -> PlanDecision:
        if self._threat.y >= _URGENT_Y:
            return self._decision(state, PlanStatus.INTERRUPTED, "urgent_during_wait")
        if not state.legal_wait:
            return self._decision(state, PlanStatus.INTERRUPTED, "wait_illegal")
        return self._decision(state, PlanStatus.WAIT, reason, TeacherTarget(0, 0, 0, 0, 0))

    def step(self, state: PlanPublicState) -> PlanDecision:
        """Re-evaluate one fresh native observation; repeated calls are inert."""

        if self._terminal is not None:
            return self._terminal
        if self._last_state_step == state.decision_step:
            assert self._last_decision is not None
            return self._last_decision
        if self._pending is not None:
            return self._decision(state, PlanStatus.INTERRUPTED, "unacknowledged_play")
        expected = self.proposal.created_step if self._last_state_step is None else self._last_state_step + 1
        if state.decision_step != expected:
            return self._decision(state, PlanStatus.INTERRUPTED, "observation_gap")
        self._last_state_step = state.decision_step
        if state.decision_step - self.proposal.created_step >= self.proposal.max_decisions:
            return self._decision(state, PlanStatus.INTERRUPTED, "plan_expired")
        enemies = _bodies(state, 1)
        if any(body.y >= _URGENT_Y for body in enemies):
            return self._decision(state, PlanStatus.INTERRUPTED, "urgent_threat")
        matches = [body for body in enemies if body.card_id == self._threat.card_id and body.is_air == self._threat.is_air and _distance(body, self._threat) <= 2.5]
        if not matches:
            return self._decision(state, PlanStatus.COMPLETE, "threat_gone")
        if len(matches) != 1:
            return self._decision(state, PlanStatus.INTERRUPTED, "ambiguous_threat")
        threat = matches[0]
        self._threat = threat
        if self._previous_enemies is not None:
            for body in enemies:
                if not _defense_zone(body):
                    continue
                associations = [prior for prior in self._previous_enemies if _defense_zone(prior) and prior.card_id == body.card_id and prior.is_air == body.is_air and _distance(body, prior) <= 2.5]
                if not associations:
                    return self._decision(state, PlanStatus.INTERRUPTED, "new_threat")
                if len(associations) != 1:
                    return self._decision(state, PlanStatus.INTERRUPTED, "ambiguous_threat")
        self._previous_enemies = enemies
        step = self.proposal.steps[self._index]
        elapsed = state.decision_step - self._last_play_step
        if elapsed > step.max_delay_steps:
            return self._decision(state, PlanStatus.INTERRUPTED, "step_expired")
        if self._index:
            nearby = sum(_distance(body, threat) <= 5 for body in enemies)
            if threat.hp <= .2 or _friendly(state, threat, "musketeer") and step.role != PlanRole.STALL:
                return self._decision(state, PlanStatus.COMPLETE, "support_not_needed")
            if step.role == PlanRole.SUPPORT and not _needs_support(threat, enemies):
                return self._decision(state, PlanStatus.COMPLETE, "support_not_needed")
            if elapsed < step.min_delay_steps:
                return self._wait(state, "minimum_delay")
            previous_role = self.proposal.steps[self._index - 1].role
            if previous_role == PlanRole.PULL and not _friendly(state, threat, "cannon", reach=9):
                return self._wait(state, "awaiting_visible_setup")
            if step.role == PlanRole.STALL:
                if not _friendly(state, threat, "musketeer"):
                    return self._wait(state, "awaiting_visible_setup")
                if threat.y < float(np.float32(.60)) and nearby < 2:
                    return self._decision(state, PlanStatus.COMPLETE, "stall_not_needed")
        target = _play(state, step, threat)
        if target is None:
            return self._wait(state, "awaiting_legal_card")
        return self._decision(state, PlanStatus.PLAY, "planned_play", target)

    def acknowledge(self, decision: PlanDecision | None = None, *, accepted: bool = True) -> None:
        """Confirm execution, never a hypothetical queued play.

        Waiting decisions need no acknowledgement. A rejected action releases
        control immediately; the controller never advances on a failed play.
        """

        decision = self._last_decision if decision is None else decision
        if decision is None or decision is not self._last_decision or not decision.active or self._terminal is not None:
            raise ValueError("acknowledgement must refer to the current active decision")
        if not accepted:
            self._counts["rejected_plays"] += int(decision.status == PlanStatus.PLAY)
            self._counts["interruptions"] += 1
            self._terminal = PlanDecision(None, PlanStatus.INTERRUPTED, "execution_rejected", decision.plan_id, self._index, decision.decision_step)
            self._pending = None
            return
        if decision.status == PlanStatus.WAIT:
            return
        if decision is not self._pending:
            raise ValueError("play was already acknowledged")
        self._counts["accepted_plays"] += 1
        self._last_play_step = decision.decision_step
        self._index += 1
        self._pending = None
        if self._index == len(self.proposal.steps):
            self._counts["completions"] += 1
            self._terminal = PlanDecision(None, PlanStatus.COMPLETE, "plan_finished", decision.plan_id, self._index, decision.decision_step)


class PlanTeacherLabeler:
    """Episode-local plan labels with a cycle-teacher fallback on every row.

    Call :meth:`label` at every native observation, then :meth:`acknowledge`
    with the action actually executed. Merely generating a label never
    commits its hypothetical setup: an alternative or rejected action ends
    that intent. For independent counterfactual labels, use a fresh instance
    per row. This object owns no simulator or privileged state.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._controller: DefensePlanController | None = None
        self._counts = {name: 0 for name in PlanCounters.__dataclass_fields__}
        self._last_step: int | None = None
        self._emitted: TeacherTarget | None = None
        self._waiting_ack = False
        self.last_decision: PlanDecision | None = None
        self.last_source = "cycle"

    @property
    def counters(self) -> PlanCounters:
        counts = dict(self._counts)
        if self._controller is not None:
            live = self._controller.counters
            for name in counts:
                counts[name] += getattr(live, name)
        return PlanCounters(**counts)

    def _retire(self) -> None:
        if self._controller is not None:
            counts = self._controller.counters
            for name in self._counts:
                self._counts[name] += getattr(counts, name)
            self._controller = None

    def label(
        self, *, decision_step: int, hand_tokens: np.ndarray,
        entity_tokens: np.ndarray, entity_mask: np.ndarray,
        legal_play: np.ndarray, legal_wait: bool, own_elixir: float,
        decision_interval_us: int = DECISION_INTERVAL_US,
    ) -> TeacherTarget:
        state = PlanPublicState(
            decision_step, hand_tokens, entity_tokens, entity_mask, legal_play,
            legal_wait, own_elixir, decision_interval_us,
        )
        if self._last_step == decision_step:
            assert self._emitted is not None
            return self._emitted
        missing_ack = self._waiting_ack
        if missing_ack and self._controller is not None and self._controller.active:
            assert self.last_decision is not None
            self._controller.acknowledge(self.last_decision, accepted=False)
        if self._controller is not None and not self._controller.active:
            self._retire()
        self.last_decision = None
        self.last_source = "cycle"
        if self._controller is None and not missing_ack:
            proposal = choose(state)
            if proposal is not None:
                self._controller = DefensePlanController(proposal)
        target = None
        if self._controller is not None:
            self.last_decision = self._controller.step(state)
            target = self.last_decision.target
            if self.last_decision.active:
                self.last_source = "plan"
            else:
                self._retire()
        if target is None:
            target = cycle_teacher_label(
                hand_tokens=state.hand_tokens, entity_tokens=state.entity_tokens,
                entity_mask=state.entity_mask, legal_play=state.legal_play,
                legal_wait=state.legal_wait, own_elixir=state.own_elixir,
            )
        self._last_step = decision_step
        self._emitted = target
        self._waiting_ack = True
        return target

    def acknowledge(self, executed_target: TeacherTarget | None, *, accepted: bool = True) -> None:
        """Record execution of the current row, including fallback decisions."""

        if not self._waiting_ack:
            raise ValueError("no current unacknowledged label")
        if self._controller is not None and self._controller.active:
            assert self.last_decision is not None
            self._controller.acknowledge(
                self.last_decision,
                accepted=bool(accepted and executed_target == self._emitted),
            )
        self._waiting_ack = False


__all__ = [
    "DECISION_INTERVAL_US", "TACTICAL_PLANS_VERSION", "PlanPublicState",
    "PlanRole", "PlanStatus", "PublicThreat", "DefensePlanStep",
    "DefensePlanProposal", "PlanDecision", "PlanCounters",
    "propose_defense_plans", "choose", "DefensePlanController", "PlanTeacherLabeler",
]
