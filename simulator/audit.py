"""Automated determinism and legal-action fuzz audits.

The audit path is deliberately read-only: it executes in-memory battles and
returns JSON-safe data.  It never promotes a failure into a regression oracle
or writes generated scenarios to the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .actions import PlayCardAction, SimAction, WaitAction, action_to_dict
from .engine import ENGINE_VERSION, BattleEngine
from .events import SimEvent
from .fixed import DeterministicRng, PERMILLE
from .state import BattleState


AUDIT_SCHEMA_VERSION = 1
MAX_SOAK_TICK_BUDGET = 10_000_000
_ACTION_SEED_SALT = 0xA5B35705D17EF10D
_UINT64_MASK = (1 << 64) - 1

EngineFactory = Callable[[], BattleEngine]
ControllerFactory = Callable[[int], "LegalFuzzController"]


class DeterminismAuditError(RuntimeError):
    """Raised when replicas, actions, or invariant checks diverge."""

    def __init__(
        self,
        *,
        seed: int,
        tick: int,
        detail: str,
        first_hash: str | None = None,
        second_hash: str | None = None,
    ) -> None:
        self.seed = seed
        self.tick = tick
        self.detail = detail
        self.first_hash = first_hash
        self.second_hash = second_hash
        hashes = ""
        if first_hash is not None or second_hash is not None:
            hashes = f" (replica_a={first_hash}, replica_b={second_hash})"
        super().__init__(f"determinism audit failed at seed={seed}, tick={tick}: {detail}{hashes}")


class LegalFuzzController:
    """Generate reproducible waits and currently legal card placements.

    Each player owns an independent RNG stream, so action selection does not
    depend on the order in which callers query the two players.  A periodic
    forced wait guarantees that long audits exercise the explicit Wait action;
    all other choices are randomized from affordable slots and legal cells.
    """

    def __init__(
        self,
        seed: int,
        *,
        random_wait_permille: int = 150,
        force_wait_every: int | None = 7,
    ) -> None:
        if not (0 <= random_wait_permille <= PERMILLE):
            raise ValueError("random_wait_permille must be between 0 and 1000")
        if force_wait_every is not None and force_wait_every <= 0:
            raise ValueError("force_wait_every must be positive or None")
        seed_rng = DeterministicRng(seed & _UINT64_MASK)
        self._rngs = (DeterministicRng(seed_rng.next_u64()), DeterministicRng(seed_rng.next_u64()))
        self._decision_counts = [0, 0]
        self.random_wait_permille = random_wait_permille
        self.force_wait_every = force_wait_every

    @property
    def decision_counts(self) -> tuple[int, int]:
        return self._decision_counts[0], self._decision_counts[1]

    def choose_action(self, engine: BattleEngine, state: BattleState, player: int) -> SimAction:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        ordinal = self._decision_counts[player]
        self._decision_counts[player] += 1
        rng = self._rngs[player]

        random_wait = rng.randbelow(PERMILLE) < self.random_wait_permille
        forced_wait = self.force_wait_every is not None and ordinal % self.force_wait_every == 0
        if random_wait or forced_wait:
            return WaitAction(player)

        legal_by_slot: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        player_state = state.players[player]
        for slot, card_id in enumerate(player_state.hand):
            card = engine.ruleset.card(card_id)
            if card.elixir_milli > player_state.elixir_milli:
                continue
            cells = engine.legal_cells(state, player, card_id)
            if cells:
                legal_by_slot.append((slot, cells))
        if not legal_by_slot:
            return WaitAction(player)

        slot, cells = legal_by_slot[rng.randbelow(len(legal_by_slot))]
        action = PlayCardAction(player, slot, cells[rng.randbelow(len(cells))])
        reason = engine.validate_action(state, action)
        if reason is not None:
            raise RuntimeError(f"legal-action generator produced an invalid action: {reason}")
        return action

    def choose_actions(self, engine: BattleEngine, state: BattleState) -> tuple[SimAction, SimAction]:
        return self.choose_action(engine, state, 0), self.choose_action(engine, state, 1)


@dataclass(frozen=True, slots=True)
class AuditSeedResult:
    seed: int
    ticks: int
    actions: int
    card_plays: int
    waits: int
    events: int
    completed: bool
    winner: int | None
    terminal_reason: str | None
    final_hash: str
    event_log_hash: str
    replay_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "ticks": self.ticks,
            "actions": self.actions,
            "card_plays": self.card_plays,
            "waits": self.waits,
            "events": self.events,
            "completed": self.completed,
            "winner": self.winner,
            "terminal_reason": self.terminal_reason,
            "final_hash": self.final_hash,
            "event_log_hash": self.event_log_hash,
            "replay_hash": self.replay_hash,
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    mode: str
    engine_version: str
    ruleset_id: str
    ruleset_hash: str
    level: int
    tick_us: int
    decision_interval_ticks: int
    seed_start: int
    seed_count: int
    max_ticks_per_seed: int
    tick_budget: int
    runs: tuple[AuditSeedResult, ...]

    @property
    def total_ticks(self) -> int:
        return sum(run.ticks for run in self.runs)

    @property
    def total_actions(self) -> int:
        return sum(run.actions for run in self.runs)

    @property
    def total_card_plays(self) -> int:
        return sum(run.card_plays for run in self.runs)

    @property
    def total_waits(self) -> int:
        return sum(run.waits for run in self.runs)

    @property
    def total_events(self) -> int:
        return sum(run.events for run in self.runs)

    @property
    def completions(self) -> int:
        return sum(run.completed for run in self.runs)

    def to_dict(self) -> dict[str, object]:
        """Return stable, timestamp-free data suitable for canonical JSON."""

        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "kind": "simulator_determinism_audit",
            "mode": self.mode,
            "engine_version": self.engine_version,
            "ruleset": {
                "id": self.ruleset_id,
                "hash": self.ruleset_hash,
                "level": self.level,
                "tick_us": self.tick_us,
            },
            "config": {
                "seed_start": self.seed_start,
                "seed_count": self.seed_count,
                "decision_interval_ticks": self.decision_interval_ticks,
                "max_ticks_per_seed": self.max_ticks_per_seed,
                "tick_budget": self.tick_budget,
            },
            "totals": {
                "ticks": self.total_ticks,
                "actions": self.total_actions,
                "card_plays": self.total_card_plays,
                "waits": self.total_waits,
                "events": self.total_events,
                "completions": self.completions,
            },
            "final_hashes": [
                {
                    "seed": run.seed,
                    "state_hash": run.final_hash,
                    "event_log_hash": run.event_log_hash,
                    "replay_hash": run.replay_hash,
                }
                for run in self.runs
            ],
            "runs": [run.to_dict() for run in self.runs],
        }


def run_determinism_audit(
    *,
    seed_count: int = 4,
    seed_start: int = 0,
    max_ticks_per_seed: int | None = None,
    decision_interval_ticks: int | None = None,
    engine_factory: EngineFactory | None = None,
    controller_factory: ControllerFactory | None = None,
) -> AuditReport:
    """Run independent replicas in lockstep for a contiguous range of seeds."""

    first_engine, second_engine = _make_engines(engine_factory)
    if max_ticks_per_seed is None:
        max_ticks_per_seed = _complete_match_tick_limit(first_engine)
    cadence = (
        first_engine.decision_interval_ticks
        if decision_interval_ticks is None
        else decision_interval_ticks
    )
    _validate_run_bounds(seed_count, max_ticks_per_seed, cadence)
    return _run_audit(
        mode="determinism",
        first_engine=first_engine,
        second_engine=second_engine,
        seed_count=seed_count,
        seed_start=seed_start,
        max_ticks_per_seed=max_ticks_per_seed,
        decision_interval_ticks=cadence,
        tick_budget=seed_count * max_ticks_per_seed,
        controller_factory=controller_factory,
    )


def run_soak_audit(
    *,
    seed_count: int = 16,
    seed_start: int = 0,
    tick_budget: int = 100_000,
    max_ticks_per_seed: int | None = None,
    decision_interval_ticks: int | None = None,
    engine_factory: EngineFactory | None = None,
    controller_factory: ControllerFactory | None = None,
) -> AuditReport:
    """Run a deterministically bounded soak audit.

    The cap is based on simulated ticks rather than wall time.  This makes the
    selected seeds, actions, and final report repeatable on fast and slow hosts.
    The budget is divided evenly between seeds; unused capacity is not silently
    reassigned after an early match completion.
    """

    if not (1 <= tick_budget <= MAX_SOAK_TICK_BUDGET):
        raise ValueError(f"tick_budget must be between 1 and {MAX_SOAK_TICK_BUDGET}")
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    ticks_per_seed_from_budget = tick_budget // seed_count
    if ticks_per_seed_from_budget <= 0:
        raise ValueError("tick_budget must provide at least one tick per seed")

    first_engine, second_engine = _make_engines(engine_factory)
    requested_limit = (
        _complete_match_tick_limit(first_engine)
        if max_ticks_per_seed is None
        else max_ticks_per_seed
    )
    effective_limit = min(requested_limit, ticks_per_seed_from_budget)
    cadence = (
        first_engine.decision_interval_ticks
        if decision_interval_ticks is None
        else decision_interval_ticks
    )
    _validate_run_bounds(seed_count, effective_limit, cadence)
    return _run_audit(
        mode="soak",
        first_engine=first_engine,
        second_engine=second_engine,
        seed_count=seed_count,
        seed_start=seed_start,
        max_ticks_per_seed=effective_limit,
        decision_interval_ticks=cadence,
        tick_budget=tick_budget,
        controller_factory=controller_factory,
    )


def _run_audit(
    *,
    mode: str,
    first_engine: BattleEngine,
    second_engine: BattleEngine,
    seed_count: int,
    seed_start: int,
    max_ticks_per_seed: int,
    decision_interval_ticks: int,
    tick_budget: int,
    controller_factory: ControllerFactory | None,
) -> AuditReport:
    runs: list[AuditSeedResult] = []
    for seed in range(seed_start, seed_start + seed_count):
        action_seed = (seed ^ _ACTION_SEED_SALT) & _UINT64_MASK
        make_controller = LegalFuzzController if controller_factory is None else controller_factory
        first_controller = make_controller(action_seed)
        second_controller = make_controller(action_seed)
        if first_controller is second_controller:
            raise ValueError("controller_factory must return independent controllers")

        first_state = first_engine.new_battle(seed=seed)
        second_state = second_engine.new_battle(seed=seed)
        _validate_state(first_engine, first_state, seed=seed, replica="a")
        _validate_state(second_engine, second_state, seed=seed, replica="b")
        if first_state.events != second_state.events:
            _raise_divergence(first_state, second_state, seed, 0, "initial event stream")
        _compare_hashes(first_state, second_state, seed=seed, tick=0)

        action_count = 0
        card_play_count = 0
        wait_count = 0
        while not first_state.terminal and first_state.tick < max_ticks_per_seed:
            if second_state.terminal:
                _raise_divergence(first_state, second_state, seed, first_state.tick, "terminal state")
            actions_a: tuple[SimAction, ...] = ()
            actions_b: tuple[SimAction, ...] = ()
            if first_state.tick % decision_interval_ticks == 0:
                actions_a = first_controller.choose_actions(first_engine, first_state)
                actions_b = second_controller.choose_actions(second_engine, second_state)
                if actions_a != actions_b:
                    detail = (
                        "action generators diverged: "
                        f"{[action_to_dict(action) for action in actions_a]} != "
                        f"{[action_to_dict(action) for action in actions_b]}"
                    )
                    _raise_divergence(first_state, second_state, seed, first_state.tick, detail)
                action_count += len(actions_a)
                card_play_count += sum(isinstance(action, PlayCardAction) for action in actions_a)
                wait_count += sum(isinstance(action, WaitAction) for action in actions_a)

            events_a = _step(first_engine, first_state, actions_a, seed=seed, replica="a")
            events_b = _step(second_engine, second_state, actions_b, seed=seed, replica="b")
            if events_a != events_b:
                _raise_divergence(first_state, second_state, seed, first_state.tick, "event stream")
            expected_plays = sum(isinstance(action, PlayCardAction) for action in actions_a)
            actual_plays = sum(event.kind == "card_played" for event in events_a)
            if actual_plays != expected_plays:
                raise DeterminismAuditError(
                    seed=seed,
                    tick=first_state.tick,
                    detail=(
                        "legal-action controller action was rejected "
                        f"(expected {expected_plays} card plays, observed {actual_plays})"
                    ),
                    first_hash=first_state.state_hash(),
                    second_hash=second_state.state_hash(),
                )
            _validate_state(first_engine, first_state, seed=seed, replica="a")
            _validate_state(second_engine, second_state, seed=seed, replica="b")
            _compare_hashes(first_state, second_state, seed=seed, tick=first_state.tick)

        if second_state.terminal != first_state.terminal or second_state.tick != first_state.tick:
            _raise_divergence(first_state, second_state, seed, first_state.tick, "final state")
        final_hash = _compare_hashes(
            first_state,
            second_state,
            seed=seed,
            tick=first_state.tick,
        )
        runs.append(
            AuditSeedResult(
                seed=seed,
                ticks=first_state.tick,
                actions=action_count,
                card_plays=card_play_count,
                waits=wait_count,
                events=len(first_state.events),
                completed=first_state.terminal,
                winner=first_state.winner,
                terminal_reason=first_state.terminal_reason,
                final_hash=final_hash,
                event_log_hash=first_state.event_log_hash(),
                replay_hash=first_state.replay_hash(),
            )
        )

    ruleset = first_engine.ruleset
    return AuditReport(
        mode=mode,
        engine_version=ENGINE_VERSION,
        ruleset_id=ruleset.ruleset_id,
        ruleset_hash=ruleset.content_hash,
        level=ruleset.level,
        tick_us=ruleset.tick_us,
        decision_interval_ticks=decision_interval_ticks,
        seed_start=seed_start,
        seed_count=seed_count,
        max_ticks_per_seed=max_ticks_per_seed,
        tick_budget=tick_budget,
        runs=tuple(runs),
    )


def _make_engines(engine_factory: EngineFactory | None) -> tuple[BattleEngine, BattleEngine]:
    factory = BattleEngine if engine_factory is None else engine_factory
    first = factory()
    second = factory()
    if first is second:
        raise ValueError("engine_factory must return independent engines")
    first_ruleset = first.ruleset
    second_ruleset = second.ruleset
    if (
        first_ruleset.ruleset_id != second_ruleset.ruleset_id
        or first_ruleset.content_hash != second_ruleset.content_hash
    ):
        raise ValueError("replica engines must use the same ruleset ID and hash")
    if first.decision_interval_ticks != second.decision_interval_ticks:
        raise ValueError("replica engines must use the same policy cadence")
    return first, second


def _complete_match_tick_limit(engine: BattleEngine) -> int:
    match = engine.ruleset.match
    return (match.regulation_us + match.overtime_us) // engine.ruleset.tick_us + 2


def _validate_run_bounds(seed_count: int, max_ticks_per_seed: int, cadence: int) -> None:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if max_ticks_per_seed <= 0:
        raise ValueError("max_ticks_per_seed must be positive")
    if cadence <= 0:
        raise ValueError("decision_interval_ticks must be positive")


def _validate_state(
    engine: BattleEngine,
    state: BattleState,
    *,
    seed: int,
    replica: str,
) -> None:
    try:
        engine.validate_state(state)
    except Exception as error:
        raise DeterminismAuditError(
            seed=seed,
            tick=state.tick,
            detail=f"replica {replica} invariant violation: {error}",
            first_hash=_safe_state_hash(state),
        ) from error


def _step(
    engine: BattleEngine,
    state: BattleState,
    actions: tuple[SimAction, ...],
    *,
    seed: int,
    replica: str,
) -> tuple[SimEvent, ...]:
    try:
        return engine.step(state, actions)
    except Exception as error:
        raise DeterminismAuditError(
            seed=seed,
            tick=state.tick,
            detail=f"replica {replica} step/invariant failure: {error}",
            first_hash=_safe_state_hash(state),
        ) from error


def _safe_state_hash(state: BattleState) -> str | None:
    try:
        return state.state_hash()
    except Exception:
        return None


def _compare_hashes(
    first: BattleState,
    second: BattleState,
    *,
    seed: int,
    tick: int,
) -> str:
    first_hash = first.state_hash()
    second_hash = second.state_hash()
    if first_hash != second_hash:
        raise DeterminismAuditError(
            seed=seed,
            tick=tick,
            detail="canonical state hash divergence",
            first_hash=first_hash,
            second_hash=second_hash,
        )
    return first_hash


def _raise_divergence(
    first: BattleState,
    second: BattleState,
    seed: int,
    tick: int,
    detail: str,
) -> None:
    raise DeterminismAuditError(
        seed=seed,
        tick=tick,
        detail=detail,
        first_hash=first.state_hash(),
        second_hash=second.state_hash(),
    )
