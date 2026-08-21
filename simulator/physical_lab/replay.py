"""Replay a physical experiment's logical actions through the reference engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..actions import PlayCardAction
from ..engine import BASE_HOG_CYCLE_DECK, ENGINE_VERSION, BattleEngine
from ..ruleset import Ruleset, load_ruleset
from ..scenario import Scenario, ScheduledAction
from ..state import BattleState, battle_state_from_primitive
from .schema import ExperimentSpec, PhysicalAction, PhysicalLabError


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _side_to_player(side: str) -> int:
    if side == "A":
        return 0
    if side == "B":
        return 1
    raise PhysicalLabError(f"unknown physical side: {side!r}")


def _default_decks(spec: ExperimentSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    decks = spec.initial_conditions.decks
    return (
        tuple(decks.get("A", BASE_HOG_CYCLE_DECK)),
        tuple(decks.get("B", BASE_HOG_CYCLE_DECK)),
    )


@dataclass(frozen=True, slots=True)
class ReplayAction:
    action_id: str
    side: str
    match_time_us: int
    simulator_tick: int
    card_slot: int
    cell: tuple[int, int]
    accepted_boundary_error_us: int

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "side": self.side,
            "match_time_us": self.match_time_us,
            "simulator_tick": self.simulator_tick,
            "card_slot": self.card_slot,
            "cell": list(self.cell),
            "accepted_boundary_error_us": self.accepted_boundary_error_us,
        }


@dataclass(frozen=True, slots=True)
class SimulatorReplay:
    """Simulator result plus per-tick snapshots for trajectory comparison."""

    scenario: Scenario
    final_state: BattleState
    snapshots: Mapping[int, BattleState]
    actions: tuple[ReplayAction, ...]
    experiment_hash: str

    @property
    def scenario_tick_us(self) -> int:
        return load_ruleset(self.scenario.ruleset_id).tick_us

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "physical_lab_simulator_replay",
            "scenario_id": self.scenario.scenario_id,
            "experiment_hash": self.experiment_hash,
            "scenario": self.scenario.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "snapshot_ticks": sorted(self.snapshots),
            "snapshot_count": len(self.snapshots),
            "final_state_hash": self.final_state.state_hash(),
            "event_log_hash": self.final_state.event_log_hash(),
            "replay_hash": self.final_state.replay_hash(),
        }


def _action_time_map(
    spec: ExperimentSpec,
    action_times: Mapping[str, int] | None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for action in spec.actions:
        if action.trigger.type.value == "match_time_us":
            result[action.action_id] = action.trigger.value
        else:
            if action_times is None or action.action_id not in action_times:
                raise PhysicalLabError(
                    f"after_observation action {action.action_id!r} lacks an observed match time"
                )
            value = action_times[action.action_id]
            if type(value) is not int or value < 0:
                raise PhysicalLabError(f"action time for {action.action_id!r} must be non-negative")
            result[action.action_id] = value
    return result


def _card_slot(spec: ExperimentSpec, action: PhysicalAction, decks: tuple[tuple[str, ...], tuple[str, ...]]) -> int:
    if action.card_slot is not None:
        return action.card_slot
    mapped = spec.initial_conditions.hand_slots.get(action.side, {}).get(action.card_id)
    if mapped is not None:
        return mapped
    deck = decks[_side_to_player(action.side)]
    try:
        return deck.index(action.card_id)
    except ValueError as error:
        raise PhysicalLabError(
            f"action card {action.card_id!r} is absent from side {action.side} deck"
        ) from error


def build_scenario(
    spec: ExperimentSpec,
    *,
    action_times: Mapping[str, int] | None = None,
    ruleset: Ruleset | None = None,
) -> tuple[Scenario, tuple[ReplayAction, ...]]:
    """Convert logical match-time actions to the engine's explicit tick API."""

    if spec.engine_version != ENGINE_VERSION:
        raise PhysicalLabError(
            f"experiment engine_version {spec.engine_version!r} does not match {ENGINE_VERSION!r}"
        )
    ruleset = ruleset or load_ruleset(spec.ruleset_id)
    if ruleset.content_hash != spec.ruleset_hash:
        raise PhysicalLabError(
            f"experiment ruleset hash does not match {ruleset.ruleset_id}: "
            f"{spec.ruleset_hash} != {ruleset.content_hash}"
        )
    decks = _default_decks(spec)
    times = _action_time_map(spec, action_times)
    scheduled: list[ScheduledAction] = []
    replay_actions: list[ReplayAction] = []
    for index, action in enumerate(spec.actions):
        match_time_us = times[action.action_id]
        tick = _ceil_div(match_time_us, ruleset.tick_us)
        slot = _card_slot(spec, action, decks)
        scheduled.append(
            ScheduledAction(
                tick=tick,
                action=PlayCardAction(
                    player=_side_to_player(action.side),
                    card_slot=slot,
                    cell=action.arena_cell,
                ),
            )
        )
        replay_actions.append(
            ReplayAction(
                action_id=action.action_id,
                side=action.side,
                match_time_us=match_time_us,
                simulator_tick=tick,
                card_slot=slot,
                cell=action.arena_cell,
                accepted_boundary_error_us=tick * ruleset.tick_us - match_time_us,
            )
        )
    # Scenario actions must be sorted by engine tick.  Preserve experiment
    # order for equal ticks, which is part of the canonical action boundary.
    ordered = sorted(
        enumerate(zip(scheduled, replay_actions)),
        key=lambda pair: (pair[1][0].tick, pair[0]),
    )
    scheduled = [item[1][0] for item in ordered]
    replay_actions = [item[1][1] for item in ordered]
    max_ticks = max(
        1,
        _ceil_div(spec.duration_us, ruleset.tick_us) + 1,
        *(item.simulator_tick + 1 for item in replay_actions),
    )
    scenario = Scenario(
        scenario_id=f"physical-{spec.experiment_id}-{spec.experiment_hash()[-12:]}",
        ruleset_id=ruleset.ruleset_id,
        ruleset_hash=ruleset.content_hash,
        engine_version=ENGINE_VERSION,
        seed=spec.seed,
        decks=decks,
        actions=tuple(scheduled),
        max_ticks=max_ticks,
        shuffle_decks=False,
        split=spec.evidence_split.value,
        tags=("physical_lab", spec.experiment_id),
        oracle={"promoted": False, "source": "physical_lab_logical_replay"},
    )
    return scenario, tuple(replay_actions)


def _initial_state(
    engine: BattleEngine,
    scenario: Scenario,
    requested_elixir_milli: Mapping[str, int] | None = None,
) -> BattleState:
    if scenario.initial_state is not None:
        state = battle_state_from_primitive(scenario.to_dict()["initial_state"])
    else:
        state = engine.new_battle(
            scenario.decks,
            seed=scenario.seed,
            shuffle_decks=scenario.shuffle_decks,
        )
    # The physical specification may deliberately request a full-elixir
    # challenge setup even though the normal engine match starts at 5 elixir.
    # Apply that logical initial condition before the first action and validate
    # it through the engine's normal state contract.
    requested = requested_elixir_milli
    if requested is not None:
        for player_index, side in enumerate(("A", "B")):
            value = requested.get(side)
            if value is None:
                continue
            if value > engine.ruleset.match.max_elixir_milli:
                raise PhysicalLabError(
                    f"requested initial elixir for {side} exceeds ruleset maximum"
                )
            state.players[player_index].elixir_milli = value
            state.players[player_index].elixir_remainder = 0
    engine.validate_state(state)
    return state


def run_simulator_replay(
    spec: ExperimentSpec,
    *,
    action_times: Mapping[str, int] | None = None,
    capture_snapshots: bool = True,
    validate_every_tick: bool = True,
) -> SimulatorReplay:
    """Run the exact logical actions with the pinned reference engine."""

    if spec.initial_conditions.tower_state != "default":
        raise PhysicalLabError(
            f"unsupported physical initial tower_state: {spec.initial_conditions.tower_state!r}"
        )
    scenario, replay_actions = build_scenario(spec, action_times=action_times)
    engine = BattleEngine(load_ruleset(spec.ruleset_id), validate_every_tick=validate_every_tick)
    state = _initial_state(
        engine,
        scenario,
        requested_elixir_milli=spec.initial_conditions.requested_elixir_milli,
    )
    actions_by_tick: dict[int, list[Any]] = defaultdict(list)
    for scheduled in scenario.actions:
        actions_by_tick[scheduled.tick].append(scheduled.action)
    snapshots: dict[int, BattleState] = {}
    maximum = scenario.max_ticks or 0
    while not state.terminal and state.tick < maximum:
        if capture_snapshots:
            snapshots[state.tick] = battle_state_from_primitive(
                state.to_primitive(include_events=False)
            )
        engine.step(state, actions_by_tick.get(state.tick, ()))
    if capture_snapshots:
        snapshots[state.tick] = battle_state_from_primitive(
            state.to_primitive(include_events=False)
        )
    return SimulatorReplay(
        scenario=scenario,
        final_state=state,
        snapshots=snapshots,
        actions=replay_actions,
        experiment_hash=spec.experiment_hash(),
    )


def replay_hash_pair(spec: ExperimentSpec, *, action_times: Mapping[str, int] | None = None) -> tuple[str, str, str]:
    """Run the software replay twice for a deterministic Phase-0 audit."""

    first = run_simulator_replay(spec, action_times=action_times)
    second = run_simulator_replay(spec, action_times=action_times)
    first_hashes = (first.final_state.state_hash(), first.final_state.event_log_hash(), first.final_state.replay_hash())
    second_hashes = (second.final_state.state_hash(), second.final_state.event_log_hash(), second.final_state.replay_hash())
    if first_hashes != second_hashes:
        raise PhysicalLabError(f"simulator replay is not deterministic: {first_hashes} != {second_hashes}")
    return first_hashes


__all__ = [
    "ReplayAction",
    "SimulatorReplay",
    "build_scenario",
    "replay_hash_pair",
    "run_simulator_replay",
]
