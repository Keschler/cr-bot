"""Scenario and scheduled-action runners for reproducible experiments."""

from __future__ import annotations

from collections import defaultdict

from .engine import ENGINE_VERSION, BattleEngine
from .scenario import Scenario
from .state import BattleState, battle_state_from_primitive


def run_scenario(engine: BattleEngine, scenario: Scenario) -> BattleState:
    state, _ = run_scenario_with_snapshots(engine, scenario)
    return state


def run_scenario_with_snapshots(
    engine: BattleEngine,
    scenario: Scenario,
    *,
    snapshot_ticks: tuple[int, ...] = (),
) -> tuple[BattleState, dict[int, BattleState]]:
    """Run a scenario and capture requested pre-action tick boundaries."""

    if scenario.ruleset_id != engine.ruleset.ruleset_id:
        raise ValueError("scenario ruleset ID does not match engine")
    if scenario.ruleset_hash != engine.ruleset.content_hash:
        raise ValueError("scenario ruleset hash does not match pinned ruleset")
    if scenario.engine_version != ENGINE_VERSION:
        raise ValueError("scenario engine version does not match reference engine")
    if scenario.initial_state is None:
        state = engine.new_battle(
            scenario.decks,
            seed=scenario.seed,
            shuffle_decks=scenario.shuffle_decks,
        )
    else:
        state = battle_state_from_primitive(scenario.to_dict()["initial_state"])
        engine.validate_state(state)
        if state.seed != scenario.seed:
            raise ValueError("scenario seed does not match initial state")
        if tuple(player.deck for player in state.players) != scenario.decks:
            raise ValueError("scenario decks do not match initial state")
        if scenario.shuffle_decks:
            raise ValueError("initial-state scenarios cannot request deck shuffling")
        if state.terminal:
            raise ValueError("scenario initial state is already terminal")
    actions_by_tick = defaultdict(list)
    for scheduled in scenario.actions:
        if scheduled.tick < state.tick:
            raise ValueError("scenario action precedes initial state tick")
        actions_by_tick[scheduled.tick].append(scheduled.action)
    total_us = engine.ruleset.match.regulation_us + engine.ruleset.match.overtime_us
    maximum = (
        total_us // engine.ruleset.tick_us + 2
        if scenario.max_ticks is None
        else scenario.max_ticks
    )
    if maximum <= state.tick:
        raise ValueError("scenario max_ticks must exceed initial state tick")
    requested = set(snapshot_ticks)
    if any(type(tick) is not int or tick < state.tick or tick > maximum for tick in requested):
        raise ValueError("snapshot ticks must be within the scenario execution window")
    snapshots: dict[int, BattleState] = {}
    while not state.terminal and state.tick < maximum:
        if state.tick in requested:
            snapshots[state.tick] = battle_state_from_primitive(
                state.to_primitive(include_events=False)
            )
        engine.step(state, actions_by_tick.get(state.tick, ()))
    if state.tick in requested:
        snapshots[state.tick] = battle_state_from_primitive(
            state.to_primitive(include_events=False)
        )
    missing = requested - set(snapshots)
    if missing:
        raise ValueError(f"scenario ended before requested snapshot ticks: {sorted(missing)}")
    return state, snapshots
