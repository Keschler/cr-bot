"""Shared imports, constants, and small value objects for the engine package.

The mechanics live in focused mixins under :mod:`simulator.engine`.  This
module intentionally contains only dependencies and stateless helpers so the
mixins can call one another through the composed ``BattleEngine`` without
creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Protocol

from ..actions import PlayCardAction, SimAction, UseAbilityAction, WaitAction
from ..events import SimEvent
from ..fixed import (
    ELIXIR_SCALE,
    PERMILLE,
    POSITION_SCALE,
    SECOND_US,
    DeterministicRng,
    ceil_div,
    distance_mtile,
    move_towards,
)
from ..geometry import (
    GRID_COLS,
    GRID_ROWS,
    TOWER_SITES,
    cell_center_mtile,
    is_basic_deploy_cell,
    is_ground_cell,
    is_spell_cell,
    position_to_cell,
)
from ..navigation import (
    NavigationObstacle,
    plan_route,
    point_is_walkable,
    segment_is_walkable,
)
from ..ruleset import CardDefinition, Ruleset, RulesetError, TowerDefinition, load_ruleset
from ..roster import PLAYER_DECK
from ..state import (
    AreaEffectState,
    BattleState,
    EntityState,
    PlayerState,
    ProjectileState,
    StatusState,
)


# Keep the engine default, policy observation slots, scenario factory, and
# physical Testspiel fixed-deck order on one canonical contract.
BASE_HOG_CYCLE_DECK = PLAYER_DECK

# Behavior-changing mechanics are part of the engine identity.  Replays and
# mined evidence must never be silently interpreted by a newer algorithm.
ENGINE_VERSION = "reference-0.36.0"
_SEED_MASK = (1 << 64) - 1
_SLOW_STATUS_KINDS = frozenset({"slow", "freeze", "poison-slow", "earthquake-slow"})
_CARD_CYCLE_COOLDOWN_SINGLE_US = 2_000_000
_CARD_CYCLE_COOLDOWN_ACCELERATED_US = 1_000_000


# The policy action grid is fixed by the observation contract.  Keeping its
# valid cells as immutable tuples lets the legality path avoid rebuilding the
# same coordinate pairs for every hand slot.
_POLICY_GRID_CELLS: tuple[tuple[int, int], ...] = tuple(
    (col, row)
    for row in range(GRID_ROWS)
    for col in range(GRID_COLS)
)
_GROUND_CELLS: tuple[tuple[int, int], ...] = tuple(
    cell for cell in _POLICY_GRID_CELLS if is_ground_cell(cell)
)
_BASIC_DEPLOY_CELLS: tuple[frozenset[tuple[int, int]], ...] = tuple(
    frozenset(
        cell
        for cell in _GROUND_CELLS
        if is_basic_deploy_cell(player, cell)
    )
    for player in (0, 1)
)


@lru_cache(maxsize=512)
def _blocked_deployment_cells(
    obstacles: tuple[tuple[int, int, int], ...],
    radius: int,
) -> frozenset[tuple[int, int]]:
    """Cache grid cells blocked by one obstacle layout and card radius."""

    if not obstacles:
        return frozenset()
    blocked: set[tuple[int, int]] = set()
    center_offset = POSITION_SCALE // 2
    for col, row in _POLICY_GRID_CELLS:
        x = col * POSITION_SCALE + center_offset
        y = row * POSITION_SCALE + center_offset
        for obstacle_x, obstacle_y, obstacle_radius in obstacles:
            if distance_mtile(x, y, obstacle_x, obstacle_y) < radius + obstacle_radius:
                blocked.add((col, row))
                break
    return frozenset(blocked)


@lru_cache(maxsize=128)
def _footprint_cells_in_allowed(
    allowed_cells: frozenset[tuple[int, int]],
    size: int,
) -> tuple[tuple[int, int], ...]:
    """Cache building center cells for an unchanged territory layout."""

    low = -(size // 2)
    high = size - size // 2
    return tuple(
        (col, row)
        for col, row in _POLICY_GRID_CELLS
        if all(
            (col + dcol, row + drow) in allowed_cells
            for drow in range(low, high)
            for dcol in range(low, high)
        )
    )


@dataclass(frozen=True, slots=True)
class ActionResult:
    player: int
    accepted: bool
    reason: str | None = None
    card_id: str | None = None


class Controller(Protocol):
    def choose_action(self, engine: "BattleEngine", state: BattleState, player: int) -> SimAction: ...


# Mixins use star-import from this module to keep the extracted method bodies
# readable.  Include single-underscore engine constants/helpers explicitly;
# Python's default star-import filtering would otherwise hide them.
__all__ = tuple(
    name
    for name in globals()
    if not name.startswith("__")
)
