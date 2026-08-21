"""Arena coordinates, legal placement, mirroring, and bridge routing."""

from __future__ import annotations

from dataclasses import dataclass

from .fixed import POSITION_SCALE


GRID_COLS = 18
GRID_ROWS = 32
RIVER_ROWS = (15, 16)

# This is the existing policy grid's walkability map, represented without a
# NumPy dependency so the headless physics core remains stdlib-only.
_GROUND_ROWS = (
    "000000111111000000",
    "111111100001111111",
    "111111100001111111",
    "111111100001111111",
    "111111100001111111",
    "110001111111100011",
    "110001111111100011",
    "110001111111100011",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "011111111111111110",
    "000100000000001000",
    "000100000000001000",
    "011111111111111110",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "111111111111111111",
    "110001111111100011",
    "110001111111100011",
    "110001111111100011",
    "111111100001111111",
    "111111100001111111",
    "111111100001111111",
    "111111100001111111",
    "000000111111000000",
)


@dataclass(frozen=True, slots=True)
class TowerSite:
    owner: int
    role: str
    x_mtile: int
    y_mtile: int


TOWER_SITES = (
    TowerSite(0, "left", 3_500, 25_500),
    TowerSite(0, "king", 9_000, 28_500),
    TowerSite(0, "right", 14_500, 25_500),
    TowerSite(1, "left", 14_500, 6_500),
    TowerSite(1, "king", 9_000, 3_500),
    TowerSite(1, "right", 3_500, 6_500),
)


def validate_cell(cell: tuple[int, int]) -> None:
    if not isinstance(cell, tuple) or len(cell) != 2:
        raise ValueError(f"cell must be a (column, row) tuple: {cell!r}")
    col, row = cell
    if type(col) is not int or type(row) is not int:
        raise ValueError(f"cell coordinates must be integers: {cell!r}")
    if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
        raise ValueError(f"cell out of bounds: {cell!r}")


def cell_center_mtile(cell: tuple[int, int]) -> tuple[int, int]:
    validate_cell(cell)
    col, row = cell
    return col * POSITION_SCALE + POSITION_SCALE // 2, row * POSITION_SCALE + POSITION_SCALE // 2


def position_to_cell(x_mtile: int, y_mtile: int) -> tuple[int, int] | None:
    col = x_mtile // POSITION_SCALE
    row = y_mtile // POSITION_SCALE
    if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
        return col, row
    return None


def mirror_cell(cell: tuple[int, int]) -> tuple[int, int]:
    validate_cell(cell)
    col, row = cell
    return GRID_COLS - 1 - col, GRID_ROWS - 1 - row


def mirror_position(x_mtile: int, y_mtile: int) -> tuple[int, int]:
    return GRID_COLS * POSITION_SCALE - x_mtile, GRID_ROWS * POSITION_SCALE - y_mtile


def is_ground_cell(cell: tuple[int, int]) -> bool:
    try:
        validate_cell(cell)
    except (TypeError, ValueError):
        return False
    col, row = cell
    return _GROUND_ROWS[row][col] == "1"


def is_basic_deploy_cell(player: int, cell: tuple[int, int]) -> bool:
    if type(player) is not int or player not in (0, 1) or not is_ground_cell(cell):
        return False
    _, row = cell
    return row >= 17 if player == 0 else row <= 14


def is_spell_cell(cell: tuple[int, int]) -> bool:
    try:
        validate_cell(cell)
    except (TypeError, ValueError):
        return False
    return True


def building_footprint_fits(player: int, cell: tuple[int, int], size: int = 3) -> bool:
    """Check a square footprint around a placement-center cell.

    Vision action labels represent the deployment/object center.  The legacy
    NumPy mask treats the same cell as a top-left anchor; the observation
    adapter exposes that old mask separately for reproducibility, while the
    authoritative engine consistently uses center semantics.
    """

    col, row = cell
    low = -(size // 2)
    high = size - size // 2
    for drow in range(low, high):
        for dcol in range(low, high):
            if not is_basic_deploy_cell(player, (col + dcol, row + drow)):
                return False
    return True

