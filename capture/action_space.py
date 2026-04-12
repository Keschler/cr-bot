from dataclasses import dataclass

import numpy as np


GRID_SIZE = (18, 32)  # (columns, rows)
KATACR_BACKGROUND_SIZE = (1080, 2400)  # (width, height)
KATACR_GRID_CELL_SIZE = (56, 46)  # (width, height)
KATACR_GRID_TOP_LEFT = (28, 320)
KATACR_GRID_XYXY = (
    KATACR_GRID_TOP_LEFT[0],
    KATACR_GRID_TOP_LEFT[1],
    KATACR_GRID_TOP_LEFT[0] + GRID_SIZE[0] * KATACR_GRID_CELL_SIZE[0],
    KATACR_GRID_TOP_LEFT[1] + GRID_SIZE[1] * KATACR_GRID_CELL_SIZE[1],
)
OWN_SIDE_FIRST_ROW = 17

map_ground = np.array(
    [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)
map_ground.setflags(write=False)


@dataclass(frozen=True)
class GridSpec:
    cols: int
    rows: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def norm_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        if not (self.x0 <= x < self.x1 and self.y0 <= y < self.y1):
            return None
        col = int((x - self.x0) / self.width * self.cols)
        row = int((y - self.y0) / self.height * self.rows)
        return col, row

    def pixel_to_cell(
        self,
        x: float,
        y: float,
        arena_px: tuple[float, float, float, float],
    ) -> tuple[int, int] | None:
        ax, ay, aw, ah = arena_px
        return self.norm_to_cell((x - ax) / aw, (y - ay) / ah)

    def cell_to_norm_center(self, col: int, row: int) -> tuple[float, float]:
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            raise ValueError(f"cell out of bounds: col={col}, row={row}")
        x = self.x0 + (col + 0.5) / self.cols * self.width
        y = self.y0 + (row + 0.5) / self.rows * self.height
        return x, y

    def cell_to_pixel_center(
        self,
        col: int,
        row: int,
        arena_px: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        ax, ay, aw, ah = arena_px
        x, y = self.cell_to_norm_center(col, row)
        return ax + x * aw, ay + y * ah


def _katacr_grid_spec() -> GridSpec:
    bg_w, bg_h = KATACR_BACKGROUND_SIZE
    x0, y0, x1, y1 = KATACR_GRID_XYXY
    cols, rows = GRID_SIZE
    return GridSpec(cols, rows, x0 / bg_w, y0 / bg_h, x1 / bg_w, y1 / bg_h)


ACTION_GRID = _katacr_grid_spec()


def build_ground_mask() -> np.ndarray:
    return map_ground > 0


def build_own_ground_mask() -> np.ndarray:
    mask = build_ground_mask()
    mask[:OWN_SIDE_FIRST_ROW, :] = False
    return mask


LEGAL_GROUND = build_ground_mask()
LEGAL_OWN_GROUND = build_own_ground_mask()
LEGAL_GROUND.setflags(write=False)
LEGAL_OWN_GROUND.setflags(write=False)

OWN_LEFT_PRINCESS_DOWN_PATCH = np.zeros_like(LEGAL_OWN_GROUND)
OWN_LEFT_PRINCESS_DOWN_PATCH[24:27, 2:5] = True

OWN_RIGHT_PRINCESS_DOWN_PATCH = np.zeros_like(LEGAL_OWN_GROUND)
OWN_RIGHT_PRINCESS_DOWN_PATCH[24:27, 13:16] = True

ENEMY_LEFT_PRINCESS_DOWN_PATCH = np.zeros_like(LEGAL_OWN_GROUND)
ENEMY_LEFT_PRINCESS_DOWN_PATCH[11:15, 0:9] = True
ENEMY_LEFT_PRINCESS_DOWN_PATCH[15:17, 3] = True
ENEMY_LEFT_PRINCESS_DOWN_PATCH &= LEGAL_GROUND

ENEMY_RIGHT_PRINCESS_DOWN_PATCH = np.zeros_like(LEGAL_OWN_GROUND)
ENEMY_RIGHT_PRINCESS_DOWN_PATCH[11:15, 9:18] = True
ENEMY_RIGHT_PRINCESS_DOWN_PATCH[15:17, 14] = True
ENEMY_RIGHT_PRINCESS_DOWN_PATCH &= LEGAL_GROUND


OWN_LEFT_PRINCESS_DOWN_PATCH.setflags(write=False)
OWN_RIGHT_PRINCESS_DOWN_PATCH.setflags(write=False)
ENEMY_LEFT_PRINCESS_DOWN_PATCH.setflags(write=False)
ENEMY_RIGHT_PRINCESS_DOWN_PATCH.setflags(write=False)


# TODO ENEMY WHEN BOTH PRINCESS TOWERS ARE DOWN + 


def build_own_deploy_mask(
    own_left_princess_down: bool = False,
    own_right_princess_down: bool = False,
    enemy_left_princess_down: bool = False,
    enemy_right_princess_down: bool = False,
) -> np.ndarray:
    mask = LEGAL_OWN_GROUND.copy()

    if own_left_princess_down:
        mask |= OWN_LEFT_PRINCESS_DOWN_PATCH
    if own_right_princess_down:
        mask |= OWN_RIGHT_PRINCESS_DOWN_PATCH
    if enemy_left_princess_down:
        mask |= ENEMY_LEFT_PRINCESS_DOWN_PATCH
    if enemy_right_princess_down:
        mask |= ENEMY_RIGHT_PRINCESS_DOWN_PATCH

    return mask


@dataclass(frozen=True)
class Region:
    id: int
    name: str
    mask: np.ndarray
    allowed_classes: frozenset[str]
    requires_left_princess_down: bool = False
    requires_right_princess_down: bool = False
    requires_left_enemy_princess_down: bool = False
    requires_right_enemy_princess_down: bool = False

    def contains_cell(self, col: int, row: int) -> bool:
        rows, cols = self.mask.shape
        return 0 <= row < rows and 0 <= col < cols and bool(self.mask[row, col])

    def contains_norm(self, x: float, y: float, grid: GridSpec = ACTION_GRID) -> bool:
        cell = grid.norm_to_cell(x, y)
        return cell is not None and self.contains_cell(*cell)

    def contains_pixel(
        self,
        x: float,
        y: float,
        arena_px: tuple[float, float, float, float],
        grid: GridSpec = ACTION_GRID,
    ) -> bool:
        cell = grid.pixel_to_cell(x, y, arena_px)
        return cell is not None and self.contains_cell(*cell)


REGIONS = [
    Region(
        id=0,
        name="troops_my_side",
        mask=LEGAL_OWN_GROUND,
        allowed_classes=frozenset({"own_ground"}),
    ),
]
