from dataclasses import dataclass

import numpy as np

from cr_bot.domain.card_metadata import CARD_METADATA


GRID_SIZE = (18, 32)  # (columns, rows)
KATACR_BACKGROUND_SIZE = (568, 896)  # part2 arena crop size used by KataCR
KATACR_GRID_XYXY = (-0.9320463320463317, 72.54622356495467, 569.2610038610038, 879.9748640483384)
OWN_SIDE_FIRST_ROW = 17
RIVER_ROWS = (15, 16)
BRIDGE_COLS = (3, 14)
ENEMY_KING_TOWER_ROWS = (1, 2, 3, 4)
OWN_KING_TOWER_ROWS = (27, 28, 29, 30)
KING_TOWER_COLS = (7, 8, 9, 10)
ENEMY_PRINCESS_TOWER_ROWS = (5, 6, 7)
OWN_PRINCESS_TOWER_ROWS = (24, 25, 26)
LEFT_PRINCESS_TOWER_COLS = (2, 3, 4)
RIGHT_PRINCESS_TOWER_COLS = (13, 14, 15)

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


def build_river_mask() -> np.ndarray:
    mask = np.zeros_like(map_ground, dtype=bool)
    mask[list(RIVER_ROWS), :] = True
    mask[np.ix_(RIVER_ROWS, BRIDGE_COLS)] = False
    return mask


def build_bridge_mask() -> np.ndarray:
    mask = np.zeros_like(map_ground, dtype=bool)
    mask[np.ix_(RIVER_ROWS, BRIDGE_COLS)] = True
    return mask


def build_rect_mask(rows: tuple[int, ...], cols: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros_like(map_ground, dtype=bool)
    mask[np.ix_(rows, cols)] = True
    return mask


def build_left_princess_tower_site(rows: tuple[int, ...]) -> np.ndarray:
    return build_rect_mask(rows, LEFT_PRINCESS_TOWER_COLS)


def build_right_princess_tower_site(rows: tuple[int, ...]) -> np.ndarray:
    return build_rect_mask(rows, RIGHT_PRINCESS_TOWER_COLS)


def build_own_princess_tower_sites() -> np.ndarray:
    return OWN_LEFT_PRINCESS_TOWER_SITE | OWN_RIGHT_PRINCESS_TOWER_SITE


def build_own_king_tower_site() -> np.ndarray:
    return build_rect_mask(OWN_KING_TOWER_ROWS, KING_TOWER_COLS)


def build_enemy_princess_tower_sites() -> np.ndarray:
    return ENEMY_LEFT_PRINCESS_TOWER_SITE | ENEMY_RIGHT_PRINCESS_TOWER_SITE


def build_enemy_king_tower_site() -> np.ndarray:
    return build_rect_mask(ENEMY_KING_TOWER_ROWS, KING_TOWER_COLS)


def build_own_ground_mask() -> np.ndarray:
    mask = build_ground_mask()
    mask[:OWN_SIDE_FIRST_ROW, :] = False
    return mask


def build_own_half_mask() -> np.ndarray:
    mask = np.zeros_like(map_ground, dtype=bool)
    mask[OWN_SIDE_FIRST_ROW:, :] = True
    return mask

LEGAL_GROUND = build_ground_mask()
RIVER_MASK = build_river_mask()
BRIDGE_MASK = build_bridge_mask()
OWN_LEFT_PRINCESS_TOWER_SITE = build_left_princess_tower_site(OWN_PRINCESS_TOWER_ROWS)
OWN_RIGHT_PRINCESS_TOWER_SITE = build_right_princess_tower_site(OWN_PRINCESS_TOWER_ROWS)
OWN_PRINCESS_TOWER_SITES = build_own_princess_tower_sites()
OWN_KING_TOWER_SITE = build_own_king_tower_site()
ENEMY_LEFT_PRINCESS_TOWER_SITE = build_left_princess_tower_site(ENEMY_PRINCESS_TOWER_ROWS)
ENEMY_RIGHT_PRINCESS_TOWER_SITE = build_right_princess_tower_site(ENEMY_PRINCESS_TOWER_ROWS)
ENEMY_PRINCESS_TOWER_SITES = build_enemy_princess_tower_sites()
ENEMY_KING_TOWER_SITE = build_enemy_king_tower_site()
LEGAL_OWN_GROUND = build_own_ground_mask()
LEGAL_OWN_HALF = build_own_half_mask()
LEGAL_SPELL_ANYWHERE = np.ones_like(LEGAL_GROUND, dtype=bool)
LEGAL_GLOBAL_TARGET = LEGAL_GROUND

LEGAL_SPELL_ANYWHERE.setflags(write=False)
LEGAL_GROUND.setflags(write=False)
RIVER_MASK.setflags(write=False)
BRIDGE_MASK.setflags(write=False)
OWN_LEFT_PRINCESS_TOWER_SITE.setflags(write=False)
OWN_RIGHT_PRINCESS_TOWER_SITE.setflags(write=False)
OWN_PRINCESS_TOWER_SITES.setflags(write=False)
OWN_KING_TOWER_SITE.setflags(write=False)
ENEMY_LEFT_PRINCESS_TOWER_SITE.setflags(write=False)
ENEMY_RIGHT_PRINCESS_TOWER_SITE.setflags(write=False)
ENEMY_PRINCESS_TOWER_SITES.setflags(write=False)
ENEMY_KING_TOWER_SITE.setflags(write=False)
LEGAL_OWN_GROUND.setflags(write=False)
LEGAL_OWN_HALF.setflags(write=False)

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

ENEMY_LEFT_SPELL_DOWN_PATCH = np.zeros_like(LEGAL_OWN_HALF)
ENEMY_LEFT_SPELL_DOWN_PATCH[11:17, 0:9] = True

ENEMY_RIGHT_SPELL_DOWN_PATCH = np.zeros_like(LEGAL_OWN_HALF)
ENEMY_RIGHT_SPELL_DOWN_PATCH[11:17, 9:18] = True

ENEMY_LEFT_SPELL_DOWN_PATCH.setflags(write=False)
ENEMY_RIGHT_SPELL_DOWN_PATCH.setflags(write=False)

# BUILDINGS

BUILDING_FOOTPRINT = (3, 3)
TESLA_FOOTPRINT = (2, 2)

    

def build_footprint_anchor_mask(base_mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    mask = np.zeros_like(base_mask, dtype=bool)
    max_row = base_mask.shape[0] - rows + 1
    max_col = base_mask.shape[1] - cols + 1

    for row in range(max_row):
        for col in range(max_col):
            # True means the building footprint can start at this top-left cell.
            mask[row, col] = bool(base_mask[row : row + rows, col : col + cols].all())

    return mask


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


def build_restricted_spell_deploy_mask(
    enemy_left_princess_down: bool = False,
    enemy_right_princess_down: bool = False,
) -> np.ndarray:
    mask = LEGAL_OWN_HALF.copy()

    if enemy_left_princess_down:
        mask |= ENEMY_LEFT_SPELL_DOWN_PATCH
    if enemy_right_princess_down:
        mask |= ENEMY_RIGHT_SPELL_DOWN_PATCH

    return mask


def build_building_deploy_mask(
    footprint_rows: int,
    footprint_cols: int,
    own_left_princess_down: bool = False,
    own_right_princess_down: bool = False,
    enemy_left_princess_down: bool = False,
    enemy_right_princess_down: bool = False,
) -> np.ndarray:
    deploy_mask = build_own_deploy_mask(
        own_left_princess_down=own_left_princess_down,
        own_right_princess_down=own_right_princess_down,
        enemy_left_princess_down=enemy_left_princess_down,
        enemy_right_princess_down=enemy_right_princess_down,
    )
    return build_footprint_anchor_mask(deploy_mask, footprint_rows, footprint_cols)


def build_standard_building_deploy_mask(
    own_left_princess_down: bool = False,
    own_right_princess_down: bool = False,
    enemy_left_princess_down: bool = False,
    enemy_right_princess_down: bool = False,
) -> np.ndarray:
    return build_building_deploy_mask(
        *BUILDING_FOOTPRINT,
        own_left_princess_down=own_left_princess_down,
        own_right_princess_down=own_right_princess_down,
        enemy_left_princess_down=enemy_left_princess_down,
        enemy_right_princess_down=enemy_right_princess_down,
    )


def build_tesla_deploy_mask(
    own_left_princess_down: bool = False,
    own_right_princess_down: bool = False,
    enemy_left_princess_down: bool = False,
    enemy_right_princess_down: bool = False,
) -> np.ndarray:
    return build_building_deploy_mask(
        *TESLA_FOOTPRINT,
        own_left_princess_down=own_left_princess_down,
        own_right_princess_down=own_right_princess_down,
        enemy_left_princess_down=enemy_left_princess_down,
        enemy_right_princess_down=enemy_right_princess_down,
    )


def normalize_card_name(card_name: str) -> str:
    return card_name.strip().lower().replace(" ", "-")


def get_card_deploy_mask(
    card_name: str,
    own_left_princess_down: bool = False,
    own_right_princess_down: bool = False,
    enemy_left_princess_down: bool = False,
    enemy_right_princess_down: bool = False,
) -> np.ndarray:
    card_key = normalize_card_name(card_name)
    metadata = CARD_METADATA.get(card_key)
    if metadata is None:
        raise KeyError(f"unknown card: {card_name!r}")

    placement_class = metadata.get("placement_class")

    if placement_class == "own_ground":
        return build_own_deploy_mask(
            own_left_princess_down=own_left_princess_down,
            own_right_princess_down=own_right_princess_down,
            enemy_left_princess_down=enemy_left_princess_down,
            enemy_right_princess_down=enemy_right_princess_down,
        )
    if placement_class == "building":
        if card_key == "tesla":
            return build_tesla_deploy_mask(
                own_left_princess_down=own_left_princess_down,
                own_right_princess_down=own_right_princess_down,
                enemy_left_princess_down=enemy_left_princess_down,
                enemy_right_princess_down=enemy_right_princess_down,
            )
        return build_standard_building_deploy_mask(
            own_left_princess_down=own_left_princess_down,
            own_right_princess_down=own_right_princess_down,
            enemy_left_princess_down=enemy_left_princess_down,
            enemy_right_princess_down=enemy_right_princess_down,
        )
    if placement_class == "spell_anywhere":
        return LEGAL_SPELL_ANYWHERE
    if placement_class == "global_target":
        return LEGAL_GLOBAL_TARGET
    if placement_class == "spells":
        return build_restricted_spell_deploy_mask(
            enemy_left_princess_down=enemy_left_princess_down,
            enemy_right_princess_down=enemy_right_princess_down,
        )

    raise ValueError(f"unsupported placement_class {placement_class!r} for {card_key!r}")
